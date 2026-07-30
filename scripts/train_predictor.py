#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from specstream.io import atomic_json, seed_everything, utc_now
from specstream.predictor import LayerwiseExpertPredictor
from specstream.traces import load_split, multi_hot, recall_at_k


def validation_metrics(
    model: LayerwiseExpertPredictor,
    data,
    batch_size: int,
    device: torch.device,
    ready_counts: torch.Tensor,
) -> dict[str, float]:
    recall_at_8_values = []
    ready_recall_values = []
    max_ready = int(ready_counts.max().item())
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            stop = min(start + batch_size, len(data))
            layers = data.layer[start:stop].to(device)
            targets = data.targets[start:stop].to(device)
            logits = model(
                data.hidden[start:stop].to(device, torch.float32),
                layers,
            )
            recall_at_8_values.append(recall_at_k(logits, targets, 8).cpu())
            if max_ready == 0:
                ready_recall_values.append(torch.zeros(stop - start))
                continue
            predictions = logits.topk(max_ready, dim=-1).indices
            within_deadline = (
                torch.arange(max_ready, device=device)[None]
                < ready_counts[layers][:, None]
            )
            hits = (
                (predictions[:, :, None] == targets[:, None, :])
                & within_deadline[:, :, None]
            ).any(dim=1)
            ready_recall_values.append(
                (hits.float().sum(dim=-1) / targets.shape[-1]).cpu()
            )
    return {
        "validation_recall_at_8": torch.cat(recall_at_8_values).mean().item(),
        "validation_ready_recall": torch.cat(ready_recall_values).mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument(
        "--architecture",
        choices=["layer_aware", "low_rank"],
        default="layer_aware",
    )
    parser.add_argument("--layer-embedding-width", type=int, default=32)
    parser.add_argument("--loss", choices=["bce", "kl"], default="bce")
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--maximum-train-pairs-per-prompt", type=int, default=4096)
    parser.add_argument(
        "--feature-key",
        choices=["features", "router_features"],
        default="router_features",
    )
    parser.add_argument(
        "--target-horizon",
        type=int,
        default=1,
        help="Predict routes this many layers after the anchor feature.",
    )
    parser.add_argument("--deadline-profile", type=Path, required=True)
    parser.add_argument("--deadline-weighted-loss", action="store_true")
    parser.add_argument("--seed", type=int, default=260724787)
    args = parser.parse_args()

    seed_everything(args.seed)
    train = load_split(
        args.traces,
        "train",
        args.maximum_train_pairs_per_prompt,
        args.feature_key,
        args.target_horizon,
    )
    validation = load_split(
        args.traces,
        "validation",
        feature_key=args.feature_key,
        target_horizon=args.target_horizon,
    )
    hidden_size = train.hidden.shape[-1]
    num_layers = max(train.num_layers, validation.num_layers)
    num_experts = max(train.num_experts, validation.num_experts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LayerwiseExpertPredictor(
        hidden_size,
        num_layers,
        num_experts,
        args.width,
        architecture=args.architecture,
        layer_embedding_width=args.layer_embedding_width,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    positive_weight = torch.tensor(
        [(num_experts - train.targets.shape[-1]) / train.targets.shape[-1]],
        device=device,
    )
    deadline_profile = json.loads(args.deadline_profile.read_text())
    ready_counts = torch.tensor(
        deadline_profile["ready_counts"], device=device, dtype=torch.long
    )
    if len(ready_counts) != num_layers:
        raise ValueError("deadline profile does not match model layer count")
    layer_weights = torch.ones(num_layers, device=device)
    if args.deadline_weighted_loss:
        layer_weights = torch.tensor(
            deadline_profile["weights"], device=device, dtype=torch.float32
        )
        layer_weights /= layer_weights.mean()

    loader_generator = torch.Generator().manual_seed(args.seed)
    if args.loss == "kl":
        if train.teacher_logits is None:
            raise ValueError("KL training requires router_logits in every trace")
        training_dataset = TensorDataset(
            train.hidden, train.layer, train.targets, train.teacher_logits
        )
    else:
        training_dataset = TensorDataset(train.hidden, train.layer, train.targets)
    loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        pin_memory=True,
    )
    history = []
    best_recall = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            hidden, layer, target_indices = batch[:3]
            hidden = hidden.to(device, torch.float32, non_blocking=True)
            layer = layer.to(device, non_blocking=True)
            logits = model(hidden, layer)
            if args.loss == "kl":
                teacher = batch[3].to(device, torch.float32, non_blocking=True)
                element_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(logits, dim=-1),
                    torch.nn.functional.softmax(teacher, dim=-1),
                    reduction="none",
                ).sum(dim=-1)
            else:
                target = multi_hot(target_indices.to(device), num_experts)
                element_loss = (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        logits,
                        target,
                        pos_weight=positive_weight,
                        reduction="none",
                    ).mean(dim=-1)
                )
            loss = (element_loss * layer_weights[layer]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())
        metrics = validation_metrics(
            model, validation, args.batch_size, device, ready_counts
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            **metrics,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result), flush=True)
        if metrics["validation_ready_recall"] > best_recall:
            best_recall = metrics["validation_ready_recall"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    assert best_state is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_metadata = {
        **model.metadata(),
        "trace_feature_key": args.feature_key,
        "target_horizon": args.target_horizon,
        "deadline_weighted_loss": args.deadline_weighted_loss,
        "training_loss": args.loss,
    }
    torch.save(
        {
            "format_version": 1,
            "captured_at_utc": utc_now(),
            "model_metadata": model_metadata,
            "state_dict": best_state,
            "best_validation_ready_recall": best_recall,
            "history": history,
            "seed": args.seed,
            "layer_weights": layer_weights.cpu(),
            "deadline_profile": deadline_profile,
        },
        args.output,
    )
    atomic_json(
        args.output.with_suffix(".json"),
        {
            "captured_at_utc": utc_now(),
            "model_metadata": model_metadata,
            "best_validation_ready_recall": best_recall,
            "history": history,
            "seed": args.seed,
            "training_pairs": len(train),
            "validation_pairs": len(validation),
            "deadline_profile": deadline_profile,
        },
    )


if __name__ == "__main__":
    main()
