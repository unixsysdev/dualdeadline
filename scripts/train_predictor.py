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


def validation_recall(
    model: LayerwiseExpertPredictor,
    data,
    batch_size: int,
    device: torch.device,
) -> float:
    values = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            stop = min(start + batch_size, len(data))
            logits = model(
                data.hidden[start:stop].to(device, torch.float32),
                data.layer[start:stop].to(device),
            )
            values.append(
                recall_at_k(logits, data.targets[start:stop].to(device), 8).cpu()
            )
    return torch.cat(values).mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--maximum-train-pairs-per-prompt", type=int, default=4096)
    parser.add_argument("--layer-weights", type=Path)
    parser.add_argument("--seed", type=int, default=260724787)
    args = parser.parse_args()

    seed_everything(args.seed)
    train = load_split(
        args.traces, "train", args.maximum_train_pairs_per_prompt
    )
    validation = load_split(args.traces, "validation")
    hidden_size = train.hidden.shape[-1]
    num_layers = max(train.num_layers, validation.num_layers)
    num_experts = max(train.num_experts, validation.num_experts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LayerwiseExpertPredictor(
        hidden_size, num_layers, num_experts, args.width
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    positive_weight = torch.tensor(
        [(num_experts - train.targets.shape[-1]) / train.targets.shape[-1]],
        device=device,
    )
    layer_weights = torch.ones(num_layers, device=device)
    if args.layer_weights:
        values = json.loads(args.layer_weights.read_text())
        layer_weights = torch.tensor(values["weights"], device=device)
        layer_weights /= layer_weights.mean()

    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(train.hidden, train.layer, train.targets),
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
        for hidden, layer, target_indices in loader:
            hidden = hidden.to(device, torch.float32, non_blocking=True)
            layer = layer.to(device, non_blocking=True)
            target = multi_hot(target_indices.to(device), num_experts)
            logits = model(hidden, layer)
            element_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                target,
                pos_weight=positive_weight,
                reduction="none",
            ).mean(dim=-1)
            loss = (element_loss * layer_weights[layer]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().item())
        recall = validation_recall(model, validation, args.batch_size, device)
        epoch_result = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "validation_recall_at_8": recall,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result), flush=True)
        if recall > best_recall:
            best_recall = recall
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    assert best_state is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "captured_at_utc": utc_now(),
            "model_metadata": model.metadata(),
            "state_dict": best_state,
            "best_validation_recall_at_8": best_recall,
            "history": history,
            "seed": args.seed,
            "layer_weights": layer_weights.cpu(),
        },
        args.output,
    )
    atomic_json(
        args.output.with_suffix(".json"),
        {
            "captured_at_utc": utc_now(),
            "model_metadata": model.metadata(),
            "best_validation_recall_at_8": best_recall,
            "history": history,
            "seed": args.seed,
            "training_pairs": len(train),
            "validation_pairs": len(validation),
        },
    )


if __name__ == "__main__":
    main()
