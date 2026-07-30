#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from specstream.io import atomic_json, utc_now
from specstream.predictor import LayerwiseExpertPredictor
from specstream.traces import load_split, recall_at_k


def popularity(directory: Path, layers: int, experts: int) -> torch.Tensor:
    counts = torch.zeros(layers, experts, dtype=torch.float64)
    for path in sorted(directory.glob("*.pt")):
        trace = torch.load(path, map_location="cpu", weights_only=False)
        if trace["split"] != "train":
            continue
        routes = trace["route_ids"].long()
        for layer in range(routes.shape[1]):
            counts[layer].scatter_add_(
                0,
                routes[:, layer].reshape(-1),
                torch.ones(routes[:, layer].numel(), dtype=torch.float64),
            )
    return torch.log1p(counts).float()


def previous_routes(
    directory: Path,
    popularity_scores: torch.Tensor,
    target_horizon: int,
) -> torch.Tensor:
    score_parts = []
    for path in sorted(directory.glob("*.pt")):
        trace = torch.load(path, map_location="cpu", weights_only=False)
        if trace["split"] != "test":
            continue
        routes = trace["route_ids"].long()
        steps, layers, _ = routes.shape
        target_layers = range(target_horizon, layers)
        scores = popularity_scores[None, target_horizon:].expand(
            steps, -1, -1
        ).clone()
        for output_layer, layer in enumerate(target_layers):
            if layer == 0:
                continue
            previous = routes[:, layer - 1]
            scores[:, output_layer].scatter_(
                1,
                previous,
                100.0
                + torch.arange(previous.shape[-1], 0, -1, dtype=torch.float32)[
                    None
                ].expand(steps, -1),
            )
        score_parts.append(scores.reshape(steps * len(target_layers), -1))
    return torch.cat(score_parts)


def prompt_summary(values: torch.Tensor, prompt_index: torch.Tensor) -> np.ndarray:
    result = []
    for prompt in range(prompt_index.max().item() + 1):
        result.append(values[prompt_index == prompt].mean().item())
    return np.asarray(result)


def interval(values: np.ndarray, rng: np.random.Generator, resamples: int) -> dict:
    means = np.empty(resamples)
    for index in range(resamples):
        selection = rng.integers(0, len(values), len(values))
        means[index] = values[selection].mean()
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "prompt_count": int(len(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4, 8, 12, 16])
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=260724787)
    parser.add_argument(
        "--inference-dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint["model_metadata"]
    feature_key = metadata.get("trace_feature_key", "features")
    target_horizon = metadata.get("target_horizon", 0)
    test = load_split(
        args.traces,
        "test",
        feature_key=feature_key,
        target_horizon=target_horizon,
    )
    model = LayerwiseExpertPredictor(
        metadata["hidden_size"],
        metadata["num_layers"],
        metadata["num_experts"],
        metadata["width"],
        architecture=metadata.get("architecture", "layer_aware"),
        layer_embedding_width=metadata.get(
            "layer_embedding_width", metadata["width"]
        ),
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtypes = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    inference_dtype = dtypes[args.inference_dtype]
    model.to(device=device, dtype=inference_dtype).eval()

    learned_parts = []
    with torch.inference_mode():
        for start in range(0, len(test), args.batch_size):
            stop = min(start + args.batch_size, len(test))
            learned_parts.append(
                model(
                    test.hidden[start:stop].to(device, inference_dtype),
                    test.layer[start:stop].to(device),
                ).cpu()
            )
    learned = torch.cat(learned_parts)
    popular_by_layer = popularity(
        args.traces, metadata["num_layers"], metadata["num_experts"]
    )
    popular = popular_by_layer[test.layer]
    previous = previous_routes(args.traces, popular_by_layer, target_horizon)
    generator = torch.Generator().manual_seed(args.seed)
    random_scores = torch.rand(learned.shape, generator=generator)
    oracle = torch.zeros_like(learned).scatter_(1, test.targets, 1.0)

    policies = {
        "random": random_scores,
        "training_popularity": popular,
        "previous_layer_routes": previous,
        "learned": learned,
        "oracle": oracle,
    }
    rng = np.random.default_rng(args.seed)
    report = {
        "captured_at_utc": utc_now(),
        "checkpoint": str(args.checkpoint),
        "test_pairs": len(test),
        "test_prompts": len(test.prompt_ids),
        "trace_feature_key": feature_key,
        "target_horizon": target_horizon,
        "inference_dtype": args.inference_dtype,
        "budgets": args.budgets,
        "policies": {},
        "paired_differences_learned_minus_baseline": {},
    }
    rows = []
    prompt_values: dict[tuple[str, int], np.ndarray] = {}
    for policy_name, scores in policies.items():
        report["policies"][policy_name] = {}
        for budget in args.budgets:
            pair_recall = recall_at_k(scores, test.targets, budget)
            values = prompt_summary(pair_recall, test.prompt_index)
            prompt_values[(policy_name, budget)] = values
            stats = interval(values, rng, args.bootstrap_resamples)
            stats["by_source"] = {}
            prompt_sources = np.asarray(test.prompt_sources)
            for source in sorted(set(test.prompt_sources)):
                stats["by_source"][source] = interval(
                    values[prompt_sources == source],
                    rng,
                    args.bootstrap_resamples,
                )
            report["policies"][policy_name][str(budget)] = stats
            rows.append(
                {
                    "policy": policy_name,
                    "budget": budget,
                    **{key: value for key, value in stats.items() if key != "by_source"},
                }
            )

    for baseline in ("random", "training_popularity", "previous_layer_routes"):
        report["paired_differences_learned_minus_baseline"][baseline] = {}
        for budget in args.budgets:
            difference = (
                prompt_values[("learned", budget)]
                - prompt_values[(baseline, budget)]
            )
            stats = interval(difference, rng, args.bootstrap_resamples)
            stats["by_source"] = {}
            prompt_sources = np.asarray(test.prompt_sources)
            for source in sorted(set(test.prompt_sources)):
                stats["by_source"][source] = interval(
                    difference[prompt_sources == source],
                    rng,
                    args.bootstrap_resamples,
                )
            report["paired_differences_learned_minus_baseline"][baseline][
                str(budget)
            ] = stats

    atomic_json(args.output, report)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
