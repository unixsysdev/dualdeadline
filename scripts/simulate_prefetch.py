#!/usr/bin/env python3
"""Trace-driven comparison of monolithic and staged exact prefetch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

from specstream.io import atomic_json, utc_now
from specstream.predictor import LayerwiseExpertPredictor
from specstream.traces import load_split


def transfer_fit(path: Path) -> tuple[float, float]:
    report = json.loads(path.read_text())
    sizes = np.asarray([row["size_bytes"] for row in report["results"]], dtype=float)
    times = np.asarray([row["mean_ms"] for row in report["results"]], dtype=float)
    slope, intercept = np.polyfit(sizes, times, 1)
    return max(float(intercept), 0.0), max(float(slope), 0.0)


def popular_scores(directory: Path, layers: int, experts: int) -> torch.Tensor:
    counts = torch.zeros(layers, experts, dtype=torch.float64)
    for path in sorted(directory.glob("*.pt")):
        trace = torch.load(path, map_location="cpu", weights_only=False)
        if trace["split"] != "train":
            continue
        routes = trace["route_ids"].long()
        for layer in range(layers):
            values = routes[:, layer].reshape(-1)
            counts[layer].scatter_add_(
                0, values, torch.ones_like(values, dtype=torch.float64)
            )
    return torch.log1p(counts).float()


def prior_layer_scores(
    directory: Path, popular: torch.Tensor
) -> torch.Tensor:
    parts = []
    for path in sorted(directory.glob("*.pt")):
        trace = torch.load(path, map_location="cpu", weights_only=False)
        if trace["split"] != "test":
            continue
        routes = trace["route_ids"].long()
        steps, layers, top_k = routes.shape
        scores = popular[None].expand(steps, -1, -1).clone()
        previous = torch.roll(routes, 1, dims=1)
        bonus = 100.0 + torch.arange(top_k, 0, -1, dtype=torch.float32)
        for layer in range(1, layers):
            scores[:, layer].scatter_(1, previous[:, layer], bonus[None])
        parts.append(scores.reshape(steps * layers, -1))
    return torch.cat(parts)


def prompt_means(values: torch.Tensor, prompt_index: torch.Tensor) -> np.ndarray:
    return np.asarray(
        [
            values[prompt_index == prompt].mean().item()
            for prompt in range(prompt_index.max().item() + 1)
        ]
    )


def bootstrap(
    values: np.ndarray, rng: np.random.Generator, resamples: int
) -> dict[str, float]:
    selections = rng.integers(0, len(values), (resamples, len(values)))
    means = values[selections].mean(axis=1)
    return {
        "mean_ms": float(values.mean()),
        "ci_low_ms": float(np.quantile(means, 0.025)),
        "ci_high_ms": float(np.quantile(means, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--transfer-timing", type=Path, required=True)
    parser.add_argument("--model-timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--byte-budget-expert-equivalents",
        type=int,
        nargs="+",
        default=[0, 2, 4, 8, 12, 16],
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=260724787)
    args = parser.parse_args()

    test = load_split(args.traces, "test")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint["model_metadata"]
    predictor = LayerwiseExpertPredictor(
        metadata["hidden_size"],
        metadata["num_layers"],
        metadata["num_experts"],
        metadata["width"],
    )
    predictor.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor.to(device).eval()
    learned_parts = []
    with torch.inference_mode():
        for start in range(0, len(test), 2048):
            stop = min(start + 2048, len(test))
            learned_parts.append(
                predictor(
                    test.hidden[start:stop].to(device, torch.float32),
                    test.layer[start:stop].to(device),
                ).cpu()
            )
    learned = torch.cat(learned_parts)

    popular_by_layer = popular_scores(
        args.traces, metadata["num_layers"], metadata["num_experts"]
    )
    generator = torch.Generator().manual_seed(args.seed)
    policies = {
        "random": torch.rand(learned.shape, generator=generator),
        "training_popularity": popular_by_layer[test.layer],
        "previous_layer_routes": prior_layer_scores(args.traces, popular_by_layer),
        "learned": learned,
        "oracle": torch.zeros_like(learned).scatter_(1, test.targets, 1.0),
    }

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    text = config.text_config if hasattr(config, "text_config") else config
    expert_intermediate_size = getattr(
        text, "moe_intermediate_size", None
    ) or getattr(text, "intermediate_size")
    dtype_bytes = 2
    gate_up_bytes = (
        2 * text.hidden_size * expert_intermediate_size * dtype_bytes
    )
    down_bytes = text.hidden_size * expert_intermediate_size * dtype_bytes
    full_expert_bytes = gate_up_bytes + down_bytes
    latency_ms, ms_per_byte = transfer_fit(args.transfer_timing)

    def component_time(size: int) -> float:
        return latency_ms + ms_per_byte * size

    gate_up_transfer_ms = component_time(gate_up_bytes)
    down_transfer_ms = component_time(down_bytes)
    full_transfer_ms = component_time(full_expert_bytes)
    timing = json.loads(args.model_timing.read_text())
    pre_moe_ms = torch.tensor(
        [layer["pre_moe"]["p50_ms"] for layer in timing["layers"]]
    )
    gate_compute_ms = timing["expert_gate_up_compute"]["p50_ms"]
    top_k = test.targets.shape[-1]
    rng = np.random.default_rng(args.seed)
    report = {
        "captured_at_utc": utc_now(),
        "semantics": "trace-driven simulation; not an end-to-end serving measurement",
        "assumptions": {
            "dma_channels": 1,
            "persistent_cache": False,
            "prefetch_window": "measured layer-start to router invocation p50",
            "staged_policy": (
                "predict gate/up; after the authoritative router fires, transfer "
                "actual down projections during gate/up compute"
            ),
        },
        "bytes": {
            "gate_up_per_expert": gate_up_bytes,
            "down_per_expert": down_bytes,
            "full_per_expert": full_expert_bytes,
        },
        "transfer_model": {
            "intercept_ms": latency_ms,
            "ms_per_byte": ms_per_byte,
            "gate_up_ms": gate_up_transfer_ms,
            "down_ms": down_transfer_ms,
            "full_ms": full_transfer_ms,
        },
        "gate_up_compute_top_k_ms": gate_compute_ms,
        "results": {},
    }

    actual = test.targets
    slack = pre_moe_ms[test.layer]
    for policy_name, scores in policies.items():
        report["results"][policy_name] = {}
        ranking = scores.argsort(dim=-1, descending=True)
        for equivalents in args.byte_budget_expert_equivalents:
            byte_budget = equivalents * full_expert_bytes

            monolithic_candidates = min(
                equivalents, metadata["num_experts"]
            )
            monolithic_deadline = torch.floor(slack / full_transfer_ms).long()
            monolithic_ready = torch.minimum(
                monolithic_deadline,
                torch.full_like(monolithic_deadline, monolithic_candidates),
            )
            monolithic_hits = torch.zeros(len(test))
            for ready_count in torch.unique(monolithic_ready):
                mask = monolithic_ready == ready_count
                if ready_count:
                    selected = ranking[mask, : int(ready_count)]
                    monolithic_hits[mask] = (
                        selected[:, :, None] == actual[mask, None, :]
                    ).any(dim=1).sum(dim=-1)
            monolithic_stall = (top_k - monolithic_hits) * full_transfer_ms

            staged_candidates = min(
                byte_budget // gate_up_bytes, metadata["num_experts"]
            )
            staged_deadline = torch.floor(slack / gate_up_transfer_ms).long()
            staged_ready = torch.minimum(
                staged_deadline,
                torch.full_like(staged_deadline, staged_candidates),
            )
            staged_hits = torch.zeros(len(test))
            for ready_count in torch.unique(staged_ready):
                mask = staged_ready == ready_count
                if ready_count:
                    selected = ranking[mask, : int(ready_count)]
                    staged_hits[mask] = (
                        selected[:, :, None] == actual[mask, None, :]
                    ).any(dim=1).sum(dim=-1)
            missing_gate = top_k - staged_hits
            staged_stall = missing_gate * gate_up_transfer_ms + max(
                0.0, top_k * down_transfer_ms - gate_compute_ms
            )

            monolithic_prompt = prompt_means(
                monolithic_stall, test.prompt_index
            )
            staged_prompt = prompt_means(staged_stall, test.prompt_index)
            difference = staged_prompt - monolithic_prompt
            report["results"][policy_name][str(equivalents)] = {
                "byte_budget": byte_budget,
                "monolithic": bootstrap(
                    monolithic_prompt, rng, args.bootstrap_resamples
                ),
                "staged": bootstrap(staged_prompt, rng, args.bootstrap_resamples),
                "paired_staged_minus_monolithic": bootstrap(
                    difference, rng, args.bootstrap_resamples
                ),
            }

    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
