#!/usr/bin/env python3
"""Benchmark exact post-training materialization of layer embeddings."""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from specstream.io import atomic_json, utc_now
from specstream.predictor import LayerwiseExpertPredictor
from specstream.traces import load_split, recall_at_k


def summarize(values: list[float]) -> dict[str, float | int]:
    values = sorted(values)
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": values[round(0.50 * (len(values) - 1))],
        "p95_ms": values[round(0.95 * (len(values) - 1))],
        "min_ms": values[0],
        "max_ms": values[-1],
        "samples": len(values),
    }


def measure(operation, warmup: int, iterations: int) -> dict[str, float | int]:
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize()
        values = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operation()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
    return summarize(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layer-embedding-width", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--traces", type=Path)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    model = LayerwiseExpertPredictor(
        args.hidden_size,
        args.num_layers,
        args.num_experts,
        args.width,
        architecture="layer_aware",
        layer_embedding_width=args.layer_embedding_width,
    ).cuda().eval()
    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])
    hidden_bf16 = torch.randn(
        1, args.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    layer = torch.tensor([1], device="cuda")
    assert model.layer_embedding is not None
    with torch.inference_mode():
        projected_layer_table = model.layer_projection(
            model.layer_embedding.weight
        ).contiguous()

    def eager():
        return model(hidden_bf16.float(), layer)

    def precomputed():
        state = model.hidden_projection(
            model.normalization(hidden_bf16.float())
        )
        state = torch.nn.functional.gelu(
            state + projected_layer_table[layer]
        )
        return model.output(state)

    with torch.inference_mode():
        reference = eager()
        optimized = precomputed()
        difference = optimized.float() - reference.float()

    held_out_equivalence = None
    if args.traces is not None:
        if checkpoint is None:
            raise ValueError("--traces requires --checkpoint")
        metadata = checkpoint["model_metadata"]
        test = load_split(
            args.traces,
            "test",
            feature_key=metadata.get("trace_feature_key", "features"),
            target_horizon=metadata.get("target_horizon", 0),
        )
        baseline_parts = []
        materialized_parts = []
        with torch.inference_mode():
            for start in range(0, len(test), args.batch_size):
                stop = min(start + args.batch_size, len(test))
                batch_hidden = test.hidden[start:stop].cuda()
                batch_layer = test.layer[start:stop].cuda()
                baseline_parts.append(
                    model(batch_hidden.float(), batch_layer).cpu()
                )
                state = model.hidden_projection(
                    model.normalization(batch_hidden.float())
                )
                state = torch.nn.functional.gelu(
                    state + projected_layer_table[batch_layer]
                )
                materialized_parts.append(model.output(state).cpu())
        baseline_logits = torch.cat(baseline_parts)
        materialized_logits = torch.cat(materialized_parts)
        logit_difference = (
            materialized_logits.float() - baseline_logits.float()
        )
        held_out_equivalence = {
            "pairs": len(test),
            "max_abs_logit_difference": (
                logit_difference.abs().max().item()
            ),
            "topk": {},
        }
        for budget in (2, 4, 8):
            baseline_topk = baseline_logits.topk(budget, dim=-1).indices
            materialized_topk = materialized_logits.topk(
                budget, dim=-1
            ).indices
            baseline_sorted = baseline_topk.sort(dim=-1).values
            materialized_sorted = materialized_topk.sort(dim=-1).values
            held_out_equivalence["topk"][str(budget)] = {
                "set_agreement": (
                    baseline_sorted == materialized_sorted
                ).all(dim=-1).float().mean().item(),
                "baseline_recall": recall_at_k(
                    baseline_logits, test.targets, budget
                ).mean().item(),
                "materialized_recall": recall_at_k(
                    materialized_logits, test.targets, budget
                ).mean().item(),
            }

    report = {
        "captured_at_utc": utc_now(),
        "device": torch.cuda.get_device_name(),
        "model_dtype": "float32",
        "source_dtype": "bfloat16",
        "batch_size": 1,
        "architecture": "layer_aware",
        "width": args.width,
        "layer_embedding_width": args.layer_embedding_width,
        "parameters": model.metadata()["parameters"],
        "materialized_layer_table_elements": projected_layer_table.numel(),
        "correctness": {
            "max_abs_error": difference.abs().max().item(),
            "root_mean_square_error": difference.square().mean().sqrt().item(),
        },
        "eager": measure(eager, args.warmup, args.iterations),
        "precomputed_layer_table": measure(
            precomputed, args.warmup, args.iterations
        ),
        "held_out_equivalence": held_out_equivalence,
    }
    atomic_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
