#!/usr/bin/env python3
"""Compare eager and torch.compile execution of the layer-aware predictor."""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from specstream.io import atomic_json, utc_now
from specstream.predictor import LayerwiseExpertPredictor


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
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    model = LayerwiseExpertPredictor(
        args.hidden_size,
        args.num_layers,
        args.num_experts,
        args.width,
        architecture="layer_aware",
        layer_embedding_width=args.layer_embedding_width,
    ).cuda().eval()
    hidden = torch.randn(1, args.hidden_size, device="cuda")
    layer = torch.tensor([1], device="cuda")
    compiled = torch.compile(
        model,
        mode="reduce-overhead",
        fullgraph=True,
        dynamic=False,
    )

    eager = lambda: model(hidden, layer)
    compiled_operation = lambda: compiled(hidden, layer)
    compile_started = time.perf_counter()
    with torch.inference_mode():
        compiled_output = compiled_operation()
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_started
    with torch.inference_mode():
        reference = eager()
    difference = (compiled_output - reference).float()

    report = {
        "captured_at_utc": utc_now(),
        "device": torch.cuda.get_device_name(),
        "dtype": "float32",
        "batch_size": 1,
        "architecture": "layer_aware",
        "width": args.width,
        "layer_embedding_width": args.layer_embedding_width,
        "parameters": model.metadata()["parameters"],
        "compile_seconds": compile_seconds,
        "correctness": {
            "max_abs_error": difference.abs().max().item(),
            "root_mean_square_error": difference.square().mean().sqrt().item(),
        },
        "eager": measure(eager, args.warmup, args.iterations),
        "compiled": measure(
            compiled_operation, args.warmup, args.iterations
        ),
    }
    atomic_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
