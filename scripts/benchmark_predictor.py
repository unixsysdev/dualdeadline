#!/usr/bin/env python3
"""Measure batch-one predictor latency on the target accelerator."""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from specstream.io import atomic_json, utc_now
from specstream.predictor import LayerwiseExpertPredictor


def summarize(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": ordered[round(0.50 * (len(ordered) - 1))],
        "p95_ms": ordered[round(0.95 * (len(ordered) - 1))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples": len(ordered),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument(
        "--architecture",
        choices=["layer_aware", "low_rank"],
        default="layer_aware",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda")
    results = []
    for width in args.widths:
        model = LayerwiseExpertPredictor(
            args.hidden_size,
            args.num_layers,
            args.num_experts,
            width,
            architecture=args.architecture,
        ).to(device).eval()
        hidden = torch.randn(1, args.hidden_size, device=device)
        layer = torch.tensor([1], device=device)
        values = []
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(hidden, layer)
            torch.cuda.synchronize()
            for _ in range(args.iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                model(hidden, layer)
                end.record()
                end.synchronize()
                values.append(start.elapsed_time(end))
        results.append(
            {
                "width": width,
                "parameters": model.metadata()["parameters"],
                **summarize(values),
            }
        )
        print(width, results[-1], flush=True)

    atomic_json(
        args.output,
        {
            "captured_at_utc": utc_now(),
            "device": torch.cuda.get_device_name(),
            "dtype": "float32",
            "architecture": args.architecture,
            "batch_size": 1,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "num_experts": args.num_experts,
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
