#!/usr/bin/env python3
"""Measure the transfer primitive used by the trace-driven simulator."""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from specstream.io import atomic_json, utc_now


MIB = 1024 * 1024


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def at(q: float) -> float:
        return ordered[round(q * (len(ordered) - 1))]

    return {
        "mean_ms": statistics.fmean(ordered),
        "std_ms": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "p50_ms": at(0.50),
        "p95_ms": at(0.95),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def measure(size_bytes: int, repetitions: int, warmup: int) -> dict:
    source = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
    destination = torch.empty(size_bytes, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    samples: list[float] = []
    for iteration in range(warmup + repetitions):
        with torch.cuda.stream(stream):
            start.record(stream)
            destination.copy_(source, non_blocking=True)
            end.record(stream)
        end.synchronize()
        elapsed = start.elapsed_time(end)
        if iteration >= warmup:
            samples.append(elapsed)

    stats = quantiles(samples)
    stats.update(
        {
            "size_bytes": size_bytes,
            "effective_mean_gbps": size_bytes / stats["mean_ms"] / 1e6,
            "repetitions": repetitions,
        }
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--sizes-mib",
        type=float,
        nargs="+",
        default=[0.5, 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64],
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    torch.cuda.set_device(0)
    results = [
        measure(round(size_mib * MIB), args.repetitions, args.warmup)
        for size_mib in args.sizes_mib
    ]
    report = {
        "captured_at_utc": utc_now(),
        "device": torch.cuda.get_device_name(0),
        "method": "pinned host to preallocated device, nonblocking copy, dedicated stream",
        "results": results,
    }
    atomic_json(args.output, report)
    for row in results:
        print(
            f"{row['size_bytes'] / MIB:6.1f} MiB  "
            f"{row['mean_ms']:8.4f} ms  "
            f"{row['effective_mean_gbps']:7.2f} GB/s"
        )


if __name__ == "__main__":
    main()

