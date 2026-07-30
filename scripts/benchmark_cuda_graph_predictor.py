#!/usr/bin/env python3
"""Compare eager predictor execution with a static CUDA Graph replay."""
from __future__ import annotations

import argparse
import statistics
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
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2000)
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
    static_hidden = torch.empty_like(hidden)
    static_layer = torch.empty_like(layer)

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream), torch.inference_mode():
        static_hidden.copy_(hidden)
        static_layer.copy_(layer)
        for _ in range(20):
            model(static_hidden, static_layer)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.inference_mode():
        graph_output = model(static_hidden, static_layer)

    eager = lambda: model(hidden, layer)

    def graph_replay():
        graph.replay()
        return graph_output

    def copy_and_graph_replay():
        static_hidden.copy_(hidden)
        static_layer.copy_(layer)
        graph.replay()
        return graph_output

    with torch.inference_mode():
        reference = eager()
        copy_and_graph_replay()
        difference = graph_output.float() - reference.float()

    report = {
        "captured_at_utc": utc_now(),
        "device": torch.cuda.get_device_name(),
        "dtype": "float32",
        "batch_size": 1,
        "architecture": "layer_aware",
        "width": args.width,
        "layer_embedding_width": args.layer_embedding_width,
        "parameters": model.metadata()["parameters"],
        "correctness": {
            "max_abs_error": difference.abs().max().item(),
            "root_mean_square_error": difference.square().mean().sqrt().item(),
        },
        "eager": measure(eager, args.warmup, args.iterations),
        "graph_replay_only": measure(
            graph_replay, args.warmup, args.iterations
        ),
        "device_copy_and_graph_replay": measure(
            copy_and_graph_replay, args.warmup, args.iterations
        ),
    }
    atomic_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
