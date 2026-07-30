#!/usr/bin/env python3
"""Benchmark exact algebraic and Triton fusion of the linear low-rank adapter."""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl

from specstream.io import atomic_json, utc_now
from specstream.predictor import LayerwiseExpertPredictor


@triton.jit
def dense_gemv_kernel(
    hidden_ptr,
    weight_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    num_experts: tl.constexpr,
    block_hidden: tl.constexpr,
    block_experts: tl.constexpr,
):
    expert_offsets = (
        tl.program_id(0) * block_experts + tl.arange(0, block_experts)
    )
    hidden_offsets = tl.arange(0, block_hidden)
    hidden = tl.load(
        hidden_ptr + hidden_offsets,
        mask=hidden_offsets < hidden_size,
        other=0.0,
    )
    weights = tl.load(
        weight_ptr
        + expert_offsets[:, None] * hidden_size
        + hidden_offsets[None, :],
        mask=(expert_offsets[:, None] < num_experts)
        & (hidden_offsets[None, :] < hidden_size),
        other=0.0,
    )
    values = tl.sum(weights * hidden[None, :], axis=1)
    tl.store(
        output_ptr + expert_offsets,
        values,
        mask=expert_offsets < num_experts,
    )


def summary(values: list[float]) -> dict[str, float | int]:
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
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    model = LayerwiseExpertPredictor(
        args.hidden_size,
        args.num_layers,
        args.num_experts,
        args.width,
        architecture="low_rank",
    ).cuda().eval()
    hidden = torch.randn(1, args.hidden_size, device="cuda")
    layer = torch.tensor([1], device="cuda")
    fused_weight = (
        model.output.weight @ model.hidden_projection.weight
    ).contiguous()
    triton_output = torch.empty(args.num_experts, device="cuda")
    def factorized() -> torch.Tensor:
        return model(hidden, layer)

    def fused_torch() -> torch.Tensor:
        return torch.nn.functional.linear(hidden, fused_weight)

    def make_fused_triton(block_experts: int, num_warps: int):
        grid = (triton.cdiv(args.num_experts, block_experts),)

        def operation() -> torch.Tensor:
            dense_gemv_kernel[grid](
                hidden,
                fused_weight,
                triton_output,
                hidden_size=args.hidden_size,
                num_experts=args.num_experts,
                block_hidden=triton.next_power_of_2(args.hidden_size),
                block_experts=block_experts,
                num_warps=num_warps,
            )
            return triton_output[None]

        return operation

    implementations = {
        "factorized_torch": factorized,
        "materialized_torch": fused_torch,
    }
    for block_experts, num_warps in (
        (1, 4),
        (2, 4),
        (4, 4),
        (4, 8),
        (8, 4),
        (8, 8),
        (16, 8),
    ):
        implementations[
            f"triton_e{block_experts}_w{num_warps}"
        ] = make_fused_triton(block_experts, num_warps)
    with torch.inference_mode():
        reference = factorized().clone()
        correctness = {}
        timings = {}
        for name, operation in implementations.items():
            output = operation()
            torch.cuda.synchronize()
            difference = (output - reference).float()
            correctness[name] = {
                "max_abs_error": difference.abs().max().item(),
                "root_mean_square_error": difference.square().mean().sqrt().item(),
            }
            for _ in range(args.warmup):
                operation()
            torch.cuda.synchronize()
            values = []
            for _ in range(args.iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                operation()
                end.record()
                end.synchronize()
                values.append(start.elapsed_time(end))
            timings[name] = summary(values)
            print(name, timings[name], correctness[name], flush=True)

    atomic_json(
        args.output,
        {
            "captured_at_utc": utc_now(),
            "device": torch.cuda.get_device_name(),
            "dtype": "float32",
            "batch_size": 1,
            "hidden_size": args.hidden_size,
            "num_experts": args.num_experts,
            "width": args.width,
            "factorized_parameters": model.metadata()["parameters"],
            "materialized_parameters": fused_weight.numel(),
            "timings": timings,
            "correctness": correctness,
        },
    )


if __name__ == "__main__":
    main()
