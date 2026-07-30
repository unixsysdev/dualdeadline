#!/usr/bin/env python3
"""Benchmark a two-kernel Triton implementation of the layer-aware predictor."""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl

from specstream.io import atomic_json, utc_now
from specstream.predictor import LayerwiseExpertPredictor
from specstream.traces import load_split


@triton.jit
def layernorm_projection_gelu_kernel(
    hidden_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    projection_ptr,
    layer_table_ptr,
    layer_ptr,
    state_ptr,
    hidden_size: tl.constexpr,
    width: tl.constexpr,
    block_hidden: tl.constexpr,
):
    row = tl.program_id(0)
    hidden_offsets = tl.arange(0, block_hidden)
    hidden_mask = hidden_offsets < hidden_size
    hidden = tl.load(
        hidden_ptr + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(hidden, axis=0) / hidden_size
    centered = tl.where(hidden_mask, hidden - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / hidden_size
    normalized = centered * tl.rsqrt(variance + 1.0e-5)
    normalized = (
        normalized
        * tl.load(
            norm_weight_ptr + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        )
        + tl.load(
            norm_bias_ptr + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        )
    )
    projection = tl.load(
        projection_ptr + row * hidden_size + hidden_offsets,
        mask=(row < width) & hidden_mask,
        other=0.0,
    )
    value = tl.sum(normalized * projection, axis=0)
    layer = tl.load(layer_ptr)
    value += tl.load(
        layer_table_ptr + layer * width + row,
        mask=row < width,
        other=0.0,
    )
    value = 0.5 * value * (
        1.0 + tl.erf(value * 0.7071067811865476)
    )
    tl.store(state_ptr + row, value, mask=row < width)


@triton.jit
def output_projection_kernel(
    state_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    width: tl.constexpr,
    num_experts: tl.constexpr,
    block_width: tl.constexpr,
):
    expert = tl.program_id(0)
    offsets = tl.arange(0, block_width)
    mask = offsets < width
    state = tl.load(state_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(
        weight_ptr + expert * width + offsets,
        mask=(expert < num_experts) & mask,
        other=0.0,
    )
    value = tl.sum(state * weight, axis=0)
    value += tl.load(bias_ptr + expert, mask=expert < num_experts)
    tl.store(output_ptr + expert, value, mask=expert < num_experts)


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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layer-embedding-width", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--traces", type=Path)
    parser.add_argument(
        "--validation-pairs",
        type=int,
        default=0,
        help="Pairs to validate (0 means the full held-out split).",
    )
    args = parser.parse_args()

    model = LayerwiseExpertPredictor(
        args.hidden_size,
        args.num_layers,
        args.num_experts,
        args.width,
        architecture="layer_aware",
        layer_embedding_width=args.layer_embedding_width,
    ).cuda().eval()
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    assert model.layer_embedding is not None
    assert isinstance(model.normalization, torch.nn.LayerNorm)
    with torch.inference_mode():
        layer_table = model.layer_projection(
            model.layer_embedding.weight
        ).contiguous()

    hidden = torch.randn(
        1, args.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    layer = torch.tensor([1], device="cuda", dtype=torch.int64)
    state = torch.empty(args.width, device="cuda", dtype=torch.float32)
    output = torch.empty(
        args.num_experts, device="cuda", dtype=torch.float32
    )

    def eager():
        return model(hidden.float(), layer)

    def fused():
        layernorm_projection_gelu_kernel[(args.width,)](
            hidden,
            model.normalization.weight,
            model.normalization.bias,
            model.hidden_projection.weight,
            layer_table,
            layer,
            state,
            hidden_size=args.hidden_size,
            width=args.width,
            block_hidden=triton.next_power_of_2(args.hidden_size),
            num_warps=4,
        )
        output_projection_kernel[(args.num_experts,)](
            state,
            model.output.weight,
            model.output.bias,
            output,
            width=args.width,
            num_experts=args.num_experts,
            block_width=triton.next_power_of_2(args.width),
            num_warps=4,
        )
        return output[None]

    with torch.inference_mode():
        reference = eager()
        candidate = fused()
        torch.cuda.synchronize()
        difference = candidate.float() - reference.float()

    held_out_equivalence = None
    if args.traces is not None:
        metadata = checkpoint["model_metadata"]
        test = load_split(
            args.traces,
            "test",
            feature_key=metadata.get("trace_feature_key", "features"),
            target_horizon=metadata.get("target_horizon", 0),
        )
        pairs = len(test)
        if args.validation_pairs > 0:
            pairs = min(pairs, args.validation_pairs)
        source_hidden = test.hidden[:pairs].pin_memory()
        source_layers = test.layer[:pairs].pin_memory()
        agreement_counts = {2: 0, 4: 0, 8: 0}
        maximum_error = 0.0
        pending_agreements: dict[int, list[torch.Tensor]] = {
            2: [],
            4: [],
            8: [],
        }
        pending_errors = []
        with torch.inference_mode():
            for index in range(pairs):
                hidden.copy_(
                    source_hidden[index : index + 1],
                    non_blocking=True,
                )
                layer.copy_(
                    source_layers[index : index + 1],
                    non_blocking=True,
                )
                reference_logits = eager()
                fused_logits = fused()
                pending_errors.append(
                    (fused_logits.float() - reference_logits.float())
                    .abs()
                    .max()
                )
                for budget in pending_agreements:
                    reference_topk = (
                        reference_logits.topk(budget, dim=-1)
                        .indices.sort(dim=-1)
                        .values
                    )
                    fused_topk = (
                        fused_logits.topk(budget, dim=-1)
                        .indices.sort(dim=-1)
                        .values
                    )
                    pending_agreements[budget].append(
                        (reference_topk == fused_topk).all()
                    )
                if len(pending_errors) == 512 or index + 1 == pairs:
                    maximum_error = max(
                        maximum_error,
                        torch.stack(pending_errors).max().item(),
                    )
                    pending_errors.clear()
                    for budget, values in pending_agreements.items():
                        agreement_counts[budget] += int(
                            torch.stack(values).sum().item()
                        )
                        values.clear()
        held_out_equivalence = {
            "pairs": pairs,
            "max_abs_logit_difference": maximum_error,
            "topk_set_agreement": {
                str(budget): agreement_counts[budget] / pairs
                for budget in agreement_counts
            },
        }

    report = {
        "captured_at_utc": utc_now(),
        "device": torch.cuda.get_device_name(),
        "source_dtype": "bfloat16",
        "model_dtype": "float32",
        "batch_size": 1,
        "hidden_size": args.hidden_size,
        "width": args.width,
        "num_experts": args.num_experts,
        "parameters": model.metadata()["parameters"],
        "correctness": {
            "max_abs_error": difference.abs().max().item(),
            "root_mean_square_error": difference.square().mean().sqrt().item(),
            "top8_set_agreement": (
                candidate.topk(8, dim=-1).indices.sort(dim=-1).values
                == reference.topk(8, dim=-1).indices.sort(dim=-1).values
            ).all().item(),
        },
        "eager": measure(eager, args.warmup, args.iterations),
        "fused_triton": measure(fused, args.warmup, args.iterations),
        "held_out_equivalence": held_out_equivalence,
    }
    atomic_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
