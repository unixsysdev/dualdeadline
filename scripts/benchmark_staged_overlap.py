#!/usr/bin/env python3
"""Measure actual copy/compute overlap with real MoE expert tensors."""
from __future__ import annotations

import argparse
import json
import statistics
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from specstream.io import atomic_json, utc_now


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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument(
        "--tokens",
        type=int,
        default=1,
        help="Token rows sharing the same transferred expert weights.",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()

    index = json.loads(
        (args.model / "model.safetensors.index.json").read_text()
    )["weight_map"]
    separate_prefix = f"model.layers.{args.layer}.mlp.experts"
    fused_prefix = (
        f"model.language_model.layers.{args.layer}.mlp.experts"
    )
    fused_names = {
        "gate_up": f"{fused_prefix}.gate_up_proj",
        "down": f"{fused_prefix}.down_proj",
    }
    fused_gate_up = all(name in index for name in fused_names.values())
    if fused_gate_up:
        tensor_names = list(fused_names.values())
    else:
        names = {
            projection: [
                f"{separate_prefix}.{expert}.{projection}_proj.weight"
                for expert in range(args.experts)
            ]
            for projection in ("gate", "up", "down")
        }
        missing = [
            name
            for values in names.values()
            for name in values
            if name not in index
        ]
        if missing:
            raise KeyError(
                "Unsupported expert checkpoint layout; first missing tensor: "
                f"{missing[0]}"
            )
        tensor_names = [name for values in names.values() for name in values]
    shards = {index[name] for name in tensor_names}
    with ExitStack() as stack:
        handles = {
            shard: stack.enter_context(
                safe_open(
                    args.model / shard, framework="pt", device="cpu"
                )
            )
            for shard in shards
        }
        if fused_gate_up:
            host = {}
            for projection, name in fused_names.items():
                tensor_slice = handles[index[name]].get_slice(name)
                host[projection] = [
                    tensor_slice[expert].contiguous().pin_memory()
                    for expert in range(args.experts)
                ]
        else:
            host = {
                projection: [
                    handles[index[name]]
                    .get_tensor(name)
                    .contiguous()
                    .pin_memory()
                    for name in values
                ]
                for projection, values in names.items()
            }
    device_weights = {
        projection: [
            torch.empty_like(weight, device="cuda") for weight in values
        ]
        for projection, values in host.items()
    }
    gate_key = "gate_up" if fused_gate_up else "gate"
    hidden_size = host[gate_key][0].shape[1]
    hidden = torch.randn(
        args.tokens,
        hidden_size,
        device="cuda",
        dtype=host[gate_key][0].dtype,
    )
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    def copy(projection: str, indices=range(args.experts)) -> None:
        for expert in indices:
            device_weights[projection][expert].copy_(
                host[projection][expert], non_blocking=True
            )

    def copy_gate_up(indices=range(args.experts)) -> None:
        if fused_gate_up:
            copy("gate_up", indices)
        else:
            copy("gate", indices)
            copy("up", indices)

    def gate_up(expert: int) -> tuple[torch.Tensor, torch.Tensor]:
        if fused_gate_up:
            return torch.nn.functional.linear(
                hidden, device_weights["gate_up"][expert]
            ).chunk(2, dim=-1)
        return (
            torch.nn.functional.linear(
                hidden, device_weights["gate"][expert]
            ),
            torch.nn.functional.linear(
                hidden, device_weights["up"][expert]
            ),
        )

    def gate_compute() -> list[torch.Tensor]:
        intermediate = []
        for expert in range(args.experts):
            gate, up = gate_up(expert)
            intermediate.append(torch.nn.functional.silu(gate) * up)
        return intermediate

    def gate_compute_into(
        indices,
        intermediate: list[torch.Tensor | None],
    ) -> None:
        for expert in indices:
            gate, up = gate_up(expert)
            intermediate[expert] = torch.nn.functional.silu(gate) * up

    def down_compute(
        intermediate: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        return [
            torch.nn.functional.linear(
                value, device_weights["down"][expert]
            )
            for expert, value in enumerate(intermediate)
        ]

    with torch.cuda.stream(copy_stream):
        copy_gate_up()
        copy("down")
    copy_stream.synchronize()

    def resident() -> tuple[torch.cuda.Event, torch.cuda.Event, list[torch.Tensor]]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(compute_stream):
            start.record()
            output = down_compute(gate_compute())
            end.record()
        return start, end, output

    def monolithic() -> tuple[torch.cuda.Event, torch.cuda.Event, list[torch.Tensor]]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(compute_stream):
            start.record()
            copy_gate_up()
            copy("down")
            output = down_compute(gate_compute())
            end.record()
        return start, end, output

    def staged_serial() -> tuple[torch.cuda.Event, torch.cuda.Event, list[torch.Tensor]]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(compute_stream):
            start.record()
            copy_gate_up()
            intermediate = gate_compute()
            copy("down")
            output = down_compute(intermediate)
            end.record()
        return start, end, output

    def staged_overlap() -> tuple[torch.cuda.Event, torch.cuda.Event, list[torch.Tensor]]:
        start = torch.cuda.Event(enable_timing=True)
        gate_ready = torch.cuda.Event()
        down_ready = torch.cuda.Event()
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(copy_stream):
            start.record()
            copy_gate_up()
            gate_ready.record()
            copy("down")
            down_ready.record()
        with torch.cuda.stream(compute_stream):
            compute_stream.wait_event(gate_ready)
            intermediate = gate_compute()
            compute_stream.wait_event(down_ready)
            output = down_compute(intermediate)
            end.record()
        return start, end, output

    def prefetched_gate_overlap(
    ) -> tuple[torch.cuda.Event, torch.cuda.Event, list[torch.Tensor]]:
        with torch.cuda.stream(copy_stream):
            copy_gate_up()
        copy_stream.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        down_ready = torch.cuda.Event()
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(copy_stream):
            start.record()
            copy("down")
            down_ready.record()
        with torch.cuda.stream(compute_stream):
            compute_stream.wait_event(start)
            intermediate = gate_compute()
            compute_stream.wait_event(down_ready)
            output = down_compute(intermediate)
            end.record()
        return start, end, output

    def make_partial_prefetched_overlap(prefetched: int):
        def operation(
        ) -> tuple[torch.cuda.Event, torch.cuda.Event, list[torch.Tensor]]:
            with torch.cuda.stream(copy_stream):
                copy_gate_up(range(prefetched))
            copy_stream.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            missing_gate_ready = torch.cuda.Event()
            down_ready = torch.cuda.Event()
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(copy_stream):
                start.record()
                copy_gate_up(range(prefetched, args.experts))
                missing_gate_ready.record()
                copy("down")
                down_ready.record()
            intermediate: list[torch.Tensor | None] = [None] * args.experts
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(start)
                gate_compute_into(range(prefetched), intermediate)
                compute_stream.wait_event(missing_gate_ready)
                gate_compute_into(range(prefetched, args.experts), intermediate)
                compute_stream.wait_event(down_ready)
                output = down_compute(
                    [value for value in intermediate if value is not None]
                )
                end.record()
            return start, end, output

        return operation

    strategies = {
        "resident_compute": resident,
        "monolithic_copy_then_compute": monolithic,
        "staged_serial": staged_serial,
        "staged_overlap": staged_overlap,
        "prefetched_gate_overlap": prefetched_gate_overlap,
    }
    for prefetched in (2, 4, 6):
        if prefetched >= args.experts:
            continue
        strategies[
            f"partial_prefetched_gate_{prefetched}"
        ] = make_partial_prefetched_overlap(prefetched)
    timings = {}
    outputs = {}
    with torch.inference_mode():
        for name, operation in strategies.items():
            for _ in range(args.warmup):
                _, warmup_end, _ = operation()
                warmup_end.synchronize()
            values = []
            output = None
            for _ in range(args.iterations):
                start, end, output = operation()
                end.synchronize()
                values.append(start.elapsed_time(end))
            timings[name] = summary(values)
            assert output is not None
            outputs[name] = torch.stack(output).detach().cpu()
            print(name, timings[name], flush=True)

    reference = outputs["resident_compute"].float()
    correctness = {}
    for name, output in outputs.items():
        difference = output.float() - reference
        correctness[name] = {
            "max_abs_error": difference.abs().max().item(),
            "root_mean_square_error": difference.square().mean().sqrt().item(),
        }

    resident_p50 = timings["resident_compute"]["p50_ms"]
    exposed = {
        name: values["p50_ms"] - resident_p50
        for name, values in timings.items()
    }
    atomic_json(
        args.output,
        {
            "captured_at_utc": utc_now(),
            "device": torch.cuda.get_device_name(),
            "model": str(args.model),
            "layer": args.layer,
            "experts": args.experts,
            "tokens": args.tokens,
            "expert_checkpoint_layout": (
                "fused_gate_up_3d" if fused_gate_up else "separate_per_expert"
            ),
            "dtype": str(hidden.dtype),
            "pinned_host_memory": True,
            "timings": timings,
            "exposed_over_resident_p50_ms": exposed,
            "correctness": correctness,
        },
    )


if __name__ == "__main__":
    main()
