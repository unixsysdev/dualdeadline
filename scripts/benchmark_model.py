#!/usr/bin/env python3
"""Measure decode-layer windows and the two staged expert compute phases."""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen3_5MoeForConditionalGeneration,
)

from specstream.io import atomic_json, seed_everything, utc_now


def summary(values: list[float]) -> dict[str, float]:
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
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--measurement-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=260724787)
    args = parser.parse_args()
    seed_everything(args.seed)

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    options = dict(
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
    )
    if config.model_type == "qwen3_5_moe":
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        tokenizer = processor.tokenizer
        model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            args.model, **options
        ).eval()
        layers = model.model.language_model.layers
        text_config = config.text_config
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        processor = tokenizer
        model = AutoModelForCausalLM.from_pretrained(args.model, **options).eval()
        layers = model.model.layers
        text_config = config

    text = processor.apply_chat_template(
        [{"role": "user", "content": "Explain why the sky appears blue."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = {
        key: value.to("cuda")
        for key, value in processor(text=text, return_tensors="pt").items()
    }
    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=args.warmup_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()

    starts: list[list[torch.cuda.Event]] = [[] for _ in layers]
    gates: list[list[torch.cuda.Event]] = [[] for _ in layers]
    ends: list[list[torch.cuda.Event]] = [[] for _ in layers]
    handles = []

    def record(collection, layer_index):
        def pre_hook(_module, inputs):
            hidden = inputs[0]
            if hidden.shape[-2] == 1:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                collection[layer_index].append(event)

        return pre_hook

    def record_end(layer_index):
        def hook(_module, inputs, _output):
            hidden = inputs[0]
            if hidden.shape[-2] == 1:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                ends[layer_index].append(event)

        return hook

    for layer_index, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(record(starts, layer_index)))
        handles.append(layer.mlp.gate.register_forward_pre_hook(record(gates, layer_index)))
        handles.append(layer.register_forward_hook(record_end(layer_index)))

    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=args.measurement_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    for handle in handles:
        handle.remove()

    layer_results = []
    for layer_index in range(len(layers)):
        count = min(
            len(starts[layer_index]), len(gates[layer_index]), len(ends[layer_index])
        )
        pre_moe = [
            starts[layer_index][step].elapsed_time(gates[layer_index][step])
            for step in range(count)
        ]
        moe = [
            gates[layer_index][step].elapsed_time(ends[layer_index][step])
            for step in range(count)
        ]
        total = [
            starts[layer_index][step].elapsed_time(ends[layer_index][step])
            for step in range(count)
        ]
        layer_results.append(
            {
                "layer": layer_index,
                "layer_type": getattr(text_config, "layer_types", [None] * len(layers))[
                    layer_index
                ],
                "pre_moe": summary(pre_moe),
                "moe_and_residual": summary(moe),
                "total": summary(total),
            }
        )

    # The model stores the expert's fused gate/up matrix separately from down.
    # Time those phases with eight routed experts to expose the second deadline.
    expert_module = layers[0].mlp.experts
    selected = list(range(text_config.num_experts_per_tok))
    hidden = torch.randn(
        1, text_config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    phase_gate_up = []
    phase_down = []
    with torch.inference_mode():
        for iteration in range(220):
            start = torch.cuda.Event(enable_timing=True)
            middle = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            intermediates = []
            for expert in selected:
                gate, up = torch.nn.functional.linear(
                    hidden, expert_module.gate_up_proj[expert]
                ).chunk(2, dim=-1)
                intermediates.append(torch.nn.functional.silu(gate) * up)
            middle.record()
            outputs = [
                torch.nn.functional.linear(
                    intermediate, expert_module.down_proj[expert]
                )
                for expert, intermediate in zip(selected, intermediates)
            ]
            end.record()
            end.synchronize()
            if iteration >= 20:
                phase_gate_up.append(start.elapsed_time(middle))
                phase_down.append(middle.elapsed_time(end))
            del outputs

    report = {
        "captured_at_utc": utc_now(),
        "model": str(args.model),
        "model_type": config.model_type,
        "device": torch.cuda.get_device_name(),
        "measurement_tokens_requested": args.measurement_tokens,
        "layers": layer_results,
        "expert_compute_top_k": text_config.num_experts_per_tok,
        "expert_gate_up_compute": summary(phase_gate_up),
        "expert_down_compute": summary(phase_down),
    }
    atomic_json(args.output, report)
    print(f"Wrote timing for {len(layer_results)} layers to {args.output}")


if __name__ == "__main__":
    main()

