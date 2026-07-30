#!/usr/bin/env python3
"""Collect actual autoregressive decode routes using non-invasive hooks."""
from __future__ import annotations

import argparse
import json
import time
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-prompt-tokens", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=260724787)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=0,
        help="Call torch.cuda.empty_cache every N completed prompts (0 disables it).",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    if args.limit is not None:
        rows = rows[: args.limit]

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    load_options = {
        "dtype": torch.bfloat16,
        "device_map": {"": 0},
        "attn_implementation": "sdpa",
        "local_files_only": True,
    }
    if config.model_type == "qwen3_5_moe":
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        tokenizer = processor.tokenizer
        model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            args.model, **load_options
        ).eval()
        layers = model.model.language_model.layers
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        processor = tokenizer
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_options).eval()
        layers = model.model.layers
    if not all(hasattr(layer.mlp.gate, "forward") for layer in layers):
        raise TypeError(f"{config.model_type} does not expose the expected MoE routers")
    layer_inputs: list[list[torch.Tensor]] = [[] for _ in layers]
    route_ids: list[list[torch.Tensor]] = [[] for _ in layers]
    route_scores: list[list[torch.Tensor]] = [[] for _ in layers]
    handles = []

    def layer_pre_hook(layer_index: int):
        def hook(_module, inputs):
            hidden = inputs[0]
            if hidden.shape[-2] == 1:
                layer_inputs[layer_index].append(hidden[0, 0].detach())

        return hook

    def router_hook(layer_index: int):
        def hook(_module, _inputs, output):
            logits, scores, selected = output
            if logits.shape[0] == 1:
                route_ids[layer_index].append(selected[0].detach())
                route_scores[layer_index].append(scores[0].detach())

        return hook

    for layer_index, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(layer_pre_hook(layer_index)))
        handles.append(layer.mlp.gate.register_forward_hook(router_hook(layer_index)))

    completed = 0
    total_decode_steps = 0
    collection_started = time.perf_counter()
    try:
        with torch.inference_mode():
            for position, row in enumerate(rows, 1):
                output_path = args.output_dir / f"{row['id']}.pt"
                if args.resume and output_path.exists():
                    continue
                for values in (*layer_inputs, *route_ids, *route_scores):
                    values.clear()

                text = processor.apply_chat_template(
                    [{"role": "user", "content": row["user"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = processor(
                    text=text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.maximum_prompt_tokens,
                )
                prompt_length = inputs["input_ids"].shape[-1]
                inputs = {key: value.to("cuda") for key, value in inputs.items()}
                prompt_started = time.perf_counter()
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

                steps = min(len(values) for values in layer_inputs)
                if steps == 0:
                    raise RuntimeError("No single-token decode forwards were observed")
                features = torch.stack(
                    [torch.stack(values[:steps]) for values in layer_inputs], dim=1
                ).to("cpu", torch.bfloat16)
                experts = torch.stack(
                    [torch.stack(values[:steps]) for values in route_ids], dim=1
                ).to("cpu", torch.uint8)
                scores = torch.stack(
                    [torch.stack(values[:steps]) for values in route_scores], dim=1
                ).to("cpu", torch.float16)
                generated_ids = generated[0, prompt_length : prompt_length + steps].to("cpu")
                payload = {
                    "format_version": 2,
                    "captured_at_utc": utc_now(),
                    "id": row["id"],
                    "source": row["source"],
                    "split": row["split"],
                    "model_type": config.model_type,
                    "num_layers": len(layers),
                    "num_experts": config.text_config.num_experts
                    if hasattr(config, "text_config")
                    else config.num_experts,
                    "prompt_tokens": prompt_length,
                    "generated_ids": generated_ids,
                    "features": features,
                    "route_ids": experts,
                    "route_scores": scores,
                }
                temporary_path = output_path.with_suffix(".pt.tmp")
                torch.save(payload, temporary_path)
                temporary_path.replace(output_path)
                completed += 1
                total_decode_steps += steps
                prompt_seconds = time.perf_counter() - prompt_started
                elapsed_seconds = time.perf_counter() - collection_started
                print(
                    {
                        "position": position,
                        "records": len(rows),
                        "id": row["id"],
                        "split": row["split"],
                        "prompt_tokens": prompt_length,
                        "decode_steps": steps,
                        "prompt_seconds": round(prompt_seconds, 3),
                        "aggregate_decode_tokens_per_second": round(
                            total_decode_steps / elapsed_seconds, 3
                        ),
                    },
                    flush=True,
                )
                del generated, generated_ids, features, experts, scores
                if (
                    args.empty_cache_every > 0
                    and completed % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()

    atomic_json(
        args.output_dir / "manifest.json",
        {
            "captured_at_utc": utc_now(),
            "model": str(args.model),
            "model_type": config.model_type,
            "layers": len(layers),
            "experts": config.text_config.num_experts
            if hasattr(config, "text_config")
            else config.num_experts,
            "model_dtype": "bfloat16",
            "seed": args.seed,
            "maximum_prompt_tokens": args.maximum_prompt_tokens,
            "requested_new_tokens": args.new_tokens,
            "newly_completed": completed,
            "requested_records": len(rows),
            "total_decode_steps": total_decode_steps,
            "collection_seconds": time.perf_counter() - collection_started,
            "trace_semantics": "single-token autoregressive decode forwards only",
        },
    )


if __name__ == "__main__":
    main()
