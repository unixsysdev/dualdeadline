#!/usr/bin/env python3
"""Collect exact Qwen router traces and sparse predictor-training features."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

from specstream.io import atomic_json, seed_everything, utc_now


def language_config(model):
    return model.config.text_config


def tokenize(processor, row: dict, maximum_tokens: int) -> dict[str, torch.Tensor]:
    text = processor.apply_chat_template(
        [
            {"role": "user", "content": row["user"]},
            {"role": "assistant", "content": row["assistant"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    return processor(
        text=text,
        return_tensors="pt",
        truncation=True,
        max_length=maximum_tokens,
    )


def sample_features(
    hidden_states: tuple[torch.Tensor, ...],
    route_ids: torch.Tensor,
    maximum_pairs: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # hidden_states[0] is embeddings and hidden_states[l+1] is the state exposed
    # after layer l. It can predict the router decision at layer l+1 while that
    # layer's token mixer is executing.
    layer_count, token_count, _ = route_ids.shape
    candidates = [
        (layer, token)
        for layer in range(layer_count - 1)
        for token in range(token_count)
    ]
    rng.shuffle(candidates)
    chosen = sorted(candidates[:maximum_pairs])
    indices = torch.tensor(chosen, dtype=torch.int16)
    features = torch.stack(
        [
            hidden_states[layer + 1][0, token].detach().to("cpu", torch.bfloat16)
            for layer, token in chosen
        ]
    )
    targets = torch.stack([route_ids[layer + 1, token] for layer, token in chosen])
    return indices, features, targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-tokens", type=int, default=512)
    parser.add_argument("--pairs-per-prompt", type=int, default=512)
    parser.add_argument("--seed", type=int, default=260724787)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    if args.limit is not None:
        rows = rows[: args.limit]

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    config = language_config(model)
    completed = 0

    with torch.inference_mode():
        for position, row in enumerate(rows, 1):
            output_path = args.output_dir / f"{row['id']}.pt"
            if args.resume and output_path.exists():
                continue
            inputs = tokenize(processor, row, args.maximum_tokens)
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            outputs = model(
                **inputs,
                use_cache=False,
                output_hidden_states=True,
                output_router_logits=True,
                logits_to_keep=1,
                return_dict=True,
            )
            route_logits = torch.stack(
                [tensor.view(-1, config.num_experts) for tensor in outputs.router_logits]
            )
            route_scores, route_ids = torch.topk(
                torch.softmax(route_logits.float(), dim=-1),
                k=config.num_experts_per_tok,
                dim=-1,
            )
            route_ids = route_ids.to("cpu", torch.uint8)
            route_scores = route_scores.to("cpu", torch.float16)
            indices, features, targets = sample_features(
                outputs.hidden_states,
                route_ids,
                args.pairs_per_prompt,
                rng,
            )
            artifact = {
                "format_version": 1,
                "captured_at_utc": utc_now(),
                "id": row["id"],
                "source": row["source"],
                "split": row["split"],
                "input_ids": inputs["input_ids"][0].to("cpu"),
                "route_ids": route_ids,
                "route_scores": route_scores,
                "sample_indices": indices,
                "sample_features": features,
                "sample_targets": targets,
            }
            torch.save(artifact, output_path)
            completed += 1
            print(
                f"[{position}/{len(rows)}] {row['id']} {row['split']} "
                f"tokens={route_ids.shape[1]} pairs={features.shape[0]}",
                flush=True,
            )
            del outputs, route_logits, route_scores, route_ids, features, targets

    atomic_json(
        args.output_dir / "manifest.json",
        {
            "captured_at_utc": utc_now(),
            "model": str(args.model),
            "model_dtype": "bfloat16",
            "seed": args.seed,
            "maximum_tokens": args.maximum_tokens,
            "pairs_per_prompt": args.pairs_per_prompt,
            "newly_completed": completed,
            "requested_records": len(rows),
        },
    )


if __name__ == "__main__":
    main()

