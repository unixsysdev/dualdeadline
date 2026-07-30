#!/usr/bin/env python3
"""Prompt-cold LRU sensitivity analysis for staged expert prefetch."""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

from simulate_prefetch import (
    popular_scores,
    prior_layer_scores,
    transfer_fit,
)
from specstream.io import atomic_json, utc_now
from specstream.predictor import LayerwiseExpertPredictor
from specstream.traces import load_split


def interval(
    values: np.ndarray,
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, float | int]:
    selections = rng.integers(0, len(values), (resamples, len(values)))
    means = values[selections].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "prompt_count": int(len(values)),
    }


def prompt_means(
    values: np.ndarray,
    prompt_index: torch.Tensor,
) -> np.ndarray:
    prompt_numbers = prompt_index.numpy()
    return np.asarray(
        [values[prompt_numbers == prompt].mean() for prompt in np.unique(prompt_numbers)]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--transfer-timing", type=Path, required=True)
    parser.add_argument("--model-timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--byte-budget-expert-equivalents",
        type=int,
        nargs="+",
        default=[0, 2, 4, 8, 12, 16],
    )
    parser.add_argument(
        "--cache-experts-per-layer",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument(
        "--predictor-latency-ms",
        type=float,
        help="Override checkpoint latency for timing-sensitivity analysis.",
    )
    parser.add_argument("--seed", type=int, default=260724787)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint["model_metadata"]
    feature_key = metadata.get("trace_feature_key", "features")
    target_horizon = metadata.get("target_horizon", 0)
    predictor_latency_ms = checkpoint.get("deadline_profile", {}).get(
        "predictor_latency_ms", 0.0
    )
    if args.predictor_latency_ms is not None:
        predictor_latency_ms = args.predictor_latency_ms
    test = load_split(
        args.traces,
        "test",
        feature_key=feature_key,
        target_horizon=target_horizon,
    )
    predictor = LayerwiseExpertPredictor(
        metadata["hidden_size"],
        metadata["num_layers"],
        metadata["num_experts"],
        metadata["width"],
        architecture=metadata.get("architecture", "layer_aware"),
        layer_embedding_width=metadata.get(
            "layer_embedding_width", metadata["width"]
        ),
    )
    predictor.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor.to(device).eval()
    learned_parts = []
    with torch.inference_mode():
        for start in range(0, len(test), 2048):
            stop = min(start + 2048, len(test))
            learned_parts.append(
                predictor(
                    test.hidden[start:stop].to(device, torch.float32),
                    test.layer[start:stop].to(device),
                ).cpu()
            )
    learned = torch.cat(learned_parts)
    popular_by_layer = popular_scores(
        args.traces, metadata["num_layers"], metadata["num_experts"]
    )
    generator = torch.Generator().manual_seed(args.seed)
    policies = {
        "random": torch.rand(learned.shape, generator=generator),
        "training_popularity": popular_by_layer[test.layer],
        "previous_layer_routes": prior_layer_scores(
            args.traces, popular_by_layer, target_horizon
        ),
        "learned": learned,
        "oracle": torch.zeros_like(learned).scatter_(1, test.targets, 1.0),
    }

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    text = config.text_config if hasattr(config, "text_config") else config
    intermediate = getattr(text, "moe_intermediate_size", None) or getattr(
        text, "intermediate_size"
    )
    gate_up_bytes = 2 * text.hidden_size * intermediate * 2
    down_bytes = text.hidden_size * intermediate * 2
    full_bytes = gate_up_bytes + down_bytes
    intercept_ms, ms_per_byte = transfer_fit(args.transfer_timing)

    def copy_time(size: int) -> float:
        return intercept_ms + ms_per_byte * size

    gate_up_ms = copy_time(gate_up_bytes)
    down_ms = copy_time(down_bytes)
    full_ms = copy_time(full_bytes)
    timing = json.loads(args.model_timing.read_text())
    pre_moe = torch.tensor([row["pre_moe"]["p50_ms"] for row in timing["layers"]])
    moe_residual = torch.tensor(
        [row["moe_and_residual"]["p50_ms"] for row in timing["layers"]]
    )
    total = torch.tensor([row["total"]["p50_ms"] for row in timing["layers"]])
    gate_compute_ms = timing["expert_gate_up_compute"]["p50_ms"]
    windows = torch.zeros(metadata["num_layers"])
    for target_layer in range(target_horizon, metadata["num_layers"]):
        anchor = target_layer - target_horizon
        if feature_key == "router_features":
            window = moe_residual[anchor].clone()
            first_complete = anchor + 1
        else:
            window = torch.tensor(0.0)
            first_complete = anchor
        if first_complete < target_layer:
            window += total[first_complete:target_layer].sum()
        windows[target_layer] = torch.clamp(
            window + pre_moe[target_layer] - predictor_latency_ms,
            min=0.0,
        )

    prompt_index = test.prompt_index
    prompt_starts = [
        int((prompt_index == prompt).nonzero(as_tuple=True)[0][0].item())
        for prompt in range(len(test.prompt_ids))
    ]
    prompt_stops = prompt_starts[1:] + [len(test)]
    actual_all = test.targets.long()
    rng = np.random.default_rng(args.seed)
    results = {}

    for cache_capacity in args.cache_experts_per_layer:
        cache_results = {}
        for policy_name, scores in policies.items():
            rankings = scores.argsort(dim=-1, descending=True)
            policy_results = {}
            for budget in args.byte_budget_expert_equivalents:
                mode_results = {}
                for staged in (False, True):
                    stalls = np.zeros(len(test), dtype=np.float64)
                    cache_recall = np.zeros(len(test), dtype=np.float64)
                    prefetch_recall = np.zeros(len(test), dtype=np.float64)
                    precision_numerator = np.zeros(len(test), dtype=np.float64)
                    precision_denominator = np.zeros(len(test), dtype=np.float64)
                    transfer_size = gate_up_bytes if staged else full_bytes
                    transfer_ms = gate_up_ms if staged else full_ms
                    byte_budget = budget * full_bytes
                    budget_candidates = min(
                        byte_budget // transfer_size,
                        metadata["num_experts"],
                    )

                    for prompt_start, prompt_stop in zip(prompt_starts, prompt_stops):
                        caches = [
                            OrderedDict() for _ in range(metadata["num_layers"])
                        ]
                        for pair in range(prompt_start, prompt_stop):
                            layer = int(test.layer[pair])
                            actual = [int(value) for value in actual_all[pair]]
                            actual_set = set(actual)
                            cache = caches[layer]
                            resident = set(cache)
                            resident_actual = actual_set & resident
                            missing_actual = actual_set - resident
                            ready_limit = min(
                                int(windows[layer].item() // transfer_ms),
                                int(budget_candidates),
                            )
                            selected = []
                            if ready_limit:
                                for value in rankings[pair].tolist():
                                    if value in resident:
                                        continue
                                    selected.append(value)
                                    if len(selected) == ready_limit:
                                        break
                            selected_set = set(selected)
                            prefetched_actual = missing_actual & selected_set

                            if staged:
                                missing_gate = len(missing_actual - selected_set)
                                residual_down = max(
                                    0.0,
                                    len(missing_actual) * down_ms - gate_compute_ms,
                                )
                                stalls[pair] = (
                                    missing_gate * gate_up_ms + residual_down
                                )
                            else:
                                stalls[pair] = (
                                    len(missing_actual - selected_set) * full_ms
                                )

                            cache_recall[pair] = len(resident_actual) / len(actual_set)
                            prefetch_recall[pair] = (
                                len(prefetched_actual) / len(actual_set)
                            )
                            precision_numerator[pair] = len(prefetched_actual)
                            precision_denominator[pair] = len(selected_set)

                            for expert in actual:
                                cache.pop(expert, None)
                                cache[expert] = None
                            while len(cache) > cache_capacity:
                                cache.popitem(last=False)

                    prompt_precision = []
                    prompt_numbers = prompt_index.numpy()
                    for prompt in np.unique(prompt_numbers):
                        mask = prompt_numbers == prompt
                        denominator = precision_denominator[mask].sum()
                        prompt_precision.append(
                            precision_numerator[mask].sum() / denominator
                            if denominator
                            else 0.0
                        )
                    mode_results["staged" if staged else "monolithic"] = {
                        "stall_ms": interval(
                            prompt_means(stalls, prompt_index),
                            rng,
                            args.bootstrap_resamples,
                        ),
                        "cache_recall": interval(
                            prompt_means(cache_recall, prompt_index),
                            rng,
                            args.bootstrap_resamples,
                        ),
                        "prefetch_recall": interval(
                            prompt_means(prefetch_recall, prompt_index),
                            rng,
                            args.bootstrap_resamples,
                        ),
                        "speculative_precision": interval(
                            np.asarray(prompt_precision),
                            rng,
                            args.bootstrap_resamples,
                        ),
                    }
                policy_results[str(budget)] = mode_results
            cache_results[policy_name] = policy_results
        results[str(cache_capacity)] = cache_results
        print(f"completed cache capacity {cache_capacity}", flush=True)

    atomic_json(
        args.output,
        {
            "captured_at_utc": utc_now(),
            "semantics": "trace-driven prompt-cold LRU simulation",
            "assumptions": {
                "cache": "full experts only; independent LRU per layer; reset each prompt",
                "speculative_workspace": (
                    "separate from persistent cache; unused predictions discarded "
                    "after the target layer"
                ),
                "dma_channels": 1,
                "predictor_latency_ms": predictor_latency_ms,
            },
            "test_prompts": len(test.prompt_ids),
            "test_pairs": len(test),
            "feature_key": feature_key,
            "target_horizon": target_horizon,
            "cache_experts_per_layer": args.cache_experts_per_layer,
            "byte_budget_expert_equivalents": (
                args.byte_budget_expert_equivalents
            ),
            "prefetch_window_ms_by_target_layer": windows.tolist(),
            "transfer_ms": {
                "gate_up": gate_up_ms,
                "down": down_ms,
                "full": full_ms,
            },
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
