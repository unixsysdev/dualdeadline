#!/usr/bin/env python3
"""Build the preregistered layer-wise ready-count and loss-weight profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoConfig

from specstream.io import atomic_json, utc_now


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--transfer-timing", type=Path, required=True)
    parser.add_argument("--model-timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-key",
        choices=["features", "router_features"],
        default="router_features",
    )
    parser.add_argument("--target-horizon", type=int, default=1)
    parser.add_argument("--byte-budget-expert-equivalents", type=int, default=8)
    parser.add_argument("--predictor-latency-ms", type=float, default=0.0)
    args = parser.parse_args()

    transfer = json.loads(args.transfer_timing.read_text())
    sizes = np.asarray(
        [row["size_bytes"] for row in transfer["results"]], dtype=float
    )
    times = np.asarray(
        [row["mean_ms"] for row in transfer["results"]], dtype=float
    )
    ms_per_byte, intercept_ms = np.polyfit(sizes, times, 1)
    ms_per_byte = max(float(ms_per_byte), 0.0)
    intercept_ms = max(float(intercept_ms), 0.0)

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    text = config.text_config if hasattr(config, "text_config") else config
    intermediate = getattr(text, "moe_intermediate_size", None) or getattr(
        text, "intermediate_size"
    )
    gate_up_bytes = 2 * text.hidden_size * intermediate * 2
    down_bytes = text.hidden_size * intermediate * 2
    full_expert_bytes = gate_up_bytes + down_bytes
    gate_up_transfer_ms = intercept_ms + ms_per_byte * gate_up_bytes
    candidate_count = min(
        args.byte_budget_expert_equivalents
        * full_expert_bytes
        // gate_up_bytes,
        text.num_experts,
    )

    timing = json.loads(args.model_timing.read_text())
    layers = timing["layers"]
    pre_moe = np.asarray([row["pre_moe"]["p50_ms"] for row in layers])
    moe_and_residual = np.asarray(
        [row["moe_and_residual"]["p50_ms"] for row in layers]
    )
    total = np.asarray([row["total"]["p50_ms"] for row in layers])
    num_layers = len(layers)
    windows = np.zeros(num_layers)
    ready_counts = np.zeros(num_layers, dtype=int)
    for target_layer in range(args.target_horizon, num_layers):
        anchor_layer = target_layer - args.target_horizon
        if args.feature_key == "router_features":
            window = moe_and_residual[anchor_layer]
            first_complete_layer = anchor_layer + 1
        else:
            window = 0.0
            first_complete_layer = anchor_layer
        window += total[first_complete_layer:target_layer].sum()
        window += pre_moe[target_layer]
        window = max(window - args.predictor_latency_ms, 0.0)
        windows[target_layer] = window
        ready_counts[target_layer] = min(
            candidate_count, int(np.floor(window / gate_up_transfer_ms))
        )

    target_slice = ready_counts[args.target_horizon :]
    mean_ready = float(target_slice.mean())
    if mean_ready == 0:
        weights = np.ones(num_layers)
    else:
        weights = ready_counts / mean_ready

    atomic_json(
        args.output,
        {
            "captured_at_utc": utc_now(),
            "feature_key": args.feature_key,
            "target_horizon": args.target_horizon,
            "byte_budget_expert_equivalents": (
                args.byte_budget_expert_equivalents
            ),
            "predictor_latency_ms": args.predictor_latency_ms,
            "gate_up_bytes_per_expert": gate_up_bytes,
            "full_bytes_per_expert": full_expert_bytes,
            "gate_up_transfer_ms_per_expert": gate_up_transfer_ms,
            "candidate_count_from_byte_budget": int(candidate_count),
            "prefetch_window_ms": windows.tolist(),
            "ready_counts": ready_counts.tolist(),
            "weights": weights.tolist(),
        },
    )
    print(
        f"Wrote {args.output}; ready-count range "
        f"{target_slice.min()}..{target_slice.max()}"
    )


if __name__ == "__main__":
    main()
