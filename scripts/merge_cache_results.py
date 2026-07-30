#!/usr/bin/env python3
"""Merge independently computed cache-capacity reports."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from specstream.io import atomic_json, utc_now


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()

    documents = [json.loads(path.read_text()) for path in args.inputs]
    if not documents:
        raise ValueError("At least one input report is required")

    invariant_keys = (
        "semantics",
        "assumptions",
        "test_prompts",
        "test_pairs",
        "feature_key",
        "target_horizon",
        "byte_budget_expert_equivalents",
        "prefetch_window_ms_by_target_layer",
        "transfer_ms",
    )
    reference = documents[0]
    for path, document in zip(args.inputs[1:], documents[1:]):
        for key in invariant_keys:
            if document[key] != reference[key]:
                raise ValueError(f"{path}: incompatible {key}")

    merged = deepcopy(reference)
    merged["captured_at_utc"] = utc_now()
    merged["cache_experts_per_layer"] = []
    merged["results"] = {}
    merged["merged_from"] = [str(path) for path in args.inputs]
    for path, document in zip(args.inputs, documents):
        for capacity in document["cache_experts_per_layer"]:
            key = str(capacity)
            if key in merged["results"]:
                raise ValueError(f"{path}: duplicate cache capacity {capacity}")
            merged["cache_experts_per_layer"].append(capacity)
            merged["results"][key] = document["results"][key]

    merged["cache_experts_per_layer"].sort()
    atomic_json(args.output, merged)
    print(
        f"Wrote {args.output} with capacities "
        f"{merged['cache_experts_per_layer']}"
    )


if __name__ == "__main__":
    main()
