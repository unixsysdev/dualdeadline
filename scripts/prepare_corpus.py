#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

from specstream.io import atomic_json, utc_now


def stable_id(source: str, key: str) -> str:
    return hashlib.sha256(f"{source}\0{key}".encode()).hexdigest()[:20]


def collect(seed: int, gsm8k_count: int, humaneval_count: int) -> list[dict]:
    rng = random.Random(seed)
    gsm8k = list(load_dataset("openai/gsm8k", "main", split="train"))
    humaneval = list(load_dataset("openai/openai_humaneval", split="test"))
    rng.shuffle(gsm8k)
    rng.shuffle(humaneval)

    records: list[dict] = []
    for row in gsm8k[:gsm8k_count]:
        records.append(
            {
                "id": stable_id("gsm8k", row["question"]),
                "source": "gsm8k",
                "user": row["question"],
                "assistant": row["answer"],
            }
        )
    for row in humaneval[:humaneval_count]:
        records.append(
            {
                "id": stable_id("humaneval", row["task_id"]),
                "source": "humaneval",
                "user": (
                    "Complete the following Python function. Return only the completed code.\n\n"
                    + row["prompt"]
                ),
                "assistant": row["prompt"] + row["canonical_solution"],
            }
        )

    rng.shuffle(records)
    train_end = round(0.60 * len(records))
    validation_end = train_end + round(0.20 * len(records))
    for index, row in enumerate(records):
        row["split"] = (
            "train"
            if index < train_end
            else "validation"
            if index < validation_end
            else "test"
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=260724787)
    parser.add_argument("--gsm8k-count", type=int, default=80)
    parser.add_argument("--humaneval-count", type=int, default=40)
    args = parser.parse_args()

    records = collect(args.seed, args.gsm8k_count, args.humaneval_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    atomic_json(
        args.output.with_suffix(".manifest.json"),
        {
            "captured_at_utc": utc_now(),
            "seed": args.seed,
            "counts": {
                split: sum(row["split"] == split for row in records)
                for split in ("train", "validation", "test")
            },
            "sources": {
                source: sum(row["source"] == source for row in records)
                for source in ("gsm8k", "humaneval")
            },
            "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        },
    )
    print(f"Wrote {len(records)} prompt-grouped examples to {args.output}")


if __name__ == "__main__":
    main()

