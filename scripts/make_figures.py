#!/usr/bin/env python3
"""Generate publication figures from committed JSON result artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "learned": "#0072B2",
    "training_popularity": "#E69F00",
    "previous_layer_routes": "#009E73",
    "random": "#777777",
    "oracle": "#CC79A7",
    "monolithic": "#D55E00",
    "staged": "#0072B2",
}


def save(figure: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(
            output_dir / f"{name}.{suffix}",
            dpi=240,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.22,
        }
    )

    predictor = json.loads(
        (
            args.artifacts
            / "results/olmoe_next_preregistered_e32_predictor.json"
        ).read_text()
    )
    budgets = predictor["budgets"]
    figure, axis = plt.subplots(figsize=(3.35, 2.35))
    labels = {
        "learned": "Layer-aware learned",
        "training_popularity": "Training popularity",
        "previous_layer_routes": "Previous-layer route",
        "random": "Random",
        "oracle": "Oracle",
    }
    for policy in (
        "learned",
        "training_popularity",
        "previous_layer_routes",
        "random",
        "oracle",
    ):
        means = [
            100 * predictor["policies"][policy][str(budget)]["mean"]
            for budget in budgets
        ]
        axis.plot(
            budgets,
            means,
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=COLORS[policy],
            label=labels[policy],
        )
    axis.set(
        xlabel="Prefetch candidates",
        ylabel="Held-out expert recall (%)",
        xticks=budgets,
        ylim=(0, 103),
    )
    axis.grid(axis="y")
    axis.legend(frameon=False, ncol=1, loc="upper left")
    save(figure, args.output_dir, "predictor_recall")

    stall = json.loads(
        (
            args.artifacts
            / "results/olmoe_next_preregistered_e32_stall.json"
        ).read_text()
    )
    stall_budgets = sorted(
        int(value) for value in stall["results"]["learned"]
    )
    figure, axis = plt.subplots(figsize=(3.35, 2.35))
    for mode in ("monolithic", "staged"):
        means = [
            stall["results"]["learned"][str(budget)][mode]["mean_ms"]
            for budget in stall_budgets
        ]
        lows = [
            stall["results"]["learned"][str(budget)][mode]["ci_low_ms"]
            for budget in stall_budgets
        ]
        highs = [
            stall["results"]["learned"][str(budget)][mode]["ci_high_ms"]
            for budget in stall_budgets
        ]
        axis.plot(
            stall_budgets,
            means,
            marker="o",
            markersize=3.5,
            linewidth=1.7,
            color=COLORS[mode],
            label=mode.capitalize(),
        )
        axis.fill_between(
            stall_budgets,
            lows,
            highs,
            color=COLORS[mode],
            alpha=0.16,
            linewidth=0,
        )
    axis.set(
        xlabel="Byte budget (full-expert equiv.)",
        ylabel="Exposed fetch stall (ms)",
        xticks=stall_budgets,
    )
    axis.grid(axis="y")
    axis.legend(frameon=False)
    save(figure, args.output_dir, "cacheless_stall")

    cache = json.loads(
        (
            args.artifacts
            / "results/olmoe_next_preregistered_e32_cache.json"
        ).read_text()
    )
    capacities = [int(value) for value in cache["cache_experts_per_layer"]]
    figure, axis = plt.subplots(figsize=(3.35, 2.35))
    for mode in ("monolithic", "staged"):
        entries = [
            cache["results"][str(capacity)]["learned"]["4"][mode]["stall_ms"]
            for capacity in capacities
        ]
        means = np.asarray([entry["mean"] for entry in entries])
        lower = means - np.asarray([entry["ci_low"] for entry in entries])
        upper = np.asarray([entry["ci_high"] for entry in entries]) - means
        axis.errorbar(
            capacities,
            means,
            yerr=np.vstack([lower, upper]),
            marker="o",
            markersize=4,
            capsize=2,
            linewidth=1.6,
            color=COLORS[mode],
            label=mode.capitalize(),
        )
    axis.set(
        xlabel="Persistent LRU capacity (experts/layer)",
        ylabel="Exposed fetch stall (ms)",
        xticks=capacities,
    )
    axis.grid(axis="y")
    axis.legend(frameon=False)
    save(figure, args.output_dir, "cache_sensitivity")

    overlap = json.loads(
        (
            args.artifacts
            / "timing/olmoe_staged_overlap_partial_h200.json"
        ).read_text()
    )
    timing = overlap["timings"]
    ready = [0, 2, 4, 6, 8]
    staged_names = [
        "staged_overlap",
        "partial_prefetched_gate_2",
        "partial_prefetched_gate_4",
        "partial_prefetched_gate_6",
        "prefetched_gate_overlap",
    ]
    staged_values = [timing[name]["p50_ms"] for name in staged_names]
    figure, axis = plt.subplots(figsize=(3.35, 2.35))
    axis.plot(
        ready,
        staged_values,
        marker="o",
        markersize=4,
        linewidth=1.7,
        color=COLORS["staged"],
        label="Component-staged",
    )
    axis.scatter(
        [0],
        [timing["monolithic_copy_then_compute"]["p50_ms"]],
        marker="s",
        s=28,
        color=COLORS["monolithic"],
        label="Monolithic on-demand",
        zorder=3,
    )
    axis.axhline(
        timing["resident_compute"]["p50_ms"],
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label="All weights resident",
    )
    axis.set(
        xlabel="Gate/up components ready at router",
        ylabel="Measured block latency (ms)",
        xticks=ready,
    )
    axis.grid(axis="y")
    axis.legend(frameon=False)
    save(figure, args.output_dir, "measured_overlap")


if __name__ == "__main__":
    main()
