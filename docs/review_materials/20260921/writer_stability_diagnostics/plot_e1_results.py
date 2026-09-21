#!/usr/bin/env python3
"""Regenerate E1 overview and per-task closed-loop figures."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ASSETS = ("S1000", "M300", "O1200", "O1500", "N1000", "N1800")
LABELS = ("Source", "MT-BC300", "Old1200", "Old1500", "New1000", "New1800")
COLORS = ("#777777", "#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.input.resolve()
    probe = json.loads((root / "e1_probe_summary.json").read_text())["by_asset"]
    with (root / "e1_rollout_summary.csv").open(newline="") as handle:
        summary = list(csv.DictReader(handle))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    x = np.arange(len(ASSETS))
    axes[0].bar(x, [probe[a]["inference_first5_mse"] for a in ASSETS], color=COLORS)
    axes[0].set_ylabel("10-step inference first-5 MSE")
    axes[0].set_title("Fixed action probes (288/model)")
    axes[1].bar(x, [probe[a]["action_delta_mse_vs_source"] for a in ASSETS], color=COLORS)
    axes[1].set_ylabel("Action MSE vs Source")
    axes[1].set_title("Functional action displacement")
    aggregate = {(r["asset"], r["panel"]): float(r["success_rate"]) for r in summary if r["global_task_id"] == "all"}
    width = 0.36
    axes[2].bar(x - width / 2, [aggregate.get((a, "common_held"), np.nan) for a in ASSETS], width,
                label="common held", color="#7A5195")
    axes[2].bar(x + width / 2, [aggregate.get((a, "train"), np.nan) for a in ASSETS], width,
                label="train panel", color="#EF5675")
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Success rate")
    axes[2].set_title("Fixed closed loop (4 states/task)")
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xticks(x, LABELS, rotation=35, ha="right")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("E1 frozen-function and small closed-loop diagnostics", fontsize=13)
    fig.tight_layout()
    fig.savefig(root / "e1_overview.png", dpi=220, bbox_inches="tight")
    fig.savefig(root / "e1_overview.svg", bbox_inches="tight")
    plt.close(fig)

    with (root / "frozen_rollout_rows.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    tasks = {}
    counts = defaultdict(int)
    for row in rows:
        key = (row["panel"], int(row["global_task_id"]))
        tasks[key] = (row["suite"], int(row["task_id"]), row["language"])
        counts[(row["asset"], key)] += row["success"] == "True"
    task_keys = sorted(tasks, key=lambda key: (0 if key[0] == "common_held" else 1, key[1]))
    matrix = np.full((len(ASSETS), len(task_keys)), np.nan)
    for i, asset in enumerate(ASSETS):
        for j, key in enumerate(task_keys):
            if any(r["asset"] == asset and r["panel"] == key[0] and int(r["global_task_id"]) == key[1] for r in rows):
                matrix[i, j] = counts[(asset, key)]
    names = [f"{key[0][0].upper()}:{suite.replace('libero_', '')}/{task_id}" for key, (suite, task_id, _language) in ((key, tasks[key]) for key in task_keys)]
    fig, axis = plt.subplots(figsize=(12.8, 4.5))
    image = axis.imshow(matrix, vmin=0, vmax=4, cmap="viridis", aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                axis.text(j, i, str(int(matrix[i, j])), ha="center", va="center",
                          color="white" if matrix[i, j] < 2.5 else "black", fontsize=8)
    axis.set_xticks(range(len(task_keys)), names, rotation=45, ha="right")
    axis.set_yticks(range(len(ASSETS)), LABELS)
    axis.set_title("E1-B successes per task (out of 4 fixed states)")
    fig.colorbar(image, ax=axis, label="successes / 4", ticks=range(5))
    fig.tight_layout()
    fig.savefig(root / "e1_closed_loop_by_task.png", dpi=220, bbox_inches="tight")
    fig.savefig(root / "e1_closed_loop_by_task.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
