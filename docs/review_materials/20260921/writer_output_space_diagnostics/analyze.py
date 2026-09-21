#!/usr/bin/env python3
"""Export the frozen Writer output-space diagnostic into a portable report bundle."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt
import numpy as np


ARMS = ("O1200", "SELF", "NEWSPACE", "SHRINK")
TASKS = (3, 11, 26, 31)
FAMILIES = ("q_b", "v_b")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def truth(value: str) -> bool:
    return value.lower() == "true"


def load_rollouts(root: Path) -> dict[str, dict[tuple[int, int], dict[str, str]]]:
    tables: dict[str, dict[tuple[int, int], dict[str, str]]] = {}
    baseline = [
        row
        for row in read_csv(root / "baseline_o1200_rows.csv")
        if int(row["global_task_id"]) in TASKS and int(row["init_state_id"]) < 4
    ]
    tables["O1200"] = {
        (int(row["global_task_id"]), int(row["init_state_id"])): row
        for row in baseline
    }
    for arm in ARMS[1:]:
        rows = read_csv(root / f"a2_rollout_rows_{arm}.csv")
        tables[arm] = {
            (int(row["global_task_id"]), int(row["init_state_id"])): row
            for row in rows
        }
    expected = {(task, state) for task in TASKS for state in range(4)}
    for arm, table in tables.items():
        if set(table) != expected:
            raise ValueError(f"{arm} panel keys differ from the fixed 16-row panel")
    return tables


def portable_rollout_rows(
    tables: dict[str, dict[tuple[int, int], dict[str, str]]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        for key in sorted(tables[arm]):
            raw = tables[arm][key]
            stage = json.loads(raw["stage_predicates"])
            rows.append(
                {
                    "arm": arm,
                    "suite": raw["suite"],
                    "task_id": int(raw["task_id"]),
                    "global_task_id": key[0],
                    "language": raw["language"],
                    "init_state_id": key[1],
                    "env_seed": int(raw["env_seed"]),
                    "policy_seed_root": int(raw["policy_seed_root"]),
                    "success": truth(raw["success"]),
                    "steps": int(raw["steps"]),
                    "wall_seconds": float(raw["wall_seconds"]),
                    "final_satisfied_count": sum(stage["final_satisfied"]),
                    "ever_satisfied_count": sum(stage["ever_satisfied"]),
                    "peak_satisfied_count": int(stage["peak_satisfied_count"]),
                    "predicate_count": len(stage["predicates"]),
                    "stage_predicates": json.dumps(stage, separators=(",", ":")),
                }
            )
    return rows


def paired_summaries(
    tables: dict[str, dict[tuple[int, int], dict[str, str]]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    baseline = tables["O1200"]
    summary: list[dict[str, object]] = []
    per_task: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    for arm in ARMS:
        panel = tables[arm]
        success = sum(truth(row["success"]) for row in panel.values())
        retained = gained = lost = stable_failure = 0
        for key in sorted(baseline):
            old = truth(baseline[key]["success"])
            new = truth(panel[key]["success"])
            retained += old and new
            gained += (not old) and new
            lost += old and (not new)
            stable_failure += (not old) and (not new)
            pairs.append(
                {
                    "arm": arm,
                    "global_task_id": key[0],
                    "init_state_id": key[1],
                    "baseline_success": old,
                    "arm_success": new,
                    "transition": (
                        "retained" if old and new else
                        "gained" if (not old) and new else
                        "lost" if old and (not new) else "stable_failure"
                    ),
                    "baseline_steps": int(baseline[key]["steps"]),
                    "arm_steps": int(panel[key]["steps"]),
                }
            )
        summary.append(
            {
                "arm": arm,
                "successes": success,
                "trials": 16,
                "success_rate": success / 16,
                "retained": retained,
                "gained": gained,
                "lost": lost,
                "stable_failure": stable_failure,
                "churn": gained + lost,
                "successful_steps_mean": (
                    fmean(int(row["steps"]) for row in panel.values() if truth(row["success"]))
                    if success else ""
                ),
                "total_wall_seconds": sum(float(row["wall_seconds"]) for row in panel.values()),
            }
        )
        for task in TASKS:
            task_rows = [row for key, row in panel.items() if key[0] == task]
            per_task.append(
                {
                    "arm": arm,
                    "global_task_id": task,
                    "language": task_rows[0]["language"],
                    "successes": sum(truth(row["success"]) for row in task_rows),
                    "trials": 4,
                    "mean_steps": fmean(int(row["steps"]) for row in task_rows),
                }
            )
    return summary, per_task, pairs


def a1_summary(root: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for asset in ("O1200", "N1000"):
        codes = read_csv(root / f"a1_code_geometry_{asset}.csv")
        cross = read_csv(root / f"a1_cross_video_geometry_{asset}.csv")
        factors = read_csv(root / f"a1_factor_geometry_{asset}.csv")
        for family in FAMILIES:
            c = [row for row in codes if row["family"] == family]
            x = [row for row in cross if row["family"] == family]
            f = [row for row in factors if row["family"] == family]
            rank_gram = [json.loads(row["rank_gram"])["off_diagonal_mean"] for row in c]
            layer_gram = [json.loads(row["layer_gram"])["off_diagonal_mean"] for row in c]
            b_top1, delta_top1 = [], []
            for row in f:
                bsv = np.asarray(json.loads(row["b_singular_values"]), dtype=np.float64)
                dsv = np.asarray(json.loads(row["delta_singular_values"]), dtype=np.float64)
                b_top1.append(float(bsv[0] ** 2 / np.square(bsv).sum()))
                delta_top1.append(float(dsv[0] ** 2 / np.square(dsv).sum()))
            output.append(
                {
                    "asset": asset,
                    "family": family,
                    "compilations": len(c),
                    "rank_centered_energy_ratio_mean": fmean(float(row["rank_centered_energy_ratio"]) for row in c),
                    "layer_centered_energy_ratio_mean": fmean(float(row["layer_centered_energy_ratio"]) for row in c),
                    "rank_gram_offdiag_mean": fmean(rank_gram),
                    "layer_gram_offdiag_mean": fmean(layer_gram),
                    "cross_video_cosine_mean": fmean(float(row["flattened_cosine"]) for row in x),
                    "cross_video_relative_l2_mean": fmean(float(row["relative_l2"]) for row in x),
                    "cross_video_centered_energy_ratio_mean": fmean(float(row["centered_energy_ratio"]) for row in x),
                    "b_top1_energy_ratio_mean": fmean(b_top1),
                    "delta_top1_energy_ratio_mean": fmean(delta_top1),
                    "reconstruction_relative_l2_mean": fmean(float(row["reconstruction_relative_l2"]) for row in f),
                    "reconstruction_relative_l2_max": max(float(row["reconstruction_relative_l2"]) for row in f),
                }
            )
    return output


def a2_geometry(root: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for arm in ARMS[1:]:
        rows = read_csv(root / f"a2_projection_rows_{arm}.csv")
        for family in FAMILIES:
            selected = [row for row in rows if row["family"] == family]
            output.append(
                {
                    "arm": arm,
                    "family": family,
                    "rows": len(selected),
                    "candidate_to_original_norm_ratio_mean": fmean(float(row["candidate_to_original_norm_ratio"]) for row in selected),
                    "candidate_to_original_norm_ratio_min": min(float(row["candidate_to_original_norm_ratio"]) for row in selected),
                    "rho_mean": fmean(float(row["rho"]) for row in selected),
                    "rho_max": max(float(row["rho"]) for row in selected),
                    "b_projection_relative_l2_mean": fmean(float(row["b_projection_relative_l2"]) for row in selected),
                    "b_projection_relative_l2_max": max(float(row["b_projection_relative_l2"]) for row in selected),
                    "finite": all(truth(row["finite"]) for row in selected),
                }
            )
    return output


def plots(
    out: Path,
    summary: list[dict[str, object]],
    per_task: list[dict[str, object]],
    code: list[dict[str, object]],
    geometry: list[dict[str, object]],
) -> None:
    colors = {"O1200": "#4c78a8", "SELF": "#72b7b2", "NEWSPACE": "#f58518", "SHRINK": "#e45756"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    x = np.arange(len(TASKS))
    width = 0.19
    for index, arm in enumerate(ARMS):
        vals = [next(row["successes"] for row in per_task if row["arm"] == arm and row["global_task_id"] == task) for task in TASKS]
        axes[0].bar(x + (index - 1.5) * width, vals, width, label=arm, color=colors[arm])
    axes[0].set_xticks(x, [f"global {task}" for task in TASKS])
    axes[0].set_ylim(0, 4.3)
    axes[0].set_ylabel("successes / 4")
    axes[0].set_title("Fixed held panel by task")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar([row["arm"] for row in summary], [row["successes"] for row in summary], color=[colors[row["arm"]] for row in summary])
    axes[1].set_ylim(0, 16)
    axes[1].set_ylabel("successes / 16")
    axes[1].set_title("Panel total")
    for ext in ("png", "svg"):
        fig.savefig(out / f"paired_success.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for index, asset in enumerate(("O1200", "N1000")):
        vals = [next(row["rank_centered_energy_ratio_mean"] for row in code if row["asset"] == asset and row["family"] == family) for family in FAMILIES]
        axes[0].bar(np.arange(2) + (index - .5) * .35, vals, .35, label=asset)
        vals = [next(row["cross_video_cosine_mean"] for row in code if row["asset"] == asset and row["family"] == family) for family in FAMILIES]
        axes[1].bar(np.arange(2) + (index - .5) * .35, vals, .35, label=asset)
    for ax, title, ylabel in zip(axes, ("Rank-centered code energy", "Demo46 vs demo48 code cosine"), ("energy ratio", "cosine")):
        ax.set_xticks(np.arange(2), FAMILIES)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
    axes[1].set_ylim(.98, 1.0)
    for ext in ("png", "svg"):
        fig.savefig(out / f"code_geometry.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    arms = ARMS[1:]
    for index, family in enumerate(FAMILIES):
        axes[0].bar(np.arange(3) + (index - .5) * .35, [next(row["rho_mean"] for row in geometry if row["arm"] == arm and row["family"] == family) for arm in arms], .35, label=family)
        axes[1].bar(np.arange(3) + (index - .5) * .35, [next(row["candidate_to_original_norm_ratio_mean"] for row in geometry if row["arm"] == arm and row["family"] == family) for arm in arms], .35, label=family)
    axes[0].set_ylabel("mean relative update change (rho)")
    axes[1].set_ylabel("mean update norm ratio")
    for ax, title in zip(axes, ("Direction intervention", "Magnitude retained")):
        ax.set_xticks(np.arange(3), arms)
        ax.set_title(title)
        ax.legend(frameon=False)
    for ext in ("png", "svg"):
        fig.savefig(out / f"projection_geometry.{ext}", dpi=180)
    plt.close(fig)
    for path in out.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def copy_raw(root: Path, out: Path) -> None:
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    names = [
        "a0_completion.json", "a0_space_summary.csv",
        "a0_space_overlap.csv", "a0_singular_and_principal_values.csv",
        "a1_code_geometry_O1200.csv", "a1_code_geometry_N1000.csv",
        "a1_cross_video_geometry_O1200.csv", "a1_cross_video_geometry_N1000.csv",
        "a1_factor_geometry_O1200.csv", "a1_factor_geometry_N1000.csv",
        "a1_completion_O1200.json", "a1_completion_N1000.json",
        "a2_self_first_replan.csv",
    ]
    for arm in ARMS[1:]:
        names.extend((f"a2_projection_rows_{arm}.csv", f"a2_completion_{arm}.json"))
    for name in names:
        shutil.copy2(root / name, raw / name)
    registration = json.loads((root / "registration.json").read_text())
    for asset, record in registration["assets"].items():
        record["checkpoint"] = asset
        for kind in ("weights", "manifest"):
            record[kind]["path"] = Path(record[kind]["path"]).name
    (raw / "registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )
    shutil.copy2(root / "launch/a1_layer_order_incident.json", raw / "a1_layer_order_incident.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root, out = args.study_root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tables = load_rollouts(root)
    summary, per_task, pairs = paired_summaries(tables)
    code = a1_summary(root)
    geometry = a2_geometry(root)
    write_csv(out / "rollout_rows.csv", portable_rollout_rows(tables))
    write_csv(out / "paired_summary.csv", summary)
    write_csv(out / "per_task_success.csv", per_task)
    write_csv(out / "paired_rows.csv", pairs)
    write_csv(out / "a1_geometry_summary.csv", code)
    write_csv(out / "a2_geometry_summary.csv", geometry)
    copy_raw(root, out)
    plots(out, summary, per_task, code, geometry)
    completion = {
        "schema_version": "ember_writer_output_space_export_v1",
        "base_runtime_commit": "2848684f681d7c1ccedee57ba1a58583b73cb5da",
        "a1_repair_runtime_commit": "36e7f791dc8993569ca68dbe1873842a61fedde1",
        "a0_assets": 3,
        "a1_compilations": 16,
        "a2_rollouts": 48,
        "baseline_rows": 16,
        "all_formal_exits_zero": True,
        "backward": False,
        "optimizer_updates": 0,
        "test_use": False,
    }
    (out / "completion.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
