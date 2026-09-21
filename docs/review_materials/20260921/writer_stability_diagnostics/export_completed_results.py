#!/usr/bin/env python3
"""Export only completed E0/E1 Writer stability evidence into the repository."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

ASSETS = ("O1200", "O1500", "N1000", "N1800", "M300", "S1000")
EXPECTED_ROLLOUTS = {"O1200": 20, "O1500": 52, "N1000": 52, "N1800": 52, "M300": 52, "S1000": 52}


def scrub(value: Any, replacements: tuple[tuple[str, str], ...]) -> Any:
    if isinstance(value, dict):
        return {key: scrub(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item, replacements) for item in value]
    if isinstance(value, str):
        result = value
        for prefix, replacement in replacements:
            result = result.replace(prefix, replacement)
        for pattern, replacement in (
            (r'/[^"\n]*?/ember_runs/coverage_retraining_20260920/', "<coverage-study>/"),
            (r'/[^"\n]*?/ember_runs/video_teaching_20260919/', "<old-video-study>/"),
            (r'/[^"\n]*?/projects/EMBER/', "<repo>/"),
        ):
            result = re.sub(pattern, replacement, result)
        return result
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty export table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study = args.study.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    replacements = (
        (str(study) + "/", "<writer-stability-study>/"),
    )

    e0 = json.loads((study / "e0_integrity.json").read_text())
    if e0.get("status") != "complete" or len(e0.get("rows", ())) != 4:
        raise ValueError("E0 is not complete")
    if not all(row.get("all_parameter_gradients_finite") for row in e0["rows"]):
        raise ValueError("E0 contains nonfinite gradients")
    write_json(output / "e0_integrity.json", e0)

    registration = json.loads((study / "registration.json").read_text())
    registration["status"] = "paused_after_e1"
    registration["output_root"] = "<writer-stability-study>"
    write_json(output / "registration.json", registration)
    write_json(output / "assets.json", scrub(json.loads((study / "assets.json").read_text()), replacements))

    probes = read_csv(study / "frozen_probe_rows.csv")
    probe_key = ("asset", "global_task_id", "teacher_demo", "query_demo", "query_fraction", "query_position", "noise_seed")
    if len(probes) != 1728 or len({tuple(row[key] for key in probe_key) for row in probes}) != 1728:
        raise ValueError("E1-A probe panel is incomplete or duplicated")
    numeric = ("full_mse", "first5_mse", "tau1_first5_mse", "inference_first5_mse", "action_delta_mse_vs_source")
    if any(not math.isfinite(float(row[key])) for row in probes for key in numeric):
        raise ValueError("E1-A contains nonfinite metrics")
    write_csv(output / "frozen_probe_rows.csv", probes)
    shutil.copyfile(study / "probe_manifest.jsonl", output / "probe_manifest.jsonl")

    rollouts = read_csv(study / "frozen_rollout_rows.csv")
    if len(rollouts) != 280:
        raise ValueError("E1-B rollout panel is incomplete")
    for asset in ASSETS:
        subset = [row for row in rollouts if row["asset"] == asset]
        if len(subset) != EXPECTED_ROLLOUTS[asset]:
            raise ValueError(f"E1-B row count differs for {asset}")
    scrubbed_rollouts = [scrub(row, replacements) for row in rollouts]
    write_csv(output / "frozen_rollout_rows.csv", scrubbed_rollouts)
    trajectories = [scrub(row, replacements) for row in read_csv(study / "trajectory_manifest.csv")]
    if len(trajectories) != 280:
        raise ValueError("trajectory index is incomplete")
    write_csv(output / "trajectory_manifest.csv", trajectories)

    shutil.copyfile(study / "analysis/e1_probe_summary.json", output / "e1_probe_summary.json")
    shutil.copyfile(study / "analysis/e1_rollout_summary.csv", output / "e1_rollout_summary.csv")

    conditions: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rollouts:
        key = (row["panel"], row["global_task_id"], row["init_state_id"])
        target = conditions.setdefault(key, {
            "panel": row["panel"], "suite": row["suite"], "task_id": row["task_id"],
            "global_task_id": row["global_task_id"], "init_state_id": row["init_state_id"],
            "paired_teacher_demo": row["paired_teacher_demo"], "language": row["language"],
        })
        target[f"{row['asset']}_success"] = row["success"]
        target[f"{row['asset']}_steps"] = row["steps"]
    wide = [conditions[key] for key in sorted(conditions, key=lambda item: (item[0], int(item[1]), int(item[2])))]
    if len(wide) != 52:
        raise ValueError("paired E1-B condition table differs")
    write_csv(output / "e1_closed_loop_paired.csv", wide)

    status = {
        "schema_version": "ember_writer_stability_remote_bundle_v1",
        "complete_stages": ["E0", "E1-A", "E1-B"],
        "incomplete_stages": ["E2", "E3"],
        "e0_rows": 4,
        "e1_probe_rows": 1728,
        "e1_rollout_rows": 280,
        "trajectory_payloads_exported": False,
        "trajectory_index_rows": 280,
        "fresh_training": False,
        "test_use": False,
        "checkpoint_selection_use": False,
        "paused_by_owner": True,
        "partial_e2_exported": False,
    }
    write_json(output / "completion_status.json", status)

    for path in output.iterdir():
        if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".md", ".py", ".svg"}:
            text = path.read_text(encoding="utf-8")
            private_roots = ("/" + "data0/user/", "/" + "data1/user/")
            if any(root in text for root in private_roots):
                raise ValueError(f"private absolute path remained in {path}")


if __name__ == "__main__":
    main()
