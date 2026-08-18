#!/usr/bin/env python3
"""Build the exact macro25/50 lost-gained-retained occupancy panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ember.pi05_eval.paired_metrics import episode_key, index_rows
from ember.pi05_source_checkpoint import write_json_atomic


def _load(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("mode") != "formal"
        or result.get("role") != "validation"
        or result.get("adapter", {}).get("video_condition") != "correct"
        or len(result.get("rows", ())) != 400
    ):
        raise ValueError("occupancy source is not one correct-video paired400 panel")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro25", type=Path, required=True)
    parser.add_argument("--macro50", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left = _load(args.macro25.resolve())
    right = _load(args.macro50.resolve())
    left_rows = index_rows(left["rows"])
    right_rows = index_rows(right["rows"])
    if set(left_rows) != set(right_rows):
        raise ValueError("macro25/50 occupancy panels are not exactly paired")
    rows = []
    for key in sorted(left_rows):
        success25 = bool(left_rows[key]["success"])
        success50 = bool(right_rows[key]["success"])
        category = (
            "retained"
            if success25 and success50
            else "lost" if success25 else "gained" if success50 else None
        )
        if category is None:
            continue
        suite, task_id, init_state_id = episode_key(left_rows[key])
        rows.append(
            {
                "suite": suite,
                "task_id": task_id,
                "init_state_id": init_state_id,
                "category": category,
                "macro25_success": success25,
                "macro50_success": success50,
                "env_seed": int(left_rows[key]["env_seed"]),
                "policy_seed_root": int(left_rows[key]["policy_seed_root"]),
            }
        )
    counts = {
        category: sum(row["category"] == category for row in rows)
        for category in ("lost", "gained", "retained")
    }
    if counts != {"lost": 52, "gained": 13, "retained": 71}:
        raise ValueError(f"macro25/50 occupancy category counts changed: {counts}")
    output = {
        "schema_version": "ember_writer_occupancy_selection_v1",
        "source_panels": {
            "macro25": str(args.macro25.resolve()),
            "macro50": str(args.macro50.resolve()),
        },
        "category_counts": counts,
        "rows": rows,
        "information_wall": {
            "selection_uses_validation_success_for_diagnosis_only": True,
            "training_gradient_use": False,
            "checkpoint_selection_use": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output.resolve(), output)
    print(json.dumps({"event": "complete", "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
