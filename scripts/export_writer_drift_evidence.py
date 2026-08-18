#!/usr/bin/env python3
"""Export remote-safe Writer drift and Program/FactorHead evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-decode", type=Path, required=True)
    parser.add_argument("--drift-diagnosis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cross = _read(args.cross_decode)
    diagnosis = _read(args.drift_diagnosis)
    if (
        cross.get("schema_version") != "ember_lmmpc_checkpoint_cross_decode_v1"
        or diagnosis.get("schema_version")
        != "ember_lmmpc_drift_diagnosis_v1"
    ):
        raise ValueError("unexpected Writer drift evidence schema")

    evidence = {
        "schema_version": "ember_external_review_writer_drift_evidence_v1",
        "source_artifact_schemas": {
            "cross_decode": cross["schema_version"],
            "drift_diagnosis": diagnosis["schema_version"],
        },
        "conditions_per_task": cross["conditions_per_task"],
        "task_keys": cross["task_keys"],
        "checkpoint_scores": diagnosis["checkpoint_scores"],
        "checkpoint_diagnostics": diagnosis["checkpoints"],
        "strict_stability": diagnosis["strict_stability"],
        "strict_task_trajectory": diagnosis["strict_task_trajectory"],
        "intervals": diagnosis["intervals"],
        "native_pairwise": cross["native_pairwise"],
        "factor_decomposition": cross["factor_decomposition"],
        "stage_trajectory": cross["stage_trajectory"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
