#!/usr/bin/env python3
"""Export remote-safe fixed-occupancy evidence for the external review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon


ANALYSIS_SCHEMA = "ember_writer_occupancy_counterfactual_analysis_v1"
EVIDENCE_SCHEMA = "ember_external_review_occupancy_evidence_v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _delta_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    values = np.asarray(
        [
            float(row["macro50_occupancy_action_rms_mean"])
            - float(row["macro25_occupancy_action_rms_mean"])
            for row in selected
        ],
        dtype=np.float64,
    )
    statistic, p_value = wilcoxon(values, alternative="two-sided")
    return {
        "rows": len(selected),
        "mean_macro50_minus_macro25": float(values.mean()),
        "median_macro50_minus_macro25": float(np.median(values)),
        "positive_rows": int((values > 0).sum()),
        "negative_rows": int((values < 0).sum()),
        "zero_rows": int((values == 0).sum()),
        "two_sided_wilcoxon_statistic": float(statistic),
        "two_sided_wilcoxon_p": float(p_value),
    }


def _initial_comparison(
    rows: list[Mapping[str, Any]], left: str, right: str
) -> dict[str, Any]:
    left_values = np.asarray(
        [
            float(row["initial_state_action_rms"])
            for row in rows
            if row["category"] == left
        ],
        dtype=np.float64,
    )
    right_values = np.asarray(
        [
            float(row["initial_state_action_rms"])
            for row in rows
            if row["category"] == right
        ],
        dtype=np.float64,
    )
    statistic, p_value = mannwhitneyu(
        left_values, right_values, alternative="two-sided"
    )
    return {
        "left": left,
        "right": right,
        "left_rows": int(left_values.size),
        "right_rows": int(right_values.size),
        "left_mean": float(left_values.mean()),
        "right_mean": float(right_values.mean()),
        "two_sided_mann_whitney_u": float(statistic),
        "two_sided_mann_whitney_p": float(p_value),
    }


def _contract_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    capture = contract.get("diagnostic_occupancy_capture", {})
    git = contract.get("git", {})
    if (
        contract.get("mode") != "formal"
        or capture.get("schema_version") != "ember_writer_occupancy_capture_v1"
        or git.get("dirty_paths") != []
    ):
        raise ValueError("occupancy capture is not a clean formal diagnostic")
    return {
        "contract_reference": contract.get("contract_reference"),
        "git_commit": git.get("commit"),
        "git_dirty_paths": git.get("dirty_paths"),
        "host": contract.get("host"),
        "mode": contract.get("mode"),
        "diagnostic_occupancy_capture": {
            "schema_version": capture.get("schema_version"),
            "selected_rows": capture.get("selected_rows"),
            "category_counts": capture.get("category_counts"),
            "training_gradient_use": capture.get("training_gradient_use"),
            "validation_action_or_reward_gradient_use": capture.get(
                "validation_action_or_reward_gradient_use"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--macro25-contract", type=Path, required=True)
    parser.add_argument("--macro50-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    analysis = _read(args.analysis)
    if analysis.get("schema_version") != ANALYSIS_SCHEMA:
        raise ValueError("unexpected occupancy analysis schema")
    rows = list(analysis["rows"])
    categories = ("lost", "gained", "retained")
    delta_by_category = {}
    replay_consistent_delta_by_category = {}
    for category in categories:
        selected = [row for row in rows if row["category"] == category]
        delta_by_category[category] = _delta_summary(selected)
        replay_consistent_delta_by_category[category] = _delta_summary(
            row
            for row in selected
            if row["canonical_macro25_success"]
            == row["captured_macro25_success"]
            and row["canonical_macro50_success"]
            == row["captured_macro50_success"]
        )

    evidence = {
        "schema_version": EVIDENCE_SCHEMA,
        "source_schema_version": analysis["schema_version"],
        "provenance": {
            "macro25_capture": _contract_summary(_read(args.macro25_contract)),
            "macro50_capture": _contract_summary(_read(args.macro50_contract)),
        },
        "offline_b20_functional_loss": analysis["offline_b20_functional_loss"],
        "capture_outcome_replay": analysis["capture_outcome_replay"],
        "reference_boundary": analysis["reference_boundary"],
        "by_category": analysis["by_category"],
        "macro50_vs_macro25_occupancy_delta": delta_by_category,
        "replay_consistent_occupancy_delta": replay_consistent_delta_by_category,
        "initial_state_category_comparisons": [
            _initial_comparison(rows, "lost", "retained"),
            _initial_comparison(rows, "lost", "gained"),
            _initial_comparison(rows, "gained", "retained"),
        ],
        "rows": rows,
        "interpretation_boundary": (
            "Action RMS is the matched disagreement between macro25 and macro50 "
            "on the fixed union of both rollout occupancies. It is not expert-action "
            "error because validation experts are unavailable and teacher actions "
            "were not read."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
