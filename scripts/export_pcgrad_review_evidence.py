#!/usr/bin/env python3
"""Export remote-safe evidence for the matched PCGrad review intervention."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


EVIDENCE_SCHEMA = "ember_external_review_pcgrad_training_evidence_v1"
AGGREGATION = "deterministic_pcgrad_v1"
METRIC_FIELDS = (
    "macro",
    "global_mean_functional_loss",
    "gradient_norm_before_clip",
    "pcgrad_arithmetic_mean_norm",
    "pcgrad_aggregate_norm",
    "pcgrad_cosine_to_arithmetic_mean",
    "pcgrad_ordered_pair_count",
    "pcgrad_projection_count",
    "macro_seconds",
    "max_cuda_allocated_bytes",
    "max_cuda_reserved_bytes",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [row.get("macro") for row in rows] != list(range(1, 51)):
        raise ValueError("PCGrad evidence requires complete macros 1 through 50")
    if any(row.get("task_gradient_aggregation") != AGGREGATION for row in rows):
        raise ValueError("training metrics changed aggregation contract")
    return [
        {field: row[field] for field in METRIC_FIELDS}
        for row in rows
    ]


def _checkpoint_summary(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    return {
        "schema_version": manifest["schema_version"],
        "run_contract_schema": manifest["run_contract_schema"],
        "next_macro": manifest["next_macro"],
        "world_size": manifest["world_size"],
        "files": manifest["files"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run = args.training_run.resolve()
    contract = _read_json(run / "run_contract.json")
    intervention = contract.get("diagnostic_intervention", {})
    if intervention.get("kind") != "replace_arithmetic_mean_with_deterministic_pcgrad_v1":
        raise ValueError("run is not the registered PCGrad intervention")
    metrics = _read_metrics(run / "metrics.jsonl")
    projections = [int(row["pcgrad_projection_count"]) for row in metrics]
    cosines = [float(row["pcgrad_cosine_to_arithmetic_mean"]) for row in metrics]
    norm_ratios = [
        float(row["pcgrad_aggregate_norm"])
        / float(row["pcgrad_arithmetic_mean_norm"])
        for row in metrics
    ]
    evidence = {
        "schema_version": EVIDENCE_SCHEMA,
        "repository_commit": contract["git"]["commit"],
        "diagnostic_intervention": intervention,
        "fixed_training_contract": {
            "conditioning_training": contract["conditioning_training"],
            "data": contract["data"],
            "optimizer": contract["optimization"]["optimizer"],
            "scheduler": contract["optimization"]["scheduler"],
            "precision": contract["optimization"]["precision"],
            "world_size": contract["optimization"]["distributed"].get(
                "world_size", 6
            ),
        },
        "checkpoint_manifests": {
            "macro25": _checkpoint_summary(
                run / "checkpoints/macro_00000025/checkpoint_manifest.json"
            ),
            "macro50": _checkpoint_summary(
                run / "checkpoints/macro_00000050/checkpoint_manifest.json"
            ),
        },
        "aggregate": {
            "functional_loss": {
                "macro1": metrics[0]["global_mean_functional_loss"],
                "macro25": metrics[24]["global_mean_functional_loss"],
                "macro50": metrics[49]["global_mean_functional_loss"],
                "minimum": min(
                    float(row["global_mean_functional_loss"]) for row in metrics
                ),
            },
            "projection_count": {
                "minimum": min(projections),
                "maximum": max(projections),
                "mean": statistics.fmean(projections),
            },
            "cosine_to_arithmetic_mean": {
                "minimum": min(cosines),
                "maximum": max(cosines),
                "mean": statistics.fmean(cosines),
            },
            "aggregate_to_mean_norm_ratio_mean": statistics.fmean(norm_ratios),
            "macro_seconds_mean": statistics.fmean(
                float(row["macro_seconds"]) for row in metrics
            ),
            "max_cuda_allocated_bytes": max(
                int(row["max_cuda_allocated_bytes"]) for row in metrics
            ),
            "max_cuda_reserved_bytes": max(
                int(row["max_cuda_reserved_bytes"]) for row in metrics
            ),
        },
        "per_macro_metrics": metrics,
        "causal_boundary": (
            "This arm changes only equal-task arithmetic-mean aggregation to the "
            "registered deterministic PCGrad rule while retaining AdamW and its "
            "moment dynamics. It can adjudicate gradient aggregation under the "
            "same optimizer, but cannot independently adjudicate AdamW moments."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
