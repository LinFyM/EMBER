"""Strict row-product validation and aggregation for endpoint diagnostics."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from ember.writer.endpoint_validation import (
    ENDPOINT_SUMMARY_SCHEMA,
    METRICS,
    SEALED_PANEL_PAYLOAD_SHA256,
    SUITES,
    EndpointCandidate,
    endpoint_noise_seed,
)
from ember.writer.model import WriterModelError


def _finite_array(label: str, values: Any) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise WriterModelError(
            f"endpoint diagnostic {label} aggregation shape changed"
        ) from error
    if array.size == 0 or not bool(np.isfinite(array).all()):
        raise WriterModelError(
            f"endpoint diagnostic {label} aggregation input is non-finite"
        )
    return array


def _metric_summary(
    values: Any,
    dimensions: Any,
    label: str,
) -> dict[str, Any]:
    metric_values = _finite_array(f"{label} scalar", values)
    metric_dimensions = _finite_array(f"{label} dimensions", dimensions)
    if metric_dimensions.ndim != 2 or metric_dimensions.shape[1] != 7:
        raise WriterModelError(
            f"endpoint diagnostic {label} dimensions changed"
        )
    mean_value = float(metric_values.mean())
    mean_dimensions = metric_dimensions.mean(axis=0)
    if not math.isfinite(mean_value) or not bool(
        np.isfinite(mean_dimensions).all()
    ):
        raise WriterModelError(
            f"endpoint diagnostic {label} aggregation is non-finite"
        )
    return {
        "mse": mean_value,
        "quality": -mean_value,
        "per_action_dimension_mse": mean_dimensions.tolist(),
    }


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[EndpointCandidate],
) -> dict[str, Any]:
    records = []
    for candidate in candidates:
        selected = [
            row
            for row in rows
            if row["candidate_id"] == candidate.candidate_id
        ]
        per_task = {}
        for task_id in sorted(
            {int(row["global_task_id"]) for row in selected}
        ):
            task_rows = [
                row
                for row in selected
                if int(row["global_task_id"]) == task_id
            ]
            metric_rows = {}
            for name in METRICS:
                metric_rows[name] = _metric_summary(
                    [row["metrics"][name]["mse"] for row in task_rows],
                    [
                        row["metrics"][name]["per_action_dimension_mse"]
                        for row in task_rows
                    ],
                    f"{candidate.candidate_id} task {task_id} {name}",
                )
            per_task[str(task_id)] = {
                "rows": len(task_rows),
                "suite": task_rows[0]["suite"],
                "metrics": metric_rows,
            }
        aggregate = {}
        for name in METRICS:
            aggregate[name] = _metric_summary(
                [row["metrics"][name]["mse"] for row in per_task.values()],
                [
                    row["metrics"][name]["per_action_dimension_mse"]
                    for row in per_task.values()
                ],
                f"{candidate.candidate_id} task-balanced {name}",
            )
        per_suite = {}
        for suite in SUITES:
            suite_tasks = [
                row for row in per_task.values() if row["suite"] == suite
            ]
            per_suite[suite] = {
                name: float(
                    _finite_array(
                        f"{candidate.candidate_id} {suite} {name}",
                        [row["metrics"][name]["mse"] for row in suite_tasks],
                    ).mean()
                )
                for name in METRICS
            }
        records.append(
            {
                **candidate.record(),
                "rows": len(selected),
                "aggregate": aggregate,
                "per_suite": per_suite,
                "per_task": per_task,
            }
        )
    return {
        "schema_version": ENDPOINT_SUMMARY_SCHEMA,
        "metrics": list(METRICS),
        "primary_metric": METRICS[0],
        "candidates": records,
    }


def _selected_panel_by_ordinal(
    manifest: Mapping[str, Any],
    max_groups_per_task: int | None,
) -> dict[int, Mapping[str, Any]]:
    selected = [
        row
        for row in manifest.get("rows", [])
        if max_groups_per_task is None
        or int(row["video_group"]) < max_groups_per_task
    ]
    result: dict[int, Mapping[str, Any]] = {}
    for row in selected:
        ordinal = int(row["ordinal"])
        if ordinal in result:
            raise WriterModelError(
                "endpoint diagnostic sealed panel duplicated an ordinal"
            )
        result[ordinal] = row
    if not result:
        raise WriterModelError("endpoint diagnostic selected an empty panel")
    return result


def _validate_output_metrics(
    candidate_id: str,
    ordinal: int,
    metrics: Any,
) -> None:
    if not isinstance(metrics, Mapping) or set(metrics) != set(METRICS):
        raise WriterModelError(
            "endpoint diagnostic output metric set changed"
        )
    for name in METRICS:
        record = metrics[name]
        if not isinstance(record, Mapping):
            raise WriterModelError(
                "endpoint diagnostic output metric record changed"
            )
        _finite_array(
            f"{candidate_id} ordinal {ordinal} {name} scalar",
            [record.get("mse")],
        )
        dimensions = _finite_array(
            f"{candidate_id} ordinal {ordinal} {name} dimensions",
            record.get("per_action_dimension_mse"),
        )
        if dimensions.shape != (7,):
            raise WriterModelError(
                "endpoint diagnostic output metric dimensions changed"
            )


def _validate_endpoint_output_row(
    row: Mapping[str, Any],
    candidate: EndpointCandidate,
    panel_row: Mapping[str, Any],
    ordinal: int,
) -> None:
    if (
        row.get("family") != candidate.family
        or int(row.get("checkpoint_cursor", -1))
        != candidate.checkpoint_cursor
    ):
        raise WriterModelError(
            "endpoint diagnostic output candidate identity changed"
        )
    if any(row.get(field) != value for field, value in panel_row.items()):
        raise WriterModelError(
            "endpoint diagnostic output panel identity changed"
        )
    global_task_id = int(panel_row["global_task_id"])
    if (
        row.get("suite") != SUITES[global_task_id // 10]
        or int(row.get("suite_task_id", -1)) != global_task_id % 10
    ):
        raise WriterModelError(
            "endpoint diagnostic output task identity changed"
        )
    if int(row.get("endpoint_noise_seed", -1)) != endpoint_noise_seed(
        SEALED_PANEL_PAYLOAD_SHA256, panel_row
    ):
        raise WriterModelError(
            "endpoint diagnostic output noise identity changed"
        )
    _validate_output_metrics(
        candidate.candidate_id,
        ordinal,
        row.get("metrics"),
    )
    wall_seconds = float(row.get("group_wall_seconds", float("nan")))
    if not math.isfinite(wall_seconds) or wall_seconds < 0:
        raise WriterModelError(
            "endpoint diagnostic output timing is non-finite"
        )


def _validate_endpoint_output_rows(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[EndpointCandidate],
    manifest: Mapping[str, Any],
    max_groups_per_task: int | None,
) -> int:
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in candidates
    }
    if not candidate_by_id or len(candidate_by_id) != len(candidates):
        raise WriterModelError("endpoint diagnostic candidates are duplicated")
    panel_by_ordinal = _selected_panel_by_ordinal(
        manifest,
        max_groups_per_task,
    )
    expected = {
        (candidate_id, ordinal)
        for candidate_id in candidate_by_id
        for ordinal in panel_by_ordinal
    }
    observed: set[tuple[str, int]] = set()
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        try:
            ordinal = int(row["ordinal"])
        except (KeyError, TypeError, ValueError) as error:
            raise WriterModelError(
                "endpoint diagnostic output ordinal changed"
            ) from error
        key = (candidate_id, ordinal)
        if key in observed:
            raise WriterModelError(
                "endpoint diagnostic output duplicated a candidate-panel row"
            )
        observed.add(key)
        candidate = candidate_by_id.get(candidate_id)
        panel_row = panel_by_ordinal.get(ordinal)
        if candidate is None or panel_row is None:
            raise WriterModelError(
                "endpoint diagnostic output escaped its candidate-panel product"
            )
        _validate_endpoint_output_row(
            row,
            candidate,
            panel_row,
            ordinal,
        )

    if observed != expected:
        raise WriterModelError(
            "endpoint diagnostic output candidate-panel product is incomplete"
        )
    return len(expected)
