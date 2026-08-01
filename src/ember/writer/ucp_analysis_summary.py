"""Recursive scalar/vector aggregation for UCP internal-analysis records."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ember.writer.model import CompleteLoRAWriter, WriterModelError
from ember.writer.ucp_analysis import (
    CONDITIONS,
    STAGES,
    coordinate_summary,
    effective_metrics,
    fixed_sequence,
    lora_geometry,
    mapping_metrics,
    program_signature,
    reader_attention_summary,
    relative_metrics,
)


def stage_signatures(
    encoded: Mapping[str, Any], row: int,
) -> dict[str, torch.Tensor]:
    """Create fixed-shape signatures for every canonical evidence/Program stage."""

    token_valid = encoded["valid_tokens"][row]
    frame_valid = encoded["valid_frames"][row]
    interval_valid = encoded["valid_intervals"][row]
    semantic_valid = encoded["valid_semantics"][row]
    result = {
        "q_text": encoded["q_text"][row, token_valid].float(),
        "multimodal_m": fixed_sequence(
            encoded["packed_m"][row, :, token_valid], frame_valid,
        ),
        "grounded_g": fixed_sequence(
            encoded["packed_g"][row, :, token_valid], frame_valid,
        ),
        "absolute_x": fixed_sequence(
            encoded["packed_x"][row, :, token_valid], frame_valid,
        ),
        "raw_action": fixed_sequence(encoded["packed_raw"][row], frame_valid),
        "action_probe": fixed_sequence(encoded["packed_action"][row], frame_valid),
        "coordinates": encoded["coordinates"][row].float(),
    }
    values = (
        ("initial", encoded["initial"][row]),
        ("program_block_1", encoded["blocks"][0][row]),
        ("program_block_2", encoded["blocks"][1][row]),
        ("final", encoded["final"][row]),
    )
    for name, value in values:
        key = f"{name}_program" if name == "initial" else name
        result[key] = program_signature(value, interval_valid, semantic_valid)
        if name in {"initial", "final"}:
            for kind in ("x", "a", "d"):
                result[f"{name}_{kind}"] = program_signature(
                    value, interval_valid, semantic_valid, kind=kind,
                )
    result["final_program"] = result.pop("final")
    return result


def matched_diagnostics(
    writer: CompleteLoRAWriter, encoded: Mapping[str, Any]
) -> dict[str, Any]:
    signatures = [stage_signatures(encoded, row) for row in range(len(CONDITIONS))]
    comparisons = {}
    for row, condition in enumerate(CONDITIONS):
        comparisons[condition] = {
            stage: relative_metrics(signatures[0][stage], signatures[row][stage])
            for stage in STAGES
        }
        comparisons[condition].update({
            "factor_output": mapping_metrics(
                encoded["factor_states"][0], encoded["factor_states"][row]
            ),
            "public_a": mapping_metrics(
                encoded["states"][0], encoded["states"][row], select="a"
            ),
            "public_b": mapping_metrics(
                encoded["states"][0], encoded["states"][row], select="b"
            ),
            "effective_ba": effective_metrics(
                writer, encoded["states"][0], encoded["states"][row]
            ),
            "policy_action": relative_metrics(
                encoded["actions"][0], encoded["actions"][row]
            ),
        })
    return {
        "comparisons": comparisons,
        "readers": {
            condition: reader_attention_summary(
                encoded["attention"][row : row + 1],
                encoded["valid_intervals"][row : row + 1],
                encoded["valid_semantics"][row : row + 1],
            )
            for row, condition in enumerate(CONDITIONS)
        },
        "coordinates": {
            condition: coordinate_summary(encoded["coordinates"][row])
            for row, condition in enumerate(CONDITIONS)
        },
        "geometry": {
            condition: lora_geometry(writer, encoded["states"][row])
            for row, condition in enumerate(CONDITIONS)
        },
    }


def validate_finite_tree(value: Any, path: str = "root") -> None:
    """Reject a non-finite numeric leaf before any retained JSON is written."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            validate_finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise WriterModelError(f"non-finite UCP result at {path}")


def validate_rank_payloads(
    payloads: Sequence[Mapping[str, Any]], references_per_task: int
) -> list[Mapping[str, Any]]:
    """Validate the exact eight-task by reference-ordinal Cartesian panel."""

    if references_per_task <= 0:
        raise WriterModelError("invalid UCP reference count")
    rows = []
    for rank, payload in enumerate(payloads):
        validate_finite_tree(payload, f"rank_{rank}")
        if payload.get("rank") != rank or not isinstance(payload.get("rows"), list):
            raise WriterModelError("UCP analysis rank payload identity changed")
        rows.extend(payload["rows"])
    task_ids = sorted({int(row["global_task_id"]) for row in rows})
    actual = [
        (int(row["global_task_id"]), int(row["reference_ordinal"]))
        for row in rows
    ]
    expected = [
        (task_id, ordinal) for task_id in task_ids
        for ordinal in range(references_per_task)
    ]
    if len(task_ids) != 8 or sorted(actual) != expected or len(actual) != len(set(actual)):
        raise WriterModelError("UCP analysis panel is not exact 8-task x references")
    identities = {}
    for row in rows:
        task_id = int(row["global_task_id"])
        value = (str(row["suite"]), int(row["task_id"]))
        if task_id in identities and identities[task_id] != value:
            raise WriterModelError("UCP task identity changed across references")
        identities[task_id] = value
    return sorted(rows, key=lambda row: (
        int(row["global_task_id"]), int(row["reference_ordinal"]),
    ))


def _numeric_summary(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0 or not np.isfinite(values).all():
        raise WriterModelError("non-finite or empty UCP summary values")
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
    }


def aggregate_numeric_records(records: Sequence[Any]) -> dict[str, Any]:
    """Recursively summarize shared numeric leaves and fixed-length vectors."""

    if not records:
        raise WriterModelError("cannot aggregate empty UCP records")
    mapping_flags = [isinstance(record, Mapping) for record in records]
    if any(mapping_flags) and not all(mapping_flags):
        raise WriterModelError("UCP summary record changed mapping type")
    if all(mapping_flags):
        key_sets = [set(record) for record in records]
        if any(keys != key_sets[0] for keys in key_sets[1:]):
            raise WriterModelError("UCP summary records changed key set")
        keys = key_sets[0]
        result = {}
        for key in sorted(keys):
            nested = aggregate_numeric_records([record[key] for record in records])
            if nested:
                result[key] = nested
        return result
    scalar_types = (int, float, np.integer, np.floating)
    if all(
        isinstance(record, scalar_types) and not isinstance(record, bool)
        for record in records
    ):
        return _numeric_summary(np.asarray(records, dtype=np.float64))
    sequence_flags = [isinstance(record, (list, tuple)) for record in records]
    if any(sequence_flags) and not all(sequence_flags):
        raise WriterModelError("UCP summary record changed sequence type")
    if all(sequence_flags):
        if any(len(record) != len(records[0]) for record in records):
            raise WriterModelError("UCP summary numeric vector changed length")
        try:
            values = np.asarray(records, dtype=np.float64)
        except (TypeError, ValueError):
            return {}
        if values.ndim >= 2:
            return _numeric_summary(values)
    return {}


def _variance_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = aggregate_numeric_records([
        row["same_task_video_variance"] for row in rows
    ])
    for kind in ("effective_ba", "fixed_policy_action"):
        flags = [row["same_task_video_variance"][kind]["estimable"] for row in rows]
        if not all(isinstance(flag, bool) for flag in flags) or len(set(flags)) != 1:
            raise WriterModelError("UCP video-variance estimability changed within task")
        result.setdefault(kind, {})["estimable"] = flags[0]
    return result


def _condition_summary(
    rows: Sequence[Mapping[str, Any]], metric_stages: Sequence[str]
) -> dict[str, Any]:
    result = {
        "conditions": {}, "reader_attention": {},
        "coordinate_routing": {}, "lora_geometry": {},
    }
    for condition in CONDITIONS:
        result["conditions"][condition] = {
            stage: aggregate_numeric_records([
                row["comparisons_to_correct"][condition][stage] for row in rows
            ])
            for stage in metric_stages
        }
        for key in ("reader_attention", "coordinate_routing", "lora_geometry"):
            result[key][condition] = aggregate_numeric_records([
                row[key][condition] for row in rows
            ])
    return result


def summarize_records(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise WriterModelError("cannot summarize empty UCP analysis")
    metric_stages = (
        *STAGES, "factor_output", "public_a", "public_b", "effective_ba",
        "policy_action",
    )
    counterfactual_keys = (
        "type_ablations", "fixed_x_vary_a_d", "fixed_a_d_vary_x",
        "dynamic_scale", "variant_recompute", "target_identity_permutation",
        "rank_gauge_permutation",
    )
    result = _condition_summary(rows, metric_stages)
    result["canonical_program_parity"] = aggregate_numeric_records([
        row["canonical_program_parity"] for row in rows
    ])
    result["counterfactuals"] = {
        key: aggregate_numeric_records([row[key] for row in rows])
        for key in counterfactual_keys
    }
    result["per_task"] = {}
    for task_id in sorted({int(row["global_task_id"]) for row in rows}):
        selected = [row for row in rows if int(row["global_task_id"]) == task_id]
        key = f"{selected[0]['suite']}:task_{int(selected[0]['task_id']):02d}"
        task = _condition_summary(selected, metric_stages)
        task.update({
            "global_task_id": task_id,
            "references": len(selected),
            "same_task_video_variance": _variance_summary(selected),
            "canonical_program_parity": aggregate_numeric_records([
                row["canonical_program_parity"] for row in selected
            ]),
            "counterfactuals": {
                name: aggregate_numeric_records([row[name] for row in selected])
                for name in counterfactual_keys
            },
        })
        result["per_task"][key] = task
    result["factor_gauge_caveat"] = (
        "raw factor/public A/B coordinates are gauge-dependent; effective BA and "
        "canonical fixed-query action are primary functional evidence. A rank "
        "gauge permutation preserves mathematical BA but can change the "
        "factorized bf16 reduction order, so its action drift is a measured "
        "low-precision execution effect rather than the gauge sanity gate"
    )
    return result
