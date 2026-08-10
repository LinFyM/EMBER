"""Cross-family historical baseline transition analysis."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from ember.pi05_eval.writer_family_registry import (
    HISTORICAL_TRANSITION_BASELINE_FAMILIES,
    HISTORICAL_TRANSITION_CANDIDATE_MACROS,
)


HISTORICAL_BASELINE_TRANSITION_SCHEMA = (
    "ember_pi05_v6_historical_baseline_transition_analysis_v2"
)


def _historical_projection(
    result: Mapping[str, Any],
    shared: Any,
) -> dict[str, Any]:
    projection = shared._scientific_projection(
        result,
        allow_checkpoint_change=True,
    )
    projection.pop("git", None)
    writer = projection.get("writer")
    tokenizer = projection.get("tokenizer")
    normalization = projection.get("normalization")
    if not all(isinstance(value, dict) for value in (writer, tokenizer, normalization)):
        shared._fail("historical transition is missing its shared scientific contract")
    for key in (
        "execution_backend",
        "config",
        "writer_asset",
        "evaluation_authority",
    ):
        writer.pop(key, None)
    tokenizer.pop("manifest_path", None)
    normalization.pop("path", None)
    return projection


def _indexed_historical_pair(
    results_by_root: Mapping[str, Mapping[str, Any]],
    shared: Any,
) -> tuple[str, str, tuple[Any, ...], tuple[Any, ...]]:
    if len(results_by_root) != 2:
        shared._fail("historical baseline transition requires exactly two roots")
    by_family = {}
    for root, result in results_by_root.items():
        indexed = shared._formal_panel_index(result)
        if result["adapter"]["video_condition"] != "correct":
            shared._fail(
                "historical baseline transition accepts only correct-video roots"
            )
        family = shared._writer_family(result["adapter"])[0]
        if family in by_family:
            shared._fail(
                "historical baseline transition contains a duplicate method family"
            )
        by_family[family] = (root, result, indexed)
    supported = [
        (baseline_family, candidate_family)
        for candidate_family, baseline_family in (
            HISTORICAL_TRANSITION_BASELINE_FAMILIES.items()
        )
        if set(by_family) == {baseline_family, candidate_family}
    ]
    if len(supported) != 1:
        shared._fail(
            "historical baseline transition requires exactly one supported "
            "baseline and current candidate family pair"
        )
    baseline_family, candidate_family = supported[0]
    baseline = by_family[baseline_family]
    candidate = by_family[candidate_family]
    shared._method_macro(
        baseline[1],
        allowed_macros=(0,),
        context="historical baseline transition baseline",
    )
    shared._method_macro(
        candidate[1],
        allowed_macros=HISTORICAL_TRANSITION_CANDIDATE_MACROS[candidate_family],
        context="historical baseline transition candidate",
    )
    if _historical_projection(baseline[1], shared) != _historical_projection(
        candidate[1],
        shared,
    ):
        shared._fail(
            "historical baseline transition changed its shared scientific contract"
        )
    shared._assert_row_pairing(
        baseline[2],
        candidate[2],
        require_same_actual_video=True,
    )
    return baseline_family, candidate_family, baseline, candidate


def _root_evidence(value: tuple[Any, ...], shared: Any) -> dict[str, Any]:
    root, result, _ = value
    adapter = result["adapter"]
    family, family_contract = shared._writer_family(adapter)
    asset = adapter["writer_asset"]
    return {
        "root": root,
        "method_family": family,
        "method_macro": int(asset["method_macro"]),
        "checkpoint_kind": asset["kind"],
        "adapter_schema": family_contract["adapter_schema"],
        "config_schema": family_contract["config_schema"],
        "contract_reference": result["contract_reference"],
        "git": copy.deepcopy(result["paired_control"]["git"]),
        "parallel_provenance": copy.deepcopy(
            result["paired_control"].get("parallel", {})
        ),
    }


def historical_baseline_transition_analysis(
    results_by_root: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare one sealed candidate with its explicit immutable macro0 baseline."""

    from ember.pi05_eval import analysis as shared

    baseline_family, candidate_family, baseline, candidate = (
        _indexed_historical_pair(results_by_root, shared)
    )
    panels = {
        "correct80": {
            "historical_baseline": shared._prefix_rows(baseline[2], 10),
            "current_candidate": shared._prefix_rows(candidate[2], 10),
        },
        "correct400": {
            "historical_baseline": list(baseline[2].values()),
            "current_candidate": list(candidate[2].values()),
        },
    }
    return {
        "schema_version": HISTORICAL_BASELINE_TRANSITION_SCHEMA,
        "analysis_scope": (
            "cross_family_historical_baseline_transition_not_checkpoint_curve"
        ),
        "method_families": {
            "historical_baseline": baseline_family,
            "current_candidate": candidate_family,
        },
        "contract_audit": {
            "native_family_validation_each_root": True,
            "formal_validation_8x50_each": True,
            "same_shared_scientific_contract": True,
            "same_state_rng_language_and_correct_video_identity": True,
            "family_labels_preserved": True,
            "checkpoint_curve_membership_claimed": False,
        },
        "row_selection": {
            "correct80": "same validated correct400 root rows with init_state_id < 10",
            "correct400": "all formal validation rows",
        },
        "roots": {
            "historical_baseline": _root_evidence(baseline, shared),
            "current_candidate": _root_evidence(candidate, shared),
        },
        "panels": {
            panel: {
                role: shared.summarize_panel(rows)
                for role, rows in role_rows.items()
            }
            for panel, role_rows in panels.items()
        },
        "baseline_to_candidate": {
            panel: shared.paired_transition_summary(
                role_rows["historical_baseline"],
                role_rows["current_candidate"],
            )
            for panel, role_rows in panels.items()
        },
        "metric_definitions": {
            "gained": (
                "historical failure and current success on the identical episode key"
            ),
            "lost": (
                "historical success and current failure on the identical episode key"
            ),
            "churn": "gained plus lost",
            "nonzero_task_breadth": (
                "tasks with at least one success in this exact panel"
            ),
            "cross_family_warning": (
                "native family labels are retained; this artifact is not a "
                "within-family checkpoint curve"
            ),
        },
    }
