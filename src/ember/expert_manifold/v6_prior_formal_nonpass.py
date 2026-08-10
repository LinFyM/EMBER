"""Immutable evidence gate for the RLS macro10 closed-loop non-pass."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ember.expert_manifold.contract import ExpertManifoldError


TRANSITION_SCHEMA = "ember_pi05_v6_historical_baseline_transition_analysis_v2"


def formal_metrics_match(path: Path, *, expected_rows: int = 25) -> bool:
    """Validate the contiguous pre-update macro ledger."""

    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return False
    return (
        len(rows) == expected_rows
        and all(isinstance(row, Mapping) for row in rows)
        and [row.get("macro") for row in rows]
        == list(range(1, expected_rows + 1))
        and [row.get("schedule_macro") for row in rows]
        == list(range(expected_rows))
    )


def _load_nonpass_evidence(
    config: Mapping[str, Any], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    from ember.expert_manifold import v6_prior_contract as contract

    required = {
        "path",
        "bytes",
        "schema",
        "run_commit",
        "metrics_bytes",
        "checkpoint_manifest",
        "strict_correct_results",
        "transition_analysis",
    }
    if set(evidence) != required:
        raise ValueError("unexpected non-pass evidence fields")
    checkpoint = evidence["checkpoint_manifest"]
    strict = evidence["strict_correct_results"]
    transition = evidence["transition_analysis"]
    if not all(
        isinstance(row, Mapping) for row in (checkpoint, strict, transition)
    ):
        raise TypeError("non-pass artifact pointers must be mappings")

    completion_path = contract._runtime_artifact_path(evidence["path"])
    training_root = completion_path.parent
    strict_path = contract._runtime_artifact_path(strict["path"])
    transition_path = contract._runtime_artifact_path(transition["path"])
    baseline = Path(
        config["formal_run"]["decision_evaluation"]["macro0_reference_root"]
    ) / "results.json"
    return {
        "contract": contract,
        "evidence": evidence,
        "checkpoint": checkpoint,
        "strict": strict,
        "transition": transition,
        "completion_path": completion_path,
        "training_root": training_root,
        "metrics_path": training_root / "metrics.jsonl",
        "run": json.loads(
            (training_root / "run_contract.json").read_text(encoding="utf-8")
        ),
        "completion": json.loads(completion_path.read_text(encoding="utf-8")),
        "checkpoint_path": contract._runtime_artifact_path(checkpoint["path"]),
        "strict_path": strict_path,
        "transition_path": transition_path,
        "analysis": json.loads(transition_path.read_text(encoding="utf-8")),
        "baseline_root": contract._runtime_artifact_path(baseline).parent,
    }


def _training_contract_matches(
    config: Mapping[str, Any], loaded: Mapping[str, Any]
) -> bool:
    contract = loaded["contract"]
    evidence = loaded["evidence"]
    completion_path = loaded["completion_path"]
    metrics_path = loaded["metrics_path"]
    commit = str(evidence["run_commit"])
    expected_completion = {
        "schema_version": contract.V6_PRIOR_COMPLETION_SCHEMA,
        "mode": "formal",
        "completed_macro": 10,
        "metrics_rows": 10,
        "content_hash_policy": "disabled_by_owner",
    }
    return (
        completion_path.name == "completion.json"
        and completion_path.stat().st_size == int(evidence["bytes"])
        and evidence["schema"] == contract.V6_PRIOR_COMPLETION_SCHEMA
        and metrics_path.stat().st_size == int(evidence["metrics_bytes"])
        and loaded["completion"] == expected_completion
        and formal_metrics_match(metrics_path, expected_rows=10)
        and contract._run_science_matches(
            config,
            loaded["run"],
            commit,
            mode="formal",
            schedule_start=0,
            schedule_stop=25,
        )
        and contract._runtime_matches(
            loaded["run"].get("runtime"),
            total_macros=25,
            schedule_origin=0,
            checkpoint_macros=[10, 25],
        )
        and contract.git_commit_in_active_authority_lineage(commit)
    )


def _checkpoint_contract_matches(
    config: Mapping[str, Any], loaded: Mapping[str, Any]
) -> bool:
    from ember.expert_manifold.v6_prior_checkpoint import (
        V6_PRIOR_CHECKPOINT_SCHEMA,
        inspect_v6_prior_checkpoint,
    )
    from ember.expert_manifold.v6_prior_run_contract import (
        checkpoint_contract,
        cursor_contract,
    )

    checkpoint = loaded["checkpoint"]
    checkpoint_path = loaded["checkpoint_path"]
    expected_path = (
        loaded["training_root"] / "checkpoints/macro_00000010/manifest.json"
    ).resolve()
    inspection = inspect_v6_prior_checkpoint(
        checkpoint_path.parent,
        expected_cursor_contract=cursor_contract(config, 10),
        expected_checkpoint_contract=checkpoint_contract(loaded["run"]),
        validate_payload_values=False,
    )
    return (
        set(checkpoint) == {"path", "bytes", "schema"}
        and checkpoint_path == expected_path
        and checkpoint_path.stat().st_size == int(checkpoint["bytes"])
        and checkpoint["schema"] == V6_PRIOR_CHECKPOINT_SCHEMA
        and int(inspection["next_macro"]) == 10
    )


def _transition_identity_matches(loaded: Mapping[str, Any]) -> bool:
    analysis = loaded["analysis"]
    roots = analysis["roots"]
    transition = loaded["transition"]
    transition_path = loaded["transition_path"]
    strict = loaded["strict"]
    strict_path = loaded["strict_path"]
    return (
        set(strict) == {"path", "bytes"}
        and strict_path.name == "results.json"
        and strict_path.stat().st_size == int(strict["bytes"])
        and set(transition) == {"path", "bytes", "schema"}
        and transition_path.name == "analysis.json"
        and transition_path.stat().st_size == int(transition["bytes"])
        and transition["schema"] == TRANSITION_SCHEMA
        and analysis.get("schema_version") == TRANSITION_SCHEMA
        and analysis.get("method_families")
        == {
            "historical_baseline": "v6_condition_residual_v2",
            "current_candidate": "v6_anchored_reconciliation_v3",
        }
        and Path(roots["historical_baseline"]["root"]).resolve()
        == loaded["baseline_root"]
        and Path(roots["current_candidate"]["root"]).resolve()
        == strict_path.parent.resolve()
    )


def _transition_outcome_matches(loaded: Mapping[str, Any]) -> bool:
    analysis = loaded["analysis"]
    panels = analysis["panels"]["correct400"]
    delta = analysis["baseline_to_candidate"]["correct400"]["overall"]
    return (
        panels["historical_baseline"]["overall"]["successes"] == 134
        and panels["current_candidate"]["overall"]["successes"] == 140
        and panels["current_candidate"]["nonzero_task_breadth"] == 6
        and delta["gained"] == 21
        and delta["lost"] == 15
        and delta["net"] == 6
        and delta["churn"] == 36
    )


def formal_macro10_nonpass_artifact_matches(
    config: Mapping[str, Any], evidence: Mapping[str, Any]
) -> bool:
    """Rebind the retired state to training, checkpoint, and strict rows."""

    try:
        loaded = _load_nonpass_evidence(config, evidence)
        return all(
            (
                _training_contract_matches(config, loaded),
                _checkpoint_contract_matches(config, loaded),
                _transition_identity_matches(loaded),
                _transition_outcome_matches(loaded),
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        ExpertManifoldError,
    ):
        return False
