"""Freeze bounded target-support fits before source closed-loop screening."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ember.gate_zero_oracle_artifacts import atomic_json, sha256_file
from ember.gate_zero_oracle_report import _fit_evidence
from ember.gate_zero_support.contract import load_target_support_screen_spec


GRANT_STATUS = "target_support_fit_selections_frozen_before_closed_loop_screening"
GRANT_NAME = "screening_grant.json"


class GateZeroTargetSupportScreenError(RuntimeError):
    """Raised when fit evidence cannot authorize the frozen screening surface."""


def canonical_support_screening_shards(
    *, variants: list[str] | None = None, task_ids: list[int] | None = None
) -> list[list[tuple[int, str]]]:
    """Partition each bounded support stage into deterministic two-arm shards."""

    variants = (
        ["last_two_qv_r8", "all_expert_qv_r8", "official_default_r8"]
        if variants is None
        else variants
    )
    task_ids = [3, 4] if task_ids is None else task_ids
    if task_ids != [3, 4] or len(variants) not in {1, 3} or len(set(variants)) != len(variants):
        raise GateZeroTargetSupportScreenError("invalid bounded support-screening scope")
    shards = []
    for task_id in task_ids:
        arms = [(task_id, condition) for condition in ["frozen_base", *variants]]
        shards.extend(arms[index : index + 2] for index in range(0, len(arms), 2))
    return shards


def assigned_support_screening_arms(
    *,
    rank: int,
    world_size: int,
    variants: list[str] | None = None,
    task_ids: list[int] | None = None,
) -> list[tuple[int, str]]:
    if world_size not in {1, 2, 4} or rank < 0 or rank >= world_size:
        raise GateZeroTargetSupportScreenError("invalid support-screening topology")
    shards = canonical_support_screening_shards(
        variants=variants, task_ids=task_ids
    )
    if world_size > len(shards):
        raise GateZeroTargetSupportScreenError("support-screening has idle ranks")
    return [
        arm
        for index, shard in enumerate(shards)
        if index % world_size == rank
        for arm in shard
    ]


def _load_result(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateZeroTargetSupportScreenError("invalid support fit result") from error
    if not isinstance(value, dict):
        raise GateZeroTargetSupportScreenError("invalid support fit result")
    return value


def _support_fit_evidence(
    output: Path,
    *,
    variant: str,
    task_id: int,
    variant_spec: Mapping[str, Any],
    expected_authorities: Mapping[str, str],
) -> dict[str, Any]:
    evidence = _fit_evidence(
        output,
        variant=variant,
        task_id=task_id,
        expected_authorities=expected_authorities,
    )
    result = _load_result(output / "fit_selection_result.json")
    expected = {
        "pilot_scope": (
            "source_only_gate_zero_target_support_audit_"
            "not_final_writer_target_support"
        ),
        "capacity_role": "matched_target_support_audit_candidate",
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise GateZeroTargetSupportScreenError(f"support fit changed {key}")
    authorities = result.get("authorities", {})
    if (
        authorities.get("validation_numeric_access") is not False
        or authorities.get("held_numeric_access") is not False
    ):
        raise GateZeroTargetSupportScreenError("support fit opened forbidden numeric access")
    expected_parameters = variant_spec["expected_trainable_parameters"]
    if (
        evidence["selected_trainable_parameters"] != expected_parameters
        or result.get("trainable", {}).get("trainable_parameters")
        != expected_parameters
    ):
        raise GateZeroTargetSupportScreenError("support fit parameter count changed")
    return evidence


def _query_metrics(evidence: Mapping[str, Any]) -> dict[str, Any]:
    metrics = evidence.get("selected_query_metrics", {})
    base = metrics.get("base_query_flow_mse")
    query = metrics.get("query_flow_mse")
    drift = metrics.get("action_drift_proxy")
    step = evidence.get("selected_step")
    row_hash = metrics.get("query_row_keys_sha256")
    sample_count = metrics.get("query_sample_count")
    anchor_hash = metrics.get("anchor_row_keys_sha256")
    anchor_count = metrics.get("anchor_count")
    if (
        not isinstance(base, (int, float))
        or not isinstance(query, (int, float))
        or not isinstance(drift, (int, float))
        or not all(math.isfinite(float(value)) for value in (base, query, drift))
        or base <= 0
        or query < 0
        or drift < 0
        or not isinstance(step, int)
        or step < 0
        or not isinstance(row_hash, str)
        or len(row_hash) != 64
        or not isinstance(anchor_hash, str)
        or len(anchor_hash) != 64
        or not isinstance(sample_count, int)
        or sample_count <= 0
        or not isinstance(anchor_count, int)
        or anchor_count <= 0
    ):
        raise GateZeroTargetSupportScreenError("selected query metrics are invalid")
    return {
        "selected_step": step,
        "base_query_flow_mse": float(base),
        "query_flow_mse": float(query),
        "query_reduction_fraction": (float(base) - float(query)) / float(base),
        "action_drift_proxy": float(drift),
        "query_row_keys_sha256": row_hash,
        "query_sample_count": sample_count,
        "anchor_row_keys_sha256": anchor_hash,
        "anchor_count": anchor_count,
    }


def rank_query_supports(
    *,
    evidence: Mapping[str, Mapping[str, Any]],
    variants: list[str],
    task_ids: list[int],
    parameter_counts: Mapping[str, int],
    drift_proxy_max: float,
) -> list[dict[str, Any]]:
    """Apply the frozen query ranking without opening rollout/report outcomes."""

    rows: list[dict[str, Any]] = []
    identity_by_task: dict[int, tuple[Any, ...]] = {}
    for variant in variants:
        task_metrics: dict[str, dict[str, Any]] = {}
        for task_id in task_ids:
            metrics = _query_metrics(evidence[f"{variant}:task{task_id}"])
            if metrics["action_drift_proxy"] > drift_proxy_max:
                raise GateZeroTargetSupportScreenError("selected candidate exceeds drift cap")
            identity = (
                metrics["query_row_keys_sha256"],
                metrics["query_sample_count"],
                metrics["anchor_row_keys_sha256"],
                metrics["anchor_count"],
            )
            if task_id in identity_by_task and identity_by_task[task_id] != identity:
                raise GateZeroTargetSupportScreenError("query identity differs across supports")
            identity_by_task[task_id] = identity
            task_metrics[str(task_id)] = metrics
        reductions = [
            task_metrics[str(task_id)]["query_reduction_fraction"]
            for task_id in task_ids
        ]
        positive = sum(
            task_metrics[str(task_id)]["selected_step"] > 0 and reduction > 0
            for task_id, reduction in zip(task_ids, reductions, strict=True)
        )
        rows.append(
            {
                "variant": variant,
                "positive_query_task_count": positive,
                "median_query_reduction_fraction": statistics.median(reductions),
                "trainable_parameters": parameter_counts[variant],
                "task_metrics": task_metrics,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -row["positive_query_task_count"],
            -row["median_query_reduction_fraction"],
            row["trainable_parameters"],
            row["variant"],
        ),
    )


def _validate_screening_arms(
    *,
    arms: list[dict[str, Any]],
    variants: list[str],
    task_ids: list[int],
    expected_init_state_indices: list[int],
    expected_seeds: list[int],
) -> dict[tuple[int, str], dict[str, Any]]:
    expected = {
        (task_id, condition)
        for task_id in task_ids
        for condition in ["frozen_base", *variants]
    }
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for arm in arms:
        key = (arm.get("task_id"), arm.get("condition"))
        successes = arm.get("successes")
        if (
            key not in expected
            or key in by_key
            or arm.get("mechanics_valid") is not True
            or not isinstance(successes, list)
            or len(successes) != len(expected_seeds)
            or any(not isinstance(value, bool) for value in successes)
            or arm.get("official_rollout_init_state_indices")
            != expected_init_state_indices
            or arm.get("seeds") != expected_seeds
        ):
            raise GateZeroTargetSupportScreenError("invalid support-screening arm")
        by_key[key] = arm
    if set(by_key) != expected:
        raise GateZeroTargetSupportScreenError("support-screening arms are incomplete")
    return by_key


def _decision_query_metrics(
    grant: Mapping[str, Any], *, variant: str, task_id: int
) -> tuple[float, float]:
    metrics = grant.get("fit_evidence", {}).get(
        f"{variant}:task{task_id}", {}
    ).get("selected_query_metrics", {})
    base = metrics.get("base_query_flow_mse")
    query = metrics.get("query_flow_mse")
    drift = metrics.get("action_drift_proxy")
    if (
        not isinstance(base, (int, float))
        or not isinstance(query, (int, float))
        or not isinstance(drift, (int, float))
        or not all(math.isfinite(float(value)) for value in (base, query, drift))
        or base <= 0
        or query < 0
        or drift < 0
    ):
        raise GateZeroTargetSupportScreenError("grant query decision metrics are invalid")
    return (float(base) - float(query)) / float(base), float(drift)


def decide_support_screening(
    *,
    arms: list[dict[str, Any]],
    grant: Mapping[str, Any],
    variants: list[str],
    task_ids: list[int],
    parameter_counts: Mapping[str, int],
    thresholds: Mapping[str, Any],
    expected_init_state_indices: list[int],
    expected_seeds: list[int],
    rank_stage: str = "rank8",
) -> dict[str, Any]:
    """Select a support only when every frozen screening test passes."""

    if rank_stage not in {"rank8", "rank16", "mature_positive_control"}:
        raise GateZeroTargetSupportScreenError("invalid support-screening rank stage")

    by_key = _validate_screening_arms(
        arms=arms,
        variants=variants,
        task_ids=task_ids,
        expected_init_state_indices=expected_init_state_indices,
        expected_seeds=expected_seeds,
    )
    candidates = []
    for variant in variants:
        gains = []
        reductions = []
        drifts = []
        task_metrics = {}
        for task_id in task_ids:
            base_successes = sum(by_key[(task_id, "frozen_base")]["successes"])
            own_successes = sum(by_key[(task_id, variant)]["successes"])
            gain = 100.0 * (own_successes - base_successes) / len(expected_seeds)
            reduction, drift = _decision_query_metrics(
                grant, variant=variant, task_id=task_id
            )
            gains.append(gain)
            reductions.append(reduction)
            drifts.append(drift)
            task_metrics[str(task_id)] = {
                "base_successes": base_successes,
                "own_successes": own_successes,
                "success_gain_pp": gain,
                "query_loss_reduction_fraction": reduction,
                "selection_drift_proxy": drift,
            }
        positive_count = sum(value > 0 for value in gains)
        aggregate = {
            "median_success_gain_pp": statistics.median(gains),
            "positive_task_count": positive_count,
            "positive_task_fraction": positive_count / len(task_ids),
            "median_query_loss_reduction_fraction": statistics.median(reductions),
            "median_selection_drift_proxy": statistics.median(drifts),
        }
        drift_is_diagnostic = bool(
            thresholds.get("selection_drift_is_diagnostic_only", False)
        )
        checks = {
            "median_success_gain": aggregate["median_success_gain_pp"]
            >= thresholds["median_success_gain_pp_min"],
            "two_task_positive_count": positive_count
            >= thresholds["two_task_positive_count_required"],
            "positive_task_fraction": aggregate["positive_task_fraction"]
            >= thresholds["positive_task_fraction_min"],
            "median_query_loss_reduction": aggregate[
                "median_query_loss_reduction_fraction"
            ]
            >= thresholds["median_locked_action_loss_reduction_fraction_min"],
            "selection_drift": drift_is_diagnostic
            or aggregate["median_selection_drift_proxy"]
            <= thresholds["median_selection_drift_proxy_max"],
        }
        candidates.append(
            {
                "variant": variant,
                "trainable_parameters": parameter_counts[variant],
                "task_metrics": task_metrics,
                "aggregate": aggregate,
                "threshold_checks": checks,
                "screening_passed": all(checks.values()),
            }
        )
    passing = sorted(
        (candidate for candidate in candidates if candidate["screening_passed"]),
        key=lambda candidate: (candidate["trainable_parameters"], candidate["variant"]),
    )
    if passing and rank_stage == "mature_positive_control":
        selected = passing[0]["variant"]
        status = "mature_lora_positive_control_passed"
        rank16_scope = None
        rank16_authorized = False
    elif passing:
        selected = passing[0]["variant"]
        status = f"{rank_stage}_support_selected_pending_confirmation"
        rank16_scope = None
        rank16_authorized = False
    elif rank_stage == "rank8":
        ranking = grant.get("query_ranking")
        if (
            not isinstance(ranking, list)
            or not ranking
            or ranking[0].get("variant") not in variants
        ):
            raise GateZeroTargetSupportScreenError("grant query ranking is invalid")
        selected = None
        status = "rank8_support_screen_failed_rank16_authorized"
        rank16_scope = ranking[0]["variant"]
        rank16_authorized = True
    elif rank_stage == "rank16":
        selected = None
        status = "rank16_support_screen_failed"
        rank16_scope = None
        rank16_authorized = False
    else:
        selected = None
        status = "mature_lora_positive_control_failed_bounded_recovery_required"
        rank16_scope = None
        rank16_authorized = False
    mature_pass = rank_stage == "mature_positive_control" and selected is not None
    return {
        "status": status,
        "candidates": candidates,
        "selected_variant": selected,
        "confirmation_authorized": selected is not None and not mature_pass,
        "rank16_authorized": rank16_authorized,
        "rank16_scope": rank16_scope,
        "gate_zero_authorized": mature_pass,
        "writer_authorized": mature_pass,
        "final_writer_target_contract_sealed": mature_pass,
    }


def _collect_fit_evidence(
    *,
    spec: dict[str, Any],
    config_path: Path,
    parent_path: Path,
    phase0_path: Path,
    competence_path: Path,
    fit_outputs: Mapping[tuple[str, int], Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = {
        (variant, task_id)
        for variant in spec["variants"]
        for task_id in spec["task_ids"]
    }
    if set(fit_outputs) != required:
        raise GateZeroTargetSupportScreenError(
            "screening grant requires the exact frozen fit-output set"
        )
    expected_authorities = {
        "execution_contract_sha256": sha256_file(config_path),
        "gate_zero_contract_sha256": sha256_file(parent_path),
        "phase0_contract_sha256": sha256_file(phase0_path),
        "source_competence_contract_sha256": sha256_file(competence_path),
    }
    evidence = {
        f"{variant}:task{task_id}": _support_fit_evidence(
            Path(fit_outputs[(variant, task_id)]),
            variant=variant,
            task_id=task_id,
            variant_spec=spec["fit"][variant],
            expected_authorities=expected_authorities,
        )
        for variant, task_id in sorted(required)
    }
    ranking = rank_query_supports(
        evidence=evidence,
        variants=spec["variants"],
        task_ids=spec["task_ids"],
        parameter_counts={
            variant: spec["fit"][variant]["expected_trainable_parameters"]
            for variant in spec["variants"]
        },
        drift_proxy_max=spec["selection"]["drift_proxy_max"],
    )
    return evidence, ranking


def create_support_screening_grant(
    *,
    config_path: Path,
    parent_path: Path,
    phase0_path: Path,
    competence_path: Path,
    fit_outputs: Mapping[tuple[str, int], Path],
    grant_path: Path,
) -> dict[str, Any]:
    """Freeze every task/support state before source rollout screening."""

    spec = load_target_support_screen_spec(
        config_path,
        gate_zero_path=parent_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
        prior_execution_path=config_path.with_name("gate_zero_oracle_execution.toml"),
    )
    if not grant_path.is_absolute() or grant_path.exists() or grant_path.parent.exists():
        raise GateZeroTargetSupportScreenError(
            "screening grant output must be a fresh absolute directory"
        )
    evidence, ranking = _collect_fit_evidence(
        spec=spec,
        config_path=config_path,
        parent_path=parent_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
        fit_outputs=fit_outputs,
    )
    grant = {
        "schema_version": 1,
        "status": GRANT_STATUS,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "target_support_contract_sha256": sha256_file(config_path),
        "gate_zero_contract_sha256": sha256_file(parent_path),
        "phase0_contract_sha256": sha256_file(phase0_path),
        "source_competence_contract_sha256": sha256_file(competence_path),
        "task_ids": spec["task_ids"],
        "variants": spec["variants"],
        "fit_evidence": evidence,
        "query_ranking": ranking,
        "screening_rollout_authorized": True,
        "screening_init_state_indices": spec["screening_rollout"][
            "init_state_indices"
        ],
        "screening_conditions": spec["screening_rollout"]["conditions"],
        "locked_report_access_authorized": False,
        "rank16_authorized": False,
        "gate_zero_authorized": False,
        "writer_authorized": False,
        "final_writer_target_contract_sealed": False,
        "validation_numeric_access": False,
        "held_numeric_access": False,
    }
    if spec.get("screening_stage") in {"rank16", "mature_positive_control"}:
        grant["screening_stage"] = spec["screening_stage"]
    grant_path.parent.mkdir(parents=True, exist_ok=False)
    atomic_json(grant_path, grant)
    (grant_path.parent / "checksums.sha256").write_text(
        f"{sha256_file(grant_path)}  {grant_path.name}\n", encoding="utf-8"
    )
    return grant


def validate_support_screening_grant(
    *,
    grant_path: Path,
    config_path: Path,
    parent_path: Path,
    phase0_path: Path,
    competence_path: Path,
    fit_outputs: Mapping[tuple[str, int], Path],
) -> dict[str, Any]:
    """Revalidate the grant, selected states, and query ranking before rollout."""

    try:
        grant = json.loads(grant_path.read_text(encoding="utf-8"))
        checksum = (grant_path.parent / "checksums.sha256").read_text(
            encoding="utf-8"
        ).split()
    except (OSError, json.JSONDecodeError) as error:
        raise GateZeroTargetSupportScreenError("invalid screening grant") from error
    spec = load_target_support_screen_spec(
        config_path,
        gate_zero_path=parent_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
        prior_execution_path=config_path.with_name("gate_zero_oracle_execution.toml"),
    )
    expected = {
        "schema_version": 1,
        "status": GRANT_STATUS,
        "target_support_contract_sha256": sha256_file(config_path),
        "gate_zero_contract_sha256": sha256_file(parent_path),
        "phase0_contract_sha256": sha256_file(phase0_path),
        "source_competence_contract_sha256": sha256_file(competence_path),
        "task_ids": spec["task_ids"],
        "variants": spec["variants"],
        "screening_rollout_authorized": True,
        "screening_init_state_indices": spec["screening_rollout"][
            "init_state_indices"
        ],
        "screening_conditions": spec["screening_rollout"]["conditions"],
        "locked_report_access_authorized": False,
        "rank16_authorized": False,
        "gate_zero_authorized": False,
        "writer_authorized": False,
        "final_writer_target_contract_sealed": False,
        "validation_numeric_access": False,
        "held_numeric_access": False,
    }
    if spec.get("screening_stage") in {"rank16", "mature_positive_control"}:
        expected["screening_stage"] = spec["screening_stage"]
    for key, value in expected.items():
        if grant.get(key) != value:
            raise GateZeroTargetSupportScreenError(f"screening grant changed {key}")
    evidence, ranking = _collect_fit_evidence(
        spec=spec,
        config_path=config_path,
        parent_path=parent_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
        fit_outputs=fit_outputs,
    )
    if grant.get("fit_evidence") != evidence or grant.get("query_ranking") != ranking:
        raise GateZeroTargetSupportScreenError("screening grant fit evidence changed")
    if checksum != [sha256_file(grant_path), grant_path.name]:
        raise GateZeroTargetSupportScreenError("screening grant checksum changed")
    return grant


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate-zero-contract", type=Path, required=True)
    parser.add_argument("--phase0-contract", type=Path, required=True)
    parser.add_argument("--source-competence-contract", type=Path, required=True)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--grant-path", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for name in (
        "config",
        "gate_zero_contract",
        "phase0_contract",
        "source_competence_contract",
        "fit_root",
        "grant_path",
    ):
        setattr(args, name, getattr(args, name).absolute())
    spec = load_target_support_screen_spec(
        args.config,
        gate_zero_path=args.gate_zero_contract,
        phase0_path=args.phase0_contract,
        competence_path=args.source_competence_contract,
        prior_execution_path=args.config.with_name("gate_zero_oracle_execution.toml"),
    )
    outputs = {
        (variant, task_id): args.fit_root / f"{variant}_task{task_id}"
        for variant in spec["variants"]
        for task_id in spec["task_ids"]
    }
    grant = create_support_screening_grant(
        config_path=args.config,
        parent_path=args.gate_zero_contract,
        phase0_path=args.phase0_contract,
        competence_path=args.source_competence_contract,
        fit_outputs=outputs,
        grant_path=args.grant_path,
    )
    print(json.dumps(grant, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
