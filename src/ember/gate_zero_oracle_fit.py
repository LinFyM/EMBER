"""Fit and query-select one frozen Gate 0 task-local oracle candidate set."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from ember.eval_artifacts import update_latest_link
from ember.gate_zero_oracle_artifacts import (
    atomic_json,
    candidate_evidence,
    cleanup_completed_fit_state,
    load_recovery_artifact,
    publish_selected_artifact,
    save_candidate_artifact,
    save_recovery_artifact,
    sha256_file,
    validate_candidate_artifact,
    validate_fit_output,
    validate_recovery_artifact,
    validate_selected_artifact,
    write_output_checksums,
)
from ember.gate_zero_oracle_contract import (
    oracle_fit_authorities,
    validate_oracle_fit_prerequisites,
)
from ember.gate_zero_oracle_metrics import (
    FixedQueryReference,
    evenly_spaced_anchor_indices,
    select_action_mse_candidate,
)
from ember.gate_zero_oracle_session import (
    OracleModelSession,
    build_oracle_optimizer,
    capture_trainable_state,
    close_loader,
    make_support_loader,
    open_oracle_model_session,
    train_oracle_step,
    validate_fit_job,
)


RESULT_NAME = "fit_selection_result.json"
CANDIDATE_RECORDS_NAME = "candidate_metrics.json"


class GateZeroOracleFitError(RuntimeError):
    """Raised when oracle fitting or query-only selection changes contract."""


def select_drift_safe_candidate(
    candidates: list[dict[str, Any]], *, drift_proxy_max: float
) -> dict[str, Any]:
    """Select minimum query MSE among candidates satisfying the frozen drift cap."""

    import math

    if not candidates or not math.isfinite(drift_proxy_max) or drift_proxy_max < 0:
        raise GateZeroOracleFitError("invalid query-selection candidates or drift cap")
    seen_steps: set[int] = set()
    safe: list[dict[str, Any]] = []
    for candidate in candidates:
        step = candidate.get("step")
        query = candidate.get("query_flow_mse")
        drift = candidate.get("action_drift_proxy")
        if (
            not isinstance(step, int)
            or isinstance(step, bool)
            or step < 0
            or step in seen_steps
            or not isinstance(query, (int, float))
            or not isinstance(drift, (int, float))
            or not math.isfinite(float(query))
            or not math.isfinite(float(drift))
            or query < 0
            or drift < 0
        ):
            raise GateZeroOracleFitError("invalid query-selection candidate")
        seen_steps.add(step)
        if drift <= drift_proxy_max:
            safe.append(candidate)
    if not safe:
        raise GateZeroOracleFitError("no drift-safe candidate")
    return min(safe, key=lambda value: (float(value["query_flow_mse"]), value["step"]))


def select_fixed_final_candidate(
    candidates: list[dict[str, Any]], *, final_step: int
) -> dict[str, Any]:
    """Select only the predeclared final step for a mature recipe control."""

    if not isinstance(final_step, int) or isinstance(final_step, bool) or final_step <= 0:
        raise GateZeroOracleFitError("invalid fixed final optimizer step")
    matching = [candidate for candidate in candidates if candidate.get("step") == final_step]
    if len(matching) != 1:
        raise GateZeroOracleFitError("fixed final optimizer-step candidate is missing or duplicated")
    selected = matching[0]
    for key in ("query_flow_mse", "action_drift_proxy"):
        value = selected.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise GateZeroOracleFitError("fixed final candidate metrics are invalid")
    return selected


def resolve_training_target_step(
    *,
    start_step: int,
    optimizer_steps: int,
    candidate_steps: list[int],
    stop_after_step: int | None,
) -> int:
    """Resolve one resumable segment without changing the scientific budget."""

    if stop_after_step is None:
        return optimizer_steps
    if (
        not isinstance(stop_after_step, int)
        or isinstance(stop_after_step, bool)
        or stop_after_step not in candidate_steps
        or stop_after_step <= start_step
        or stop_after_step >= optimizer_steps
    ):
        raise GateZeroOracleFitError(
            "staged stop must be a future predeclared non-final candidate"
        )
    return stop_after_step


def _reference_evidence(reference: FixedQueryReference) -> dict[str, Any]:
    evidence = {
        "base_query_flow_mse": reference.query_flow_mse,
        "query_sample_count": reference.query_sample_count,
        "query_row_keys_sha256": reference.query_row_keys_sha256,
        "anchor_count": reference.anchor_count,
        "anchor_row_keys_sha256": reference.anchor_row_keys_sha256,
    }
    if reference.action_error_mse_by_noise_seed:
        evidence["base_query_action_mse_by_noise_seed"] = (
            reference.action_error_mse_by_noise_seed
        )
        evidence["base_query_action_mse_mean"] = sum(
            reference.action_error_mse_by_noise_seed.values()
        ) / len(reference.action_error_mse_by_noise_seed)
    return evidence


def _existing_candidates(
    output_dir: Path,
    *,
    candidate_steps: list[int],
    start_step: int,
    variant: str,
    task_id: int,
    authorities: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    root = output_dir / "candidates"
    records: dict[int, dict[str, Any]] = {}
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if not path.is_dir() or not path.name.isdigit():
                raise GateZeroOracleFitError("candidate directory contains unknown state")
            manifest = validate_candidate_artifact(
                path,
                expected={"variant": variant, "task_id": task_id, "authorities": authorities},
            )
            step = int(manifest["step"])
            if step not in candidate_steps or step > start_step:
                raise GateZeroOracleFitError("candidate step differs from recovery authority")
            records[step] = manifest["metrics"]
    required_prior = {step for step in candidate_steps if step < start_step}
    if not required_prior <= set(records):
        raise GateZeroOracleFitError("recovery lacks prior candidate evidence")
    return records


def _evaluate_and_save_candidate(
    *,
    evaluator: FixedQueryEvaluator,
    model: Any,
    reference: FixedQueryReference,
    output_dir: Path,
    variant: str,
    task_id: int,
    step: int,
    support_record: dict[str, Any] | None,
    authorities: dict[str, Any],
) -> dict[str, Any]:
    metrics = evaluator.evaluate_candidate(model, reference, step=step)
    metrics["support_training_record"] = support_record
    candidate_dir = save_candidate_artifact(
        output_dir,
        variant=variant,
        task_id=task_id,
        step=step,
        trainable_state=capture_trainable_state(model),
        metrics=metrics,
        authorities=authorities,
    )
    validate_candidate_artifact(candidate_dir)
    return metrics


def _require_step_zero(
    metrics: dict[str, Any], reference: FixedQueryReference, *, variant: str
) -> None:
    if (
        metrics["query_flow_mse"] != reference.query_flow_mse
        or metrics["action_drift_proxy"] != 0.0
    ):
        raise GateZeroOracleFitError(f"{variant} step zero is not a functional base identity")
    if reference.action_error_mse_by_noise_seed is not None:
        if metrics.get("query_action_mse_by_noise_seed") != reference.action_error_mse_by_noise_seed:
            raise GateZeroOracleFitError(
                f"{variant} step zero action metric is not a functional base identity"
            )


def _initialize_tracker(
    args: argparse.Namespace, spec: dict[str, Any], session: OracleModelSession
) -> Any:
    import trackio

    trackio.init(
        project=spec["resources"]["tracking_project"],
        name=args.output_dir.name,
        group=spec["resources"]["tracking_group_fit"],
        config={
            "variant": args.variant,
            "task_id": args.task_id,
            "optimizer_steps": spec["fit"]["optimizer_steps"],
            "effective_batch_size": spec["fit"]["effective_batch_size"],
            "trainable_parameters": session.trainable_summary["trainable_parameters"],
            "physical_gpu": args.physical_gpu,
        },
        auto_log_gpu=True,
        gpu_log_interval=1.0,
        auto_log_cpu=True,
        cpu_log_interval=1.0,
    )
    return trackio


def _restore_training(
    args: argparse.Namespace,
    spec: dict[str, Any],
    variant_spec: dict[str, Any],
    authorities: dict[str, Any],
    session: OracleModelSession,
) -> tuple[Any, Any, int, dict[int, dict[str, Any]]]:
    expected = {
        "variant": args.variant,
        "task_id": args.task_id,
        "authorities": authorities,
    }
    start_step = 0
    recovery = None
    if args.resume:
        recovery = (args.output_dir / "recovery" / "last").resolve(strict=True)
        start_step = int(validate_recovery_artifact(recovery, expected=expected)["step"])
    loader = make_support_loader(
        session,
        fit=spec["fit"],
        seed=variant_spec["seed"],
        start_step=start_step,
    )
    iterator = iter(loader)
    if recovery is not None:
        restored = load_recovery_artifact(
            recovery,
            model=session.model,
            optimizer=session.optimizer,
            scheduler=session.scheduler,
            expected=expected,
        )
        if restored != start_step:
            raise GateZeroOracleFitError("recovery step changed while loading")
    candidates = _existing_candidates(
        args.output_dir,
        candidate_steps=spec["fit"]["candidate_steps"],
        start_step=start_step,
        variant=args.variant,
        task_id=args.task_id,
        authorities=authorities,
    )
    return loader, iterator, start_step, candidates


def _ensure_start_candidate(
    args: argparse.Namespace,
    spec: dict[str, Any],
    authorities: dict[str, Any],
    session: OracleModelSession,
    start_step: int,
    candidates: dict[int, dict[str, Any]],
) -> None:
    if start_step not in spec["fit"]["candidate_steps"] or start_step in candidates:
        return
    if not args.resume:
        save_recovery_artifact(
            args.output_dir,
            variant=args.variant,
            task_id=args.task_id,
            step=start_step,
            trainable_state=capture_trainable_state(session.model),
            optimizer=session.optimizer,
            scheduler=session.scheduler,
            authorities=authorities,
        )
    candidates[start_step] = _evaluate_and_save_candidate(
        evaluator=session.evaluator,
        model=session.model,
        reference=session.reference,
        output_dir=args.output_dir,
        variant=args.variant,
        task_id=args.task_id,
        step=start_step,
        support_record=None,
        authorities=authorities,
    )
    if start_step == 0:
        _require_step_zero(candidates[0], session.reference, variant=args.variant)


def _log_progress(
    tracker: Any,
    args: argparse.Namespace,
    spec: dict[str, Any],
    step: int,
    record: dict[str, Any],
) -> None:
    tracked = {
        "fit/support_objective_loss": record["support_objective_loss"],
        "fit/gradient_norm": record["gradient_norm"],
        "fit/samples_per_second": record["samples_per_second"],
        "fit/learning_rate": record["learning_rate"],
    }
    if "support_flow_loss" in record:
        tracked["fit/support_flow_loss"] = record["support_flow_loss"]
    if "support_action_mse" in record:
        tracked["fit/support_action_mse"] = record["support_action_mse"]
    tracker.log(tracked, step=step)
    print(
        json.dumps(
            {
                "event": "gate_zero_oracle_progress",
                "variant": args.variant,
                "task_id": args.task_id,
                "step": step,
                "target_step": spec["fit"]["optimizer_steps"],
                "segment_target_step": args.stop_after_step
                or spec["fit"]["optimizer_steps"],
                **record,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _save_trained_candidate(
    args: argparse.Namespace,
    authorities: dict[str, Any],
    session: OracleModelSession,
    tracker: Any,
    step: int,
    record: dict[str, Any],
) -> dict[str, Any]:
    save_recovery_artifact(
        args.output_dir,
        variant=args.variant,
        task_id=args.task_id,
        step=step,
        trainable_state=capture_trainable_state(session.model),
        optimizer=session.optimizer,
        scheduler=session.scheduler,
        authorities=authorities,
    )
    metrics = _evaluate_and_save_candidate(
        evaluator=session.evaluator,
        model=session.model,
        reference=session.reference,
        output_dir=args.output_dir,
        variant=args.variant,
        task_id=args.task_id,
        step=step,
        support_record=record,
        authorities=authorities,
    )
    tracked = {
        "selection/query_flow_mse": metrics["query_flow_mse"],
        "selection/action_drift_proxy": metrics["action_drift_proxy"],
    }
    if "query_action_mse_mean" in metrics:
        tracked.update(
            {
                "selection/query_action_mse_mean": metrics[
                    "query_action_mse_mean"
                ],
                "selection/query_action_mse_reduction_fraction": metrics[
                    "query_action_mse_reduction_fraction"
                ],
            }
        )
    tracker.log(tracked, step=step)
    return metrics


def _train_to_budget(
    args: argparse.Namespace,
    spec: dict[str, Any],
    parent: dict[str, Any],
    variant_spec: dict[str, Any],
    authorities: dict[str, Any],
    session: OracleModelSession,
    tracker: Any,
    iterator: Any,
    start_step: int,
    target_step: int,
    candidates: dict[int, dict[str, Any]],
) -> None:
    candidate_steps = spec["fit"]["candidate_steps"]
    log_every = parent["tracking"]["log_every_optimizer_steps"]
    for step in range(start_step + 1, target_step + 1):
        record = train_oracle_step(
            iterator,
            session=session,
            gradient_clip_norm=variant_spec["gradient_clip_norm"],
            optimizer_step=step,
            variant_spec=variant_spec,
        )
        if step == 1 or step % log_every == 0 or step in candidate_steps:
            _log_progress(tracker, args, spec, step, record)
        if step in candidate_steps:
            candidates[step] = _save_trained_candidate(
                args, authorities, session, tracker, step, record
            )


def _build_stage_result(
    args: argparse.Namespace,
    spec: dict[str, Any],
    authorities: dict[str, Any],
    *,
    stage_step: int,
) -> dict[str, Any]:
    """Validate and summarize a resumable candidate boundary without selecting."""

    expected = {
        "variant": args.variant,
        "task_id": args.task_id,
        "authorities": authorities,
    }
    recovery_dir = (args.output_dir / "recovery" / "last").resolve(strict=True)
    recovery = validate_recovery_artifact(recovery_dir, expected=expected)
    if recovery["step"] != stage_step:
        raise GateZeroOracleFitError("staged recovery did not reach the stop boundary")
    candidate = candidate_evidence(
        args.output_dir / "candidates" / f"{stage_step:06d}"
    )
    if candidate["step"] != stage_step:
        raise GateZeroOracleFitError("staged query candidate changed step")
    base = float(candidate["base_query_flow_mse"])
    query = float(candidate["query_flow_mse"])
    result = {
        "schema_version": 1,
        "status": "oracle_fit_stage_complete_resumable",
        "variant": args.variant,
        "task_id": args.task_id,
        "stage_step": stage_step,
        "maximum_optimizer_steps": spec["fit"]["optimizer_steps"],
        "query_loss_reduction_fraction": (base - query) / base,
        "candidate": candidate,
        "recovery": {
            "step": recovery["step"],
            "manifest_sha256": sha256_file(recovery_dir / "recovery_manifest.json"),
            "files": recovery["files"],
        },
        "selection_frozen": False,
        "final_closed_loop_accessed": False,
        "validation_numeric_access": False,
        "held_numeric_access": False,
        "continuation_requires_stage_contract": True,
    }
    if "query_action_mse_reduction_fraction" in candidate:
        result.update(
            {
                "primary_stage_metric": "generated_action_query_mse",
                "query_action_mse_reduction_fraction": candidate[
                    "query_action_mse_reduction_fraction"
                ],
                "query_action_mse_mean": candidate["query_action_mse_mean"],
                "base_query_action_mse_mean": candidate[
                    "base_query_action_mse_mean"
                ],
            }
        )
    return result


def _build_result(
    args: argparse.Namespace,
    spec: dict[str, Any],
    authorities: dict[str, Any],
    session: OracleModelSession,
    candidate_records: list[dict[str, Any]],
    selected_dir: Path,
    selected: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    selection = spec["selection"]
    variant_spec = spec["fit"][args.variant]
    if spec.get("screening_stage") == "action_aligned_lora_acquisition_recovery":
        capacity_role = "action_aligned_task_local_lora_acquisition_recovery"
        pilot_scope = "source_only_gate_zero_action_aligned_recovery_pending_closed_loop"
    elif spec.get("screening_stage") == "mature_positive_control":
        capacity_role = "mature_recipe_task_local_lora_positive_control"
        pilot_scope = (
            "source_only_mature_lora_competence_control_"
            "pending_fresh_closed_loop"
        )
    elif variant_spec.get("adaptation_kind") == "lora":
        capacity_role = "matched_target_support_audit_candidate"
        pilot_scope = (
            "source_only_gate_zero_target_support_audit_"
            "not_final_writer_target_support"
        )
    elif args.variant == "lora":
        capacity_role = "matched_primary_lora_pilot"
        pilot_scope = "source_only_gate_zero_pilot_not_final_writer_target_support"
    else:
        capacity_role = "non_matched_partial_update_upper_bound_only"
        pilot_scope = "source_only_gate_zero_pilot_not_final_writer_target_support"
    return {
        "schema_version": 1,
        "status": "oracle_fit_selection_complete_pending_global_report_grant",
        "variant": args.variant,
        "task_id": args.task_id,
        "pilot_scope": pilot_scope,
        "authorities": authorities,
        "task_authorities": session.task_authorities,
        "support": {
            "episode_bounds": spec["fit"]["support_episode_bounds"],
            "frame_count": len(session.support_dataset),
            "optimizer_steps": spec["fit"]["optimizer_steps"],
            "effective_batch_size": spec["fit"]["effective_batch_size"],
        },
        "query_reference": _reference_evidence(session.reference),
        "candidates": candidate_records,
        "selection": {
            "rule": selection["candidate_rule"],
            "drift_proxy_max": selection["drift_proxy_max"],
            "selected_step": selected["selected_step"],
            "selected_metrics": selected["selected_metrics"],
            "selected_trainable_state_sha256": selected["trainable_state_sha256"],
            "selected_manifest_sha256": sha256_file(
                selected_dir / "selected_manifest.json"
            ),
            "locked_report_accessed": False,
        },
        "trainable": session.trainable_summary,
        "capacity_role": capacity_role,
        "gate_zero_authorized": False,
        "writer_authorized": False,
        "final_writer_target_contract_sealed": False,
        "resources": {
            "physical_gpu": args.physical_gpu,
            "gpu_count": 1,
            "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() // (1024 * 1024),
            "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() // (1024 * 1024),
            "wall_seconds": time.perf_counter() - started,
        },
        "tracking": {
            "backend": "trackio",
            "project": spec["resources"]["tracking_project"],
            "run": args.output_dir.name,
            "dashboard_command": "trackio show --project EMBER_gate0",
        },
    }


def _finalize_fit(
    args: argparse.Namespace,
    spec: dict[str, Any],
    authorities: dict[str, Any],
    session: OracleModelSession,
    candidates: dict[int, dict[str, Any]],
    tracker: Any,
    started: float,
) -> dict[str, Any]:
    steps = spec["fit"]["candidate_steps"]
    if sorted(candidates) != steps:
        raise GateZeroOracleFitError("completed fit lacks every predeclared candidate")
    candidate_metrics = [candidates[step] for step in steps]
    if spec["selection"]["candidate_rule"] == "fixed_final_optimizer_step":
        selected_metrics = select_fixed_final_candidate(
            candidate_metrics,
            final_step=spec["selection"]["fixed_final_optimizer_step"],
        )
    elif spec["selection"]["candidate_rule"] == "minimum_mean_generated_action_mse_with_drift_cap":
        selected_metrics = select_action_mse_candidate(
            candidate_metrics,
            drift_proxy_max=spec["selection"]["drift_proxy_max"],
        )
    else:
        selected_metrics = select_drift_safe_candidate(
            candidate_metrics,
            drift_proxy_max=spec["selection"]["drift_proxy_max"],
        )
    records = [
        candidate_evidence(args.output_dir / "candidates" / f"{step:06d}")
        for step in steps
    ]
    atomic_json(args.output_dir / CANDIDATE_RECORDS_NAME, records)
    selected_dir = publish_selected_artifact(
        args.output_dir,
        args.output_dir / "candidates" / f"{selected_metrics['step']:06d}",
    )
    selected = validate_selected_artifact(
        selected_dir, expected={"variant": args.variant, "task_id": args.task_id}
    )
    result = _build_result(
        args, spec, authorities, session, records, selected_dir, selected, started
    )
    atomic_json(args.output_dir / RESULT_NAME, result)
    cleanup_completed_fit_state(args.output_dir, variant=args.variant)
    write_output_checksums(args.output_dir)
    update_latest_link(args.output_dir, args.latest_link)
    tracked = {
            "selection/selected_step": selected["selected_step"],
            "selection/selected_query_flow_mse": selected["selected_metrics"][
                "query_flow_mse"
            ],
            "selection/selected_action_drift_proxy": selected["selected_metrics"][
                "action_drift_proxy"
            ],
            "selection/complete": 1,
    }
    if "query_action_mse_mean" in selected["selected_metrics"]:
        tracked["selection/selected_query_action_mse_mean"] = selected[
            "selected_metrics"
        ]["query_action_mse_mean"]
    tracker.log(tracked, step=spec["fit"]["optimizer_steps"])
    return result


def run_oracle_fit(args: argparse.Namespace) -> dict[str, Any]:
    validate_fit_output(args.output_dir, result_name=RESULT_NAME, resume=args.resume)
    started = time.perf_counter()
    spec, parent, phase0, checkpoint = validate_oracle_fit_prerequisites(
        config_path=args.config,
        gate_zero_path=args.gate_zero_contract,
        phase0_path=args.phase0_contract,
        competence_path=args.source_competence_contract,
        competence_result_path=args.source_competence_result,
        source_base_checkpoint=args.source_base_checkpoint,
    )
    variant_spec = validate_fit_job(spec, variant=args.variant, task_id=args.task_id)
    authorities = oracle_fit_authorities(
        config_path=args.config,
        gate_zero_path=args.gate_zero_contract,
        phase0_path=args.phase0_contract,
        competence_path=args.source_competence_contract,
        competence_result_path=args.source_competence_result,
        source_base_checkpoint=args.source_base_checkpoint,
        manifest_path=args.manifest,
        spec=spec,
        parent=parent,
    )
    session = loader = tracker = None
    try:
        session = open_oracle_model_session(
            spec=spec,
            parent=parent,
            phase0=phase0,
            checkpoint=checkpoint,
            manifest=args.manifest,
            dataset_root=args.dataset_root,
            source_base_checkpoint=args.source_base_checkpoint,
            variant=args.variant,
            task_id=args.task_id,
            variant_spec=variant_spec,
        )
        tracker = _initialize_tracker(args, spec, session)
        loader, iterator, start_step, candidates = _restore_training(
            args, spec, variant_spec, authorities, session
        )
        target_step = resolve_training_target_step(
            start_step=start_step,
            optimizer_steps=spec["fit"]["optimizer_steps"],
            candidate_steps=spec["fit"]["candidate_steps"],
            stop_after_step=args.stop_after_step,
        )
        _ensure_start_candidate(
            args, spec, authorities, session, start_step, candidates
        )
        _train_to_budget(
            args,
            spec,
            parent,
            variant_spec,
            authorities,
            session,
            tracker,
            iterator,
            start_step,
            target_step,
            candidates,
        )
        if target_step < spec["fit"]["optimizer_steps"]:
            result = _build_stage_result(
                args, spec, authorities, stage_step=target_step
            )
            tracker.log(
                {
                    "stage/complete": 1,
                    "stage/query_loss_reduction_fraction": result[
                        "query_loss_reduction_fraction"
                    ],
                },
                step=target_step,
            )
            tracker.finish()
            tracker = None
            return result
        result = _finalize_fit(
            args, spec, authorities, session, candidates, tracker, started
        )
        tracker.finish()
        tracker = None
        return result
    finally:
        if tracker is not None:
            tracker.finish()
        close_loader(loader)
        if session is not None:
            session.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate-zero-contract", type=Path, required=True)
    parser.add_argument("--phase0-contract", type=Path, required=True)
    parser.add_argument("--source-competence-contract", type=Path, required=True)
    parser.add_argument("--source-competence-result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latest-link", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--physical-gpu", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-step", type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for name in (
        "config",
        "gate_zero_contract",
        "phase0_contract",
        "source_competence_contract",
        "source_competence_result",
        "manifest",
        "dataset_root",
        "source_base_checkpoint",
        "output_dir",
        "latest_link",
    ):
        setattr(args, name, getattr(args, name).absolute())
    result = run_oracle_fit(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
