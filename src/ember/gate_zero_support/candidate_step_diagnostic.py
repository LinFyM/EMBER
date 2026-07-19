"""Evaluate earlier states on the failed Gate-0 LoRA trajectory."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
import tomllib
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

from ember.eval_artifacts import build_eval_gallery, update_latest_link
from ember.evaluation_identity import _load_policy
from ember.gate_zero_checkpoint import CHECKPOINT_MANIFEST, validate_source_base_checkpoint
from ember.gate_zero_contract import load_gate_zero_contract
from ember.gate_zero_oracle_artifacts import (
    atomic_json,
    restore_trainable_state,
    sha256_file,
    validate_candidate_artifact,
    write_output_checksums,
)
from ember.gate_zero_oracle_report_runtime import _closed_loop_metrics, _task_authority
from ember.gate_zero_oracle_session import configure_oracle_variant
from ember.gate_zero_support.mature_lora_lr_contract import (
    load_mature_lora_lr_recovery_spec,
)
from ember.gate_zero_support.screen_runtime import (
    ParallelContext,
    _broadcast,
    _close_parallel,
    _gather,
    _initialize_parallel,
)


EXPECTED_NAME = "smolvla_libero90_gate_zero_mature_lora_candidate_step_diagnostic_v1"
EXPECTED_STATUS = (
    "predeclared_after_failed_proposal_a_before_candidate_step_closed_loop_outcomes"
)
RESULT_NAME = "candidate_step_diagnostic_result.json"


class GateZeroCandidateStepDiagnosticError(RuntimeError):
    """Raised when the post-failure diagnostic changes its frozen contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_fields(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key, value in expected.items():
        if actual.get(key) != value:
            raise GateZeroCandidateStepDiagnosticError(f"{label} {key} changed")


def validate_candidate_step_diagnostic_spec(
    spec: dict[str, Any],
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    fit_path: Path,
    proposal_a_path: Path,
) -> None:
    """Fail closed on tasks, states, old evidence, and recovery thresholds."""
    _require_fields(spec, {
        "schema_version": 1,
        "name": EXPECTED_NAME,
        "status": EXPECTED_STATUS,
        "task_ids": [3, 4],
        "candidate_steps": [500, 750],
        "variant": "mature_official_default_r32_lr25e6_recovery",
    }, "root")
    authority = spec.get("authority", {})
    _require_fields(authority, {
        "gate_zero_contract_sha256": _sha256(gate_zero_path),
        "phase0_contract_sha256": _sha256(phase0_path),
        "fit_contract_sha256": _sha256(fit_path),
        "proposal_a_contract_sha256": _sha256(proposal_a_path),
        "validation_numeric_access": False,
        "held_numeric_access": False,
        "locked_report_numeric_access": False,
        "continuation_past_step1000": False,
    }, "authority")
    _require_fields(spec.get("proposal_a_failure", {}), {
        "role": "immutable_failed_reference_not_a_threshold_source",
        "task3_base_successes": 3,
        "task3_step1000_successes": 2,
        "task4_base_successes": 3,
        "task4_step1000_successes": 4,
        "aggregate_paired_net_wins": 0,
        "gate_zero_authorized": False,
        "writer_authorized": False,
    }, "Proposal-A failure")
    _require_fields(spec.get("diagnostic_rollout", {}), {
        "reuse_proposal_a_base_arms_without_rerun": True,
        "reuse_proposal_a_step1000_arms_without_rerun": True,
        "evaluate_only_candidate_steps": [500, 750],
        "init_state_indices": list(range(40, 48)),
        "batch_size": 8,
        "seed_start": 5800,
        "warmup_seed_start": 5792,
        "policy_rng_seed": 2026071836,
        "conditions": ["step500", "step750"],
    }, "diagnostic rollout")
    frozen_rule = {
        "minimum_each_task_query_reduction_fraction": 0.02,
        "maximum_each_task_selection_drift_proxy": 0.02,
        "minimum_positive_task_count": 2,
        "minimum_median_success_gain_pp": 15.0,
    }
    _require_fields(spec.get("decision", {}), {
        **frozen_rule,
        "minimum_each_task_success_gain_exclusive_pp": 0.0,
        "selection_rule": "max_aggregate_paired_net_wins_then_earliest_step",
        "selected_step_must_pass_every_rule": True,
        "diagnostic_cannot_authorize_gate_zero_or_writer": True,
    }, "decision")
    _require_fields(spec.get("fresh_recovery_gate", {}), {
        "task_ids": [3, 4],
        "init_state_indices": list(range(40, 48)),
        "seed_start": 6000,
        **frozen_rule,
        "selected_step_and_state_hash_grant_required_before_rollout": True,
        "selection_changes_after_grant_forbidden": True,
    }, "fresh recovery Gate")
    _require_fields(spec.get("lora", {}), {
        "target_count": 37,
        "rank": 32,
        "alpha": 16,
        "dropout": 0.0,
        "trainable_parameters": 1_485_312,
    }, "LoRA")
    _require_fields(spec.get("resources", {}), {
        "maximum_concurrent_gpus": 4,
        "diagnostic_gpus": 2,
        "minimum_free_memory_mib": 10_240,
        "maximum_new_rollout_episodes": 32,
    }, "resources")


def load_candidate_step_diagnostic_spec(
    path: Path,
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    fit_path: Path,
    proposal_a_path: Path,
) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            spec = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroCandidateStepDiagnosticError("invalid candidate-step TOML") from error
    validate_candidate_step_diagnostic_spec(
        spec,
        gate_zero_path=gate_zero_path,
        phase0_path=phase0_path,
        fit_path=fit_path,
        proposal_a_path=proposal_a_path,
    )
    return spec


def assigned_candidate_step_arms(
    *, rank: int, world_size: int, spec: Mapping[str, Any]
) -> list[tuple[int, int]]:
    if world_size not in {1, 2} or rank < 0 or rank >= world_size:
        raise GateZeroCandidateStepDiagnosticError("invalid candidate-step topology")
    by_task = [
        [(task_id, step) for step in spec["candidate_steps"]]
        for task_id in spec["task_ids"]
    ]
    return [arm for rows in by_task for arm in rows] if world_size == 1 else by_task[rank]


def _validated_arms(
    spec: Mapping[str, Any], arms: list[dict[str, Any]]
) -> dict[tuple[int, int], dict[str, Any]]:
    rollout = spec["diagnostic_rollout"]
    expected = {
        (task_id, step) for task_id in spec["task_ids"] for step in spec["candidate_steps"]
    }
    expected_seeds = list(range(rollout["seed_start"], rollout["seed_start"] + 8))
    by_key = {}
    for arm in arms:
        key = (arm.get("task_id"), arm.get("candidate_step"))
        successes = arm.get("successes")
        valid = (
            key in expected
            and key not in by_key
            and arm.get("condition") == f"step{key[1]}"
            and arm.get("mechanics_valid") is True
            and arm.get("official_rollout_init_state_indices") == rollout["init_state_indices"]
            and arm.get("seeds") == expected_seeds
            and isinstance(successes, list)
            and len(successes) == 8
            and all(isinstance(value, bool) for value in successes)
        )
        if not valid:
            raise GateZeroCandidateStepDiagnosticError("invalid candidate-step arm")
        by_key[key] = arm
    if set(by_key) != expected:
        raise GateZeroCandidateStepDiagnosticError("candidate-step arms are incomplete")
    return by_key


def decide_candidate_step_diagnostic(
    spec: Mapping[str, Any], arms: list[dict[str, Any]]
) -> dict[str, Any]:
    """Apply the original two-task positive-improvement rule."""

    by_key = _validated_arms(spec, arms)
    threshold = spec["decision"]
    evidence = spec["candidate_evidence"]
    base = {3: 3, 4: 3}
    candidates = []
    for index, step in enumerate(spec["candidate_steps"]):
        metrics, gains, reductions, drifts, total = {}, [], [], [], 0
        for task_id in spec["task_ids"]:
            successes = sum(by_key[(task_id, step)]["successes"])
            net = successes - base[task_id]
            gain = 12.5 * net
            reduction = evidence[f"task{task_id}_query_reduction_fraction"][index]
            drift = evidence[f"task{task_id}_selection_drift_proxy"][index]
            gains.append(gain)
            reductions.append(reduction)
            drifts.append(drift)
            total += net
            metrics[str(task_id)] = {
                "base_successes": base[task_id],
                "candidate_successes": successes,
                "paired_net_wins": net,
                "success_gain_pp": gain,
                "query_loss_reduction_fraction": reduction,
                "selection_drift_proxy": drift,
            }
        positive = sum(value > 0 for value in gains)
        median_gain = statistics.median(gains)
        checks = {
            "each_query_reduction": min(reductions)
            >= threshold["minimum_each_task_query_reduction_fraction"],
            "each_selection_drift": max(drifts)
            <= threshold["maximum_each_task_selection_drift_proxy"],
            "positive_task_count": positive >= threshold["minimum_positive_task_count"],
            "median_success_gain": median_gain >= threshold["minimum_median_success_gain_pp"],
        }
        candidates.append(
            {
                "step": step,
                "task_metrics": metrics,
                "aggregate": {
                    "paired_net_wins": total,
                    "positive_task_count": positive,
                    "median_success_gain_pp": median_gain,
                },
                "threshold_checks": checks,
                "diagnostic_passed": all(checks.values()),
            }
        )
    passing = sorted(
        (row for row in candidates if row["diagnostic_passed"]),
        key=lambda row: (-row["aggregate"]["paired_net_wins"], row["step"]),
    )
    selected_step = passing[0]["step"] if passing else None
    selected_index = spec["candidate_steps"].index(selected_step) if passing else None
    return {
        "status": (
            "earlier_candidate_step_selected_for_fresh_recovery_gate"
            if passing
            else "candidate_step_magnitude_recovery_not_supported"
        ),
        "failure_class_tested": threshold["failure_class_tested"],
        "candidates": candidates,
        "selected_step": selected_step,
        "selected_state_sha256_by_task": (
            {
                str(task_id): evidence[f"task{task_id}_state_sha256"][selected_index]
                for task_id in spec["task_ids"]
            }
            if selected_index is not None
            else None
        ),
        "fresh_recovery_gate_grant_authorized": bool(passing),
        "gate_zero_authorized": False,
        "writer_authorized": False,
        "validation_numeric_access": False,
        "held_numeric_access": False,
    }


def _proposal_a_reference(path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    if _sha256(path) != spec["authority"]["proposal_a_result_sha256"]:
        raise GateZeroCandidateStepDiagnosticError("Proposal-A result hash changed")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("status") != "mature_lora_headroom_control_failed_gate_recovery_required"
        or result.get("gate_zero_authorized") is not False
        or result.get("writer_authorized") is not False
    ):
        raise GateZeroCandidateStepDiagnosticError("Proposal-A failure boundary changed")
    counts = {
        (arm["task_id"], arm["condition"]): sum(arm["successes"])
        for arm in result.get("arms", [])
    }
    expected = {
        (3, "frozen_base"): 3,
        (4, "frozen_base"): 3,
        (3, spec["variant"]): 2,
        (4, spec["variant"]): 4,
    }
    if counts != expected:
        raise GateZeroCandidateStepDiagnosticError("Proposal-A arm counts changed")
    return result


def _candidate_state(
    spec: Mapping[str, Any], *, fit_root: Path, task_id: int, step: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    index = spec["candidate_steps"].index(step)
    task_root = fit_root / f"{spec['variant']}_task{task_id}"
    output = task_root / "candidates" / f"{step:06d}"
    manifest = validate_candidate_artifact(
        output, expected={"variant": spec["variant"], "task_id": task_id, "step": step}
    )
    evidence = spec["candidate_evidence"]
    if (
        _sha256(output / "candidate_manifest.json")
        != evidence[f"task{task_id}_manifest_sha256"][index]
        or _sha256(output / "trainable_state.safetensors")
        != evidence[f"task{task_id}_state_sha256"][index]
    ):
        raise GateZeroCandidateStepDiagnosticError("candidate artifact hash changed")
    zero = validate_candidate_artifact(
        task_root / "candidates" / "000000",
        expected={"variant": spec["variant"], "task_id": task_id, "step": 0},
    )
    base = float(zero["metrics"]["query_flow_mse"])
    metrics = manifest["metrics"]
    reduction = (base - float(metrics["query_flow_mse"])) / base
    if (
        reduction != evidence[f"task{task_id}_query_reduction_fraction"][index]
        or float(metrics["action_drift_proxy"])
        != evidence[f"task{task_id}_selection_drift_proxy"][index]
    ):
        raise GateZeroCandidateStepDiagnosticError("candidate query evidence changed")
    return load_file(output / "trainable_state.safetensors"), manifest


def _rollout_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    rollout = spec["diagnostic_rollout"]
    return {
        "report": {
            "rollout_batch_size": 8,
            "official_rollout_init_state_indices": rollout["init_state_indices"],
            "seed_start": rollout["seed_start"],
            "warmup_seed_start": rollout["warmup_seed_start"],
            "policy_rng_seed": rollout["policy_rng_seed"],
        },
        "resources": {
            "retain_one_video_per_report_arm": spec["resources"]["retain_one_video_per_arm"]
        },
    }


def _eval_info(arms: list[dict[str, Any]], decision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "overall": {
            "surface": "source_only_post_failure_candidate_step_closed_loop_diagnostic",
            "status": decision["status"],
            "episodes": sum(len(arm["successes"]) for arm in arms),
            "selected_step": decision["selected_step"],
            "gate_zero_authorized": False,
            "writer_authorized": False,
        },
        "per_task": [
            {
                "task_group": f"libero_90:step{arm['candidate_step']}",
                "task_id": arm["task_id"],
                "metrics": {
                    key: arm[key]
                    for key in ("successes", "sum_rewards", "max_rewards", "video_paths")
                },
            }
            for arm in arms
        ],
    }


def _prepare_tracker(
    context: ParallelContext, spec: Mapping[str, Any], args: argparse.Namespace
) -> Any:
    error, tracker = None, None
    if context.is_primary:
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            unexpected = [
                path.name
                for path in args.output_dir.iterdir()
                if not (path.name.startswith("gpu_telemetry_") and path.suffix == ".csv")
            ]
            if unexpected:
                raise GateZeroCandidateStepDiagnosticError(
                    f"refusing non-fresh diagnostic output: {unexpected}"
                )
            import trackio

            trackio.init(
                project=spec["resources"]["tracking_project"],
                name=args.output_dir.name,
                group=spec["resources"]["tracking_group"],
                config={"world_size": context.world_size, "candidate_steps": [500, 750]},
                auto_log_gpu=True,
                gpu_log_interval=1.0,
                auto_log_cpu=True,
                cpu_log_interval=1.0,
            )
            tracker = trackio
        except BaseException as caught:
            error = f"{type(caught).__name__}: {caught}"
    error = _broadcast(context, error)
    if error:
        raise GateZeroCandidateStepDiagnosticError(error)
    return tracker


def _evaluate_local_arms(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    fit_spec: Mapping[str, Any],
    parent: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    context: ParallelContext,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arms, authorities = [], {}
    for task_id, step in assigned_candidate_step_arms(
        rank=context.rank, world_size=context.world_size, spec=spec
    ):
        language, authority = _task_authority(task_id, list(range(40, 48)))
        authorities[task_id] = authority
        runtime = list(
            _load_policy(
                args.source_base_checkpoint / "pretrained_model",
                {"task_suite": "libero_90", "task_id": task_id},
            )
        )
        model, summary = configure_oracle_variant(
            runtime[0],
            parent=parent,
            checkpoint=checkpoint,
            variant=spec["variant"],
            variant_spec=fit_spec["fit"][spec["variant"]],
        )
        state, manifest = _candidate_state(
            spec, fit_root=args.fit_root, task_id=task_id, step=step
        )
        restore_trainable_state(model, state)
        model.eval()
        runtime[0] = model
        arm = {
            "task_id": task_id,
            "condition": f"step{step}",
            "candidate_step": step,
            "state_authority": {
                "candidate_manifest_sha256": spec["candidate_evidence"][
                    f"task{task_id}_manifest_sha256"
                ][spec["candidate_steps"].index(step)],
                "trainable_state_sha256": manifest["files"][
                    "trainable_state.safetensors"
                ]["sha256"],
                "trainable_parameters": summary["trainable_parameters"],
            },
            **_closed_loop_metrics(
                runtime=tuple(runtime),
                task_id=task_id,
                condition=f"step{step}",
                language=language,
                spec=_rollout_spec(spec),
                output_dir=args.output_dir,
            ),
        }
        arms.append(arm)
        print(
            json.dumps(
                {
                    "event": "gate_zero_candidate_step_diagnostic_arm",
                    "rank": context.rank,
                    "task_id": task_id,
                    "step": step,
                    "successes": sum(arm["successes"]),
                    "episodes": 8,
                    "mechanics_valid": arm["mechanics_valid"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del runtime, model, state
        gc.collect()
        torch.cuda.empty_cache()
    return arms, list(authorities.values())


def _publish_result(
    *,
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    reference: Mapping[str, Any],
    fit_path: Path,
    gathered_arms: list[list[dict[str, Any]]],
    gathered_authorities: list[list[dict[str, Any]]],
    tracker: Any,
    context: ParallelContext,
    started: float,
) -> dict[str, Any]:
    arms = sorted(
        (arm for rows in gathered_arms for arm in rows),
        key=lambda arm: (arm["task_id"], arm["candidate_step"]),
    )
    decision = decide_candidate_step_diagnostic(spec, arms)
    authorities = {row["task_id"]: row for rows in gathered_authorities for row in rows}
    result = {
            "schema_version": 1,
            "status": decision["status"],
            "surface": spec["surface"],
            "diagnostic_contract_sha256": sha256_file(args.config),
            "fit_contract_sha256": sha256_file(fit_path),
            "proposal_a_result_sha256": sha256_file(args.proposal_a_result),
            "proposal_a_reference": {
                "status": reference["status"],
                **{
                    key: spec["proposal_a_failure"][key]
                    for key in (
                        "task3_base_successes",
                        "task3_step1000_successes",
                        "task4_base_successes",
                        "task4_step1000_successes",
                        "aggregate_paired_net_wins",
                    )
                },
            },
            "task_authorities": [authorities[task_id] for task_id in [3, 4]],
            "arms": arms,
            "decision": decision,
            "fresh_recovery_gate_contract": spec["fresh_recovery_gate"],
            "gate_zero_authorized": False,
            "writer_authorized": False,
            "validation_numeric_access": False,
            "held_numeric_access": False,
            "locked_report_numeric_access": False,
            "continuation_past_step1000": False,
            "resources": {
                "physical_gpus": args.physical_gpus,
                "gpu_count": context.world_size,
                "wall_seconds": time.perf_counter() - started,
            },
            "tracking": {
                "backend": "trackio",
                "project": spec["resources"]["tracking_project"],
                "run": args.output_dir.name,
                "dashboard_command": "trackio show --project EMBER_gate0",
            },
            "interpretation": spec["interpretation"],
    }
    atomic_json(args.output_dir / RESULT_NAME, result)
    atomic_json(args.output_dir / "eval_info.json", _eval_info(arms, decision))
    build_eval_gallery(args.output_dir)
    write_output_checksums(args.output_dir)
    update_latest_link(args.output_dir, args.latest_link)
    tracker.log(
        {
            "candidate_step/selected_step": decision["selected_step"] or 0,
            "candidate_step/fresh_gate_authorized": int(
                decision["fresh_recovery_gate_grant_authorized"]
            ),
        }
    )
    tracker.finish()
    return result


def run_candidate_step_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    fit_path = args.config.with_name("gate_zero_mature_lora_lr_recovery.toml")
    spec = load_candidate_step_diagnostic_spec(
        args.config,
        gate_zero_path=args.gate_zero_contract,
        phase0_path=args.phase0_contract,
        fit_path=fit_path,
        proposal_a_path=args.config.with_name("gate_zero_mature_lora_headroom_screen.toml"),
    )
    fit_spec = load_mature_lora_lr_recovery_spec(
        fit_path,
        gate_zero_path=args.gate_zero_contract,
        phase0_path=args.phase0_contract,
        competence_path=args.config.with_name("gate_zero_source_competence.toml"),
    )
    parent = load_gate_zero_contract(args.gate_zero_contract, args.phase0_contract)
    checkpoint = validate_source_base_checkpoint(
        args.source_base_checkpoint,
        expected={
            "step": spec["authority"]["source_base_checkpoint_step"],
            "checkpoint_role": spec["authority"]["source_base_checkpoint_role"],
        },
    )
    if _sha256(args.source_base_checkpoint / CHECKPOINT_MANIFEST) != spec["authority"][
        "source_base_checkpoint_manifest_sha256"
    ]:
        raise GateZeroCandidateStepDiagnosticError("source-base checkpoint changed")
    reference = _proposal_a_reference(args.proposal_a_result, spec)
    context = _initialize_parallel()
    tracker = None
    try:
        tracker = _prepare_tracker(context, spec, args)
        started = time.perf_counter()
        local_arms, local_authorities = _evaluate_local_arms(
            args, spec, fit_spec, parent, checkpoint, context
        )
        gathered_arms = _gather(context, local_arms)
        gathered_authorities = _gather(context, local_authorities)
        if not context.is_primary:
            return {"status": "non_primary_rank_complete", "rank": context.rank}
        result = _publish_result(
            args=args,
            spec=spec,
            reference=reference,
            fit_path=fit_path,
            gathered_arms=gathered_arms,
            gathered_authorities=gathered_authorities,
            tracker=tracker,
            context=context,
            started=started,
        )
        tracker = None
        return result
    finally:
        if tracker is not None:
            tracker.finish()
        _close_parallel(context)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config",
        "gate-zero-contract",
        "phase0-contract",
        "source-base-checkpoint",
        "fit-root",
        "proposal-a-result",
        "output-dir",
        "latest-link",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--physical-gpus", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for name in (
        "config",
        "gate_zero_contract",
        "phase0_contract",
        "source_base_checkpoint",
        "fit_root",
        "proposal_a_result",
        "output_dir",
        "latest_link",
    ):
        setattr(args, name, getattr(args, name).absolute())
    try:
        result = run_candidate_step_diagnostic(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.output_dir / f"failure_packet_rank_{os.environ.get('RANK', '0')}.json",
            {
                "schema_version": 1,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "validation_numeric_access": False,
                "held_numeric_access": False,
                "continuation_past_step1000": False,
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
