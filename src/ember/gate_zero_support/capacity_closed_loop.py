"""Evaluate the frozen non-matched action-expert capacity upper bound."""

from __future__ import annotations

import argparse
import gc
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
from ember.gate_zero_support.mature_contract import load_mature_lora_positive_control_spec
from ember.gate_zero_support.screen_runtime import (
    ParallelContext,
    _broadcast,
    _close_parallel,
    _gather,
    _initialize_parallel,
)


EXPECTED_NAME = "smolvla_libero90_gate_zero_action_expert_capacity_closed_loop_v1"
EXPECTED_STATUS = (
    "predeclared_after_signed_ratio_failure_before_action_expert_closed_loop_outcomes"
)
EXPECTED_VARIANT = "mature_action_expert_lr25e6_recovery"
EXPECTED_PROPOSAL_SHA = "84116faaffd5115a72f4d49efa2f2467445ca0ac61edac265e571a1e8564c98f"
EXPECTED_SIGNED_SHA = "73d681caf4f5d6b67519eb33636e9af905aec7412c18c5d85cd1aaf8d3488703"
RESULT_NAME = "capacity_closed_loop_result.json"


class GateZeroCapacityClosedLoopError(RuntimeError):
    """Raised when the capacity diagnostic escapes its frozen authority."""


def _require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GateZeroCapacityClosedLoopError(f"{label} changed")


def validate_capacity_closed_loop_spec(
    spec: Mapping[str, Any],
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
    fit_path: Path,
) -> None:
    """Fail closed on all artifacts, rollout identities, and interpretation."""

    for key, expected in (
        ("schema_version", 1),
        ("name", EXPECTED_NAME),
        ("status", EXPECTED_STATUS),
        ("surface", "source_only_nonmatched_action_expert_capacity_upper_bound"),
        ("task_ids", [3, 4]),
        ("variant", EXPECTED_VARIANT),
        ("candidate_step", 1000),
    ):
        _require(spec.get(key), expected, key)
    authority = spec.get("authority", {})
    for key, path in (
        ("gate_zero_contract_sha256", gate_zero_path),
        ("phase0_contract_sha256", phase0_path),
        ("source_competence_contract_sha256", competence_path),
        ("fit_contract_sha256", fit_path),
    ):
        _require(authority.get(key), sha256_file(path), key)
    _require(authority.get("proposal_a_result_sha256"), EXPECTED_PROPOSAL_SHA, "Proposal A")
    _require(authority.get("signed_ratio_result_sha256"), EXPECTED_SIGNED_SHA, "signed result")
    for key in ("validation_numeric_access", "held_numeric_access", "locked_report_numeric_access"):
        _require(authority.get(key), False, key)
    evidence = spec.get("candidate_evidence", {})
    expected_evidence = {
        "task3_candidate_manifest_sha256": "a1eaaf1d7e81b6d602743b5c943eb65a142d07fad9ec66f038fde3326fce24ab",
        "task3_trainable_state_sha256": "67a2ffe2054e0cd1985211bcfe6fbf929d7203ab4572f56fc12b1cbc089962b2",
        "task3_query_reduction_fraction": 0.07883101786715387,
        "task3_action_drift_proxy": 0.027359457686543465,
        "task4_candidate_manifest_sha256": "ca5f1586b9b735014e00c778e003246235ef4707eb8030d62e78684d2d0dbe1c",
        "task4_trainable_state_sha256": "d5a94c0f8b34b06d939bfea3e1b62fc9db48db648ab8827f65d705709736226c",
        "task4_query_reduction_fraction": 0.03710404010162202,
        "task4_action_drift_proxy": 0.02131560631096363,
        "trainable_parameters": 99_880_992,
        "trainable_tensors": 155,
        "matched_lora_baseline": False,
        "may_authorize_gate_zero": False,
        "may_authorize_writer": False,
        "may_seal_writer_target_contract": False,
    }
    for key, expected in expected_evidence.items():
        _require(evidence.get(key), expected, f"candidate {key}")
    base = [True, False, False, False, True, False, True, False]
    _require(spec.get("base_reference", {}).get("task3_successes"), base, "task3 base")
    _require(spec.get("base_reference", {}).get("task4_successes"), base, "task4 base")
    rollout = spec.get("rollout", {})
    rollout_expected = {
        "init_state_indices": list(range(40, 48)),
        "batch_size": 8,
        "seed_start": 5800,
        "warmup_seed_start": 5792,
        "policy_rng_seed": 2026071836,
        "reuse_base_without_rerun": True,
        "conditions": [EXPECTED_VARIANT],
        "retain_one_video_per_arm": True,
    }
    for key, expected in rollout_expected.items():
        _require(rollout.get(key), expected, f"rollout {key}")
    decision = spec.get("decision", {})
    for key, expected in (
        ("minimum_each_task_success_gain_exclusive_pp", 0.0),
        ("minimum_positive_task_count", 2),
        ("minimum_median_success_gain_pp", 15.0),
        ("pass_status", "nonmatched_action_expert_capacity_behavioral_signal_present"),
        ("fail_status", "nonmatched_action_expert_capacity_behavioral_signal_absent"),
        ("result_cannot_authorize_gate_zero_or_writer", True),
        ("no_threshold_or_task_change_after_outcome", True),
    ):
        _require(decision.get(key), expected, f"decision {key}")
    resources = spec.get("resources", {})
    for key, expected in (
        ("maximum_concurrent_gpus", 4),
        ("diagnostic_gpus", 2),
        ("minimum_free_memory_mib", 10_240),
        ("maximum_new_rollout_episodes", 16),
    ):
        _require(resources.get(key), expected, f"resource {key}")
    load_mature_lora_positive_control_spec(
        fit_path,
        gate_zero_path=gate_zero_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
    )


def load_capacity_closed_loop_spec(
    path: Path,
    *,
    gate_zero_path: Path,
    phase0_path: Path,
    competence_path: Path,
    fit_path: Path,
) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            spec = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise GateZeroCapacityClosedLoopError("invalid capacity TOML") from error
    validate_capacity_closed_loop_spec(
        spec,
        gate_zero_path=gate_zero_path,
        phase0_path=phase0_path,
        competence_path=competence_path,
        fit_path=fit_path,
    )
    return spec


def assigned_capacity_arms(
    *, rank: int, world_size: int, spec: Mapping[str, Any]
) -> list[int]:
    if world_size not in {1, 2} or not 0 <= rank < world_size:
        raise GateZeroCapacityClosedLoopError("invalid capacity topology")
    return list(spec["task_ids"]) if world_size == 1 else [spec["task_ids"][rank]]


def _validated_arms(
    spec: Mapping[str, Any], arms: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    rollout = spec["rollout"]
    expected_seeds = list(range(rollout["seed_start"], rollout["seed_start"] + 8))
    by_task = {}
    for arm in arms:
        task_id = arm.get("task_id")
        successes = arm.get("successes")
        if (
            task_id not in spec["task_ids"]
            or task_id in by_task
            or arm.get("condition") != spec["variant"]
            or arm.get("mechanics_valid") is not True
            or arm.get("seeds") != expected_seeds
            or arm.get("official_rollout_init_state_indices") != rollout["init_state_indices"]
            or not isinstance(successes, list)
            or len(successes) != 8
            or any(not isinstance(value, bool) for value in successes)
        ):
            raise GateZeroCapacityClosedLoopError("invalid capacity arm")
        by_task[task_id] = arm
    if set(by_task) != set(spec["task_ids"]):
        raise GateZeroCapacityClosedLoopError("capacity arms are incomplete")
    return by_task


def decide_capacity_closed_loop(
    spec: Mapping[str, Any], arms: list[dict[str, Any]]
) -> dict[str, Any]:
    by_task = _validated_arms(spec, arms)
    nets, gains = {}, []
    for task_id in spec["task_ids"]:
        base = spec["base_reference"][f"task{task_id}_successes"]
        net = sum(int(right) - int(left) for left, right in zip(base, by_task[task_id]["successes"], strict=True))
        nets[str(task_id)] = net
        gains.append(net * 12.5)
    threshold = spec["decision"]
    positive = sum(value > threshold["minimum_each_task_success_gain_exclusive_pp"] for value in gains)
    median = statistics.median(gains)
    passed = positive >= threshold["minimum_positive_task_count"] and median >= threshold[
        "minimum_median_success_gain_pp"
    ]
    return {
        "status": threshold["pass_status"] if passed else threshold["fail_status"],
        "capacity_signal_present": passed,
        "paired_net_wins_by_task": nets,
        "success_gain_pp_by_task": {str(task): gain for task, gain in zip(spec["task_ids"], gains, strict=True)},
        "positive_task_count": positive,
        "median_success_gain_pp": median,
        "gate_zero_authorized": False,
        "writer_authorized": False,
        "validation_numeric_access": False,
        "held_numeric_access": False,
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateZeroCapacityClosedLoopError(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise GateZeroCapacityClosedLoopError(f"invalid {label}")
    return value


def _validate_upstream_results(args: argparse.Namespace, spec: Mapping[str, Any]) -> None:
    authority = spec["authority"]
    if sha256_file(args.proposal_a_result) != authority["proposal_a_result_sha256"]:
        raise GateZeroCapacityClosedLoopError("Proposal-A result hash changed")
    if sha256_file(args.signed_ratio_result) != authority["signed_ratio_result_sha256"]:
        raise GateZeroCapacityClosedLoopError("signed-ratio result hash changed")
    proposal = _load_json(args.proposal_a_result, "Proposal-A result")
    signed = _load_json(args.signed_ratio_result, "signed-ratio result")
    base = {
        arm["task_id"]: arm["successes"]
        for arm in proposal.get("arms", [])
        if arm.get("condition") == "frozen_base"
    }
    if (
        proposal.get("status") != "mature_lora_headroom_control_failed_gate_recovery_required"
        or base != {task: spec["base_reference"][f"task{task}_successes"] for task in (3, 4)}
        or signed.get("status") != "task_local_rl_early_check_not_supported"
        or signed.get("gate_zero_authorized") is not False
        or signed.get("writer_authorized") is not False
    ):
        raise GateZeroCapacityClosedLoopError("upstream failure boundary changed")


def _candidate_state(
    spec: Mapping[str, Any], *, fit_root: Path, task_id: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    output = fit_root / f"{spec['variant']}_task{task_id}" / "candidates" / "001000"
    manifest = validate_candidate_artifact(
        output, expected={"variant": spec["variant"], "task_id": task_id, "step": 1000}
    )
    evidence = spec["candidate_evidence"]
    if (
        sha256_file(output / "candidate_manifest.json")
        != evidence[f"task{task_id}_candidate_manifest_sha256"]
        or sha256_file(output / "trainable_state.safetensors")
        != evidence[f"task{task_id}_trainable_state_sha256"]
        or manifest.get("trainable_parameters") != evidence["trainable_parameters"]
        or manifest.get("trainable_tensors") != evidence["trainable_tensors"]
    ):
        raise GateZeroCapacityClosedLoopError("capacity candidate hash or identity changed")
    metrics = manifest["metrics"]
    reduction = (metrics["base_query_flow_mse"] - metrics["query_flow_mse"]) / metrics[
        "base_query_flow_mse"
    ]
    if (
        reduction != evidence[f"task{task_id}_query_reduction_fraction"]
        or metrics["action_drift_proxy"] != evidence[f"task{task_id}_action_drift_proxy"]
    ):
        raise GateZeroCapacityClosedLoopError("capacity query evidence changed")
    return load_file(output / "trainable_state.safetensors"), manifest


def _rollout_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    rollout = spec["rollout"]
    return {
        "report": {
            "rollout_batch_size": rollout["batch_size"],
            "official_rollout_init_state_indices": rollout["init_state_indices"],
            "seed_start": rollout["seed_start"],
            "warmup_seed_start": rollout["warmup_seed_start"],
            "policy_rng_seed": rollout["policy_rng_seed"],
        },
        "resources": {"retain_one_video_per_report_arm": rollout["retain_one_video_per_arm"]},
    }


def _evaluate_local(
    args: argparse.Namespace,
    spec: Mapping[str, Any],
    fit: Mapping[str, Any],
    parent: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    context: ParallelContext,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arms, authorities = [], []
    for task_id in assigned_capacity_arms(rank=context.rank, world_size=context.world_size, spec=spec):
        language, authority = _task_authority(task_id, spec["rollout"]["init_state_indices"])
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
            variant_spec=fit["fit"][spec["variant"]],
        )
        state, manifest = _candidate_state(spec, fit_root=args.fit_root, task_id=task_id)
        restore_trainable_state(model, state)
        model.eval()
        runtime[0] = model
        arm = {
            "task_id": task_id,
            "condition": spec["variant"],
            "candidate_step": 1000,
            "state_authority": {
                "candidate_manifest_sha256": spec["candidate_evidence"][f"task{task_id}_candidate_manifest_sha256"],
                "trainable_state_sha256": manifest["files"]["trainable_state.safetensors"]["sha256"],
                "trainable_parameters": summary["trainable_parameters"],
            },
            **_closed_loop_metrics(
                runtime=tuple(runtime),
                task_id=task_id,
                condition=spec["variant"],
                language=language,
                spec=_rollout_spec(spec),
                output_dir=args.output_dir,
            ),
        }
        arms.append(arm)
        authorities.append(authority)
        print(json.dumps({"event": "capacity_closed_loop_arm", "task_id": task_id, "successes": sum(arm["successes"]), "mechanics_valid": arm["mechanics_valid"]}, sort_keys=True), flush=True)
        del runtime, model, state
        gc.collect()
        torch.cuda.empty_cache()
    return arms, authorities


def _prepare_tracker(context: ParallelContext, args: argparse.Namespace, spec: Mapping[str, Any]) -> Any:
    error, tracker = None, None
    if context.is_primary:
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            unexpected = [
                path.name
                for path in args.output_dir.iterdir()
                if not path.name.startswith("gpu_telemetry_")
            ]
            if unexpected:
                raise GateZeroCapacityClosedLoopError(f"non-fresh output: {unexpected}")
            import trackio

            trackio.init(
                project=spec["resources"]["tracking_project"],
                name=args.output_dir.name,
                group=spec["resources"]["tracking_group"],
                config={
                    "world_size": context.world_size,
                    "task_ids": spec["task_ids"],
                    "nonmatched": True,
                },
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
        raise GateZeroCapacityClosedLoopError(error)
    return tracker


def _eval_info(arms: list[dict[str, Any]], decision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "overall": {
            "surface": "source_only_nonmatched_action_expert_capacity_upper_bound",
            "status": decision["status"],
            "gate_zero_authorized": False,
            "writer_authorized": False,
        },
        "per_task": [
            {
                "task_group": f"capacity_upper_bound:{arm['condition']}",
                "task_id": arm["task_id"],
                "metrics": {key: arm[key] for key in ("successes", "sum_rewards", "max_rewards", "video_paths")},
            }
            for arm in arms
        ],
    }


def run_capacity_closed_loop(args: argparse.Namespace) -> dict[str, Any]:
    spec = load_capacity_closed_loop_spec(
        args.config,
        gate_zero_path=args.gate_zero_contract,
        phase0_path=args.phase0_contract,
        competence_path=args.source_competence_contract,
        fit_path=args.fit_contract,
    )
    _validate_upstream_results(args, spec)
    fit = load_mature_lora_positive_control_spec(
        args.fit_contract,
        gate_zero_path=args.gate_zero_contract,
        phase0_path=args.phase0_contract,
        competence_path=args.source_competence_contract,
    )
    parent = load_gate_zero_contract(args.gate_zero_contract, args.phase0_contract)
    checkpoint = validate_source_base_checkpoint(
        args.source_base_checkpoint,
        expected={
            "step": spec["authority"]["source_base_checkpoint_step"],
            "checkpoint_role": spec["authority"]["source_base_checkpoint_role"],
        },
    )
    if sha256_file(args.source_base_checkpoint / CHECKPOINT_MANIFEST) != spec["authority"][
        "source_base_checkpoint_manifest_sha256"
    ]:
        raise GateZeroCapacityClosedLoopError("source checkpoint hash changed")
    context = _initialize_parallel()
    tracker = None
    try:
        tracker = _prepare_tracker(context, args, spec)
        started = time.perf_counter()
        local_arms, local_authorities = _evaluate_local(
            args, spec, fit, parent, checkpoint, context
        )
        gathered_arms = _gather(context, local_arms)
        gathered_authorities = _gather(context, local_authorities)
        if not context.is_primary:
            return {"status": "non_primary_rank_complete", "rank": context.rank}
        arms = sorted((arm for rows in gathered_arms for arm in rows), key=lambda arm: arm["task_id"])
        authorities = {row["task_id"]: row for rows in gathered_authorities for row in rows}
        decision = decide_capacity_closed_loop(spec, arms)
        result = {
            "schema_version": 1,
            "status": decision["status"],
            "surface": spec["surface"],
            "contract_sha256": sha256_file(args.config),
            "fit_contract_sha256": sha256_file(args.fit_contract),
            "proposal_a_result_sha256": sha256_file(args.proposal_a_result),
            "signed_ratio_result_sha256": sha256_file(args.signed_ratio_result),
            "task_authorities": [authorities[task] for task in spec["task_ids"]],
            "base_reference": spec["base_reference"],
            "arms": arms,
            "decision": decision,
            "interpretation": spec["interpretation"],
            "gate_zero_authorized": False,
            "writer_authorized": False,
            "validation_numeric_access": False,
            "held_numeric_access": False,
            "resources": {"physical_gpus": args.physical_gpus, "gpu_count": context.world_size, "wall_seconds": time.perf_counter() - started},
            "tracking": {"backend": "trackio", "project": spec["resources"]["tracking_project"], "run": args.output_dir.name, "dashboard_command": "trackio show --project EMBER_gate0"},
        }
        atomic_json(args.output_dir / RESULT_NAME, result)
        atomic_json(args.output_dir / "eval_info.json", _eval_info(arms, decision))
        build_eval_gallery(args.output_dir)
        write_output_checksums(args.output_dir)
        update_latest_link(args.output_dir, args.latest_link)
        tracker.log({"capacity/signal_present": int(decision["capacity_signal_present"]), "capacity/median_gain_pp": decision["median_success_gain_pp"]})
        tracker.finish()
        tracker = None
        return result
    finally:
        if tracker is not None:
            tracker.finish()
        _close_parallel(context)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config", "gate-zero-contract", "phase0-contract", "source-competence-contract",
        "fit-contract", "source-base-checkpoint", "fit-root", "proposal-a-result",
        "signed-ratio-result", "output-dir", "latest-link",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--physical-gpus", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.absolute())
    try:
        result = run_capacity_closed_loop(args)
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
                "gate_zero_authorized": False,
                "writer_authorized": False,
                "validation_numeric_access": False,
                "held_numeric_access": False,
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
