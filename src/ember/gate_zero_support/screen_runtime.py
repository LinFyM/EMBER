"""Run a frozen source closed-loop screen for the bounded LoRA supports."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from dataclasses import dataclass
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
    validate_selected_artifact,
    write_output_checksums,
)
from ember.gate_zero_oracle_report_runtime import _closed_loop_metrics, _task_authority
from ember.gate_zero_oracle_session import configure_oracle_variant
from ember.gate_zero_support.contract import load_target_support_screen_spec
from ember.gate_zero_support.screen import (
    assigned_support_screening_arms,
    canonical_support_screening_shards,
    decide_support_screening,
    validate_support_screening_grant,
)


RESULT_NAME = "support_screening_result.json"


class GateZeroTargetSupportScreenRuntimeError(RuntimeError):
    """Raised when source rollout screening differs from its frozen grant."""


@dataclass(frozen=True)
class ParallelContext:
    rank: int
    local_rank: int
    world_size: int
    initialized: bool

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def support_state_authority(
    task_id: int, condition: str, *, variants: list[str] | None = None
) -> tuple[str | None, int | None]:
    if task_id not in {3, 4}:
        raise GateZeroTargetSupportScreenRuntimeError("screening task is invalid")
    if condition == "frozen_base":
        return None, None
    variants = (
        ["last_two_qv_r8", "all_expert_qv_r8", "official_default_r8"]
        if variants is None
        else variants
    )
    if condition in variants:
        return condition, task_id
    raise GateZeroTargetSupportScreenRuntimeError("screening condition is invalid")


def _initialize_parallel() -> ParallelContext:
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError as error:
        raise GateZeroTargetSupportScreenRuntimeError("invalid rank environment") from error
    if world_size not in {1, 2, 4} or rank < 0 or rank >= world_size:
        raise GateZeroTargetSupportScreenRuntimeError("invalid rank environment")
    if not torch.cuda.is_available():
        raise GateZeroTargetSupportScreenRuntimeError("screening requires CUDA")
    torch.cuda.set_device(local_rank)
    initialized = False
    if world_size > 1:
        torch.distributed.init_process_group(backend="gloo", init_method="env://")
        initialized = True
    return ParallelContext(rank, local_rank, world_size, initialized)


def _broadcast(context: ParallelContext, value: Any) -> Any:
    if context.world_size == 1:
        return value
    payload = [value if context.is_primary else None]
    torch.distributed.broadcast_object_list(payload, src=0)
    return payload[0]


def _gather(context: ParallelContext, value: Any) -> list[Any] | None:
    if context.world_size == 1:
        return [value]
    result = [None] * context.world_size if context.is_primary else None
    torch.distributed.gather_object(value, result, dst=0)
    return result


def _close_parallel(context: ParallelContext) -> None:
    if context.initialized:
        torch.distributed.destroy_process_group()


def _fit_outputs(
    fit_root: Path, spec: Mapping[str, Any]
) -> dict[tuple[str, int], Path]:
    return {
        (variant, task_id): fit_root / f"{variant}_task{task_id}"
        for variant in spec["variants"]
        for task_id in spec["task_ids"]
    }


def _load_authorities(
    arguments: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    spec = load_target_support_screen_spec(
        arguments.config,
        gate_zero_path=arguments.gate_zero_contract,
        phase0_path=arguments.phase0_contract,
        competence_path=arguments.source_competence_contract,
        prior_execution_path=arguments.config.with_name(
            "gate_zero_oracle_execution.toml"
        ),
    )
    parent = load_gate_zero_contract(
        arguments.gate_zero_contract, arguments.phase0_contract
    )
    checkpoint = validate_source_base_checkpoint(
        arguments.source_base_checkpoint,
        expected={
            "step": spec["authority"]["source_base_checkpoint_step"],
            "checkpoint_role": spec["authority"]["source_base_checkpoint_role"],
        },
    )
    if (
        sha256_file(arguments.source_base_checkpoint / CHECKPOINT_MANIFEST)
        != spec["authority"]["source_base_checkpoint_manifest_sha256"]
    ):
        raise GateZeroTargetSupportScreenRuntimeError(
            "source-base checkpoint authority changed"
        )
    outputs = _fit_outputs(arguments.fit_root, spec)
    grant = validate_support_screening_grant(
        grant_path=arguments.screening_grant,
        config_path=arguments.config,
        parent_path=arguments.gate_zero_contract,
        phase0_path=arguments.phase0_contract,
        competence_path=arguments.source_competence_contract,
        fit_outputs=outputs,
    )
    return spec, parent, checkpoint, grant


def _selected_state(
    *,
    spec: Mapping[str, Any],
    variant: str,
    task_id: int,
    fit_root: Path,
    grant: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    output = fit_root / f"{variant}_task{task_id}"
    if spec.get("screening_stage") == "mature_lora_headroom_control":
        from ember.gate_zero_support.mature_headroom import (
            load_staged_candidate_state,
        )

        return load_staged_candidate_state(
            output=output,
            task_id=task_id,
            grant=grant,
        )
    selected_dir = output / "selected"
    selected = validate_selected_artifact(
        selected_dir, expected={"variant": variant, "task_id": task_id}
    )
    expected = grant["fit_evidence"][f"{variant}:task{task_id}"]
    if (
        selected["trainable_state_sha256"]
        != expected["selected_trainable_state_sha256"]
    ):
        raise GateZeroTargetSupportScreenRuntimeError(
            "selected state differs from screening grant"
        )
    return load_file(selected_dir / "trainable_state.safetensors"), selected


def _open_arm_runtime(
    *,
    task_id: int,
    condition: str,
    spec: dict[str, Any],
    parent: dict[str, Any],
    checkpoint: dict[str, Any],
    grant: dict[str, Any],
    arguments: argparse.Namespace,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    runtime = list(
        _load_policy(
            arguments.source_base_checkpoint / "pretrained_model",
            {"task_suite": "libero_90", "task_id": task_id},
        )
    )
    variant, state_task = support_state_authority(
        task_id, condition, variants=spec["variants"]
    )
    evidence: dict[str, Any] = {"variant": variant, "state_task_id": state_task}
    if variant is not None and state_task is not None:
        model, summary = configure_oracle_variant(
            runtime[0],
            parent=parent,
            checkpoint=checkpoint,
            variant=variant,
            variant_spec=spec["fit"][variant],
        )
        state, selected = _selected_state(
            spec=spec,
            variant=variant,
            task_id=state_task,
            fit_root=arguments.fit_root,
            grant=grant,
        )
        restore_trainable_state(model, state)
        runtime[0] = model
        evidence.update(
            {
                "selected_step": selected["selected_step"],
                "selected_trainable_state_sha256": selected[
                    "trainable_state_sha256"
                ],
                "trainable_parameters": summary["trainable_parameters"],
            }
        )
    runtime[0].eval()
    return tuple(runtime), evidence


def _rollout_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    screening = spec["screening_rollout"]
    return {
        "report": {
            "rollout_batch_size": screening["batch_size"],
            "official_rollout_init_state_indices": screening[
                "init_state_indices"
            ],
            "seed_start": screening["seed_start"],
            "warmup_seed_start": screening["warmup_seed_start"],
            "policy_rng_seed": screening["policy_rng_seed"],
        },
        "resources": {
            "retain_one_video_per_report_arm": spec["resources"][
                "retain_one_video_per_rollout_arm"
            ]
        },
    }


def _evaluate_local_arms(
    *,
    context: ParallelContext,
    spec: dict[str, Any],
    parent: dict[str, Any],
    checkpoint: dict[str, Any],
    grant: dict[str, Any],
    arguments: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    local = []
    authorities = {}
    rollout_spec = _rollout_spec(spec)
    for task_id, condition in assigned_support_screening_arms(
        rank=context.rank,
        world_size=context.world_size,
        variants=spec["variants"],
        task_ids=spec["task_ids"],
    ):
        language, authority = _task_authority(
            task_id, spec["screening_rollout"]["init_state_indices"]
        )
        authorities[task_id] = authority
        runtime, state_evidence = _open_arm_runtime(
            task_id=task_id,
            condition=condition,
            spec=spec,
            parent=parent,
            checkpoint=checkpoint,
            grant=grant,
            arguments=arguments,
        )
        metrics = _closed_loop_metrics(
            runtime=runtime,
            task_id=task_id,
            condition=condition,
            language=language,
            spec=rollout_spec,
            output_dir=arguments.output_dir,
        )
        arm = {
            "task_id": task_id,
            "condition": condition,
            "state_authority": state_evidence,
            **metrics,
        }
        local.append(arm)
        print(
            json.dumps(
                {
                    "event": "gate_zero_target_support_screen_arm",
                    "rank": context.rank,
                    "task_id": task_id,
                    "condition": condition,
                    "successes": sum(arm["successes"]),
                    "episodes": len(arm["successes"]),
                    "mechanics_valid": arm["mechanics_valid"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del runtime
        gc.collect()
        torch.cuda.empty_cache()
    return local, list(authorities.values())


def _prepare_output(
    context: ParallelContext, arguments: argparse.Namespace, spec: dict[str, Any]
) -> Any:
    error = None
    tracker = None
    if context.is_primary:
        try:
            arguments.output_dir.mkdir(parents=True, exist_ok=True)
            unexpected = [
                path.name
                for path in arguments.output_dir.iterdir()
                if not (
                    path.is_file()
                    and path.name.startswith("gpu_telemetry_")
                    and path.suffix == ".csv"
                )
            ]
            if unexpected:
                raise GateZeroTargetSupportScreenRuntimeError(
                    f"refusing non-fresh screening output: {unexpected}"
                )
            import trackio

            trackio.init(
                project=spec["resources"]["tracking_project"],
                name=arguments.output_dir.name,
                group=f"target_support_{spec.get('screening_stage', 'rank8')}_screening",
                config={
                    "world_size": context.world_size,
                    "surface": (
                        "source_recovery_init_"
                        f"{spec['screening_rollout']['init_state_indices'][0]}_"
                        f"{spec['screening_rollout']['init_state_indices'][-1]}"
                    ),
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
    if error is not None:
        raise GateZeroTargetSupportScreenRuntimeError(error)
    return tracker


def _ordered_arms(
    gathered: list[list[dict[str, Any]]], spec: Mapping[str, Any]
) -> list[dict[str, Any]]:
    arms = [arm for values in gathered for arm in values]
    order = {
        arm: index
        for index, arm in enumerate(
            arm
            for shard in canonical_support_screening_shards(
                variants=spec["variants"], task_ids=spec["task_ids"]
            )
            for arm in shard
        )
    }
    arms.sort(key=lambda arm: order[(arm["task_id"], arm["condition"])])
    return arms


def _eval_info(arms: list[dict[str, Any]], decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall": {
            "surface": "source_target_support_closed_loop_screening",
            "status": decision["status"],
            "episodes": sum(len(arm["successes"]) for arm in arms),
            "successes": sum(sum(arm["successes"]) for arm in arms),
            "writer_authorized": decision["writer_authorized"],
        },
        "per_task": [
            {
                "task_group": f"libero_90:{arm['condition']}",
                "task_id": arm["task_id"],
                "metrics": {
                    "successes": arm["successes"],
                    "sum_rewards": arm["sum_rewards"],
                    "max_rewards": arm["max_rewards"],
                    "video_paths": arm["video_paths"],
                },
            }
            for arm in arms
        ],
    }


def _selected_support_freeze(
    *, decision: Mapping[str, Any], spec: Mapping[str, Any], grant: Mapping[str, Any]
) -> dict[str, Any] | None:
    variant = decision["selected_variant"]
    if variant is None:
        return None
    contract = spec["fit"][variant]
    return {
        "variant": variant,
        "target_modules": contract["target_modules"],
        "rank": contract["rank"],
        "alpha": contract["alpha"],
        "dropout": contract["dropout"],
        "trainable_parameters": contract["expected_trainable_parameters"],
        "selected_trainable_state_sha256_by_task": {
            str(task_id): grant["fit_evidence"][f"{variant}:task{task_id}"][
                "selected_trainable_state_sha256"
            ]
            for task_id in spec["task_ids"]
        },
        "selection_changes_after_freeze_forbidden": True,
        "final_confirmation_required": spec.get("screening_stage")
        not in {"mature_positive_control", "mature_lora_headroom_control"},
    }


def _publish(
    *,
    arguments: argparse.Namespace,
    spec: dict[str, Any],
    grant: dict[str, Any],
    arms: list[dict[str, Any]],
    task_authorities: list[dict[str, Any]],
    context: ParallelContext,
    tracker: Any,
    wall_seconds: float,
) -> dict[str, Any]:
    decision = decide_support_screening(
        arms=arms,
        grant=grant,
        variants=spec["variants"],
        task_ids=spec["task_ids"],
        parameter_counts={
            variant: spec["fit"][variant]["expected_trainable_parameters"]
            for variant in spec["variants"]
        },
        thresholds=spec["decision"],
        expected_init_state_indices=spec["screening_rollout"][
            "init_state_indices"
        ],
        expected_seeds=list(
            range(
                spec["screening_rollout"]["seed_start"],
                spec["screening_rollout"]["seed_start"]
                + spec["screening_rollout"]["batch_size"],
            )
        ),
        rank_stage=spec.get("screening_stage", "rank8"),
    )
    result = {
        "schema_version": 1,
        "status": decision["status"],
        "surface": "source_target_support_closed_loop_screening",
        "target_support_contract_sha256": sha256_file(arguments.config),
        "screening_grant_sha256": sha256_file(arguments.screening_grant),
        "task_authorities": sorted(
            task_authorities, key=lambda value: value["task_id"]
        ),
        "arms": arms,
        "decision": decision,
        "selected_support_freeze": _selected_support_freeze(
            decision=decision, spec=spec, grant=grant
        ),
        "confirmation_authorized": decision["confirmation_authorized"],
        "locked_report_access_authorized": False,
        "selection_changes_after_screening_forbidden": True,
        "gate_zero_authorized": decision["gate_zero_authorized"],
        "writer_authorized": decision["writer_authorized"],
        "final_writer_target_contract_sealed": decision[
            "final_writer_target_contract_sealed"
        ],
        "validation_numeric_access": False,
        "held_numeric_access": False,
        "parallel": {
            "world_size": context.world_size,
            "shards": canonical_support_screening_shards(
                variants=spec["variants"], task_ids=spec["task_ids"]
            ),
        },
        "resources": {
            "physical_gpus": arguments.physical_gpus,
            "gpu_count": context.world_size,
            "wall_seconds": wall_seconds,
        },
        "tracking": {
            "backend": "trackio",
            "project": spec["resources"]["tracking_project"],
            "run": arguments.output_dir.name,
            "dashboard_command": "trackio show --project EMBER_gate0",
        },
    }
    atomic_json(arguments.output_dir / RESULT_NAME, result)
    atomic_json(arguments.output_dir / "eval_info.json", _eval_info(arms, decision))
    build_eval_gallery(arguments.output_dir)
    write_output_checksums(arguments.output_dir)
    update_latest_link(arguments.output_dir, arguments.latest_link)
    tracker.log(
        {
            "support_screen/selected": int(decision["selected_variant"] is not None),
            "support_screen/rank16_authorized": int(decision["rank16_authorized"]),
        }
    )
    tracker.finish()
    return result


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    context = _initialize_parallel()
    tracker = None
    try:
        spec, parent, checkpoint, grant = _load_authorities(arguments)
        tracker = _prepare_output(context, arguments, spec)
        started = time.perf_counter()
        local_arms, local_authorities = _evaluate_local_arms(
            context=context,
            spec=spec,
            parent=parent,
            checkpoint=checkpoint,
            grant=grant,
            arguments=arguments,
        )
        gathered_arms = _gather(context, local_arms)
        gathered_authorities = _gather(context, local_authorities)
        if not context.is_primary:
            return {"status": "non_primary_rank_complete", "rank": context.rank}
        arms = _ordered_arms(gathered_arms, spec)
        by_task = {
            row["task_id"]: row
            for values in gathered_authorities
            for row in values
        }
        result = _publish(
            arguments=arguments,
            spec=spec,
            grant=grant,
            arms=arms,
            task_authorities=list(by_task.values()),
            context=context,
            tracker=tracker,
            wall_seconds=time.perf_counter() - started,
        )
        tracker = None
        return result
    finally:
        if tracker is not None and context.is_primary:
            try:
                tracker.finish()
            except Exception:
                pass
        _close_parallel(context)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate-zero-contract", type=Path, required=True)
    parser.add_argument("--phase0-contract", type=Path, required=True)
    parser.add_argument("--source-competence-contract", type=Path, required=True)
    parser.add_argument("--source-base-checkpoint", type=Path, required=True)
    parser.add_argument("--screening-grant", type=Path, required=True)
    parser.add_argument("--fit-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latest-link", type=Path, required=True)
    parser.add_argument("--physical-gpus", required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    try:
        result = run(arguments)
    except BaseException as error:
        rank = os.environ.get("RANK", "0")
        try:
            arguments.output_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(
                arguments.output_dir / f"failure_packet_rank_{rank}.json",
                {
                    "schema_version": 1,
                    "status": "failed",
                    "rank": rank,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "gate_zero_authorized": False,
                    "writer_authorized": False,
                },
            )
        finally:
            raise
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
