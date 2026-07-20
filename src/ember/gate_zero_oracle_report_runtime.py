"""Run the frozen four-shard Gate 0 offline and closed-loop report."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
import tomllib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file

from ember.eval_artifacts import build_eval_gallery, update_latest_link
from ember.evaluation_identity import _load_policy
from ember.gate_zero_checkpoint import CHECKPOINT_MANIFEST, validate_source_base_checkpoint
from ember.gate_zero_contract import load_gate_zero_contract
from ember.gate_zero_data import (
    GateZeroSurface,
    SourceHdf5Dataset,
    load_surface_authorities,
)
from ember.gate_zero_oracle_artifacts import (
    atomic_json,
    restore_trainable_state,
    sha256_file,
    validate_selected_artifact,
    write_output_checksums,
)
from ember.gate_zero_oracle_contract import load_oracle_execution_spec
from ember.gate_zero_oracle_metrics import FixedQueryEvaluator
from ember.gate_zero_oracle_report import (
    assigned_report_arms,
    canonical_report_shards,
    decide_gate_zero_report,
    validate_selection_freeze_grant,
)
from ember.gate_zero_oracle_session import configure_oracle_variant
from ember.gate_zero_support.closed_loop import (
    ClosedLoopMetricsSession,
    EpisodeDiagnosticEnv,
    GateZeroOracleReportRuntimeError,
    closed_loop_metrics as _closed_loop_metrics,
    report_warmup_seed_batches,
    validate_report_reset_identity,
)
from ember.specification_probe import (
    paired_gap_summary,
)


@dataclass(frozen=True)
class ParallelContext:
    rank: int
    local_rank: int
    world_size: int
    initialized: bool

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def report_state_authority(task_id: int, condition: str) -> tuple[str | None, int | None]:
    if task_id not in {3, 4}:
        raise GateZeroOracleReportRuntimeError("report task is outside the frozen pilot")
    if condition == "frozen_base":
        return None, None
    if condition == "own_adapter":
        return "lora", task_id
    if condition == "swapped_adapter":
        return "lora", 4 if task_id == 3 else 3
    if condition == "partial_upper_bound":
        return "partial_upper_bound", task_id
    raise GateZeroOracleReportRuntimeError("unknown report condition")


def _initialize_parallel() -> ParallelContext:
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError as error:
        raise GateZeroOracleReportRuntimeError("invalid torchrun rank environment") from error
    assigned_report_arms(rank=rank, world_size=world_size)
    if not torch.cuda.is_available():
        raise GateZeroOracleReportRuntimeError("locked report requires CUDA")
    torch.cuda.set_device(local_rank)
    initialized = False
    if world_size > 1:
        torch.distributed.init_process_group(backend="gloo", init_method="env://")
        initialized = True
    return ParallelContext(rank, local_rank, world_size, initialized)


def _close_parallel(context: ParallelContext) -> None:
    if context.initialized:
        torch.distributed.destroy_process_group()


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


def _fit_outputs(arguments: argparse.Namespace) -> dict[tuple[str, int], Path]:
    return {
        ("lora", 3): arguments.lora_task3,
        ("lora", 4): arguments.lora_task4,
        ("partial_upper_bound", 3): arguments.partial_task3,
        ("partial_upper_bound", 4): arguments.partial_task4,
    }


def checkpoint_manifest_path(checkpoint_dir: Path) -> Path:
    """Resolve the manifest through the canonical checkpoint owner."""

    return checkpoint_dir / CHECKPOINT_MANIFEST


def _load_authorities(arguments: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    spec = load_oracle_execution_spec(
        arguments.config,
        gate_zero_path=arguments.gate_zero_contract,
        phase0_path=arguments.phase0_contract,
        competence_path=arguments.source_competence_contract,
    )
    parent = load_gate_zero_contract(
        arguments.gate_zero_contract, arguments.phase0_contract
    )
    with arguments.phase0_contract.open("rb") as handle:
        phase0 = tomllib.load(handle)
    checkpoint = validate_source_base_checkpoint(arguments.source_base_checkpoint)
    if spec["parallel"].get("report_shards") != [
        "task3:base_own",
        "task3:swapped_partial",
        "task4:base_own",
        "task4:swapped_partial",
    ]:
        raise GateZeroOracleReportRuntimeError("report shard contract changed")
    expected = spec["authority"]
    if (
        checkpoint.get("step") != expected["source_base_checkpoint_step"]
        or checkpoint.get("checkpoint_role") != expected["source_base_checkpoint_role"]
        or sha256_file(checkpoint_manifest_path(arguments.source_base_checkpoint))
        != expected["source_base_checkpoint_manifest_sha256"]
    ):
        raise GateZeroOracleReportRuntimeError("source-base checkpoint authority changed")
    grant = validate_selection_freeze_grant(
        grant_path=arguments.selection_freeze_grant,
        execution_path=arguments.config,
        parent_path=arguments.gate_zero_contract,
        phase0_path=arguments.phase0_contract,
        competence_path=arguments.source_competence_contract,
        fit_outputs=_fit_outputs(arguments),
    )
    return spec, parent, phase0, checkpoint, grant


def _task_authority(task_id: int, init_indices: Sequence[int]) -> tuple[str, dict[str, Any]]:
    from libero.libero import get_libero_path
    from lerobot.envs.libero import _get_suite, get_task_init_states

    suite = _get_suite("libero_90")
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    init_states = np.asarray(get_task_init_states(suite, task_id))
    return task.language, {
        "task_id": task_id,
        "task_name": task.name,
        "language": task.language,
        "bddl_filename": task.bddl_file,
        "bddl_sha256": sha256_file(bddl),
        "init_state_filename": task.init_states_file,
        "init_state_file_sha256": sha256_file(init_path),
        "init_state_indices": list(init_indices),
        "init_state_sha256": [
            hashlib.sha256(np.ascontiguousarray(init_states[index]).tobytes()).hexdigest()
            for index in init_indices
        ],
    }


def _selected_state(
    *,
    variant: str,
    state_task_id: int,
    fit_outputs: Mapping[tuple[str, int], Path],
    grant: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    selected_dir = fit_outputs[(variant, state_task_id)] / "selected"
    selected = validate_selected_artifact(
        selected_dir, expected={"variant": variant, "task_id": state_task_id}
    )
    grant_key = (
        "selected_adapter_sha256_by_task"
        if variant == "lora"
        else "selected_capacity_upper_bound_sha256_by_task"
    )
    if grant[grant_key][str(state_task_id)] != selected["trainable_state_sha256"]:
        raise GateZeroOracleReportRuntimeError("selected state differs from freeze grant")
    return load_file(selected_dir / "trainable_state.safetensors"), selected


def _open_arm_runtime(
    *,
    task_id: int,
    condition: str,
    spec: dict[str, Any],
    parent: dict[str, Any],
    checkpoint: dict[str, Any],
    source_base_checkpoint: Path,
    fit_outputs: Mapping[tuple[str, int], Path],
    grant: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, torch.Tensor] | None, dict[str, Any]]:
    runtime = list(
        _load_policy(
            source_base_checkpoint / "pretrained_model",
            {"task_suite": "libero_90", "task_id": task_id},
        )
    )
    variant, state_task_id = report_state_authority(task_id, condition)
    evidence: dict[str, Any] = {"variant": variant, "state_task_id": state_task_id}
    state = None
    if variant is not None and state_task_id is not None:
        model, summary = configure_oracle_variant(
            runtime[0],
            parent=parent,
            checkpoint=checkpoint,
            variant=variant,
            variant_spec=spec["fit"][variant],
        )
        state, selected = _selected_state(
            variant=variant,
            state_task_id=state_task_id,
            fit_outputs=fit_outputs,
            grant=grant,
        )
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
    return tuple(runtime), state, evidence


def _offline_metrics(
    *,
    model: Any,
    preprocessor: Any,
    selected_state: dict[str, torch.Tensor] | None,
    task_id: int,
    verify_sha256: bool,
    spec: dict[str, Any],
    parent: dict[str, Any],
    phase0: dict[str, Any],
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    authorities, demos = load_surface_authorities(
        parent,
        phase0,
        manifest_path=arguments.manifest,
        dataset_root=arguments.dataset_root,
        surface=GateZeroSurface.REPORT,
        oracle_task_id=task_id,
        report_access_grant=arguments.selection_freeze_grant,
    )
    dataset = SourceHdf5Dataset(
        authorities,
        demo_indices=demos,
        action_chunk_size=parent["data"]["action_chunk_size"],
        verify_sha256=verify_sha256,
    )
    evaluator = FixedQueryEvaluator(
        dataset,
        preprocessor=preprocessor,
        batch_size=spec["report"]["offline_batch_size"],
        num_workers=spec["fit"]["num_workers"],
        anchor_count_per_demo=spec["selection"]["anchor_frames_per_demo"],
        action_chunk_size=parent["data"]["action_chunk_size"],
        fixed_noise_seed=spec["selection"]["fixed_noise_seed"],
        fixed_time_seed=spec["selection"]["fixed_time_seed"],
        inference_noise_seed=spec["report"]["policy_rng_seed"],
    )
    try:
        reference = evaluator.capture_base_reference(model)
        if selected_state is not None:
            restore_trainable_state(model, selected_state)
        candidate = evaluator.evaluate_candidate(model, reference, step=0)
    finally:
        evaluator.close()
    return {
        "offline_flow_mse": candidate["query_flow_mse"],
        "base_offline_flow_mse": reference.query_flow_mse,
        "offline_flow_reduction_fraction": (
            reference.query_flow_mse - candidate["query_flow_mse"]
        )
        / reference.query_flow_mse,
        "offline_sample_count": candidate["query_sample_count"],
        "offline_row_keys_sha256": candidate["query_row_keys_sha256"],
        "report_action_drift_proxy": candidate["action_drift_proxy"],
        "report_anchor_count": candidate["anchor_count"],
        "report_anchor_row_keys_sha256": candidate["anchor_row_keys_sha256"],
    }


def _evaluate_local_arms(
    *,
    context: ParallelContext,
    spec: dict[str, Any],
    parent: dict[str, Any],
    phase0: dict[str, Any],
    checkpoint: dict[str, Any],
    grant: dict[str, Any],
    arguments: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fit_outputs = _fit_outputs(arguments)
    local: list[dict[str, Any]] = []
    authorities: dict[int, dict[str, Any]] = {}
    verified_hdf5: set[int] = set()
    for task_id, condition in assigned_report_arms(
        rank=context.rank, world_size=context.world_size
    ):
        language, task_authority = _task_authority(
            task_id, spec["report"]["official_rollout_init_state_indices"]
        )
        authorities[task_id] = task_authority
        runtime, selected_state, state_evidence = _open_arm_runtime(
            task_id=task_id,
            condition=condition,
            spec=spec,
            parent=parent,
            checkpoint=checkpoint,
            source_base_checkpoint=arguments.source_base_checkpoint,
            fit_outputs=fit_outputs,
            grant=grant,
        )
        offline = _offline_metrics(
            model=runtime[0],
            preprocessor=runtime[1],
            selected_state=selected_state,
            task_id=task_id,
            verify_sha256=task_id not in verified_hdf5,
            spec=spec,
            parent=parent,
            phase0=phase0,
            arguments=arguments,
        )
        verified_hdf5.add(task_id)
        rollout = _closed_loop_metrics(
            runtime=runtime,
            task_id=task_id,
            condition=condition,
            language=language,
            spec=spec,
            output_dir=arguments.output_dir,
        )
        arm = {
            "task_id": task_id,
            "condition": condition,
            "state_authority": state_evidence,
            **offline,
            **rollout,
        }
        local.append(arm)
        print(
            json.dumps(
                {
                    "event": "gate_zero_locked_report_arm",
                    "rank": context.rank,
                    "task_id": task_id,
                    "condition": condition,
                    "successes": sum(arm["successes"]),
                    "episodes": len(arm["successes"]),
                    "offline_flow_mse": arm["offline_flow_mse"],
                    "mechanics_valid": arm["mechanics_valid"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del runtime, selected_state
        gc.collect()
        torch.cuda.empty_cache()
    return local, list(authorities.values())


def _ordered_arms(gathered: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    arms = [arm for values in gathered for arm in values]
    order = {
        arm: index
        for index, arm in enumerate(
            arm for shard in canonical_report_shards() for arm in shard
        )
    }
    arms.sort(key=lambda value: order[(value["task_id"], value["condition"])])
    return arms


def _eval_info(arms: list[dict[str, Any]], decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall": {
            "surface": "sealed_source_locked_report_tasks_3_4",
            "status": decision["status"],
            "episodes": sum(len(arm["successes"]) for arm in arms),
            "successes": sum(sum(arm["successes"]) for arm in arms),
            "writer_authorized": False,
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
                raise GateZeroOracleReportRuntimeError(
                    f"refusing non-fresh report output: {unexpected}"
                )
            import trackio

            trackio.init(
                project=spec["resources"]["tracking_project"],
                name=arguments.output_dir.name,
                group=spec["resources"]["tracking_group_report"],
                config={"world_size": context.world_size, "surface": "locked_source_report"},
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
        raise GateZeroOracleReportRuntimeError(error)
    return tracker


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
    selected_drift = {
        task_id: float(
            grant["fit_evidence"][f"lora:task{task_id}"]["selected_query_metrics"][
                "action_drift_proxy"
            ]
        )
        for task_id in spec["task_ids"]
    }
    decision = decide_gate_zero_report(
        arms=arms,
        selected_lora_drift=selected_drift,
        thresholds=spec["decision"],
    )
    comparison = {
        "own_vs_frozen_base": paired_gap_summary(
            arms,
            left="own_adapter",
            right="frozen_base",
            seed=spec["report"]["bootstrap_seed"],
            replicates=spec["report"]["bootstrap_replicates"],
        ),
        "own_vs_swapped_adapter": paired_gap_summary(
            arms,
            left="own_adapter",
            right="swapped_adapter",
            seed=spec["report"]["bootstrap_seed"] + 1,
            replicates=spec["report"]["bootstrap_replicates"],
        ),
    }
    result = {
        "schema_version": 1,
        "status": decision["status"],
        "surface": "sealed_source_locked_report_tasks_3_4",
        "execution_contract_sha256": sha256_file(arguments.config),
        "selection_freeze_grant_sha256": sha256_file(
            arguments.selection_freeze_grant
        ),
        "task_authorities": sorted(task_authorities, key=lambda value: value["task_id"]),
        "arms": arms,
        "paired_success_comparisons": comparison,
        "decision": decision,
        "interpretation": {
            "authorized_claim": "two-task source-only Gate 0 pilot evidence",
            "forbidden_claims": [
                "Writer authorization",
                "final Writer target support seal",
                "validation or held performance",
                "language or video Writer utility",
            ],
        },
        "parallel": {"world_size": context.world_size, "shards": canonical_report_shards()},
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
    atomic_json(arguments.output_dir / "oracle_report_result.json", result)
    atomic_json(arguments.output_dir / "eval_info.json", _eval_info(arms, decision))
    build_eval_gallery(arguments.output_dir)
    write_output_checksums(arguments.output_dir)
    update_latest_link(arguments.output_dir, arguments.latest_link)
    tracker.log(
        {
            "report/gate_zero_pilot_passed": int(decision["gate_zero_pilot_passed"]),
            "report/median_success_gain_pp": decision["aggregate"][
                "median_success_gain_pp"
            ],
            "report/median_locked_action_loss_reduction_fraction": decision[
                "aggregate"
            ]["median_locked_action_loss_reduction_fraction"],
        }
    )
    tracker.finish()
    return result


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    context = _initialize_parallel()
    tracker = None
    try:
        spec, parent, phase0, checkpoint, grant = _load_authorities(arguments)
        tracker = _prepare_output(context, arguments, spec)
        started = time.perf_counter()
        local_arms, local_authorities = _evaluate_local_arms(
            context=context,
            spec=spec,
            parent=parent,
            phase0=phase0,
            checkpoint=checkpoint,
            grant=grant,
            arguments=arguments,
        )
        gathered_arms = _gather(context, local_arms)
        gathered_authorities = _gather(context, local_authorities)
        if not context.is_primary:
            return {"status": "non_primary_rank_complete", "rank": context.rank}
        arms = _ordered_arms(gathered_arms)
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-base-checkpoint", type=Path, required=True)
    parser.add_argument("--selection-freeze-grant", type=Path, required=True)
    parser.add_argument("--lora-task3", type=Path, required=True)
    parser.add_argument("--lora-task4", type=Path, required=True)
    parser.add_argument("--partial-task3", type=Path, required=True)
    parser.add_argument("--partial-task4", type=Path, required=True)
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
