"""Canonical source-base trainer and exact checkpoint/resume probe for Gate 0."""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path
from typing import Any

from ember.gate_zero_base_probe import run_resume_probe, run_topology_probe
from ember.gate_zero_base_session import (
    GateZeroBaseTrainError,
    TrainingRuntime as _TrainingRuntime,
    assert_exact_resume_equivalence,
    build_source_base_checkpoint_metadata,
    canonical_state_sha256,
    checkpoint_evidence as _checkpoint_evidence,
    close_training_runtime as _close_runtime,
    initialize_tracking as _initialize_tracking,
    log_training_progress as _log_training_progress,
    new_training_runtime as _new_training_runtime,
    require_base_fit_authorization,
    run_one_step as _run_one_step,
    save_runtime_checkpoint as _save_runtime_checkpoint,
    should_log_training_step,
    validate_output as _validate_output,
    write_result as _write_result,
)
from ember.gate_zero_checkpoint import rotate_source_base_recovery_checkpoints
from ember.gate_zero_contract import load_gate_zero_contract
from ember.gate_zero_distributed import (
    DistributedContext,
    TrainingTopology,
    broadcast_primary_error,
    close_distributed_context,
    distributed_barrier,
    initialize_distributed_context,
    load_distributed_topology_spec,
    require_topology_mode_authorization,
    topology_for_world_size,
)


def _commit_training_checkpoint(
    runtime: _TrainingRuntime, args: argparse.Namespace, spec: dict[str, Any]
) -> None:
    checkpoint_dir = args.output_dir / "checkpoints" / f"{runtime.completed_step:06d}"
    _save_runtime_checkpoint(runtime, args, spec, checkpoint_dir)
    removed = None
    primary_error: BaseException | None = None
    if runtime.distributed_context.is_primary:
        try:
            removed = rotate_source_base_recovery_checkpoints(
                checkpoint_dir.parent,
                keep=spec["base_fit"]["recoverable_checkpoints_to_keep"],
            )
            print(
                json.dumps(
                    {
                        "event": "source_base_checkpoint_committed",
                        "completed_step": runtime.completed_step,
                        "world_size": runtime.topology.world_size,
                        "removed_recovery_steps": removed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except BaseException as error:
            primary_error = error
    broadcast_primary_error(runtime.distributed_context, primary_error)


def _build_training_result(
    args: argparse.Namespace,
    spec: dict[str, Any],
    *,
    target_step: int,
    started: float,
    topology: TrainingTopology,
) -> dict[str, Any]:
    final_checkpoint = args.output_dir / "checkpoints" / f"{target_step:06d}"
    return {
        "schema_version": 1,
        "status": "source_base_fit_completed_pending_competence",
        "completed_steps": target_step,
        "effective_batch_size": spec["base_fit"]["effective_batch_size"],
        "world_size": topology.world_size,
        "topology": topology.as_manifest(),
        "source_policy_outcome_recorded": False,
        "gate_zero_authorized": False,
        "writer_authorized": False,
        "authorities": build_source_base_checkpoint_metadata(
            spec,
            config_path=args.config,
            phase0_path=args.phase0_contract,
            completed_step=target_step,
            topology_config_path=args.topology_config,
            topology=topology,
        )["authorities"],
        "final_checkpoint": _checkpoint_evidence(final_checkpoint),
        "wall_seconds": time.perf_counter() - started,
        "tracking": {
            "backend": "trackio",
            "project": spec["tracking"]["project"],
            "run": args.output_dir.name,
            "dashboard_command": spec["tracking"]["dashboard_command"],
        },
    }


def run_training(
    args: argparse.Namespace,
    context: DistributedContext,
    topology: TrainingTopology,
) -> dict[str, Any]:
    _validate_output(args, result_name="training_result.json", context=context)
    spec = load_gate_zero_contract(args.config, args.phase0_contract)
    phase0 = tomllib.loads(args.phase0_contract.read_text(encoding="utf-8"))
    require_base_fit_authorization(spec, mode="train")
    target_step = spec["base_fit"]["steps"]
    started = time.perf_counter()
    trackio = _initialize_tracking(spec, args, context, topology)
    runtime: _TrainingRuntime | None = None
    last_record: dict[str, Any] | None = None
    try:
        runtime = _new_training_runtime(
            args,
            spec,
            phase0,
            target_step=target_step,
            topology=topology,
            distributed_context=context,
            resume_from=args.resume_from,
        )
        while runtime.completed_step < target_step:
            last_record = _run_one_step(runtime, spec)
            _log_training_progress(trackio, runtime, last_record, spec, target_step=target_step)
            checkpoint_due = (
                runtime.completed_step % spec["base_fit"]["checkpoint_every_steps"] == 0
                or runtime.completed_step == target_step
            )
            if checkpoint_due:
                _commit_training_checkpoint(runtime, args, spec)
        if last_record is None:
            raise GateZeroBaseTrainError("formal base fit performed no optimizer step")
        result = None
        if context.is_primary:
            result = _build_training_result(
                args,
                spec,
                target_step=target_step,
                started=started,
                topology=topology,
            )
            trackio.finish()
            _write_result(
                result,
                args,
                result_name="training_result.json",
                context=context,
            )
        distributed_barrier(context)
        if context.is_primary:
            return result
        return {"status": "non_primary_rank_complete", "rank": context.rank}
    except BaseException:
        if trackio is not None:
            trackio.finish()
        raise
    finally:
        if runtime is not None:
            _close_runtime(runtime)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("resume-probe", "topology-probe", "train"), required=True
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase0-contract", type=Path, required=True)
    parser.add_argument("--topology-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latest-link", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.output_dir = args.output_dir.absolute()
    args.latest_link = args.latest_link.absolute()
    topology_spec = load_distributed_topology_spec(
        args.topology_config, args.config, args.phase0_contract
    )
    context = initialize_distributed_context(topology_spec)
    topology = topology_for_world_size(topology_spec, context.world_size)
    require_topology_mode_authorization(
        topology_spec, mode=args.mode, world_size=context.world_size
    )
    try:
        if args.mode == "resume-probe":
            result = run_resume_probe(args, context, topology)
        elif args.mode == "topology-probe":
            result = run_topology_probe(
                args, context, topology, topology_spec
            )
        else:
            result = run_training(args, context, topology)
        if context.is_primary:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    finally:
        close_distributed_context(context)


if __name__ == "__main__":
    raise SystemExit(main())
