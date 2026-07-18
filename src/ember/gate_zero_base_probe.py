"""Matched resume and 1/2/4-GPU topology probes for the canonical Gate 0 trainer."""

from __future__ import annotations

import argparse
import shutil
import time
import tomllib
from pathlib import Path
from typing import Any

import torch

from ember.gate_zero_base_session import (
    GateZeroBaseTrainError,
    TrainingRuntime,
    assert_exact_resume_equivalence,
    build_source_base_checkpoint_metadata,
    canonical_state_sha256,
    capture_runtime_authority,
    checkpoint_evidence,
    close_training_runtime,
    initialize_tracking,
    log_training_progress,
    new_training_runtime,
    require_base_fit_authorization,
    run_one_step,
    save_runtime_checkpoint,
    validate_output,
    write_result,
)
from ember.gate_zero_checkpoint import validate_source_base_checkpoint
from ember.gate_zero_contract import load_gate_zero_contract
from ember.gate_zero_distributed import (
    DistributedContext,
    TrainingTopology,
    assert_same_topology,
    broadcast_primary_error,
    broadcast_primary_object,
    distributed_max,
    gather_rank_objects,
)
from ember.gate_zero_runtime import sha256_file


def _cleanup_probe_checkpoint(checkpoint_dir: Path) -> None:
    validate_source_base_checkpoint(checkpoint_dir)
    checkpoint_root = checkpoint_dir.parent
    sidecar = checkpoint_root / f"{checkpoint_dir.name}.manifest.sha256"
    last = checkpoint_root / "last"
    if last.is_symlink():
        last.unlink()
    shutil.rmtree(checkpoint_dir)
    sidecar.unlink()
    if any(checkpoint_root.iterdir()):
        raise GateZeroBaseTrainError("unexpected files prevent transient checkpoint cleanup")
    checkpoint_root.rmdir()


def _advance_runtime(
    runtime: TrainingRuntime, spec: dict[str, Any], *, target_step: int
) -> None:
    while runtime.completed_step < target_step:
        run_one_step(runtime, spec)


def _probe_runtime_authority(
    args: argparse.Namespace,
    spec: dict[str, Any],
    phase0: dict[str, Any],
    *,
    target_step: int,
    topology: TrainingTopology,
    context: DistributedContext,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    runtime = new_training_runtime(
        args,
        spec,
        phase0,
        target_step=target_step,
        topology=topology,
        distributed_context=context,
        resume_from=resume_from,
    )
    try:
        _advance_runtime(runtime, spec, target_step=target_step)
        return capture_runtime_authority(runtime, spec)
    finally:
        close_training_runtime(runtime)


def _create_probe_checkpoint(
    args: argparse.Namespace,
    spec: dict[str, Any],
    phase0: dict[str, Any],
    *,
    checkpoint_step: int,
    topology: TrainingTopology,
    context: DistributedContext,
) -> tuple[Path, dict[str, Any]]:
    runtime = new_training_runtime(
        args,
        spec,
        phase0,
        target_step=checkpoint_step,
        topology=topology,
        distributed_context=context,
    )
    try:
        _advance_runtime(runtime, spec, target_step=checkpoint_step)
        checkpoint_dir = args.output_dir / "checkpoints" / f"{checkpoint_step:06d}"
        save_runtime_checkpoint(runtime, args, spec, checkpoint_dir)
        evidence = checkpoint_evidence(checkpoint_dir) if context.is_primary else None
        return checkpoint_dir, broadcast_primary_object(context, evidence)
    finally:
        close_training_runtime(runtime)


def _verify_probe_resume_and_cleanup(
    uninterrupted: dict[str, Any],
    resumed: dict[str, Any],
    checkpoint_dir: Path,
    context: DistributedContext,
) -> dict[str, Any]:
    comparison = None
    primary_error: BaseException | None = None
    if context.is_primary:
        try:
            comparison = assert_exact_resume_equivalence(uninterrupted, resumed)
            _cleanup_probe_checkpoint(checkpoint_dir)
        except BaseException as error:
            primary_error = error
    broadcast_primary_error(context, primary_error)
    return broadcast_primary_object(context, comparison)


def _resume_probe_result(
    args: argparse.Namespace,
    spec: dict[str, Any],
    topology: TrainingTopology,
    *,
    checkpoint_step: int,
    uninterrupted: dict[str, Any],
    resumed: dict[str, Any],
    comparison: dict[str, Any],
    checkpoint_evidence_value: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    metadata = build_source_base_checkpoint_metadata(
        spec,
        config_path=args.config,
        phase0_path=args.phase0_contract,
        completed_step=checkpoint_step,
        topology_config_path=args.topology_config,
        topology=topology,
    )
    return {
        "schema_version": 1,
        "status": "resume_probe_passed",
        "scientific_outcome_metrics_recorded": False,
        "source_policy_outcome_recorded": False,
        "scientific_gate_decision_authorized": False,
        "writer_authorized": False,
        "gpu_count": topology.world_size,
        "authorities": metadata["authorities"],
        "topology": metadata["topology"],
        "uninterrupted": uninterrupted,
        "resumed": resumed,
        "comparison": comparison,
        "checkpoint_evidence": checkpoint_evidence_value,
        "transient_full_checkpoint_cleaned": True,
        "wall_seconds": time.perf_counter() - started,
        "tracking": {
            "backend": "trackio",
            "project": spec["tracking"]["project"],
            "run": args.output_dir.name,
            "dashboard_command": spec["tracking"]["dashboard_command"],
        },
    }


def run_resume_probe(
    args: argparse.Namespace,
    context: DistributedContext,
    topology: TrainingTopology,
) -> dict[str, Any]:
    if args.resume_from is not None:
        raise GateZeroBaseTrainError("resume probe does not accept an external checkpoint")
    validate_output(args, result_name="resume_probe_result.json", context=context)
    spec = load_gate_zero_contract(args.config, args.phase0_contract)
    phase0 = tomllib.loads(args.phase0_contract.read_text(encoding="utf-8"))
    require_base_fit_authorization(spec, mode="resume-probe")
    probe = spec["base_fit"]["resume_probe"]
    target_step = probe["uninterrupted_target_step"]
    checkpoint_step = probe["interrupted_checkpoint_step"]
    if target_step != 2 or checkpoint_step != 1:
        raise GateZeroBaseTrainError("resume probe step contract changed")
    started = time.perf_counter()
    trackio = initialize_tracking(spec, args, context, topology)
    try:
        uninterrupted = _probe_runtime_authority(
            args, spec, phase0, target_step=target_step, topology=topology, context=context
        )
        checkpoint_dir, checkpoint_evidence_value = _create_probe_checkpoint(
            args,
            spec,
            phase0,
            checkpoint_step=checkpoint_step,
            topology=topology,
            context=context,
        )
        resumed = _probe_runtime_authority(
            args,
            spec,
            phase0,
            target_step=target_step,
            topology=topology,
            context=context,
            resume_from=checkpoint_dir,
        )
        expected_topology = build_source_base_checkpoint_metadata(
            spec,
            config_path=args.config,
            phase0_path=args.phase0_contract,
            completed_step=checkpoint_step,
            topology_config_path=args.topology_config,
            topology=topology,
        )["topology"]
        assert_same_topology(expected_topology, checkpoint_evidence_value["topology"])
        comparison = _verify_probe_resume_and_cleanup(
            uninterrupted, resumed, checkpoint_dir, context
        )
        result = _resume_probe_result(
            args,
            spec,
            topology,
            checkpoint_step=checkpoint_step,
            uninterrupted=uninterrupted,
            resumed=resumed,
            comparison=comparison,
            checkpoint_evidence_value=checkpoint_evidence_value,
            started=started,
        )
        if context.is_primary:
            trackio.log(
                {f"resume/{key}": int(value) for key, value in comparison["surfaces"].items()}
            )
            trackio.log({"resume/all_exact": 1})
            trackio.finish()
            write_result(result, args, result_name="resume_probe_result.json", context=context)
            return result
        return {"status": "non_primary_rank_complete", "rank": context.rank}
    except BaseException:
        if trackio is not None:
            trackio.finish()
        raise


def _run_topology_uninterrupted_path(
    args: argparse.Namespace,
    spec: dict[str, Any],
    phase0: dict[str, Any],
    *,
    checkpoint_step: int,
    target_step: int,
    topology: TrainingTopology,
    context: DistributedContext,
    trackio: Any,
) -> dict[str, Any]:
    runtime: TrainingRuntime | None = None
    try:
        startup_started = time.perf_counter()
        runtime = new_training_runtime(
            args,
            spec,
            phase0,
            target_step=target_step,
            topology=topology,
            distributed_context=context,
        )
        device = next(runtime.base_policy.parameters()).device
        startup_seconds = distributed_max(
            time.perf_counter() - startup_started, context, device=device
        )
        initial_authority = _capture_initial_authority(runtime, context)
        records: list[dict[str, Any]] = []
        while runtime.completed_step < checkpoint_step:
            record = run_one_step(runtime, spec)
            records.append(dict(record))
            log_training_progress(trackio, runtime, record, spec, target_step=checkpoint_step)
        checkpoint_dir = args.output_dir / "checkpoints" / f"{checkpoint_step:06d}"
        checkpoint_started = time.perf_counter()
        save_runtime_checkpoint(runtime, args, spec, checkpoint_dir)
        checkpoint_seconds = distributed_max(
            time.perf_counter() - checkpoint_started, context, device=device
        )
        evidence = checkpoint_evidence(checkpoint_dir) if context.is_primary else None
        evidence = broadcast_primary_object(context, evidence)
        expected = build_source_base_checkpoint_metadata(
            spec,
            config_path=args.config,
            phase0_path=args.phase0_contract,
            completed_step=checkpoint_step,
            topology_config_path=args.topology_config,
            topology=topology,
        )["topology"]
        assert_same_topology(expected, evidence["topology"])
        run_one_step(runtime, spec)
        return {
            "startup_seconds": startup_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_evidence": evidence,
            "initial_authority": initial_authority,
            "records": records,
            "uninterrupted": capture_runtime_authority(runtime, spec),
        }
    finally:
        if runtime is not None:
            close_training_runtime(runtime)


def _capture_initial_authority(
    runtime: TrainingRuntime, context: DistributedContext
) -> dict[str, Any]:
    authority = None
    primary_error: BaseException | None = None
    if context.is_primary:
        try:
            if runtime.completed_step != 0:
                raise GateZeroBaseTrainError("topology probe did not start at step zero")
            authority = {
                "initial_model_state_sha256": canonical_state_sha256(
                    runtime.base_policy.state_dict()
                ),
                "initial_optimizer_state_sha256": canonical_state_sha256(
                    runtime.optimizer.state_dict()
                ),
                "initial_scheduler_state_sha256": canonical_state_sha256(
                    runtime.scheduler.state_dict()
                ),
            }
        except BaseException as error:
            primary_error = error
    broadcast_primary_error(context, primary_error)
    return broadcast_primary_object(context, authority)


def _run_topology_resumed_path(
    args: argparse.Namespace,
    spec: dict[str, Any],
    phase0: dict[str, Any],
    *,
    target_step: int,
    topology: TrainingTopology,
    context: DistributedContext,
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], float]:
    runtime: TrainingRuntime | None = None
    try:
        started = time.perf_counter()
        runtime = new_training_runtime(
            args,
            spec,
            phase0,
            target_step=target_step,
            topology=topology,
            distributed_context=context,
            resume_from=checkpoint_dir,
        )
        resume_seconds = distributed_max(
            time.perf_counter() - started,
            context,
            device=next(runtime.base_policy.parameters()).device,
        )
        run_one_step(runtime, spec)
        return capture_runtime_authority(runtime, spec), resume_seconds
    finally:
        if runtime is not None:
            close_training_runtime(runtime)


def _topology_probe_measurement(
    probe: dict[str, Any],
    records: list[dict[str, Any]],
    topology: TrainingTopology,
    context: DistributedContext,
    *,
    initial_authority: dict[str, Any],
    startup_seconds: float,
    checkpoint_seconds: float,
    resume_startup_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    measured = records[probe["warmup_optimizer_steps"] :]
    if len(measured) != probe["measured_optimizer_steps"]:
        raise GateZeroBaseTrainError("topology probe measurement window changed")
    measured_seconds = sum(record["wall_seconds"] for record in measured)
    throughput = topology.global_effective_batch_size * len(measured) / measured_seconds
    local_peak_mib = (
        int(torch.cuda.max_memory_allocated() // (1024 * 1024))
        if torch.cuda.is_available()
        else 0
    )
    measurement = {
        "global_effective_samples_per_second": throughput,
        "measured_wall_seconds": measured_seconds,
        "startup_wall_seconds": startup_seconds,
        "checkpoint_wall_seconds": checkpoint_seconds,
        "resume_startup_wall_seconds": resume_startup_seconds,
        "per_rank_peak_allocated_memory_mib": gather_rank_objects(local_peak_mib, context),
        "step_wall_seconds": [record["wall_seconds"] for record in measured],
    }
    authority = {
        **initial_authority,
        "row_keys_sha256_by_step": [record["row_keys_sha256"] for record in records],
        "flow_rng_state_sha256_by_step": [
            record["flow_rng_state_sha256"] for record in records
        ],
        "flow_input_sha256_by_step": [
            record["flow_input_sha256"] for record in records
        ],
        "global_slot_count_by_step": [record["global_slot_count"] for record in records],
        "unique_global_slot_count_by_step": [
            record["unique_global_slot_count"] for record in records
        ],
    }
    return measurement, authority


def _topology_probe_result(
    args: argparse.Namespace,
    spec: dict[str, Any],
    topology: TrainingTopology,
    probe: dict[str, Any],
    *,
    measurement: dict[str, Any],
    authority: dict[str, Any],
    comparison: dict[str, Any],
    checkpoint_evidence_value: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "topology_probe_completed_pending_cross_topology_selection",
        "scientific_outcome_metrics_recorded": False,
        "source_policy_outcome_recorded": False,
        "gate_zero_authorized": False,
        "writer_authorized": False,
        "topology_config_sha256": sha256_file(args.topology_config),
        "topology": topology.as_manifest(),
        "probe_budget": {
            "warmup_optimizer_steps": probe["warmup_optimizer_steps"],
            "measured_optimizer_steps": probe["measured_optimizer_steps"],
            "checkpoint_optimizer_step": probe["checkpoint_after_optimizer_steps"],
            "resume_validation_optimizer_steps": 1,
            "global_effective_batch_size": topology.global_effective_batch_size,
        },
        "measurement": measurement,
        "global_authority": authority,
        "same_topology_resume": comparison,
        "checkpoint_evidence": checkpoint_evidence_value,
        "transient_full_checkpoint_cleaned": True,
        "wall_seconds": time.perf_counter() - started,
        "tracking": {
            "backend": "trackio",
            "project": spec["tracking"]["project"],
            "run": args.output_dir.name,
            "dashboard_command": spec["tracking"]["dashboard_command"],
        },
    }


def run_topology_probe(
    args: argparse.Namespace,
    context: DistributedContext,
    topology: TrainingTopology,
    topology_spec: dict[str, Any],
) -> dict[str, Any]:
    """Measure one frozen topology and prove exact same-topology interruption recovery."""

    if args.resume_from is not None:
        raise GateZeroBaseTrainError("topology probe owns its transient checkpoint")
    validate_output(args, result_name="topology_probe_result.json", context=context)
    spec = load_gate_zero_contract(args.config, args.phase0_contract)
    phase0 = tomllib.loads(args.phase0_contract.read_text(encoding="utf-8"))
    require_base_fit_authorization(spec, mode="topology-probe")
    probe = topology_spec["probe"]
    checkpoint_step = probe["checkpoint_after_optimizer_steps"]
    target_step = checkpoint_step + probe["resume_validation_optimizer_steps"]
    if checkpoint_step != probe["warmup_optimizer_steps"] + probe["measured_optimizer_steps"]:
        raise GateZeroBaseTrainError("topology probe step accounting changed")
    if target_step != checkpoint_step + 1:
        raise GateZeroBaseTrainError("topology probe requires one resumed validation step")
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trackio = initialize_tracking(spec, args, context, topology)
    try:
        path = _run_topology_uninterrupted_path(
            args,
            spec,
            phase0,
            checkpoint_step=checkpoint_step,
            target_step=target_step,
            topology=topology,
            context=context,
            trackio=trackio,
        )
        resumed, resume_seconds = _run_topology_resumed_path(
            args,
            spec,
            phase0,
            target_step=target_step,
            topology=topology,
            context=context,
            checkpoint_dir=path["checkpoint_dir"],
        )
        comparison = _verify_probe_resume_and_cleanup(
            path["uninterrupted"], resumed, path["checkpoint_dir"], context
        )
        measurement, authority = _topology_probe_measurement(
            probe,
            path["records"],
            topology,
            context,
            initial_authority=path["initial_authority"],
            startup_seconds=path["startup_seconds"],
            checkpoint_seconds=path["checkpoint_seconds"],
            resume_startup_seconds=resume_seconds,
        )
        result = _topology_probe_result(
            args,
            spec,
            topology,
            probe,
            measurement=measurement,
            authority=authority,
            comparison=comparison,
            checkpoint_evidence_value=path["checkpoint_evidence"],
            started=started,
        )
        if context.is_primary:
            trackio.log(
                {
                    "topology/world_size": topology.world_size,
                    "topology/global_samples_per_second": measurement[
                        "global_effective_samples_per_second"
                    ],
                    "topology/checkpoint_seconds": path["checkpoint_seconds"],
                    "topology/resume_exact": 1,
                },
                step=checkpoint_step,
            )
            trackio.finish()
            write_result(result, args, result_name="topology_probe_result.json", context=context)
            return result
        return {"status": "non_primary_rank_complete", "rank": context.rank}
    except BaseException:
        if trackio is not None:
            trackio.finish()
        raise
