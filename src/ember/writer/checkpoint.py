"""Atomic exact-resume checkpoints for PI05 Action-Supervised Writer training."""

from __future__ import annotations

import json
import math
import os
import random
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from ember.pi05_source_checkpoint import (
    DistributedContext,
    canonical_hash,
    read_json,
    restore_rng,
    sha256_file,
    source_reference_matches,
    write_json_atomic,
)
from ember.writer.as_sampling import (
    MixedTaskBatchSampler,
    TeacherVideoSchedule,
)
from ember.writer.as_contract import AS_WRITER_LAUNCH_SCHEMA
from ember.writer.checkpoint_schema import (
    AS_WRITER_CHECKPOINT_SCHEMA,
    AS_WRITER_CYCLE_NORMALIZED_GROUP4_CHECKPOINT_SCHEMA,
    AS_WRITER_CYCLE_NORMALIZED_GROUP4_RANK_STATE_SCHEMA,
    AS_WRITER_RANK_STATE_SCHEMA,
    AS_WRITER_SERIAL4_CHECKPOINT_SCHEMA,
    AS_WRITER_SERIAL4_RANK_STATE_SCHEMA,
    HISTORICAL_V6_LAUNCH_SCHEMA,
    checkpoint_schema_matches,
    state_schemas as _state_schemas,
)
from ember.writer.model import CompleteLoRAWriter, WriterModelError
from ember.writer.update_schedule import cycle_matched_weight_decay


_HASHLESS_CHECKPOINT_FAMILIES = frozenset(
    {
        "k4_energy_preserving_policy_layer_trace_m2p_full24_v1",
        "k4_evidence_factorized_policy_layer_trace_m2p_full24_v1",
        "k4_sparse_semantic_expert_policy_layer_trace_m2p_full24_v1",
    }
)


def _scheduler_update_cursor(task_cycle: int, checkpoint_state_family: str) -> int:
    return (
        min(task_cycle, 50)
        if checkpoint_state_family
        == "condition_kernel_program_memory_full24_v1"
        else task_cycle
    )


def _rng_state(context: DistributedContext) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(context.device),
    }


def _checkpoint_nonce(context: DistributedContext) -> str:
    nonce = uuid.uuid4().hex
    if context.world_size == 1:
        return nonce
    encoded = torch.zeros(16, dtype=torch.uint8, device=context.device)
    if context.is_main:
        encoded.copy_(
            torch.tensor(
                list(bytes.fromhex(nonce)), dtype=torch.uint8, device=context.device
            )
        )
    dist.broadcast(encoded, src=0)
    return bytes(encoded.cpu().tolist()).hex()


def _raise_distributed_errors(
    context: DistributedContext, phase: str, error: Exception | None
) -> None:
    local = None if error is None else repr(error)
    failures: list[str | None] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(failures, local)
    else:
        failures[0] = local
    observed = [f"rank {rank}: {value}" for rank, value in enumerate(failures) if value]
    if observed:
        raise WriterModelError(f"AS-Writer checkpoint {phase} failed; " + "; ".join(observed))


def _write_rank_state(
    path: Path,
    *,
    step: int,
    context: DistributedContext,
    sampler: MixedTaskBatchSampler,
    video_schedule: TeacherVideoSchedule,
    saved_rng: Mapping[str, Any],
    videos_per_task_visit: int,
    tasks_per_rank_per_update: int,
    optimizer_updates_per_task_cycle: int,
    rank_state_schema: str,
    checkpoint_state_family: str | None = None,
) -> None:
    task_cycle, task_cycle_phase = divmod(
        step, optimizer_updates_per_task_cycle
    )
    resolved_family = checkpoint_state_family or (
        "cvadr_legacy_full24_v1"
        if optimizer_updates_per_task_cycle == 1
        else "cvadr_legacy_serial4_v1"
    )
    torch.save(
        {
            "schema_version": rank_state_schema,
            "next_step": step,
            "next_data_step": step * tasks_per_rank_per_update,
            "next_task_cycle": task_cycle,
            "next_task_cycle_phase": task_cycle_phase,
            "scheduler_logical_updates": _scheduler_update_cursor(
                task_cycle, resolved_family
            ),
            "rank": context.rank,
            "world_size": context.world_size,
            "per_rank_batch_size": sampler.per_rank_batch_size,
            "per_rank_batch_cycle": sampler.per_rank_batch_cycle,
            "sampler_seed": sampler.seed,
            "teacher_video_seed": video_schedule.seed,
            "teacher_videos_per_task_visit": videos_per_task_visit,
            "tasks_per_rank_per_optimizer_update": tasks_per_rank_per_update,
            "optimizer_updates_per_task_cycle": optimizer_updates_per_task_cycle,
            "checkpoint_state_family": resolved_family,
            "rng": saved_rng,
        },
        path,
    )


def _write_shared_state(
    temporary: Path,
    final: Path,
    *,
    output_dir: Path,
    step: int,
    writer: CompleteLoRAWriter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler: MixedTaskBatchSampler,
    video_schedule: TeacherVideoSchedule,
    contract: Mapping[str, Any],
    require_full_coverage: bool,
    metrics_rows: int,
) -> dict[str, Any]:
    videos_per_task_visit = int(
        contract["runtime"]["teacher_videos_per_task_visit"]
    )
    tasks_per_rank_per_update = int(
        contract["runtime"]["tasks_per_rank_per_optimizer_update"]
    )
    optimizer_updates_per_task_cycle = int(
        contract["runtime"].get("optimizer_updates_per_task_cycle", 1)
    )
    checkpoint_state_family = str(
        contract["runtime"].get(
            "checkpoint_state_family",
            (
                "cvadr_legacy_full24_v1"
                if optimizer_updates_per_task_cycle == 1
                else "cvadr_legacy_serial4_v1"
            ),
        )
    )
    task_cycle, task_cycle_phase = divmod(
        step, optimizer_updates_per_task_cycle
    )
    scheduler_cursor = _scheduler_update_cursor(
        task_cycle, checkpoint_state_family
    )
    checkpoint_schema, trainer_state_schema, _ = _state_schemas(
        optimizer_updates_per_task_cycle,
        checkpoint_state_family,
    )
    scheduler_state = scheduler.state_dict()
    hashless = checkpoint_state_family in _HASHLESS_CHECKPOINT_FAMILIES
    if int(scheduler_state.get("last_epoch", -1)) != scheduler_cursor:
        raise WriterModelError("AS-Writer scheduler exposure cursor changed")
    data_stop_step = step
    save_file(
        {
            name: value.detach().to(device="cpu").contiguous()
            for name, value in writer.state_dict().items()
        },
        str(temporary / "writer.safetensors"),
    )
    torch.save(
        {
            "schema_version": trainer_state_schema,
            "next_step": step,
            "next_task_cycle": task_cycle,
            "next_task_cycle_phase": task_cycle_phase,
            "scheduler_logical_updates": scheduler_cursor,
            "checkpoint_state_family": checkpoint_state_family,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler_state,
            "amp_scaler": {"enabled": False, "state": {}},
            "contract_reference": (
                str(contract["schema_version"])
                if hashless
                else canonical_hash(contract)
            ),
            "metrics_rows": metrics_rows,
        },
        temporary / "trainer_state.pt",
    )
    coverage = sampler.coverage_for_steps(0, data_stop_step)
    schedule = video_schedule.consumed_identity_summary(
        sampler,
        0,
        data_stop_step,
    )
    if require_full_coverage and (
        any(len(episodes) != sampler.episodes_per_task for episodes in coverage.values())
        or schedule["min_unique_videos_per_task"] != len(video_schedule.demo_indices)
    ):
        raise WriterModelError("final AS-Writer checkpoint lacks full data coverage")
    consumed = {
        **schedule,
        "declared_task_count": len(coverage),
        "tasks_with_action_signal": sum(bool(episodes) for episodes in coverage.values()),
        "min_action_episodes_per_task": min(map(len, coverage.values())),
        "max_action_episodes_per_task": max(map(len, coverage.values())),
        "next_step": step,
        "next_data_step": step * tasks_per_rank_per_update,
        "next_task_cycle": task_cycle,
        "next_task_cycle_phase": task_cycle_phase,
        "scheduler_logical_updates": scheduler_cursor,
        "optimizer_updates_per_task_cycle": optimizer_updates_per_task_cycle,
        "checkpoint_state_family": checkpoint_state_family,
        "teacher_videos_per_task_visit": videos_per_task_visit,
    }
    files = {
        str(path.relative_to(temporary)): {"bytes": path.stat().st_size}
        for path in sorted(value for value in temporary.rglob("*") if value.is_file())
    }
    manifest = {
        "schema_version": checkpoint_schema,
        "contract_reference": (
            str(contract["schema_version"])
            if hashless
            else canonical_hash(contract)
        ),
        "consumed": consumed,
        "files": files,
    }
    if not hashless:
        manifest["canonical_payload_sha256"] = canonical_hash(manifest)
        for relative, record in files.items():
            record["sha256"] = sha256_file(temporary / relative)
    write_json_atomic(temporary / "checkpoint_manifest.json", manifest)
    os.replace(temporary, final)
    write_json_atomic(output_dir / "latest_checkpoint.json", {"path": str(final), "step": step})
    return consumed


def save_writer_checkpoint(
    *,
    output_dir: Path,
    step: int,
    context: DistributedContext,
    writer: CompleteLoRAWriter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler: MixedTaskBatchSampler,
    video_schedule: TeacherVideoSchedule,
    contract: dict[str, Any],
    mode: str,
    metrics_rows: int,
) -> Path:
    total_steps = int(contract.get("runtime", {}).get("total_steps", -1))
    if mode not in {"profile", "formal"} or not 0 < step <= total_steps:
        raise WriterModelError("AS-Writer checkpoint step is outside its launch contract")
    require_full_coverage = mode == "formal" and step == total_steps
    nonce = _checkpoint_nonce(context)
    temporary = output_dir / "checkpoints" / f".step_{step:08d}.{nonce}.partial"
    final = output_dir / "checkpoints" / f"step_{step:08d}"
    error: Exception | None = None
    try:
        if context.is_main:
            if final.exists():
                raise WriterModelError(f"Writer checkpoint already exists: {final}")
            temporary.mkdir(parents=True)
    except Exception as caught:
        error = caught
    _raise_distributed_errors(context, "initialization", error)

    saved_rng = _rng_state(context)
    videos_per_task_visit = int(
        contract["runtime"]["teacher_videos_per_task_visit"]
    )
    tasks_per_rank_per_update = int(
        contract["runtime"]["tasks_per_rank_per_optimizer_update"]
    )
    optimizer_updates_per_task_cycle = int(
        contract["runtime"].get("optimizer_updates_per_task_cycle", 1)
    )
    checkpoint_state_family = str(
        contract["runtime"].get(
            "checkpoint_state_family",
            (
                "cvadr_legacy_full24_v1"
                if optimizer_updates_per_task_cycle == 1
                else "cvadr_legacy_serial4_v1"
            ),
        )
    )
    _, _, rank_state_schema = _state_schemas(
        optimizer_updates_per_task_cycle,
        checkpoint_state_family,
    )
    error = None
    try:
        _write_rank_state(
            temporary / f"rank_{context.rank:02d}_state.pt",
            step=step,
            context=context,
            sampler=sampler,
            video_schedule=video_schedule,
            saved_rng=saved_rng,
            videos_per_task_visit=videos_per_task_visit,
            tasks_per_rank_per_update=tasks_per_rank_per_update,
            optimizer_updates_per_task_cycle=optimizer_updates_per_task_cycle,
            rank_state_schema=rank_state_schema,
            checkpoint_state_family=checkpoint_state_family,
        )
    except Exception as caught:
        error = caught
    _raise_distributed_errors(context, "rank-state write", error)

    error = None
    try:
        if context.is_main:
            consumed = _write_shared_state(
                temporary,
                final,
                output_dir=output_dir,
                step=step,
                writer=writer,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                video_schedule=video_schedule,
                contract=contract,
                require_full_coverage=require_full_coverage,
                metrics_rows=metrics_rows,
            )
            print(
                json.dumps({"event": "checkpoint", "path": str(final), **consumed}),
                flush=True,
            )
    except Exception as caught:
        error = caught
    _raise_distributed_errors(context, "publication", error)
    restore_rng(saved_rng, context)
    return final


def validate_writer_checkpoint_files(
    checkpoint: Path,
    *,
    world_size: int,
    contract_sha256: str,
    allow_historical_v6_warmstart: bool = False,
) -> dict[str, Any]:
    """Verify every checkpoint file before any pickle payload is loaded."""

    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    expected_files = {
        "writer.safetensors",
        "trainer_state.pt",
        *(f"rank_{rank:02d}_state.pt" for rank in range(world_size)),
    }
    files = manifest.get("files", {})
    updates_per_cycle = int(
        manifest.get("consumed", {}).get("optimizer_updates_per_task_cycle", 1)
    )
    checkpoint_state_family = manifest.get("consumed", {}).get(
        "checkpoint_state_family"
    )
    hashless = checkpoint_state_family in _HASHLESS_CHECKPOINT_FAMILIES
    schema_matches = checkpoint_schema_matches(
        manifest.get("schema_version"),
        updates_per_cycle,
        str(checkpoint_state_family) if checkpoint_state_family is not None else None,
        allow_historical_v6_warmstart=allow_historical_v6_warmstart,
    )
    if (
        not schema_matches
        or manifest.get(
            "contract_reference", manifest.get("contract_sha256")
        ) != contract_sha256
        or not isinstance(files, dict)
        or set(files) != expected_files
    ):
        raise WriterModelError("AS-Writer checkpoint manifest changed")
    for relative, record in files.items():
        path = checkpoint / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("bytes", -1))
            or (
                not hashless
                and sha256_file(path) != record.get("sha256")
            )
        ):
            raise WriterModelError(f"AS-Writer checkpoint file changed: {relative}")
    return manifest


def inspect_writer_checkpoint(
    checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Validate a published Writer checkpoint against its owning run contract."""

    checkpoint = checkpoint.resolve()
    if checkpoint.parent.name != "checkpoints":
        raise WriterModelError("AS-Writer initialization is outside a training run")
    contract_path = checkpoint.parent.parent / "run_contract.json"
    if not contract_path.is_file():
        raise WriterModelError("AS-Writer initialization lost its run contract")
    contract = read_json(contract_path)
    contract_sha256 = (
        str(contract["schema_version"])
        if contract.get("runtime", {}).get("checkpoint_state_family")
        in _HASHLESS_CHECKPOINT_FAMILIES
        else canonical_hash(contract)
    )
    world_size = int(contract.get("runtime", {}).get("world_size", -1))
    if world_size <= 0:
        raise WriterModelError("AS-Writer initialization topology is invalid")
    manifest = validate_writer_checkpoint_files(
        checkpoint,
        world_size=world_size,
        contract_sha256=contract_sha256,
        allow_historical_v6_warmstart=True,
    )
    return contract, manifest, contract_sha256


def initialize_writer_phase(
    checkpoint: Path | None,
    context: DistributedContext,
    stage: str,
    source: Mapping[str, Any],
    authorities: Mapping[str, Any],
    writer_config: Mapping[str, Any],
    writer: CompleteLoRAWriter,
    lora_contract_sha256: str,
) -> dict[str, Any]:
    """Load a compatible Writer state while deliberately resetting optimization."""

    if checkpoint is None:
        return {
            "mode": "functional_identity_init",
            "optimizer": "fresh",
            "scheduler": "fresh",
            "rng": "fresh_seed",
        }
    validation: list[Any] = [None]
    if context.is_main:
        try:
            training, manifest, contract_sha256 = inspect_writer_checkpoint(checkpoint)
            cursor = int(manifest.get("consumed", {}).get("next_step", -1))
            writer_record = manifest.get("files", {}).get("writer.safetensors", {})
            if (
                training.get("schema_version")
                not in {AS_WRITER_LAUNCH_SCHEMA, HISTORICAL_V6_LAUNCH_SCHEMA}
                or training.get("stage", "development") != stage
                or not source_reference_matches(training.get("source"), source)
                or training.get("authorities") != dict(authorities)
                or training.get("writer") != dict(writer_config)
                or training.get("trainable", {}).get("lora_contract_sha256")
                != lora_contract_sha256
                or cursor <= 0
                or checkpoint.name != f"step_{cursor:08d}"
                or not isinstance(writer_record.get("sha256"), str)
            ):
                raise WriterModelError("AS-Writer warm-start authority changed")
            validation[0] = {
                "mode": "writer_weight_warm_start",
                "source_checkpoint": str(checkpoint),
                "source_run_contract_sha256": contract_sha256,
                "source_checkpoint_manifest_sha256": sha256_file(
                    checkpoint / "checkpoint_manifest.json"
                ),
                "source_writer_state_sha256": writer_record["sha256"],
                "source_optimizer_step": cursor,
                "optimizer": "fresh",
                "scheduler": "fresh",
                "rng": "fresh_seed",
            }
        except Exception as error:
            validation[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(validation, src=0, device=context.device)
    if validation[0].get("error"):
        raise WriterModelError(validation[0]["error"])
    writer.load_state_dict(
        load_file(str(checkpoint / "writer.safetensors"), device=str(context.device))
    )
    return dict(validation[0])


def _trainer_resume_cursor(
    trainer: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    trainer_state_schema: str,
    contract_sha256: str,
    optimizer_updates_per_task_cycle: int,
    checkpoint_state_family: str,
) -> tuple[int, int, int]:
    next_step = int(trainer.get("next_step", -1))
    task_cycle, task_cycle_phase = divmod(
        next_step, optimizer_updates_per_task_cycle
    )
    scheduler_cursor = _scheduler_update_cursor(
        task_cycle, checkpoint_state_family
    )
    serial4 = optimizer_updates_per_task_cycle > 1
    valid = (
        trainer.get("schema_version") == trainer_state_schema
        and trainer.get(
            "checkpoint_state_family",
            (
                checkpoint_state_family
                if checkpoint_state_family.startswith("cvadr_legacy_")
                else ""
            ),
        )
        == checkpoint_state_family
        and trainer.get(
            "contract_reference", trainer.get("contract_sha256")
        ) == contract_sha256
        and int(trainer.get("metrics_rows", -1)) >= 0
        and int(
            trainer.get("next_task_cycle", -1 if serial4 else task_cycle)
        )
        == task_cycle
        and int(
            trainer.get(
                "next_task_cycle_phase", -1 if serial4 else task_cycle_phase
            )
        )
        == task_cycle_phase
        and int(
            trainer.get(
                "scheduler_logical_updates", -1 if serial4 else scheduler_cursor
            )
        )
        == scheduler_cursor
        and int(manifest.get("consumed", {}).get("next_step", -1))
        == next_step
    )
    if not valid:
        raise WriterModelError("Writer resume contract changed")
    return next_step, task_cycle, task_cycle_phase


def _validate_rank_resume_cursor(
    rank_state: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    checkpoint: Path,
    context: DistributedContext,
    rank_state_schema: str,
    next_step: int,
    task_cycle: int,
    task_cycle_phase: int,
    per_rank_batch_size: int,
    per_rank_batch_cycle: tuple[int, ...],
    videos_per_task_visit: int,
    tasks_per_rank_per_update: int,
    optimizer_updates_per_task_cycle: int,
    sampler_seed: int,
    teacher_video_seed: int,
    checkpoint_state_family: str,
) -> None:
    serial4 = optimizer_updates_per_task_cycle > 1
    scheduler_cursor = _scheduler_update_cursor(
        task_cycle, checkpoint_state_family
    )
    expected = (
        next_step,
        context.rank,
        context.world_size,
        per_rank_batch_size,
        per_rank_batch_cycle,
        videos_per_task_visit,
        tasks_per_rank_per_update,
        optimizer_updates_per_task_cycle,
        next_step * tasks_per_rank_per_update,
        task_cycle,
        task_cycle_phase,
        scheduler_cursor,
        sampler_seed,
        teacher_video_seed,
        checkpoint_state_family,
    )
    actual = (
        int(rank_state["next_step"]),
        int(rank_state["rank"]),
        int(rank_state["world_size"]),
        int(rank_state["per_rank_batch_size"]),
        tuple(int(value) for value in rank_state.get("per_rank_batch_cycle", ())),
        int(rank_state.get("teacher_videos_per_task_visit", -1)),
        int(rank_state.get("tasks_per_rank_per_optimizer_update", -1)),
        int(
            rank_state.get(
                "optimizer_updates_per_task_cycle", -1 if serial4 else 1
            )
        ),
        int(rank_state.get("next_data_step", -1)),
        int(rank_state.get("next_task_cycle", -1 if serial4 else task_cycle)),
        int(
            rank_state.get(
                "next_task_cycle_phase", -1 if serial4 else task_cycle_phase
            )
        ),
        int(
            rank_state.get(
                "scheduler_logical_updates", -1 if serial4 else task_cycle
            )
        ),
        int(rank_state["sampler_seed"]),
        int(rank_state["teacher_video_seed"]),
        str(
            rank_state.get(
                "checkpoint_state_family",
                (
                    checkpoint_state_family
                    if checkpoint_state_family.startswith("cvadr_legacy_")
                    else ""
                ),
            )
        ),
    )
    valid = (
        rank_state.get("schema_version") == rank_state_schema
        and actual == expected
        and checkpoint.name == f"step_{next_step:08d}"
        and int(manifest.get("consumed", {}).get("next_data_step", -1))
        == expected[8]
    )
    if not valid:
        raise WriterModelError("Writer rank resume state changed")


def _validate_cycle_normalized_optimizer_resume(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    next_step: int,
    task_cycle_phase: int,
    checkpoint_state_family: str,
) -> None:
    condition_kernel = (
        checkpoint_state_family == "condition_kernel_program_memory_full24_v1"
    )
    if checkpoint_state_family.startswith("cvadr_legacy_") or (
        checkpoint_state_family in {
            "target_bound_role_rawfull24_v1",
            "semantic_factor_basis_rawfull24_v1",
        }
    ):
        return
    group4 = (
        checkpoint_state_family in {
            "cvadr_cycle_normalized_randomized_group4_v2",
            "target_bound_role_cycle_normalized_randomized_group4_v1",
        }
    )
    if (
        not group4
        and checkpoint_state_family
        not in {
            "cvadr_task_query_keyed_rawfull24_v2",
            "target_bound_role_task_query_keyed_rawfull24_v1",
            "semantic_factor_basis_task_query_keyed_rawfull24_v1",
            "v6_relative_flow_coldstart_task_query_keyed_rawfull24_v1",
            "condition_kernel_program_memory_full24_v1",
            "k4_energy_preserving_policy_layer_trace_m2p_full24_v1",
            "k4_evidence_factorized_policy_layer_trace_m2p_full24_v1",
            "k4_sparse_semantic_expert_policy_layer_trace_m2p_full24_v1",
        }
    ):
        raise WriterModelError("unknown cycle-normalized optimizer resume family")
    expected_betas = (
        (0.9825931938526898, 0.9914875553891529)
        if group4
        else (0.9, 0.95)
    )
    divisor = 6 if group4 else 1
    logical_lrs = tuple(float(value) for value in scheduler.get_last_lr())
    if len(logical_lrs) != len(optimizer.param_groups):
        raise WriterModelError("optimizer resume lost logical LR groups")
    for group, logical_lr in zip(
        optimizer.param_groups, logical_lrs, strict=True
    ):
        expected_lr = (
            logical_lr
            if task_cycle_phase == 0
            else logical_lr / divisor
        )
        if (
            tuple(float(value) for value in group["betas"]) != expected_betas
            or float(group["eps"]) != 1e-8
            or not math.isclose(
                float(group["lr"]), expected_lr, rel_tol=1e-12, abs_tol=1e-15
            )
        ):
            raise WriterModelError("optimizer resume clock changed")
        if task_cycle_phase != 0:
            expected_weight_decay = cycle_matched_weight_decay(
                logical_lr,
                1e-4,
                divisor,
            )
            if not math.isclose(
                float(group["weight_decay"]),
                expected_weight_decay,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise WriterModelError("optimizer resume decay clock changed")
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            step = state.get("step")
            expected_step = min(next_step, 50) if condition_kernel else next_step
            if step is None or int(step.item()) != expected_step:
                raise WriterModelError("optimizer resume bias cursor changed")


def load_writer_checkpoint(
    *,
    checkpoint: Path,
    context: DistributedContext,
    writer: CompleteLoRAWriter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler_seed: int,
    teacher_video_seed: int,
    per_rank_batch_size: int,
    per_rank_batch_cycle: tuple[int, ...],
    videos_per_task_visit: int,
    tasks_per_rank_per_update: int,
    optimizer_updates_per_task_cycle: int,
    contract_sha256: str,
    checkpoint_state_family: str | None = None,
) -> tuple[int, dict[str, Any], int]:
    validation: list[Any] = [None]
    if context.is_main:
        try:
            validation[0] = validate_writer_checkpoint_files(
                checkpoint,
                world_size=context.world_size,
                contract_sha256=contract_sha256,
            )
        except Exception as error:
            validation[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(validation, src=0, device=context.device)
    if validation[0].get("error"):
        raise WriterModelError(validation[0]["error"])
    trainer = torch.load(
        checkpoint / "trainer_state.pt",
        map_location=context.device,
        weights_only=False,
    )
    resolved_checkpoint_state_family = checkpoint_state_family or (
        "cvadr_legacy_full24_v1"
        if optimizer_updates_per_task_cycle == 1
        else "cvadr_legacy_serial4_v1"
    )
    _, trainer_state_schema, rank_state_schema = _state_schemas(
        optimizer_updates_per_task_cycle,
        resolved_checkpoint_state_family,
    )
    next_step, task_cycle, task_cycle_phase = _trainer_resume_cursor(
        trainer,
        validation[0],
        trainer_state_schema=trainer_state_schema,
        contract_sha256=contract_sha256,
        optimizer_updates_per_task_cycle=optimizer_updates_per_task_cycle,
        checkpoint_state_family=resolved_checkpoint_state_family,
    )
    writer.load_state_dict(
        load_file(str(checkpoint / "writer.safetensors"), device=str(context.device))
    )
    optimizer.load_state_dict(trainer["optimizer"])
    scheduler.load_state_dict(trainer["scheduler"])
    expected_scheduler_cursor = (
        min(task_cycle, 50)
        if resolved_checkpoint_state_family
        == "condition_kernel_program_memory_full24_v1"
        else task_cycle
    )
    if int(scheduler.state_dict().get("last_epoch", -1)) != expected_scheduler_cursor:
        raise WriterModelError("Writer scheduler resume cursor changed")
    _validate_cycle_normalized_optimizer_resume(
        optimizer,
        scheduler,
        next_step=next_step,
        task_cycle_phase=task_cycle_phase,
        checkpoint_state_family=resolved_checkpoint_state_family,
    )
    rank_state = torch.load(
        checkpoint / f"rank_{context.rank:02d}_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    _validate_rank_resume_cursor(
        rank_state,
        validation[0],
        checkpoint=checkpoint,
        context=context,
        rank_state_schema=rank_state_schema,
        next_step=next_step,
        task_cycle=task_cycle,
        task_cycle_phase=task_cycle_phase,
        per_rank_batch_size=per_rank_batch_size,
        per_rank_batch_cycle=per_rank_batch_cycle,
        videos_per_task_visit=videos_per_task_visit,
        tasks_per_rank_per_update=tasks_per_rank_per_update,
        optimizer_updates_per_task_cycle=optimizer_updates_per_task_cycle,
        sampler_seed=sampler_seed,
        teacher_video_seed=teacher_video_seed,
        checkpoint_state_family=resolved_checkpoint_state_family,
    )
    return (
        next_step,
        rank_state["rng"],
        int(trainer["metrics_rows"]),
    )
