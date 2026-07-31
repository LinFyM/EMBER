"""Atomic exact-resume checkpoints for PI05 Action-Supervised Writer training."""

from __future__ import annotations

import json
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
    write_json_atomic,
)
from ember.writer.as_sampling import (
    MixedTaskBatchSampler,
    TeacherVideoSchedule,
)
from ember.writer.as_contract import AS_WRITER_LAUNCH_SCHEMA
from ember.writer.model import CompleteLoRAWriter, WriterModelError


AS_WRITER_CHECKPOINT_SCHEMA = (
    "ember_pi05_target_spectral_writer_checkpoint_v1"
)
AS_WRITER_TRAINER_STATE_SCHEMA = (
    "ember_pi05_target_spectral_writer_trainer_state_v1"
)
AS_WRITER_RANK_STATE_SCHEMA = (
    "ember_pi05_target_spectral_writer_rank_state_v1"
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
) -> None:
    torch.save(
        {
            "schema_version": AS_WRITER_RANK_STATE_SCHEMA,
            "next_step": step,
            "next_data_step": step * tasks_per_rank_per_update,
            "rank": context.rank,
            "world_size": context.world_size,
            "per_rank_batch_size": sampler.per_rank_batch_size,
            "per_rank_batch_cycle": sampler.per_rank_batch_cycle,
            "sampler_seed": sampler.seed,
            "teacher_video_seed": video_schedule.seed,
            "teacher_videos_per_task_visit": videos_per_task_visit,
            "tasks_per_rank_per_optimizer_update": tasks_per_rank_per_update,
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
            "schema_version": AS_WRITER_TRAINER_STATE_SCHEMA,
            "next_step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "amp_scaler": {"enabled": False, "state": {}},
            "contract_sha256": canonical_hash(contract),
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
        "teacher_videos_per_task_visit": videos_per_task_visit,
    }
    files = {
        str(path.relative_to(temporary)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(value for value in temporary.rglob("*") if value.is_file())
    }
    manifest = {
        "schema_version": AS_WRITER_CHECKPOINT_SCHEMA,
        "contract_sha256": canonical_hash(contract),
        "consumed": consumed,
        "files": files,
    }
    manifest["canonical_payload_sha256"] = canonical_hash(manifest)
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
) -> dict[str, Any]:
    """Verify every checkpoint file before any pickle payload is loaded."""

    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    payload = dict(manifest)
    digest = payload.pop("canonical_payload_sha256", None)
    expected_files = {
        "writer.safetensors",
        "trainer_state.pt",
        *(f"rank_{rank:02d}_state.pt" for rank in range(world_size)),
    }
    files = manifest.get("files", {})
    if (
        manifest.get("schema_version") != AS_WRITER_CHECKPOINT_SCHEMA
        or manifest.get("contract_sha256") != contract_sha256
        or canonical_hash(payload) != digest
        or not isinstance(files, dict)
        or set(files) != expected_files
    ):
        raise WriterModelError("AS-Writer checkpoint manifest changed")
    for relative, record in files.items():
        path = checkpoint / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(path) != record.get("sha256")
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
    contract_sha256 = canonical_hash(contract)
    world_size = int(contract.get("runtime", {}).get("world_size", -1))
    if world_size <= 0:
        raise WriterModelError("AS-Writer initialization topology is invalid")
    manifest = validate_writer_checkpoint_files(
        checkpoint,
        world_size=world_size,
        contract_sha256=contract_sha256,
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
                != AS_WRITER_LAUNCH_SCHEMA
                or training.get("stage", "development") != stage
                or training.get("source") != dict(source)
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
    contract_sha256: str,
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
    if (
        trainer.get("schema_version") != AS_WRITER_TRAINER_STATE_SCHEMA
        or trainer.get("contract_sha256") != contract_sha256
        or int(trainer.get("metrics_rows", -1)) < 0
        or int(validation[0].get("consumed", {}).get("next_step", -1))
        != int(trainer.get("next_step", -2))
    ):
        raise WriterModelError("Writer resume contract changed")
    writer.load_state_dict(
        load_file(str(checkpoint / "writer.safetensors"), device=str(context.device))
    )
    optimizer.load_state_dict(trainer["optimizer"])
    scheduler.load_state_dict(trainer["scheduler"])
    rank_state = torch.load(
        checkpoint / f"rank_{context.rank:02d}_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    expected = (
        int(trainer["next_step"]),
        context.rank,
        context.world_size,
        per_rank_batch_size,
        per_rank_batch_cycle,
        videos_per_task_visit,
        tasks_per_rank_per_update,
        int(trainer["next_step"]) * tasks_per_rank_per_update,
        sampler_seed,
        teacher_video_seed,
    )
    actual = (
        int(rank_state["next_step"]),
        int(rank_state["rank"]),
        int(rank_state["world_size"]),
        int(rank_state["per_rank_batch_size"]),
        tuple(int(value) for value in rank_state.get("per_rank_batch_cycle", ())),
        int(rank_state.get("teacher_videos_per_task_visit", -1)),
        int(rank_state.get("tasks_per_rank_per_optimizer_update", -1)),
        int(rank_state.get("next_data_step", -1)),
        int(rank_state["sampler_seed"]),
        int(rank_state["teacher_video_seed"]),
    )
    if (
        rank_state.get("schema_version") != AS_WRITER_RANK_STATE_SCHEMA
        or actual != expected
        or checkpoint.name != f"step_{expected[0]:08d}"
        or int(validation[0].get("consumed", {}).get("next_data_step", -1))
        != expected[7]
    ):
        raise WriterModelError("Writer rank resume state changed")
    return (
        int(trainer["next_step"]),
        rank_state["rng"],
        int(trainer["metrics_rows"]),
    )
