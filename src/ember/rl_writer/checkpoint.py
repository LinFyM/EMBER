"""Atomic exact-resume checkpoints for task-relative Flow-Credit Writer."""

from __future__ import annotations

import os
import random
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from ember.pi05_source_checkpoint import (
    DistributedContext,
    barrier,
    canonical_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.reward.ledger import InteractionCursors
from ember.reward.protocol import RewardProtocolError, RewardTask
from ember.rl_writer.contract import schedule_summary
from ember.writer.as_sampling import TeacherVideoSchedule
from ember.writer.model import CompleteLoRAWriter


RL_WRITER_CHECKPOINT_SCHEMA = "ember_pi05_task_grounded_progress_credit_checkpoint_v1"
RL_WRITER_TRAINER_SCHEMA = "ember_pi05_task_grounded_progress_credit_trainer_state_v1"
RL_WRITER_RANK_SCHEMA = "ember_pi05_task_grounded_progress_credit_rank_state_v1"


def _rng_state(context: DistributedContext) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state(context.device)
            if context.device.type == "cuda"
            else None
        ),
    }


def restore_rng(state: Mapping[str, Any], context: DistributedContext) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if context.device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], context.device)


def _nonce(context: DistributedContext) -> str:
    value = uuid.uuid4().hex
    if context.world_size == 1:
        return value
    encoded = torch.zeros(16, dtype=torch.uint8, device=context.device)
    if context.is_main:
        encoded.copy_(
            torch.tensor(
                list(bytes.fromhex(value)), dtype=torch.uint8, device=context.device
            )
        )
    dist.broadcast(encoded, src=0)
    return bytes(encoded.cpu().tolist()).hex()


def _distributed_error(
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
        raise RewardProtocolError(
            f"Flow-Credit checkpoint {phase} failed; " + "; ".join(observed)
        )


def _file_records(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(value for value in root.rglob("*") if value.is_file())
    }


def validate_rl_writer_checkpoint_files(
    checkpoint: Path,
    *,
    world_size: int,
    contract_sha256: str | None = None,
    rank_ledgers: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
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
        manifest.get("schema_version") != RL_WRITER_CHECKPOINT_SCHEMA
        or canonical_hash(payload) != digest
        or not isinstance(files, dict)
        or set(files) != expected_files
        or len(manifest.get("rank_ledgers", [])) != world_size
        or (
            contract_sha256 is not None
            and manifest.get("contract_sha256") != contract_sha256
        )
        or (
            rank_ledgers is not None
            and list(map(dict, rank_ledgers)) != manifest.get("rank_ledgers")
        )
    ):
        raise RewardProtocolError("Flow-Credit checkpoint manifest changed")
    for relative, record in files.items():
        path = checkpoint / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise RewardProtocolError(f"Flow-Credit checkpoint file changed: {relative}")
    return manifest


def _write_rank_state(
    path: Path,
    *,
    next_cycle: int,
    context: DistributedContext,
    cursors: InteractionCursors,
    successes: int,
    reward_sum: float,
    wall_nanoseconds: int,
    ledger_summary: Mapping[str, Any],
    rng: Mapping[str, Any],
) -> None:
    torch.save(
        {
            "schema_version": RL_WRITER_RANK_SCHEMA,
            "next_cycle": next_cycle,
            "rank": context.rank,
            "world_size": context.world_size,
            "cursors": cursors.to_dict(),
            "successes": successes,
            "reward_sum": reward_sum,
            "wall_nanoseconds": wall_nanoseconds,
            "ledger_summary": dict(ledger_summary),
            "rng": dict(rng),
        },
        path,
    )


def _publish_checkpoint(
    *,
    temporary: Path,
    final: Path,
    output_dir: Path,
    next_cycle: int,
    writer: CompleteLoRAWriter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    contract: Mapping[str, Any],
    cursors: InteractionCursors,
    metrics_rows: int,
    consumed: Mapping[str, Any],
    rank_ledgers: Sequence[Mapping[str, Any]],
) -> None:
    contract_sha = canonical_hash(contract)
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in writer.state_dict().items()},
        str(temporary / "writer.safetensors"),
    )
    torch.save(
        {
            "schema_version": RL_WRITER_TRAINER_SCHEMA,
            "contract_sha256": contract_sha,
            "next_cycle": next_cycle,
            "optimizer_update_cursor": cursors.optimizer_updates,
            "metrics_rows": metrics_rows,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        temporary / "trainer_state.pt",
    )
    files = _file_records(temporary)
    manifest = {
        "schema_version": RL_WRITER_CHECKPOINT_SCHEMA,
        "contract_sha256": contract_sha,
        "next_cycle": next_cycle,
        "consumed": dict(consumed),
        "rank_ledgers": list(map(dict, rank_ledgers)),
        "files": files,
    }
    manifest["canonical_payload_sha256"] = canonical_hash(manifest)
    write_json_atomic(temporary / "checkpoint_manifest.json", manifest)
    os.replace(temporary, final)
    write_json_atomic(
        output_dir / "latest_checkpoint.json",
        {
            "schema_version": "ember_pi05_task_grounded_progress_credit_latest_v1",
            "next_cycle": next_cycle,
            "path": str(final),
            "checkpoint_manifest_sha256": sha256_file(
                final / "checkpoint_manifest.json"
            ),
        },
    )


def save_rl_writer_checkpoint(
    *,
    output_dir: Path,
    next_cycle: int,
    context: DistributedContext,
    writer: CompleteLoRAWriter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    tasks: Sequence[RewardTask],
    task_schedule_seed: int,
    rollouts_per_task: int,
    video_schedule: TeacherVideoSchedule,
    contract: Mapping[str, Any],
    cursors: InteractionCursors,
    successes: int,
    reward_sum: float,
    wall_nanoseconds: int,
    ledger_summary: Mapping[str, Any],
    metrics_rows: int,
    learning_epochs: int,
) -> Path:
    per_rank_tasks = len(tasks) // context.world_size
    expected_rollouts = next_cycle * per_rank_tasks * rollouts_per_task
    if (
        cursors.rollout != expected_rollouts
        or cursors.environment_actions < cursors.rollout
        or cursors.optimizer_updates != next_cycle * learning_epochs
        or not 0 <= successes <= cursors.rollout
        or min(reward_sum, wall_nanoseconds, metrics_rows) < 0
        or int(ledger_summary.get("rollout_cursor", -1)) != cursors.rollout
        or int(ledger_summary.get("environment_action_cursor", -1))
        != cursors.environment_actions
    ):
        raise RewardProtocolError("Flow-Credit checkpoint cursors are inconsistent")
    consumed = schedule_summary(
        tasks,
        world_size=context.world_size,
        next_cycle=next_cycle,
        seed=task_schedule_seed,
        rollouts_per_task=rollouts_per_task,
        video_schedule=video_schedule,
    )
    rank_ledgers: list[Any] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(rank_ledgers, dict(ledger_summary))
    else:
        rank_ledgers[0] = dict(ledger_summary)

    nonce = _nonce(context)
    temporary = output_dir / "checkpoints" / f".cycle_{next_cycle:08d}.{nonce}.partial"
    final = output_dir / "checkpoints" / f"cycle_{next_cycle:08d}"
    error: Exception | None = None
    try:
        if context.is_main:
            if final.exists():
                raise RewardProtocolError(f"Flow-Credit checkpoint exists: {final}")
            temporary.mkdir(parents=True)
    except Exception as caught:
        error = caught
    _distributed_error(context, "initialization", error)

    saved_rng = _rng_state(context)
    error = None
    try:
        _write_rank_state(
            temporary / f"rank_{context.rank:02d}_state.pt",
            next_cycle=next_cycle,
            context=context,
            cursors=cursors,
            successes=successes,
            reward_sum=reward_sum,
            wall_nanoseconds=wall_nanoseconds,
            ledger_summary=ledger_summary,
            rng=saved_rng,
        )
    except Exception as caught:
        error = caught
    _distributed_error(context, "rank-state write", error)

    error = None
    try:
        if context.is_main:
            _publish_checkpoint(
                temporary=temporary,
                final=final,
                output_dir=output_dir,
                next_cycle=next_cycle,
                writer=writer,
                optimizer=optimizer,
                scheduler=scheduler,
                contract=contract,
                cursors=cursors,
                metrics_rows=metrics_rows,
                consumed=consumed,
                rank_ledgers=rank_ledgers,
            )
    except Exception as caught:
        error = caught
    _distributed_error(context, "publication", error)
    restore_rng(saved_rng, context)
    barrier(context)
    return final


def load_rl_writer_checkpoint(
    *,
    checkpoint: Path,
    context: DistributedContext,
    writer: CompleteLoRAWriter | torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    contract_sha256: str,
    tasks: Sequence[RewardTask],
    task_schedule_seed: int,
    rollouts_per_task: int,
    video_schedule: TeacherVideoSchedule,
    ledger_summary: Mapping[str, Any],
    learning_epochs: int,
) -> tuple[int, InteractionCursors, dict[str, Any], int, dict[str, Any]]:
    ledgers: list[Any] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(ledgers, dict(ledger_summary))
    else:
        ledgers[0] = dict(ledger_summary)
    validation: list[Any] = [None]
    if context.is_main:
        try:
            validation[0] = validate_rl_writer_checkpoint_files(
                checkpoint,
                world_size=context.world_size,
                contract_sha256=contract_sha256,
                rank_ledgers=ledgers,
            )
        except Exception as error:
            validation[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(validation, src=0, device=context.device)
    if validation[0].get("error"):
        raise RewardProtocolError(validation[0]["error"])
    manifest = validation[0]
    next_cycle = int(manifest.get("next_cycle", -1))
    expected_consumed = schedule_summary(
        tasks,
        world_size=context.world_size,
        next_cycle=next_cycle,
        seed=task_schedule_seed,
        rollouts_per_task=rollouts_per_task,
        video_schedule=video_schedule,
    )
    if manifest.get("consumed") != expected_consumed:
        raise RewardProtocolError("Flow-Credit resume schedule changed")
    trainer = torch.load(
        checkpoint / "trainer_state.pt", map_location=context.device, weights_only=False
    )
    rank_state = torch.load(
        checkpoint / f"rank_{context.rank:02d}_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    if (
        trainer.get("schema_version") != RL_WRITER_TRAINER_SCHEMA
        or rank_state.get("schema_version") != RL_WRITER_RANK_SCHEMA
        or trainer.get("contract_sha256") != contract_sha256
        or int(trainer.get("next_cycle", -2)) != next_cycle
        or int(rank_state.get("next_cycle", -2)) != next_cycle
        or int(rank_state.get("rank", -1)) != context.rank
        or int(rank_state.get("world_size", -1)) != context.world_size
        or rank_state.get("ledger_summary") != dict(ledger_summary)
        or checkpoint.name != f"cycle_{next_cycle:08d}"
    ):
        raise RewardProtocolError("Flow-Credit resume state changed")
    values = rank_state["cursors"]
    cursors = InteractionCursors(
        rollout=int(values["rollout_cursor"]),
        environment_actions=int(values["environment_action_cursor"]),
        optimizer_updates=int(values["optimizer_update_cursor"]),
    )
    expected_rollouts = next_cycle * (len(tasks) // context.world_size) * rollouts_per_task
    if (
        cursors.rollout != expected_rollouts
        or cursors.optimizer_updates != next_cycle * learning_epochs
        or cursors.optimizer_updates
        != int(trainer.get("optimizer_update_cursor", -1))
        or int(trainer.get("metrics_rows", -1)) < 0
    ):
        raise RewardProtocolError("Flow-Credit resume cursors changed")
    writer.load_state_dict(
        load_file(str(checkpoint / "writer.safetensors"), device=str(context.device)),
        strict=True,
    )
    optimizer.load_state_dict(trainer["optimizer"])
    scheduler.load_state_dict(trainer["scheduler"])
    counters = {
        "successes": int(rank_state["successes"]),
        "reward_sum": float(rank_state["reward_sum"]),
        "wall_nanoseconds": int(rank_state["wall_nanoseconds"]),
    }
    return next_cycle, cursors, rank_state["rng"], int(trainer["metrics_rows"]), counters
