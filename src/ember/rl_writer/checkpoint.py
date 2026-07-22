"""Atomic exact-resume checkpoints for PI05 Reward-Trained Writer."""

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
from ember.writer.data import TeacherVideoSchedule
from ember.writer.model import CompleteLoRAWriter


RL_WRITER_CHECKPOINT_SCHEMA = "ember_pi05_rl_writer_checkpoint_v1"
RL_WRITER_TRAINER_SCHEMA = "ember_pi05_rl_writer_trainer_state_v1"
RL_WRITER_RANK_SCHEMA = "ember_pi05_rl_writer_rank_state_v1"


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
            torch.tensor(list(bytes.fromhex(value)), dtype=torch.uint8, device=context.device)
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
            f"RL-Writer checkpoint {phase} failed; " + "; ".join(observed)
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
    """Hash every file and ledger prefix before loading optimizer/RNG pickles."""

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
        raise RewardProtocolError("RL-Writer checkpoint manifest changed")
    for relative, record in files.items():
        path = checkpoint / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise RewardProtocolError(f"RL-Writer checkpoint file changed: {relative}")
    return manifest


def _write_rank_state(
    path: Path,
    *,
    next_update: int,
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
            "next_update": next_update,
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
    next_update: int,
    writer: CompleteLoRAWriter | torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    contract: Mapping[str, Any],
    cursors: InteractionCursors,
    metrics_rows: int,
    consumed: Mapping[str, Any],
    rank_ledgers: Sequence[Mapping[str, Any]],
) -> None:
    save_file(
        {
            name: value.detach().to(device="cpu").contiguous()
            for name, value in writer.state_dict().items()
        },
        str(temporary / "writer.safetensors"),
    )
    torch.save(
        {
            "schema_version": RL_WRITER_TRAINER_SCHEMA,
            "next_update": next_update,
            "optimizer_update_cursor": cursors.optimizer_updates,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "amp_scaler": {"enabled": False, "state": {}},
            "contract_sha256": canonical_hash(contract),
            "metrics_rows": metrics_rows,
        },
        temporary / "trainer_state.pt",
    )
    manifest = {
        "schema_version": RL_WRITER_CHECKPOINT_SCHEMA,
        "contract_sha256": canonical_hash(contract),
        "next_update": next_update,
        "optimizer_update_cursor": cursors.optimizer_updates,
        "consumed": dict(consumed),
        "rank_ledgers": list(map(dict, rank_ledgers)),
        "files": _file_records(temporary),
    }
    manifest["canonical_payload_sha256"] = canonical_hash(manifest)
    write_json_atomic(temporary / "checkpoint_manifest.json", manifest)
    os.replace(temporary, final)
    write_json_atomic(
        output_dir / "latest_checkpoint.json",
        {"path": str(final), "next_update": next_update},
    )


def save_rl_writer_checkpoint(
    *,
    output_dir: Path,
    next_update: int,
    context: DistributedContext,
    writer: CompleteLoRAWriter | torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    tasks: Sequence[RewardTask],
    task_schedule_seed: int,
    rollouts_per_task_update: int,
    video_schedule: TeacherVideoSchedule,
    contract: Mapping[str, Any],
    cursors: InteractionCursors,
    successes: int,
    reward_sum: float,
    wall_nanoseconds: int,
    ledger_summary: Mapping[str, Any],
    metrics_rows: int,
    formal: bool,
) -> Path:
    expected_rollouts = next_update * rollouts_per_task_update
    if (
        cursors.rollout != expected_rollouts
        or cursors.environment_actions < cursors.rollout
        or not 0 <= successes <= cursors.rollout
        or reward_sum < 0
        or wall_nanoseconds < 0
        or metrics_rows < 0
        or int(ledger_summary.get("rollout_cursor", -1)) != cursors.rollout
        or int(ledger_summary.get("environment_action_cursor", -1))
        != cursors.environment_actions
    ):
        raise RewardProtocolError("RL-Writer checkpoint cursors are inconsistent")
    consumed = schedule_summary(
        tasks,
        world_size=context.world_size,
        next_update=next_update,
        seed=task_schedule_seed,
        rollouts_per_task_update=rollouts_per_task_update,
        video_schedule=video_schedule,
    )
    if formal and consumed["cycle_slot_cursor"] != 0:
        raise RewardProtocolError("formal RL-Writer checkpoint is not a full task cycle")
    rank_ledgers: list[Any] = [None] * context.world_size
    local_ledger = dict(ledger_summary)
    if context.world_size > 1:
        dist.all_gather_object(rank_ledgers, local_ledger)
    else:
        rank_ledgers[0] = local_ledger

    nonce = _nonce(context)
    temporary = output_dir / "checkpoints" / f".update_{next_update:08d}.{nonce}.partial"
    final = output_dir / "checkpoints" / f"update_{next_update:08d}"
    error: Exception | None = None
    try:
        if context.is_main:
            if final.exists():
                raise RewardProtocolError(f"RL-Writer checkpoint exists: {final}")
            temporary.mkdir(parents=True)
    except Exception as caught:
        error = caught
    _distributed_error(context, "initialization", error)

    saved_rng = _rng_state(context)
    error = None
    try:
        _write_rank_state(
            temporary / f"rank_{context.rank:02d}_state.pt",
            next_update=next_update,
            context=context,
            cursors=cursors,
            successes=successes,
            reward_sum=reward_sum,
            wall_nanoseconds=wall_nanoseconds,
            ledger_summary=local_ledger,
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
                next_update=next_update,
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
    rollouts_per_task_update: int,
    video_schedule: TeacherVideoSchedule,
    ledger_summary: Mapping[str, Any],
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
    next_update = int(manifest.get("next_update", -1))
    expected_consumed = schedule_summary(
        tasks,
        world_size=context.world_size,
        next_update=next_update,
        seed=task_schedule_seed,
        rollouts_per_task_update=rollouts_per_task_update,
        video_schedule=video_schedule,
    )
    if manifest.get("consumed") != expected_consumed:
        raise RewardProtocolError("RL-Writer resume schedule changed")
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
        or int(trainer.get("next_update", -2)) != next_update
        or int(rank_state.get("next_update", -2)) != next_update
        or int(rank_state.get("rank", -1)) != context.rank
        or int(rank_state.get("world_size", -1)) != context.world_size
        or rank_state.get("ledger_summary") != dict(ledger_summary)
        or checkpoint.name != f"update_{next_update:08d}"
    ):
        raise RewardProtocolError("RL-Writer resume state changed")
    cursor_values = rank_state["cursors"]
    cursors = InteractionCursors(
        rollout=int(cursor_values["rollout_cursor"]),
        environment_actions=int(cursor_values["environment_action_cursor"]),
        optimizer_updates=int(cursor_values["optimizer_update_cursor"]),
    )
    if (
        cursors.rollout != next_update * rollouts_per_task_update
        or cursors.optimizer_updates
        != int(trainer.get("optimizer_update_cursor", -1))
        or int(trainer.get("metrics_rows", -1)) < 0
    ):
        raise RewardProtocolError("RL-Writer resume cursors changed")
    writer.load_state_dict(
        load_file(str(checkpoint / "writer.safetensors"), device=str(context.device))
    )
    optimizer.load_state_dict(trainer["optimizer"])
    scheduler.load_state_dict(trainer["scheduler"])
    counters = {
        "successes": int(rank_state["successes"]),
        "reward_sum": float(rank_state["reward_sum"]),
        "wall_nanoseconds": int(rank_state["wall_nanoseconds"]),
    }
    return next_update, cursors, rank_state["rng"], int(trainer["metrics_rows"]), counters
