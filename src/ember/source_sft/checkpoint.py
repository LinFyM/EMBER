"""Atomic exact-resume checkpoints for the shared PI05 Source-SFT LoRA."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from ember.lora import (
    LoRAContract,
    copy_task_lora_state_,
    task_lora_state_dict,
    validate_lora_state,
)
from ember.pi05_source_checkpoint import (
    DistributedContext,
    canonical_hash,
    capture_rng,
    read_json,
    restore_rng,
    sha256_file,
    write_json_atomic,
)
from ember.source_sft.contract import Pi05SourceSFTError
from ember.writer.data import MixedTaskBatchSampler


SOURCE_SFT_CHECKPOINT_SCHEMA = "ember_pi05_source_sft_checkpoint_v1"
SOURCE_SFT_TRAINER_SCHEMA = "ember_pi05_source_sft_trainer_state_v1"
SOURCE_SFT_RANK_SCHEMA = "ember_pi05_source_sft_rank_state_v1"


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


def _raise_distributed(
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
        raise Pi05SourceSFTError(
            f"Source-SFT checkpoint {phase} failed; " + "; ".join(observed)
        )


def _checkpoint_files(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _write_rank_state(
    path: Path,
    *,
    step: int,
    context: DistributedContext,
    sampler: MixedTaskBatchSampler,
    contract: Mapping[str, Any],
    saved_rng: Mapping[str, Any],
) -> None:
    torch.save(
        {
            "schema_version": SOURCE_SFT_RANK_SCHEMA,
            "next_step": step,
            "next_optimizer_step": step,
            "next_micro_step": step,
            "rank": context.rank,
            "world_size": context.world_size,
            "per_rank_batch_size": sampler.per_rank_batch_size,
            "sampler_seed": sampler.seed,
            "dataloader_generator_seed": int(
                contract["runtime"]["dataloader_generator_seed_base"]
            )
            + context.rank,
            "worker_random_transforms": False,
            "rng": saved_rng,
        },
        path,
    )


def _publish_shared_checkpoint(
    *,
    temporary: Path,
    final: Path,
    output_dir: Path,
    step: int,
    total_steps: int,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler: MixedTaskBatchSampler,
    contract: Mapping[str, Any],
    mode: str,
    metrics_rows: int,
) -> None:
    state = task_lora_state_dict(policy, clone=True)
    save_file(
        {name: value.to(device="cpu").contiguous() for name, value in state.items()},
        str(temporary / "lora.safetensors"),
    )
    torch.save(
        {
            "schema_version": SOURCE_SFT_TRAINER_SCHEMA,
            "next_step": step,
            "next_optimizer_step": step,
            "next_micro_step": step,
            "gradient_accumulation_offset": 0,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "amp_scaler": {"enabled": False, "state": {}},
            "contract_sha256": canonical_hash(contract),
            "metrics_rows": metrics_rows,
        },
        temporary / "trainer_state.pt",
    )
    coverage = sampler.coverage_for_steps(0, step)
    if mode == "formal" and step == total_steps and any(
        len(episodes) != sampler.episodes_per_task for episodes in coverage.values()
    ):
        raise Pi05SourceSFTError(
            "final formal Source-SFT checkpoint lacks all declared episodes"
        )
    consumed = {
        **sampler.consumed_identity_summary(0, step),
        "declared_task_count": len(coverage),
        "tasks_with_action_signal": sum(bool(value) for value in coverage.values()),
        "min_action_episodes_per_task": min(map(len, coverage.values())),
        "max_action_episodes_per_task": max(map(len, coverage.values())),
        "next_step": step,
        "next_optimizer_step": step,
        "next_micro_step": step,
    }
    manifest = {
        "schema_version": SOURCE_SFT_CHECKPOINT_SCHEMA,
        "contract_sha256": canonical_hash(contract),
        "stage": contract["stage"],
        "consumed": consumed,
        "files": _checkpoint_files(temporary),
    }
    manifest["canonical_payload_sha256"] = canonical_hash(manifest)
    write_json_atomic(temporary / "checkpoint_manifest.json", manifest)
    os.replace(temporary, final)
    write_json_atomic(
        output_dir / "latest_checkpoint.json", {"path": str(final), "step": step}
    )


def save_source_sft_checkpoint(
    *,
    output_dir: Path,
    step: int,
    context: DistributedContext,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler: MixedTaskBatchSampler,
    contract: Mapping[str, Any],
    mode: str,
    metrics_rows: int,
) -> Path:
    total_steps = int(contract.get("runtime", {}).get("total_steps", -1))
    if mode not in {"profile", "formal"} or not 0 < step <= total_steps:
        raise Pi05SourceSFTError("Source-SFT checkpoint step is outside its contract")
    temporary = (
        output_dir
        / "checkpoints"
        / f".step_{step:08d}.{_nonce(context)}.partial"
    )
    final = output_dir / "checkpoints" / f"step_{step:08d}"
    error: Exception | None = None
    try:
        if context.is_main:
            if final.exists():
                raise Pi05SourceSFTError(f"Source-SFT checkpoint exists: {final}")
            temporary.mkdir(parents=True)
    except Exception as caught:
        error = caught
    _raise_distributed(context, "initialization", error)

    saved_rng = capture_rng(context)
    error = None
    try:
        _write_rank_state(
            temporary / f"rank_{context.rank:02d}_state.pt",
            step=step,
            context=context,
            sampler=sampler,
            contract=contract,
            saved_rng=saved_rng,
        )
    except Exception as caught:
        error = caught
    _raise_distributed(context, "rank-state write", error)

    error = None
    try:
        if context.is_main:
            _publish_shared_checkpoint(
                temporary=temporary,
                final=final,
                output_dir=output_dir,
                step=step,
                total_steps=total_steps,
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                contract=contract,
                mode=mode,
                metrics_rows=metrics_rows,
            )
    except Exception as caught:
        error = caught
    _raise_distributed(context, "publication", error)
    restore_rng(saved_rng, context)
    return final


def validate_source_sft_checkpoint_files(
    checkpoint: Path,
    *,
    world_size: int,
    contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify every file before any optimizer or RNG pickle is read."""

    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    payload = dict(manifest)
    digest = payload.pop("canonical_payload_sha256", None)
    expected = {
        "lora.safetensors",
        "trainer_state.pt",
        *(f"rank_{rank:02d}_state.pt" for rank in range(world_size)),
    }
    files = manifest.get("files", {})
    if (
        manifest.get("schema_version") != SOURCE_SFT_CHECKPOINT_SCHEMA
        or canonical_hash(payload) != digest
        or not isinstance(files, dict)
        or set(files) != expected
        or (
            contract_sha256 is not None
            and manifest.get("contract_sha256") != contract_sha256
        )
    ):
        raise Pi05SourceSFTError("Source-SFT checkpoint manifest changed")
    for relative, record in files.items():
        path = checkpoint / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise Pi05SourceSFTError(f"Source-SFT checkpoint file changed: {relative}")
    return manifest


def load_source_sft_checkpoint(
    *,
    checkpoint: Path,
    context: DistributedContext,
    policy: torch.nn.Module,
    lora_contract: LoRAContract,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    per_rank_batch_size: int,
    sampler_seed: int,
    dataloader_generator_seed: int,
    contract_sha256: str,
) -> tuple[int, dict[str, Any], int]:
    validation: list[Any] = [None]
    if context.is_main:
        try:
            validation[0] = validate_source_sft_checkpoint_files(
                checkpoint,
                world_size=context.world_size,
                contract_sha256=contract_sha256,
            )
        except Exception as error:
            validation[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(validation, src=0, device=context.device)
    if validation[0].get("error"):
        raise Pi05SourceSFTError(validation[0]["error"])
    trainer = torch.load(
        checkpoint / "trainer_state.pt",
        map_location=context.device,
        weights_only=False,
    )
    rank_state = torch.load(
        checkpoint / f"rank_{context.rank:02d}_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    next_step = int(trainer.get("next_step", -1))
    expected = (
        next_step,
        context.rank,
        context.world_size,
        per_rank_batch_size,
        sampler_seed,
        dataloader_generator_seed,
        False,
    )
    actual = (
        int(rank_state.get("next_step", -2)),
        int(rank_state.get("rank", -1)),
        int(rank_state.get("world_size", -1)),
        int(rank_state.get("per_rank_batch_size", -1)),
        int(rank_state.get("sampler_seed", -1)),
        int(rank_state.get("dataloader_generator_seed", -1)),
        bool(rank_state.get("worker_random_transforms", True)),
    )
    if (
        trainer.get("schema_version") != SOURCE_SFT_TRAINER_SCHEMA
        or trainer.get("contract_sha256") != contract_sha256
        or rank_state.get("schema_version") != SOURCE_SFT_RANK_SCHEMA
        or actual != expected
        or int(trainer.get("next_optimizer_step", -1)) != next_step
        or int(trainer.get("next_micro_step", -1)) != next_step
        or int(trainer.get("gradient_accumulation_offset", -1)) != 0
        or checkpoint.name != f"step_{next_step:08d}"
        or int(validation[0].get("consumed", {}).get("next_step", -1)) != next_step
        or int(trainer.get("metrics_rows", -1)) < 0
    ):
        raise Pi05SourceSFTError("Source-SFT resume state changed")
    state = load_file(str(checkpoint / "lora.safetensors"), device=str(context.device))
    validate_lora_state(state, lora_contract)
    copy_task_lora_state_(policy, state, lora_contract)
    optimizer.load_state_dict(trainer["optimizer"])
    scheduler.load_state_dict(trainer["scheduler"])
    return next_step, rank_state["rng"], int(trainer["metrics_rows"])
