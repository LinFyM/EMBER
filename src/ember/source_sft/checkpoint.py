"""Atomic exact-resume checkpoints for the shared PI05 Source-SFT LoRA."""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch
from ember.source_sft.control import dynamic_control, checkpoint_declared
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
from ember.source_sft.sampler import HierarchicalMixedBatchSampler


LEGACY_SOURCE_SFT_CHECKPOINT_SCHEMA = "ember_pi05_source_sft_checkpoint_v2"
PREVIOUS_SOURCE_SFT_CHECKPOINT_SCHEMA = "ember_pi05_source_sft_checkpoint_v4"
SOURCE_SFT_CHECKPOINT_SCHEMA = "ember_pi05_source_sft_checkpoint_v5"
SOURCE_SFT_TRAINER_SCHEMA = "ember_pi05_source_sft_trainer_state_v5"
SOURCE_SFT_RANK_SCHEMA = "ember_pi05_source_sft_rank_state_v5"
_RANK_FILE = re.compile(r"rank_([0-9]{2})_state\.pt")


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
    sampler: HierarchicalMixedBatchSampler,
    contract: Mapping[str, Any],
    saved_rng: Mapping[str, Any],
    micro_step_offset: int,
) -> None:
    torch.save(
        {
            "schema_version": SOURCE_SFT_RANK_SCHEMA,
            "next_step": step,
            "next_optimizer_step": step,
            "next_micro_step": step * sampler.accumulation + micro_step_offset,
            "micro_step_offset": micro_step_offset,
            "sampler": sampler.resume_contract(),
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
    sampler: HierarchicalMixedBatchSampler,
    contract: Mapping[str, Any],
    mode: str,
    metrics_rows: int,
    micro_step_offset: int,
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
            "next_micro_step": step * sampler.accumulation + micro_step_offset,
            "micro_step_offset": micro_step_offset,
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
        **sampler.consumed_summary(0, step),
        "declared_task_count": len(coverage),
        "tasks_with_action_signal": sum(bool(value) for value in coverage.values()),
        "min_action_episodes_per_task": min(map(len, coverage.values())),
        "max_action_episodes_per_task": max(map(len, coverage.values())),
        "next_step": step,
        "next_optimizer_step": step,
        "next_micro_step": step * sampler.accumulation + micro_step_offset,
    }
    manifest = {
        "schema_version": SOURCE_SFT_CHECKPOINT_SCHEMA,
        "contract_sha256": canonical_hash(contract),
        "stage": contract["stage"],
        "physical_world_size": sampler.world_size,
        "physical_packing": sampler.physical_packing,
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
    sampler: HierarchicalMixedBatchSampler,
    contract: Mapping[str, Any],
    mode: str,
    metrics_rows: int,
    micro_step_offset: int = 0,
) -> Path:
    total_steps = int(contract.get("runtime", {}).get("total_steps", -1))
    if (
        mode not in {"profile", "formal"}
        or (not checkpoint_declared(contract, step) if dynamic_control(contract) else not 0 < step <= total_steps)
    ):
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
            micro_step_offset=micro_step_offset,
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
                micro_step_offset=micro_step_offset,
            )
    except Exception as caught:
        error = caught
    _raise_distributed(context, "publication", error)
    restore_rng(saved_rng, context)
    return final


def validate_source_sft_checkpoint_files(
    checkpoint: Path,
    *,
    world_size: int | None,
    contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify every file before any optimizer or RNG pickle is read."""

    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    payload = dict(manifest)
    digest = payload.pop("canonical_payload_sha256", None)
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        raise Pi05SourceSFTError("Source-SFT checkpoint manifest changed")
    rank_numbers = sorted(int(match.group(1)) for name in files
                          if (match := _RANK_FILE.fullmatch(name)))
    source_world_size = len(rank_numbers) if world_size is None else world_size
    expected = {
        "lora.safetensors",
        "trainer_state.pt",
        *(f"rank_{rank:02d}_state.pt" for rank in range(source_world_size)),
    }
    if (
        manifest.get("schema_version")
        not in {
            LEGACY_SOURCE_SFT_CHECKPOINT_SCHEMA,
            "ember_pi05_source_sft_checkpoint_v3",
            PREVIOUS_SOURCE_SFT_CHECKPOINT_SCHEMA,
            SOURCE_SFT_CHECKPOINT_SCHEMA,
        }
        or canonical_hash(payload) != digest
        or not isinstance(files, dict)
        or source_world_size <= 0
        or rank_numbers != list(range(source_world_size))
        or (manifest.get("schema_version") == SOURCE_SFT_CHECKPOINT_SCHEMA
            and (manifest.get("physical_world_size") != source_world_size
                 or manifest.get("physical_packing") not in {"contiguous", "task_striped"}))
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
    sampler: HierarchicalMixedBatchSampler,
    dataloader_generator_seed: int,
    contract_sha256: str,
    allow_physical_resume: bool = False,
) -> tuple[int, dict[str, Any], int, int]:
    validation: list[Any] = [None]
    if context.is_main:
        try:
            validation[0] = validate_source_sft_checkpoint_files(
                checkpoint,
                world_size=None if allow_physical_resume else context.world_size,
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
    source_world_size = sum(bool(_RANK_FILE.fullmatch(name))
                            for name in validation[0]["files"])
    source_rank = min(context.rank, source_world_size - 1)
    rank_state = torch.load(checkpoint / f"rank_{source_rank:02d}_state.pt",
                            map_location="cpu", weights_only=False)
    next_step = int(trainer.get("next_step", -1))
    source_sampler = rank_state.get("sampler", {})
    source_accumulation = int(source_sampler.get("gradient_accumulation_steps", -1))
    old_offset = int(trainer.get("micro_step_offset", 0))
    expected_micro = next_step * source_accumulation + old_offset
    logical_keys = (
        "sampler_kind", "sampler_seed", "task_ids", "logical_world_size",
        "logical_per_rank_batch_size", "global_batch_size",
    )
    logical_match = all(source_sampler.get(key) == sampler.resume_contract().get(key)
                        for key in logical_keys)
    physical_match = source_sampler == sampler.resume_contract()
    trainer_schema = trainer.get("schema_version")
    rank_schema = rank_state.get("schema_version")
    if (
        trainer_schema not in {"ember_pi05_source_sft_trainer_state_v4",
                               SOURCE_SFT_TRAINER_SCHEMA}
        or trainer.get("contract_sha256") != contract_sha256
        or rank_schema not in {"ember_pi05_source_sft_rank_state_v4",
                               SOURCE_SFT_RANK_SCHEMA}
        or not logical_match
        or (not allow_physical_resume and not physical_match)
        or (not allow_physical_resume and source_world_size != context.world_size)
        or int(source_sampler.get("world_size", -1)) != source_world_size
        or int(source_sampler.get("rank", -1)) != source_rank
        or int(rank_state.get("micro_step_offset", 0)) != old_offset
        or int(rank_state.get("next_step", -1)) != next_step
        or int(rank_state.get("next_optimizer_step", -1)) != next_step
        or int(rank_state.get("next_micro_step", -1)) != expected_micro
        or rank_state.get("dataloader_generator_seed") != (
            dataloader_generator_seed + source_rank - context.rank)
        or rank_state.get("worker_random_transforms") is not False
        or int(trainer.get("next_optimizer_step", -1)) != next_step
        or int(trainer.get("next_micro_step", -1)) != expected_micro
        or int(trainer.get("gradient_accumulation_offset", -1)) != 0
        or checkpoint.name != f"step_{next_step:08d}"
        or validation[0].get("schema_version") not in {
            PREVIOUS_SOURCE_SFT_CHECKPOINT_SCHEMA, SOURCE_SFT_CHECKPOINT_SCHEMA}
        or int(validation[0].get("consumed", {}).get("next_step", -1)) != next_step
        or int(validation[0].get("consumed", {}).get("next_micro_step", -1)) != expected_micro
        or int(trainer.get("metrics_rows", -1)) < 0
    ):
        raise Pi05SourceSFTError("Source-SFT resume state changed")
    state = load_file(str(checkpoint / "lora.safetensors"), device=str(context.device))
    validate_lora_state(state, lora_contract)
    copy_task_lora_state_(policy, state, lora_contract)
    optimizer.load_state_dict(trainer["optimizer"])
    scheduler.load_state_dict(trainer["scheduler"])
    rng = rank_state["rng"] if context.rank < source_world_size else capture_rng(context)
    micro_step_offset = expected_micro - next_step * sampler.accumulation
    return next_step, rng, int(trainer["metrics_rows"]), micro_step_offset
