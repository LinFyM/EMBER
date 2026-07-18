"""Shared source-base optimization mechanics for Gate 0 calibration and training."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ember.gate_zero_data import (
    GateZeroSurface,
    SourceHdf5Dataset,
    TaskDemoFrameBatchSampler,
    load_surface_authorities,
    verify_task_authority,
)
from ember.gate_zero_distributed import (
    DistributedContext,
    TrainingTopology,
    broadcast_primary_object,
    distributed_max,
    distributed_mean,
    gather_rank_objects,
    global_effective_slots,
    merge_rank_provenance,
    native_global_flow_inputs,
    primary_rng_state_sha256,
    unwrap_distributed_model,
)
from ember.gate_zero_runtime import (
    batch_provenance_keys,
    deterministic_flow_inputs,
    load_smolvla_policy,
    load_source_normalization,
    preprocess_smolvla_batch,
    sha256_file,
    smolvla_flow_loss,
)


class GateZeroBaseRuntimeError(RuntimeError):
    """Raised when source-base optimization mechanics drift."""


def gradient_accumulation_steps(
    effective_batch_size: int, micro_batch_size: int, *, world_size: int = 1
) -> int:
    if effective_batch_size <= 0 or micro_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if world_size <= 0:
        raise ValueError("world size must be positive")
    distributed_micro_batch = micro_batch_size * world_size
    if effective_batch_size % distributed_micro_batch:
        raise ValueError("distributed microbatch must divide the effective batch")
    return effective_batch_size // distributed_micro_batch


def build_base_optimizer(
    parameters: Iterable[torch.nn.Parameter], spec: dict[str, Any]
) -> torch.optim.AdamW:
    base_fit = spec["base_fit"]
    return torch.optim.AdamW(
        list(parameters),
        lr=base_fit["learning_rate"],
        betas=tuple(base_fit["betas"]),
        eps=base_fit["epsilon"],
        weight_decay=base_fit["weight_decay"],
    )


def build_base_scheduler(
    optimizer: torch.optim.Optimizer, spec: dict[str, Any]
) -> torch.optim.lr_scheduler.LambdaLR:
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

    base_fit = spec["base_fit"]
    config = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=base_fit["warmup_steps"],
        num_decay_steps=base_fit["decay_steps"],
        peak_lr=base_fit["learning_rate"],
        decay_lr=base_fit["decay_learning_rate"],
    )
    return config.build(optimizer, num_training_steps=base_fit["steps"])


def optimizer_state_summary(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Report actual parameter and optimizer-state tensor dtypes without values."""

    parameter_dtypes: dict[str, int] = {}
    seen_parameters: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) in seen_parameters:
                continue
            seen_parameters.add(id(parameter))
            key = str(parameter.dtype)
            parameter_dtypes[key] = parameter_dtypes.get(key, 0) + parameter.numel()
    state_dtypes: dict[str, int] = {}
    state_keys: dict[str, int] = {}
    for state in optimizer.state.values():
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            dtype = str(value.dtype)
            state_dtypes[dtype] = state_dtypes.get(dtype, 0) + value.numel()
            state_keys[key] = state_keys.get(key, 0) + value.numel()
    return {
        "parameter_dtype_elements": dict(sorted(parameter_dtypes.items())),
        "state_tensor_elements_by_dtype": dict(sorted(state_dtypes.items())),
        "state_tensor_elements_by_key": dict(sorted(state_keys.items())),
    }


def capture_trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Keep one CPU copy of trainable tensors for matched technical branches."""

    return {
        name: value.detach().to(device="cpu", copy=True)
        for name, value in model.named_parameters()
        if value.requires_grad
    }


def restore_trainable_state(
    model: torch.nn.Module, snapshot: dict[str, torch.Tensor]
) -> None:
    trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if set(trainable) != set(snapshot):
        raise GateZeroBaseRuntimeError("trainable parameter identity changed")
    with torch.no_grad():
        for name, value in trainable.items():
            authority = snapshot[name]
            if authority.shape != value.shape or authority.dtype != value.dtype:
                raise GateZeroBaseRuntimeError(f"trainable parameter metadata changed: {name}")
            value.copy_(authority, non_blocking=False)


def make_base_loader(
    dataset: SourceHdf5Dataset,
    *,
    micro_batch_size: int,
    effective_batch_size: int,
    optimizer_steps: int,
    start_optimizer_step: int,
    sampler_seed: int,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    accumulation_steps = gradient_accumulation_steps(
        effective_batch_size, micro_batch_size, world_size=world_size
    )
    sampler = TaskDemoFrameBatchSampler(
        dataset,
        micro_batch_size=micro_batch_size,
        optimizer_steps=optimizer_steps,
        gradient_accumulation_steps=accumulation_steps,
        seed=sampler_seed,
        start_optimizer_step=start_optimizer_step,
        rank=rank,
        world_size=world_size,
        global_effective_batch_size=effective_batch_size,
    )
    worker_generator = torch.Generator(device="cpu").manual_seed(
        sampler_seed + start_optimizer_step + rank * 1_000_003
    )
    kwargs: dict[str, Any] = {}
    if num_workers:
        kwargs.update(
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=worker_generator,
        **kwargs,
    )


def load_base_training_components(
    *,
    spec: dict[str, Any],
    phase0: dict[str, Any],
    manifest_path: Path,
    normalization_path: Path,
    dataset_root: Path,
    base_path: Path,
    vlm_path: Path,
    verify_dataset_sha256: bool = True,
    verify_base_weight_sha256: bool = True,
) -> tuple[SourceHdf5Dataset, Any, Any, Any]:
    """Load the single all-source dataset/policy/processor authority used by Gate 0."""

    if (
        verify_base_weight_sha256
        and sha256_file(base_path / "model.safetensors")
        != spec["authority"]["model_weight_sha256"]
    ):
        raise GateZeroBaseRuntimeError("base policy weight authority changed")
    authorities, demo_indices = load_surface_authorities(
        spec,
        phase0,
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        surface=GateZeroSurface.BASE_FIT,
    )
    dataset = SourceHdf5Dataset(
        authorities,
        demo_indices=demo_indices,
        action_chunk_size=spec["data"]["action_chunk_size"],
        verify_sha256=verify_dataset_sha256,
    )
    stats = load_source_normalization(
        normalization_path,
        expected_sha256=spec["authority"]["source_normalization_sha256"],
        expected_task_ids=phase0["splits"]["source"],
        expected_count=183555,
    )
    policy = load_smolvla_policy(base_path, vlm_path, spec)
    policy.train()
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    preprocessor, postprocessor = make_smolvla_pre_post_processors(
        policy.config, dataset_stats=stats
    )
    return dataset, policy, preprocessor, postprocessor


def validate_base_training_files_authority(
    spec: dict[str, Any],
    phase0: dict[str, Any],
    *,
    manifest_path: Path,
    dataset_root: Path,
    base_path: Path,
) -> dict[str, int]:
    """Hash shared model/data files once before rank-local loading reuses them."""

    if sha256_file(base_path / "model.safetensors") != spec["authority"]["model_weight_sha256"]:
        raise GateZeroBaseRuntimeError("base policy weight authority changed")
    authorities, _ = load_surface_authorities(
        spec,
        phase0,
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        surface=GateZeroSurface.BASE_FIT,
    )
    for authority in authorities:
        verify_task_authority(authority, verify_sha256=True)
    return {
        "task_count": len(authorities),
        "total_bytes": sum(authority.expected_bytes for authority in authorities),
        "model_bytes": (base_path / "model.safetensors").stat().st_size,
    }


def training_row_keys(
    raw_batch: dict[str, Any],
    *,
    optimizer_step: int,
    effective_batch_start_slot: int,
) -> list[str]:
    return [
        f"{key}/step{optimizer_step}/effective_slot{effective_batch_start_slot + slot}"
        for slot, key in enumerate(batch_provenance_keys(raw_batch))
    ]


def optimizer_step(
    iterator: Iterator[dict[str, Any]],
    *,
    policy: Any,
    preprocessor: Any,
    optimizer: torch.optim.Optimizer,
    spec: dict[str, Any],
    optimizer_step_index: int,
    accumulation_steps: int,
    fixed_flow_seed: int | None = None,
    topology: TrainingTopology | None = None,
    distributed_context: DistributedContext | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    mean_loss: torch.Tensor | None = None
    flow_rng_state_sha256: str | None = None
    flow_input_sha256: str | None = None
    local_row_keys: list[str] = []
    if (topology is None) != (distributed_context is None):
        raise GateZeroBaseRuntimeError("topology and distributed context must be provided together")
    owner = unwrap_distributed_model(policy)
    for accumulation_step in range(accumulation_steps):
        raw_batch = next(iterator)
        if topology is None:
            effective_start_slot = accumulation_step * len(raw_batch["task_id"])
        else:
            slots = global_effective_slots(
                topology,
                rank=distributed_context.rank,
                accumulation_step=accumulation_step,
            )
            if len(slots) != len(raw_batch["task_id"]):
                raise GateZeroBaseRuntimeError("local batch differs from distributed slot shard")
            effective_start_slot = slots[0]
        keys = training_row_keys(
            raw_batch,
            optimizer_step=optimizer_step_index,
            effective_batch_start_slot=effective_start_slot,
        )
        local_row_keys.extend(keys)
        batch = preprocess_smolvla_batch(
            raw_batch, preprocessor, list(owner.config.image_features)
        )
        if fixed_flow_seed is None and topology is None:
            loss = smolvla_flow_loss(policy, batch)
        elif fixed_flow_seed is None:
            flow_rng_state_sha256 = primary_rng_state_sha256(distributed_context)
            noise, flow_time, flow_input_sha256 = native_global_flow_inputs(
                policy,
                topology,
                distributed_context,
                action_shape=(spec["data"]["action_chunk_size"], owner.config.max_action_dim),
                device=next(owner.parameters()).device,
            )
            loss = smolvla_flow_loss(policy, batch, noise, flow_time)
        else:
            noise, flow_time = deterministic_flow_inputs(
                keys,
                action_shape=(spec["data"]["action_chunk_size"], owner.config.max_action_dim),
                noise_seed=fixed_flow_seed,
                time_seed=fixed_flow_seed + 1,
                device=torch.device("cuda"),
            )
            loss = smolvla_flow_loss(policy, batch, noise, flow_time)
        (loss / accumulation_steps).backward()
        contribution = loss.detach() / accumulation_steps
        mean_loss = contribution if mean_loss is None else mean_loss + contribution
    trainable = [value for value in policy.parameters() if value.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(
        trainable, spec["base_fit"]["gradient_clip_norm"]
    )
    if mean_loss is None or not torch.isfinite(grad_norm):
        raise GateZeroBaseRuntimeError("non-finite or empty source-base optimizer step")
    learning_rate = float(optimizer.param_groups[0]["lr"])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available() and next(owner.parameters()).is_cuda:
        torch.cuda.synchronize()
    local_wall_seconds = time.perf_counter() - started
    if topology is None:
        row_digest = hashlib.sha256()
        for key in local_row_keys:
            row_digest.update(key.encode("utf-8") + b"\0")
        row_summary = {
            "sha256": row_digest.hexdigest(),
            "global_slot_count": len(local_row_keys),
            "unique_global_slot_count": len(local_row_keys),
        }
        global_mean_loss = mean_loss
        wall_seconds = local_wall_seconds
    else:
        gathered = gather_rank_objects(local_row_keys, distributed_context)
        row_summary = (
            merge_rank_provenance(topology, gathered)
            if distributed_context.is_primary
            else None
        )
        row_summary = broadcast_primary_object(distributed_context, row_summary)
        global_mean_loss = distributed_mean(mean_loss, distributed_context)
        wall_seconds = distributed_max(
            local_wall_seconds,
            distributed_context,
            device=next(owner.parameters()).device,
        )
    return {
        "loss": float(global_mean_loss),
        "gradient_norm": float(grad_norm),
        "learning_rate_used": learning_rate,
        "row_keys_sha256": row_summary["sha256"],
        "global_slot_count": row_summary["global_slot_count"],
        "unique_global_slot_count": row_summary["unique_global_slot_count"],
        "flow_rng_state_sha256": flow_rng_state_sha256,
        "flow_input_sha256": flow_input_sha256,
        "wall_seconds": wall_seconds,
    }
