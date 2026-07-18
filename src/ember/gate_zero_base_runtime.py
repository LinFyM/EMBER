"""Shared source-base optimization mechanics for Gate 0 calibration and training."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch.utils.data import DataLoader

from ember.gate_zero_data import SourceHdf5Dataset, TaskDemoFrameBatchSampler
from ember.gate_zero_runtime import (
    batch_provenance_keys,
    deterministic_flow_inputs,
    preprocess_smolvla_batch,
    smolvla_flow_loss,
)


class GateZeroBaseRuntimeError(RuntimeError):
    """Raised when source-base optimization mechanics drift."""


def gradient_accumulation_steps(effective_batch_size: int, micro_batch_size: int) -> int:
    if effective_batch_size <= 0 or micro_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if effective_batch_size % micro_batch_size:
        raise ValueError("microbatch must divide the effective batch")
    return effective_batch_size // micro_batch_size


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
) -> DataLoader:
    accumulation_steps = gradient_accumulation_steps(effective_batch_size, micro_batch_size)
    sampler = TaskDemoFrameBatchSampler(
        dataset,
        micro_batch_size=micro_batch_size,
        optimizer_steps=optimizer_steps,
        gradient_accumulation_steps=accumulation_steps,
        seed=sampler_seed,
        start_optimizer_step=start_optimizer_step,
    )
    worker_generator = torch.Generator(device="cpu").manual_seed(
        sampler_seed + start_optimizer_step
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
) -> dict[str, Any]:
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    mean_loss: torch.Tensor | None = None
    row_digest = hashlib.sha256()
    for accumulation_step in range(accumulation_steps):
        raw_batch = next(iterator)
        keys = training_row_keys(
            raw_batch,
            optimizer_step=optimizer_step_index,
            effective_batch_start_slot=accumulation_step * len(raw_batch["task_id"]),
        )
        for key in keys:
            row_digest.update(key.encode("utf-8") + b"\0")
        batch = preprocess_smolvla_batch(
            raw_batch, preprocessor, list(policy.config.image_features)
        )
        if fixed_flow_seed is None:
            loss = smolvla_flow_loss(policy, batch)
        else:
            noise, flow_time = deterministic_flow_inputs(
                keys,
                action_shape=(spec["data"]["action_chunk_size"], policy.config.max_action_dim),
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
    torch.cuda.synchronize()
    return {
        "loss": float(mean_loss),
        "gradient_norm": float(grad_norm),
        "learning_rate_used": learning_rate,
        "row_keys_sha256": row_digest.hexdigest(),
        "wall_seconds": time.perf_counter() - started,
    }
