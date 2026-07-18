"""Live model, data, and optimizer session for the single Gate 0 oracle fitter."""

from __future__ import annotations

import contextlib
import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ember.evaluation_identity import _load_policy
from ember.gate_zero_data import (
    GateZeroSurface,
    SourceHdf5Dataset,
    TaskDemoFrameBatchSampler,
    load_surface_authorities,
)
from ember.gate_zero_oracle_metrics import FixedQueryEvaluator, FixedQueryReference
from ember.gate_zero_runtime import (
    batch_provenance_keys,
    build_lora_config,
    parameter_summary,
    physical_lora_deltas,
    preprocess_smolvla_batch,
    set_global_seed,
    smolvla_flow_loss,
)


MATURE_SUPPORT_QUERY_STAGES = frozenset(
    {
        "mature_positive_control",
        "mature_capacity_upper_bound",
        "mature_capacity_lr_recovery",
    }
)


class GateZeroOracleSessionError(RuntimeError):
    """Raised when live oracle model/data/training mechanics drift."""


@dataclass
class OracleModelSession:
    support_dataset: SourceHdf5Dataset
    evaluator: FixedQueryEvaluator
    model: Any
    preprocessor: Any
    optimizer: torch.optim.AdamW
    scheduler: Any | None
    reference: FixedQueryReference
    trainable_summary: dict[str, Any]
    task_authorities: list[dict[str, Any]]

    def close(self) -> None:
        self.evaluator.close()
        self.support_dataset.close()


def validate_fit_job(
    spec: dict[str, Any], *, variant: str, task_id: int
) -> dict[str, Any]:
    if variant not in spec.get("variants", []):
        raise GateZeroOracleSessionError("fit variant is not predeclared")
    if task_id not in spec.get("task_ids", []):
        raise GateZeroOracleSessionError("fit task is not predeclared")
    variant_spec = spec.get("fit", {}).get(variant)
    if not isinstance(variant_spec, dict):
        raise GateZeroOracleSessionError("fit variant contract is missing")
    return variant_spec


def capture_trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: value.detach().to(device="cpu", copy=True).contiguous()
        for name, value in model.named_parameters()
        if value.requires_grad
    }
    if not state:
        raise GateZeroOracleSessionError("oracle model has no trainable parameters")
    return state


def build_oracle_optimizer(
    model: torch.nn.Module, variant_spec: dict[str, Any]
) -> torch.optim.AdamW:
    parameters = [value for value in model.parameters() if value.requires_grad]
    if not parameters or variant_spec.get("optimizer") != "adamw":
        raise GateZeroOracleSessionError("oracle optimizer contract is invalid")
    return torch.optim.AdamW(
        parameters,
        lr=variant_spec["learning_rate"],
        betas=tuple(variant_spec["betas"]),
        eps=variant_spec["epsilon"],
        weight_decay=variant_spec["weight_decay"],
    )


def build_oracle_scheduler(
    optimizer: torch.optim.Optimizer,
    variant_spec: dict[str, Any],
    *,
    optimizer_steps: int,
) -> Any | None:
    """Build the optional SmolVLA-native warmup/cosine schedule."""

    scheduler = variant_spec.get("scheduler")
    if scheduler is None:
        return None
    if scheduler != "linear_warmup_cosine_decay" or optimizer_steps <= 0:
        raise GateZeroOracleSessionError("oracle scheduler contract is invalid")
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

    config = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=variant_spec["warmup_steps"],
        num_decay_steps=variant_spec["decay_steps"],
        peak_lr=variant_spec["learning_rate"],
        decay_lr=variant_spec["decay_learning_rate"],
    )
    return config.build(optimizer, num_training_steps=optimizer_steps)


def augment_support_images(
    batch: dict[str, Any],
    *,
    row_keys: list[str],
    optimizer_step: int,
    seed: int,
    scale_min: float,
    scale_max: float,
) -> dict[str, Any]:
    """Apply deterministic per-row random-resized crops to training cameras."""

    image_keys = sorted(key for key in batch if key.startswith("observation.images."))
    if not image_keys or optimizer_step <= 0 or not row_keys:
        raise GateZeroOracleSessionError("image augmentation authority is invalid")
    reference = batch[image_keys[0]]
    if reference.ndim != 4 or reference.shape[0] != len(row_keys):
        raise GateZeroOracleSessionError("image augmentation batch identity changed")
    if not (0 < scale_min <= scale_max <= 1.0):
        raise GateZeroOracleSessionError("image augmentation scale is invalid")
    batch_size, channels, height, width = reference.shape
    if channels != 3 or height <= 1 or width <= 1:
        raise GateZeroOracleSessionError("image augmentation shape is invalid")
    for key in image_keys:
        value = batch[key]
        if value.shape != reference.shape or value.dtype != torch.uint8:
            raise GateZeroOracleSessionError("training camera shape or dtype changed")
    augmented = {key: [] for key in image_keys}
    for index, row_key in enumerate(row_keys):
        digest = hashlib.sha256(
            f"{seed}\0{optimizer_step}\0{row_key}".encode("utf-8")
        ).digest()
        unit_scale = int.from_bytes(digest[:8], "little") / float(2**64 - 1)
        scale = scale_min + (scale_max - scale_min) * unit_scale
        crop_height = min(height, max(1, math.floor(height * math.sqrt(scale))))
        crop_width = min(width, max(1, math.floor(width * math.sqrt(scale))))
        top_slots = height - crop_height + 1
        left_slots = width - crop_width + 1
        top = int.from_bytes(digest[8:16], "little") % top_slots
        left = int.from_bytes(digest[16:24], "little") % left_slots
        cameras = torch.stack(
            [batch[key][index, :, top : top + crop_height, left : left + crop_width] for key in image_keys]
        ).to(dtype=torch.float32).div_(255.0)
        cameras = F.interpolate(
            cameras,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        for camera_index, key in enumerate(image_keys):
            augmented[key].append(cameras[camera_index])
    for key in image_keys:
        batch[key] = torch.stack(augmented[key])
    return batch


def _load_task_datasets(
    *,
    spec: dict[str, Any],
    parent: dict[str, Any],
    phase0: dict[str, Any],
    manifest: Path,
    dataset_root: Path,
    task_id: int,
) -> tuple[SourceHdf5Dataset, SourceHdf5Dataset, list[dict[str, Any]]]:
    support_authorities, parent_support_demos = load_surface_authorities(
        parent,
        phase0,
        manifest_path=manifest,
        dataset_root=dataset_root,
        surface=GateZeroSurface.SUPPORT,
        oracle_task_id=task_id,
    )
    query_authorities, parent_query_demos = load_surface_authorities(
        parent,
        phase0,
        manifest_path=manifest,
        dataset_root=dataset_root,
        surface=GateZeroSurface.QUERY,
        oracle_task_id=task_id,
    )
    if support_authorities != query_authorities:
        raise GateZeroOracleSessionError("support/query task authority differs")
    support_bounds = spec["fit"]["support_episode_bounds"]
    query_bounds = spec["selection"]["query_episode_bounds"]
    support_demos = list(range(support_bounds[0], support_bounds[1] + 1))
    query_demos = list(range(query_bounds[0], query_bounds[1] + 1))
    if spec.get("screening_stage") not in MATURE_SUPPORT_QUERY_STAGES:
        if support_demos != parent_support_demos or query_demos != parent_query_demos:
            raise GateZeroOracleSessionError("legacy support/query data authority changed")
    elif set(support_demos) & set(query_demos) or min(support_demos + query_demos) < 0 or max(
        support_demos + query_demos
    ) >= 50:
        raise GateZeroOracleSessionError("mature support/query isolation changed")
    support = SourceHdf5Dataset(
        support_authorities,
        demo_indices=support_demos,
        action_chunk_size=parent["data"]["action_chunk_size"],
        verify_sha256=True,
    )
    query = SourceHdf5Dataset(
        query_authorities,
        demo_indices=query_demos,
        action_chunk_size=parent["data"]["action_chunk_size"],
        verify_sha256=False,
    )
    evidence = [
        {
            "task_id": value.task_id,
            "hdf5_bytes": value.expected_bytes,
            "hdf5_sha256": value.expected_sha256,
            "language": value.language,
        }
        for value in support_authorities
    ]
    return support, query, evidence


def resolve_lora_variant_spec(
    *,
    parent: dict[str, Any],
    variant: str,
    variant_spec: dict[str, Any],
) -> dict[str, Any] | None:
    """Resolve legacy and named audit variants through one PEFT model path."""

    if variant == "lora":
        oracle = parent["oracle"]
        return {
            "target_modules": list(oracle["target_modules"]),
            "rank": variant_spec["rank"],
            "alpha": variant_spec["alpha"],
            "dropout": variant_spec["dropout"],
            "init_lora_weights": oracle["init_lora_weights"],
        }
    if variant_spec.get("adaptation_kind") != "lora":
        return None
    targets = variant_spec.get("target_modules")
    if (
        not isinstance(targets, list)
        or not targets
        or len(targets) != len(set(targets))
        or any(not isinstance(target, str) or not target for target in targets)
    ):
        raise GateZeroOracleSessionError("named LoRA variant has invalid targets")
    return {
        "target_modules": list(targets),
        "rank": variant_spec["rank"],
        "alpha": variant_spec["alpha"],
        "dropout": variant_spec["dropout"],
        "init_lora_weights": variant_spec["init_lora_weights"],
    }


def configure_oracle_variant(
    policy: Any,
    *,
    parent: dict[str, Any],
    checkpoint: dict[str, Any],
    variant: str,
    variant_spec: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    set_global_seed(variant_spec["seed"])
    lora = resolve_lora_variant_spec(
        parent=parent, variant=variant, variant_spec=variant_spec
    )
    if lora is not None:
        peft_config = build_lora_config(
            targets=lora["target_modules"],
            rank=lora["rank"],
            alpha=lora["alpha"],
            dropout=lora["dropout"],
            init_lora_weights=lora["init_lora_weights"],
            base_revision=parent["authority"]["model_revision"],
        )
        model = policy.wrap_with_peft(peft_config=peft_config)
        actual_targets = sorted(model.base_model.targeted_module_names)
        if actual_targets != sorted(lora["target_modules"]):
            raise GateZeroOracleSessionError("PEFT resolved a different target set")
        deltas = physical_lora_deltas(model, lora["target_modules"])
        if not deltas or any(torch.count_nonzero(value).item() for value in deltas.values()):
            raise GateZeroOracleSessionError("LoRA initialization is not an exact physical zero")
    else:
        model = policy
        actual = [
            {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in model.named_parameters()
            if value.requires_grad
        ]
        if actual != checkpoint.get("trainable_parameters"):
            raise GateZeroOracleSessionError("partial upper-bound trainable identity changed")
        actual_targets = [value["name"] for value in actual]
    model.train()
    summary = parameter_summary(model)
    if summary["trainable_parameters"] != variant_spec["expected_trainable_parameters"]:
        raise GateZeroOracleSessionError("oracle trainable parameter count changed")
    return model, {**summary, "resolved_targets": actual_targets}


def open_oracle_model_session(
    *,
    spec: dict[str, Any],
    parent: dict[str, Any],
    phase0: dict[str, Any],
    checkpoint: dict[str, Any],
    manifest: Path,
    dataset_root: Path,
    source_base_checkpoint: Path,
    variant: str,
    task_id: int,
    variant_spec: dict[str, Any],
) -> OracleModelSession:
    support, query, task_authorities = _load_task_datasets(
        spec=spec,
        parent=parent,
        phase0=phase0,
        manifest=manifest,
        dataset_root=dataset_root,
        task_id=task_id,
    )
    evaluator = None
    try:
        runtime = _load_policy(
            source_base_checkpoint / "pretrained_model",
            {"task_suite": "libero_90", "task_id": task_id},
        )
        policy, preprocessor = runtime[0], runtime[1]
        model, summary = configure_oracle_variant(
            policy,
            parent=parent,
            checkpoint=checkpoint,
            variant=variant,
            variant_spec=variant_spec,
        )
        selection = spec["selection"]
        evaluator = FixedQueryEvaluator(
            query,
            preprocessor=preprocessor,
            batch_size=selection["evaluation_batch_size"],
            num_workers=spec["fit"]["num_workers"],
            anchor_count_per_demo=selection["anchor_frames_per_demo"],
            action_chunk_size=parent["data"]["action_chunk_size"],
            fixed_noise_seed=selection["fixed_noise_seed"],
            fixed_time_seed=selection["fixed_time_seed"],
            inference_noise_seed=selection["inference_noise_seed"],
        )
        lora = resolve_lora_variant_spec(
            parent=parent, variant=variant, variant_spec=variant_spec
        )
        base_context = (
            model.disable_adapter() if lora is not None else contextlib.nullcontext()
        )
        with base_context:
            reference = evaluator.capture_base_reference(model)
        optimizer = build_oracle_optimizer(model, variant_spec)
        scheduler = build_oracle_scheduler(
            optimizer,
            variant_spec,
            optimizer_steps=spec["fit"]["optimizer_steps"],
        )
        return OracleModelSession(
            support,
            evaluator,
            model,
            preprocessor,
            optimizer,
            scheduler,
            reference,
            summary,
            task_authorities,
        )
    except BaseException:
        if evaluator is not None:
            evaluator.close()
        else:
            query.close()
        support.close()
        raise


def make_support_loader(
    session: OracleModelSession,
    *,
    fit: dict[str, Any],
    seed: int,
    start_step: int,
) -> DataLoader:
    remaining_steps = fit["optimizer_steps"] - start_step
    if remaining_steps < 0:
        raise GateZeroOracleSessionError("recovery step exceeds oracle budget")
    sampler = TaskDemoFrameBatchSampler(
        session.support_dataset,
        micro_batch_size=fit["micro_batch_size"],
        optimizer_steps=max(1, remaining_steps),
        gradient_accumulation_steps=fit["gradient_accumulation_steps"],
        seed=seed,
        start_optimizer_step=start_step,
        global_effective_batch_size=fit["effective_batch_size"],
    )
    kwargs: dict[str, Any] = {}
    if fit["num_workers"]:
        kwargs.update(
            prefetch_factor=fit["prefetch_factor"],
            persistent_workers=fit["persistent_workers"],
        )
    return DataLoader(
        session.support_dataset,
        batch_sampler=sampler,
        num_workers=fit["num_workers"],
        pin_memory=fit["pin_memory"],
        generator=torch.Generator(device="cpu").manual_seed(seed + start_step),
        **kwargs,
    )


def close_loader(loader: DataLoader | None) -> None:
    if loader is not None and (iterator := getattr(loader, "_iterator", None)) is not None:
        iterator._shutdown_workers()


def train_oracle_step(
    iterator: Any,
    *,
    session: OracleModelSession,
    gradient_clip_norm: float,
    optimizer_step: int,
    variant_spec: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    raw_batch = next(iterator)
    row_keys = batch_provenance_keys(raw_batch)
    if variant_spec.get("augmentation") is not None:
        if variant_spec["augmentation"] != "random_resized_crop":
            raise GateZeroOracleSessionError("unknown oracle training augmentation")
        raw_batch = augment_support_images(
            raw_batch,
            row_keys=row_keys,
            optimizer_step=optimizer_step,
            seed=variant_spec["augmentation_seed"],
            scale_min=variant_spec["augmentation_scale_min"],
            scale_max=variant_spec["augmentation_scale_max"],
        )
    model = session.model
    owner = model.get_base_model() if hasattr(model, "get_base_model") else model
    batch = preprocess_smolvla_batch(
        raw_batch, session.preprocessor, list(owner.config.image_features)
    )
    session.optimizer.zero_grad(set_to_none=True)
    loss = smolvla_flow_loss(model, batch)
    loss.backward()
    trainable = [value for value in model.parameters() if value.requires_grad]
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, gradient_clip_norm)
    if not torch.isfinite(gradient_norm):
        raise GateZeroOracleSessionError("oracle gradient norm is non-finite")
    learning_rate = float(session.optimizer.param_groups[0]["lr"])
    session.optimizer.step()
    if session.scheduler is not None:
        session.scheduler.step()
    session.optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    digest = hashlib.sha256()
    for key in row_keys:
        digest.update(key.encode("utf-8") + b"\0")
    return {
        "support_flow_loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm),
        "wall_seconds": wall_seconds,
        "samples_per_second": len(row_keys) / wall_seconds,
        "row_keys_sha256": digest.hexdigest(),
        "sample_count": len(row_keys),
        "unique_source_rows": len(set(row_keys)),
        "learning_rate": learning_rate,
        "next_learning_rate": float(session.optimizer.param_groups[0]["lr"]),
    }
