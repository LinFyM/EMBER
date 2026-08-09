"""Hashless atomic exact-resume checkpoints for the v6-prior Writer."""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior import (
    V6_PRIOR_FROZEN_PARAMETER_COUNT,
    V6_PRIOR_FROZEN_ROOTS,
    V6_PRIOR_TRAINABLE_PARAMETER_COUNT,
    V6_PRIOR_TRAINABLE_ROOTS,
    V6_WRITER_STATE_TENSOR_COUNT,
    configure_v6_prior_trainability,
)
from ember.pi05_source_checkpoint import (
    DistributedContext,
    read_json,
    write_json_atomic,
)
from ember.writer.model import CompleteLoRAWriter


V6_PRIOR_CHECKPOINT_SCHEMA = "ember_pi05_v6_prior_writer_checkpoint_v1"
V6_PRIOR_TRAINER_SCHEMA = "ember_pi05_v6_prior_writer_trainer_v1"
V6_PRIOR_RNG_SCHEMA = "ember_pi05_v6_prior_writer_rank_rng_v1"
V6_PRIOR_CHECKPOINT_INSPECTION_SCHEMA = "ember_pi05_v6_prior_checkpoint_inspection_v1"
V6_PRIOR_CHECKPOINT_COMPARISON_SCHEMA = "ember_pi05_v6_prior_checkpoint_comparison_v1"
V6_PRIOR_WORLD_SIZE = 6
V6_PRIOR_FROZEN_PARAMETER_TENSOR_COUNT = 482
V6_PRIOR_FROZEN_STATE_TENSOR_COUNT = 483
V6_PRIOR_TRAINABLE_TENSOR_COUNT = 41
V6_PRIOR_TEMPLATE_TENSOR_COUNT = 76
_CURSOR_KEYS = {
    "next_macro",
    "task_visits_per_task",
    "sampler_seed",
    "teacher_video_seed",
    "counterfactual_seed",
    "counterfactual_phase",
    "videos_per_task_visit",
    "action_queries_per_task",
}
_CHECKPOINT_CONTRACT_KEYS = {
    "run_schema",
    "mode",
    "git_commit",
    "config",
    "source",
    "initialization",
    "expert_bank_root",
    "expert_step",
    "objective",
    "ownership",
    "world_size",
}


@dataclass(frozen=True)
class _CheckpointInspection:
    summary: dict[str, Any]
    manifest: dict[str, Any]
    writer: dict[str, torch.Tensor]
    trainer: dict[str, Any]
    rng_by_rank: tuple[dict[str, Any], ...]
    trainable_names: tuple[str, ...]


def _barrier(context: DistributedContext) -> None:
    if context.world_size > 1:
        dist.barrier(device_ids=[context.local_rank])


def _rng_state(context: DistributedContext) -> dict[str, Any]:
    cuda = (
        torch.cuda.get_rng_state(context.device)
        if context.device.type == "cuda"
        else None
    )
    return {
        "schema_version": V6_PRIOR_RNG_SCHEMA,
        "rank": context.rank,
        "world_size": context.world_size,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda,
    }


def _restore_rng(value: Mapping[str, Any], context: DistributedContext) -> None:
    if (
        value.get("schema_version") != V6_PRIOR_RNG_SCHEMA
        or int(value.get("rank", -1)) != context.rank
        or int(value.get("world_size", -1)) != context.world_size
        or (context.device.type == "cuda") != (value.get("torch_cuda") is not None)
    ):
        raise ExpertManifoldError("v6-prior rank RNG state changed")
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"].cpu())
    if context.device.type == "cuda":
        torch.cuda.set_rng_state(value["torch_cuda"].cpu(), context.device)


def _move_optimizer_state_(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for values in optimizer.state.values():
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                values[name] = value.to(device)


def save_v6_prior_checkpoint(
    *,
    output_dir: Path,
    macro: int,
    writer: CompleteLoRAWriter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    context: DistributedContext,
    metrics_rows: int,
    cursor_contract: Mapping[str, Any],
    checkpoint_contract: Mapping[str, Any],
) -> Path:
    """Publish one complete checkpoint only at a macro boundary."""

    if (
        macro <= 0
        or metrics_rows != macro
        or int(cursor_contract.get("next_macro", -1)) != macro
    ):
        raise ExpertManifoldError("v6-prior checkpoint cursor changed")
    checkpoints = output_dir / "checkpoints"
    final = checkpoints / f"macro_{macro:08d}"
    temporary = checkpoints / f".macro_{macro:08d}.tmp"
    if context.is_main:
        checkpoints.mkdir(parents=True, exist_ok=True)
        if final.exists() or temporary.exists():
            raise ExpertManifoldError("v6-prior checkpoint already exists")
        temporary.mkdir()
    _barrier(context)
    saved_rng = _rng_state(context)
    rng_name = f"rng_rank_{context.rank:03d}.pt"
    torch.save(saved_rng, temporary / rng_name)
    _barrier(context)
    if context.is_main:
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in writer.state_dict().items()
        }
        if len(state) != V6_WRITER_STATE_TENSOR_COUNT:
            raise ExpertManifoldError("v6-prior checkpoint Writer state changed")
        writer_path = temporary / "writer.safetensors"
        trainer_path = temporary / "trainer.pt"
        save_file(state, str(writer_path))
        torch.save(
            {
                "schema_version": V6_PRIOR_TRAINER_SCHEMA,
                "next_macro": macro,
                "metrics_rows": metrics_rows,
                "world_size": context.world_size,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "amp_scaler": {"enabled": False, "state": {}},
            },
            trainer_path,
        )
        files = {
            "writer.safetensors": writer_path.stat().st_size,
            "trainer.pt": trainer_path.stat().st_size,
            **{
                f"rng_rank_{rank:03d}.pt": (temporary / f"rng_rank_{rank:03d}.pt")
                .stat()
                .st_size
                for rank in range(context.world_size)
            },
        }
        write_json_atomic(
            temporary / "manifest.json",
            {
                "schema_version": V6_PRIOR_CHECKPOINT_SCHEMA,
                "next_macro": macro,
                "metrics_rows": metrics_rows,
                "world_size": context.world_size,
                "cursor_contract": dict(cursor_contract),
                "checkpoint_contract": dict(checkpoint_contract),
                "files": files,
                "content_hash_policy": "disabled_by_owner",
            },
        )
        os.replace(temporary, final)
    _barrier(context)
    _restore_rng(saved_rng, context)
    return final


def load_v6_prior_checkpoint(
    *,
    checkpoint: Path,
    writer: CompleteLoRAWriter,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    context: DistributedContext,
    expected_cursor_contract: Mapping[str, Any],
    expected_checkpoint_contract: Mapping[str, Any],
) -> tuple[int, int]:
    """Restore only this method's exact-resume schema, never historical v6 state."""

    checkpoint = checkpoint.resolve()
    manifest = read_json(checkpoint / "manifest.json")
    macro = int(manifest.get("next_macro", -1))
    expected_files = {
        "writer.safetensors",
        "trainer.pt",
        *(f"rng_rank_{rank:03d}.pt" for rank in range(context.world_size)),
    }
    files = manifest.get("files", {})
    if (
        manifest.get("schema_version") != V6_PRIOR_CHECKPOINT_SCHEMA
        or macro <= 0
        or checkpoint.name != f"macro_{macro:08d}"
        or int(manifest.get("world_size", -1)) != context.world_size
        or int(manifest.get("metrics_rows", -1)) != macro
        or manifest.get("cursor_contract") != dict(expected_cursor_contract)
        or manifest.get("checkpoint_contract") != dict(expected_checkpoint_contract)
        or set(files) != expected_files
        or manifest.get("content_hash_policy") != "disabled_by_owner"
    ):
        raise ExpertManifoldError("v6-prior checkpoint manifest changed")
    for name, expected_bytes in files.items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(expected_bytes):
            raise ExpertManifoldError("v6-prior checkpoint file changed")
    state = load_file(str(checkpoint / "writer.safetensors"), device="cpu")
    if len(state) != V6_WRITER_STATE_TENSOR_COUNT:
        raise ExpertManifoldError("v6-prior resume Writer state changed")
    writer.load_state_dict(state, strict=True)
    configure_v6_prior_trainability(writer)
    trainer = torch.load(
        checkpoint / "trainer.pt",
        map_location="cpu",
        weights_only=False,
    )
    if (
        trainer.get("schema_version") != V6_PRIOR_TRAINER_SCHEMA
        or int(trainer.get("next_macro", -1)) != macro
        or int(trainer.get("metrics_rows", -1)) != macro
        or int(trainer.get("world_size", -1)) != context.world_size
        or trainer.get("amp_scaler") != {"enabled": False, "state": {}}
    ):
        raise ExpertManifoldError("v6-prior trainer state changed")
    optimizer.load_state_dict(trainer["optimizer"])
    _move_optimizer_state_(optimizer, context.device)
    scheduler.load_state_dict(trainer["scheduler"])
    rng = torch.load(
        checkpoint / f"rng_rank_{context.rank:03d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    _restore_rng(rng, context)
    return macro, int(trainer["metrics_rows"])


def _inspection_error(component: str) -> ExpertManifoldError:
    return ExpertManifoldError(f"v6-prior checkpoint inspection failed: {component}")


def _strict_int(value: object) -> int | None:
    return value if type(value) is int else None


def _load_mapping(path: Path, component: str) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise _inspection_error(component) from error
    if not isinstance(value, dict):
        raise _inspection_error(component)
    return value


def _finite_scalar_tree(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(
            value.numel() > 0
            and (
                not (value.is_floating_point() or value.is_complex())
                or torch.isfinite(value).all().item()
            )
        )
    if isinstance(value, np.ndarray):
        return bool(
            value.size > 0
            and (not np.issubdtype(value.dtype, np.inexact) or np.isfinite(value).all())
        )
    if isinstance(value, Mapping):
        return all(_finite_scalar_tree(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_scalar_tree(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return value is None or isinstance(value, (str, bool, int, np.integer))


def _validate_cursor(cursor: object, macro: int) -> dict[str, Any]:
    if not isinstance(cursor, Mapping) or set(cursor) != _CURSOR_KEYS:
        raise _inspection_error("cursor contract")
    integers = {name: _strict_int(cursor.get(name)) for name in _CURSOR_KEYS}
    if (
        any(value is None for value in integers.values())
        or integers["next_macro"] != macro
        or integers["task_visits_per_task"] != macro
        or integers["counterfactual_phase"] != macro % 3
        or integers["videos_per_task_visit"] != 1
        or integers["action_queries_per_task"] != 20
        or any(
            integers[name] < 0
            for name in (
                "sampler_seed",
                "teacher_video_seed",
                "counterfactual_seed",
            )
        )
    ):
        raise _inspection_error("cursor contract")
    return dict(cursor)


def _validate_checkpoint_contract(
    contract: object,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    if not isinstance(contract, Mapping) or set(contract) != _CHECKPOINT_CONTRACT_KEYS:
        raise _inspection_error("checkpoint contract")
    config = contract.get("config")
    source = contract.get("source")
    initialization = contract.get("initialization")
    objective = contract.get("objective")
    ownership = contract.get("ownership")
    if not all(
        isinstance(value, Mapping)
        for value in (config, source, initialization, objective, ownership)
    ):
        raise _inspection_error("checkpoint contract")
    names = ownership.get("trainable_tensor_names")
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise _inspection_error("checkpoint ownership")
    names_tuple = tuple(names)
    if (
        len(names_tuple) != V6_PRIOR_TRAINABLE_TENSOR_COUNT
        or len(set(names_tuple)) != len(names_tuple)
        or any(
            name.split(".", 1)[0] not in V6_PRIOR_TRAINABLE_ROOTS
            for name in names_tuple
        )
        or contract.get("run_schema") != "ember_pi05_v6_prior_writer_launch_v1"
        or contract.get("mode") not in {"profile", "formal"}
        or not isinstance(contract.get("git_commit"), str)
        or len(str(contract.get("git_commit"))) < 7
        or not isinstance(config.get("path"), str)
        or config.get("schema") != "ember_pi05_v6_prior_policy_effective_writer_v1"
        or (_strict_int(config.get("bytes")) or -1) <= 0
        or not source
        or initialization.get("mode") != "historical_v6_macro400_load_only"
        or not isinstance(initialization.get("checkpoint"), str)
        or not str(initialization.get("checkpoint"))
        or _strict_int(initialization.get("writer_state_tensor_count"))
        != V6_WRITER_STATE_TENSOR_COUNT
        or (_strict_int(initialization.get("writer_state_value_count")) or -1) <= 0
        or initialization.get("optimizer") != "fresh"
        or initialization.get("scheduler") != "fresh"
        or initialization.get("rng") != "fresh_seed"
        or not isinstance(contract.get("expert_bank_root"), str)
        or not str(contract.get("expert_bank_root"))
        or _strict_int(contract.get("expert_step")) != 2000
        or not objective
        or _strict_int(contract.get("world_size")) != V6_PRIOR_WORLD_SIZE
        or _strict_int(ownership.get("frozen_parameter_count"))
        != V6_PRIOR_FROZEN_PARAMETER_COUNT
        or _strict_int(ownership.get("trainable_parameter_count"))
        != V6_PRIOR_TRAINABLE_PARAMETER_COUNT
        or _strict_int(ownership.get("frozen_tensor_count"))
        != V6_PRIOR_FROZEN_PARAMETER_TENSOR_COUNT
        or _strict_int(ownership.get("trainable_tensor_count"))
        != V6_PRIOR_TRAINABLE_TENSOR_COUNT
        or _strict_int(ownership.get("source_policy_trainable_parameter_count")) != 0
    ):
        raise _inspection_error("checkpoint contract")
    return dict(contract), names_tuple


def _validate_writer(
    path: Path,
    trainable_names: Sequence[str],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    try:
        state = load_file(str(path), device="cpu")
    except Exception as error:
        raise _inspection_error("Writer state") from error
    trainable = set(trainable_names)
    expected_templates = {
        f"template_{index:03d}" for index in range(V6_PRIOR_TEMPLATE_TENSOR_COUNT)
    }
    observed_templates = {name for name in state if name.startswith("template_")}
    observed_trainable = {
        name for name in state if name.split(".", 1)[0] in V6_PRIOR_TRAINABLE_ROOTS
    }
    frozen_state = {
        name for name in state if name.split(".", 1)[0] in V6_PRIOR_FROZEN_ROOTS
    }
    frozen_buffers = {"semantic_encoder.fixed_suffix_noise"}
    frozen_parameters = frozen_state - frozen_buffers
    allowed_names = expected_templates | observed_trainable | frozen_state
    if (
        len(state) != V6_WRITER_STATE_TENSOR_COUNT
        or observed_templates != expected_templates
        or observed_trainable != trainable
        or len(observed_trainable) != V6_PRIOR_TRAINABLE_TENSOR_COUNT
        or len(frozen_state) != V6_PRIOR_FROZEN_STATE_TENSOR_COUNT
        or len(frozen_parameters) != V6_PRIOR_FROZEN_PARAMETER_TENSOR_COUNT
        or frozen_buffers - set(state)
        or set(state) != allowed_names
    ):
        raise _inspection_error("Writer tensor topology")
    dtype_counts: dict[str, int] = {}
    for name, value in state.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.numel() <= 0
            or not value.is_floating_point()
            or not torch.isfinite(value).all().item()
        ):
            raise _inspection_error(f"Writer tensor {name}")
        dtype = str(value.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    if (
        sum(state[name].numel() for name in observed_trainable)
        != V6_PRIOR_TRAINABLE_PARAMETER_COUNT
        or sum(state[name].numel() for name in frozen_parameters)
        != V6_PRIOR_FROZEN_PARAMETER_COUNT
    ):
        raise _inspection_error("Writer tensor shapes")
    return state, {
        "state_tensor_count": len(state),
        "state_value_count": sum(value.numel() for value in state.values()),
        "trainable_tensor_count": len(observed_trainable),
        "trainable_parameter_count": sum(
            state[name].numel() for name in observed_trainable
        ),
        "frozen_state_tensor_count": len(frozen_state),
        "frozen_parameter_tensor_count": len(frozen_parameters),
        "frozen_parameter_count": sum(
            state[name].numel() for name in frozen_parameters
        ),
        "template_tensor_count": len(observed_templates),
        "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
        "finite": True,
    }


def _validate_python_rng(value: object) -> bool:
    if not isinstance(value, tuple) or len(value) != 3:
        return False
    version, internal, gaussian = value
    return bool(
        version in {2, 3}
        and isinstance(internal, tuple)
        and len(internal) == 625
        and all(type(item) is int for item in internal)
        and (gaussian is None or isinstance(gaussian, float))
        and (gaussian is None or math.isfinite(gaussian))
    )


def _validate_numpy_rng(value: object) -> bool:
    if not isinstance(value, tuple) or len(value) != 5:
        return False
    kind, state, position, has_gaussian, cached_gaussian = value
    return bool(
        kind == "MT19937"
        and isinstance(state, np.ndarray)
        and state.shape == (624,)
        and state.dtype == np.uint32
        and type(position) is int
        and 0 <= position <= 624
        and type(has_gaussian) is int
        and has_gaussian in {0, 1}
        and isinstance(cached_gaussian, float)
        and math.isfinite(cached_gaussian)
    )


def _validate_rng(path: Path, rank: int) -> dict[str, Any]:
    value = _load_mapping(path, f"rank {rank} RNG")
    cpu = value.get("torch_cpu")
    cuda = value.get("torch_cuda")
    if (
        set(value)
        != {
            "schema_version",
            "rank",
            "world_size",
            "python",
            "numpy",
            "torch_cpu",
            "torch_cuda",
        }
        or value.get("schema_version") != V6_PRIOR_RNG_SCHEMA
        or _strict_int(value.get("rank")) != rank
        or _strict_int(value.get("world_size")) != V6_PRIOR_WORLD_SIZE
        or not _validate_python_rng(value.get("python"))
        or not _validate_numpy_rng(value.get("numpy"))
        or not isinstance(cpu, torch.Tensor)
        or cpu.dtype != torch.uint8
        or cpu.ndim != 1
        or cpu.numel() <= 0
        or not isinstance(cuda, torch.Tensor)
        or cuda.dtype != torch.uint8
        or cuda.ndim != 1
        or cuda.numel() <= 0
    ):
        raise _inspection_error(f"rank {rank} RNG")
    return value


def _validate_optimizer(
    optimizer: object,
    *,
    macro: int,
    trainable_names: Sequence[str],
    writer: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if not isinstance(optimizer, Mapping) or set(optimizer) != {
        "state",
        "param_groups",
    }:
        raise _inspection_error("optimizer state")
    state = optimizer.get("state")
    groups = optimizer.get("param_groups")
    if (
        not isinstance(state, Mapping)
        or not isinstance(groups, list)
        or len(groups) != 1
    ):
        raise _inspection_error("optimizer state")
    group = groups[0]
    if not isinstance(group, Mapping) or not _finite_scalar_tree(group):
        raise _inspection_error("optimizer parameter group")
    parameter_ids = group.get("params")
    if (
        not isinstance(parameter_ids, list)
        or len(parameter_ids) != V6_PRIOR_TRAINABLE_TENSOR_COUNT
        or any(type(index) is not int for index in parameter_ids)
        or len(set(parameter_ids)) != len(parameter_ids)
        or set(state) != set(parameter_ids)
    ):
        raise _inspection_error("optimizer parameter topology")
    amsgrad = group.get("amsgrad") is True
    expected_state_keys = {"step", "exp_avg", "exp_avg_sq"}
    if amsgrad:
        expected_state_keys.add("max_exp_avg_sq")
    for parameter_id, name in zip(parameter_ids, trainable_names, strict=True):
        values = state.get(parameter_id)
        if not isinstance(values, Mapping) or set(values) != expected_state_keys:
            raise _inspection_error(f"optimizer state for {name}")
        step = values.get("step")
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or not step.is_floating_point()
            or not torch.isfinite(step).all().item()
            or float(step.item()) != float(macro)
        ):
            raise _inspection_error(f"optimizer step for {name}")
        for field in expected_state_keys - {"step"}:
            tensor = values.get(field)
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != tuple(writer[name].shape)
                or not tensor.is_floating_point()
                or not torch.isfinite(tensor).all().item()
            ):
                raise _inspection_error(f"optimizer {field} for {name}")
    return {
        "parameter_group_count": len(groups),
        "parameter_count": len(parameter_ids),
        "state_count": len(state),
    }


def _validate_trainer(
    path: Path,
    *,
    macro: int,
    trainable_names: Sequence[str],
    writer: Mapping[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, Any]]:
    trainer = _load_mapping(path, "trainer state")
    if (
        set(trainer)
        != {
            "schema_version",
            "next_macro",
            "metrics_rows",
            "world_size",
            "optimizer",
            "scheduler",
            "amp_scaler",
        }
        or trainer.get("schema_version") != V6_PRIOR_TRAINER_SCHEMA
        or _strict_int(trainer.get("next_macro")) != macro
        or _strict_int(trainer.get("metrics_rows")) != macro
        or _strict_int(trainer.get("world_size")) != V6_PRIOR_WORLD_SIZE
        or trainer.get("amp_scaler") != {"enabled": False, "state": {}}
    ):
        raise _inspection_error("trainer state")
    optimizer_summary = _validate_optimizer(
        trainer.get("optimizer"),
        macro=macro,
        trainable_names=trainable_names,
        writer=writer,
    )
    scheduler = trainer.get("scheduler")
    if (
        not isinstance(scheduler, Mapping)
        or _strict_int(scheduler.get("last_epoch")) != macro
        or _strict_int(scheduler.get("_step_count")) != macro + 1
        or not _finite_scalar_tree(scheduler)
    ):
        raise _inspection_error("scheduler state")
    groups = trainer["optimizer"]["param_groups"]
    last_lr = scheduler.get("_last_lr")
    if (
        not isinstance(last_lr, list)
        or len(last_lr) != len(groups)
        or any(
            not math.isclose(
                float(lr), float(group.get("lr")), rel_tol=0.0, abs_tol=0.0
            )
            for lr, group in zip(last_lr, groups, strict=True)
        )
    ):
        raise _inspection_error("scheduler learning rate")
    return trainer, {
        "optimizer": optimizer_summary,
        "scheduler_last_epoch": macro,
        "scheduler_step_count": macro + 1,
        "amp_scaler": {"enabled": False, "state": {}},
    }


def _inspect_v6_prior_checkpoint(checkpoint: Path) -> _CheckpointInspection:
    checkpoint = checkpoint.resolve()
    if (
        not checkpoint.is_dir()
        or checkpoint.parent.name != "checkpoints"
        or checkpoint.is_symlink()
    ):
        raise _inspection_error("checkpoint directory")
    try:
        manifest = read_json(checkpoint / "manifest.json")
    except Exception as error:
        raise _inspection_error("manifest") from error
    macro = _strict_int(manifest.get("next_macro"))
    files = manifest.get("files")
    expected_files = {
        "writer.safetensors",
        "trainer.pt",
        *(f"rng_rank_{rank:03d}.pt" for rank in range(V6_PRIOR_WORLD_SIZE)),
    }
    if (
        set(manifest)
        != {
            "schema_version",
            "next_macro",
            "metrics_rows",
            "world_size",
            "cursor_contract",
            "checkpoint_contract",
            "files",
            "content_hash_policy",
        }
        or manifest.get("schema_version") != V6_PRIOR_CHECKPOINT_SCHEMA
        or macro is None
        or macro <= 0
        or checkpoint.name != f"macro_{macro:08d}"
        or _strict_int(manifest.get("metrics_rows")) != macro
        or _strict_int(manifest.get("world_size")) != V6_PRIOR_WORLD_SIZE
        or manifest.get("content_hash_policy") != "disabled_by_owner"
        or not isinstance(files, Mapping)
        or set(files) != expected_files
    ):
        raise _inspection_error("manifest")
    file_rows: list[dict[str, Any]] = []
    for name in sorted(expected_files):
        expected_bytes = _strict_int(files.get(name))
        path = checkpoint / name
        if (
            expected_bytes is None
            or expected_bytes <= 0
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected_bytes
        ):
            raise _inspection_error(f"declared file {name}")
        file_rows.append({"name": name, "bytes": expected_bytes})
    cursor = _validate_cursor(manifest.get("cursor_contract"), macro)
    contract, trainable_names = _validate_checkpoint_contract(
        manifest.get("checkpoint_contract")
    )
    writer, writer_summary = _validate_writer(
        checkpoint / "writer.safetensors", trainable_names
    )
    if (
        writer_summary["state_value_count"]
        != contract["initialization"]["writer_state_value_count"]
    ):
        raise _inspection_error("Writer tensor shapes")
    trainer, trainer_summary = _validate_trainer(
        checkpoint / "trainer.pt",
        macro=macro,
        trainable_names=trainable_names,
        writer=writer,
    )
    rng_by_rank = tuple(
        _validate_rng(checkpoint / f"rng_rank_{rank:03d}.pt", rank)
        for rank in range(V6_PRIOR_WORLD_SIZE)
    )
    summary = {
        "schema_version": V6_PRIOR_CHECKPOINT_INSPECTION_SCHEMA,
        "checkpoint": str(checkpoint),
        "checkpoint_schema": V6_PRIOR_CHECKPOINT_SCHEMA,
        "next_macro": macro,
        "metrics_rows": macro,
        "world_size": V6_PRIOR_WORLD_SIZE,
        "cursor_contract": cursor,
        "checkpoint_contract": contract,
        "files": file_rows,
        "writer": writer_summary,
        "trainer": trainer_summary,
        "rng": {
            "schema_version": V6_PRIOR_RNG_SCHEMA,
            "rank_count": len(rng_by_rank),
            "ranks": list(range(V6_PRIOR_WORLD_SIZE)),
            "cuda_state_present": True,
        },
        "content_hash_policy": "disabled_by_owner",
    }
    return _CheckpointInspection(
        summary=summary,
        manifest=manifest,
        writer=writer,
        trainer=trainer,
        rng_by_rank=rng_by_rank,
        trainable_names=trainable_names,
    )


def inspect_v6_prior_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Read and validate a six-rank checkpoint without restoring any live state."""

    return _inspect_v6_prior_checkpoint(checkpoint).summary


def _exact_semantic_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return bool(
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return bool(
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_exact_semantic_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return bool(
            type(left) is type(right)
            and len(left) == len(right)  # type: ignore[arg-type]
            and all(
                _exact_semantic_equal(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)  # type: ignore[arg-type]
            )
        )
    return type(left) is type(right) and left == right


@dataclass
class _DifferenceAccumulator:
    sum_squared_difference: float = 0.0
    sum_squared_left: float = 0.0
    sum_squared_right: float = 0.0
    max_abs: float = 0.0
    worst: str | None = None
    tensor_count: int = 0

    def add(self, path: str, left: torch.Tensor, right: torch.Tensor) -> None:
        difference = left.to(torch.float64) - right.to(torch.float64)
        current = float(difference.abs().max().item())
        if current > self.max_abs or self.worst is None:
            self.max_abs = current
            self.worst = path
        self.sum_squared_difference += float(difference.square().sum().item())
        self.sum_squared_left += float(left.to(torch.float64).square().sum().item())
        self.sum_squared_right += float(right.to(torch.float64).square().sum().item())
        self.tensor_count += 1

    def summary(self) -> dict[str, Any]:
        denominator = max(
            math.sqrt(self.sum_squared_left),
            math.sqrt(self.sum_squared_right),
        )
        relative = (
            math.sqrt(self.sum_squared_difference) / denominator
            if denominator > 0.0
            else (0.0 if self.sum_squared_difference == 0.0 else math.inf)
        )
        return {
            "tensor_count": self.tensor_count,
            "max_abs": self.max_abs,
            "global_relative_l2": relative,
            "worst_tensor": self.worst,
        }


def _compare_writer_states(
    left: _CheckpointInspection,
    right: _CheckpointInspection,
    *,
    scientific_atol: float,
    scientific_rtol: float,
    max_abs_tolerance: float,
    relative_l2_tolerance: float,
) -> dict[str, Any]:
    if set(left.writer) != set(right.writer):
        raise _inspection_error("compared Writer names")
    trainable = set(left.trainable_names)
    if trainable != set(right.trainable_names):
        raise _inspection_error("compared Writer ownership")
    accumulator = _DifferenceAccumulator()
    frozen_count = 0
    for name in sorted(left.writer):
        left_value = left.writer[name]
        right_value = right.writer[name]
        if left_value.dtype != right_value.dtype or tuple(left_value.shape) != tuple(
            right_value.shape
        ):
            raise _inspection_error(f"compared Writer schema for {name}")
        if name not in trainable:
            frozen_count += 1
            if not torch.equal(left_value, right_value):
                raise _inspection_error(f"compared frozen Writer tensor {name}")
            continue
        accumulator.add(name, left_value, right_value)
        if not torch.allclose(
            left_value,
            right_value,
            atol=scientific_atol,
            rtol=scientific_rtol,
        ):
            raise _inspection_error(f"compared trainable Writer tensor {name}")
    summary = accumulator.summary()
    if (
        float(summary["max_abs"]) > max_abs_tolerance
        or float(summary["global_relative_l2"]) > relative_l2_tolerance
    ):
        raise _inspection_error("compared trainable Writer tolerance")
    return {
        "tensor_schema_equal": True,
        "state_tensor_count": len(left.writer),
        "frozen_exact": True,
        "frozen_tensor_count": frozen_count,
        "trainable_tensor_count": len(trainable),
        "scientific_atol": scientific_atol,
        "scientific_rtol": scientific_rtol,
        "max_abs_tolerance": max_abs_tolerance,
        "global_relative_l2_tolerance": relative_l2_tolerance,
        **summary,
    }


def _compare_optimizer_states(
    left: _CheckpointInspection,
    right: _CheckpointInspection,
    *,
    scientific_atol: float,
    scientific_rtol: float,
    max_abs_tolerance: float,
    relative_l2_tolerance: float,
) -> dict[str, Any]:
    left_optimizer = left.trainer["optimizer"]
    right_optimizer = right.trainer["optimizer"]
    if not _exact_semantic_equal(
        left_optimizer["param_groups"], right_optimizer["param_groups"]
    ):
        raise _inspection_error("compared optimizer parameter groups")
    left_state = left_optimizer["state"]
    right_state = right_optimizer["state"]
    if set(left_state) != set(right_state):
        raise _inspection_error("compared optimizer state topology")
    accumulator = _DifferenceAccumulator()
    field_accumulators: dict[str, _DifferenceAccumulator] = {}
    for parameter_id in left_optimizer["param_groups"][0]["params"]:
        left_values = left_state[parameter_id]
        right_values = right_state[parameter_id]
        if set(left_values) != set(right_values):
            raise _inspection_error("compared optimizer state topology")
        for field in sorted(left_values):
            left_value = left_values[field]
            right_value = right_values[field]
            path = f"parameter_{parameter_id}.{field}"
            if field == "step":
                if not _exact_semantic_equal(left_value, right_value):
                    raise _inspection_error(f"compared optimizer {path}")
                continue
            if (
                not isinstance(left_value, torch.Tensor)
                or not isinstance(right_value, torch.Tensor)
                or left_value.dtype != right_value.dtype
                or tuple(left_value.shape) != tuple(right_value.shape)
            ):
                raise _inspection_error(f"compared optimizer {path}")
            accumulator.add(path, left_value, right_value)
            field_accumulators.setdefault(field, _DifferenceAccumulator()).add(
                path, left_value, right_value
            )
            if not torch.allclose(
                left_value,
                right_value,
                atol=scientific_atol,
                rtol=scientific_rtol,
            ):
                raise _inspection_error(f"compared optimizer {path}")
    field_summaries = {
        field: field_accumulator.summary()
        for field, field_accumulator in sorted(field_accumulators.items())
    }
    for field, summary in field_summaries.items():
        if (
            float(summary["max_abs"]) > max_abs_tolerance
            or float(summary["global_relative_l2"]) > relative_l2_tolerance
        ):
            raise _inspection_error(f"compared optimizer {field} tolerance")
    return {
        "param_groups_equal": True,
        "scientific_atol": scientific_atol,
        "scientific_rtol": scientific_rtol,
        "max_abs_tolerance": max_abs_tolerance,
        "global_relative_l2_tolerance": relative_l2_tolerance,
        "moment_fields": field_summaries,
        **accumulator.summary(),
    }


def compare_v6_prior_checkpoints(
    left: Path,
    right: Path,
    *,
    scientific_atol: float = 2e-4,
    scientific_rtol: float = 2e-3,
    writer_max_abs_tolerance: float = 7.5e-6,
    writer_relative_l2_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Compare fresh/resumed checkpoints without mutating any live training state."""

    tolerances = (
        scientific_atol,
        scientific_rtol,
        writer_max_abs_tolerance,
        writer_relative_l2_tolerance,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
        raise _inspection_error("comparison tolerances")
    left_inspection = _inspect_v6_prior_checkpoint(left)
    right_inspection = _inspect_v6_prior_checkpoint(right)
    if left_inspection.summary["next_macro"] != right_inspection.summary["next_macro"]:
        raise _inspection_error("compared macro")
    if not _exact_semantic_equal(
        left_inspection.manifest["cursor_contract"],
        right_inspection.manifest["cursor_contract"],
    ):
        raise _inspection_error("compared cursor contract")
    if not _exact_semantic_equal(
        left_inspection.manifest["checkpoint_contract"],
        right_inspection.manifest["checkpoint_contract"],
    ):
        raise _inspection_error("compared checkpoint contract")
    for rank, (left_rng, right_rng) in enumerate(
        zip(
            left_inspection.rng_by_rank,
            right_inspection.rng_by_rank,
            strict=True,
        )
    ):
        if not _exact_semantic_equal(left_rng, right_rng):
            raise _inspection_error(f"compared rank {rank} RNG")
    if not _exact_semantic_equal(
        left_inspection.trainer["scheduler"],
        right_inspection.trainer["scheduler"],
    ):
        raise _inspection_error("compared scheduler")
    if not _exact_semantic_equal(
        left_inspection.trainer["amp_scaler"],
        right_inspection.trainer["amp_scaler"],
    ):
        raise _inspection_error("compared AMP scaler")
    writer_summary = _compare_writer_states(
        left_inspection,
        right_inspection,
        scientific_atol=scientific_atol,
        scientific_rtol=scientific_rtol,
        max_abs_tolerance=writer_max_abs_tolerance,
        relative_l2_tolerance=writer_relative_l2_tolerance,
    )
    optimizer_summary = _compare_optimizer_states(
        left_inspection,
        right_inspection,
        scientific_atol=scientific_atol,
        scientific_rtol=scientific_rtol,
        max_abs_tolerance=writer_max_abs_tolerance,
        relative_l2_tolerance=writer_relative_l2_tolerance,
    )
    return {
        "schema_version": V6_PRIOR_CHECKPOINT_COMPARISON_SCHEMA,
        "left": {
            "checkpoint": left_inspection.summary["checkpoint"],
            "next_macro": left_inspection.summary["next_macro"],
        },
        "right": {
            "checkpoint": right_inspection.summary["checkpoint"],
            "next_macro": right_inspection.summary["next_macro"],
        },
        "next_macro": left_inspection.summary["next_macro"],
        "cursor": {"semantic_equal": True},
        "checkpoint_contract": {"semantic_equal": True},
        "rng": {
            "semantic_equal": True,
            "rank_count": V6_PRIOR_WORLD_SIZE,
        },
        "trainer": {
            "scheduler_semantic_equal": True,
            "amp_semantic_equal": True,
            "optimizer": optimizer_summary,
        },
        "writer": writer_summary,
        "content_hash_policy": "disabled_by_owner",
    }
