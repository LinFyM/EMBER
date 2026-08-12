"""Atomic memory-only checkpoints for the frozen-v6 Program residual."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ember.expert_manifold.contract import ExpertManifoldError
from ember.pi05_source_checkpoint import (
    DistributedContext,
    read_json,
    write_json_atomic,
)
from ember.writer.condition_update import ProgramResidualMemory


V6_PRIOR_CHECKPOINT_SCHEMA = (
    "ember_pi05_v6_paired_video_joint_functional_credit_checkpoint_v1"
)
V6_PRIOR_RNG_SCHEMA = "ember_pi05_v6_paired_video_joint_functional_credit_rank_rng_v1"
V6_PRIOR_CHECKPOINT_INSPECTION_SCHEMA = (
    "ember_pi05_v6_paired_video_joint_functional_credit_inspection_v1"
)
V6_PRIOR_WORLD_SIZE = 6
FORMAL_PROGRAM_MEMORY_SHAPE = (256, 320, 256)
PROGRAM_MEMORY_FILE = "program_memory.safetensors"
PROGRAM_MEMORY_KEY = "program_memory.value"
_CONTENT_HASH_POLICY = "disabled_by_owner"
_MANIFEST_KEYS = {
    "schema_version",
    "next_macro",
    "metrics_rows",
    "world_size",
    "program_memory_shape",
    "cursor_contract",
    "checkpoint_contract",
    "files",
    "content_hash_policy",
}
_RNG_KEYS = {
    "schema_version",
    "rank",
    "world_size",
    "device_type",
    "python",
    "numpy",
    "torch_cpu",
    "torch_cuda",
}


@dataclass(frozen=True)
class _CheckpointLayout:
    checkpoint: Path
    manifest: dict[str, Any]
    macro: int
    metrics_rows: int
    world_size: int
    memory_shape: tuple[int, int, int]


def _error(component: str) -> ExpertManifoldError:
    return ExpertManifoldError(
        f"v6 Program-residual checkpoint validation failed: {component}"
    )


def _raise_distributed(
    context: DistributedContext, phase: str, error: Exception | None
) -> None:
    """Make every rank observe a checkpoint I/O failure before continuing."""

    local = None if error is None else repr(error)
    failures: list[str | None] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(failures, local)
    else:
        failures[0] = local
    observed = [
        f"rank {rank}: {value}"
        for rank, value in enumerate(failures)
        if value is not None
    ]
    if observed:
        raise ExpertManifoldError(
            f"v6 Program-residual checkpoint {phase} failed; " + "; ".join(observed)
        )


def _shape(value: Sequence[int]) -> tuple[int, int, int]:
    try:
        dimensions = tuple(value)
    except TypeError as error:
        raise _error("Program memory shape") from error
    if len(dimensions) != 3 or any(
        type(dimension) is not int or dimension <= 0 for dimension in dimensions
    ):
        raise _error("Program memory shape")
    return dimensions


def _json_object(value: Mapping[str, Any], component: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise _error(component)
    try:
        normalized = json.loads(
            json.dumps(dict(value), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise _error(component) from error
    if not isinstance(normalized, dict):
        raise _error(component)
    return normalized


def _memory_tensor(
    memory: ProgramResidualMemory | torch.Tensor,
) -> torch.Tensor:
    if isinstance(memory, ProgramResidualMemory):
        return memory.value
    if isinstance(memory, torch.Tensor):
        return memory
    raise _error("Program memory owner")


def _validate_live_memory(
    memory: ProgramResidualMemory | torch.Tensor,
    expected_shape: tuple[int, int, int],
    *,
    require_finite: bool,
) -> torch.Tensor:
    value = _memory_tensor(memory)
    if value.dtype != torch.float32 or tuple(value.shape) != expected_shape:
        raise _error("Program memory tensor schema")
    if require_finite and not bool(torch.isfinite(value).all()):
        raise _error("non-finite Program memory")
    return value


def _rng_state(context: DistributedContext) -> dict[str, Any]:
    cuda_state = None
    if context.device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(context.device).cpu()
    return {
        "schema_version": V6_PRIOR_RNG_SCHEMA,
        "rank": context.rank,
        "world_size": context.world_size,
        "device_type": context.device.type,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_state,
    }


def _valid_rng_tensor(value: object) -> bool:
    return bool(
        isinstance(value, torch.Tensor)
        and value.device.type == "cpu"
        and value.dtype == torch.uint8
        and value.ndim == 1
        and value.numel() > 0
    )


def _validate_rng(
    value: object,
    *,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != _RNG_KEYS
        or value.get("schema_version") != V6_PRIOR_RNG_SCHEMA
        or type(value.get("rank")) is not int
        or value.get("rank") != rank
        or type(value.get("world_size")) is not int
        or value.get("world_size") != world_size
        or value.get("device_type") not in {"cpu", "cuda"}
        or not _valid_rng_tensor(value.get("torch_cpu"))
    ):
        raise _error(f"rank {rank} RNG")
    cuda = value.get("torch_cuda")
    if (value["device_type"] == "cuda") != _valid_rng_tensor(cuda):
        raise _error(f"rank {rank} RNG")
    if value["device_type"] == "cpu" and cuda is not None:
        raise _error(f"rank {rank} RNG")
    try:
        random.Random().setstate(value["python"])
        np.random.RandomState().set_state(value["numpy"])
        torch.Generator(device="cpu").set_state(value["torch_cpu"])
    except (TypeError, ValueError, RuntimeError) as error:
        raise _error(f"rank {rank} RNG") from error
    return value


def _load_rng(path: Path, *, rank: int, world_size: int) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise _error(f"rank {rank} RNG") from error
    return _validate_rng(value, rank=rank, world_size=world_size)


def _restore_rng(value: Mapping[str, Any], context: DistributedContext) -> None:
    validated = _validate_rng(
        dict(value), rank=context.rank, world_size=context.world_size
    )
    if validated["device_type"] != context.device.type:
        raise _error(f"rank {context.rank} RNG device")
    if context.device.type == "cuda":
        torch.cuda.set_rng_state(validated["torch_cuda"], context.device)
    random.setstate(validated["python"])
    np.random.set_state(validated["numpy"])
    torch.set_rng_state(validated["torch_cpu"])


def _load_program_memory(
    path: Path,
    expected_shape: tuple[int, int, int],
) -> torch.Tensor:
    try:
        state = load_file(str(path), device="cpu")
    except Exception as error:
        raise _error("Program memory file") from error
    if set(state) != {PROGRAM_MEMORY_KEY}:
        raise _error("Program memory key")
    value = state[PROGRAM_MEMORY_KEY]
    if value.dtype != torch.float32 or tuple(value.shape) != expected_shape:
        raise _error("Program memory tensor schema")
    if not bool(torch.isfinite(value).all()):
        raise _error("non-finite Program memory")
    return value


def _inspect_program_memory_metadata(
    path: Path,
    expected_shape: tuple[int, int, int],
) -> dict[str, Any]:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {PROGRAM_MEMORY_KEY}:
                raise _error("Program memory key")
            value = handle.get_slice(PROGRAM_MEMORY_KEY)
            if value.get_dtype() != "F32" or tuple(value.get_shape()) != expected_shape:
                raise _error("Program memory tensor schema")
    except ExpertManifoldError:
        raise
    except Exception as error:
        raise _error("Program memory file") from error
    return {
        "dtype": "torch.float32",
        "shape": list(expected_shape),
        "value_count": math.prod(expected_shape),
    }


def _read_manifest(checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    if checkpoint.is_symlink():
        raise _error("checkpoint directory")
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir() or checkpoint.parent.name != "checkpoints":
        raise _error("checkpoint directory")
    try:
        manifest = read_json(checkpoint / "manifest.json")
    except Exception as error:
        raise _error("manifest") from error
    return checkpoint, manifest


def _manifest_identity(
    checkpoint: Path,
    manifest: Mapping[str, Any],
    *,
    expected_memory_shape: tuple[int, int, int],
    expected_world_size: int,
) -> tuple[int, int]:
    macro = manifest.get("next_macro")
    metrics_rows = manifest.get("metrics_rows")
    recorded_shape = manifest.get("program_memory_shape")
    if (
        set(manifest) != _MANIFEST_KEYS
        or manifest.get("schema_version") != V6_PRIOR_CHECKPOINT_SCHEMA
        or type(macro) is not int
        or macro <= 0
        or checkpoint.name != f"macro_{macro:08d}"
        or type(metrics_rows) is not int
        or metrics_rows != macro
        or manifest.get("world_size") != expected_world_size
        or not isinstance(recorded_shape, list)
        or tuple(recorded_shape) != expected_memory_shape
        or manifest.get("content_hash_policy") != _CONTENT_HASH_POLICY
        or not isinstance(manifest.get("files"), dict)
    ):
        raise _error("manifest")
    return macro, metrics_rows


def _validate_payload_files(
    checkpoint: Path,
    files: Mapping[str, Any],
    world_size: int,
) -> None:
    expected_files = {
        PROGRAM_MEMORY_FILE,
        *(f"rng_rank_{rank:03d}.pt" for rank in range(world_size)),
    }
    physical_files = {path.name for path in checkpoint.iterdir()}
    if set(files) != expected_files or physical_files != expected_files | {
        "manifest.json"
    }:
        raise _error("manifest")
    for name in expected_files:
        expected_bytes = files.get(name)
        path = checkpoint / name
        if (
            type(expected_bytes) is not int
            or expected_bytes <= 0
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != expected_bytes
        ):
            raise _error(f"declared file {name}")


def _validate_contracts(
    manifest: Mapping[str, Any],
    *,
    macro: int,
    expected_cursor_contract: Mapping[str, Any] | None,
    expected_checkpoint_contract: Mapping[str, Any] | None,
) -> None:
    cursor = _json_object(manifest.get("cursor_contract"), "cursor contract")
    contract = _json_object(manifest.get("checkpoint_contract"), "checkpoint contract")
    if cursor.get("next_macro") != macro:
        raise _error("cursor contract")
    if expected_cursor_contract is not None and cursor != _json_object(
        expected_cursor_contract, "expected cursor contract"
    ):
        raise _error("cursor contract")
    if expected_checkpoint_contract is not None and contract != _json_object(
        expected_checkpoint_contract, "expected checkpoint contract"
    ):
        raise _error("checkpoint contract")


def _validate_layout(
    checkpoint: Path,
    *,
    expected_memory_shape: tuple[int, int, int],
    expected_world_size: int,
    expected_cursor_contract: Mapping[str, Any] | None,
    expected_checkpoint_contract: Mapping[str, Any] | None,
) -> _CheckpointLayout:
    checkpoint, manifest = _read_manifest(checkpoint)
    macro, metrics_rows = _manifest_identity(
        checkpoint,
        manifest,
        expected_memory_shape=expected_memory_shape,
        expected_world_size=expected_world_size,
    )
    _validate_payload_files(checkpoint, manifest["files"], expected_world_size)
    _validate_contracts(
        manifest,
        macro=macro,
        expected_cursor_contract=expected_cursor_contract,
        expected_checkpoint_contract=expected_checkpoint_contract,
    )
    return _CheckpointLayout(
        checkpoint=checkpoint,
        manifest=manifest,
        macro=macro,
        metrics_rows=metrics_rows,
        world_size=expected_world_size,
        memory_shape=expected_memory_shape,
    )


def save_v6_prior_checkpoint(
    *,
    output_dir: Path,
    macro: int,
    memory: ProgramResidualMemory | torch.Tensor,
    context: DistributedContext,
    metrics_rows: int,
    cursor_contract: Mapping[str, Any],
    checkpoint_contract: Mapping[str, Any],
    expected_memory_shape: Sequence[int] = FORMAL_PROGRAM_MEMORY_SHAPE,
) -> Path:
    """Atomically publish only Program memory and per-rank RNG at a macro boundary."""

    shape = _shape(expected_memory_shape)
    if (
        type(macro) is not int
        or macro <= 0
        or type(metrics_rows) is not int
        or metrics_rows != macro
        or context.world_size <= 0
        or not 0 <= context.rank < context.world_size
    ):
        raise _error("checkpoint cursor")
    cursor = _json_object(cursor_contract, "cursor contract")
    contract = _json_object(checkpoint_contract, "checkpoint contract")
    if cursor.get("next_macro") != macro:
        raise _error("cursor contract")
    value = _validate_live_memory(memory, shape, require_finite=False)
    checkpoints = output_dir / "checkpoints"
    final = checkpoints / f"macro_{macro:08d}"
    temporary = checkpoints / f".macro_{macro:08d}.tmp"
    error: Exception | None = None
    try:
        if context.is_main:
            _validate_live_memory(memory, shape, require_finite=True)
            checkpoints.mkdir(parents=True, exist_ok=True)
            if temporary.exists() and not final.exists():
                failure_packets = output_dir / "failure_packets"
                failure_packets.mkdir(parents=True, exist_ok=True)
                os.replace(
                    temporary,
                    failure_packets
                    / (f"incomplete_checkpoint_macro_{macro:08d}_" f"{time.time_ns()}"),
                )
            if final.exists() or temporary.exists():
                raise _error("checkpoint already exists")
            temporary.mkdir()
    except Exception as caught:
        error = caught
    _raise_distributed(context, "initialization", error)

    saved_rng: Mapping[str, Any] | None = None
    error = None
    rng_name = f"rng_rank_{context.rank:03d}.pt"
    try:
        saved_rng = _rng_state(context)
        torch.save(saved_rng, temporary / rng_name)
    except Exception as caught:
        error = caught
    _raise_distributed(context, "rank RNG write", error)

    error = None
    try:
        if context.is_main:
            save_file(
                {PROGRAM_MEMORY_KEY: value.detach().cpu().contiguous()},
                str(temporary / PROGRAM_MEMORY_FILE),
            )
            payload_names = {
                PROGRAM_MEMORY_FILE,
                *(f"rng_rank_{rank:03d}.pt" for rank in range(context.world_size)),
            }
            files = {name: (temporary / name).stat().st_size for name in payload_names}
            write_json_atomic(
                temporary / "manifest.json",
                {
                    "schema_version": V6_PRIOR_CHECKPOINT_SCHEMA,
                    "next_macro": macro,
                    "metrics_rows": metrics_rows,
                    "world_size": context.world_size,
                    "program_memory_shape": list(shape),
                    "cursor_contract": cursor,
                    "checkpoint_contract": contract,
                    "files": files,
                    "content_hash_policy": _CONTENT_HASH_POLICY,
                },
            )
            os.replace(temporary, final)
    except Exception as caught:
        error = caught
    _raise_distributed(context, "publication", error)

    error = None
    try:
        if saved_rng is None:
            raise _error("rank RNG capture")
        _restore_rng(saved_rng, context)
    except Exception as caught:
        error = caught
    _raise_distributed(context, "rank RNG restoration", error)
    return final


@torch.no_grad()
def load_v6_prior_checkpoint(
    *,
    checkpoint: Path,
    memory: ProgramResidualMemory | torch.Tensor,
    context: DistributedContext,
    expected_cursor_contract: Mapping[str, Any],
    expected_checkpoint_contract: Mapping[str, Any],
    expected_memory_shape: Sequence[int] = FORMAL_PROGRAM_MEMORY_SHAPE,
) -> tuple[int, int]:
    """Restore Program memory and this rank's RNG without accepting base state."""

    destination: torch.Tensor | None = None
    layout: _CheckpointLayout | None = None
    restored: torch.Tensor | None = None
    rng: Mapping[str, Any] | None = None
    error: Exception | None = None
    try:
        shape = _shape(expected_memory_shape)
        destination = _validate_live_memory(memory, shape, require_finite=False)
        layout = _validate_layout(
            checkpoint,
            expected_memory_shape=shape,
            expected_world_size=context.world_size,
            expected_cursor_contract=expected_cursor_contract,
            expected_checkpoint_contract=expected_checkpoint_contract,
        )
        restored = _load_program_memory(layout.checkpoint / PROGRAM_MEMORY_FILE, shape)
        rng = _load_rng(
            layout.checkpoint / f"rng_rank_{context.rank:03d}.pt",
            rank=context.rank,
            world_size=context.world_size,
        )
        if rng["device_type"] != context.device.type:
            raise _error(f"rank {context.rank} RNG device")
    except Exception as caught:
        error = caught
    _raise_distributed(context, "resume payload load", error)

    error = None
    try:
        if destination is None or layout is None or restored is None or rng is None:
            raise _error("resume payload agreement")
        destination.copy_(restored.to(device=destination.device))
        _restore_rng(rng, context)
    except Exception as caught:
        error = caught
    _raise_distributed(context, "resume state restoration", error)
    if layout is None:
        raise _error("resume payload agreement")
    return layout.macro, layout.metrics_rows


def inspect_v6_prior_checkpoint(
    checkpoint: Path,
    *,
    expected_memory_shape: Sequence[int] = FORMAL_PROGRAM_MEMORY_SHAPE,
    expected_world_size: int = V6_PRIOR_WORLD_SIZE,
    expected_cursor_contract: Mapping[str, Any] | None = None,
    expected_checkpoint_contract: Mapping[str, Any] | None = None,
    validate_payload_values: bool = True,
) -> dict[str, Any]:
    """Validate one checkpoint without mutating live memory or global RNG."""

    shape = _shape(expected_memory_shape)
    if type(expected_world_size) is not int or expected_world_size <= 0:
        raise _error("expected world size")
    layout = _validate_layout(
        checkpoint,
        expected_memory_shape=shape,
        expected_world_size=expected_world_size,
        expected_cursor_contract=expected_cursor_contract,
        expected_checkpoint_contract=expected_checkpoint_contract,
    )
    if validate_payload_values:
        memory = _load_program_memory(layout.checkpoint / PROGRAM_MEMORY_FILE, shape)
        memory_metadata = {
            "dtype": str(memory.dtype),
            "shape": list(memory.shape),
            "value_count": memory.numel(),
        }
        rng = tuple(
            _load_rng(
                layout.checkpoint / f"rng_rank_{rank:03d}.pt",
                rank=rank,
                world_size=layout.world_size,
            )
            for rank in range(layout.world_size)
        )
        finite: bool | None = True
        device_types = [value["device_type"] for value in rng]
        payload_validation = "full_values"
    else:
        memory_metadata = _inspect_program_memory_metadata(
            layout.checkpoint / PROGRAM_MEMORY_FILE,
            shape,
        )
        finite = None
        device_types = []
        payload_validation = "deployment_metadata_only"
    return {
        "schema_version": V6_PRIOR_CHECKPOINT_INSPECTION_SCHEMA,
        "checkpoint": str(layout.checkpoint),
        "checkpoint_schema": V6_PRIOR_CHECKPOINT_SCHEMA,
        "next_macro": layout.macro,
        "metrics_rows": layout.metrics_rows,
        "world_size": layout.world_size,
        "cursor_contract": layout.manifest["cursor_contract"],
        "checkpoint_contract": layout.manifest["checkpoint_contract"],
        "files": [
            {"name": name, "bytes": layout.manifest["files"][name]}
            for name in sorted(layout.manifest["files"])
        ],
        "program_memory": {
            "file": PROGRAM_MEMORY_FILE,
            "key": PROGRAM_MEMORY_KEY,
            "tensor_count": 1,
            **memory_metadata,
            "finite": finite,
        },
        "rng": {
            "schema_version": V6_PRIOR_RNG_SCHEMA,
            "rank_count": layout.world_size,
            "ranks": list(range(layout.world_size)),
            "device_types": device_types,
        },
        "payload_value_validation": payload_validation,
        "content_hash_policy": _CONTENT_HASH_POLICY,
    }
