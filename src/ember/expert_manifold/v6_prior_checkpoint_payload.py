"""Tensor payload codec for frozen-v6 Program reconciliation checkpoints."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ember.expert_manifold.contract import ExpertManifoldError
from ember.writer.condition_update import (
    ProgramReconciliationState,
    ProgramResidualMemory,
)


PROGRAM_MEMORY_FILE = "program_memory.safetensors"
PROGRAM_MEMORY_KEY = "program_memory.value"
RECONCILIATION_FILE = "reconciliation.safetensors"
RECONCILIATION_KEY = "reconciliation.precision"


def _error(component: str) -> ExpertManifoldError:
    return ExpertManifoldError(
        f"v6 Program-residual checkpoint validation failed: {component}"
    )


def reconciliation_shape(
    memory_shape: tuple[int, int, int],
) -> tuple[int, int]:
    return memory_shape[0], memory_shape[0]


def validate_live_memory(
    memory: ProgramResidualMemory | torch.Tensor,
    expected_shape: tuple[int, int, int],
    *,
    require_finite: bool,
) -> torch.Tensor:
    if isinstance(memory, ProgramResidualMemory):
        value = memory.value
    elif isinstance(memory, torch.Tensor):
        value = memory
    else:
        raise _error("Program memory owner")
    if value.dtype != torch.float32 or tuple(value.shape) != expected_shape:
        raise _error("Program memory tensor schema")
    if require_finite and not bool(torch.isfinite(value).all()):
        raise _error("non-finite Program memory")
    return value


def validate_live_reconciliation(
    reconciliation: ProgramReconciliationState,
    expected_shape: tuple[int, int],
    *,
    expected_assimilated_rows: int,
    require_finite: bool,
    require_positive_definite: bool = False,
) -> torch.Tensor:
    if not isinstance(reconciliation, ProgramReconciliationState):
        raise _error("reconciliation owner")
    value = reconciliation.precision
    if (
        value.dtype != torch.float64
        or tuple(value.shape) != expected_shape
        or reconciliation.assimilated_rows != expected_assimilated_rows
    ):
        raise _error("reconciliation tensor schema")
    if require_finite and not bool(torch.isfinite(value).all()):
        raise _error("non-finite reconciliation precision")
    if require_positive_definite:
        try:
            torch.linalg.cholesky(value)
        except RuntimeError as error:
            raise _error(
                "reconciliation precision positive definiteness"
            ) from error
    return value


def write_state_payload(
    directory: Path,
    memory: torch.Tensor,
    precision: torch.Tensor,
) -> None:
    save_file(
        {PROGRAM_MEMORY_KEY: memory.detach().cpu().contiguous()},
        str(directory / PROGRAM_MEMORY_FILE),
    )
    save_file(
        {RECONCILIATION_KEY: precision.detach().cpu().contiguous()},
        str(directory / RECONCILIATION_FILE),
    )


def load_program_memory(
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


def inspect_program_memory_metadata(
    path: Path,
    expected_shape: tuple[int, int, int],
) -> dict[str, object]:
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


def load_reconciliation_precision(
    path: Path,
    expected_shape: tuple[int, int],
) -> torch.Tensor:
    try:
        state = load_file(str(path), device="cpu")
    except Exception as error:
        raise _error("reconciliation file") from error
    if set(state) != {RECONCILIATION_KEY}:
        raise _error("reconciliation key")
    value = state[RECONCILIATION_KEY]
    if value.dtype != torch.float64 or tuple(value.shape) != expected_shape:
        raise _error("reconciliation tensor schema")
    if not bool(torch.isfinite(value).all()):
        raise _error("non-finite reconciliation precision")
    try:
        torch.linalg.cholesky(value)
    except RuntimeError as error:
        raise _error("reconciliation precision positive definiteness") from error
    return value


def inspect_reconciliation_metadata(
    path: Path,
    expected_shape: tuple[int, int],
) -> dict[str, object]:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {RECONCILIATION_KEY}:
                raise _error("reconciliation key")
            value = handle.get_slice(RECONCILIATION_KEY)
            if value.get_dtype() != "F64" or tuple(value.get_shape()) != expected_shape:
                raise _error("reconciliation tensor schema")
    except ExpertManifoldError:
        raise
    except Exception as error:
        raise _error("reconciliation file") from error
    return {
        "dtype": "torch.float64",
        "shape": list(expected_shape),
        "value_count": math.prod(expected_shape),
    }
