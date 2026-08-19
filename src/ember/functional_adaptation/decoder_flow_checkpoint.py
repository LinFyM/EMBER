"""Atomic exact-resume state for the canonical fixed-decoder flow trainer."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic


CHECKPOINT_SCHEMA = "ember_functional_decoder_flow_checkpoint_v1"
TRAINER_SCHEMA = "ember_functional_decoder_flow_trainer_v1"
RUN_SCHEMA = "ember_functional_decoder_flow_run_v1"
PHASES = ("decoder", "held_code")


@dataclass(frozen=True)
class DecoderFlowCursor:
    phase: str
    step: int
    metrics_rows: int
    visits: tuple[int, ...]


def decoder_flow_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state(state["torch_cuda"].cpu())


def _checkpoint_name(phase: str, step: int) -> str:
    if phase not in PHASES or step <= 0:
        raise ValueError("invalid functional-decoder checkpoint cursor")
    return f"{phase}_step_{step:08d}"


def inspect_decoder_flow_checkpoint(checkpoint: Path) -> DecoderFlowCursor:
    checkpoint = checkpoint.resolve()
    manifest = read_json(checkpoint / "manifest.json")
    phase = str(manifest.get("phase"))
    step = int(manifest.get("step", -1))
    metrics_rows = int(manifest.get("metrics_rows", -1))
    if (
        manifest.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.name != _checkpoint_name(phase, step)
        or metrics_rows < step
        or set(manifest.get("files", {}))
        != {"model.safetensors", "trainer.pt"}
        or manifest.get("content_hash_policy") != "disabled_by_owner"
    ):
        raise ValueError("functional-decoder checkpoint manifest changed")
    for name, expected_bytes in manifest["files"].items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(expected_bytes):
            raise ValueError("functional-decoder checkpoint file changed")
    trainer = torch.load(
        checkpoint / "trainer.pt", map_location="cpu", weights_only=False
    )
    visits = tuple(int(value) for value in trainer.get("visits", ()))
    if (
        trainer.get("schema_version") != TRAINER_SCHEMA
        or trainer.get("phase") != phase
        or int(trainer.get("step", -1)) != step
        or int(trainer.get("metrics_rows", -1)) != metrics_rows
        or not visits
        or sum(visits) != step
    ):
        raise ValueError("functional-decoder trainer cursor changed")
    return DecoderFlowCursor(phase, step, metrics_rows, visits)


def save_decoder_flow_checkpoint(
    *,
    output_dir: Path,
    phase: str,
    step: int,
    metrics_rows: int,
    visits: Sequence[int],
    system: torch.nn.Module,
    held_codes: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> Path:
    cursor_visits = tuple(int(value) for value in visits)
    if (
        phase not in PHASES
        or step <= 0
        or metrics_rows < step
        or not cursor_visits
        or sum(cursor_visits) != step
    ):
        raise ValueError("invalid functional-decoder checkpoint state")
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    name = _checkpoint_name(phase, step)
    final = checkpoints / name
    temporary = checkpoints / f".{name}.tmp-{os.getpid()}"
    if final.exists() or temporary.exists():
        raise ValueError("functional-decoder checkpoint already exists")
    temporary.mkdir()
    model_path = temporary / "model.safetensors"
    trainer_path = temporary / "trainer.pt"
    state = {
        f"system.{name}": value.detach().cpu().contiguous()
        for name, value in system.state_dict().items()
    }
    state["held_codes"] = held_codes.detach().cpu().contiguous()
    save_file(state, str(model_path))
    torch.save(
        {
            "schema_version": TRAINER_SCHEMA,
            "phase": phase,
            "step": step,
            "metrics_rows": metrics_rows,
            "visits": list(cursor_visits),
            "optimizer": optimizer.state_dict(),
            "rng": decoder_flow_rng_state(),
        },
        trainer_path,
    )
    write_json_atomic(
        temporary / "manifest.json",
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "phase": phase,
            "step": step,
            "metrics_rows": metrics_rows,
            "files": {
                "model.safetensors": model_path.stat().st_size,
                "trainer.pt": trainer_path.stat().st_size,
            },
            "content_hash_policy": "disabled_by_owner",
        },
    )
    os.replace(temporary, final)
    return final


def load_decoder_flow_checkpoint(
    *,
    checkpoint: Path,
    expected_phase: str,
    system: torch.nn.Module,
    held_codes: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
) -> DecoderFlowCursor:
    checkpoint = checkpoint.resolve()
    cursor = inspect_decoder_flow_checkpoint(checkpoint)
    if cursor.phase != expected_phase:
        raise ValueError("functional-decoder resume phase changed")
    state = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    system_state = {
        name.removeprefix("system."): value
        for name, value in state.items()
        if name.startswith("system.")
    }
    if "held_codes" not in state:
        raise ValueError("functional-decoder held codes are missing")
    system.load_state_dict(system_state, strict=True)
    held_codes.data.copy_(state["held_codes"].to(held_codes))
    trainer = torch.load(
        checkpoint / "trainer.pt", map_location="cpu", weights_only=False
    )
    optimizer.load_state_dict(trainer["optimizer"])
    _restore_rng(trainer["rng"])
    return cursor
