"""Immutable initialization bundles and exact-resume PI05 task-local checkpoints."""

from __future__ import annotations

import os
import random
import re
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ember.lora import (
    LoRAContract,
    canonical_contract_sha256,
    copy_task_lora_state_,
    lora_state_sha256,
    task_lora_state_dict,
    validate_lora_state,
)
from ember.pi05_source_checkpoint import canonical_hash, read_json, sha256_file, write_json_atomic
from ember.reward.ledger import InteractionCursors
from ember.reward.protocol import RewardProtocolError


INITIALIZATION_SCHEMA = "ember_pi05_task_local_initialization_v1"
CHECKPOINT_SCHEMA = "ember_pi05_task_local_checkpoint_v1"
TRAINER_SCHEMA = "ember_pi05_task_local_trainer_state_v1"
RNG_SCHEMA = "ember_pi05_task_local_rng_state_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def _validate_initialization_evidence(evidence: Mapping[str, Any]) -> None:
    arm = evidence.get("arm")
    teacher_used = evidence.get("teacher_video_used")
    demo = evidence.get("teacher_demo_index")
    if (
        arm not in {"identity", "as_writer", "rl_writer"}
        or int(evidence.get("global_task_id", -1)) < 0
        or int(evidence.get("adaptation_seed", -1)) < 0
        or _SHA256.fullmatch(str(evidence.get("source_checkpoint_manifest_sha256", "")))
        is None
        or (arm == "identity" and (teacher_used is not False or demo is not None))
        or (
            arm != "identity"
            and (
                teacher_used is not True
                or not 0 <= int(demo) < 50
                or _SHA256.fullmatch(str(evidence.get("writer_state_sha256", "")))
                is None
            )
        )
        or evidence.get("stacked_source_sft") is not False
    ):
        raise RewardProtocolError("task-local initialization evidence changed")


def validate_initialization_bundle(
    root: Path,
    *,
    contract: LoRAContract,
    unit_contract_sha256: str,
) -> dict[str, Any]:
    manifest_path = root / "initialization_manifest.json"
    manifest = read_json(manifest_path)
    payload = dict(manifest)
    digest = payload.pop("canonical_payload_sha256", None)
    lora_path = root / "initialization.safetensors"
    _validate_initialization_evidence(manifest.get("evidence", {}))
    if (
        manifest.get("schema_version") != INITIALIZATION_SCHEMA
        or canonical_hash(payload) != digest
        or manifest.get("unit_contract_sha256") != unit_contract_sha256
        or manifest.get("lora_contract_sha256") != canonical_contract_sha256(contract)
        or not lora_path.is_file()
        or lora_path.stat().st_size != int(manifest.get("lora_file", {}).get("bytes", -1))
        or sha256_file(lora_path) != manifest.get("lora_file", {}).get("sha256")
    ):
        raise RewardProtocolError("task-local initialization bundle changed")
    state = load_file(str(lora_path), device="cpu")
    validate_lora_state(state, contract)
    if lora_state_sha256(state) != manifest.get("lora_state_sha256"):
        raise RewardProtocolError("task-local initialization LoRA changed")
    return manifest


def write_initialization_bundle(
    *,
    unit_dir: Path,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    unit_contract_sha256: str,
    evidence: Mapping[str, Any],
) -> Path:
    validate_lora_state(state, contract)
    _validate_initialization_evidence(evidence)
    root = unit_dir / "initialization"
    if root.exists():
        observed = validate_initialization_bundle(
            root, contract=contract, unit_contract_sha256=unit_contract_sha256
        )
        if observed.get("evidence") != dict(evidence):
            raise RewardProtocolError("task-local initialization replay changed")
        return root
    temporary = unit_dir / f".initialization.{uuid.uuid4().hex}.partial"
    temporary.mkdir(parents=True)
    lora_path = temporary / "initialization.safetensors"
    save_file(
        {
            name: value.detach().to(device="cpu").contiguous()
            for name, value in state.items()
        },
        str(lora_path),
    )
    manifest = {
        "schema_version": INITIALIZATION_SCHEMA,
        "unit_contract_sha256": unit_contract_sha256,
        "lora_contract_sha256": canonical_contract_sha256(contract),
        "lora_state_sha256": lora_state_sha256(state),
        "lora_file": {
            "bytes": lora_path.stat().st_size,
            "sha256": sha256_file(lora_path),
        },
        "evidence": dict(evidence),
    }
    manifest["canonical_payload_sha256"] = canonical_hash(manifest)
    write_json_atomic(temporary / "initialization_manifest.json", manifest)
    os.replace(temporary, root)
    return root


def validate_task_local_checkpoint_files(
    checkpoint: Path,
    *,
    unit_contract_sha256: str | None = None,
    initialization_manifest_sha256: str | None = None,
    ledger_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    payload = dict(manifest)
    digest = payload.pop("canonical_payload_sha256", None)
    files = manifest.get("files", {})
    expected = {"lora.safetensors", "trainer_state.pt", "rng_state.pt"}
    if (
        manifest.get("schema_version") != CHECKPOINT_SCHEMA
        or canonical_hash(payload) != digest
        or not isinstance(files, dict)
        or set(files) != expected
        or (
            unit_contract_sha256 is not None
            and manifest.get("unit_contract_sha256") != unit_contract_sha256
        )
        or (
            initialization_manifest_sha256 is not None
            and manifest.get("initialization_manifest_sha256")
            != initialization_manifest_sha256
        )
        or (
            ledger_summary is not None
            and manifest.get("ledger_summary") != dict(ledger_summary)
        )
    ):
        raise RewardProtocolError("task-local checkpoint manifest changed")
    for relative, record in files.items():
        path = checkpoint / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(path) != record.get("sha256")
        ):
            raise RewardProtocolError(f"task-local checkpoint file changed: {relative}")
    return manifest


def save_task_local_checkpoint(
    *,
    unit_dir: Path,
    next_update: int,
    policy: torch.nn.Module,
    contract: LoRAContract,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    unit_contract_sha256: str,
    initialization_manifest_sha256: str,
    cursors: InteractionCursors,
    successes: int,
    reward_sum: float,
    wall_nanoseconds: int,
    rollouts_per_update: int,
    segment_successes: int,
    segment_rollouts: int,
    ledger_summary: Mapping[str, Any],
    device: torch.device,
) -> Path:
    if (
        next_update <= 0
        or cursors.rollout != next_update * rollouts_per_update
        or cursors.environment_actions < cursors.rollout
        or not 0 <= cursors.optimizer_updates <= next_update
        or not 0 <= successes <= cursors.rollout
        or reward_sum < 0
        or wall_nanoseconds < 0
        or not 0 <= segment_successes <= segment_rollouts
        or not 0 < segment_rollouts <= cursors.rollout
        or int(ledger_summary.get("rollout_cursor", -1)) != cursors.rollout
        or int(ledger_summary.get("environment_action_cursor", -1))
        != cursors.environment_actions
    ):
        raise RewardProtocolError("task-local checkpoint cursors are inconsistent")
    final = unit_dir / "checkpoints" / f"update_{next_update:08d}"
    if final.exists():
        raise RewardProtocolError(f"task-local checkpoint exists: {final}")
    temporary = unit_dir / "checkpoints" / f".update_{next_update:08d}.{uuid.uuid4().hex}.partial"
    temporary.mkdir(parents=True)
    state = task_lora_state_dict(policy, clone=True)
    save_file(
        {name: value.to(device="cpu").contiguous() for name, value in state.items()},
        str(temporary / "lora.safetensors"),
    )
    torch.save(
        {
            "schema_version": TRAINER_SCHEMA,
            "next_update": next_update,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "amp_scaler": {"enabled": False, "state": {}},
            "unit_contract_sha256": unit_contract_sha256,
            "initialization_manifest_sha256": initialization_manifest_sha256,
            "rollouts_per_update": rollouts_per_update,
            "cursors": cursors.to_dict(),
            "successes": successes,
            "reward_sum": reward_sum,
            "wall_nanoseconds": wall_nanoseconds,
        },
        temporary / "trainer_state.pt",
    )
    saved_rng = _rng_state(device)
    torch.save(
        {
            "schema_version": RNG_SCHEMA,
            "next_update": next_update,
            "unit_contract_sha256": unit_contract_sha256,
            "rng": saved_rng,
        },
        temporary / "rng_state.pt",
    )
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(value for value in temporary.iterdir() if value.is_file())
    }
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA,
        "unit_contract_sha256": unit_contract_sha256,
        "initialization_manifest_sha256": initialization_manifest_sha256,
        "next_update": next_update,
        **cursors.to_dict(),
        "segment_successes": segment_successes,
        "segment_rollouts": segment_rollouts,
        "ledger_summary": dict(ledger_summary),
        "files": files,
    }
    manifest["canonical_payload_sha256"] = canonical_hash(manifest)
    write_json_atomic(temporary / "checkpoint_manifest.json", manifest)
    os.replace(temporary, final)
    write_json_atomic(
        unit_dir / "latest_checkpoint.json",
        {"path": str(final), "next_update": next_update},
    )
    restore_rng(saved_rng, device)
    return final


def load_task_local_checkpoint(
    *,
    checkpoint: Path,
    initialization_root: Path,
    policy: torch.nn.Module,
    contract: LoRAContract,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    unit_contract_sha256: str,
    rollouts_per_update: int,
    ledger_summary: Mapping[str, Any],
    device: torch.device,
) -> tuple[int, InteractionCursors, dict[str, Any], dict[str, Any]]:
    initialization = validate_initialization_bundle(
        initialization_root,
        contract=contract,
        unit_contract_sha256=unit_contract_sha256,
    )
    init_sha = sha256_file(initialization_root / "initialization_manifest.json")
    manifest = validate_task_local_checkpoint_files(
        checkpoint,
        unit_contract_sha256=unit_contract_sha256,
        initialization_manifest_sha256=init_sha,
        ledger_summary=ledger_summary,
    )
    trainer = torch.load(
        checkpoint / "trainer_state.pt", map_location=device, weights_only=False
    )
    rng_state = torch.load(
        checkpoint / "rng_state.pt", map_location="cpu", weights_only=False
    )
    next_update = int(manifest.get("next_update", -1))
    if (
        trainer.get("schema_version") != TRAINER_SCHEMA
        or rng_state.get("schema_version") != RNG_SCHEMA
        or trainer.get("unit_contract_sha256") != unit_contract_sha256
        or rng_state.get("unit_contract_sha256") != unit_contract_sha256
        or int(trainer.get("next_update", -2)) != next_update
        or int(rng_state.get("next_update", -2)) != next_update
        or int(trainer.get("rollouts_per_update", -1)) != rollouts_per_update
        or checkpoint.name != f"update_{next_update:08d}"
    ):
        raise RewardProtocolError("task-local resume authority changed")
    values = trainer["cursors"]
    cursors = InteractionCursors(
        rollout=int(values["rollout_cursor"]),
        environment_actions=int(values["environment_action_cursor"]),
        optimizer_updates=int(values["optimizer_update_cursor"]),
    )
    if cursors.rollout != next_update * rollouts_per_update:
        raise RewardProtocolError("task-local resume cursor changed")
    state = load_file(str(checkpoint / "lora.safetensors"), device=str(device))
    validate_lora_state(state, contract)
    copy_task_lora_state_(policy, state, contract)
    optimizer.load_state_dict(trainer["optimizer"])
    scheduler.load_state_dict(trainer["scheduler"])
    counters = {
        "successes": int(trainer["successes"]),
        "reward_sum": float(trainer["reward_sum"]),
        "wall_nanoseconds": int(trainer["wall_nanoseconds"]),
    }
    return next_update, cursors, rng_state["rng"], counters
