"""Atomic, hash-bound trainable-state artifacts for Gate 0 oracles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


CANDIDATE_MANIFEST = "candidate_manifest.json"
TRAINABLE_STATE = "trainable_state.safetensors"
RECOVERY_MANIFEST = "recovery_manifest.json"
OPTIMIZER_STATE = "optimizer.pt"
RNG_STATE = "rng_state.safetensors"
SELECTED_MANIFEST = "selected_manifest.json"


class GateZeroOracleArtifactError(RuntimeError):
    """Raised when an oracle candidate is incomplete, changed, or overwritten."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def normalized_trainable_state(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not isinstance(state, dict) or not state:
        raise GateZeroOracleArtifactError("trainable state is empty")
    normalized: dict[str, torch.Tensor] = {}
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not name or not torch.is_tensor(value):
            raise GateZeroOracleArtifactError("trainable state has an invalid entry")
        tensor = value.detach().to(device="cpu").contiguous()
        if not torch.isfinite(tensor).all():
            raise GateZeroOracleArtifactError(f"trainable state is non-finite: {name}")
        normalized[name] = tensor
    return normalized


def save_candidate_artifact(
    output_root: Path,
    *,
    variant: str,
    task_id: int,
    step: int,
    trainable_state: dict[str, torch.Tensor],
    metrics: dict[str, Any],
    authorities: dict[str, Any],
) -> Path:
    """Atomically publish one immutable query-evaluated candidate."""

    if variant not in {"lora", "partial_upper_bound"}:
        raise GateZeroOracleArtifactError("candidate variant is invalid")
    if task_id < 0 or step < 0 or metrics.get("step") != step:
        raise GateZeroOracleArtifactError("candidate task/step metadata is invalid")
    root = output_root / "candidates"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{step:06d}"
    if destination.exists() or destination.is_symlink():
        raise GateZeroOracleArtifactError("refusing to overwrite candidate artifact")
    staging = root / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    state = normalized_trainable_state(trainable_state)
    try:
        staging.mkdir(parents=False, exist_ok=False)
        state_path = staging / TRAINABLE_STATE
        save_file(state, state_path)
        manifest = {
            "schema_version": 1,
            "status": "query_evaluated_candidate",
            "variant": variant,
            "task_id": task_id,
            "step": step,
            "metrics": metrics,
            "authorities": authorities,
            "trainable_parameters": sum(value.numel() for value in state.values()),
            "trainable_tensors": len(state),
            "files": {
                TRAINABLE_STATE: {
                    "bytes": state_path.stat().st_size,
                    "sha256": sha256_file(state_path),
                }
            },
        }
        _atomic_json(staging / CANDIDATE_MANIFEST, manifest)
        _fsync_directory(staging)
        os.replace(staging, destination)
        _fsync_directory(root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


def validate_candidate_artifact(
    candidate_dir: Path, *, expected: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate exact candidate files and optional authority fields."""

    if not candidate_dir.is_dir():
        raise GateZeroOracleArtifactError("candidate artifact is missing")
    manifest_path = candidate_dir / CANDIDATE_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateZeroOracleArtifactError("candidate manifest is invalid") from error
    if manifest.get("schema_version") != 1 or manifest.get("status") != "query_evaluated_candidate":
        raise GateZeroOracleArtifactError("candidate manifest schema/status changed")
    actual = {path.name for path in candidate_dir.iterdir() if path.is_file()}
    if actual != {CANDIDATE_MANIFEST, TRAINABLE_STATE}:
        raise GateZeroOracleArtifactError("candidate file set changed")
    record = manifest.get("files", {}).get(TRAINABLE_STATE, {})
    state_path = candidate_dir / TRAINABLE_STATE
    if state_path.stat().st_size != record.get("bytes"):
        raise GateZeroOracleArtifactError("candidate state hash/bytes changed")
    if sha256_file(state_path) != record.get("sha256"):
        raise GateZeroOracleArtifactError("candidate state hash changed")
    try:
        state = load_file(state_path)
    except Exception as error:
        raise GateZeroOracleArtifactError("candidate state cannot be loaded") from error
    if (
        len(state) != manifest.get("trainable_tensors")
        or sum(value.numel() for value in state.values()) != manifest.get("trainable_parameters")
        or any(not torch.isfinite(value).all() for value in state.values())
    ):
        raise GateZeroOracleArtifactError("candidate trainable-state summary changed")
    for key, value in (expected or {}).items():
        if manifest.get(key) != value:
            raise GateZeroOracleArtifactError(f"candidate authority changed: {key}")
    return manifest


def load_candidate_state(candidate_dir: Path) -> dict[str, torch.Tensor]:
    validate_candidate_artifact(candidate_dir)
    return load_file(candidate_dir / TRAINABLE_STATE)


def restore_trainable_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if set(trainable) != set(state):
        raise GateZeroOracleArtifactError("trainable parameter identity changed")
    with torch.no_grad():
        for name, value in trainable.items():
            authority = state[name]
            if authority.shape != value.shape or authority.dtype != value.dtype:
                raise GateZeroOracleArtifactError(f"trainable parameter metadata changed: {name}")
            value.copy_(authority.to(device=value.device), non_blocking=False)


def _update_last(root: Path, destination: Path) -> None:
    last = root / "last"
    if last.exists() and not last.is_symlink():
        raise GateZeroOracleArtifactError("recovery last pointer is not a symlink")
    temporary = root / f".last.tmp-{uuid.uuid4().hex}"
    try:
        temporary.symlink_to(destination.name)
        os.replace(temporary, last)
        _fsync_directory(root)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def save_recovery_artifact(
    output_root: Path,
    *,
    variant: str,
    task_id: int,
    step: int,
    trainable_state: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    authorities: dict[str, Any],
) -> Path:
    """Publish a resumable candidate-bound state and atomically advance ``last``."""

    if variant not in {"lora", "partial_upper_bound"} or task_id < 0 or step < 0:
        raise GateZeroOracleArtifactError("recovery variant/task/step is invalid")
    root = output_root / "recovery"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{step:06d}"
    if destination.exists() or destination.is_symlink():
        raise GateZeroOracleArtifactError("refusing to overwrite recovery artifact")
    staging = root / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    state = normalized_trainable_state(trainable_state)
    try:
        staging.mkdir(parents=False, exist_ok=False)
        state_path = staging / TRAINABLE_STATE
        optimizer_path = staging / OPTIMIZER_STATE
        rng_path = staging / RNG_STATE
        save_file(state, state_path)
        torch.save(optimizer.state_dict(), optimizer_path)
        from lerobot.utils.random_utils import serialize_rng_state

        save_file(serialize_rng_state(), rng_path)
        files = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (state_path, optimizer_path, rng_path)
        }
        manifest = {
            "schema_version": 1,
            "status": "resumable_oracle_candidate",
            "variant": variant,
            "task_id": task_id,
            "step": step,
            "authorities": authorities,
            "trainable_parameters": sum(value.numel() for value in state.values()),
            "trainable_tensors": len(state),
            "files": files,
        }
        _atomic_json(staging / RECOVERY_MANIFEST, manifest)
        _fsync_directory(staging)
        os.replace(staging, destination)
        _fsync_directory(root)
        _update_last(root, destination)
        for prior in sorted(root.iterdir()):
            if (
                prior.is_dir()
                and not prior.is_symlink()
                and prior != destination
                and not prior.name.startswith(".")
            ):
                shutil.rmtree(prior)
        _fsync_directory(root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


def validate_recovery_artifact(
    recovery_dir: Path, *, expected: dict[str, Any] | None = None
) -> dict[str, Any]:
    if not recovery_dir.is_dir():
        raise GateZeroOracleArtifactError("recovery artifact is missing")
    try:
        manifest = json.loads((recovery_dir / RECOVERY_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateZeroOracleArtifactError("recovery manifest is invalid") from error
    if manifest.get("schema_version") != 1 or manifest.get("status") != "resumable_oracle_candidate":
        raise GateZeroOracleArtifactError("recovery manifest schema/status changed")
    expected_files = {RECOVERY_MANIFEST, TRAINABLE_STATE, OPTIMIZER_STATE, RNG_STATE}
    actual_files = {path.name for path in recovery_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise GateZeroOracleArtifactError("recovery file set changed")
    for name, record in manifest.get("files", {}).items():
        path = recovery_dir / name
        if name not in expected_files - {RECOVERY_MANIFEST} or not path.is_file():
            raise GateZeroOracleArtifactError("recovery file authority changed")
        if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
            raise GateZeroOracleArtifactError(f"recovery hash/bytes changed: {name}")
    if set(manifest.get("files", {})) != expected_files - {RECOVERY_MANIFEST}:
        raise GateZeroOracleArtifactError("recovery file manifest is incomplete")
    for key, value in (expected or {}).items():
        if manifest.get(key) != value:
            raise GateZeroOracleArtifactError(f"recovery authority changed: {key}")
    return manifest


def load_recovery_artifact(
    recovery_dir: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected: dict[str, Any] | None = None,
) -> int:
    """Restore trainable state, optimizer, and RNG after loader construction."""

    manifest = validate_recovery_artifact(recovery_dir, expected=expected)
    state = load_file(recovery_dir / TRAINABLE_STATE)
    restore_trainable_state(model, state)
    try:
        optimizer_state = torch.load(
            recovery_dir / OPTIMIZER_STATE, map_location="cpu", weights_only=True
        )
        optimizer.load_state_dict(optimizer_state)
        from lerobot.utils.random_utils import deserialize_rng_state

        deserialize_rng_state(load_file(recovery_dir / RNG_STATE))
    except Exception as error:
        raise GateZeroOracleArtifactError("recovery optimizer/RNG state cannot be restored") from error
    return int(manifest["step"])


def publish_selected_artifact(output_root: Path, candidate_dir: Path) -> Path:
    """Copy exactly one validated candidate into the immutable selected role."""

    candidate = validate_candidate_artifact(candidate_dir)
    destination = output_root / "selected"
    if destination.exists() or destination.is_symlink():
        raise GateZeroOracleArtifactError("refusing to overwrite selected artifact")
    staging = output_root / f".selected.tmp-{uuid.uuid4().hex}"
    try:
        staging.mkdir(parents=False, exist_ok=False)
        source = candidate_dir / TRAINABLE_STATE
        target = staging / TRAINABLE_STATE
        shutil.copyfile(source, target)
        manifest = {
            "schema_version": 1,
            "status": "task_oracle_selection_frozen_pending_global_report_grant",
            "variant": candidate["variant"],
            "task_id": candidate["task_id"],
            "selected_step": candidate["step"],
            "selected_metrics": candidate["metrics"],
            "candidate_manifest_sha256": sha256_file(candidate_dir / CANDIDATE_MANIFEST),
            "trainable_state_sha256": sha256_file(target),
            "trainable_state_bytes": target.stat().st_size,
            "trainable_parameters": candidate["trainable_parameters"],
            "trainable_tensors": candidate["trainable_tensors"],
            "authorities": candidate["authorities"],
        }
        _atomic_json(staging / SELECTED_MANIFEST, manifest)
        _fsync_directory(staging)
        os.replace(staging, destination)
        _fsync_directory(output_root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


def validate_selected_artifact(
    selected_dir: Path, *, expected: dict[str, Any] | None = None
) -> dict[str, Any]:
    if not selected_dir.is_dir():
        raise GateZeroOracleArtifactError("selected artifact is missing")
    try:
        manifest = json.loads((selected_dir / SELECTED_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateZeroOracleArtifactError("selected manifest is invalid") from error
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status")
        != "task_oracle_selection_frozen_pending_global_report_grant"
    ):
        raise GateZeroOracleArtifactError("selected manifest schema/status changed")
    actual = {path.name for path in selected_dir.iterdir() if path.is_file()}
    if actual != {SELECTED_MANIFEST, TRAINABLE_STATE}:
        raise GateZeroOracleArtifactError("selected file set changed")
    state_path = selected_dir / TRAINABLE_STATE
    if (
        state_path.stat().st_size != manifest.get("trainable_state_bytes")
        or sha256_file(state_path) != manifest.get("trainable_state_sha256")
    ):
        raise GateZeroOracleArtifactError("selected state hash/bytes changed")
    state = load_file(state_path)
    if (
        len(state) != manifest.get("trainable_tensors")
        or sum(value.numel() for value in state.values())
        != manifest.get("trainable_parameters")
    ):
        raise GateZeroOracleArtifactError("selected trainable-state summary changed")
    for key, value in (expected or {}).items():
        if manifest.get(key) != value:
            raise GateZeroOracleArtifactError(f"selected authority changed: {key}")
    return manifest


def atomic_json(path: Path, value: Any) -> None:
    """Expose the module's atomic JSON publisher to the sole fit orchestrator."""

    _atomic_json(path, value)


def validate_fit_output(output_dir: Path, *, result_name: str, resume: bool) -> None:
    if not output_dir.is_absolute():
        raise GateZeroOracleArtifactError("output directory must be absolute")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / result_name).exists():
        raise GateZeroOracleArtifactError("refusing to overwrite completed oracle fit")
    entries = [
        path
        for path in output_dir.iterdir()
        if not (
            path.is_file()
            and path.name.startswith("gpu_telemetry_")
            and path.suffix == ".csv"
        )
    ]
    if resume:
        unexpected = [
            path.name for path in entries if path.name not in {"candidates", "recovery"}
        ]
        if unexpected or not (output_dir / "recovery" / "last").is_symlink():
            raise GateZeroOracleArtifactError(
                "resume output is incomplete or contains unknown state"
            )
    elif entries:
        raise GateZeroOracleArtifactError(
            f"refusing non-fresh output: {[path.name for path in entries]}"
        )


def candidate_evidence(candidate_dir: Path) -> dict[str, Any]:
    manifest = validate_candidate_artifact(candidate_dir)
    return {
        **manifest["metrics"],
        "candidate_manifest_sha256": sha256_file(candidate_dir / CANDIDATE_MANIFEST),
        "trainable_state_sha256": manifest["files"][TRAINABLE_STATE]["sha256"],
        "trainable_state_bytes": manifest["files"][TRAINABLE_STATE]["bytes"],
    }


def cleanup_completed_fit_state(output_dir: Path, *, variant: str) -> None:
    recovery = output_dir / "recovery"
    if recovery.exists():
        shutil.rmtree(recovery)
    if variant == "partial_upper_bound" and (candidates := output_dir / "candidates").exists():
        shutil.rmtree(candidates)


def write_output_checksums(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if (
            path.is_file()
            and path.name != "checksums.sha256"
            and not (path.name.startswith("gpu_telemetry_") and path.suffix == ".csv")
        ):
            rows.append(f"{sha256_file(path)}  {path.relative_to(output_dir)}")
    (output_dir / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")
