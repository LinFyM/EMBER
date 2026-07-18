"""Hash-bound checkpoint and runtime-policy manifests for Gate 0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class GateZeroCheckpointError(RuntimeError):
    """Raised when a checkpoint or policy runtime view is incomplete or changed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hashed_tree(root: Path, *, excluded_names: set[str] | None = None) -> dict[str, dict[str, Any]]:
    excluded = excluded_names or set()
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in excluded:
            relative = path.relative_to(root).as_posix()
            result[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if not result:
        raise GateZeroCheckpointError("cannot hash an empty artifact tree")
    return result


def validate_hashed_tree(root: Path, files: dict[str, dict[str, Any]]) -> None:
    if not isinstance(files, dict) or not files:
        raise GateZeroCheckpointError("hashed file manifest is empty")
    actual_paths = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual_paths != set(files):
        raise GateZeroCheckpointError("hashed artifact file set changed")
    for relative, authority in files.items():
        path = root / relative
        if path.stat().st_size != authority.get("bytes"):
            raise GateZeroCheckpointError(f"artifact bytes changed: {relative}")
        if sha256_file(path) != authority.get("sha256"):
            raise GateZeroCheckpointError(f"artifact hash changed: {relative}")


def build_policy_runtime_manifest(
    policy_dir: Path,
    *,
    policy_role: str,
    training_step: int,
    base_revision: str,
    base_weight_sha256: str,
    normalization_sha256: str,
    contract_sha256: str,
) -> dict[str, Any]:
    required = {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    }
    files = hashed_tree(policy_dir, excluded_names={"policy_runtime_manifest.json"})
    if not required <= set(files):
        raise GateZeroCheckpointError("full policy runtime view lacks required files")
    try:
        config = json.loads((policy_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateZeroCheckpointError("invalid policy config") from error
    if config.get("use_peft") is not False:
        raise GateZeroCheckpointError("full policy runtime manifest cannot bind PEFT")
    if any(name.startswith("adapter_") for name in files):
        raise GateZeroCheckpointError("full policy runtime view contains adapter files")
    return {
        "schema_version": 2,
        "policy_role": policy_role,
        "artifact_kind": "full_policy",
        "policy_type": "smolvla",
        "training_step": training_step,
        "base": {"revision": base_revision, "weight_sha256": base_weight_sha256},
        "normalization_authority_sha256": normalization_sha256,
        "gate_zero_contract_sha256": contract_sha256,
        "files": files,
    }
