"""Hash-bound checkpoint and runtime-policy manifests for Gate 0."""

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


CHECKPOINT_MANIFEST = "ember_checkpoint_manifest.json"


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


def validate_hashed_tree(
    root: Path,
    files: dict[str, dict[str, Any]],
    *,
    excluded_names: set[str] | None = None,
) -> None:
    if not isinstance(files, dict) or not files:
        raise GateZeroCheckpointError("hashed file manifest is empty")
    excluded = excluded_names or set()
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in excluded
    }
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


def trainable_parameter_manifest(policy: torch.nn.Module) -> list[dict[str, Any]]:
    return [
        {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in policy.named_parameters()
        if value.requires_grad
    ]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _manifest_sidecar(checkpoint_dir: Path) -> Path:
    return checkpoint_dir.parent / f"{checkpoint_dir.name}.manifest.sha256"


def _atomic_write_sidecar(checkpoint_dir: Path) -> Path:
    manifest_path = checkpoint_dir / CHECKPOINT_MANIFEST
    sidecar = _manifest_sidecar(checkpoint_dir)
    temporary = sidecar.parent / f".{sidecar.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_text(
            f"{sha256_file(manifest_path)}  {CHECKPOINT_MANIFEST}\n", encoding="utf-8"
        )
        os.replace(temporary, sidecar)
        _fsync_directory(sidecar.parent)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
    return sidecar


def _atomic_update_last(checkpoint_dir: Path) -> None:
    last = checkpoint_dir.parent / "last"
    if last.exists() and not last.is_symlink():
        raise GateZeroCheckpointError("refusing to replace non-symlink last checkpoint")
    temporary = checkpoint_dir.parent / f".last.tmp-{uuid.uuid4().hex}"
    try:
        temporary.symlink_to(checkpoint_dir.name)
        os.replace(temporary, last)
        _fsync_directory(checkpoint_dir.parent)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def _last_target(checkpoint_root: Path) -> str | None:
    last = checkpoint_root / "last"
    if last.is_symlink():
        return os.readlink(last)
    if last.exists():
        raise GateZeroCheckpointError("refusing to replace non-symlink last checkpoint")
    return None


def _restore_last_target(checkpoint_root: Path, previous_target: str | None) -> None:
    last = checkpoint_root / "last"
    current_target = os.readlink(last) if last.is_symlink() else None
    if previous_target is None:
        if current_target is not None:
            last.unlink()
    elif current_target != previous_target:
        temporary = checkpoint_root / f".last.rollback-{uuid.uuid4().hex}"
        try:
            temporary.symlink_to(previous_target)
            os.replace(temporary, last)
        finally:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()


def _rollback_checkpoint_publication(
    checkpoint_dir: Path, *, previous_last_target: str | None
) -> None:
    _restore_last_target(checkpoint_dir.parent, previous_last_target)
    sidecar = _manifest_sidecar(checkpoint_dir)
    if sidecar.exists() or sidecar.is_symlink():
        sidecar.unlink()
    if checkpoint_dir.is_dir():
        shutil.rmtree(checkpoint_dir)
    elif checkpoint_dir.exists() or checkpoint_dir.is_symlink():
        checkpoint_dir.unlink()
    _fsync_directory(checkpoint_dir.parent)


def _validate_checkpoint_request(
    checkpoint_dir: Path, step: int, metadata: dict[str, Any]
) -> None:
    if not checkpoint_dir.is_absolute():
        raise GateZeroCheckpointError("checkpoint path must be absolute")
    if checkpoint_dir.exists() or checkpoint_dir.is_symlink() or _manifest_sidecar(checkpoint_dir).exists():
        raise GateZeroCheckpointError("refusing to overwrite checkpoint")
    if checkpoint_dir.name != f"{step:06d}":
        raise GateZeroCheckpointError("checkpoint directory does not match completed step")
    required = {"checkpoint_role", "topology", "authorities", "sampler"}
    if not required <= set(metadata):
        raise GateZeroCheckpointError("checkpoint metadata is incomplete")
    if metadata["sampler"].get("next_optimizer_step") != step:
        raise GateZeroCheckpointError("checkpoint sampler step differs from completed step")
    topology = metadata["topology"]
    world_size = topology.get("world_size")
    if (
        not isinstance(world_size, int)
        or isinstance(world_size, bool)
        or world_size <= 0
        or world_size > 4
    ):
        raise GateZeroCheckpointError("checkpoint world size is invalid")
    if world_size > 1 and topology.get("checkpoint_writer_rank") != 0:
        raise GateZeroCheckpointError("distributed checkpoint must be written by rank 0")


def _write_distributed_rng_states(
    staging: Path,
    states: list[dict[str, torch.Tensor]],
    *,
    world_size: int,
) -> dict[str, Any]:
    if len(states) != world_size:
        raise GateZeroCheckpointError("distributed RNG rank count differs from topology")
    root = staging / "distributed_rng"
    root.mkdir(parents=True, exist_ok=False)
    names = []
    for rank, state in enumerate(states):
        if not isinstance(state, dict) or not state:
            raise GateZeroCheckpointError("distributed RNG state is empty")
        if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
            raise GateZeroCheckpointError("distributed RNG state must be a tensor dictionary")
        name = f"rank_{rank:05d}.safetensors"
        save_file(
            {
                key: value.detach().to(device="cpu").contiguous()
                for key, value in state.items()
            },
            root / name,
        )
        names.append(name)
    return {
        "format": "lerobot_flat_rng_safetensors_v1",
        "world_size": world_size,
        "files": names,
    }


def save_source_base_checkpoint(
    checkpoint_dir: Path,
    *,
    step: int,
    policy: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    preprocessor: Any,
    postprocessor: Any,
    metadata: dict[str, Any],
    rank_rng_states: list[dict[str, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    """Write one LeRobot-format full-state checkpoint and publish it atomically."""

    _validate_checkpoint_request(checkpoint_dir, step, metadata)
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    previous_last_target = _last_target(checkpoint_dir.parent)
    staging = checkpoint_dir.parent / f".{checkpoint_dir.name}.tmp-{uuid.uuid4().hex}"
    published = False
    try:
        pretrained = staging / "pretrained_model"
        policy.save_pretrained(pretrained)
        preprocessor.save_pretrained(pretrained)
        postprocessor.save_pretrained(pretrained)
        from lerobot.common.train_utils import save_training_state

        save_training_state(
            staging,
            step,
            optimizer,
            scheduler,
            num_processes=metadata["topology"]["world_size"],
            batch_size=metadata["topology"]["micro_batch_size"],
        )
        world_size = metadata["topology"]["world_size"]
        if world_size > 1 and rank_rng_states is None:
            raise GateZeroCheckpointError("distributed checkpoint lacks every-rank RNG state")
        distributed_rng = None
        if rank_rng_states is not None:
            distributed_rng = _write_distributed_rng_states(
                staging, rank_rng_states, world_size=world_size
            )
        authorities = metadata["authorities"]
        policy_runtime = build_policy_runtime_manifest(
            pretrained,
            policy_role=metadata["checkpoint_role"],
            training_step=step,
            base_revision=authorities["base_revision"],
            base_weight_sha256=authorities["base_weight_sha256"],
            normalization_sha256=authorities["normalization_sha256"],
            contract_sha256=authorities["gate_zero_contract_sha256"],
        )
        manifest = {
            "schema_version": 3 if distributed_rng is not None else 2,
            "step": step,
            **metadata,
            "trainable_parameters": trainable_parameter_manifest(policy),
            "policy_runtime": policy_runtime,
        }
        if distributed_rng is not None:
            manifest["distributed_rng"] = distributed_rng
        manifest["files"] = hashed_tree(staging, excluded_names={CHECKPOINT_MANIFEST})
        _write_json(staging / CHECKPOINT_MANIFEST, manifest)
        _fsync_directory(staging)
        os.replace(staging, checkpoint_dir)
        published = True
        _fsync_directory(checkpoint_dir.parent)
        _atomic_write_sidecar(checkpoint_dir)
        _atomic_update_last(checkpoint_dir)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        if published:
            _rollback_checkpoint_publication(
                checkpoint_dir, previous_last_target=previous_last_target
            )
        raise


def _require_expected(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    for key, value in expected.items():
        if actual.get(key) != value:
            raise GateZeroCheckpointError(f"checkpoint authority changed: {key}")


def validate_source_base_checkpoint(
    checkpoint_dir: Path, *, expected: dict[str, Any] | None = None
) -> dict[str, Any]:
    if not checkpoint_dir.is_dir():
        raise GateZeroCheckpointError("checkpoint directory is missing")
    manifest_path = checkpoint_dir / CHECKPOINT_MANIFEST
    sidecar = _manifest_sidecar(checkpoint_dir)
    try:
        fields = sidecar.read_text(encoding="utf-8").strip().split()
    except OSError as error:
        raise GateZeroCheckpointError("checkpoint manifest sidecar is missing") from error
    if len(fields) != 2 or fields[1] != CHECKPOINT_MANIFEST:
        raise GateZeroCheckpointError("invalid checkpoint manifest sidecar")
    if sha256_file(manifest_path) != fields[0]:
        raise GateZeroCheckpointError("checkpoint manifest hash changed")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateZeroCheckpointError("invalid checkpoint manifest") from error
    if manifest.get("schema_version") not in {2, 3}:
        raise GateZeroCheckpointError("unsupported checkpoint schema")
    validate_hashed_tree(
        checkpoint_dir,
        manifest.get("files"),
        excluded_names={CHECKPOINT_MANIFEST},
    )
    validate_hashed_tree(
        checkpoint_dir / "pretrained_model", manifest["policy_runtime"]["files"]
    )
    if manifest["schema_version"] == 3:
        distributed_rng = manifest.get("distributed_rng")
        world_size = manifest.get("topology", {}).get("world_size")
        if (
            not isinstance(world_size, int)
            or isinstance(world_size, bool)
            or world_size <= 0
            or world_size > 4
        ):
            raise GateZeroCheckpointError("distributed checkpoint world size is invalid")
        expected_files = [f"rank_{rank:05d}.safetensors" for rank in range(world_size)]
        if (
            not isinstance(distributed_rng, dict)
            or distributed_rng.get("format") != "lerobot_flat_rng_safetensors_v1"
            or distributed_rng.get("world_size") != world_size
            or distributed_rng.get("files") != expected_files
        ):
            raise GateZeroCheckpointError("distributed RNG manifest differs from topology")
        actual_files = sorted(
            path.name for path in (checkpoint_dir / "distributed_rng").glob("rank_*.safetensors")
        )
        if actual_files != expected_files:
            raise GateZeroCheckpointError("distributed RNG files differ from topology")
    if expected is not None:
        _require_expected(manifest, expected)
    return manifest


def load_source_base_training_state_without_rng(
    checkpoint_dir: Path,
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    expected: dict[str, Any] | None = None,
) -> tuple[int, torch.optim.Optimizer, Any]:
    """Restore model/optimizer/scheduler while deliberately leaving RNG untouched."""

    manifest = validate_source_base_checkpoint(checkpoint_dir, expected=expected)
    if trainable_parameter_manifest(policy) != manifest["trainable_parameters"]:
        raise GateZeroCheckpointError("ordered trainable parameter manifest changed")
    model_state = load_file(checkpoint_dir / "pretrained_model" / "model.safetensors")
    incompatible = policy.load_state_dict(model_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise GateZeroCheckpointError("checkpoint model state is incomplete")
    from lerobot.common.train_utils import load_training_step
    from lerobot.optim import load_optimizer_state, load_scheduler_state

    training_state = checkpoint_dir / "training_state"
    step = load_training_step(training_state)
    if step != manifest["step"]:
        raise GateZeroCheckpointError("training step differs from checkpoint manifest")
    optimizer = load_optimizer_state(optimizer, training_state)
    scheduler = load_scheduler_state(scheduler, training_state)
    return step, optimizer, scheduler


def restore_source_base_checkpoint_rng(
    checkpoint_dir: Path, *, rank: int = 0, world_size: int = 1
) -> None:
    """Restore RNG only after the resumed loader and iterator have been constructed."""

    manifest = validate_source_base_checkpoint(checkpoint_dir)
    checkpoint_world_size = manifest["topology"]["world_size"]
    if checkpoint_world_size != world_size or not 0 <= rank < world_size:
        raise GateZeroCheckpointError("resume RNG topology differs from checkpoint")
    if manifest["schema_version"] == 2:
        if world_size != 1 or rank != 0:
            raise GateZeroCheckpointError("legacy checkpoint RNG is single-rank only")
        from lerobot.utils.random_utils import load_rng_state

        load_rng_state(checkpoint_dir / "training_state")
        return
    from lerobot.utils.random_utils import deserialize_rng_state

    state = load_file(checkpoint_dir / "distributed_rng" / f"rank_{rank:05d}.safetensors")
    deserialize_rng_state(state)


def rotate_source_base_recovery_checkpoints(
    checkpoint_root: Path, *, keep: int
) -> list[int]:
    """Remove only validated, superseded recovery checkpoints; preserve other roles."""

    if keep < 1:
        raise GateZeroCheckpointError("at least one recovery checkpoint must be retained")
    recovery: list[tuple[int, Path]] = []
    for path in sorted(checkpoint_root.iterdir()):
        if not path.is_dir() or not path.name.isdigit():
            continue
        manifest = validate_source_base_checkpoint(path)
        if manifest["checkpoint_role"] == "source_base_training_recovery":
            recovery.append((int(path.name), path))
    removed: list[int] = []
    for step, path in recovery[:-keep]:
        trash = checkpoint_root / f".{path.name}.trash-{uuid.uuid4().hex}"
        os.replace(path, trash)
        sidecar = _manifest_sidecar(path)
        if sidecar.exists():
            sidecar.unlink()
        shutil.rmtree(trash)
        removed.append(step)
    if removed:
        _fsync_directory(checkpoint_root)
    return removed
