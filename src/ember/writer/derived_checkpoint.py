"""Inference-only derived checkpoints for canonical PI05 Writer states."""

from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from ember.pi05_source_checkpoint import (
    canonical_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.writer.checkpoint import validate_writer_checkpoint_files
from ember.writer.model import WriterModelError


AS_WRITER_DERIVED_CHECKPOINT_SCHEMA = (
    "ember_pi05_language_axial_writer_average_checkpoint_v1"
)
UNIFORM_PARAMETER_AVERAGE_ALGORITHM = (
    "uniform_arithmetic_mean_float32_then_cast_original_dtype"
)
_OUTPUT_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def _valid_source_record(row: Any, source_count: int) -> bool:
    if not isinstance(row, Mapping):
        return False
    try:
        return (
            isinstance(row.get("path"), str)
            and bool(row["path"])
            and int(row.get("cursor", -1)) > 0
            and int(row.get("weight_numerator", -1)) == 1
            and int(row.get("weight_denominator", -1)) == source_count
            and all(
                _is_sha256(row.get(field))
                for field in (
                    "checkpoint_manifest_file_sha256",
                    "checkpoint_manifest_payload_sha256",
                    "writer_state_sha256",
                )
            )
        )
    except (TypeError, ValueError):
        return False


def _source_record(
    checkpoint: Path,
    *,
    run_root: Path,
    run_contract: Mapping[str, Any],
    contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    checkpoint = checkpoint.resolve()
    if checkpoint.parent != run_root / "checkpoints":
        raise WriterModelError(
            "derived Writer source is outside the selected training run"
        )
    world_size = int(run_contract.get("runtime", {}).get("world_size", -1))
    if world_size <= 0:
        raise WriterModelError("derived Writer source topology is invalid")
    manifest = validate_writer_checkpoint_files(
        checkpoint,
        world_size=world_size,
        contract_sha256=contract_sha256,
    )
    cursor = int(manifest.get("consumed", {}).get("next_step", -1))
    checkpoint_steps = {
        int(value)
        for value in run_contract.get("runtime", {}).get("checkpoint_steps", [])
    }
    writer_record = manifest.get("files", {}).get("writer.safetensors", {})
    if (
        cursor <= 0
        or cursor not in checkpoint_steps
        or checkpoint.name != f"step_{cursor:08d}"
        or not isinstance(writer_record.get("sha256"), str)
    ):
        raise WriterModelError("derived Writer source checkpoint authority changed")
    state = load_file(str(checkpoint / "writer.safetensors"), device="cpu")
    return (
        {
            "path": str(checkpoint),
            "cursor": cursor,
            "checkpoint_manifest_file_sha256": sha256_file(
                checkpoint / "checkpoint_manifest.json"
            ),
            "checkpoint_manifest_payload_sha256": manifest[
                "canonical_payload_sha256"
            ],
            "writer_state_sha256": writer_record["sha256"],
        },
        state,
    )


def _validate_state_contract(
    reference: Mapping[str, torch.Tensor],
    observed: Mapping[str, torch.Tensor],
) -> None:
    if set(observed) != set(reference):
        raise WriterModelError("derived Writer source tensor names changed")
    for name, expected in reference.items():
        actual = observed[name]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise WriterModelError(
                f"derived Writer source tensor contract changed: {name}"
            )


def derive_uniform_writer_average_checkpoint(
    *,
    source_run: Path,
    source_checkpoints: Sequence[Path],
    output_name: str,
) -> tuple[Path, dict[str, Any]]:
    """Average compatible raw Writer checkpoints into one inference-only state."""

    run_root = source_run.resolve()
    if not (run_root / "run_contract.json").is_file():
        raise WriterModelError("derived Writer source run lost its contract")
    if _OUTPUT_NAME.fullmatch(output_name) is None:
        raise WriterModelError("invalid derived Writer checkpoint name")
    normalized = tuple(path.resolve() for path in source_checkpoints)
    if len(normalized) < 2 or len(set(normalized)) != len(normalized):
        raise WriterModelError(
            "uniform Writer averaging requires at least two distinct checkpoints"
        )
    output_root = run_root / "derived_checkpoints"
    final = output_root / output_name
    if final.exists():
        raise WriterModelError(f"derived Writer checkpoint already exists: {final}")

    run_contract = read_json(run_root / "run_contract.json")
    contract_sha256 = canonical_hash(run_contract)
    source_records: list[dict[str, Any]] = []
    reference: dict[str, torch.Tensor] | None = None
    accumulators: dict[str, torch.Tensor] = {}
    fixed: dict[str, torch.Tensor] = {}
    original_dtypes: dict[str, torch.dtype] = {}
    tensor_shapes: dict[str, tuple[int, ...]] = {}
    for checkpoint in normalized:
        record, state = _source_record(
            checkpoint,
            run_root=run_root,
            run_contract=run_contract,
            contract_sha256=contract_sha256,
        )
        source_records.append(record)
        if reference is None:
            reference = state
            for name, value in state.items():
                if value.is_complex():
                    raise WriterModelError(
                        f"complex derived Writer tensor is unsupported: {name}"
                    )
                original_dtypes[name] = value.dtype
                tensor_shapes[name] = tuple(value.shape)
                if value.is_floating_point():
                    accumulators[name] = value.to(dtype=torch.float32)
                else:
                    fixed[name] = value.clone()
            continue
        _validate_state_contract(reference, state)
        for name, value in state.items():
            if name in accumulators:
                accumulators[name].add_(value.to(dtype=torch.float32))
            elif not torch.equal(value, fixed[name]):
                raise WriterModelError(
                    f"non-floating derived Writer tensor differs: {name}"
                )

    assert reference is not None
    divisor = len(normalized)
    averaged = {
        name: (
            accumulators[name].div_(divisor).to(dtype=original_dtypes[name])
            if name in accumulators
            else fixed[name]
        ).contiguous()
        for name in reference
    }
    parameter_count = sum(value.numel() for value in averaged.values())

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / f".{output_name}.{uuid.uuid4().hex}.partial"
    try:
        temporary.mkdir()
        save_file(averaged, str(temporary / "writer.safetensors"))
        writer_path = temporary / "writer.safetensors"
        manifest: dict[str, Any] = {
            "schema_version": AS_WRITER_DERIVED_CHECKPOINT_SCHEMA,
            "contract_sha256": contract_sha256,
            "inference_only": True,
            "derivation": {
                "algorithm": UNIFORM_PARAMETER_AVERAGE_ALGORITHM,
                "source_cursor_axis": "optimizer_step",
                "source_checkpoints": [
                    {
                        **record,
                        "weight_numerator": 1,
                        "weight_denominator": divisor,
                    }
                    for record in source_records
                ],
                "tensor_count": len(averaged),
                "parameter_count": parameter_count,
                "floating_accumulator_dtype": "torch.float32",
                "output_tensor_dtypes": {
                    name: str(original_dtypes[name]) for name in sorted(averaged)
                },
                "output_tensor_shapes": {
                    name: list(tensor_shapes[name]) for name in sorted(averaged)
                },
            },
            "files": {
                "writer.safetensors": {
                    "bytes": writer_path.stat().st_size,
                    "sha256": sha256_file(writer_path),
                }
            },
        }
        manifest["canonical_payload_sha256"] = canonical_hash(manifest)
        write_json_atomic(temporary / "checkpoint_manifest.json", manifest)
        os.replace(temporary, final)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return final, manifest


def validate_derived_writer_checkpoint_files(
    checkpoint: Path,
    *,
    contract_sha256: str,
) -> dict[str, Any]:
    """Validate a derived state and its immutable, inference-only provenance."""

    checkpoint = checkpoint.resolve()
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    payload = dict(manifest)
    digest = payload.pop("canonical_payload_sha256", None)
    files = manifest.get("files", {})
    derivation = manifest.get("derivation", {})
    sources = (
        derivation.get("source_checkpoints", [])
        if isinstance(derivation, Mapping)
        else []
    )
    valid_sources = (
        isinstance(sources, list)
        and len(sources) >= 2
        and all(_valid_source_record(row, len(sources)) for row in sources)
        and len({str(row["path"]) for row in sources}) == len(sources)
    )
    try:
        tensor_count = int(derivation.get("tensor_count", -1))
        parameter_count = int(derivation.get("parameter_count", -1))
    except (AttributeError, TypeError, ValueError):
        tensor_count = -1
        parameter_count = -1
    if (
        checkpoint.parent.name != "derived_checkpoints"
        or manifest.get("schema_version")
        != AS_WRITER_DERIVED_CHECKPOINT_SCHEMA
        or manifest.get("contract_sha256") != contract_sha256
        or manifest.get("inference_only") is not True
        or canonical_hash(payload) != digest
        or not isinstance(files, dict)
        or set(files) != {"writer.safetensors"}
        or not isinstance(derivation, Mapping)
        or derivation.get("algorithm") != UNIFORM_PARAMETER_AVERAGE_ALGORITHM
        or derivation.get("source_cursor_axis") != "optimizer_step"
        or not valid_sources
        or tensor_count <= 0
        or parameter_count <= 0
    ):
        raise WriterModelError("derived AS-Writer checkpoint manifest changed")
    record = files["writer.safetensors"]
    writer_path = checkpoint / "writer.safetensors"
    if (
        not writer_path.is_file()
        or writer_path.stat().st_size != int(record.get("bytes", -1))
        or sha256_file(writer_path) != record.get("sha256")
    ):
        raise WriterModelError(
            "derived AS-Writer checkpoint file changed: writer.safetensors"
        )
    return manifest


def validate_derived_writer_checkpoint_provenance(
    checkpoint: Path,
    *,
    run_root: Path,
    run_contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[int, ...]:
    """Recheck the small source manifests without reloading optimizer payloads."""

    run_root = run_root.resolve()
    checkpoint_steps = {
        int(value)
        for value in run_contract.get("runtime", {}).get("checkpoint_steps", [])
    }
    cursors: list[int] = []
    for record in manifest["derivation"]["source_checkpoints"]:
        source = Path(str(record["path"])).resolve()
        cursor = int(record["cursor"])
        if (
            source.parent != run_root / "checkpoints"
            or cursor not in checkpoint_steps
            or source.name != f"step_{cursor:08d}"
        ):
            raise WriterModelError(
                "derived AS-Writer source checkpoint authority changed"
            )
        source_manifest_path = source / "checkpoint_manifest.json"
        source_manifest = read_json(source_manifest_path)
        writer_record = source_manifest.get("files", {}).get(
            "writer.safetensors", {}
        )
        if (
            sha256_file(source_manifest_path)
            != record["checkpoint_manifest_file_sha256"]
            or source_manifest.get("canonical_payload_sha256")
            != record["checkpoint_manifest_payload_sha256"]
            or writer_record.get("sha256") != record["writer_state_sha256"]
            or source_manifest.get("contract_sha256")
            != manifest.get("contract_sha256")
        ):
            raise WriterModelError(
                "derived AS-Writer source checkpoint provenance changed"
            )
        cursors.append(cursor)
    if len(set(cursors)) != len(cursors):
        raise WriterModelError("derived AS-Writer source cursors are duplicated")
    return tuple(cursors)
