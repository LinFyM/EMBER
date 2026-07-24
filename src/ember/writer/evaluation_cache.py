"""Atomic, reusable per-episode Writer-LoRA caches for PI05 evaluation."""

from __future__ import annotations

import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from ember.lora import (
    LoRAContract,
    canonical_contract_sha256,
    lora_state_sha256,
    validate_lora_state,
)
from ember.pi05_source_checkpoint import (
    canonical_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.writer.model import WriterModelError


WRITER_LORA_CACHE_SCHEMA = "ember_pi05_writer_lora_cache_v1"
WRITER_LORA_CACHE_ENTRY_SCHEMA = "ember_pi05_writer_lora_cache_entry_v1"
WRITER_LORA_CACHE_MANIFEST_SCHEMA = "ember_pi05_writer_lora_cache_manifest_v1"
WRITER_LORA_GENERATOR_MARKER_SCHEMA = "ember_pi05_writer_lora_generator_marker_v1"
WRITER_LORA_REQUEST_ORDER = (
    "suite/task order from the sealed evaluation contract, then ascending init_state_id"
)
WRITER_LORA_ASSIGNMENT = (
    "request_ordinal modulo generator_worker_count; each worker batches its assigned "
    "subsequence in ordinal order"
)


@dataclass(frozen=True)
class WriterCacheRequest:
    suite: str
    task_id: int
    init_state_id: int
    ordinal: int

    @property
    def entry_id(self) -> str:
        if re.fullmatch(r"[a-z0-9_]+", self.suite) is None:
            raise WriterModelError("Writer cache suite is unsafe")
        return (
            f"{self.suite}_task_{self.task_id:02d}_"
            f"state_{self.init_state_id:03d}"
        )


def is_writer_adapter(adapter: Mapping[str, Any] | None) -> bool:
    return adapter is not None and adapter.get("kind") in {"as_writer", "rl_writer"}


def writer_cache_requests(contract: Mapping[str, Any]) -> tuple[WriterCacheRequest, ...]:
    requests: list[WriterCacheRequest] = []
    observed: set[tuple[str, int, int]] = set()
    for task in contract.get("tasks", []):
        suite = str(task["suite"])
        task_id = int(task["task_id"])
        state_ids = tuple(int(value) for value in task["init_state_ids"])
        if (
            not state_ids
            or len(set(state_ids)) != len(state_ids)
            or tuple(sorted(state_ids)) != state_ids
        ):
            raise WriterModelError("Writer cache task states are invalid")
        for state_id in state_ids:
            key = suite, task_id, state_id
            if key in observed:
                raise WriterModelError("Writer cache requests are duplicated")
            observed.add(key)
            requests.append(
                WriterCacheRequest(
                    suite=suite,
                    task_id=task_id,
                    init_state_id=state_id,
                    ordinal=len(requests),
                )
            )
    if not requests:
        raise WriterModelError("Writer cache has no episode requests")
    return tuple(requests)


def _cache_identity_payload(
    contract: Mapping[str, Any],
    generation_recipe: Mapping[str, Any],
) -> dict[str, Any]:
    adapter = contract.get("adapter")
    if not is_writer_adapter(adapter):
        raise WriterModelError("Writer cache requires a Writer evaluation adapter")
    tasks = [
        {
            "suite": str(task["suite"]),
            "task_id": int(task["task_id"]),
            "init_state_ids": [int(value) for value in task["init_state_ids"]],
        }
        for task in contract["tasks"]
    ]
    return {
        "schema_version": WRITER_LORA_CACHE_SCHEMA,
        "adapter": dict(adapter),
        "model": dict(contract["model"]),
        "tokenizer": dict(contract["tokenizer"]),
        "tasks": tasks,
        "policy": dict(contract["policy"]),
        "rng": {"inference_seed": int(contract["rng"]["inference_seed"])},
        "generation_recipe": dict(generation_recipe),
    }


def build_writer_lora_cache_descriptor(
    contract: Mapping[str, Any],
    *,
    root: Path,
    generators_per_gpu: int,
    generation_batch_size: int,
    lora_parameter_count: int,
    lora_tensor_count: int,
) -> dict[str, Any]:
    physical_gpu_count = int(contract["parallel"]["physical_gpu_count"])
    generator_worker_count = physical_gpu_count * generators_per_gpu
    if (
        generators_per_gpu <= 0
        or generators_per_gpu > int(contract["parallel"]["replicas_per_gpu"])
        or generation_batch_size <= 0
        or lora_parameter_count <= 0
        or lora_tensor_count <= 0
    ):
        raise WriterModelError("Writer cache generation topology is invalid")
    generation_recipe = {
        "generators_per_gpu": generators_per_gpu,
        "generator_worker_count": generator_worker_count,
        "generation_batch_size": generation_batch_size,
        "request_order": WRITER_LORA_REQUEST_ORDER,
        "assignment": WRITER_LORA_ASSIGNMENT,
        "precision": "bfloat16",
    }
    identity = _cache_identity_payload(contract, generation_recipe)
    entry_count = len(writer_cache_requests(contract))
    tensor_bytes = entry_count * lora_parameter_count * torch.bfloat16.itemsize
    return {
        "schema_version": WRITER_LORA_CACHE_SCHEMA,
        "root": str(root.resolve()),
        "identity_sha256": canonical_hash(identity),
        "identity": identity,
        "entry_count": entry_count,
        "lora_contract_sha256": str(contract["adapter"]["lora_contract_sha256"]),
        "lora_parameter_count": lora_parameter_count,
        "lora_tensor_count": lora_tensor_count,
        "estimated_tensor_bytes": tensor_bytes,
        "estimated_peak_new_bytes": tensor_bytes + entry_count * 16_384 + 1_048_576,
        "generation_recipe": generation_recipe,
        "persistent_source_policy_handoff": True,
        "writer_modules_released_before_rollout_scale_out": True,
    }


def _descriptor(contract: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = contract.get("writer_lora_cache")
    if not isinstance(descriptor, Mapping):
        raise WriterModelError("Writer evaluation contract lacks its LoRA cache")
    observed = _cache_identity_payload(contract, descriptor["generation_recipe"])
    if (
        descriptor.get("schema_version") != WRITER_LORA_CACHE_SCHEMA
        or descriptor.get("identity") != observed
        or descriptor.get("identity_sha256") != canonical_hash(observed)
        or int(descriptor.get("entry_count", -1))
        != len(writer_cache_requests(contract))
    ):
        raise WriterModelError("Writer LoRA cache descriptor changed")
    return dict(descriptor)


def writer_cache_root(contract: Mapping[str, Any]) -> Path:
    return Path(_descriptor(contract)["root"]).resolve()


def assigned_writer_cache_requests(
    contract: Mapping[str, Any],
    *,
    generator_index: int,
) -> tuple[WriterCacheRequest, ...]:
    descriptor = _descriptor(contract)
    worker_count = int(descriptor["generation_recipe"]["generator_worker_count"])
    if not 0 <= generator_index < worker_count:
        raise WriterModelError("Writer generator index is invalid")
    return tuple(
        request
        for request in writer_cache_requests(contract)
        if request.ordinal % worker_count == generator_index
    )


def _entry_root(root: Path, request: WriterCacheRequest) -> Path:
    return root / "entries" / request.entry_id


def _validated_payload(path: Path, schema: str) -> dict[str, Any]:
    payload = read_json(path)
    digest = payload.pop("canonical_payload_sha256", None)
    payload["canonical_payload_sha256"] = digest
    unhashed = {
        key: value for key, value in payload.items() if key != "canonical_payload_sha256"
    }
    if payload.get("schema_version") != schema or canonical_hash(unhashed) != digest:
        raise WriterModelError(f"Writer cache metadata changed: {path}")
    return payload


def validate_writer_cache_entry_record(
    contract: Mapping[str, Any],
    request: WriterCacheRequest,
) -> dict[str, Any]:
    descriptor = _descriptor(contract)
    entry_root = _entry_root(Path(descriptor["root"]), request)
    record_path = entry_root / "entry.json"
    lora_path = entry_root / "lora.safetensors"
    if not record_path.is_file() or not lora_path.is_file():
        raise WriterModelError(f"Writer cache entry is incomplete: {request.entry_id}")
    record = _validated_payload(record_path, WRITER_LORA_CACHE_ENTRY_SCHEMA)
    expected_request = asdict(request)
    file_record = record.get("lora_file", {})
    if (
        record.get("cache_identity_sha256") != descriptor["identity_sha256"]
        or record.get("request") != expected_request
        or record.get("entry_id") != request.entry_id
        or record.get("lora_contract_sha256")
        != descriptor["lora_contract_sha256"]
        or re.fullmatch(
            r"[0-9a-f]{64}", str(record.get("lora_state_sha256", ""))
        )
        is None
        or lora_path.stat().st_size != int(file_record.get("bytes", -1))
        or sha256_file(lora_path) != file_record.get("sha256")
    ):
        raise WriterModelError(f"Writer cache entry changed: {request.entry_id}")
    return record


def writer_cache_entry_is_complete(
    contract: Mapping[str, Any],
    request: WriterCacheRequest,
) -> bool:
    entry_root = _entry_root(writer_cache_root(contract), request)
    if not entry_root.exists():
        return False
    validate_writer_cache_entry_record(contract, request)
    return True


def write_writer_cache_entry(
    contract: Mapping[str, Any],
    request: WriterCacheRequest,
    *,
    state: Mapping[str, torch.Tensor],
    evidence: Mapping[str, Any],
    generation: Mapping[str, Any],
    lora_contract: LoRAContract,
) -> dict[str, Any]:
    descriptor = _descriptor(contract)
    if canonical_contract_sha256(lora_contract) != descriptor["lora_contract_sha256"]:
        raise WriterModelError("Writer cache LoRA contract changed")
    validate_lora_state(state, lora_contract)
    state_sha256 = lora_state_sha256(state)
    if evidence.get("lora_sha256") != state_sha256:
        raise WriterModelError("Writer cache evidence and LoRA state disagree")
    final = _entry_root(Path(descriptor["root"]), request)
    if final.exists():
        observed = validate_writer_cache_entry_record(contract, request)
        if observed.get("lora_state_sha256") != state_sha256:
            raise WriterModelError("existing Writer cache entry has another LoRA")
        return observed
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.parent / f".{request.entry_id}.{uuid.uuid4().hex}.partial"
    temporary.mkdir()
    try:
        lora_path = temporary / "lora.safetensors"
        save_file(
            {
                name: value.detach().to(device="cpu").contiguous()
                for name, value in state.items()
            },
            str(lora_path),
        )
        record = {
            "schema_version": WRITER_LORA_CACHE_ENTRY_SCHEMA,
            "cache_identity_sha256": descriptor["identity_sha256"],
            "entry_id": request.entry_id,
            "request": asdict(request),
            "lora_contract_sha256": descriptor["lora_contract_sha256"],
            "lora_state_sha256": state_sha256,
            "lora_file": {
                "bytes": lora_path.stat().st_size,
                "sha256": sha256_file(lora_path),
            },
            "evidence": dict(evidence),
            "generation": dict(generation),
        }
        record["canonical_payload_sha256"] = canonical_hash(record)
        write_json_atomic(temporary / "entry.json", record)
        os.replace(temporary, final)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_writer_cache_entry_record(contract, request)


def load_writer_cache_entry(
    contract: Mapping[str, Any],
    request: WriterCacheRequest,
    *,
    lora_contract: LoRAContract,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    descriptor = _descriptor(contract)
    if canonical_contract_sha256(lora_contract) != descriptor["lora_contract_sha256"]:
        raise WriterModelError("Writer cache runtime LoRA contract changed")
    record = validate_writer_cache_entry_record(contract, request)
    state = load_file(
        str(_entry_root(Path(descriptor["root"]), request) / "lora.safetensors"),
        device=str(device),
    )
    validate_lora_state(state, lora_contract)
    if lora_state_sha256(state) != record["lora_state_sha256"]:
        raise WriterModelError(f"Writer cache LoRA changed: {request.entry_id}")
    return state, dict(record["evidence"])


def generator_marker_path(
    contract: Mapping[str, Any], invocation_id: str, worker_id: str
) -> Path:
    if re.fullmatch(r"[0-9a-f]{32}", invocation_id) is None:
        raise WriterModelError("Writer generator invocation ID is invalid")
    if re.fullmatch(r"[0-9]+-r[0-9]+", worker_id) is None:
        raise WriterModelError("Writer generator worker ID is invalid")
    return (
        writer_cache_root(contract)
        / "generator_invocations"
        / invocation_id
        / f"{worker_id}.json"
    )


def write_generator_marker(
    contract: Mapping[str, Any],
    *,
    invocation_id: str,
    worker_id: str,
    generator_index: int,
    summary: Mapping[str, Any],
) -> Path:
    requests = assigned_writer_cache_requests(
        contract, generator_index=generator_index
    )
    marker = {
        "schema_version": WRITER_LORA_GENERATOR_MARKER_SCHEMA,
        "cache_identity_sha256": _descriptor(contract)["identity_sha256"],
        "invocation_id": invocation_id,
        "worker_id": worker_id,
        "generator_index": generator_index,
        "entry_ids": [request.entry_id for request in requests],
        "summary": dict(summary),
    }
    marker["canonical_payload_sha256"] = canonical_hash(marker)
    path = generator_marker_path(contract, invocation_id, worker_id)
    if path.exists():
        observed = _validated_payload(path, WRITER_LORA_GENERATOR_MARKER_SCHEMA)
        if observed != marker:
            raise WriterModelError("Writer generator marker changed")
    else:
        write_json_atomic(path, marker)
    return path


def validate_generator_markers(
    contract: Mapping[str, Any],
    *,
    invocation_id: str,
    worker_ids: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    descriptor = _descriptor(contract)
    expected_workers = int(
        descriptor["generation_recipe"]["generator_worker_count"]
    )
    if len(worker_ids) != expected_workers or len(set(worker_ids)) != len(worker_ids):
        raise WriterModelError("Writer generator worker layout changed")
    markers = []
    covered: list[str] = []
    for generator_index, worker_id in enumerate(worker_ids):
        marker = _validated_payload(
            generator_marker_path(contract, invocation_id, worker_id),
            WRITER_LORA_GENERATOR_MARKER_SCHEMA,
        )
        expected = assigned_writer_cache_requests(
            contract, generator_index=generator_index
        )
        if (
            marker.get("cache_identity_sha256") != descriptor["identity_sha256"]
            or marker.get("invocation_id") != invocation_id
            or marker.get("worker_id") != worker_id
            or int(marker.get("generator_index", -1)) != generator_index
            or marker.get("entry_ids") != [request.entry_id for request in expected]
            or marker.get("summary", {}).get("source_policy_reused_for_rollout")
            is not True
            or marker.get("summary", {}).get("writer_modules_released") is not True
        ):
            raise WriterModelError("Writer generator marker contract changed")
        covered.extend(marker["entry_ids"])
        markers.append(marker)
    expected_entry_ids = [
        request.entry_id for request in writer_cache_requests(contract)
    ]
    if sorted(covered) != sorted(expected_entry_ids):
        raise WriterModelError("Writer generator markers do not cover the panel")
    return tuple(markers)


def writer_cache_manifest_path(contract: Mapping[str, Any]) -> Path:
    return writer_cache_root(contract) / "cache_manifest.json"


def validate_writer_cache_manifest(
    contract: Mapping[str, Any],
    *,
    verify_entry_files: bool,
) -> dict[str, Any]:
    descriptor = _descriptor(contract)
    path = writer_cache_manifest_path(contract)
    manifest = _validated_payload(path, WRITER_LORA_CACHE_MANIFEST_SCHEMA)
    expected_ids = [request.entry_id for request in writer_cache_requests(contract)]
    entries = manifest.get("entries", [])
    if (
        manifest.get("cache_identity_sha256") != descriptor["identity_sha256"]
        or manifest.get("descriptor") != descriptor
        or manifest.get("entry_ids") != expected_ids
        or len(entries) != len(expected_ids)
        or [row.get("entry_id") for row in entries] != expected_ids
    ):
        raise WriterModelError("Writer LoRA cache manifest changed")
    if verify_entry_files:
        for request, summary in zip(
            writer_cache_requests(contract), entries, strict=True
        ):
            record = validate_writer_cache_entry_record(contract, request)
            record_path = _entry_root(
                Path(descriptor["root"]), request
            ) / "entry.json"
            if (
                summary.get("record_bytes") != record_path.stat().st_size
                or summary.get("record_sha256") != sha256_file(record_path)
                or summary.get("lora_state_sha256")
                != record.get("lora_state_sha256")
            ):
                raise WriterModelError("Writer cache manifest entry changed")
    return manifest


def writer_cache_manifest_is_ready(contract: Mapping[str, Any]) -> bool:
    path = writer_cache_manifest_path(contract)
    if not path.is_file():
        return False
    validate_writer_cache_manifest(contract, verify_entry_files=False)
    return True


def finalize_writer_cache(
    contract: Mapping[str, Any],
    *,
    invocation_id: str,
    worker_ids: Sequence[str],
) -> dict[str, Any]:
    path = writer_cache_manifest_path(contract)
    if path.exists():
        return validate_writer_cache_manifest(contract, verify_entry_files=True)
    markers = validate_generator_markers(
        contract,
        invocation_id=invocation_id,
        worker_ids=worker_ids,
    )
    descriptor = _descriptor(contract)
    entries = []
    entry_ids = []
    tensor_bytes = 0
    for request in writer_cache_requests(contract):
        record = validate_writer_cache_entry_record(contract, request)
        record_path = _entry_root(Path(descriptor["root"]), request) / "entry.json"
        entries.append(
            {
                "entry_id": request.entry_id,
                "record_bytes": record_path.stat().st_size,
                "record_sha256": sha256_file(record_path),
                "lora_state_sha256": record["lora_state_sha256"],
            }
        )
        entry_ids.append(request.entry_id)
        tensor_bytes += int(record["lora_file"]["bytes"])
    manifest = {
        "schema_version": WRITER_LORA_CACHE_MANIFEST_SCHEMA,
        "cache_identity_sha256": descriptor["identity_sha256"],
        "descriptor": descriptor,
        "entry_ids": entry_ids,
        "entries": entries,
        "tensor_file_bytes": tensor_bytes,
        "generator_invocation_id": invocation_id,
        "generator_markers": [
            {
                "worker_id": marker["worker_id"],
                "canonical_payload_sha256": marker["canonical_payload_sha256"],
            }
            for marker in markers
        ],
    }
    manifest["canonical_payload_sha256"] = canonical_hash(manifest)
    write_json_atomic(path, manifest)
    return validate_writer_cache_manifest(contract, verify_entry_files=True)
