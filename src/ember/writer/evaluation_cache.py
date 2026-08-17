"""Atomic per-episode Writer-LoRA caches with method-specific schemas."""

from __future__ import annotations

import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from ember.eval_adapters import (
    DYNAMIC_K_WRITER_KIND,
    WRITER_ADAPTER_KINDS,
    validate_writer_episode,
    writer_episode_schema,
)
from ember.lora import LoRAContract, validate_lora_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.errors import WriterModelError


WRITER_LORA_REQUEST_ORDER = "sealed suite/task order then ascending init_state_id"
DYNAMIC_K_WRITER_LORA_CACHE_SCHEMA = "ember_pi05_dynamic_k_writer_lora_cache_v1"
DYNAMIC_K_WRITER_LORA_CACHE_ENTRY_SCHEMA = (
    "ember_pi05_dynamic_k_writer_lora_cache_entry_v1"
)
DYNAMIC_K_WRITER_LORA_CACHE_MANIFEST_SCHEMA = (
    "ember_pi05_dynamic_k_writer_lora_cache_manifest_v1"
)
DYNAMIC_K_WRITER_LORA_GENERATOR_MARKER_SCHEMA = (
    "ember_pi05_dynamic_k_writer_lora_generator_marker_v1"
)
DYNAMIC_K_WRITER_LORA_VIDEO_KEY_ALGORITHM = (
    "one_entry_per_episode_dynamic_k_k1_video_set_v1"
)
DYNAMIC_K_WRITER_LORA_VIDEO_SET_KEY_ALGORITHM = (
    "one_entry_per_episode_dynamic_k_nested_video_set_v1"
)
WRITER_LORA_VIDEO_REQUEST_ORDER = WRITER_LORA_REQUEST_ORDER
WRITER_LORA_ASSIGNMENT = (
    "sealed request order chunked by generation_batch_size into contiguous global "
    "batches then batch ordinal modulo generator worker count"
)
WRITER_LORA_LEGACY_ASSIGNMENT = "request ordinal modulo generator worker count"
_TORCH_DTYPE_NAMES = {
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
    torch.float64: "F64",
}


def _writer_cache_schemas(adapter: Mapping[str, Any]) -> dict[str, str]:
    kind = adapter.get("kind")
    if kind == DYNAMIC_K_WRITER_KIND:
        evaluation_k = int(
            adapter.get("information_wall", {}).get("evaluation_k", 1)
        )
        return {
            "cache": DYNAMIC_K_WRITER_LORA_CACHE_SCHEMA,
            "entry": DYNAMIC_K_WRITER_LORA_CACHE_ENTRY_SCHEMA,
            "manifest": DYNAMIC_K_WRITER_LORA_CACHE_MANIFEST_SCHEMA,
            "marker": DYNAMIC_K_WRITER_LORA_GENERATOR_MARKER_SCHEMA,
            "key_algorithm": (
                DYNAMIC_K_WRITER_LORA_VIDEO_KEY_ALGORITHM
                if evaluation_k == 1
                else DYNAMIC_K_WRITER_LORA_VIDEO_SET_KEY_ALGORITHM
            ),
        }
    raise WriterModelError("Writer cache adapter kind changed")


@dataclass(frozen=True)
class WriterCacheRequest:
    suite: str
    task_id: int
    init_state_id: int
    ordinal: int

    def record(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "task_id": self.task_id,
            "init_state_id": self.init_state_id,
            "ordinal": self.ordinal,
        }

    @property
    def entry_id(self) -> str:
        if re.fullmatch(r"[a-z0-9_]+", self.suite) is None:
            raise WriterModelError("Writer cache suite is unsafe")
        return f"{self.suite}_task_{self.task_id:02d}_state_{self.init_state_id:03d}"


@dataclass(frozen=True)
class WriterCacheGenerationBatch:
    ordinal: int
    requests: tuple[WriterCacheRequest, ...]
    canonical_global: bool


def is_writer_adapter(adapter: Mapping[str, Any] | None) -> bool:
    return adapter is not None and adapter.get("kind") in WRITER_ADAPTER_KINDS


def _writer_cache_layout(
    contract: Mapping[str, Any],
) -> tuple[
    tuple[WriterCacheRequest, ...], dict[tuple[str, int, int], WriterCacheRequest]
]:
    requests = []
    by_episode = {}
    for task in contract.get("tasks", []):
        suite = str(task["suite"])
        task_id = int(task["task_id"])
        states = tuple(int(value) for value in task["init_state_ids"])
        if (
            not states
            or len(set(states)) != len(states)
            or tuple(sorted(states)) != states
        ):
            raise WriterModelError("Writer cache task states are invalid")
        for state_id in states:
            key = suite, task_id, state_id
            if key in by_episode:
                raise WriterModelError("Writer cache requests are duplicated")
            request = WriterCacheRequest(suite, task_id, state_id, len(requests))
            requests.append(request)
            by_episode[key] = request
    if not requests:
        raise WriterModelError("Writer cache has no episode requests")
    return tuple(requests), by_episode


def writer_cache_requests(
    contract: Mapping[str, Any]
) -> tuple[WriterCacheRequest, ...]:
    return _writer_cache_layout(contract)[0]


def writer_cache_episode_request_map(
    contract: Mapping[str, Any],
) -> dict[tuple[str, int, int], WriterCacheRequest]:
    return _writer_cache_layout(contract)[1]


def _cache_identity_payload(
    contract: Mapping[str, Any], generation_recipe: Mapping[str, Any]
) -> dict[str, Any]:
    adapter = contract.get("adapter")
    if not is_writer_adapter(adapter):
        raise WriterModelError("Writer cache requires a Writer adapter")
    return {
        "schema_version": _writer_cache_schemas(adapter)["cache"],
        "adapter": dict(adapter),
        "model_step": int(contract["model"]["optimizer_step"]),
        "tokenizer_path": str(contract["tokenizer"]["path"]),
        "tasks": [
            {
                "suite": str(task["suite"]),
                "task_id": int(task["task_id"]),
                "init_state_ids": [int(value) for value in task["init_state_ids"]],
            }
            for task in contract["tasks"]
        ],
        "inference_seed": int(contract["rng"]["inference_seed"]),
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
    lora_storage_per_entry: Mapping[str, Any],
) -> dict[str, Any]:
    configured_worker_count = (
        int(contract["parallel"]["physical_gpu_count"]) * generators_per_gpu
    )
    if (
        generators_per_gpu <= 0
        or generators_per_gpu > int(contract["parallel"]["replicas_per_gpu"])
        or generation_batch_size <= 0
        or min(lora_parameter_count, lora_tensor_count) <= 0
    ):
        raise WriterModelError("Writer cache generation topology is invalid")
    try:
        storage = {
            "tensor_count": int(lora_storage_per_entry["tensor_count"]),
            "parameter_count": int(lora_storage_per_entry["parameter_count"]),
            "tensor_bytes": int(lora_storage_per_entry["tensor_bytes"]),
            "dtype_tensor_counts": {
                str(name): int(value)
                for name, value in lora_storage_per_entry["dtype_tensor_counts"].items()
            },
            "dtype_parameter_counts": {
                str(name): int(value)
                for name, value in lora_storage_per_entry[
                    "dtype_parameter_counts"
                ].items()
            },
            "dtype_by_name": {
                str(name): str(value)
                for name, value in lora_storage_per_entry["dtype_by_name"].items()
            },
        }
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise WriterModelError(
            "Writer cache LoRA storage contract is invalid"
        ) from error
    if (
        storage["tensor_count"] != lora_tensor_count
        or storage["parameter_count"] != lora_parameter_count
        or storage["tensor_bytes"] <= 0
        or sum(storage["dtype_tensor_counts"].values()) != lora_tensor_count
        or sum(storage["dtype_parameter_counts"].values()) != lora_parameter_count
        or len(storage["dtype_by_name"]) != lora_tensor_count
        or set(storage["dtype_by_name"].values()) - set(storage["dtype_tensor_counts"])
    ):
        raise WriterModelError("Writer cache LoRA storage contract changed")
    adapter = contract["adapter"]
    recipe = {
        "generators_per_gpu": generators_per_gpu,
        "generator_worker_count": configured_worker_count,
        "generation_batch_size": generation_batch_size,
        "cache_key_algorithm": _writer_cache_schemas(adapter)["key_algorithm"],
        "episode_evidence_schema": writer_episode_schema(adapter),
        "request_order": WRITER_LORA_VIDEO_REQUEST_ORDER,
        "assignment": WRITER_LORA_ASSIGNMENT,
        "precision": "bfloat16_compute_template_native_mixed_lora_state",
        "storage_per_entry": storage,
    }
    identity = _cache_identity_payload(contract, recipe)
    entry_count = len(writer_cache_requests(contract))
    tensor_bytes = entry_count * storage["tensor_bytes"]
    return {
        "schema_version": _writer_cache_schemas(adapter)["cache"],
        "root": str(root.resolve()),
        "reference": (
            f"{contract['adapter']['writer_asset']['reference']}:"
            f"{entry_count}episodes:seed{contract['rng']['inference_seed']}:"
            f"batch{generation_batch_size}:native{storage['tensor_bytes']}bytes"
        ),
        "identity": identity,
        "entry_count": entry_count,
        "lora_contract": dict(contract["adapter"]["lora_contract"]),
        "lora_parameter_count": lora_parameter_count,
        "lora_tensor_count": lora_tensor_count,
        "lora_storage_per_entry": storage,
        "estimated_tensor_bytes": tensor_bytes,
        "estimated_peak_new_bytes": tensor_bytes + entry_count * 16_384 + 1_048_576,
        "generation_recipe": recipe,
        "persistent_source_policy_handoff": True,
        "writer_modules_released_before_rollout_scale_out": True,
    }


def stage_writer_lora_states_to_cpu(
    states: Sequence[Mapping[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor], ...]:
    """Submit one batch of GPU-to-host copies and synchronize only once."""

    if not states:
        raise WriterModelError("Writer cache staging batch is empty")
    result: list[dict[str, torch.Tensor]] = [dict() for _ in states]
    pending_devices: dict[torch.device, torch.cuda.Stream] = {}
    staged_buffers: list[torch.Tensor] = []
    cpu_values: list[torch.Tensor] = []
    pending_values: dict[
        tuple[torch.device, torch.dtype],
        list[tuple[int, str, torch.Tensor]],
    ] = {}
    for state_index, state in enumerate(states):
        for name, value in state.items():
            detached = value.detach()
            if detached.device.type == "cuda":
                pending_values.setdefault((detached.device, detached.dtype), []).append(
                    (state_index, name, detached)
                )
            else:
                cpu_value = detached.to(device="cpu").contiguous()
                result[state_index][name] = cpu_value
                cpu_values.append(cpu_value)
    for (device, dtype), values in pending_values.items():
        flat = torch.empty(
            sum(value.numel() for _, _, value in values),
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
        offset = 0
        for state_index, name, value in values:
            stop = offset + value.numel()
            destination = flat[offset:stop].view(value.shape)
            destination.copy_(value, non_blocking=True)
            result[state_index][name] = destination
            offset = stop
        pending_devices[device] = torch.cuda.current_stream(device)
        staged_buffers.append(flat)
    for stream in pending_devices.values():
        stream.synchronize()
    if any(
        not bool(torch.isfinite(value).all().item())
        for value in (*staged_buffers, *cpu_values)
    ):
        raise WriterModelError("Writer cache staging produced a non-finite LoRA")
    return tuple(result)


def _descriptor(contract: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = contract.get("writer_lora_cache")
    if not isinstance(descriptor, Mapping):
        raise WriterModelError("Writer evaluation lacks its LoRA cache")
    observed = _cache_identity_payload(contract, descriptor["generation_recipe"])
    if (
        descriptor.get("schema_version")
        != _writer_cache_schemas(contract["adapter"])["cache"]
        or descriptor.get("identity") != observed
        or int(descriptor.get("entry_count", -1))
        != len(writer_cache_requests(contract))
    ):
        raise WriterModelError("Writer LoRA cache descriptor changed")
    return dict(descriptor)


def writer_cache_root(contract: Mapping[str, Any]) -> Path:
    return Path(_descriptor(contract)["root"]).resolve()


def assigned_writer_cache_requests(
    contract: Mapping[str, Any], *, generator_index: int
) -> tuple[WriterCacheRequest, ...]:
    return tuple(
        request
        for batch in assigned_writer_cache_batches(
            contract,
            generator_index=generator_index,
        )
        for request in batch.requests
    )


def assigned_writer_cache_batches(
    contract: Mapping[str, Any], *, generator_index: int
) -> tuple[WriterCacheGenerationBatch, ...]:
    descriptor = _descriptor(contract)
    recipe = descriptor["generation_recipe"]
    workers = int(recipe["generator_worker_count"])
    if not 0 <= generator_index < workers:
        raise WriterModelError("Writer generator index is invalid")
    batch_size = int(recipe["generation_batch_size"])
    requests = writer_cache_requests(contract)
    assignment = str(recipe.get("assignment", ""))
    if assignment == WRITER_LORA_ASSIGNMENT:
        global_batches = tuple(
            WriterCacheGenerationBatch(
                ordinal=batch_ordinal,
                requests=requests[offset : offset + batch_size],
                canonical_global=True,
            )
            for batch_ordinal, offset in enumerate(range(0, len(requests), batch_size))
        )
        return tuple(
            batch
            for batch in global_batches
            if batch.ordinal % workers == generator_index
        )
    if assignment == WRITER_LORA_LEGACY_ASSIGNMENT:
        assigned = tuple(
            request
            for request in requests
            if request.ordinal % workers == generator_index
        )
        return tuple(
            WriterCacheGenerationBatch(
                ordinal=batch_ordinal,
                requests=assigned[offset : offset + batch_size],
                canonical_global=False,
            )
            for batch_ordinal, offset in enumerate(range(0, len(assigned), batch_size))
        )
    raise WriterModelError("Writer cache assignment algorithm is unsupported")


def _entry_root(root: Path, request: WriterCacheRequest) -> Path:
    return root / "entries" / request.entry_id


def validate_writer_cache_entry_record(
    contract: Mapping[str, Any], request: WriterCacheRequest
) -> dict[str, Any]:
    descriptor = _descriptor(contract)
    root = _entry_root(Path(descriptor["root"]), request)
    record_path, lora_path = root / "entry.json", root / "lora.safetensors"
    if not record_path.is_file() or not lora_path.is_file():
        raise WriterModelError(f"Writer cache entry is incomplete: {request.entry_id}")
    record = read_json(record_path)
    if (
        record.get("schema_version")
        != _writer_cache_schemas(contract["adapter"])["entry"]
        or record.get("cache_reference") != descriptor["reference"]
        or record.get("request") != request.record()
        or record.get("entry_id") != request.entry_id
        or record.get("cache_identity") != descriptor["identity"]
        or record.get("lora_contract") != descriptor["lora_contract"]
        or record.get("lora_storage") != descriptor["lora_storage_per_entry"]
        or lora_path.stat().st_size != int(record.get("lora_file", {}).get("bytes", -1))
    ):
        raise WriterModelError(f"Writer cache entry changed: {request.entry_id}")
    return record


def writer_cache_entry_is_complete(
    contract: Mapping[str, Any], request: WriterCacheRequest
) -> bool:
    root = _entry_root(writer_cache_root(contract), request)
    if not root.exists():
        return False
    validate_writer_cache_entry_record(contract, request)
    return True


def _validate_lora_contract(
    descriptor: Mapping[str, Any], contract: LoRAContract
) -> None:
    if (
        int(descriptor["lora_parameter_count"]) != contract.parameter_count
        or int(descriptor["lora_tensor_count"]) != contract.state_tensor_count
    ):
        raise WriterModelError("Writer cache LoRA topology changed")


def lora_state_storage(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Describe one native LoRA state without copying or changing its dtype."""

    dtype_tensor_counts: dict[str, int] = {}
    dtype_parameter_counts: dict[str, int] = {}
    tensor_bytes = 0
    dtype_by_name: dict[str, str] = {}
    for name, value in state.items():
        dtype = _TORCH_DTYPE_NAMES.get(value.dtype)
        if dtype is None:
            raise WriterModelError("Writer cache LoRA dtype changed")
        dtype_tensor_counts[dtype] = dtype_tensor_counts.get(dtype, 0) + 1
        dtype_parameter_counts[dtype] = (
            dtype_parameter_counts.get(dtype, 0) + value.numel()
        )
        tensor_bytes += value.numel() * value.element_size()
        dtype_by_name[name] = dtype
    return {
        "tensor_count": len(state),
        "parameter_count": sum(value.numel() for value in state.values()),
        "tensor_bytes": tensor_bytes,
        "dtype_tensor_counts": dict(sorted(dtype_tensor_counts.items())),
        "dtype_parameter_counts": dict(sorted(dtype_parameter_counts.items())),
        "dtype_by_name": dict(sorted(dtype_by_name.items())),
    }


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
    _validate_lora_contract(descriptor, lora_contract)
    validate_lora_state(state, lora_contract)
    if lora_state_storage(state) != descriptor["lora_storage_per_entry"]:
        raise WriterModelError("Writer cache LoRA native storage changed")
    if not validate_writer_episode(
        contract["adapter"],
        evidence,
        suite=request.suite,
        task_id=request.task_id,
        init_state_id=request.init_state_id,
    ):
        raise WriterModelError("Writer cache episode evidence changed")
    final = _entry_root(Path(descriptor["root"]), request)
    if final.exists():
        return validate_writer_cache_entry_record(contract, request)
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.parent / f".{request.entry_id}.{uuid.uuid4().hex}.partial"
    temporary.mkdir()
    try:
        lora_path = temporary / "lora.safetensors"
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in state.items()},
            str(lora_path),
        )
        write_json_atomic(
            temporary / "entry.json",
            {
                "schema_version": _writer_cache_schemas(contract["adapter"])[
                    "entry"
                ],
                "cache_reference": descriptor["reference"],
                "cache_identity": descriptor["identity"],
                "entry_id": request.entry_id,
                "request": request.record(),
                "lora_contract": descriptor["lora_contract"],
                "lora_storage": descriptor["lora_storage_per_entry"],
                "lora_file": {"bytes": lora_path.stat().st_size},
                "evidence": dict(evidence),
                "generation": dict(generation),
            },
        )
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
    _validate_lora_contract(descriptor, lora_contract)
    record = validate_writer_cache_entry_record(contract, request)
    state = load_file(
        str(_entry_root(Path(descriptor["root"]), request) / "lora.safetensors"),
        device=str(device),
    )
    validate_lora_state(state, lora_contract)
    if lora_state_storage(state) != descriptor["lora_storage_per_entry"]:
        raise WriterModelError("Writer cache loaded LoRA storage changed")
    return state, dict(record["evidence"])


def generator_marker_path(
    contract: Mapping[str, Any], invocation_id: str, worker_id: str
) -> Path:
    if (
        re.fullmatch(r"[0-9a-f]{32}", invocation_id) is None
        or re.fullmatch(r"[0-9]+-r[0-9]+", worker_id) is None
    ):
        raise WriterModelError("Writer generator invocation identity is invalid")
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
    requests = assigned_writer_cache_requests(contract, generator_index=generator_index)
    marker = {
        "schema_version": _writer_cache_schemas(contract["adapter"])["marker"],
        "cache_reference": _descriptor(contract)["reference"],
        "invocation_id": invocation_id,
        "worker_id": worker_id,
        "generator_index": generator_index,
        "entry_ids": [request.entry_id for request in requests],
        "summary": dict(summary),
    }
    path = generator_marker_path(contract, invocation_id, worker_id)
    if path.exists():
        if read_json(path) != marker:
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
    workers = int(descriptor["generation_recipe"]["generator_worker_count"])
    if len(worker_ids) != workers or len(set(worker_ids)) != len(worker_ids):
        raise WriterModelError("Writer generator worker layout changed")
    markers, covered = [], []
    for index, worker_id in enumerate(worker_ids):
        marker = read_json(generator_marker_path(contract, invocation_id, worker_id))
        expected = assigned_writer_cache_requests(contract, generator_index=index)
        if (
            marker.get("schema_version")
            != _writer_cache_schemas(contract["adapter"])["marker"]
            or marker.get("cache_reference") != descriptor["reference"]
            or marker.get("invocation_id") != invocation_id
            or marker.get("worker_id") != worker_id
            or int(marker.get("generator_index", -1)) != index
            or marker.get("entry_ids") != [request.entry_id for request in expected]
            or marker.get("summary", {}).get("source_policy_reused_for_rollout")
            is not True
            or marker.get("summary", {}).get("writer_modules_released") is not True
        ):
            raise WriterModelError("Writer generator marker contract changed")
        covered.extend(marker["entry_ids"])
        markers.append(marker)
    expected_ids = [request.entry_id for request in writer_cache_requests(contract)]
    if sorted(covered) != sorted(expected_ids):
        raise WriterModelError("Writer generator markers do not cover the panel")
    return tuple(markers)


def writer_cache_manifest_path(contract: Mapping[str, Any]) -> Path:
    return writer_cache_root(contract) / "cache_manifest.json"


def validate_writer_cache_manifest(
    contract: Mapping[str, Any], *, verify_entry_files: bool
) -> dict[str, Any]:
    descriptor = _descriptor(contract)
    manifest = read_json(writer_cache_manifest_path(contract))
    expected_ids = [request.entry_id for request in writer_cache_requests(contract)]
    if (
        manifest.get("schema_version")
        != _writer_cache_schemas(contract["adapter"])["manifest"]
        or manifest.get("cache_reference") != descriptor["reference"]
        or manifest.get("descriptor") != descriptor
        or manifest.get("entry_ids") != expected_ids
        or not isinstance(manifest.get("generator_invocation_id"), str)
        or not isinstance(manifest.get("generator_workers"), list)
    ):
        raise WriterModelError("Writer LoRA cache manifest changed")
    if verify_entry_files:
        for request in writer_cache_requests(contract):
            validate_writer_cache_entry_record(contract, request)
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
    validate_generator_markers(
        contract, invocation_id=invocation_id, worker_ids=worker_ids
    )
    descriptor = _descriptor(contract)
    entries, tensor_bytes = [], 0
    for request in writer_cache_requests(contract):
        record = validate_writer_cache_entry_record(contract, request)
        entries.append(
            {"entry_id": request.entry_id, "lora_bytes": record["lora_file"]["bytes"]}
        )
        tensor_bytes += int(record["lora_file"]["bytes"])
    manifest = {
        "schema_version": _writer_cache_schemas(contract["adapter"])["manifest"],
        "cache_reference": descriptor["reference"],
        "descriptor": descriptor,
        "entry_ids": [row["entry_id"] for row in entries],
        "entries": entries,
        "tensor_file_bytes": tensor_bytes,
        "generator_invocation_id": invocation_id,
        "generator_workers": list(worker_ids),
    }
    write_json_atomic(path, manifest)
    return validate_writer_cache_manifest(contract, verify_entry_files=True)
