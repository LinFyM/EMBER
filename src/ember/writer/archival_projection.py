"""Read-only canonicalization of sealed historical per-episode Writer LoRAs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ember.eval_adapters import ARCHIVAL_WRITER_CACHE_KIND
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json
from ember.writer.errors import WriterModelError


ARCHIVAL_PROJECTION_SCHEMA = "ember_pi05_archival_writer_rank_projection_v1"
ARCHIVAL_ADAPTER_SCHEMA = "ember_pi05_archival_writer_lora_cache_eval_adapter_v1"
ARCHIVAL_EPISODE_SCHEMA = "ember_pi05_archival_writer_lora_cache_episode_v1"
ARCHIVAL_WRITER_ASSET_KIND = "gomq_cycle2_effective_rank16_archival_projection"
ARCHIVAL_THROUGHPUT_POLICY = "precomputed_archival_cache_only"
REPEATED_A_FOLDED_B = "A_concat_A0_A0_B_concat_B0_deltaB_to_A0_B0_plus_deltaB"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_repo_path(value: Any) -> Path:
    path = Path(str(value))
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _relative_reference(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _source_entry_id(suite: str, task_id: int, init_state_id: int) -> str:
    return f"{suite}_task_{task_id:02d}_state_{init_state_id:03d}"


def _source_entry_path(
    adapter: Mapping[str, Any], suite: str, task_id: int, init_state_id: int
) -> Path:
    root = Path(str(adapter["archival_projection"]["source_cache_root"]))
    return root / "entries" / _source_entry_id(suite, task_id, init_state_id)


def _expected_source_request(
    adapter: Mapping[str, Any], suite: str, task_id: int, init_state_id: int
) -> dict[str, Any]:
    ordinal = 0
    found = False
    for task in adapter["archival_projection"]["source_tasks"]:
        task_suite = str(task["suite"])
        task_id_value = int(task["task_id"])
        for state_id in task["init_state_ids"]:
            state_id_value = int(state_id)
            if (task_suite, task_id_value, state_id_value) == (
                suite,
                task_id,
                init_state_id,
            ):
                found = True
                break
            ordinal += 1
        if found:
            break
    if not found:
        raise WriterModelError("archival episode is outside the source cache")
    return {
        "suite": suite,
        "task_id": task_id,
        "init_state_id": init_state_id,
        "ordinal": ordinal,
    }


def source_entry_record(
    adapter: Mapping[str, Any], suite: str, task_id: int, init_state_id: int
) -> dict[str, Any]:
    entry_id = _source_entry_id(suite, task_id, init_state_id)
    root = _source_entry_path(adapter, suite, task_id, init_state_id)
    record = read_json(root / "entry.json")
    lora_path = root / "lora.safetensors"
    projection = adapter["archival_projection"]
    if (
        record.get("entry_id") != entry_id
        or record.get("request")
        != _expected_source_request(adapter, suite, task_id, init_state_id)
        or record.get("cache_reference") != projection["source_cache_reference"]
        or record.get("lora_contract", {}).get("rank")
        != projection["source_rank"]
        or not lora_path.is_file()
        or lora_path.stat().st_size
        != int(record.get("lora_file", {}).get("bytes", -1))
    ):
        raise WriterModelError(f"archival source entry changed: {entry_id}")
    return record


def source_entry_lora_path(
    adapter: Mapping[str, Any], suite: str, task_id: int, init_state_id: int
) -> Path:
    source_entry_record(adapter, suite, task_id, init_state_id)
    return _source_entry_path(adapter, suite, task_id, init_state_id) / "lora.safetensors"


def load_archival_lora_contract(adapter: Mapping[str, Any]) -> Any:
    projection = adapter.get("archival_projection", {})
    path = Path(str(projection.get("target_lora_authority", ""))).resolve()
    contract = load_pi05_lora_contract(path)
    target_rank = int(projection.get("target_rank", -1))
    if contract.rank != target_rank:
        contract = derive_pi05_lora_rank(contract, rank=target_rank)
    reference = (
        f"{_relative_reference(path)}:{contract.state_tensor_count}tensors:"
        f"{contract.parameter_count}parameters"
    )
    if (
        target_rank != 16
        or contract.alpha != 16
        or len(contract.targets) != 38
        or adapter.get("lora_contract")
        != {
            "reference": reference,
            "rank": 16,
            "target_count": 38,
        }
    ):
        raise WriterModelError("archival target LoRA authority changed")
    return contract


def fold_repeated_a_rank32_state(
    state: Mapping[str, torch.Tensor], *, target_contract: Any
) -> dict[str, torch.Tensor]:
    """Fold [A0; A0], [B0, deltaB] into the identical effective rank-16 map."""

    source_contract = derive_pi05_lora_rank(target_contract, rank=32)
    validate_lora_state(state, source_contract)
    result: dict[str, torch.Tensor] = {}
    for target in target_contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        a = state[a_name]
        b = state[b_name]
        if not torch.equal(a[:16], a[16:]):
            raise WriterModelError(f"archival GOMQ A halves differ: {target.name}")
        result[a_name] = a[:16].clone()
        result[b_name] = (b[:, :16] + b[:, 16:]).to(dtype=b.dtype)
    validate_lora_state(result, target_contract)
    return result


def _target_storage(
    source_storage: Mapping[str, Any], *, target_parameter_count: int
) -> dict[str, Any]:
    source_parameters = int(source_storage.get("parameter_count", -1))
    source_bytes = int(source_storage.get("tensor_bytes", -1))
    dtype_parameters = {
        str(name): int(value)
        for name, value in source_storage.get("dtype_parameter_counts", {}).items()
    }
    if (
        source_parameters != 2 * target_parameter_count
        or source_bytes <= 0
        or source_bytes % 2
        or any(value % 2 for value in dtype_parameters.values())
        or int(source_storage.get("tensor_count", -1)) != 76
    ):
        raise WriterModelError("archival rank32 storage is not foldable to rank16")
    return {
        "tensor_count": 76,
        "parameter_count": target_parameter_count,
        "tensor_bytes": source_bytes // 2,
        "dtype_tensor_counts": dict(source_storage["dtype_tensor_counts"]),
        "dtype_parameter_counts": {
            name: value // 2 for name, value in dtype_parameters.items()
        },
        "dtype_by_name": dict(source_storage["dtype_by_name"]),
    }


def _inspect_source_authority(
    manifest: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    task_keys: Sequence[tuple[str, int]],
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    source_root = _resolve_repo_path(manifest["source_evaluation_root"])
    run_path = source_root / "run_contract.json"
    results_path = source_root / "results.json"
    source_run = read_json(run_path)
    results = read_json(results_path)
    adapter = source_run.get("adapter", {})
    cache = source_run.get("writer_lora_cache", {})
    cache_root = Path(str(cache.get("root", ""))).resolve()
    cache_manifest_path = cache_root / "cache_manifest.json"
    cache_manifest = read_json(cache_manifest_path)
    tasks = source_run.get("tasks", [])
    observed_keys = tuple((str(row["suite"]), int(row["task_id"])) for row in tasks)
    source_model = {
        key: str(Path(str(source_run.get("model", {}).get(key, ""))).resolve())
        for key in ("source_run", "checkpoint", "model_path")
    }
    requested_model = {
        key: str(Path(str(source[key])).resolve())
        for key in ("source_run", "checkpoint", "model_path")
    }
    result_rows = tuple(results.get("rows", ()))
    expected_entry_ids = [
        _source_entry_id(str(task["suite"]), int(task["task_id"]), int(state_id))
        for task in tasks
        for state_id in task["init_state_ids"]
    ]
    if (
        source_run.get("mode") != "formal"
        or source_run.get("role") != "validation"
        or observed_keys != tuple((str(suite), int(task_id)) for suite, task_id in task_keys)
        or len(tasks) != 8
        or any(list(map(int, task["init_state_ids"])) != list(range(50)) for task in tasks)
        or source_model != requested_model
        or int(source_run.get("rng", {}).get("inference_seed", -1)) != 7
        or adapter.get("schema_version") != manifest["source_adapter_schema"]
        or adapter.get("kind") != manifest["source_adapter_kind"]
        or adapter.get("arm") != manifest["source_arm"]
        or adapter.get("video_condition") != "correct"
        or int(adapter.get("information_wall", {}).get("evaluation_k", -1)) != 4
        or int(adapter.get("lora_contract", {}).get("rank", -1)) != 32
        or cache.get("lora_contract") != adapter.get("lora_contract")
        or cache_manifest.get("descriptor") != cache
        or cache_manifest.get("entry_ids") != expected_entry_ids
        or len(result_rows) != 400
        or sum(bool(row.get("success")) for row in result_rows)
        != int(manifest["source_successes"])
        or cache_root != (source_root / "writer_lora_cache").resolve()
    ):
        raise WriterModelError("archival GOMQ source authority changed")
    return source_run, cache_manifest, run_path, results_path


def inspect_archival_writer_projection(
    *,
    manifest_path: Path,
    source: Mapping[str, Any],
    task_keys: Sequence[tuple[str, int]],
    require_formal: bool,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != ARCHIVAL_PROJECTION_SCHEMA
        or manifest.get("projection") != REPEATED_A_FOLDED_B
        or int(manifest.get("source_rank", -1)) != 32
        or int(manifest.get("target_rank", -1)) != 16
        or manifest.get("condition") != "correct"
        or int(manifest.get("evaluation_k", -1)) != 4
        or (require_formal and manifest.get("status") != "sealed")
    ):
        raise WriterModelError("archival Writer projection manifest changed")
    source_run, cache_manifest, run_path, results_path = _inspect_source_authority(
        manifest,
        source=source,
        task_keys=task_keys,
    )
    source_adapter = source_run["adapter"]
    target_lora_path = _resolve_repo_path(manifest["target_lora_authority"])
    target_lora = load_pi05_lora_contract(target_lora_path)
    if target_lora.rank != 16 or len(target_lora.targets) != 38:
        raise WriterModelError("archival target rank16 LoRA changed")
    source_storage = source_adapter["writer_asset"]["generated_lora_storage"]
    if source_storage != source_run["writer_lora_cache"]["lora_storage_per_entry"]:
        raise WriterModelError("archival source LoRA storage changed")
    target_storage = _target_storage(
        source_storage,
        target_parameter_count=target_lora.parameter_count,
    )
    source_cache_root = Path(str(source_run["writer_lora_cache"]["root"])).resolve()
    target_reference = (
        f"{source_adapter['writer_asset']['reference']}:effective-rank16-fold"
    )
    lora_reference = (
        f"{_relative_reference(target_lora_path)}:{target_lora.state_tensor_count}tensors:"
        f"{target_lora.parameter_count}parameters"
    )
    projection = {
        "schema_version": ARCHIVAL_PROJECTION_SCHEMA,
        "manifest_path": str(manifest_path),
        "manifest_bytes": manifest_path.stat().st_size,
        "source_evaluation_root": str(run_path.parent),
        "source_run_contract": str(run_path),
        "source_results": str(results_path),
        "source_cache_root": str(source_cache_root),
        "source_cache_manifest": str(source_cache_root / "cache_manifest.json"),
        "source_cache_manifest_bytes": (
            source_cache_root / "cache_manifest.json"
        ).stat().st_size,
        "source_cache_reference": cache_manifest["cache_reference"],
        "source_tasks": list(source_run["tasks"]),
        "source_score": int(manifest["source_successes"]),
        "source_rank": 32,
        "target_rank": 16,
        "target_lora_authority": str(target_lora_path),
        "projection": REPEATED_A_FOLDED_B,
        "source_antithetic_a_halves_required_equal": True,
        "native_dtype_b_sum": True,
        "training_performed": False,
    }
    return {
        "schema_version": ARCHIVAL_ADAPTER_SCHEMA,
        "kind": ARCHIVAL_WRITER_CACHE_KIND,
        "arm": str(manifest["target_arm"]),
        "execution_backend": "sealed_archival_episode_lora_cache_only",
        "config": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "schema": ARCHIVAL_PROJECTION_SCHEMA,
        },
        "writer_asset": {
            "reference": target_reference,
            "architecture": "gomq_cycle2_effective_rank16_archival_projection",
            "kind": ARCHIVAL_WRITER_ASSET_KIND,
            "training_mode": "historical_formal_writer_cache_projection",
            "method_macro": int(source_adapter["writer_asset"]["method_macro"]),
            "writer_parameter_count": int(
                source_adapter["writer_asset"]["writer_parameter_count"]
            ),
            "deployment_trainable_parameter_count": 0,
            "generated_lora_tensor_count": target_lora.state_tensor_count,
            "generated_lora_storage": target_storage,
            "source_writer_asset_reference": source_adapter["writer_asset"][
                "reference"
            ],
        },
        "evaluation_authority": {
            "formal_status": "sealed_archival_cache_only",
            "throughput_policy": ARCHIVAL_THROUGHPUT_POLICY,
            "minimum_smoke_writer_model_batch_size": 1,
            "online_smoke_evidence": None,
        },
        "video_data": dict(source_adapter["video_data"]),
        "video_condition": "correct",
        "video_schedule": dict(source_adapter["video_schedule"]),
        "task_video_mapping": list(source_adapter["task_video_mapping"]),
        "task_video_mapping_reference": source_adapter[
            "task_video_mapping_reference"
        ],
        "pairing_reference": source_adapter["pairing_reference"],
        "lora_contract": {
            "reference": lora_reference,
            "rank": 16,
            "target_count": 38,
        },
        "source": {
            key: str(Path(str(source[key])).resolve())
            for key in ("source_run", "checkpoint", "model_path")
        },
        "information_wall": {
            **dict(source_adapter["information_wall"]),
            "archival_cache_only": True,
            "writer_generation_replayed": False,
            "training_performed": False,
        },
        "archival_projection": projection,
        "content_hash_policy": "disabled_by_owner",
    }


def reinspect_archival_writer_projection(
    adapter: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    task_keys: Sequence[tuple[str, int]],
    require_formal: bool,
) -> dict[str, Any]:
    return inspect_archival_writer_projection(
        manifest_path=Path(str(adapter["config"]["path"])),
        source=source,
        task_keys=task_keys,
        require_formal=require_formal,
    )


def archival_lora_reference(
    adapter: Mapping[str, Any], *, suite: str, task_id: int, init_state_id: int
) -> str:
    return (
        f"{adapter['writer_asset']['reference']}:{suite}:{task_id}:"
        f"{init_state_id}:rank16"
    )


def expected_archival_episode_evidence(
    adapter: Mapping[str, Any],
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
    lora_reference: str,
) -> dict[str, Any]:
    expected_reference = archival_lora_reference(
        adapter,
        suite=suite,
        task_id=task_id,
        init_state_id=init_state_id,
    )
    if lora_reference != expected_reference:
        raise WriterModelError("archival episode LoRA reference changed")
    source_record = source_entry_record(adapter, suite, task_id, init_state_id)
    source_evidence = dict(source_record["evidence"])
    source_reference = str(source_evidence["lora_reference"])
    result = {
        **source_evidence,
        "schema_version": ARCHIVAL_EPISODE_SCHEMA,
        "writer_method": ARCHIVAL_WRITER_CACHE_KIND,
        "method_arm": adapter["arm"],
        "writer_asset_reference": adapter["writer_asset"]["reference"],
        "writer_checkpoint_kind": ARCHIVAL_WRITER_ASSET_KIND,
        "generated_lora_tensor_count": 76,
        "lora_contract_reference": adapter["lora_contract"]["reference"],
        "lora_reference": expected_reference,
        "archival_projection": {
            "schema_version": ARCHIVAL_PROJECTION_SCHEMA,
            "projection": REPEATED_A_FOLDED_B,
            "source_rank": 32,
            "target_rank": 16,
            "source_lora_reference": source_reference,
            "source_successes": adapter["archival_projection"]["source_score"],
            "training_performed": False,
        },
    }
    seconds = float(result.get("writer_generation_seconds", float("nan")))
    if not math.isfinite(seconds) or seconds < 0:
        raise WriterModelError("archival source generation time changed")
    return result


def validate_archival_episode_evidence(
    adapter: Mapping[str, Any],
    row: Any,
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
) -> bool:
    if not isinstance(row, Mapping):
        return False
    try:
        expected = expected_archival_episode_evidence(
            adapter,
            suite=suite,
            task_id=task_id,
            init_state_id=init_state_id,
            lora_reference=str(row.get("lora_reference", "")),
        )
    except (KeyError, TypeError, ValueError, WriterModelError):
        return False
    return dict(row) == expected
