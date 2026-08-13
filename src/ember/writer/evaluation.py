"""Deployment authority and episode evidence for the Dynamic-K Writer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from safetensors import safe_open

from ember.eval_adapters import DYNAMIC_K_WRITER_KIND
from ember.expert_manifold.video_schedule import (
    frame_order_seed,
    reference_demo_index,
    task_video_mapping,
    video_schedule_contract,
    video_selection_seed,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json
from ember.writer.as_config import (
    AS_WRITER_CONFIG_SCHEMA,
    AS_WRITER_LAUNCH_SCHEMA,
    authority_path,
    load_writer_config,
    parse_macro_boundaries,
)
from ember.writer.checkpoint import (
    CHECKPOINT_SCHEMA,
    DEPLOYMENT_CHECKPOINT_KIND,
    checkpoint_macro,
)
from ember.writer.errors import WriterModelError


DYNAMIC_K_ADAPTER_SCHEMA = (
    "ember_pi05_dynamic_k_semantic_address_direct_family_b_rank8_eval_adapter_v1"
)
DYNAMIC_K_EPISODE_SCHEMA = (
    "ember_pi05_dynamic_k_semantic_address_direct_family_b_rank8_episode_v1"
)
DYNAMIC_K_CHECKPOINT_KIND = DEPLOYMENT_CHECKPOINT_KIND
DYNAMIC_K_PAIRING_REFERENCE = "ember_pi05_dynamic_k_one_shot_pairing_v1"
DYNAMIC_K_VIDEO_CONDITIONS = frozenset({"correct"})
DYNAMIC_K_EVALUATION_STATUS = "sealed"
DYNAMIC_K_GENERATION_BATCH_SIZE = 8
DYNAMIC_K_GENERATION_PROFILE = {
    "schema": "ember_pi05_writer_generation_profile_v2",
    "path": (
        "runs/outputs/"
        "pi05_dynamic_k_semantic_address_direct_family_b_writer_generation_"
        "profile_val8x4_correct_gpu01p1_3866f50_20260813/"
        "writer_generation_profile.json"
    ),
    "selected_writer_model_batch_size": DYNAMIC_K_GENERATION_BATCH_SIZE,
}


def _target_rows(config: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    manifest = read_json(authority_path(config, "target_data_manifest"))
    rows = {int(row["global_task_id"]): dict(row) for row in manifest["tasks"]}
    if len(rows) != 40:
        raise WriterModelError("dynamic-K target manifest changed")
    return rows


def _template_lora_names(lora: Any) -> tuple[str, ...]:
    result = []
    for target in sorted(lora.targets, key=lambda value: value.name):
        result.extend(
            (
                f"{target.name}.lora_A.default.weight",
                f"{target.name}.lora_B.default.weight",
            )
        )
    return tuple(result)


def _writer_state_record(path: Path, lora: Any) -> dict[str, Any]:
    if not path.is_file():
        raise WriterModelError("dynamic-K deployment Writer state is missing")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        names = tuple(handle.keys())
        shapes = {name: tuple(handle.get_slice(name).get_shape()) for name in names}
        dtypes = {name: str(handle.get_slice(name).get_dtype()) for name in names}
    template_keys = tuple(
        sorted(name for name in names if name.startswith("lora_mapper.template_"))
    )
    lora_names = _template_lora_names(lora)
    if len(template_keys) != len(lora_names):
        raise WriterModelError("dynamic-K deployment LoRA template changed")
    item_sizes = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}
    dtype_tensor_counts: dict[str, int] = {}
    dtype_parameter_counts: dict[str, int] = {}
    dtype_by_name: dict[str, str] = {}
    for key, name in zip(template_keys, lora_names, strict=True):
        dtype = dtypes[key]
        if dtype not in item_sizes:
            raise WriterModelError("dynamic-K LoRA template dtype changed")
        count = math.prod(shapes[key])
        dtype_tensor_counts[dtype] = dtype_tensor_counts.get(dtype, 0) + 1
        dtype_parameter_counts[dtype] = dtype_parameter_counts.get(dtype, 0) + count
        dtype_by_name[name] = dtype
    parameter_count = sum(dtype_parameter_counts.values())
    if (
        parameter_count != lora.parameter_count
        or len(template_keys) != lora.state_tensor_count
    ):
        raise WriterModelError("dynamic-K LoRA template storage changed")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "state_tensor_count": len(names),
        "state_value_count": sum(math.prod(shape) for shape in shapes.values()),
        "template_lora_storage": {
            "tensor_count": len(template_keys),
            "parameter_count": parameter_count,
            "tensor_bytes": sum(
                item_sizes[dtype] * count
                for dtype, count in dtype_parameter_counts.items()
            ),
            "dtype_tensor_counts": dict(sorted(dtype_tensor_counts.items())),
            "dtype_parameter_counts": dict(sorted(dtype_parameter_counts.items())),
            "dtype_by_name": dict(sorted(dtype_by_name.items())),
        },
    }


def _writer_asset(
    *,
    config: Mapping[str, Any],
    checkpoint: Path,
    source: Mapping[str, Any],
    require_formal: bool,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    macro = checkpoint_macro(checkpoint)
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = read_json(manifest_path)
    world_size = int(manifest.get("world_size", -1))
    expected_files = {
        "writer.safetensors",
        "trainer_state.pt",
        *(f"rank_{rank:02d}_state.pt" for rank in range(world_size)),
    }
    if (
        world_size <= 0
        or manifest.get("schema_version") != CHECKPOINT_SCHEMA
        or int(manifest.get("next_macro", -1)) != macro
        or manifest.get("run_contract_schema") != AS_WRITER_LAUNCH_SCHEMA
        or set(manifest.get("files", {})) != expected_files
    ):
        raise WriterModelError("dynamic-K deployment checkpoint changed")
    for name, record in manifest["files"].items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise WriterModelError(f"dynamic-K checkpoint file changed: {name}")

    run_path = checkpoint.parent.parent / "run_contract.json"
    run = read_json(run_path)
    run_source = run.get("source", {})
    expected_source = {
        key: str(Path(str(source[key])).resolve())
        for key in ("source_run", "checkpoint", "model_path")
    }
    observed_source = {
        key: str(Path(str(run_source.get(key, ""))).resolve())
        for key in expected_source
    }
    if (
        run.get("schema_version") != AS_WRITER_LAUNCH_SCHEMA
        or observed_source != expected_source
        or int(run_source.get("optimizer_step", -1)) != 1000
        or run.get("authorities") != config["authorities"]
        or run.get("writer") != config["writer"]
        or run.get("data") != config["data"]
        or run.get("conditioning_training") != config["conditioning_training"]
        or run.get("optimization") != config["optimization"]
    ):
        raise WriterModelError("dynamic-K training authority changed")
    formal_macros = parse_macro_boundaries(
        config["formal_run"]["checkpoint_macros"],
        int(config["formal_run"]["total_macros"]),
    )
    if require_formal and (
        config["formal_run"]["status"] != "sealed"
        or run.get("mode") != "formal"
        or macro not in formal_macros
    ):
        raise WriterModelError(
            "formal dynamic-K evaluation requires a sealed formal checkpoint"
        )
    writer_state = _writer_state_record(
        checkpoint / "writer.safetensors",
        load_pi05_lora_contract(authority_path(config, "lora_contract")),
    )
    trainable = run.get("trainable", {})
    writer_parameters = int(trainable.get("writer_parameter_count", -1))
    if writer_parameters <= 0:
        raise WriterModelError("dynamic-K Writer parameter contract changed")
    return {
        "kind": DYNAMIC_K_CHECKPOINT_KIND,
        "training_mode": str(run.get("mode")),
        "method_macro": macro,
        "checkpoint": str(checkpoint),
        "checkpoint_manifest": {
            "path": str(manifest_path.resolve()),
            "bytes": manifest_path.stat().st_size,
            "schema": CHECKPOINT_SCHEMA,
            "world_size": world_size,
        },
        "training_run_contract": {
            "path": str(run_path.resolve()),
            "bytes": run_path.stat().st_size,
            "schema": AS_WRITER_LAUNCH_SCHEMA,
        },
        "writer_parameter_count": writer_parameters,
        "deployment_trainable_parameter_count": 0,
        "writer_state": writer_state,
    }


def _video_contract(
    *,
    config: Mapping[str, Any],
    video_data_root: Path,
    task_keys: Sequence[tuple[str, int]],
    video_seed: int,
    video_sampling_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows = _target_rows(config)
    by_key = {(str(row["suite"]), int(row["task_id"])): row for row in rows.values()}
    normalized = tuple((str(suite), int(task_id)) for suite, task_id in task_keys)
    if (
        not normalized
        or len(set(normalized)) != len(normalized)
        or any(key not in by_key for key in normalized)
    ):
        raise WriterModelError("dynamic-K evaluation task panel changed")
    roles = {key: str(by_key[key]["split_role"]) for key in normalized}
    mapping = list(task_video_mapping(normalized, roles, "correct"))
    root = video_data_root.resolve()
    records = []
    for global_task_id in sorted({int(row["video_global_task_id"]) for row in mapping}):
        row = rows[global_task_id]
        hdf5 = row["hdf5"]
        path = (root / str(hdf5["relative_path"])).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or path.stat().st_size != int(hdf5["bytes"])
        ):
            raise WriterModelError("dynamic-K evaluation video HDF5 changed")
        records.append(
            {
                "global_task_id": global_task_id,
                "suite": str(row["suite"]),
                "task_id": int(row["task_id"]),
                "split_role": str(row["split_role"]),
                "language": str(row["language"]),
                "relative_path": str(hdf5["relative_path"]),
                "bytes": int(hdf5["bytes"]),
            }
        )
    schedule, _ = video_schedule_contract(
        seed=video_seed,
        demo_count=50,
        sampling_mode=video_sampling_mode,
    )
    schedule.update(
        {
            "frame_stride": int(config["writer"]["frame_stride"]),
            "include_final_frame": bool(config["writer"]["include_final_frame"]),
            "backbone_total_frames_per_condition": int(
                config["writer"]["backbone_total_frames_per_condition"]
            ),
        }
    )
    return (
        mapping,
        {
            "root": str(root),
            "tasks": records,
            "teacher_action_reads": 0,
            "teacher_state_reads": 0,
            "reward_reads": 0,
            "terminal_reads": 0,
        },
        schedule,
    )


def inspect_dynamic_k_writer_evaluation(
    *,
    config_path: Path,
    checkpoint: Path,
    video_data_root: Path,
    source: Mapping[str, Any],
    task_keys: Sequence[tuple[str, int]],
    video_condition: str,
    video_seed: int,
    video_sampling_mode: str,
    require_formal: bool,
) -> dict[str, Any]:
    """Inspect one K1 deployment without loading optimizer or RNG payloads."""

    config_path = config_path.resolve()
    config = load_writer_config(config_path)
    if video_condition not in DYNAMIC_K_VIDEO_CONDITIONS:
        raise WriterModelError("dynamic-K evaluator currently supports K1 correct only")
    writer_asset = _writer_asset(
        config=config,
        checkpoint=checkpoint,
        source=source,
        require_formal=require_formal,
    )
    mapping, video_data, schedule = _video_contract(
        config=config,
        video_data_root=video_data_root,
        task_keys=task_keys,
        video_seed=video_seed,
        video_sampling_mode=video_sampling_mode,
    )
    lora_path = authority_path(config, "lora_contract")
    lora = load_pi05_lora_contract(lora_path)
    reference = (
        f"{AS_WRITER_CONFIG_SCHEMA}:{DYNAMIC_K_CHECKPOINT_KIND}:"
        f"m{writer_asset['method_macro']}:"
        f"{writer_asset['writer_state']['bytes']}bytes:rank8"
    )
    return {
        "schema_version": DYNAMIC_K_ADAPTER_SCHEMA,
        "kind": DYNAMIC_K_WRITER_KIND,
        "arm": (
            "dynamic_k_semantic_address_direct_family_b_rank8_"
            f"{video_condition}"
        ),
        "execution_backend": ("online_frozen_dynamic_k_writer_then_episode_lora_cache"),
        "config": {
            "path": str(config_path),
            "bytes": config_path.stat().st_size,
            "schema": AS_WRITER_CONFIG_SCHEMA,
        },
        "writer_asset": {
            "reference": reference,
            "architecture": str(config["writer"]["architecture"]),
            "generated_lora_tensor_count": lora.state_tensor_count,
            **writer_asset,
        },
        "evaluation_authority": {
            "formal_status": DYNAMIC_K_EVALUATION_STATUS,
            "throughput_policy": (
                "highest_measured_batch_throughput_with_device_memory_headroom"
            ),
            "minimum_smoke_writer_model_batch_size": (
                DYNAMIC_K_GENERATION_BATCH_SIZE
            ),
            "online_smoke_evidence": dict(DYNAMIC_K_GENERATION_PROFILE),
        },
        "video_data": video_data,
        "video_condition": video_condition,
        "video_schedule": schedule,
        "task_video_mapping": mapping,
        "task_video_mapping_reference": (
            f"{config['authorities']['target_data_manifest']['path']}:"
            f"{len(mapping)}tasks:{video_condition}:K1"
        ),
        "pairing_reference": DYNAMIC_K_PAIRING_REFERENCE,
        "lora_contract": {
            "reference": (
                f"{lora_path.relative_to(Path(__file__).resolve().parents[3])}:"
                f"{lora.state_tensor_count}tensors:{lora.parameter_count}parameters"
            ),
            "rank": lora.rank,
            "target_count": len(lora.targets),
        },
        "source": {
            key: str(Path(str(source[key])).resolve())
            for key in ("source_run", "checkpoint", "model_path")
        },
        "information_wall": {
            "writer_input": (
                "exact task language plus one action-hidden teacher video through "
                "the dynamic-K graph"
            ),
            "video_is_only_dynamic_value": True,
            "no_video_counterfactual": False,
            "teacher_action_reads": 0,
            "teacher_state_reads": 0,
            "reward_reads": 0,
            "terminal_reads": 0,
            "language_only_lora_path": False,
            "deployment_expert_bank_read": False,
            "evaluation_k": 1,
            "dynamic_k_training_range": [
                1,
                int(config["data"]["dynamic_k_max"]),
            ],
            "frame_stride": int(config["writer"]["frame_stride"]),
            "backbone_total_frames_per_condition": int(
                config["writer"]["backbone_total_frames_per_condition"]
            ),
        },
        "content_hash_policy": "disabled_by_owner",
    }


def dynamic_k_episode_schema(adapter: Mapping[str, Any]) -> str:
    if (
        adapter.get("kind") != DYNAMIC_K_WRITER_KIND
        or adapter.get("schema_version") != DYNAMIC_K_ADAPTER_SCHEMA
        or adapter.get("config", {}).get("schema") != AS_WRITER_CONFIG_SCHEMA
    ):
        raise WriterModelError("invalid dynamic-K deployment adapter")
    return DYNAMIC_K_EPISODE_SCHEMA


def expected_dynamic_k_episode_evidence(
    adapter: Mapping[str, Any],
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
    lora_reference: str,
) -> dict[str, Any]:
    if not lora_reference:
        raise WriterModelError("dynamic-K episode lost its LoRA reference")
    schema = dynamic_k_episode_schema(adapter)
    matches = [
        row
        for row in adapter["task_video_mapping"]
        if row["suite"] == suite and int(row["task_id"]) == task_id
    ]
    if len(matches) != 1:
        raise WriterModelError("episode task is outside dynamic-K mapping")
    mapping = matches[0]
    schedule = adapter["video_schedule"]
    seed = int(schedule["seed"])
    mode = str(schedule["sampling_mode"])
    demo_count = int(schedule["demo_count"])
    selected = reference_demo_index(
        seed,
        suite,
        task_id,
        init_state_id,
        demo_count=demo_count,
        sampling_mode=mode,
    )
    asset = adapter["writer_asset"]
    return {
        "schema_version": schema,
        "writer_method": DYNAMIC_K_WRITER_KIND,
        "method_arm": adapter["arm"],
        "condition": adapter["video_condition"],
        "evaluation_k": 1,
        "condition_video_offsets": [0, 1],
        "backbone_total_frames_per_condition": int(
            schedule["backbone_total_frames_per_condition"]
        ),
        "writer_asset_reference": asset["reference"],
        "writer_checkpoint_kind": asset["kind"],
        "writer_method_macro": int(asset["method_macro"]),
        "writer_parameter_count": int(asset["writer_parameter_count"]),
        "writer_deployment_trainable_parameter_count": 0,
        "generated_lora_tensor_count": int(asset["generated_lora_tensor_count"]),
        "lora_contract_reference": adapter["lora_contract"]["reference"],
        "lora_reference": lora_reference,
        "language_global_task_id": int(mapping["language_global_task_id"]),
        "teacher_video_kind": adapter["video_condition"],
        "teacher_video_frames_used": True,
        "teacher_video_count": 1,
        "teacher_video_seed_root": seed,
        "teacher_video_selection_seed": video_selection_seed(
            seed,
            suite,
            task_id,
            init_state_id,
            sampling_mode=mode,
        ),
        "teacher_video_sampling_mode": mode,
        "video_suite": str(mapping["video_suite"]),
        "video_task_id": int(mapping["video_task_id"]),
        "video_global_task_id": int(mapping["video_global_task_id"]),
        "video_split_role": str(mapping["video_split_role"]),
        "teacher_demo_indices": [selected],
        "teacher_reference_demo_indices": [selected],
        "task_video_mapping_reference": adapter["task_video_mapping_reference"],
        "pairing_reference": adapter["pairing_reference"],
        "writer_generation_seed_schedule": (
            "numeric_seedsequence_one_shot_frame_order_v1"
        ),
        "teacher_video_order_seeds": [frame_order_seed(seed, suite, task_id, selected)],
    }


def validate_dynamic_k_episode_evidence(
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
        seconds = float(row.get("writer_generation_seconds", float("nan")))
        expected = expected_dynamic_k_episode_evidence(
            adapter,
            suite=suite,
            task_id=task_id,
            init_state_id=init_state_id,
            lora_reference=str(row.get("lora_reference", "")),
        )
    except (KeyError, TypeError, ValueError, WriterModelError):
        return False
    observed = dict(row)
    observed.pop("writer_generation_seconds", None)
    return observed == expected and math.isfinite(seconds) and seconds >= 0
