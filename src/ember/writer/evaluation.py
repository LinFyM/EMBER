"""Deployment authority and episode evidence for the Dynamic-K Writer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from safetensors import safe_open

from ember.eval_adapters import DYNAMIC_K_WRITER_KIND
from ember.expert_manifold.video_schedule import (
    VIDEO_CONDITIONS,
    frame_order_seed,
    paired_condition_demo_indices,
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
    "ember_pi05_layer_matched_memory_program_compiler_eval_adapter_v3"
)
DYNAMIC_K_EPISODE_SCHEMA = (
    "ember_pi05_layer_matched_memory_program_compiler_episode_v3"
)
DYNAMIC_K_CHECKPOINT_KIND = DEPLOYMENT_CHECKPOINT_KIND
DYNAMIC_K_PAIRING_REFERENCE = "ember_pi05_dynamic_k_one_shot_pairing_v1"
DYNAMIC_K_VIDEO_SET_PAIRING_REFERENCE = (
    "ember_pi05_dynamic_k_nested_video_set_pairing_v1"
)
DYNAMIC_K_VIDEO_CONDITIONS = frozenset(VIDEO_CONDITIONS)
DYNAMIC_K_GENERATION_BATCH_SIZE = 32
DYNAMIC_K_GENERATION_SAFE_BATCH_SIZE = 16
DYNAMIC_K_GENERATION_PROFILES: dict[int, dict[str, Any]] = {
    4: {
        "schema_version": "ember_pi05_writer_generation_profile_v2",
        "evidence_path": (
            "runs/acceptance/"
            "pi05_lmmpc_k4_generation_profile_77f45c9_macro2_"
            "gpu01p2_20260817/writer_generation_profile.json"
        ),
        "evidence_bytes": 10774,
        "authority_commit": "77f45c99c0c241e13d85ed152d2cff5e55895b76",
        "device": "NVIDIA A40",
        "profiled_writer_model_batch_sizes": [8, 16, 32],
        "supported_writer_model_batch_sizes": [8, 16, 32],
        "selected_writer_model_batch_size": 32,
        "selection_rule": (
            "highest_measured_fixed_panel_loras_per_second_with_stable_"
            "longest_video_batch"
        ),
        "panel_entry_count": 32,
        "panel_total_sampled_frames": 4438,
        "longest_sampled_video_frames": 226,
        "warmup_runs_per_batch": 1,
        "measured_runs_per_batch": 2,
        "writer_generation_measurements": [
            {
                "batch_size": 8,
                "loras_per_second": 0.19655253615761045,
                "repeat_wall_seconds": [
                    162.87199536594562,
                    162.74069450888783,
                ],
                "peak_reserved_bytes": 16873684992,
                "memory_headroom_bytes": 30826037248,
                "stable": True,
            },
            {
                "batch_size": 16,
                "loras_per_second": 0.19805086508070918,
                "repeat_wall_seconds": [
                    161.54325406299904,
                    161.60605393885635,
                ],
                "peak_reserved_bytes": 20268974080,
                "memory_headroom_bytes": 27430748160,
                "stable": True,
            },
            {
                "batch_size": 32,
                "loras_per_second": 0.1994860904111841,
                "repeat_wall_seconds": [
                    160.4106404078193,
                    160.41373320179991,
                ],
                "peak_reserved_bytes": 25557991424,
                "memory_headroom_bytes": 22141730816,
                "stable": True,
            },
        ],
        "oom_count": 0,
        "nonfinite_count": 0,
        "teacher_action_reads": 0,
        "teacher_state_reads": 0,
        "reward_reads": 0,
        "terminal_reads": 0,
    }
}


def dynamic_k_writer_input(evaluation_k: int) -> str:
    if evaluation_k == 1:
        return (
            "exact task language plus one action-hidden teacher video through "
            "the dynamic-K graph"
        )
    return (
        f"exact task language plus {evaluation_k} action-hidden teacher videos "
        "through the dynamic-K graph"
    )


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
        sorted(name for name in names if name.startswith("template_"))
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


def _generated_lora_storage_record(
    template_storage: Mapping[str, Any],
    lora: Any,
) -> dict[str, Any]:
    """Describe a rank-derived public LoRA using the native template dtypes."""

    template_parameters = int(template_storage["parameter_count"])
    if (
        template_parameters <= 0
        or lora.parameter_count % template_parameters
        or int(template_storage["tensor_count"]) != lora.state_tensor_count
    ):
        raise WriterModelError("generated LoRA storage topology changed")
    rank_multiplier = lora.parameter_count // template_parameters
    return {
        "tensor_count": int(template_storage["tensor_count"]),
        "parameter_count": int(lora.parameter_count),
        "tensor_bytes": int(template_storage["tensor_bytes"]) * rank_multiplier,
        "dtype_tensor_counts": dict(template_storage["dtype_tensor_counts"]),
        "dtype_parameter_counts": {
            str(name): int(value) * rank_multiplier
            for name, value in template_storage["dtype_parameter_counts"].items()
        },
        "dtype_by_name": dict(template_storage["dtype_by_name"]),
    }


def _writer_asset(
    *,
    config: Mapping[str, Any],
    checkpoint: Path,
    source: Mapping[str, Any],
    require_formal: bool,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = read_json(manifest_path)
    macro = checkpoint_macro(checkpoint)
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
    evaluation_k: int,
    video_condition: str,
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
    mapping = list(task_video_mapping(normalized, roles, video_condition))
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
    if evaluation_k > 1:
        schedule.update(
            {
                "algorithm": (
                    "numeric_seeded_task_permutation_nested_k_video_set_per_"
                    "state_block"
                ),
                "videos_per_condition": evaluation_k,
                "nested_k1_prefix": True,
                "within_condition_unique": True,
            }
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


def _evaluation_contract(
    *,
    config: Mapping[str, Any],
    evaluation_k: int,
    video_condition: str,
    video_sampling_mode: str,
    require_formal: bool,
) -> Mapping[str, Any] | None:
    if not 1 <= evaluation_k <= int(config["data"]["dynamic_k_max"]):
        raise WriterModelError("dynamic-K evaluation K is outside training support")
    if video_condition not in DYNAMIC_K_VIDEO_CONDITIONS:
        raise WriterModelError("dynamic-K video condition is unsupported")
    if evaluation_k > 1 and video_sampling_mode != "without_replacement":
        raise WriterModelError(
            "multi-video evaluation requires without-replacement video sampling"
        )
    profile = DYNAMIC_K_GENERATION_PROFILES.get(evaluation_k)
    if require_formal and profile is None:
        raise WriterModelError(
            f"formal dynamic-K K{evaluation_k} evaluation requires a live profile"
        )
    return profile


def _evaluation_information_wall(
    config: Mapping[str, Any], evaluation_k: int, video_condition: str
) -> dict[str, Any]:
    return {
        "writer_input": dynamic_k_writer_input(evaluation_k),
        "video_is_only_dynamic_value": True,
        "no_video_counterfactual": video_condition == "no_video",
        "teacher_action_reads": 0,
        "teacher_state_reads": 0,
        "reward_reads": 0,
        "terminal_reads": 0,
        "language_only_lora_path": False,
        "deployment_expert_bank_read": False,
        "evaluation_k": evaluation_k,
        "dynamic_k_training_range": [1, int(config["data"]["dynamic_k_max"])],
        "frame_stride": int(config["writer"]["frame_stride"]),
        "backbone_total_frames_per_condition": int(
            config["writer"]["backbone_total_frames_per_condition"]
        ),
    }


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
    evaluation_k: int = 1,
) -> dict[str, Any]:
    """Inspect one Dynamic-K deployment without loading optimizer or RNG payloads."""

    config_path = config_path.resolve()
    config = load_writer_config(config_path)
    profile = _evaluation_contract(
        config=config,
        evaluation_k=evaluation_k,
        video_condition=video_condition,
        video_sampling_mode=video_sampling_mode,
        require_formal=require_formal,
    )
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
        evaluation_k=evaluation_k,
        video_condition=video_condition,
    )
    lora_path = authority_path(config, "lora_contract")
    lora = load_pi05_lora_contract(lora_path)
    generated_lora_storage = _generated_lora_storage_record(
        writer_asset["writer_state"]["template_lora_storage"],
        lora,
    )
    reference = (
        f"{AS_WRITER_CONFIG_SCHEMA}:{DYNAMIC_K_CHECKPOINT_KIND}:"
        f"m{writer_asset['method_macro']}:"
        f"{writer_asset['writer_state']['bytes']}bytes:rank{lora.rank}"
    )
    return {
        "schema_version": DYNAMIC_K_ADAPTER_SCHEMA,
        "kind": DYNAMIC_K_WRITER_KIND,
        "arm": "layer_matched_memory_program_compiler_" + video_condition,
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
            "generated_lora_storage": generated_lora_storage,
            **writer_asset,
        },
        "evaluation_authority": {
            "formal_status": "sealed" if profile is not None else "profile_required",
            "throughput_policy": (
                "highest_measured_batch_throughput_with_device_memory_headroom"
            ),
            "minimum_smoke_writer_model_batch_size": (
                DYNAMIC_K_GENERATION_BATCH_SIZE
                if profile is None
                else min(
                    profile.get(
                        "supported_writer_model_batch_sizes",
                        [DYNAMIC_K_GENERATION_BATCH_SIZE],
                    )
                )
            ),
            "online_smoke_evidence": None if profile is None else dict(profile),
        },
        "video_data": video_data,
        "video_condition": video_condition,
        "video_schedule": schedule,
        "task_video_mapping": mapping,
        "task_video_mapping_reference": (
            f"{config['authorities']['target_data_manifest']['path']}:"
            f"{len(mapping)}tasks:{video_condition}:K{evaluation_k}"
        ),
        "pairing_reference": (
            DYNAMIC_K_PAIRING_REFERENCE
            if evaluation_k == 1
            else DYNAMIC_K_VIDEO_SET_PAIRING_REFERENCE
        ),
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
        "information_wall": _evaluation_information_wall(
            config, evaluation_k, video_condition
        ),
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
    registered_episode_schema: str | None = None,
) -> dict[str, Any]:
    if not lora_reference:
        raise WriterModelError("dynamic-K episode lost its LoRA reference")
    schema = (
        dynamic_k_episode_schema(adapter)
        if registered_episode_schema is None
        else str(registered_episode_schema)
    )
    if not schema:
        raise WriterModelError("dynamic-K episode schema is empty")
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
    evaluation_k = int(schedule["videos_per_condition"])
    wall_evaluation_k = int(
        adapter.get("information_wall", {}).get("evaluation_k", evaluation_k)
    )
    if wall_evaluation_k != evaluation_k:
        raise WriterModelError("dynamic-K information wall K changed")
    selection = (str(adapter["video_condition"]), demo_count, mode, evaluation_k)
    reference, selected = paired_condition_demo_indices(
        seed, suite, task_id, init_state_id, *selection
    )
    frames_used = adapter["video_condition"] != "no_video"
    teacher_video_count = evaluation_k if frames_used else 0
    asset = adapter["writer_asset"]
    return {
        "schema_version": schema,
        "writer_method": DYNAMIC_K_WRITER_KIND,
        "method_arm": adapter["arm"],
        "condition": adapter["video_condition"],
        "evaluation_k": evaluation_k,
        "condition_video_offsets": [0, teacher_video_count],
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
        "teacher_video_frames_used": frames_used,
        "teacher_video_count": teacher_video_count,
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
        "teacher_demo_indices": list(selected),
        "teacher_reference_demo_indices": list(reference),
        "task_video_mapping_reference": adapter["task_video_mapping_reference"],
        "pairing_reference": adapter["pairing_reference"],
        "writer_generation_seed_schedule": (
            "numeric_seedsequence_one_shot_frame_order_v1"
            if evaluation_k == 1
            else "numeric_seedsequence_nested_video_set_frame_order_v1"
        ),
        "teacher_video_order_seeds": [
            frame_order_seed(seed, suite, task_id, demo_index)
            for demo_index in reference
        ],
    }


def validate_dynamic_k_episode_evidence(
    adapter: Mapping[str, Any],
    row: Any,
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
    registered_episode_schema: str | None = None,
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
            registered_episode_schema=registered_episode_schema,
        )
    except (KeyError, TypeError, ValueError, WriterModelError):
        return False
    observed = dict(row)
    observed.pop("writer_generation_seconds", None)
    return observed == expected and math.isfinite(seconds) and seconds >= 0
