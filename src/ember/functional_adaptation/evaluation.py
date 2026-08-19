"""Deployment authority for the fixed-decoder functional-code Writer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from safetensors import safe_open

from ember.eval_adapters import FUNCTIONAL_CODE_WRITER_KIND
from ember.expert_manifold.contract import load_task_expert_config
from ember.expert_manifold.meta_contract import meta_expert_rows
from ember.expert_manifold.video_schedule import (
    frame_order_seed,
    paired_condition_demo_indices,
    task_video_mapping,
    video_schedule_contract,
    video_selection_seed,
)
from ember.functional_adaptation.code_checkpoint import CHECKPOINT_SCHEMA, RUN_SCHEMA
from ember.functional_adaptation.decoder_training import (
    CONFIG_SCHEMA,
    authority_path,
    load_functional_adapter_config,
)
from ember.functional_adaptation.process_controls import (
    TEMPORAL_PROCESS_VIDEO_CONDITIONS,
)
from ember.lora import identity_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json
from ember.pi05_target_data import SUITE_ORDER
from ember.writer.errors import WriterModelError


REPO_ROOT = Path(__file__).resolve().parents[3]
FUNCTIONAL_CODE_WRITER_ADAPTER_SCHEMA = "ember_functional_code_writer_eval_adapter_v1"
FUNCTIONAL_CODE_WRITER_EPISODE_SCHEMA = "ember_functional_code_writer_episode_v1"
FUNCTIONAL_CODE_WRITER_CHECKPOINT_KIND = "functional_code_writer_macro_checkpoint"
FUNCTIONAL_CODE_WRITER_PAIRING_REFERENCE = "ember_functional_code_writer_pairing_v1"
FUNCTIONAL_CODE_WRITER_VIDEO_CONDITIONS = frozenset(
    {
        "correct",
        "same_task_other",
        "cross_suite_wrong",
        "wrong_task",
        "language_only",
        "video_only",
        *TEMPORAL_PROCESS_VIDEO_CONDITIONS,
    }
)
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}


def _macro(checkpoint: Path) -> int:
    try:
        value = int(checkpoint.name.removeprefix("macro_"))
    except ValueError as error:
        raise WriterModelError("functional-code checkpoint name changed") from error
    if value <= 0:
        raise WriterModelError("functional-code checkpoint macro is invalid")
    return value


def _writer_state_record(path: Path, lora: Any) -> dict[str, Any]:
    if not path.is_file():
        raise WriterModelError("functional-code Writer state is missing")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        names = tuple(handle.keys())
        shapes = {name: tuple(handle.get_slice(name).get_shape()) for name in names}
        dtypes = {name: str(handle.get_slice(name).get_dtype()) for name in names}
    template_keys = tuple(
        sorted(name for name in names if name.startswith("fixed_decoder.template_"))
    )
    expected = identity_lora_state(lora)
    lora_names = tuple(sorted(expected))
    if len(template_keys) != lora.state_tensor_count:
        raise WriterModelError("functional-code decoder template changed")
    dtype_by_name: dict[str, str] = {}
    dtype_tensor_counts: dict[str, int] = {}
    dtype_parameter_counts: dict[str, int] = {}
    for key, name in zip(template_keys, lora_names, strict=True):
        dtype = dtypes[key]
        if dtype not in _DTYPE_BYTES or shapes[key] != tuple(expected[name].shape):
            raise WriterModelError("functional-code LoRA template storage changed")
        count = math.prod(shapes[key])
        dtype_by_name[name] = dtype
        dtype_tensor_counts[dtype] = dtype_tensor_counts.get(dtype, 0) + 1
        dtype_parameter_counts[dtype] = dtype_parameter_counts.get(dtype, 0) + count
    storage = {
        "tensor_count": len(template_keys),
        "parameter_count": sum(dtype_parameter_counts.values()),
        "tensor_bytes": sum(
            _DTYPE_BYTES[dtype] * count
            for dtype, count in dtype_parameter_counts.items()
        ),
        "dtype_tensor_counts": dict(sorted(dtype_tensor_counts.items())),
        "dtype_parameter_counts": dict(sorted(dtype_parameter_counts.items())),
        "dtype_by_name": dict(sorted(dtype_by_name.items())),
    }
    if storage["parameter_count"] != lora.parameter_count:
        raise WriterModelError("functional-code generated LoRA size changed")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "state_tensor_count": len(names),
        "state_value_count": sum(math.prod(shape) for shape in shapes.values()),
        "generated_lora_storage": storage,
    }


def _writer_asset(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    checkpoint: Path,
    source: Mapping[str, Any],
    require_formal: bool,
    lora: Any,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    manifest_path = checkpoint / "manifest.json"
    manifest = read_json(manifest_path)
    macro = _macro(checkpoint)
    expected_files = {"writer.safetensors", "trainer.pt"}
    if (
        manifest.get("schema_version") != CHECKPOINT_SCHEMA
        or int(manifest.get("macro", -1)) != macro
        or int(manifest.get("metrics_rows", -1)) != macro
        or set(manifest.get("files", {})) != expected_files
        or manifest.get("content_hash_policy") != "disabled_by_owner"
    ):
        raise WriterModelError("functional-code deployment checkpoint changed")
    for name, expected_bytes in manifest["files"].items():
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size != int(expected_bytes):
            raise WriterModelError(f"functional-code checkpoint file changed: {name}")

    run_path = checkpoint.parent.parent / "run_contract.json"
    run = read_json(run_path)
    run_source = run.get("source", {})
    expected_source = {
        "run": str(Path(str(source["source_run"])).resolve()),
        "checkpoint": str(Path(str(source["checkpoint"])).resolve()),
        "model_path": str(Path(str(source["model_path"])).resolve()),
    }
    observed_source = {
        key: str(Path(str(run_source.get(key, ""))).resolve())
        for key in expected_source
    }
    decoder_profile = Path(str(run.get("decoder_profile", ""))).resolve()
    decoder_result = decoder_profile / "result.json"
    formal = config["code_inference"]["training"]["formal"]
    if (
        run.get("schema_version") != RUN_SCHEMA
        or observed_source != expected_source
        or Path(str(run.get("config", ""))).resolve() != config_path
        or not decoder_result.is_file()
        or read_json(decoder_result).get("surface") != "nonheld_meta"
        or int(run.get("tasks", {}).get("count", -1)) != 56
        or run.get("tasks", {}).get("role") != "meta_train"
        or int(run.get("trainable", {}).get("writer_parameter_count", -1)) <= 0
        or int(run.get("trainable", {}).get("fixed_decoder_trainable_parameters", -1))
        != 0
        or run.get("content_hash_policy") != "disabled_by_owner"
    ):
        raise WriterModelError("functional-code training authority changed")
    if require_formal and (
        run.get("mode") != "formal"
        or macro not in tuple(int(value) for value in formal["checkpoint_macros"])
        or int(manifest.get("world_size", -1)) != int(formal["world_size"])
    ):
        raise WriterModelError(
            "formal functional-code evaluation requires a declared formal checkpoint"
        )
    writer_state = _writer_state_record(checkpoint / "writer.safetensors", lora)
    return {
        "kind": FUNCTIONAL_CODE_WRITER_CHECKPOINT_KIND,
        "training_mode": str(run.get("mode")),
        "method_macro": macro,
        "checkpoint": str(checkpoint),
        "checkpoint_manifest": {
            "path": str(manifest_path.resolve()),
            "bytes": manifest_path.stat().st_size,
            "schema": CHECKPOINT_SCHEMA,
            "world_size": int(manifest["world_size"]),
        },
        "training_run_contract": {
            "path": str(run_path.resolve()),
            "bytes": run_path.stat().st_size,
            "schema": RUN_SCHEMA,
        },
        "decoder_profile": {
            "root": str(decoder_profile),
            "result": str(decoder_result),
            "result_bytes": decoder_result.stat().st_size,
        },
        "writer_parameter_count": int(run["trainable"]["writer_parameter_count"]),
        "deployment_trainable_parameter_count": 0,
        "writer_state": writer_state,
        "generated_lora_tensor_count": lora.state_tensor_count,
        "generated_lora_storage": writer_state["generated_lora_storage"],
    }


def _standard_task_rows(
    config: Mapping[str, Any], task_keys: Sequence[tuple[str, int]]
) -> tuple[dict[tuple[str, int], dict[str, Any]], bool]:
    normalized = tuple((str(suite), int(task_id)) for suite, task_id in task_keys)
    if not normalized or len(set(normalized)) != len(normalized):
        raise WriterModelError("functional-code evaluation tasks are invalid")
    meta = all(suite == "libero_90" for suite, _ in normalized)
    if meta:
        expert_config = load_task_expert_config(
            authority_path(config, "meta_experts", REPO_ROOT)
        )
        rows = {
            ("libero_90", int(row["task_id"])): {
                "global_task_id": int(row["task_id"]),
                "suite": "libero_90",
                "task_id": int(row["task_id"]),
                "split_role": str(row["split_role"]),
                "language": str(row["language"]),
                "relative_path": str(row["hdf5_filename"]),
                "bytes": int(row["hdf5_bytes"]),
            }
            for row in meta_expert_rows(expert_config)
        }
    else:
        if any(suite not in SUITE_ORDER for suite, _ in normalized):
            raise WriterModelError("functional-code evaluation mixes task surfaces")
        manifest = read_json(authority_path(config, "target_data_manifest", REPO_ROOT))
        rows = {
            (str(row["suite"]), int(row["task_id"])): {
                "global_task_id": int(row["global_task_id"]),
                "suite": str(row["suite"]),
                "task_id": int(row["task_id"]),
                "split_role": str(row["split_role"]),
                "language": str(row["language"]),
                "relative_path": str(row["hdf5"]["relative_path"]),
                "bytes": int(row["hdf5"]["bytes"]),
            }
            for row in manifest["tasks"]
        }
    if any(key not in rows for key in normalized):
        raise WriterModelError("functional-code evaluation task panel changed")
    return {key: rows[key] for key in normalized}, meta


def _meta_task_video_mapping(
    rows: Mapping[tuple[str, int], Mapping[str, Any]], condition: str
) -> list[dict[str, Any]]:
    if condition == "cross_suite_wrong":
        raise WriterModelError("cross-suite control is unavailable on LIBERO-90")
    by_role: dict[str, list[int]] = {}
    for row in rows.values():
        by_role.setdefault(str(row["split_role"]), []).append(int(row["task_id"]))
    for values in by_role.values():
        values.sort()
        if condition == "wrong_task" and len(values) < 2:
            raise WriterModelError("wrong-task control lacks a same-role peer")
    result = []
    for key, row in rows.items():
        peers = by_role[str(row["split_role"])]
        task_id = int(row["task_id"])
        video_task_id = (
            peers[(peers.index(task_id) + 1) % len(peers)]
            if condition == "wrong_task"
            else task_id
        )
        video = rows[("libero_90", video_task_id)]
        result.append(
            {
                "suite": key[0],
                "task_id": task_id,
                "language_global_task_id": task_id,
                "language_split_role": str(row["split_role"]),
                "video_suite": "libero_90",
                "video_task_id": video_task_id,
                "video_global_task_id": video_task_id,
                "video_split_role": str(video["split_role"]),
            }
        )
    return sorted(result, key=lambda row: int(row["task_id"]))


def _video_contract(
    *,
    config: Mapping[str, Any],
    video_data_root: Path,
    task_keys: Sequence[tuple[str, int]],
    video_condition: str,
    video_seed: int,
    video_sampling_mode: str,
    evaluation_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows, meta = _standard_task_rows(config, task_keys)
    if meta:
        mapping = _meta_task_video_mapping(rows, video_condition)
    else:
        if video_condition == "wrong_task":
            raise WriterModelError("target40 wrong control must be cross-suite")
        roles = {key: str(row["split_role"]) for key, row in rows.items()}
        mapping = list(task_video_mapping(tuple(rows), roles, video_condition))
    by_global = {int(row["global_task_id"]): row for row in rows.values()}
    root = video_data_root.resolve()
    records = []
    for global_id in sorted({int(row["video_global_task_id"]) for row in mapping}):
        row = by_global.get(global_id)
        if row is None:
            raise WriterModelError("functional-code video mapping escaped its panel")
        path = (root / str(row["relative_path"])).resolve()
        if (
            not path.is_relative_to(root)
            or not path.is_file()
            or path.stat().st_size != int(row["bytes"])
        ):
            raise WriterModelError("functional-code video HDF5 changed")
        records.append(dict(row))
    schedule, _ = video_schedule_contract(
        seed=video_seed,
        demo_count=50,
        sampling_mode=video_sampling_mode,
    )
    schedule.update(
        {
            "algorithm": (
                "numeric_seeded_task_permutation_nested_k_video_set_per_state_block"
                if evaluation_k > 1
                else schedule["algorithm"]
            ),
            "videos_per_condition": evaluation_k,
            "nested_k1_prefix": evaluation_k > 1,
            "within_condition_unique": True,
            "frame_stride": int(config["code_inference"]["training"]["frame_stride"]),
            "include_final_frame": True,
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


def _writer_input(condition: str, evaluation_k: int) -> str:
    if condition == "language_only":
        return "exact task language only"
    if condition == "video_only":
        return f"{evaluation_k} action-hidden ordered teacher video(s) only"
    return f"exact task language plus {evaluation_k} action-hidden ordered teacher video(s)"


def inspect_functional_code_writer_evaluation(
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
    config_path = config_path.resolve()
    config = load_functional_adapter_config(config_path, REPO_ROOT)
    if (
        video_condition not in FUNCTIONAL_CODE_WRITER_VIDEO_CONDITIONS
        or not 1
        <= evaluation_k
        <= int(config["code_inference"]["training"]["dynamic_k_max"])
        or evaluation_k > 1
        and video_sampling_mode != "without_replacement"
    ):
        raise WriterModelError("functional-code evaluation condition is unsupported")
    evaluation = config["code_inference"]["evaluation"]
    profile = evaluation.get("online_smoke_evidence")
    if require_formal and (
        evaluation.get("formal_status") != "sealed" or not isinstance(profile, Mapping)
    ):
        raise WriterModelError(
            "functional-code formal evaluation requires a live profile"
        )
    lora_path = authority_path(config, "lora_contract", REPO_ROOT)
    lora = load_pi05_lora_contract(lora_path)
    asset = _writer_asset(
        config_path=config_path,
        config=config,
        checkpoint=checkpoint,
        source=source,
        require_formal=require_formal,
        lora=lora,
    )
    mapping, video_data, schedule = _video_contract(
        config=config,
        video_data_root=video_data_root,
        task_keys=task_keys,
        video_condition=video_condition,
        video_seed=video_seed,
        video_sampling_mode=video_sampling_mode,
        evaluation_k=evaluation_k,
    )
    reference = (
        f"{CONFIG_SCHEMA}:{FUNCTIONAL_CODE_WRITER_CHECKPOINT_KIND}:"
        f"m{asset['method_macro']}:{asset['writer_state']['bytes']}bytes:rank{lora.rank}"
    )
    return {
        "schema_version": FUNCTIONAL_CODE_WRITER_ADAPTER_SCHEMA,
        "kind": FUNCTIONAL_CODE_WRITER_KIND,
        "arm": f"functional_code_writer_{video_condition}",
        "execution_backend": "online_frozen_functional_code_writer_then_episode_lora_cache",
        "config": {
            "path": str(config_path),
            "bytes": config_path.stat().st_size,
            "schema": CONFIG_SCHEMA,
        },
        "writer_asset": {"reference": reference, **asset},
        "evaluation_authority": {
            "formal_status": str(evaluation["formal_status"]),
            "throughput_policy": str(evaluation["throughput_policy"]),
            "minimum_smoke_writer_model_batch_size": int(
                evaluation["minimum_smoke_writer_model_batch_size"]
            ),
            "online_smoke_evidence": None if profile is None else dict(profile),
        },
        "video_data": video_data,
        "video_condition": video_condition,
        "video_schedule": schedule,
        "task_video_mapping": mapping,
        "task_video_mapping_reference": (
            f"{len(mapping)}tasks:{video_condition}:K{evaluation_k}"
        ),
        "pairing_reference": FUNCTIONAL_CODE_WRITER_PAIRING_REFERENCE,
        "lora_contract": {
            "reference": (
                f"{lora_path.relative_to(REPO_ROOT)}:"
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
            "writer_input": _writer_input(video_condition, evaluation_k),
            "language_reads": video_condition != "video_only",
            "video_frames_read": video_condition != "language_only",
            "language_only_lora_path": video_condition == "language_only",
            "video_only_lora_path": video_condition == "video_only",
            "teacher_action_reads": 0,
            "teacher_state_reads": 0,
            "reward_reads": 0,
            "terminal_reads": 0,
            "deployment_expert_bank_read": False,
            "decoder_frozen": True,
            "evaluation_k": evaluation_k,
            "dynamic_k_training_range": [
                1,
                int(config["code_inference"]["training"]["dynamic_k_max"]),
            ],
            "frame_stride": int(config["code_inference"]["training"]["frame_stride"]),
        },
        "content_hash_policy": "disabled_by_owner",
    }


def functional_code_writer_episode_schema(adapter: Mapping[str, Any]) -> str:
    if (
        adapter.get("kind") != FUNCTIONAL_CODE_WRITER_KIND
        or adapter.get("schema_version") != FUNCTIONAL_CODE_WRITER_ADAPTER_SCHEMA
        or adapter.get("config", {}).get("schema") != CONFIG_SCHEMA
    ):
        raise WriterModelError("invalid functional-code deployment adapter")
    return FUNCTIONAL_CODE_WRITER_EPISODE_SCHEMA


def expected_functional_code_writer_episode(
    adapter: Mapping[str, Any],
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
    lora_reference: str,
) -> dict[str, Any]:
    if not lora_reference:
        raise WriterModelError("functional-code episode lost its LoRA reference")
    matches = [
        row
        for row in adapter["task_video_mapping"]
        if row["suite"] == suite and int(row["task_id"]) == task_id
    ]
    if len(matches) != 1:
        raise WriterModelError("episode task is outside functional-code mapping")
    mapping = matches[0]
    schedule = adapter["video_schedule"]
    seed = int(schedule["seed"])
    mode = str(schedule["sampling_mode"])
    demo_count = int(schedule["demo_count"])
    evaluation_k = int(schedule["videos_per_condition"])
    condition = str(adapter["video_condition"])
    reference, selected = paired_condition_demo_indices(
        seed,
        suite,
        task_id,
        init_state_id,
        condition,
        demo_count,
        mode,
        evaluation_k,
    )
    frames_used = condition != "language_only"
    language_used = condition != "video_only"
    asset = adapter["writer_asset"]
    return {
        "schema_version": functional_code_writer_episode_schema(adapter),
        "writer_method": FUNCTIONAL_CODE_WRITER_KIND,
        "method_arm": adapter["arm"],
        "condition": condition,
        "inference_branch": (
            "language"
            if not frames_used
            else "video" if not language_used else "combined"
        ),
        "evaluation_k": evaluation_k,
        "condition_video_offsets": [0, evaluation_k if frames_used else 0],
        "writer_asset_reference": asset["reference"],
        "writer_checkpoint_kind": asset["kind"],
        "writer_method_macro": int(asset["method_macro"]),
        "writer_parameter_count": int(asset["writer_parameter_count"]),
        "writer_deployment_trainable_parameter_count": 0,
        "generated_lora_tensor_count": int(asset["generated_lora_tensor_count"]),
        "lora_contract_reference": adapter["lora_contract"]["reference"],
        "lora_reference": lora_reference,
        "language_global_task_id": int(mapping["language_global_task_id"]),
        "language_used": language_used,
        "teacher_video_kind": condition,
        "teacher_video_frames_used": frames_used,
        "teacher_video_count": evaluation_k if frames_used else 0,
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
        "teacher_video_order_seeds": [
            frame_order_seed(seed, suite, task_id, demo_index)
            for demo_index in reference
        ],
    }


def validate_functional_code_writer_episode(
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
        expected = expected_functional_code_writer_episode(
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
