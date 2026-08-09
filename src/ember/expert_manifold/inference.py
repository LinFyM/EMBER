"""Formal authority for the canonical v6-prior Expert-Manifold Writer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from safetensors import safe_open

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior import (
    V6_WRITER_PARAMETER_TENSOR_COUNT,
    V6_WRITER_STATE_TENSOR_COUNT,
)
from ember.expert_manifold.v6_prior_checkpoint import (
    V6_PRIOR_CHECKPOINT_SCHEMA,
    inspect_v6_prior_checkpoint,
)
from ember.expert_manifold.v6_prior_contract import (
    REPO_ROOT,
    V6_PRIOR_CANONICAL_CONFIG,
    V6_PRIOR_CONFIG_SCHEMA,
    V6_PRIOR_RUN_SCHEMA,
    authority_path,
    git_commit_in_active_authority_lineage,
    load_v6_prior_config,
)
from ember.expert_manifold.v6_prior_run_contract import cursor_contract
from ember.expert_manifold.video_schedule import (
    SAME_TASK_OTHER_OFFSET,
    VIDEO_CONDITIONS,
    condition_demo_index,
    frame_order_seed,
    reference_demo_index,
    task_video_mapping,
    video_schedule_contract,
    video_selection_seed,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json
from ember.writer.architecture import V6_WRITER_PARAMETER_COUNT


EXPERT_MANIFOLD_WRITER_KIND = "expert_manifold_writer"
EXPERT_MANIFOLD_ADAPTER_SCHEMA = (
    "ember_pi05_v6_condition_program_residual_eval_adapter_v8"
)
EXPERT_MANIFOLD_EPISODE_SCHEMA = (
    "ember_pi05_v6_condition_program_residual_episode_v8"
)


def _target_rows(config: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    manifest = read_json(authority_path(config, "target_data_manifest"))
    rows = {int(row["global_task_id"]): dict(row) for row in manifest["tasks"]}
    if len(rows) != 40:
        raise ExpertManifoldError("v6-prior target manifest changed")
    return rows


def _video_data(
    *,
    config: Mapping[str, Any],
    root: Path,
    global_task_ids: Sequence[int],
) -> dict[str, Any]:
    root = root.resolve()
    by_id = _target_rows(config)
    records = []
    allowed = set(config["information_wall"]["writer_video_split_roles"])
    for global_task_id in sorted(set(int(value) for value in global_task_ids)):
        row = by_id.get(global_task_id)
        if row is None or row["split_role"] not in allowed:
            raise ExpertManifoldError("evaluation video crossed its allowed split")
        hdf5 = row["hdf5"]
        path = root / str(hdf5["relative_path"])
        if not path.is_file() or path.stat().st_size != int(hdf5["bytes"]):
            raise ExpertManifoldError("evaluation video HDF5 changed")
        records.append(
            {
                "global_task_id": global_task_id,
                "suite": row["suite"],
                "task_id": int(row["task_id"]),
                "split_role": row["split_role"],
                "language": row["language"],
                "relative_path": hdf5["relative_path"],
                "bytes": int(hdf5["bytes"]),
            }
        )
    return {
        "root": str(root),
        "tasks": records,
        "teacher_action_reads": 0,
        "teacher_state_reads": 0,
        "reward_reads": 0,
        "terminal_reads": 0,
    }


def _evaluation_video_contract(
    config: Mapping[str, Any],
    *,
    task_keys: Sequence[tuple[str, int]],
    video_condition: str,
    video_data_root: Path,
    video_seed: int,
    video_sampling_mode: str,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any], dict[str, Any], str]:
    rows = _target_rows(config)
    by_key = {(row["suite"], int(row["task_id"])): row for row in rows.values()}
    normalized = tuple((str(suite), int(task_id)) for suite, task_id in task_keys)
    if (
        not normalized
        or len(set(normalized)) != len(normalized)
        or any(key not in by_key for key in normalized)
    ):
        raise ExpertManifoldError("v6-prior evaluation task panel changed")
    roles = {key: str(by_key[key]["split_role"]) for key in normalized}
    mapping = task_video_mapping(
        normalized,
        roles,
        "correct" if video_condition == "no_video" else video_condition,
    )
    needed = {int(row["language_global_task_id"]) for row in mapping} | {
        int(row["video_global_task_id"]) for row in mapping
    }
    video_data = _video_data(
        config=config,
        root=video_data_root,
        global_task_ids=tuple(needed),
    )
    schedule, pairing = video_schedule_contract(
        seed=video_seed,
        demo_count=50,
        sampling_mode=video_sampling_mode,
    )
    return mapping, video_data, schedule, pairing


def _template_lora_names(config: Mapping[str, Any]) -> tuple[str, ...]:
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    result = []
    for target in sorted(lora.targets, key=lambda value: value.name):
        result.extend(
            (
                f"{target.name}.lora_A.default.weight",
                f"{target.name}.lora_B.default.weight",
            )
        )
    return tuple(result)


def _writer_state_record(
    path: Path, *, template_lora_names: Sequence[str]
) -> dict[str, Any]:
    if not path.is_file():
        raise ExpertManifoldError("v6-prior Writer state is missing")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        names = tuple(handle.keys())
        value_count = sum(
            math.prod(handle.get_slice(name).get_shape()) for name in names
        )
        template_names = tuple(name for name in names if name.startswith("template_"))
        if len(template_names) != len(template_lora_names):
            raise ExpertManifoldError("v6-prior LoRA template topology changed")
        dtype_itemsize = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}
        template_dtype_tensor_counts: dict[str, int] = {}
        template_dtype_parameter_counts: dict[str, int] = {}
        template_dtype_by_name: dict[str, str] = {}
        for buffer_name, lora_name in zip(
            template_names, template_lora_names, strict=True
        ):
            value = handle.get_slice(buffer_name)
            dtype = str(value.get_dtype())
            if dtype not in dtype_itemsize:
                raise ExpertManifoldError("v6-prior LoRA template dtype changed")
            count = math.prod(value.get_shape())
            template_dtype_tensor_counts[dtype] = (
                template_dtype_tensor_counts.get(dtype, 0) + 1
            )
            template_dtype_parameter_counts[dtype] = (
                template_dtype_parameter_counts.get(dtype, 0) + count
            )
            template_dtype_by_name[lora_name] = dtype
    if len(names) != V6_WRITER_STATE_TENSOR_COUNT:
        raise ExpertManifoldError("v6-prior Writer state tensor count changed")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "state_tensor_count": len(names),
        "state_value_count": value_count,
        "template_lora_storage": {
            "tensor_count": len(template_names),
            "parameter_count": sum(template_dtype_parameter_counts.values()),
            "tensor_bytes": sum(
                dtype_itemsize[dtype] * count
                for dtype, count in template_dtype_parameter_counts.items()
            ),
            "dtype_tensor_counts": dict(sorted(template_dtype_tensor_counts.items())),
            "dtype_parameter_counts": dict(
                sorted(template_dtype_parameter_counts.items())
            ),
            "dtype_by_name": dict(sorted(template_dtype_by_name.items())),
        },
    }


def _source_shape(source: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return {
            "optimizer_step": int(source["optimizer_step"]),
            "source_training_commit": str(source["source_training_commit"]),
            "model_files": [
                {"path": str(row["path"]), "bytes": int(row["bytes"])}
                for row in source["model_files"]
            ],
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ExpertManifoldError("v6-prior source contract changed") from error


def _program_residual_shape(config: Mapping[str, Any]) -> list[int]:
    residual = config["program_residual"]
    return [
        int(residual["feature_width"]),
        int(residual["program_slots"]),
        int(residual["program_width"]),
    ]


def _historical_writer_asset(
    config: Mapping[str, Any],
    checkpoint: Path,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    writer_path = checkpoint / "writer.safetensors"
    manifest_path = checkpoint / "checkpoint_manifest.json"
    run_root = checkpoint.parent.parent
    run_contract_path = run_root / "run_contract.json"
    if not all(
        path.is_file() for path in (writer_path, manifest_path, run_contract_path)
    ):
        raise ExpertManifoldError("historical v6 warm-start asset is incomplete")
    manifest = read_json(manifest_path)
    run_contract = read_json(run_contract_path)
    if not isinstance(manifest, Mapping) or not isinstance(run_contract, Mapping):
        raise ExpertManifoldError("historical v6 warm-start manifest changed")
    files = manifest.get("files", {})
    consumed = manifest.get("consumed", {})
    if not isinstance(files, Mapping) or not isinstance(consumed, Mapping):
        raise ExpertManifoldError("historical v6 warm-start manifest changed")
    declared = files.get("writer.safetensors", {})
    try:
        valid = (
            manifest.get("schema_version")
            == "ember_pi05_language_axial_writer_checkpoint_v6"
            and int(consumed.get("next_step", -1)) == 400
            and isinstance(declared, Mapping)
            and writer_path.stat().st_size == int(declared.get("bytes", -1))
            and _source_shape(run_contract.get("source", {}))
            == _source_shape(source)
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ExpertManifoldError("historical v6 warm-start asset changed")
    return {
        "kind": "historical_v6_macro400_load_only",
        "training_mode": "historical_v6_task_complete",
        "source_macro": 400,
        "method_macro": 0,
        "checkpoint": str(checkpoint),
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "schema": manifest["schema_version"],
        },
        "writer_state": _writer_state_record(
            writer_path, template_lora_names=_template_lora_names(config)
        ),
        "residual_state": {
            "kind": "fresh_elementwise_zero",
            "path": None,
            "bytes": 0,
            "tensor_count": 0,
            "dtype": "torch.float32",
            "shape": _program_residual_shape(config),
            "value_count": int(config["program_residual"]["value_count"]),
        },
    }


def _expected_residual_ownership(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "historical_v6_base": {
            "state_tensor_count": V6_WRITER_STATE_TENSOR_COUNT,
            "parameter_tensor_count": V6_WRITER_PARAMETER_TENSOR_COUNT,
            "parameter_count": V6_WRITER_PARAMETER_COUNT,
            "trainable_parameter_count": 0,
            "checkpoint_owned": False,
            "deployment_owned": True,
        },
        "fixed_projection": {
            "shape": [
                int(config["condition_feature"]["feature_width"]),
                int(config["condition_feature"]["descriptor_width"]),
            ],
            "dtype": "torch.float32",
            "trainable": False,
            "persistent": False,
            "checkpoint_owned": False,
        },
        "program_residual_memory": {
            "shape": _program_residual_shape(config),
            "dtype": "torch.float32",
            "value_count": int(config["program_residual"]["value_count"]),
            "trainable": False,
            "manual_update": True,
            "checkpoint_owned": True,
            "deployment_owned": True,
        },
        "source_policy_trainable_parameter_count": 0,
        "optimizer": "not_instantiated",
        "scheduler": "not_instantiated",
        "scaler": "not_instantiated",
    }


def _residual_contract_matches(
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    historical: Mapping[str, Any],
    configured_writer: Path,
) -> bool:
    contract_config = contract.get("config", {})
    topology = contract.get("rank_topology")
    try:
        config_bytes = int(contract_config.get("bytes", -1))
    except (AttributeError, TypeError, ValueError):
        return False
    if (
        not isinstance(contract_config, Mapping)
        or set(contract_config) != {"path", "schema", "bytes"}
        or Path(str(contract_config.get("path", ""))).name
        != V6_PRIOR_CANONICAL_CONFIG.name
        or contract_config.get("schema") != V6_PRIOR_CONFIG_SCHEMA
        or config_bytes <= 0
        or not isinstance(topology, list)
        or len(topology) != 6
    ):
        return False
    expected_keys = {
        "run_schema",
        "mode",
        "git_commit",
        "config",
        "source",
        "initialization",
        "condition_feature",
        "program_residual",
        "update",
        "ownership",
        "world_size",
        "rank_topology",
        "content_hash_policy",
    }
    expected_initialization = {
        "mode": "strict_historical_v6_macro400_all_frozen",
        "checkpoint": str(configured_writer),
        "writer_state_tensor_count": V6_WRITER_STATE_TENSOR_COUNT,
        "writer_state_value_count": int(
            historical["writer_state"]["state_value_count"]
        ),
        "residual_memory": "fresh_zero_then_memory_only_exact_resume",
    }
    fixed = {
        "run_schema": V6_PRIOR_RUN_SCHEMA,
        "mode": "formal",
        "source": dict(source),
        "initialization": expected_initialization,
        "condition_feature": config["condition_feature"],
        "program_residual": config["program_residual"],
        "update": config["update"],
        "ownership": _expected_residual_ownership(config),
        "world_size": 6,
        "content_hash_policy": "disabled_by_owner",
    }
    return (
        set(contract) == expected_keys
        and {name: contract.get(name) for name in fixed} == fixed
        and isinstance(contract.get("git_commit"), str)
        and git_commit_in_active_authority_lineage(contract["git_commit"])
    )


def _residual_inspection(
    inspection: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[int, Mapping[str, Any]]:
    try:
        macro = int(inspection.get("next_macro", -1))
        memory = inspection.get("program_memory", {})
        if not isinstance(memory, Mapping):
            raise TypeError("memory metadata")
        identity = {
            "checkpoint_schema": inspection.get("checkpoint_schema"),
            "world_size": int(inspection.get("world_size", -1)),
            "metrics_rows": int(inspection.get("metrics_rows", -1)),
            "content_hash_policy": inspection.get("content_hash_policy"),
            "payload_value_validation": inspection.get(
                "payload_value_validation"
            ),
        }
        expected_identity = {
            "checkpoint_schema": V6_PRIOR_CHECKPOINT_SCHEMA,
            "world_size": 6,
            "metrics_rows": macro,
            "content_hash_policy": "disabled_by_owner",
            "payload_value_validation": "deployment_metadata_only",
        }
        expected_memory = {
            "tensor_count": 1,
            "dtype": "torch.float32",
            "shape": _program_residual_shape(config),
            "value_count": int(config["program_residual"]["value_count"]),
            "finite": None,
        }
        observed_memory = {
            name: memory.get(name) for name in expected_memory
        }
    except (TypeError, ValueError):
        raise ExpertManifoldError("v6-prior residual checkpoint changed") from None
    if (
        macro <= 0
        or not isinstance(memory, Mapping)
        or identity != expected_identity
        or inspection.get("cursor_contract") != cursor_contract(config, macro)
        or observed_memory != expected_memory
    ):
        raise ExpertManifoldError("v6-prior residual checkpoint changed")
    return macro, memory


def _trained_writer_asset(
    config: Mapping[str, Any],
    checkpoint: Path,
    source: Mapping[str, Any],
    *,
    require_formal: bool,
) -> dict[str, Any]:
    del require_formal
    manifest_path = checkpoint / "manifest.json"
    if not manifest_path.is_file():
        raise ExpertManifoldError("v6-prior residual checkpoint is incomplete")
    inspection = inspect_v6_prior_checkpoint(
        checkpoint, validate_payload_values=False
    )
    contract = inspection.get("checkpoint_contract", {})
    if not isinstance(contract, Mapping):
        raise ExpertManifoldError(
            "v6-prior residual checkpoint contract changed"
        )
    configured_dir = (
        REPO_ROOT / str(config["initialization"]["checkpoint"])
    ).resolve()
    configured_writer = configured_dir / "writer.safetensors"
    historical = _historical_writer_asset(config, configured_dir, source)
    if not _residual_contract_matches(
        contract, config, source, historical, configured_writer
    ):
        raise ExpertManifoldError("v6-prior residual checkpoint changed")
    macro, memory = _residual_inspection(inspection, config)
    residual_file = checkpoint / str(memory["file"])
    return {
        "kind": "v6_condition_program_residual_checkpoint",
        "training_mode": "formal",
        "source_macro": 400,
        "method_macro": macro,
        "checkpoint": str(checkpoint),
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "schema": V6_PRIOR_CHECKPOINT_SCHEMA,
        },
        "writer_state": historical["writer_state"],
        "residual_state": {
            "kind": "memory_only_checkpoint",
            "path": str(residual_file),
            "bytes": residual_file.stat().st_size,
            "tensor_count": 1,
            "key": memory["key"],
            "dtype": memory["dtype"],
            "shape": memory["shape"],
            "value_count": memory["value_count"],
        },
    }


def inspect_v6_prior_writer_asset(
    config: Mapping[str, Any],
    checkpoint: Path,
    source: Mapping[str, Any],
    *,
    require_formal: bool,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    configured = (
        REPO_ROOT / str(config["initialization"]["checkpoint"])
    ).resolve()
    if checkpoint == configured:
        return _historical_writer_asset(config, checkpoint, source)
    return _trained_writer_asset(
        config,
        checkpoint,
        source,
        require_formal=require_formal,
    )


def inspect_expert_manifold_writer_evaluation(
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
    config_path = config_path.resolve()
    config = load_v6_prior_config(config_path)
    if video_condition not in VIDEO_CONDITIONS:
        raise ExpertManifoldError("unsupported Expert-Manifold video condition")
    status = str(config["evaluation"]["formal_status"])
    if require_formal and status != "sealed_from_live_residual_deployment_profile":
        raise ExpertManifoldError(
            "formal residual evaluation requires its live deployment profile"
        )
    writer_asset = inspect_v6_prior_writer_asset(
        config,
        checkpoint,
        source,
        require_formal=require_formal,
    )
    mapping, video_data, schedule, pairing = _evaluation_video_contract(
        config,
        task_keys=task_keys,
        video_condition=video_condition,
        video_data_root=video_data_root,
        video_seed=video_seed,
        video_sampling_mode=video_sampling_mode,
    )
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    reference = (
        f"{V6_PRIOR_CONFIG_SCHEMA}:{writer_asset['kind']}:"
        f"m{int(writer_asset['method_macro'])}:"
        f"base{int(writer_asset['writer_state']['bytes'])}bytes:"
        f"residual{int(writer_asset['residual_state']['bytes'])}bytes:rank16"
    )
    return {
        "schema_version": EXPERT_MANIFOLD_ADAPTER_SCHEMA,
        "kind": EXPERT_MANIFOLD_WRITER_KIND,
        "arm": f"expert_manifold_v6_condition_residual_{video_condition}",
        "execution_backend": (
            "online_frozen_v6_condition_program_residual_then_episode_lora_cache"
        ),
        "config": {
            "path": str(config_path),
            "bytes": config_path.stat().st_size,
            "schema": config["schema_version"],
        },
        "writer_asset": {
            "reference": reference,
            "architecture": config["writer"]["architecture"],
            "writer_parameter_count": V6_WRITER_PARAMETER_COUNT,
            "deployment_trainable_parameter_count": 0,
            "program_residual_value_count": int(
                config["program_residual"]["value_count"]
            ),
            "generated_lora_tensor_count": lora.state_tensor_count,
            **writer_asset,
        },
        "evaluation_authority": {
            "formal_status": status,
            "throughput_policy": config["evaluation"]["throughput_policy"],
            "minimum_smoke_writer_model_batch_size": config["evaluation"][
                "minimum_smoke_writer_model_batch_size"
            ],
            "online_smoke_evidence": config["evaluation"].get(
                "online_smoke_evidence",
                config["evaluation"].get("inherited_online_smoke_evidence"),
            ),
        },
        "video_data": video_data,
        "video_condition": video_condition,
        "video_schedule": schedule,
        "task_video_mapping": list(mapping),
        "task_video_mapping_reference": (
            f"{config['authorities']['target_data_manifest']['path']}:"
            f"{len(mapping)}tasks:{video_condition}"
        ),
        "pairing_reference": pairing,
        "lora_contract": {
            "reference": (
                f"{config['authorities']['lora_contract']['path']}:"
                f"{lora.state_tensor_count}tensors:{lora.parameter_count}parameters"
            ),
            "rank": lora.rank,
            "target_count": len(lora.targets),
        },
        "source": {
            "source_run": str(Path(str(source["source_run"])).resolve()),
            "checkpoint": str(Path(str(source["checkpoint"])).resolve()),
            "model_path": str(Path(str(source["model_path"])).resolve()),
        },
        "information_wall": {
            "writer_input": "exact task language plus one action-hidden teacher video",
            "video_is_only_dynamic_value": True,
            "no_video_counterfactual": video_condition == "no_video",
            "teacher_action_reads": 0,
            "teacher_state_reads": 0,
            "reward_reads": 0,
            "terminal_reads": 0,
            "language_only_lora_path": False,
            "deployment_expert_bank_read": False,
        },
        "content_hash_policy": "disabled_by_owner",
    }


def expected_expert_manifold_episode_evidence(
    adapter: Mapping[str, Any],
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
    lora_reference: str,
) -> dict[str, Any]:
    if (
        adapter.get("schema_version") != EXPERT_MANIFOLD_ADAPTER_SCHEMA
        or adapter.get("kind") != EXPERT_MANIFOLD_WRITER_KIND
        or not lora_reference
    ):
        raise ExpertManifoldError("invalid Expert-Manifold episode adapter")
    matches = [
        row
        for row in adapter["task_video_mapping"]
        if row["suite"] == suite and int(row["task_id"]) == task_id
    ]
    if len(matches) != 1:
        raise ExpertManifoldError("episode task is outside Expert-Manifold mapping")
    mapping = matches[0]
    schedule = adapter["video_schedule"]
    seed = int(schedule["seed"])
    mode = str(schedule["sampling_mode"])
    demo_count = int(schedule["demo_count"])
    reference = reference_demo_index(
        seed,
        suite,
        task_id,
        init_state_id,
        demo_count=demo_count,
        sampling_mode=mode,
    )
    selected = condition_demo_index(
        seed,
        suite,
        task_id,
        init_state_id,
        condition=str(adapter["video_condition"]),
        demo_count=demo_count,
        sampling_mode=mode,
    )
    result = {
        "schema_version": EXPERT_MANIFOLD_EPISODE_SCHEMA,
        "writer_method": EXPERT_MANIFOLD_WRITER_KIND,
        "method_arm": adapter["arm"],
        "condition": adapter["video_condition"],
        "writer_asset_reference": adapter["writer_asset"]["reference"],
        "writer_checkpoint_kind": adapter["writer_asset"]["kind"],
        "writer_method_macro": int(adapter["writer_asset"]["method_macro"]),
        "writer_parameter_count": int(
            adapter["writer_asset"]["writer_parameter_count"]
        ),
        "writer_deployment_trainable_parameter_count": 0,
        "writer_program_residual_value_count": int(
            adapter["writer_asset"]["program_residual_value_count"]
        ),
        "generated_lora_tensor_count": int(
            adapter["writer_asset"]["generated_lora_tensor_count"]
        ),
        "lora_contract_reference": adapter["lora_contract"]["reference"],
        "lora_reference": lora_reference,
        "language_global_task_id": int(mapping["language_global_task_id"]),
        "teacher_video_kind": adapter["video_condition"],
        "teacher_video_frames_used": adapter["video_condition"] != "no_video",
        "teacher_video_count": int(adapter["video_condition"] != "no_video"),
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
        "teacher_reference_demo_indices": [reference],
        "task_video_mapping_reference": adapter["task_video_mapping_reference"],
        "pairing_reference": adapter["pairing_reference"],
        "writer_generation_seed_schedule": (
            "numeric_seedsequence_one_shot_frame_order_v1"
        ),
        "teacher_video_order_seeds": [
            frame_order_seed(seed, suite, task_id, reference)
        ],
    }
    if adapter["video_condition"] == "same_task_other":
        result["teacher_demo_offset"] = SAME_TASK_OTHER_OFFSET
    return result


def validate_expert_manifold_episode_evidence(
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
        expected = expected_expert_manifold_episode_evidence(
            adapter,
            suite=suite,
            task_id=task_id,
            init_state_id=init_state_id,
            lora_reference=str(row.get("lora_reference", "")),
        )
    except (ExpertManifoldError, KeyError, TypeError, ValueError):
        return False
    observed = dict(row)
    observed.pop("writer_generation_seconds", None)
    return observed == expected and math.isfinite(seconds) and seconds >= 0
