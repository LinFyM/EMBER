"""Formal authority for the canonical v6-prior Expert-Manifold Writer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from safetensors import safe_open

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior import (
    V6_PRIOR_FROZEN_PARAMETER_COUNT,
    V6_PRIOR_TRAINABLE_PARAMETER_COUNT,
    V6_WRITER_STATE_TENSOR_COUNT,
)
from ember.expert_manifold.v6_prior_checkpoint import (
    V6_PRIOR_CHECKPOINT_SCHEMA,
)
from ember.expert_manifold.v6_prior_contract import (
    REPO_ROOT,
    V6_PRIOR_CONFIG_SCHEMA,
    authority_path,
    load_v6_prior_config,
)
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
EXPERT_MANIFOLD_ADAPTER_SCHEMA = "ember_pi05_v6_prior_eval_adapter_v5"
EXPERT_MANIFOLD_EPISODE_SCHEMA = "ember_pi05_v6_prior_episode_v5"


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


def _writer_state_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExpertManifoldError("v6-prior Writer state is missing")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        names = tuple(handle.keys())
        value_count = sum(
            math.prod(handle.get_slice(name).get_shape()) for name in names
        )
    if len(names) != V6_WRITER_STATE_TENSOR_COUNT:
        raise ExpertManifoldError("v6-prior Writer state tensor count changed")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "state_tensor_count": len(names),
        "state_value_count": value_count,
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
        "writer_state": _writer_state_record(writer_path),
    }


def _trained_writer_asset(
    config: Mapping[str, Any],
    checkpoint: Path,
    source: Mapping[str, Any],
    *,
    require_formal: bool,
) -> dict[str, Any]:
    manifest_path = checkpoint / "manifest.json"
    writer_path = checkpoint / "writer.safetensors"
    if not manifest_path.is_file() or not writer_path.is_file():
        raise ExpertManifoldError("v6-prior trained Writer asset is incomplete")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ExpertManifoldError("v6-prior trained Writer manifest changed")
    contract = manifest.get("checkpoint_contract", {})
    files = manifest.get("files", {})
    if not isinstance(contract, Mapping) or not isinstance(files, Mapping):
        raise ExpertManifoldError("v6-prior trained Writer manifest changed")
    initialization = contract.get("initialization", {})
    contract_config = contract.get("config", {})
    ownership = contract.get("ownership", {})
    if not all(
        isinstance(value, Mapping)
        for value in (initialization, contract_config, ownership)
    ):
        raise ExpertManifoldError("v6-prior trained Writer manifest changed")
    configured_warm_start = (
        REPO_ROOT / str(config["initialization"]["checkpoint"])
    ).resolve() / "writer.safetensors"
    try:
        macro = int(manifest.get("next_macro", -1))
        mode = str(contract.get("mode", ""))
        valid = (
            manifest.get("schema_version") == V6_PRIOR_CHECKPOINT_SCHEMA
            and checkpoint.parent.name == "checkpoints"
            and checkpoint.name == f"macro_{macro:08d}"
            and macro > 0
            and int(manifest.get("world_size", -1)) == 6
            and int(manifest.get("metrics_rows", -1)) == macro
            and manifest.get("content_hash_policy") == "disabled_by_owner"
            and writer_path.stat().st_size
            == int(files.get("writer.safetensors", -1))
            and contract.get("run_schema")
            == "ember_pi05_v6_prior_writer_launch_v1"
            and mode in {"profile", "formal"}
            and contract.get("source") == dict(source)
            and contract.get("objective") == config["objective"]
            and contract_config.get("schema") == V6_PRIOR_CONFIG_SCHEMA
            and Path(str(initialization.get("checkpoint", ""))).resolve()
            == configured_warm_start
            and int(ownership.get("frozen_parameter_count", -1))
            == V6_PRIOR_FROZEN_PARAMETER_COUNT
            and int(ownership.get("trainable_parameter_count", -1))
            == V6_PRIOR_TRAINABLE_PARAMETER_COUNT
        )
    except (TypeError, ValueError):
        valid = False
    if require_formal:
        valid = valid and mode == "formal"
    if not valid:
        raise ExpertManifoldError("v6-prior trained Writer checkpoint changed")
    return {
        "kind": "v6_prior_trained_checkpoint",
        "training_mode": mode,
        "source_macro": 400,
        "method_macro": macro,
        "checkpoint": str(checkpoint),
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "schema": manifest["schema_version"],
        },
        "writer_state": _writer_state_record(writer_path),
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
    if require_formal and status != "sealed":
        raise ExpertManifoldError(
            "formal v6-prior evaluation requires live A40 smoke evidence"
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
        f"{int(writer_asset['writer_state']['bytes'])}bytes:rank16"
    )
    return {
        "schema_version": EXPERT_MANIFOLD_ADAPTER_SCHEMA,
        "kind": EXPERT_MANIFOLD_WRITER_KIND,
        "arm": f"expert_manifold_v6_prior_{video_condition}",
        "execution_backend": (
            "online_v6_complete_lora_writer_then_episode_lora_cache"
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
            "generated_lora_tensor_count": lora.state_tensor_count,
            **writer_asset,
        },
        "evaluation_authority": {
            "formal_status": status,
            "online_smoke_evidence": config["evaluation"].get(
                "online_smoke_evidence"
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
