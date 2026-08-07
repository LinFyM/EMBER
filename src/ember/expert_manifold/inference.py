"""Formal evaluation authority for the video-conditioned topological Writer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.expert_manifold.contract import (
    REPO_ROOT,
    ExpertManifoldError,
    authority_path,
    load_expert_manifold_config,
)
from ember.expert_manifold.video_schedule import (
    SAME_TASK_OTHER_OFFSET,
    VIDEO_CONDITIONS,
    condition_demo_index,
    frame_order_seed,
    reference_demo_index,
    video_schedule_contract,
    video_selection_seed,
)
from ember.expert_manifold.writer_checkpoint import WRITER_CHECKPOINT_SCHEMA
from ember.expert_manifold.writer_training import WRITER_RUN_SCHEMA
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json
from ember.writer.inference import task_video_mapping


EXPERT_MANIFOLD_WRITER_KIND = "expert_manifold_writer"
EXPERT_MANIFOLD_ADAPTER_SCHEMA = "ember_pi05_expert_manifold_writer_eval_adapter_v1"
EXPERT_MANIFOLD_EPISODE_SCHEMA = "ember_pi05_expert_manifold_writer_episode_v1"


def _training_checkpoint(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    checkpoint: Path,
    source: Mapping[str, Any],
    require_formal: bool,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    checkpoint = checkpoint.resolve()
    if checkpoint.parent.name != "checkpoints" or not checkpoint.name.startswith(
        "macro_"
    ):
        raise ExpertManifoldError("topological Writer checkpoint is outside its run")
    run_root = checkpoint.parent.parent
    training = read_json(run_root / "run_contract.json")
    manifest_path = checkpoint / "manifest.json"
    manifest = read_json(manifest_path)
    cursor = int(manifest.get("next_macro", -1))
    formal = config["meta_training"]["formal_run"]
    training_expert_step = int(training.get("expert_bank", {}).get("step", -1))
    selected_expert_step = formal.get("selected_expert_step")
    valid = (
        training.get("schema_version") == WRITER_RUN_SCHEMA
        and Path(str(training.get("config", {}).get("path", ""))).resolve()
        == config_path
        and training.get("config", {}).get("schema") == config["schema_version"]
        and training.get("source") == dict(source)
        and training.get("method") == config["method"]
        and training.get("information_wall") == config["information_wall"]
        and training.get("topological_writer") == config["topological_writer"]
        and training.get("meta_training") == config["meta_training"]
        and training_expert_step > 0
        and int(training.get("runtime", {}).get("world_size", -1))
        == int(formal["expected_world_size"])
        and manifest.get("schema_version") == WRITER_CHECKPOINT_SCHEMA
        and manifest.get("content_hash_policy") == "disabled_by_owner"
        and int(manifest.get("world_size", -1)) == int(formal["expected_world_size"])
        and cursor in tuple(int(value) for value in formal["checkpoint_macros"])
        and checkpoint.name == f"macro_{cursor:08d}"
    )
    if require_formal:
        valid = (
            valid
            and training.get("mode") == "formal"
            and formal.get("status") == "sealed"
            and selected_expert_step is not None
            and training_expert_step == int(selected_expert_step)
        )
    else:
        valid = valid and training.get("mode") in {"profile", "formal"}
    for name, expected_bytes in manifest.get("files", {}).items():
        path = checkpoint / name
        valid = valid and path.is_file() and path.stat().st_size == int(expected_bytes)
    if not valid:
        raise ExpertManifoldError("topological Writer training authority changed")
    return training, manifest, cursor


def _target_rows(config: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    manifest = read_json(authority_path(config, "target_data_manifest"))
    rows = {int(row["global_task_id"]): dict(row) for row in manifest["tasks"]}
    if len(rows) != 40:
        raise ExpertManifoldError("topological Writer target manifest changed")
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
    config = load_expert_manifold_config(config_path)
    if video_condition not in VIDEO_CONDITIONS:
        raise ExpertManifoldError("unsupported Expert-Manifold video condition")
    training, manifest, cursor = _training_checkpoint(
        config_path=config_path,
        config=config,
        checkpoint=checkpoint,
        source=source,
        require_formal=require_formal,
    )
    rows = _target_rows(config)
    by_key = {(row["suite"], int(row["task_id"])): row for row in rows.values()}
    normalized = tuple((str(suite), int(task_id)) for suite, task_id in task_keys)
    if not normalized or len(set(normalized)) != len(normalized) or any(
        key not in by_key for key in normalized
    ):
        raise ExpertManifoldError("Expert-Manifold evaluation task panel changed")
    roles = {key: str(by_key[key]["split_role"]) for key in normalized}
    mapping = task_video_mapping(normalized, roles, video_condition)
    needed = {
        int(row["language_global_task_id"]) for row in mapping
    } | {int(row["video_global_task_id"]) for row in mapping}
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
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    checkpoint = checkpoint.resolve()
    return {
        "schema_version": EXPERT_MANIFOLD_ADAPTER_SCHEMA,
        "kind": EXPERT_MANIFOLD_WRITER_KIND,
        "arm": f"expert_manifold_macro_{cursor}_{video_condition}",
        "execution_backend": "online_frozen_pi05_video_innovation_then_cached_topological_lora",
        "config": {
            "path": str(config_path),
            "bytes": config_path.stat().st_size,
            "schema": config["schema_version"],
        },
        "training_run": {
            "path": str(checkpoint.parent.parent / "run_contract.json"),
            "bytes": (checkpoint.parent.parent / "run_contract.json").stat().st_size,
            "schema": training["schema_version"],
            "commit": training["git"]["commit"],
        },
        "checkpoint": {
            "path": str(checkpoint),
            "cursor": cursor,
            "cursor_axis": "macro",
            "manifest_bytes": (checkpoint / "manifest.json").stat().st_size,
            "writer_bytes": int(manifest["files"]["writer.safetensors"]),
            "reference": (
                f"{WRITER_CHECKPOINT_SCHEMA}:macro{cursor}:"
                f"{int(manifest['files']['writer.safetensors'])}bytes"
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
            "teacher_action_reads": 0,
            "teacher_state_reads": 0,
            "reward_reads": 0,
            "terminal_reads": 0,
            "language_only_lora_path": False,
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
        "writer_checkpoint_axis": "macro",
        "writer_checkpoint_cursor": int(adapter["checkpoint"]["cursor"]),
        "writer_checkpoint_reference": adapter["checkpoint"]["reference"],
        "lora_contract_reference": adapter["lora_contract"]["reference"],
        "lora_reference": lora_reference,
        "language_global_task_id": int(mapping["language_global_task_id"]),
        "teacher_video_kind": adapter["video_condition"],
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
        "writer_generation_seed_schedule": "numeric_seedsequence_one_shot_frame_order_v1",
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
