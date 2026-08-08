"""Formal authority for the hard-routed policy-effective Expert-Manifold Writer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.expert_manifold.contract import (
    BARYCENTRIC_CONFIG_SCHEMA,
    ExpertManifoldError,
    authority_path,
    load_barycentric_writer_config,
    load_expert_manifold_config,
)
from ember.expert_manifold.evaluation import inspect_task_expert_bank
from ember.expert_manifold.feature_cache import inspect_feature_cache
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


EXPERT_MANIFOLD_WRITER_KIND = "expert_manifold_writer"
EXPERT_MANIFOLD_ADAPTER_SCHEMA = (
    "ember_pi05_expert_manifold_hard_routed_eval_adapter_v4"
)
EXPERT_MANIFOLD_EPISODE_SCHEMA = (
    "ember_pi05_expert_manifold_hard_routed_episode_v4"
)


def _target_rows(config: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    manifest = read_json(authority_path(config, "target_data_manifest"))
    rows = {int(row["global_task_id"]): dict(row) for row in manifest["tasks"]}
    if len(rows) != 40:
        raise ExpertManifoldError("barycentric Writer target manifest changed")
    return rows


def _train_task_keys(config: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    rows = [
        row for row in _target_rows(config).values() if row["split_role"] == "train"
    ]
    rows.sort(key=lambda row: int(row["global_task_id"]))
    if len(rows) != 24:
        raise ExpertManifoldError("barycentric Writer did not resolve train24")
    return tuple((str(row["suite"]), int(row["task_id"])) for row in rows)


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


def _validate_asset_linkage(
    *,
    config: Mapping[str, Any],
    expert: Mapping[str, Any],
    cache: Mapping[str, Any],
) -> None:
    expert_rows = sorted(expert["tasks"], key=lambda row: int(row["ordinal"]))
    cache_rows = sorted(cache["tasks"], key=lambda row: int(row["task_ordinal"]))
    valid = len(expert_rows) == len(cache_rows) == 24
    for ordinal, (expert_row, cache_row) in enumerate(
        zip(expert_rows, cache_rows, strict=True)
    ):
        valid = valid and (
            int(expert_row["ordinal"]) == ordinal
            and int(cache_row["task_ordinal"]) == ordinal
            and int(expert_row["global_task_id"]) == int(cache_row["global_task_id"])
            and expert_row["suite"] == cache_row["suite"]
            and int(expert_row["task_id"]) == int(cache_row["task_id"])
            and expert_row["language"] == cache_row["language"]
        )
    if (
        not valid
        or int(expert["step"]) != int(config["expert_basis"]["expert_step"])
        or int(cache["demo_count"])
        != int(config["expert_basis"]["centroid_videos_per_task"])
    ):
        raise ExpertManifoldError("expert basis and video centroids are misaligned")


def _inspect_fixed_assets(
    config: Mapping[str, Any],
    *,
    expert_bank_root: Path,
    feature_cache_root: Path,
    source: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    asset_config_path = authority_path(config, "asset_config").resolve()
    asset_config = load_expert_manifold_config(asset_config_path)
    for name in (
        "target_data_manifest",
        "evaluation_config",
        "lora_contract",
        "source_base_config",
    ):
        if config["authorities"][name] != asset_config["authorities"][name]:
            raise ExpertManifoldError("barycentric asset authority changed")

    expert = inspect_task_expert_bank(
        config_path=asset_config_path,
        bank_root=expert_bank_root.resolve(),
        step=int(config["expert_basis"]["expert_step"]),
        source=source,
        task_keys=_train_task_keys(config),
        evaluation_role="development_train",
        require_formal=True,
    )
    cache = inspect_feature_cache(
        asset_config_path,
        feature_cache_root.resolve(),
        source=source,
    )
    _validate_asset_linkage(config=config, expert=expert, cache=cache)
    return asset_config_path, expert, cache


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
        raise ExpertManifoldError("Expert-Manifold evaluation task panel changed")
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


def _fixed_asset_records(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    asset_config_path: Path,
    expert_bank_root: Path,
    feature_cache_root: Path,
    expert: Mapping[str, Any],
    cache: Mapping[str, Any],
) -> dict[str, Any]:
    writer = config["barycentric_writer"]
    ridge = float(writer["ridge"])
    asset_reference = (
        f"{BARYCENTRIC_CONFIG_SCHEMA}:step{int(expert['step'])}:"
        f"24experts:50centroids:ridge{ridge:g}:effectiveBA:"
        f"subspace{int(writer['effective_basis_rank'])}:hard1:rank16"
    )
    cache_manifest_path = feature_cache_root.resolve() / "cache_manifest.json"
    return {
        "config": {
            "path": str(config_path),
            "bytes": config_path.stat().st_size,
            "schema": config["schema_version"],
        },
        "writer_asset": {
            "reference": asset_reference,
            "learned_parameter_count": 0,
            "expert_step": int(expert["step"]),
            "expert_count": len(expert["tasks"]),
            "centroid_videos_per_task": int(cache["demo_count"]),
            "ridge": ridge,
            "reconstruction": writer["reconstruction"],
            "effective_basis_rank": int(writer["effective_basis_rank"]),
            "deployed_coefficient_support": int(
                writer["deployed_coefficient_support"]
            ),
        },
        "expert_basis": {
            "root": str(expert_bank_root.resolve()),
            "asset_config": str(asset_config_path),
            "training_commit": expert["training_commit"],
            "step": int(expert["step"]),
            "task_count": len(expert["tasks"]),
            "tasks": [dict(row) for row in expert["tasks"]],
        },
        "feature_cache": {
            "root": str(feature_cache_root.resolve()),
            "manifest_path": str(cache_manifest_path),
            "manifest_bytes": cache_manifest_path.stat().st_size,
            "schema": cache["schema_version"],
            "training_commit": cache["training_commit"],
            "task_count": int(cache["task_count"]),
            "demo_count": int(cache["demo_count"]),
            "tasks": [dict(row) for row in cache["tasks"]],
        },
    }


def inspect_expert_manifold_writer_evaluation(
    *,
    config_path: Path,
    expert_bank_root: Path,
    feature_cache_root: Path,
    video_data_root: Path,
    source: Mapping[str, Any],
    task_keys: Sequence[tuple[str, int]],
    video_condition: str,
    video_seed: int,
    video_sampling_mode: str,
    require_formal: bool,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_barycentric_writer_config(config_path)
    if video_condition not in VIDEO_CONDITIONS:
        raise ExpertManifoldError("unsupported Expert-Manifold video condition")
    status = str(config["evaluation"]["formal_status"])
    if require_formal and status != "sealed":
        raise ExpertManifoldError(
            "formal hard-routed evaluation requires live A40 smoke evidence"
        )
    asset_config_path, expert, cache = _inspect_fixed_assets(
        config,
        expert_bank_root=expert_bank_root,
        feature_cache_root=feature_cache_root,
        source=source,
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
    assets = _fixed_asset_records(
        config_path=config_path,
        config=config,
        asset_config_path=asset_config_path,
        expert_bank_root=expert_bank_root,
        feature_cache_root=feature_cache_root,
        expert=expert,
        cache=cache,
    )
    return {
        "schema_version": EXPERT_MANIFOLD_ADAPTER_SCHEMA,
        "kind": EXPERT_MANIFOLD_WRITER_KIND,
        "arm": f"expert_manifold_hard_routed_{video_condition}",
        "execution_backend": (
            "online_frozen_pi05_video_innovation_then_hard_routed_"
            "policy_effective_lora_cache"
        ),
        **assets,
        "evaluation_authority": {
            "formal_status": status,
            "cpu_coefficient_evidence": dict(
                config["evaluation"]["cpu_coefficient_evidence"]
            ),
            "cpu_policy_effective_compiler": dict(
                config["evaluation"]["cpu_policy_effective_compiler"]
            ),
            "cpu_runtime_evidence": dict(config["evaluation"]["cpu_runtime_evidence"]),
            "cpu_hard_route_evidence": dict(
                config["evaluation"].get("cpu_hard_route_evidence", {})
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
            "no_video_counterfactual": video_condition == "no_video",
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
        "writer_asset_reference": adapter["writer_asset"]["reference"],
        "writer_learned_parameter_count": int(
            adapter["writer_asset"]["learned_parameter_count"]
        ),
        "expert_basis_step": int(adapter["writer_asset"]["expert_step"]),
        "expert_basis_task_count": int(adapter["writer_asset"]["expert_count"]),
        "deployed_coefficient_support": int(
            adapter["writer_asset"]["deployed_coefficient_support"]
        ),
        "lora_contract_reference": adapter["lora_contract"]["reference"],
        "lora_reference": lora_reference,
        "language_global_task_id": int(mapping["language_global_task_id"]),
        "teacher_video_kind": adapter["video_condition"],
        "teacher_video_frames_used": adapter["video_condition"] != "no_video",
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
