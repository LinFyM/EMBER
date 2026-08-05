"""K4 AS-Writer evaluation authority and per-episode evidence."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json
from ember.pi05_target_data import SUITE_ORDER, target_global_task_id
from ember.writer.as_contract import (
    REPO_ROOT,
    SUPPORTED_AS_WRITER_LAUNCH_SCHEMAS,
    inspect_video_data,
    load_writer_config,
    writer_split_roles,
    writer_stage,
)
from ember.writer.checkpoint import validate_writer_checkpoint_files
from ember.writer.model import WriterModelError
from ember.writer.video_schedule import (
    SAME_TASK_OTHER_DEMO_OFFSET,
    writer_condition_demo_indices,
    writer_video_demo_indices,
    writer_video_schedule_contract,
    writer_video_selection_seed,
)


WRITER_ADAPTER_SCHEMA = "ember_pi05_k4_invariant_m2p_writer_eval_adapter_v1"
RL_WRITER_ADAPTER_SCHEMA = "ember_pi05_k4_reward_writer_eval_adapter_v1"
WRITER_ADAPTER_SCHEMAS = {WRITER_ADAPTER_SCHEMA, RL_WRITER_ADAPTER_SCHEMA}
WRITER_VIDEO_CONDITIONS = {
    "correct",
    "same_task_other",
    "cross_suite_wrong",
    "shuffled",
    "shuffled_keep_first",
    "reversed",
}
WRONG_VIDEO_CONDITIONS = {"cross_suite_wrong"}
WRITER_EPISODE_EVIDENCE = "ember_pi05_k4_invariant_m2p_episode_evidence_v1"
WRITER_GENERATION_SEED_SCHEDULE = "numeric_seedsequence_k4_frame_order_v1"


def writer_generation_seed(
    root_seed: int,
    suite: str,
    task_id: int,
    demo_index: int,
    *,
    stream: str,
    shot_ordinal: int = 0,
) -> int:
    """Return one deterministic numeric frame-order seed without hashing."""

    if (
        root_seed < 0
        or suite not in SUITE_ORDER
        or not 0 <= task_id < 10
        or demo_index < 0
        or stream != "frame_order"
        or not 0 <= shot_ordinal < 4
    ):
        raise WriterModelError("invalid K4 Writer generation seed key")
    state = np.random.SeedSequence(
        [
            root_seed,
            SUITE_ORDER.index(suite),
            task_id,
            demo_index,
            shot_ordinal,
            0xF04D,
        ]
    ).generate_state(2, dtype=np.uint32)
    return (int(state[0]) << 31 | int(state[1])) & ((1 << 63) - 1)


def writer_shuffled_frame_permutation(
    frame_count: int,
    order_seed: int,
    *,
    keep_first: bool,
) -> torch.Tensor:
    if frame_count <= 0 or order_seed < 0:
        raise WriterModelError("invalid K4 Writer frame permutation request")
    generator = torch.Generator(device="cpu").manual_seed(order_seed)
    permutation = torch.randperm(frame_count, generator=generator)
    if keep_first:
        permutation = torch.cat(
            (torch.zeros(1, dtype=permutation.dtype), permutation[permutation != 0])
        )
    return permutation


def _task_video_mapping(
    task_keys: Sequence[tuple[str, int]],
    task_roles: Mapping[tuple[str, int], str],
    condition: str,
) -> tuple[dict[str, Any], ...]:
    if condition not in WRITER_VIDEO_CONDITIONS or not task_keys:
        raise WriterModelError("invalid K4 Writer evaluation video condition")
    normalized = tuple((str(suite), int(task_id)) for suite, task_id in task_keys)
    if len(set(normalized)) != len(normalized):
        raise WriterModelError("K4 Writer evaluation tasks are duplicated")
    selected = set(normalized)
    roles = sorted({str(task_roles.get(key, "")) for key in normalized})
    if not roles or "" in roles or set(task_roles) != selected:
        raise WriterModelError("K4 Writer split-role mapping changed")
    result: list[dict[str, Any]] = []
    for role in roles:
        by_suite = {
            suite: tuple(
                sorted(
                    task_id
                    for name, task_id in normalized
                    if name == suite and task_roles[(name, task_id)] == role
                )
            )
            for suite in SUITE_ORDER
        }
        if any(not values for values in by_suite.values()) or len(
            {len(values) for values in by_suite.values()}
        ) != 1:
            raise WriterModelError("K4 cross-suite control panel is unbalanced")
        for suite in SUITE_ORDER:
            for ordinal, task_id in enumerate(by_suite[suite]):
                video_suite, video_task_id = suite, task_id
                if condition in WRONG_VIDEO_CONDITIONS:
                    video_suite = SUITE_ORDER[
                        (SUITE_ORDER.index(suite) + 1) % len(SUITE_ORDER)
                    ]
                    video_task_id = by_suite[video_suite][ordinal]
                result.append(
                    {
                        "suite": suite,
                        "task_id": task_id,
                        "language_global_task_id": target_global_task_id(suite, task_id),
                        "language_split_role": role,
                        "video_suite": video_suite,
                        "video_task_id": video_task_id,
                        "video_global_task_id": target_global_task_id(
                            video_suite, video_task_id
                        ),
                        "video_split_role": role,
                    }
                )
    return tuple(
        sorted(result, key=lambda row: (SUITE_ORDER.index(row["suite"]), row["task_id"]))
    )


task_video_mapping = _task_video_mapping


def expected_writer_episode_evidence(
    adapter: Mapping[str, Any],
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
    lora_reference: str,
    evidence_schema: str | None = None,
) -> dict[str, Any]:
    """Build all recomputable K4 episode fields without content hashes."""

    if not lora_reference or adapter.get("schema_version") not in WRITER_ADAPTER_SCHEMAS:
        raise WriterModelError("K4 Writer row lacks a valid adapter reference")
    matches = [
        row
        for row in adapter.get("task_video_mapping", [])
        if row.get("suite") == suite and int(row.get("task_id", -1)) == task_id
    ]
    if len(matches) != 1:
        raise WriterModelError("K4 Writer row task is outside its video mapping")
    mapping = matches[0]
    schedule = adapter.get("video_schedule", {})
    seed = int(schedule.get("seed", -1))
    count = int(schedule.get("demo_count", -1))
    sampling_mode = str(schedule.get("sampling_mode", "without_replacement"))
    reference = writer_video_demo_indices(
        seed,
        suite,
        task_id,
        init_state_id,
        demo_count=count,
        sampling_mode=sampling_mode,
    )
    selected = writer_condition_demo_indices(
        seed,
        suite,
        task_id,
        init_state_id,
        condition=str(adapter["video_condition"]),
        valid_conditions=WRITER_VIDEO_CONDITIONS,
        demo_count=count,
        sampling_mode=sampling_mode,
    )
    schema = evidence_schema or WRITER_EPISODE_EVIDENCE
    if schema != WRITER_EPISODE_EVIDENCE:
        raise WriterModelError("unsupported K4 Writer episode evidence")
    result = {
        "schema_version": schema,
        "writer_method": adapter.get("writer_method", "as_writer"),
        "method_arm": adapter["arm"],
        "condition": adapter["video_condition"],
        "writer_checkpoint_axis": adapter["checkpoint"].get(
            "cursor_axis", "optimizer_step"
        ),
        "writer_checkpoint_cursor": int(adapter["checkpoint"]["cursor"]),
        "writer_checkpoint_reference": adapter["checkpoint"]["reference"],
        "lora_contract_reference": adapter["lora_contract"]["reference"],
        "lora_reference": lora_reference,
        "language_global_task_id": int(mapping["language_global_task_id"]),
        "teacher_video_kind": adapter["video_condition"],
        "teacher_video_seed_root": seed,
        "teacher_video_selection_seed": writer_video_selection_seed(
            seed,
            suite,
            task_id,
            init_state_id,
            sampling_mode=sampling_mode,
        ),
        "teacher_video_sampling_mode": sampling_mode,
        "video_suite": str(mapping["video_suite"]),
        "video_task_id": int(mapping["video_task_id"]),
        "video_global_task_id": int(mapping["video_global_task_id"]),
        "video_split_role": str(mapping["video_split_role"]),
        "teacher_demo_indices": list(selected),
        "teacher_reference_demo_indices": list(reference),
        "task_video_mapping_reference": adapter["task_video_mapping_reference"],
        "pairing_reference": adapter["pairing_reference"],
        "writer_generation_seed_schedule": WRITER_GENERATION_SEED_SCHEDULE,
        "teacher_video_order_seeds": [
            writer_generation_seed(
                seed,
                suite,
                task_id,
                demo,
                stream="frame_order",
                shot_ordinal=ordinal,
            )
            for ordinal, demo in enumerate(reference)
        ],
    }
    if adapter["video_condition"] == "same_task_other":
        result["teacher_demo_offset"] = SAME_TASK_OTHER_DEMO_OFFSET
    return result


def validate_writer_episode_evidence(
    adapter: Mapping[str, Any] | None,
    row: Any,
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
) -> bool:
    if adapter is None:
        return row is None
    if not isinstance(row, Mapping):
        return False
    try:
        generation_seconds = float(row.get("writer_generation_seconds", float("nan")))
        expected = expected_writer_episode_evidence(
            adapter,
            suite=suite,
            task_id=task_id,
            init_state_id=init_state_id,
            lora_reference=str(row.get("lora_reference", "")),
            evidence_schema=str(row.get("schema_version", "")),
        )
    except (WriterModelError, TypeError, ValueError):
        return False
    observed = dict(row)
    observed.pop("writer_generation_seconds", None)
    return observed == expected and math.isfinite(generation_seconds) and generation_seconds >= 0


def _source_step_matches(recorded: Any, current: Mapping[str, Any]) -> bool:
    return (
        isinstance(recorded, Mapping)
        and int(recorded.get("optimizer_step", -1)) == 1000
        and int(current.get("optimizer_step", -1)) == 1000
    )


def _inspect_training_checkpoint(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    checkpoint: Path,
    source: Mapping[str, Any],
    require_formal: bool,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    checkpoint = checkpoint.resolve()
    if checkpoint.parent.name != "checkpoints":
        raise WriterModelError("K4 Writer checkpoint is outside a training run")
    training = read_json(checkpoint.parent.parent / "run_contract.json")
    world_size = int(training.get("runtime", {}).get("world_size", -1))
    reference = str(training.get("schema_version", ""))
    manifest = validate_writer_checkpoint_files(
        checkpoint,
        world_size=world_size,
        contract_sha256=reference,
    )
    cursor = int(manifest.get("consumed", {}).get("next_step", -1))
    target = read_json(REPO_ROOT / config["authorities"]["target_data_manifest"]["path"])
    roles = target.get("summary", {}).get("roles", {})
    source_ids = sorted(
        int(task_id)
        for role in writer_split_roles(config)
        for task_id in roles.get(role, [])
    )
    valid = (
        training.get("schema_version") in SUPPORTED_AS_WRITER_LAUNCH_SCHEMAS
        and training.get("stage", "development") == writer_stage(config)
        and Path(str(training.get("config_path", ""))).resolve() == config_path
        and _source_step_matches(training.get("source"), source)
        and training.get("authorities") == config["authorities"]
        and training.get("information_wall") == config["information_wall"]
        and training.get("writer") == config["writer"]
        and training.get("data") == config["data"]
        and training.get("task_ids") == source_ids
        and training.get("trainable", {}).get("object")
        == "shared_action_supervised_writer_only"
        and world_size == int(config["formal_run"]["expected_world_size"])
        and cursor > 0
        and cursor in training.get("runtime", {}).get("checkpoint_steps", [])
        and checkpoint.name == f"step_{cursor:08d}"
    )
    if require_formal:
        valid = valid and training.get("mode") == "formal" and config["formal_run"].get("status") == "sealed"
    else:
        valid = valid and training.get("mode") in {"profile", "formal"}
    if not valid:
        raise WriterModelError("K4 Writer training checkpoint authority changed")
    return training, manifest, cursor


def build_writer_evaluation_adapter(
    *,
    schema_version: str,
    writer_method: str,
    config_path: Path,
    checkpoint: Path,
    training: Mapping[str, Any],
    manifest: Mapping[str, Any],
    cursor: int,
    cursor_axis: str,
    video_data: Mapping[str, Any],
    lora_contract_reference: str,
    mapping: Sequence[Mapping[str, Any]],
    task_keys: Sequence[tuple[str, int]],
    source: Mapping[str, Any],
    video_condition: str,
    video_seed: int,
    forbidden_inputs: Sequence[str],
    video_sampling_mode: str | None = None,
) -> dict[str, Any]:
    del source
    if schema_version not in WRITER_ADAPTER_SCHEMAS or writer_method not in {"as_writer", "rl_writer"}:
        raise WriterModelError("invalid K4 Writer evaluation method")
    writer_record = manifest.get("files", {}).get("writer.safetensors", {})
    if int(writer_record.get("bytes", -1)) <= 0:
        raise WriterModelError("K4 Writer checkpoint lacks Writer state")
    video_schedule, pairing_schema, _ = writer_video_schedule_contract(
        video_sampling_mode, seed=video_seed, demo_count=50
    )
    checkpoint_reference = (
        f"{training['schema_version']}:{cursor}:writer-bytes={writer_record['bytes']}"
    )
    task_mapping_reference = "k4_same_role_next_suite_ordinal_v1"
    return {
        "schema_version": schema_version,
        "kind": writer_method,
        "writer_method": writer_method,
        "arm": f"{writer_method}_{video_condition}_video",
        "execution_backend": "two_stage_cached_per_sample_k4_lora_batched_replan",
        "video_condition": video_condition,
        "writer_input": "task language plus exactly four action-hidden videos jointly generating one LoRA",
        "config": {"path": str(config_path)},
        "training_run": {
            "path": str(checkpoint.parent.parent),
            "schema_version": str(training["schema_version"]),
            "mode": training["mode"],
            "git_commit": training["git"]["commit"],
        },
        "checkpoint": {
            "path": str(checkpoint),
            "cursor": cursor,
            "cursor_axis": cursor_axis,
            "reference": checkpoint_reference,
            "writer_state_bytes": int(writer_record["bytes"]),
        },
        "video_data": dict(video_data),
        "lora_contract": {"reference": lora_contract_reference},
        "video_schedule": video_schedule,
        "wrong_video_mapping": (
            "identity"
            if video_condition not in WRONG_VIDEO_CONDITIONS
            else "same role-panel ordinal in the next suite cyclically"
        ),
        "task_video_mapping_reference": task_mapping_reference,
        "task_video_mapping": list(mapping),
        "pairing_reference": f"{pairing_schema}:{writer_method}:{len(task_keys)}tasks:seed{video_seed}",
        "writer_forbidden_tensor_inputs": list(forbidden_inputs),
        "teacher_action_values_read_by_evaluator": 0,
    }


def inspect_as_writer_evaluation(
    *,
    config_path: Path,
    checkpoint: Path,
    video_data_root: Path,
    source: Mapping[str, Any],
    task_keys: Sequence[tuple[str, int]],
    video_condition: str,
    video_seed: int,
    require_formal: bool,
    video_sampling_mode: str | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    checkpoint = checkpoint.resolve()
    video_data_root = video_data_root.resolve()
    config = load_writer_config(config_path)
    target = read_json(REPO_ROOT / config["authorities"]["target_data_manifest"]["path"])
    by_key = {
        (str(row["suite"]), int(row["task_id"])): row for row in target.get("tasks", [])
    }
    keys = tuple((str(suite), int(task_id)) for suite, task_id in task_keys)
    if set(keys) - set(by_key):
        raise WriterModelError("K4 Writer evaluation task is outside target40")
    roles = {key: str(by_key[key]["split_role"]) for key in keys}
    mapping = _task_video_mapping(keys, roles, video_condition)
    needed = tuple(
        sorted(
            {int(row["language_global_task_id"]) for row in mapping}
            | {int(row["video_global_task_id"]) for row in mapping}
        )
    )
    training, manifest, cursor = _inspect_training_checkpoint(
        config_path=config_path,
        config=config,
        checkpoint=checkpoint,
        source=source,
        require_formal=require_formal,
    )
    video_data = inspect_video_data(video_data_root, config, needed, verify_hashes=False)
    if training.get("video_data", {}).get("dataset") != video_data.get("dataset"):
        raise WriterModelError("K4 Writer checkpoint and video data disagree")
    lora = load_pi05_lora_contract(REPO_ROOT / config["authorities"]["lora_contract"]["path"])
    return build_writer_evaluation_adapter(
        schema_version=WRITER_ADAPTER_SCHEMA,
        writer_method="as_writer",
        config_path=config_path,
        checkpoint=checkpoint,
        training=training,
        manifest=manifest,
        cursor=cursor,
        cursor_axis="optimizer_step",
        video_data=video_data,
        lora_contract_reference=(
            f"{config['authorities']['lora_contract']['path']}:"
            f"{lora.state_tensor_count}tensors:{lora.parameter_count}parameters"
        ),
        mapping=mapping,
        task_keys=keys,
        source=source,
        video_condition=video_condition,
        video_seed=video_seed,
        video_sampling_mode=video_sampling_mode,
        forbidden_inputs=config["information_wall"]["writer_forbidden_inputs"],
    )
