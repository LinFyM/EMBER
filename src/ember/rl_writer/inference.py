"""Frozen task-relative Flow-Credit Writer evaluation authority."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.lora import canonical_contract_sha256
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import (
    canonical_hash,
    read_json,
    sha256_file,
    source_reference_matches,
)
from ember.reward.protocol import RewardProtocolError
from ember.rl_writer.checkpoint import validate_rl_writer_checkpoint_files
from ember.rl_writer.contract import (
    RL_WRITER_LAUNCH_SCHEMA,
    authority_path,
    load_rl_writer_config,
    reward_tasks,
    schedule_summary,
)
from ember.writer.as_contract import inspect_video_data, load_writer_config
from ember.writer.as_sampling import TeacherVideoSchedule
from ember.writer.inference import (
    RL_WRITER_ADAPTER_SCHEMA,
    build_writer_evaluation_adapter,
    task_video_mapping,
)


def _inspect_training_checkpoint(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    checkpoint: Path,
    source: Mapping[str, Any],
    require_formal: bool,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    if checkpoint.parent.name != "checkpoints":
        raise RewardProtocolError("Flow-Credit checkpoint is outside a training run")
    training = read_json(checkpoint.parent.parent / "run_contract.json")
    contract_sha = canonical_hash(training)
    world_size = int(training.get("runtime", {}).get("world_size", -1))
    manifest = validate_rl_writer_checkpoint_files(
        checkpoint,
        world_size=world_size,
        contract_sha256=contract_sha,
    )
    cursor = int(manifest.get("next_cycle", -1))
    tasks = reward_tasks(config)
    task_ids = [task.global_task_id for task in tasks]
    video_schedule = TeacherVideoSchedule(
        task_ids=task_ids,
        demo_indices=range(50),
        seed=int(config["data"]["teacher_video_seed"]),
    )
    expected_consumed = schedule_summary(
        tasks,
        world_size=world_size,
        next_cycle=cursor,
        seed=int(config["data"]["task_schedule_seed"]),
        rollouts_per_task=int(config["algorithm"]["rollouts_per_task_condition"]),
        video_schedule=video_schedule,
    )
    valid = (
        training.get("schema_version") == RL_WRITER_LAUNCH_SCHEMA
        and training.get("stage") == "development"
        and training.get("config_sha256") == sha256_file(config_path)
        and source_reference_matches(training.get("source"), source)
        and training.get("authorities") == config["authorities"]
        and training.get("information_wall") == config["information_wall"]
        and [int(row["global_task_id"]) for row in training.get("tasks", [])]
        == task_ids
        and training.get("trainable", {}).get("object")
        == "sft_anchored_tangent_basis_writer_coefficients_only"
        and training.get("trainable", {}).get("coldstart_teacher_action_phase_closed")
        is True
        and world_size in {1, 2, 3, 4, 6}
        and cursor > 0
        and cursor in training.get("runtime", {}).get("checkpoint_cycles", [])
        and manifest.get("consumed") == expected_consumed
        and checkpoint.name == f"cycle_{cursor:08d}"
    )
    if require_formal:
        valid = (
            valid
            and training.get("mode") == "formal"
            and config.get("formal_run", {}).get("status") == "sealed"
        )
    elif training.get("mode") not in {"profile", "formal"}:
        valid = False
    if not valid:
        raise RewardProtocolError("Flow-Credit training checkpoint authority changed")
    return training, manifest, cursor


def inspect_rl_writer_evaluation(
    *,
    config_path: Path,
    checkpoint: Path,
    video_data_root: Path,
    source: Mapping[str, Any],
    task_keys: Sequence[tuple[str, int]],
    video_condition: str,
    video_seed: int,
    video_sampling_mode: str | None,
    require_formal: bool,
) -> dict[str, Any]:
    """Seal one reward-trained Writer without reading evaluation rewards/actions."""

    config_path = config_path.resolve()
    checkpoint = checkpoint.resolve()
    video_data_root = video_data_root.resolve()
    config = load_rl_writer_config(config_path)
    as_config = load_writer_config(authority_path(config, "as_writer_config"))
    target_manifest = read_json(authority_path(config, "target_data_manifest"))
    target_by_key = {
        (str(row["suite"]), int(row["task_id"])): row
        for row in target_manifest.get("tasks", [])
    }
    normalized_keys = tuple((str(suite), int(task_id)) for suite, task_id in task_keys)
    if set(normalized_keys) - set(target_by_key):
        raise RewardProtocolError("Flow-Credit evaluation task is outside target40")
    roles = {key: str(target_by_key[key]["split_role"]) for key in normalized_keys}
    mapping = task_video_mapping(normalized_keys, roles, video_condition)
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
    video_data = inspect_video_data(
        video_data_root, as_config, needed, verify_hashes=False
    )
    if training.get("video_data", {}).get("dataset") != video_data.get("dataset"):
        raise RewardProtocolError("Flow-Credit checkpoint and video data disagree")
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    return build_writer_evaluation_adapter(
        schema_version=RL_WRITER_ADAPTER_SCHEMA,
        writer_method="rl_writer",
        config_path=config_path,
        checkpoint=checkpoint,
        training=training,
        manifest=manifest,
        cursor=cursor,
        cursor_axis="outer_cycle",
        video_data=video_data,
        lora_contract_sha256=canonical_contract_sha256(lora),
        mapping=mapping,
        task_keys=normalized_keys,
        source=source,
        video_condition=video_condition,
        video_seed=video_seed,
        video_sampling_mode=video_sampling_mode,
        forbidden_inputs=config["information_wall"]["writer_forbidden_inputs"],
    )
