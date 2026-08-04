"""PI05 AS-Writer evaluation authority and per-sample batched LoRA execution."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ember.lora import canonical_contract_sha256
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import (
    canonical_hash,
    read_json,
    sha256_file,
    source_reference_matches,
)
from ember.pi05_target_data import SUITE_ORDER, target_global_task_id
from ember.writer.as_contract import (
    AS_WRITER_LAUNCH_SCHEMA,
    REPO_ROOT,
    inspect_video_data,
    load_writer_config,
    writer_split_roles,
    writer_stage,
)
from ember.writer.checkpoint import validate_writer_checkpoint_files
from ember.writer.model import WriterModelError
from ember.writer.video_schedule import (
    SAME_TASK_OTHER_DEMO_OFFSET,
    writer_condition_demo_index,
    writer_video_demo_index,
    writer_video_schedule_contract,
    writer_video_selection_seed,
)


WRITER_ADAPTER_SCHEMA = "ember_pi05_v6_relative_flow_writer_eval_adapter_v1"
RL_WRITER_ADAPTER_SCHEMA = "ember_pi05_task_relative_flow_credit_eval_adapter_v2"
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
WRITER_EPISODE_EVIDENCE_WITH_REPLACEMENT = (
    "ember_pi05_v6_relative_flow_episode_evidence_with_replacement_v1"
)
WRITER_EPISODE_EVIDENCE_CV = (
    "ember_pi05_v6_relative_flow_episode_evidence_v1"
)
WRITER_GENERATION_SEED_SCHEDULE = (
    "sha256 first 63 bits of canonical JSON: ember_pi05_writer_generation_v5_1/"
    "frame_order/seed/suite/task_id/demo_index"
)


def writer_generation_seed(
    root_seed: int,
    suite: str,
    task_id: int,
    demo_index: int,
    *,
    stream: str,
) -> int:
    if (
        root_seed < 0 or suite not in SUITE_ORDER or not 0 <= task_id < 10
        or demo_index < 0 or stream != "frame_order"
    ):
        raise WriterModelError("invalid AS-Writer generation seed key")
    encoded = json.dumps(
        ["ember_pi05_writer_generation_v5_1", stream, root_seed, suite, task_id, demo_index],
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def writer_shuffled_frame_permutation(
    frame_count: int,
    order_seed: int,
    *,
    keep_first: bool,
) -> torch.Tensor:
    """Reproduce the sealed shuffle, optionally restoring frame zero as anchor."""

    if frame_count <= 0 or order_seed < 0:
        raise WriterModelError("invalid AS-Writer frame permutation request")
    generator = torch.Generator(device="cpu").manual_seed(order_seed)
    permutation = torch.randperm(frame_count, generator=generator)
    if keep_first:
        permutation = torch.cat(
            (
                torch.zeros(1, dtype=permutation.dtype),
                permutation[permutation != 0],
            )
        )
    return permutation


def _task_video_mapping(
    task_keys: Sequence[tuple[str, int]],
    task_roles: Mapping[tuple[str, int], str],
    condition: str,
) -> tuple[dict[str, Any], ...]:
    if condition not in WRITER_VIDEO_CONDITIONS or not task_keys:
        raise WriterModelError("invalid AS-Writer evaluation video condition")
    normalized = tuple((str(suite), int(task_id)) for suite, task_id in task_keys)
    if len(set(normalized)) != len(normalized):
        raise WriterModelError("AS-Writer evaluation tasks are duplicated")
    selected = set(normalized)
    result: list[dict[str, Any]] = []
    roles = sorted({str(task_roles.get(key, "")) for key in normalized})
    if not roles or "" in roles or set(task_roles) != selected:
        raise WriterModelError("AS-Writer evaluation split-role mapping changed")
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
            raise WriterModelError(
                "cross-suite Writer control requires equal per-suite panels within each role"
            )
        for suite in SUITE_ORDER:
            for ordinal, task_id in enumerate(by_suite[suite]):
                video_suite = suite
                video_task_id = task_id
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
    return tuple(sorted(result, key=lambda row: (SUITE_ORDER.index(row["suite"]), row["task_id"])))


task_video_mapping = _task_video_mapping


def expected_writer_episode_evidence(
    adapter: Mapping[str, Any],
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
    lora_sha256: str,
    evidence_schema: str | None = None,
) -> dict[str, Any]:
    """Build the exact dynamic row fields implied by a sealed adapter contract."""

    if re.fullmatch(r"[0-9a-f]{64}", lora_sha256) is None:
        raise WriterModelError("AS-Writer row lacks a valid LoRA hash")
    if adapter.get("schema_version") not in WRITER_ADAPTER_SCHEMAS:
        raise WriterModelError("unsupported PI05 Writer evaluation adapter")
    matches = [
        row
        for row in adapter.get("task_video_mapping", [])
        if row.get("suite") == suite and int(row.get("task_id", -1)) == task_id
    ]
    if len(matches) != 1:
        raise WriterModelError("AS-Writer row task is outside its video mapping")
    mapping = matches[0]
    schedule = adapter.get("video_schedule", {})
    seed = int(schedule.get("seed", -1))
    count = int(schedule.get("demo_count", -1))
    sampling_mode = str(schedule.get("sampling_mode", "with_replacement"))
    reference_demo_index = writer_video_demo_index(
        seed,
        suite,
        task_id,
        init_state_id,
        demo_count=count,
        sampling_mode=sampling_mode,
    )
    demo_index = writer_condition_demo_index(
        seed,
        suite,
        task_id,
        init_state_id,
        condition=str(adapter["video_condition"]),
        valid_conditions=WRITER_VIDEO_CONDITIONS,
        demo_count=count,
        sampling_mode=sampling_mode,
    )
    selection_seed = writer_video_selection_seed(
        seed,
        suite,
        task_id,
        init_state_id,
        sampling_mode=sampling_mode,
    )
    expected_schema = (
        WRITER_EPISODE_EVIDENCE_WITH_REPLACEMENT
        if sampling_mode == "with_replacement"
        else WRITER_EPISODE_EVIDENCE_CV
    )
    if evidence_schema is None:
        evidence_schema = expected_schema
    if evidence_schema != expected_schema:
        raise WriterModelError("unsupported PI05 Writer episode evidence")
    result = {
        "schema_version": evidence_schema,
        "writer_method": adapter.get("writer_method", "as_writer"),
        "method_arm": adapter["arm"],
        "condition": adapter["video_condition"],
        "writer_checkpoint_axis": adapter["checkpoint"].get(
            "cursor_axis", "optimizer_step"
        ),
        "writer_checkpoint_cursor": int(adapter["checkpoint"]["cursor"]),
        "writer_checkpoint_manifest_sha256": adapter["checkpoint"][
            "manifest_file_sha256"
        ],
        "writer_state_sha256": adapter["checkpoint"]["writer_state_sha256"],
        "lora_contract_sha256": adapter["lora_contract_sha256"],
        "language_global_task_id": int(mapping["language_global_task_id"]),
        "teacher_video_kind": adapter["video_condition"],
        "teacher_video_seed_root": seed,
        "teacher_video_selection_seed": selection_seed,
        "video_suite": str(mapping["video_suite"]),
        "video_task_id": int(mapping["video_task_id"]),
        "video_global_task_id": int(mapping["video_global_task_id"]),
        "video_split_role": str(mapping["video_split_role"]),
        "teacher_demo_index": demo_index,
        "wrong_video_map_sha256": adapter["task_video_mapping_sha256"],
        "pairing_sha256": adapter["pairing_sha256"],
        "lora_sha256": lora_sha256,
    }
    if "sampling_mode" in schedule:
        result["teacher_video_sampling_mode"] = sampling_mode
    result.update(
        {
            "writer_generation_seed_schedule": WRITER_GENERATION_SEED_SCHEDULE,
            "teacher_video_order_seed": writer_generation_seed(
                seed,
                suite,
                task_id,
                reference_demo_index,
                stream="frame_order",
            ),
        }
    )
    if adapter["video_condition"] == "same_task_other":
        result.update(
            {
                "teacher_reference_demo_index": reference_demo_index,
                "teacher_demo_offset": SAME_TASK_OTHER_DEMO_OFFSET,
            }
        )
    return result


def validate_writer_episode_evidence(
    adapter: Mapping[str, Any] | None,
    row: Any,
    *,
    suite: str,
    task_id: int,
    init_state_id: int,
) -> bool:
    """Validate all recomputable per-rollout Writer evidence without model access."""

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
            lora_sha256=str(row.get("lora_sha256", "")),
            evidence_schema=str(row.get("schema_version", "")),
        )
    except (WriterModelError, TypeError, ValueError):
        return False
    observed = dict(row)
    observed.pop("writer_generation_seconds", None)
    return (
        observed == expected
        and math.isfinite(generation_seconds)
        and generation_seconds >= 0
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
        raise WriterModelError("AS-Writer checkpoint is outside a training run")
    run_root = checkpoint.parent.parent
    contract_path = run_root / "run_contract.json"
    training = read_json(contract_path)
    contract_sha256 = canonical_hash(training)
    world_size = int(training.get("runtime", {}).get("world_size", -1))
    manifest = validate_writer_checkpoint_files(
        checkpoint,
        world_size=world_size,
        contract_sha256=contract_sha256,
    )
    cursor = int(manifest.get("consumed", {}).get("next_step", -1))
    checkpoint_authority_valid = (
        cursor > 0
        and cursor in training.get("runtime", {}).get("checkpoint_steps", [])
        and checkpoint.name == f"step_{cursor:08d}"
    )
    target_manifest = read_json(REPO_ROOT / config["authorities"]["target_data_manifest"]["path"])
    role_ids = target_manifest.get("summary", {}).get("roles", {})
    source_ids = [
        int(task_id)
        for role in writer_split_roles(config)
        for task_id in role_ids.get(role, [])
    ]
    lora = load_pi05_lora_contract(
        REPO_ROOT / str(config["authorities"]["lora_contract"]["path"])
    )
    source_matches = source_reference_matches(training.get("source"), source)
    valid = (
        training.get("schema_version") == AS_WRITER_LAUNCH_SCHEMA
        and training.get("stage", "development") == writer_stage(config)
        and training.get("config_sha256") == sha256_file(config_path)
        and source_matches
        and training.get("authorities") == config["authorities"]
        and training.get("information_wall") == config["information_wall"]
        and training.get("writer") == config["writer"]
        and training.get("data") == config["data"]
        and training.get("task_ids") == sorted(source_ids)
        and training.get("trainable", {}).get("object")
        == "shared_action_supervised_writer_only"
        and training.get("trainable", {}).get("lora_contract_sha256")
        == canonical_contract_sha256(lora)
        and world_size
        == int(config.get("formal_run", {}).get("expected_world_size", -1))
        and checkpoint_authority_valid
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
        raise WriterModelError("AS-Writer training checkpoint authority changed")
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
    lora_contract_sha256: str,
    mapping: Sequence[Mapping[str, Any]],
    task_keys: Sequence[tuple[str, int]],
    source: Mapping[str, Any],
    video_condition: str,
    video_seed: int,
    forbidden_inputs: Sequence[str],
    video_sampling_mode: str | None = None,
) -> dict[str, Any]:
    if schema_version not in WRITER_ADAPTER_SCHEMAS or writer_method not in {
        "as_writer",
        "rl_writer",
    }:
        raise WriterModelError("invalid PI05 Writer evaluation method")
    writer_record = manifest.get("files", {}).get("writer.safetensors", {})
    if re.fullmatch(r"[0-9a-f]{64}", str(writer_record.get("sha256", ""))) is None:
        raise WriterModelError("PI05 Writer checkpoint lacks a sealed Writer state")
    mapping_sha256 = canonical_hash(list(mapping))
    video_schedule, pairing_schema, effective_sampling_mode = (
        writer_video_schedule_contract(
            video_sampling_mode,
            seed=video_seed,
            demo_count=50,
        )
    )
    pairing_sha256 = canonical_hash(
        {
            "schema_version": pairing_schema,
            "writer_method": writer_method,
            "source_run_contract_sha256": source.get("source_run_contract_sha256"),
            "source_checkpoint_manifest_sha256": source.get(
                "checkpoint_manifest_sha256"
            ),
            "writer_checkpoint_manifest_sha256": sha256_file(
                checkpoint / "checkpoint_manifest.json"
            ),
            "task_keys": [list(key) for key in task_keys],
            "video_schedule": video_schedule["algorithm"],
            **({"video_sampling_mode": effective_sampling_mode}
               if "sampling_mode" in video_schedule else {}),
            "video_seed": video_seed,
        }
    )
    checkpoint_record = {
        "path": str(checkpoint),
        "cursor": cursor,
        "cursor_axis": cursor_axis,
        "manifest_file_sha256": sha256_file(
            checkpoint / "checkpoint_manifest.json"
        ),
        "manifest_payload_sha256": manifest["canonical_payload_sha256"],
        "writer_state_sha256": writer_record["sha256"],
    }
    result = {
        "schema_version": schema_version,
        "kind": writer_method,
        "writer_method": writer_method,
        "arm": f"{writer_method}_{video_condition}_video",
        "execution_backend": "two_stage_cached_per_sample_lora_batched_replan",
        "video_condition": video_condition,
        "writer_input": (
            "task language plus exactly one raw action-hidden teacher video at inference"
        ),
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "training_run": {
            "path": str(checkpoint.parent.parent),
            "run_contract_file_sha256": sha256_file(
                checkpoint.parent.parent / "run_contract.json"
            ),
            "run_contract_sha256": canonical_hash(training),
            "mode": training["mode"],
            "git_commit": training["git"]["commit"],
        },
        "checkpoint": checkpoint_record,
        "video_data": dict(video_data),
        "lora_contract_sha256": lora_contract_sha256,
        "video_schedule": video_schedule,
        "wrong_video_mapping": (
            "identity"
            if video_condition not in WRONG_VIDEO_CONDITIONS
            else "same role-panel ordinal in the next suite cyclically"
        ),
        "task_video_mapping_sha256": mapping_sha256,
        "task_video_mapping": list(mapping),
        "pairing_sha256": pairing_sha256,
        "writer_forbidden_tensor_inputs": list(forbidden_inputs),
        "teacher_action_values_read_by_evaluator": 0,
    }
    if video_condition == "same_task_other":
        result["video_schedule"].update(
            {
                "same_task_other_demo_offset": SAME_TASK_OTHER_DEMO_OFFSET,
                "transform": (
                    "(paired correct demo + 17) modulo 50; order seed remains "
                    "paired to the correct demo"
                ),
            }
        )
    elif video_condition == "shuffled_keep_first":
        result["video_schedule"]["transform"] = (
            "use the sealed full-shuffle permutation, move original frame zero "
            "to the front, and preserve the relative order of every other frame"
        )
    return result


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
    """Seal a Writer checkpoint and raw-video authority before queue creation."""

    config_path = config_path.resolve()
    checkpoint = checkpoint.resolve()
    video_data_root = video_data_root.resolve()
    config = load_writer_config(config_path)
    target_manifest = read_json(
        REPO_ROOT / str(config["authorities"]["target_data_manifest"]["path"])
    )
    target_by_key = {
        (str(row["suite"]), int(row["task_id"])): row
        for row in target_manifest.get("tasks", [])
    }
    normalized_keys = tuple((str(suite), int(task_id)) for suite, task_id in task_keys)
    if set(normalized_keys) - set(target_by_key):
        raise WriterModelError("AS-Writer evaluation task is outside target40")
    task_roles = {key: str(target_by_key[key]["split_role"]) for key in normalized_keys}
    mapping = _task_video_mapping(normalized_keys, task_roles, video_condition)
    needed_task_ids = tuple(
        sorted(
            {
                int(row["language_global_task_id"])
                for row in mapping
            }
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
        video_data_root, config, needed_task_ids, verify_hashes=False
    )
    training_video = training.get("video_data", {})
    if training_video.get("dataset") != video_data.get("dataset"):
        raise WriterModelError("AS-Writer checkpoint and video data disagree")
    lora = load_pi05_lora_contract(
        REPO_ROOT / str(config["authorities"]["lora_contract"]["path"])
    )
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
        lora_contract_sha256=canonical_contract_sha256(lora),
        mapping=mapping,
        task_keys=normalized_keys,
        source=source,
        video_condition=video_condition,
        video_seed=video_seed,
        video_sampling_mode=video_sampling_mode,
        forbidden_inputs=config["information_wall"]["writer_forbidden_inputs"],
    )
