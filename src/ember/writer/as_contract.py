"""Authorities, data wall, and launch contracts for PI05 AS-Writer."""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import socket
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.lora import canonical_contract_sha256
from ember.pi05_eval_contract import git_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import (
    DistributedContext,
    canonical_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.pi05_source_contract import append_jsonl
from ember.writer.data import FunctionalQueryDataset
from ember.writer.feature_cache import (
    PI05_FEATURE_CACHE_MANIFEST_SCHEMA,
    PI05_TASK_FEATURE_CACHE_SCHEMA,
    FeatureCacheTask,
    load_pi05_feature_cache_config,
    load_pi05_feature_tasks,
)
from ember.writer.model import CompleteLoRAWriter, WriterModelError


REPO_ROOT = Path(__file__).resolve().parents[3]
AS_WRITER_CONFIG_SCHEMA = "ember_pi05_as_writer_v2"
AS_WRITER_LAUNCH_SCHEMA = "ember_pi05_as_writer_launch_v2"
AS_WRITER_STAGES = ("development", "final")
_CHECKPOINT_NAME = re.compile(r"step_([0-9]{8})")


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    return REPO_ROOT / str(config["authorities"][name]["path"])


def writer_stage(config: Mapping[str, Any]) -> str:
    """Return the sealed data stage, preserving old development artifacts."""

    stage = str(config.get("sealed_stage", "development"))
    if stage not in AS_WRITER_STAGES:
        raise WriterModelError("unsupported PI05 AS-Writer stage")
    return stage


def writer_split_roles(config: Mapping[str, Any]) -> tuple[str, ...]:
    if writer_stage(config) == "development":
        return ("train",)
    return ("train", "validation")


def _validate_authorities(config: Mapping[str, Any]) -> None:
    authorities = config.get("authorities", {})
    required = {
        "target_data_manifest",
        "evaluation_config",
        "feature_cache_config",
        "lora_contract",
        "source_base_config",
        "tokenizer_manifest",
    }
    if set(authorities) != required:
        raise WriterModelError("AS-Writer authority set changed")
    for name, authority in authorities.items():
        artifact = REPO_ROOT / str(authority.get("path", ""))
        if not artifact.is_file() or sha256_file(artifact) != authority.get("sha256"):
            raise WriterModelError(f"sealed AS-Writer authority changed: {name}")


def _validate_protocol(config: Mapping[str, Any]) -> None:
    target = read_json(authority_path(config, "target_data_manifest"))
    roles = target.get("summary", {}).get("roles", {})
    if (
        target.get("schema_version") != "ember_pi05_target_data_manifest_v1"
        or int(target.get("summary", {}).get("tasks", -1)) != 40
        or int(target.get("summary", {}).get("episodes", -1)) != 2000
        or {name: len(roles.get(name, [])) for name in ("train", "validation", "test")}
        != {"train": 24, "validation": 8, "test": 8}
    ):
        raise WriterModelError("AS-Writer target-data authority is not sealed 24/8/8")
    feature = load_pi05_feature_cache_config(
        authority_path(config, "feature_cache_config"), REPO_ROOT
    )
    linked = (
        "target_data_manifest",
        "evaluation_config",
        "lora_contract",
        "tokenizer_manifest",
    )
    if any(feature["authorities"][name] != config["authorities"][name] for name in linked):
        raise WriterModelError("AS-Writer and feature-cache authorities disagree")
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    if lora.source_base_config_sha256 != config["authorities"]["source_base_config"]["sha256"]:
        raise WriterModelError("AS-Writer LoRA and source-base authorities disagree")
    writer = config.get("writer", {})
    if (
        writer.get("vision_feature_dim") != feature["features"]["vision_feature_dim"]
        or writer.get("vision_spatial_tokens")
        != feature["features"]["vision_spatial_tokens"]
        or writer.get("language_feature_dim") != feature["features"]["language_feature_dim"]
        or writer.get("generated_adapter") != "complete_pi05_task_specific_lora"
    ):
        raise WriterModelError("AS-Writer architecture and PI05 features disagree")


def _validate_information_wall(config: Mapping[str, Any]) -> None:
    common = {
        "writer_input": "pure task language plus exactly one action-hidden teacher video",
        "writer_forbidden_inputs": [
            "action",
            "proprio",
            "reward",
            "terminal",
            "task_id",
            "filename",
            "policy_outcome",
        ],
        "action_owner": "frozen functional behavior loss only",
        "test_actions_read": 0,
        "test_video_values_read": 0,
    }
    if writer_stage(config) == "development":
        expected = {
            **common,
            "development_action_split_roles": ["train"],
            "development_video_split_roles": ["train"],
            "validation_actions_read": 0,
        }
    else:
        expected = {
            **common,
            "final_action_split_roles": ["train", "validation"],
            "final_video_split_roles": ["train", "validation"],
        }
    if config.get("information_wall") != expected:
        raise WriterModelError("AS-Writer information wall changed")
    data = config.get("data", {})
    required = {
        "task_count": 24 if writer_stage(config) == "development" else 32,
        "demo_indices": [0, 49],
        "episodes_per_task": 50,
        "teacher_video_sampling": "independent deterministic per-task no-replacement cycles",
        "action_query_sampling": "task-balanced deterministic no-replacement episode cycles",
        "video_action_pairing": (
            "positive video/action independent within task; contrast video from sealed paired train task"
            if writer_stage(config) == "development"
            else "positive video/action independent within task; contrast video from sealed paired final-source task"
        ),
    }
    if any(data.get(name) != value for name, value in required.items()):
        raise WriterModelError("AS-Writer sampling contract changed")


def _validate_conditioning_training(config: Mapping[str, Any]) -> None:
    value = config.get("conditioning_training", {})
    feature = load_pi05_feature_cache_config(
        authority_path(config, "feature_cache_config"), REPO_ROOT
    )
    pairs = value.get("video_task_pairs", [])
    flattened = [int(task_id) for pair in pairs for task_id in pair]
    target = read_json(authority_path(config, "target_data_manifest"))
    source_ids = sorted(
        int(task_id)
        for role in writer_split_roles(config)
        for task_id in target["summary"]["roles"][role]
    )
    weights = (
        value.get("normal_loss_weight"),
        value.get("contrast_correct_loss_weight"),
        value.get("matching_loss_weight"),
        value.get("matching_temperature"),
    )
    if (
        value.get("method")
        != "normal_full_language_contrast_generic_language_contrast_cycle"
        or value.get("step_cycle")
        != ["normal", "full_language_contrast", "generic_language_contrast"]
        or value.get("generic_writer_language")
        != feature["features"]["generic_writer_language"]
        or value.get("generic_writer_language_owner")
        != "normal_pi05_pure_language_tokenizer_and_embedding_cache"
        or value.get("policy_language_contract")
        != "correct_action_query_task_language_on_every_branch"
        or value.get("contrast_backend")
        != "paired_sequential_half_batch_with_shared_policy_rng"
        or value.get("contrast_query_fraction") != 0.5
        or not isinstance(pairs, list)
        or any(not isinstance(pair, list) or len(pair) != 2 for pair in pairs)
        or sorted(flattened) != source_ids
        or len(set(flattened)) != len(flattened)
        or any(not isinstance(weight, (int, float)) or weight <= 0 for weight in weights)
        or not isinstance(value.get("matching_margin"), (int, float))
        or value["matching_margin"] < 0
    ):
        raise WriterModelError("AS-Writer video-forced training contract changed")


def load_writer_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != AS_WRITER_CONFIG_SCHEMA:
        raise WriterModelError("unsupported PI05 AS-Writer config schema")
    writer_stage(config)
    _validate_authorities(config)
    _validate_protocol(config)
    _validate_information_wall(config)
    _validate_conditioning_training(config)
    return config


def parse_checkpoint_steps(value: str | Sequence[int], total_steps: int) -> tuple[int, ...]:
    raw = value.split(",") if isinstance(value, str) else value
    try:
        result = tuple(sorted({int(item) for item in raw}))
    except (TypeError, ValueError) as error:
        raise WriterModelError("invalid AS-Writer checkpoint steps") from error
    if not result or result[-1] != total_steps or any(step <= 0 for step in result):
        raise WriterModelError("AS-Writer checkpoints must end at total_steps")
    return result


def resume_step(checkpoint: Path | None) -> int:
    if checkpoint is None:
        return 0
    match = _CHECKPOINT_NAME.fullmatch(checkpoint.name)
    if match is None:
        raise WriterModelError("AS-Writer resume path is not a step checkpoint")
    return int(match.group(1))


def resolve_runtime(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> tuple[int, int, tuple[int, ...]]:
    if args.mode == "formal" and config["formal_run"].get("status") != "sealed":
        raise WriterModelError("formal AS-Writer config is pending a real profile")
    source = config["formal_run"] if args.mode == "formal" else config["profile_defaults"]
    total_steps = args.total_steps or int(source["total_steps"])
    batch_size = args.batch_size or int(source["per_rank_batch_size"])
    checkpoint_steps = parse_checkpoint_steps(
        args.checkpoint_steps or source["checkpoint_steps"], total_steps
    )
    default_stop = int(source.get("selected_stop_step", total_steps))
    stop_step = args.stop_after_step or default_stop
    if min(total_steps, batch_size, stop_step) <= 0 or stop_step > total_steps:
        raise WriterModelError("invalid AS-Writer runtime request")
    if context.world_size != 8:
        raise WriterModelError("AS-Writer training requires exactly eight symmetric ranks")
    if args.mode == "formal":
        formal = config["formal_run"]
        expected = (
            "sealed",
            int(formal["expected_world_size"]),
            int(formal["total_steps"]),
            int(formal["per_rank_batch_size"]),
            tuple(int(value) for value in formal["checkpoint_steps"]),
        )
        observed = (
            formal.get("status"),
            context.world_size,
            total_steps,
            batch_size,
            checkpoint_steps,
        )
        if observed != expected or stop_step != default_stop:
            raise WriterModelError("formal AS-Writer launch differs from its sealed profile")
        state = git_state(REPO_ROOT)
        if state["dirty_paths"]:
            raise WriterModelError("formal AS-Writer launch requires a clean worktree")
        if args.resume is None and state["commit"] != state["origin_main"]:
            raise WriterModelError("fresh formal AS-Writer launch must be pushed")
        if context.numa_node is None or not context.cpu_affinity:
            raise WriterModelError("formal AS-Writer launch requires GPU-local NUMA binding")
        if args.skip_data_sha:
            raise WriterModelError("formal AS-Writer launch must verify every train HDF5")
    args.stop_after_step = stop_step
    return total_steps, batch_size, checkpoint_steps


def _broadcast_validation(
    context: DistributedContext, operation: Any
) -> dict[str, Any]:
    payload: list[Any] = [None]
    if context.is_main:
        try:
            payload[0] = operation()
        except Exception as error:
            payload[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=context.device)
    if payload[0].get("error"):
        raise WriterModelError(payload[0]["error"])
    return payload[0]


def _validate_target_files(
    tasks: Sequence[FeatureCacheTask], verify_hashes: bool
) -> dict[str, Any]:
    for task in tasks:
        path = task.authority.path
        if not path.is_file() or path.stat().st_size != task.authority.expected_bytes:
            raise WriterModelError(f"AS-Writer train HDF5 size changed: {task.task_id}")
        if verify_hashes and sha256_file(path) != task.expected_hdf5_sha256:
            raise WriterModelError(f"AS-Writer train HDF5 hash changed: {task.task_id}")
    return {
        "tasks_checked": len(tasks),
        "bytes_checked": sum(task.authority.expected_bytes for task in tasks),
        "full_sha256_verified": verify_hashes,
        "hdf5_identity_sha256": canonical_hash(
            [
                [task.task_id, task.authority.expected_bytes, task.expected_hdf5_sha256]
                for task in tasks
            ]
        ),
    }


def load_training_data(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> tuple[FunctionalQueryDataset, tuple[FeatureCacheTask, ...], dict[str, Any]]:
    cache_config = load_pi05_feature_cache_config(
        authority_path(config, "feature_cache_config"), REPO_ROOT
    )
    development = load_pi05_feature_tasks(
        cache_config, REPO_ROOT, args.data_root.resolve(), role="development"
    )
    roles = set(writer_split_roles(config))
    tasks = tuple(task for task in development if task.split_role in roles)
    suite_counts: dict[str, int] = {}
    for task in tasks:
        suite_counts[str(task.suite)] = suite_counts.get(str(task.suite), 0) + 1
    per_suite = 6 if writer_stage(config) == "development" else 8
    if (
        len(tasks) != int(config["data"]["task_count"])
        or sorted(suite_counts.values()) != [per_suite] * 4
    ):
        raise WriterModelError("AS-Writer action training is not its sealed source role")
    validation = _broadcast_validation(
        context, lambda: _validate_target_files(tasks, not args.skip_data_sha)
    )
    first_demo, last_demo = map(int, config["data"]["demo_indices"])
    dataset = FunctionalQueryDataset(
        [replace(task.authority, expected_sha256=None) for task in tasks],
        demo_indices=range(first_demo, last_demo + 1),
        action_chunk_size=int(config["data"]["action_chunk_size"]),
        max_open_files_per_worker=int(config["data"]["max_open_files_per_worker"]),
    )
    return dataset, tasks, validation


def inspect_feature_cache(
    root: Path,
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    train_task_ids: Sequence[int],
) -> dict[str, Any]:
    contract_path = root / "run_contract.json"
    manifest_path = root / "cache_manifest.json"
    contract = read_json(contract_path)
    manifest = read_json(manifest_path)
    contract_payload = dict(contract)
    contract_digest = contract_payload.pop("contract_sha256", None)
    manifest_payload = dict(manifest)
    manifest_digest = manifest_payload.pop("canonical_payload_sha256", None)
    records = manifest.get("task_records", [])
    record_ids = tuple(sorted(int(record["task_id"]) for record in records))
    source_keys = (
        "source_run_contract_sha256",
        "checkpoint_manifest_sha256",
        "optimizer_step",
        "source_run_summary_sha256",
        "source_training_commit",
        "source_base_config_sha256",
        "source_authority_hashes",
        "model_files",
    )
    if (
        contract.get("schema_version") != "ember_pi05_writer_feature_cache_launch_v2"
        or contract.get("mode") != "formal"
        or contract.get("role") != "development"
        or canonical_hash(contract_payload) != contract_digest
        or contract.get("config_sha256")
        != config["authorities"]["feature_cache_config"]["sha256"]
        or contract.get("test_video_values_read") != 0
        or any(contract.get("source", {}).get(key) != source.get(key) for key in source_keys)
        or manifest.get("schema_version") != PI05_FEATURE_CACHE_MANIFEST_SCHEMA
        or canonical_hash(manifest_payload) != manifest_digest
        or manifest.get("contract_sha256") != contract_digest
        or manifest.get("extraction_sha256") != contract.get("extraction_sha256")
        or int(manifest.get("task_count", -1)) != 32
        or int(manifest.get("episode_count", -1)) != 1600
        or len(records) != 32
        or len(set(record_ids)) != 32
        or not set(train_task_ids) <= set(record_ids)
        or any(
            record.get("schema_version") != PI05_TASK_FEATURE_CACHE_SCHEMA
            or record.get("extraction_sha256") != manifest.get("extraction_sha256")
            for record in records
        )
    ):
        raise WriterModelError("formal PI05 Writer feature cache changed")
    return {
        "root": str(root.resolve()),
        "run_contract_file_sha256": sha256_file(contract_path),
        "run_contract_sha256": contract_digest,
        "cache_manifest_file_sha256": sha256_file(manifest_path),
        "cache_manifest_payload_sha256": manifest_digest,
        "extraction_sha256": manifest["extraction_sha256"],
        "task_count": 32,
        "episode_count": 1600,
        "frame_count": int(manifest["frame_count"]),
        "test_video_values_read": 0,
    }


def writer_trainable_contract(
    writer: CompleteLoRAWriter, policy: torch.nn.Module, lora: Any
) -> dict[str, Any]:
    names = sorted(name for name, value in writer.named_parameters() if value.requires_grad)
    if not names or any(parameter.requires_grad for parameter in policy.parameters()):
        raise WriterModelError("AS-Writer freeze boundary changed")
    return {
        "object": "shared_action_supervised_writer_only",
        "parameter_count": sum(value.numel() for value in writer.parameters()),
        "parameter_name_count": len(names),
        "parameter_names_sha256": canonical_hash(names),
        "generated_lora_parameter_count": lora.parameter_count,
        "generated_lora_tensor_count": lora.state_tensor_count,
        "lora_contract_sha256": canonical_contract_sha256(lora),
        "source_policy_trainable_parameter_count": 0,
    }


def _software_versions() -> dict[str, Any]:
    packages = ("lerobot", "transformers", "peft", "safetensors", "h5py")
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def build_contract(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    source: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    cache: Mapping[str, Any],
    data_validation: Mapping[str, Any],
    task_ids: Sequence[int],
    trainable: Mapping[str, Any],
    total_steps: int,
    batch_size: int,
    batch_cycle: Sequence[int],
    checkpoint_steps: Sequence[int],
) -> dict[str, Any]:
    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "device": str(context.device),
        "numa_node": context.numa_node,
        "cpu_affinity": list(context.cpu_affinity or ()),
    }
    topology: list[Any] = [None] * context.world_size
    if context.world_size > 1:
        dist.all_gather_object(topology, local)
    else:
        topology[0] = local
    return {
        "schema_version": AS_WRITER_LAUNCH_SCHEMA,
        "mode": args.mode,
        "stage": writer_stage(config),
        "git": {key: value for key, value in git_state(REPO_ROOT).items() if key in {"branch", "commit"}},
        "config_sha256": sha256_file(args.config.resolve()),
        "authorities": dict(config["authorities"]),
        "source": dict(source),
        "tokenizer": dict(tokenizer),
        "feature_cache": dict(cache),
        "target_action_data_validation": dict(data_validation),
        "information_wall": dict(config["information_wall"]),
        "writer": dict(config["writer"]),
        "data": dict(config["data"]),
        "conditioning_training": dict(config["conditioning_training"]),
        "optimization": dict(config["optimization"]),
        "task_ids": list(task_ids),
        "runtime": {
            "world_size": context.world_size,
            "one_policy_cuda_process_per_rank": True,
            "gpu0_extra_cuda_roles": 0,
            "ddp_object": "shared_writer_only",
            "per_rank_policy_sample_batch_size": batch_size,
            "per_rank_unique_action_query_cycle": list(batch_cycle),
            "global_policy_samples_per_step": context.world_size * batch_size,
            "writer_conditions_per_rank_cycle": [1, 2, 2],
            "policy_forward_calls_per_rank_cycle": [1, 2, 2],
            "teacher_videos_per_writer_invocation": 1,
            "total_steps": total_steps,
            "selected_stop_step": args.stop_after_step,
            "checkpoint_steps": list(checkpoint_steps),
            "num_workers_per_rank": args.num_workers,
            "rank_topology": topology,
        },
        "trainable": dict(trainable),
        "software": _software_versions(),
    }


def publish_contract(
    args: argparse.Namespace,
    context: DistributedContext,
    contract: Mapping[str, Any],
    contract_sha256: str,
) -> None:
    def operation() -> dict[str, bool]:
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
            raise WriterModelError(f"AS-Writer output directory is not empty: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        contract_path = args.output_dir / "run_contract.json"
        if args.resume is not None:
            if not contract_path.is_file() or canonical_hash(read_json(contract_path)) != contract_sha256:
                raise WriterModelError("AS-Writer resume launch contract changed")
        else:
            write_json_atomic(contract_path, dict(contract))
        append_jsonl(
            args.output_dir / "invocations.jsonl",
            {
                "argv": sys.argv,
                "host": socket.gethostname(),
                "resume": str(args.resume) if args.resume else None,
                "started_unix": time.time(),
            },
        )
        write_json_atomic(
            args.output_dir / "runtime_paths.json",
            {
                "source_run": str(args.source_run.resolve()),
                "source_checkpoint": str(args.checkpoint.resolve()),
                "feature_cache": str(args.feature_cache.resolve()),
                "target_data_root": str(args.data_root.resolve()),
                "tokenizer": str(args.tokenizer_path.resolve()),
            },
        )
        return {"ok": True}

    _broadcast_validation(context, operation)
