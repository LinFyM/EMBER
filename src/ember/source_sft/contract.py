"""Authorities and launch contract for the shared PI05 Source-SFT LoRA."""

from __future__ import annotations

import argparse
import importlib.metadata
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.lora import canonical_contract_sha256, task_lora_state_dict
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
from ember.writer.data import FunctionalQueryDataset, WriterTaskAuthority
from ember.writer.model import WriterModelError


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_SFT_CONFIG_SCHEMA = "ember_pi05_source_sft_v1"
SOURCE_SFT_LAUNCH_SCHEMA = "ember_pi05_source_sft_launch_v2"
SOURCE_SFT_STAGES = ("development", "final")


class Pi05SourceSFTError(WriterModelError):
    """Raised when Source-SFT crosses a sealed data or launch boundary."""


@dataclass(frozen=True)
class SourceSFTTask:
    global_task_id: int
    suite: str
    task_id: int
    split_role: str
    language: str
    authority: WriterTaskAuthority
    expected_hdf5_sha256: str


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    return REPO_ROOT / str(config["authorities"][name]["path"])


def _validate_authorities(config: Mapping[str, Any]) -> None:
    required = {
        "target_data_manifest",
        "evaluation_config",
        "lora_contract",
        "source_base_config",
        "tokenizer_manifest",
    }
    authorities = config.get("authorities", {})
    if set(authorities) != required:
        raise Pi05SourceSFTError("Source-SFT authority set changed")
    for name, record in authorities.items():
        path = REPO_ROOT / str(record.get("path", ""))
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise Pi05SourceSFTError(f"sealed Source-SFT authority changed: {name}")


def _validate_protocol(config: Mapping[str, Any]) -> None:
    manifest = read_json(authority_path(config, "target_data_manifest"))
    roles = manifest.get("summary", {}).get("roles", {})
    if (
        manifest.get("schema_version") != "ember_pi05_target_data_manifest_v1"
        or int(manifest.get("summary", {}).get("tasks", -1)) != 40
        or int(manifest.get("summary", {}).get("episodes", -1)) != 2000
        or {role: len(roles.get(role, ())) for role in ("train", "validation", "test")}
        != {"train": 24, "validation": 8, "test": 8}
    ):
        raise Pi05SourceSFTError("Source-SFT target-data authority is not sealed 24/8/8")
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    if lora.source_base_config_sha256 != config["authorities"]["source_base_config"]["sha256"]:
        raise Pi05SourceSFTError("Source-SFT LoRA and source-base authorities disagree")
    expected_stages = {
        "development": (["train"], 24, 1200),
        "final": (["train", "validation"], 32, 1600),
    }
    if set(config.get("stages", {})) != set(expected_stages):
        raise Pi05SourceSFTError("Source-SFT stage set changed")
    for stage, expected in expected_stages.items():
        value = config["stages"][stage]
        observed = (
            value.get("action_split_roles"),
            int(value.get("task_count", -1)),
            int(value.get("available_action_episodes", -1)),
        )
        if observed != expected:
            raise Pi05SourceSFTError(f"Source-SFT {stage} data wall changed")


def _validate_information_wall(config: Mapping[str, Any]) -> None:
    expected = {
        "policy_input": "correct task language plus current policy observation only",
        "forbidden_inputs": [
            "teacher_video",
            "reward",
            "terminal",
            "task_id",
            "filename",
            "policy_outcome",
        ],
        "development_action_split_roles": ["train"],
        "final_action_split_roles": ["train", "validation"],
        "test_actions_read": 0,
        "test_video_values_read": 0,
        "held_evaluation_adaptation": "none",
    }
    adapter = config.get("adapter", {})
    if config.get("information_wall") != expected:
        raise Pi05SourceSFTError("Source-SFT information wall changed")
    if (
        adapter.get("kind") != "one_shared_multitask_pi05_lora"
        or adapter.get("stacked_shared_source_adapter") is not False
        or adapter.get("per_task_adapter") is not False
    ):
        raise Pi05SourceSFTError("Source-SFT shared-adapter contract changed")


def load_source_sft_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != SOURCE_SFT_CONFIG_SCHEMA:
        raise Pi05SourceSFTError("unsupported PI05 Source-SFT config schema")
    _validate_authorities(config)
    _validate_protocol(config)
    _validate_information_wall(config)
    if config.get("sealed_stage") not in SOURCE_SFT_STAGES:
        raise Pi05SourceSFTError("Source-SFT config does not seal exactly one stage")
    return config


def parse_checkpoint_steps(value: str | Sequence[int], total_steps: int) -> tuple[int, ...]:
    raw = value.split(",") if isinstance(value, str) else value
    try:
        steps = tuple(sorted({int(item) for item in raw}))
    except (TypeError, ValueError) as error:
        raise Pi05SourceSFTError("invalid Source-SFT checkpoint steps") from error
    if not steps or steps[-1] != total_steps or any(step <= 0 for step in steps):
        raise Pi05SourceSFTError("Source-SFT checkpoints must be positive and end at total_steps")
    return steps


def resolve_runtime(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> tuple[int, int, tuple[int, ...]]:
    if args.stage not in SOURCE_SFT_STAGES:
        raise Pi05SourceSFTError("unknown Source-SFT stage")
    if args.stage != config.get("sealed_stage"):
        raise Pi05SourceSFTError(
            "Source-SFT stage needs its own immutable sealed config"
        )
    formal = config["stages"][args.stage]["formal_run"]
    if args.mode == "formal" and formal.get("status") != "sealed":
        raise Pi05SourceSFTError(f"formal Source-SFT {args.stage} config is not sealed")
    if args.stage == "final" and formal.get("status") != "sealed":
        raise Pi05SourceSFTError("final Source-SFT cannot read validation actions before selection")
    source = formal if args.mode == "formal" else config["profile_defaults"]
    total_steps = args.total_steps or int(source["total_steps"])
    batch_size = args.batch_size or int(source["per_rank_batch_size"])
    checkpoint_steps = parse_checkpoint_steps(
        args.checkpoint_steps or source["checkpoint_steps"], total_steps
    )
    default_stop = int(source.get("selected_stop_step", total_steps))
    stop_step = args.stop_after_step or default_stop
    if min(total_steps, batch_size, stop_step) <= 0 or stop_step > total_steps:
        raise Pi05SourceSFTError("invalid Source-SFT runtime request")
    expected_world_size = int(source.get("expected_world_size", 8))
    if context.world_size != expected_world_size:
        raise Pi05SourceSFTError(
            "Source-SFT training requires exactly "
            f"{expected_world_size} symmetric ranks"
        )
    if args.mode == "formal":
        expected = (
            "sealed",
            int(formal["expected_world_size"]),
            int(formal["total_steps"]),
            int(formal["per_rank_batch_size"]),
            tuple(int(step) for step in formal["checkpoint_steps"]),
        )
        observed = (
            formal.get("status"),
            context.world_size,
            total_steps,
            batch_size,
            checkpoint_steps,
        )
        stage_stops = tuple(
            int(value) for value in formal.get("stage_stop_steps", [default_stop])
        )
        if (
            observed != expected
            or not stage_stops
            or any(value not in checkpoint_steps for value in stage_stops)
            or default_stop not in stage_stops
            or stop_step not in stage_stops
        ):
            raise Pi05SourceSFTError("formal Source-SFT launch differs from its sealed profile")
        state = git_state(REPO_ROOT)
        if state["dirty_paths"]:
            raise Pi05SourceSFTError("formal Source-SFT launch requires a clean worktree")
        if args.resume is None and state["commit"] != state["origin_main"]:
            raise Pi05SourceSFTError("fresh formal Source-SFT launch must be pushed")
        if context.numa_node is None or not context.cpu_affinity:
            raise Pi05SourceSFTError("formal Source-SFT launch requires GPU-local NUMA binding")
    args.stop_after_step = stop_step
    return total_steps, batch_size, checkpoint_steps


def _broadcast(context: DistributedContext, operation: Any) -> dict[str, Any]:
    payload: list[Any] = [None]
    if context.is_main:
        try:
            payload[0] = operation()
        except Exception as error:
            payload[0] = {"error": repr(error)}
    if context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=context.device)
    if payload[0].get("error"):
        raise Pi05SourceSFTError(payload[0]["error"])
    return payload[0]


def _target_tasks(config: Mapping[str, Any], data_root: Path, stage: str) -> tuple[SourceSFTTask, ...]:
    manifest = read_json(authority_path(config, "target_data_manifest"))
    roles = set(config["stages"][stage]["action_split_roles"])
    tasks = []
    for row in manifest["tasks"]:
        if row["split_role"] not in roles:
            continue
        hdf5 = row["hdf5"]
        global_task_id = int(row["global_task_id"])
        authority = WriterTaskAuthority(
            task_id=global_task_id,
            language=str(row["language"]),
            path=data_root / str(hdf5["relative_path"]),
            expected_bytes=int(hdf5["bytes"]),
            expected_sha256=None,
        )
        tasks.append(
            SourceSFTTask(
                global_task_id=global_task_id,
                suite=str(row["suite"]),
                task_id=int(row["task_id"]),
                split_role=str(row["split_role"]),
                language=str(row["language"]),
                authority=authority,
                expected_hdf5_sha256=str(hdf5["sha256"]),
            )
        )
    tasks.sort(key=lambda task: task.global_task_id)
    if len(tasks) != int(config["stages"][stage]["task_count"]):
        raise Pi05SourceSFTError("Source-SFT selected the wrong task count")
    if any(task.split_role == "test" for task in tasks):
        raise Pi05SourceSFTError("Source-SFT attempted to read test actions")
    return tuple(tasks)


def _validate_task_files(tasks: Sequence[SourceSFTTask], verify_hashes: bool) -> dict[str, Any]:
    for task in tasks:
        path = task.authority.path
        if not path.is_file() or path.stat().st_size != task.authority.expected_bytes:
            raise Pi05SourceSFTError(f"Source-SFT HDF5 size changed: {task.global_task_id}")
        if verify_hashes and sha256_file(path) != task.expected_hdf5_sha256:
            raise Pi05SourceSFTError(f"Source-SFT HDF5 hash changed: {task.global_task_id}")
    return {
        "tasks_checked": len(tasks),
        "bytes_checked": sum(task.authority.expected_bytes for task in tasks),
        "full_sha256_verified": verify_hashes,
        "hdf5_identity_sha256": canonical_hash(
            [
                [task.global_task_id, task.authority.expected_bytes, task.expected_hdf5_sha256]
                for task in tasks
            ]
        ),
    }


def load_training_data(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
) -> tuple[FunctionalQueryDataset, tuple[SourceSFTTask, ...], dict[str, Any]]:
    tasks = _target_tasks(config, args.data_root.resolve(), args.stage)
    validation = _broadcast(
        context, lambda: _validate_task_files(tasks, not args.skip_data_sha)
    )
    first_demo, last_demo = map(int, config["data"]["demo_indices"])
    dataset = FunctionalQueryDataset(
        [task.authority for task in tasks],
        demo_indices=range(first_demo, last_demo + 1),
        action_chunk_size=int(config["data"]["action_chunk_size"]),
        max_open_files_per_worker=int(config["data"]["max_open_files_per_worker"]),
    )
    return dataset, tasks, validation


def trainable_contract(policy: torch.nn.Module, lora: Any) -> dict[str, Any]:
    state = task_lora_state_dict(policy)
    names = sorted(name for name, value in policy.named_parameters() if value.requires_grad)
    if set(names) != set(state):
        raise Pi05SourceSFTError("Source-SFT left trainable parameters outside shared LoRA")
    count = sum(value.numel() for value in state.values())
    if count != lora.parameter_count:
        raise Pi05SourceSFTError("Source-SFT LoRA parameter count changed")
    return {
        "object": "one_shared_multitask_pi05_lora_only",
        "parameter_count": count,
        "parameter_name_count": len(names),
        "parameter_names_sha256": canonical_hash(names),
        "lora_contract_sha256": canonical_contract_sha256(lora),
        "frozen_source_policy_parameter_count": sum(
            value.numel() for value in policy.parameters() if not value.requires_grad
        ),
        "per_task_adapters": 0,
        "stacked_shared_source_adapters": 0,
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


def _contract_stop_step(
    args: argparse.Namespace, config: Mapping[str, Any], total_steps: int
) -> int:
    if args.mode == "formal":
        return int(
            config["stages"][args.stage]["formal_run"].get(
                "selected_stop_step", total_steps
            )
        )
    return int(args.stop_after_step)


def build_contract(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    context: DistributedContext,
    source: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    data_validation: Mapping[str, Any],
    tasks: Sequence[SourceSFTTask],
    trainable: Mapping[str, Any],
    total_steps: int,
    batch_size: int,
    checkpoint_steps: Sequence[int],
) -> dict[str, Any]:
    contract_stop_step = _contract_stop_step(args, config, total_steps)
    task_count = len(tasks)
    recipe = config.get("training_recipe", {})
    tasks_per_rank = int(recipe.get("tasks_per_rank_per_update", -1))
    global_tasks_per_update = int(recipe.get("global_tasks_per_update", -1))
    updates_per_cycle = int(
        recipe.get("updates_per_complete_task_cycle", -1)
    )
    if (
        task_count <= 0
        or tasks_per_rank <= 1
        or batch_size % tasks_per_rank
        or context.world_size * tasks_per_rank != global_tasks_per_update
        or global_tasks_per_update * updates_per_cycle != task_count
    ):
        raise Pi05SourceSFTError(
            "Source-SFT cyclic mixed-task topology is inconsistent"
        )
    samples_per_task_per_visit = batch_size // tasks_per_rank
    if (
        total_steps % updates_per_cycle
        or contract_stop_step % updates_per_cycle
        or any(int(step) % updates_per_cycle for step in checkpoint_steps)
    ):
        raise Pi05SourceSFTError(
            "Source-SFT checkpoints and stops must be complete task-cycle boundaries"
        )
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
    stage = config["stages"][args.stage]
    return {
        "schema_version": SOURCE_SFT_LAUNCH_SCHEMA,
        "mode": args.mode,
        "stage": args.stage,
        "git": {
            key: value
            for key, value in git_state(REPO_ROOT).items()
            if key in {"branch", "commit"}
        },
        "config_sha256": sha256_file(args.config.resolve()),
        "authorities": dict(config["authorities"]),
        "source": dict(source),
        "tokenizer": dict(tokenizer),
        "target_action_data_validation": dict(data_validation),
        "information_wall": dict(config["information_wall"]),
        "adapter": dict(config["adapter"]),
        "training_recipe": dict(config.get("training_recipe", {})),
        "data": dict(config["data"]),
        "optimization": dict(config["optimization"]),
        "stage_contract": dict(stage),
        "tasks": [
            {
                "global_task_id": task.global_task_id,
                "suite": task.suite,
                "task_id": task.task_id,
                "split_role": task.split_role,
                "language": task.language,
                "hdf5_bytes": task.authority.expected_bytes,
                "hdf5_sha256": task.expected_hdf5_sha256,
            }
            for task in tasks
        ],
        "runtime": {
            "world_size": context.world_size,
            "one_policy_cuda_process_per_rank": True,
            "gpu0_extra_cuda_roles": 0,
            "ddp_object": "source_policy_with_shared_lora_only_trainable",
            "per_rank_batch_size": batch_size,
            "effective_global_batch_size": context.world_size * batch_size,
            "physical_batch_task_mixed": True,
            "tasks_per_physical_batch": tasks_per_rank,
            "global_tasks_per_update": global_tasks_per_update,
            "updates_per_complete_task_cycle": updates_per_cycle,
            "samples_per_task_per_visit": samples_per_task_per_visit,
            "global_samples_per_selected_task_per_update": (
                samples_per_task_per_visit
            ),
            "sampler_kind": "cyclic_subset_hierarchical_mixed_v2",
            "loss_reduction": "equal_samples_per_task_then_batch_mean",
            "total_steps": total_steps,
            "selected_stop_step": contract_stop_step,
            "checkpoint_steps": list(checkpoint_steps),
            "num_workers_per_rank": args.num_workers,
            "dataloader_generator_seed_base": int(config["optimization"]["seed"])
            + 0x5F7,
            "worker_random_transforms": False,
            "worker_rng_contract": "fixed DataLoader-derived worker seeds; all sample selection is a pure deterministic sampler function",
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
            raise Pi05SourceSFTError(f"Source-SFT output directory is not empty: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        contract_path = args.output_dir / "run_contract.json"
        if args.resume is not None:
            if (
                not contract_path.is_file()
                or canonical_hash(read_json(contract_path)) != contract_sha256
                or args.resume.resolve().parent.parent != args.output_dir.resolve()
            ):
                raise Pi05SourceSFTError("Source-SFT resume ownership or contract changed")
        else:
            write_json_atomic(contract_path, dict(contract))
        append_jsonl(
            args.output_dir / "invocations.jsonl",
            {
                "argv": sys.argv,
                "contract_git": dict(contract["git"]),
                "runtime_git": {
                    key: value
                    for key, value in git_state(REPO_ROOT).items()
                    if key in {"branch", "commit"}
                },
                "contract_compatible_code_resume": bool(
                    args.resume is not None
                    and contract["git"].get("commit")
                    != git_state(REPO_ROOT).get("commit")
                ),
                "contract_selected_stop_step": int(
                    contract["runtime"]["selected_stop_step"]
                ),
                "host": socket.gethostname(),
                "monotonic_stage_extension": bool(
                    args.resume is not None
                    and int(args.stop_after_step)
                    > int(contract["runtime"]["selected_stop_step"])
                ),
                "requested_stop_after_step": int(args.stop_after_step),
                "resume": str(args.resume) if args.resume else None,
                "started_unix": time.time(),
            },
        )
        write_json_atomic(
            args.output_dir / "runtime_paths.json",
            {
                "source_run": str(args.source_run.resolve()),
                "source_checkpoint": str(args.checkpoint.resolve()),
                "target_data_root": str(args.data_root.resolve()),
                "tokenizer": str(args.tokenizer_path.resolve()),
            },
        )
        return {"ok": True}

    _broadcast(context, operation)


def reconcile_resume_contract(
    args: argparse.Namespace, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Reuse an exact run contract when only the recorded code commit advanced."""

    candidate = dict(candidate)
    if args.resume is None:
        if getattr(args, "allow_contract_compatible_code_resume", False):
            raise Pi05SourceSFTError(
                "contract-compatible code resume requires a checkpoint"
            )
        return candidate
    contract_path = args.output_dir / "run_contract.json"
    if not contract_path.is_file():
        return candidate
    existing = read_json(contract_path)
    if existing == candidate:
        return existing

    existing_runtime = dict(existing.get("runtime", {}))
    candidate_runtime = dict(candidate.get("runtime", {}))
    existing_stop = int(existing_runtime.get("selected_stop_step", -1))
    candidate_stop = int(candidate_runtime.get("selected_stop_step", -1))
    if (
        existing_stop <= 0
        or candidate_stop < existing_stop
        or candidate_stop > int(existing_runtime.get("total_steps", -1))
    ):
        raise Pi05SourceSFTError(
            "Source-SFT resume cannot shorten or exceed its sealed stage axis"
        )

    normalized = dict(candidate)
    normalized["runtime"] = {
        **candidate_runtime,
        "selected_stop_step": existing_stop,
    }
    existing_git = existing.get("git", {})
    candidate_git = candidate.get("git", {})
    if existing_git != candidate_git:
        if not getattr(args, "allow_contract_compatible_code_resume", False):
            raise Pi05SourceSFTError("Source-SFT resume launch contract changed")
        if (
            existing_git.get("branch") != candidate_git.get("branch")
            or existing_git.get("commit") == candidate_git.get("commit")
        ):
            raise Pi05SourceSFTError(
                "Source-SFT code-compatible resume did not isolate one commit change"
            )
        normalized["git"] = existing_git
    if normalized != existing:
        raise Pi05SourceSFTError(
            "Source-SFT code-compatible resume changed the scientific contract"
        )
    return existing
