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
from ember.pi05_eval_contract import git_state, git_state_is_clean_pushed_or_frozen_authority
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import (
    DistributedContext,
    canonical_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.pi05_source_contract import append_jsonl
from ember.source_sft.sampler import source_batch_sizes
from ember.source_sft.control import dynamic_control, validate_coverage_manifest
from ember.writer.data import FunctionalQueryDataset, WriterTaskAuthority
from ember.writer.errors import WriterModelError


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_SFT_CONFIG_SCHEMA = "ember_pi05_source_sft_v1"
SOURCE_SFT_LAUNCH_SCHEMA = "ember_pi05_source_sft_launch_v3"
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
        if not path.is_file():
            raise Pi05SourceSFTError(f"sealed Source-SFT authority changed: {name}")


def _validate_protocol(config: Mapping[str, Any]) -> None:
    manifest = read_json(authority_path(config, "target_data_manifest"))
    roles = manifest.get("summary", {}).get("roles", {})
    coverage = manifest.get("protocol") == "configs/libero_24_8_8_coverage_v1/protocol.json"
    if coverage:
        validate_coverage_manifest(manifest, read_json(REPO_ROOT / manifest["protocol"]))
    if (
        manifest.get("schema_version") != "ember_pi05_target_data_manifest_v1"
        or int(manifest.get("summary", {}).get("tasks", -1)) != (52 if coverage else 40)
        or int(manifest.get("summary", {}).get("episodes", -1)) != (2600 if coverage else 2000)
        or {role: len(roles.get(role, ())) for role in ("train", "validation", "test")}
        != {"train": 36 if coverage else 24, "validation": 8, "test": 8}
    ):
        raise Pi05SourceSFTError("Source-SFT target-data authority is not sealed 24/8/8")
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    source_ref = config["authorities"]["source_base_config"]
    evaluation = read_json(authority_path(config, "evaluation_config"))
    if (lora.source_base_config_sha256 != source_ref["sha256"]
            or lora.source_base_config_sha256 != sha256_file(authority_path(config, "source_base_config"))
            or evaluation["authorities"]["source_base_config"]["path"] != source_ref["path"]):
        raise Pi05SourceSFTError("Source-SFT LoRA and source-base authorities disagree")
    expected_stages = {
        "development": (["train"], 36 if coverage else 24, 1800 if coverage else 1200),
        "final": (["train", "validation"], 44 if coverage else 32, 2200 if coverage else 1600),
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
    if "validation_actions_read" in config.get("information_wall", {}):
        expected["validation_actions_read"] = 0
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
    if dynamic_control(config):
        formal = config["stages"][config["sealed_stage"]]["formal_run"]
        if (formal.get("total_steps") is not None
                or config["optimization"]["scheduler"].get("kind") != "cosine_warmup_clamped_v1"):
            raise Pi05SourceSFTError("dynamic Source-SFT needs an unbounded run and clamped LR clock")
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


def validate_active_training_recipe(config: Mapping[str, Any]) -> None:
    """Historical configs remain readable evidence, never active training defaults."""
    recipe, data = config.get("training_recipe", {}), config["data"]
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    tasks = int(config["stages"]["development"]["task_count"])
    if (tasks not in (24, 36) or config.get("sealed_stage") != "development"
            or recipe.get("kind") != "hierarchical_task_episode_chunk_mixed_v1"
            or recipe.get("logical_world_size") != 4
            or recipe.get("logical_per_rank_batch_size") != 144
            or recipe.get("global_tasks_per_update") != tasks
            or recipe.get("global_samples_per_task_per_update") != 576 // tasks
            or recipe.get("rank_task_binding") != "none"
            or lora.rank != 128 or lora.parameter_count != 10297344
            or data.get("action_start_offset") != 1
            or data.get("query_alignment") != "post_action_observation_future_control_v1"
            or data.get("episodes_per_task") != 50 or data.get("demo_indices") != [0, 49]
            or config["information_wall"].get("validation_actions_read") != 0):
        raise Pi05SourceSFTError("use the aligned rank128 all-task Source-SFT recipe; historical training is retired")


def _runtime_steps(args: argparse.Namespace, defaults: Mapping[str, Any],
                   control: Mapping[str, Any] | None) -> tuple[int, tuple[int, ...]]:
    if control and args.mode == "formal":
        if (args.total_steps is not None or args.checkpoint_steps is not None
                or not args.stop_after_step
                or args.stop_after_step % control["validation_interval"]):
            raise Pi05SourceSFTError("dynamic formal training requires a validation interval endpoint only")
        endpoint = int(args.stop_after_step)
        return endpoint, tuple(range(control["checkpoint_interval"], endpoint + 1, control["checkpoint_interval"]))
    total = args.total_steps or int(defaults["total_steps"])
    return total, parse_checkpoint_steps(args.checkpoint_steps or defaults["checkpoint_steps"], total)


def resolve_runtime(
    args: argparse.Namespace, config: Mapping[str, Any], context: DistributedContext,
) -> tuple[int, int, tuple[int, ...]]:
    if args.stage != config.get("sealed_stage"):
        raise Pi05SourceSFTError("Source-SFT stage needs its own immutable sealed config")
    validate_active_training_recipe(config)
    formal = config["stages"][args.stage]["formal_run"]
    if args.mode == "formal" and formal.get("status") != "sealed":
        raise Pi05SourceSFTError("formal Source-SFT physical profile is not sealed")
    defaults = formal if args.mode == "formal" else config["profile_defaults"]
    control = dynamic_control(config)
    physical_resume = bool(getattr(args, "allow_physical_resume", False))
    if physical_resume and (
        args.mode != "formal" or args.resume is None or not control
        or not getattr(args, "allow_contract_compatible_code_resume", False)
    ):
        raise Pi05SourceSFTError(
            "physical Source-SFT resume requires a dynamic formal checkpoint and explicit code compatibility"
        )
    if (getattr(args, "physical_packing", "contiguous") != "contiguous"
            and not physical_resume and args.mode == "formal"):
        raise Pi05SourceSFTError("formal task-striped packing requires physical resume")
    total_steps, checkpoint_steps = _runtime_steps(args, defaults, control)
    batch_size = args.batch_size or int(defaults["per_rank_batch_size"])
    stop_step = args.stop_after_step or total_steps
    recipe = config["training_recipe"]
    global_batch = int(recipe["logical_world_size"]) * int(recipe["logical_per_rank_batch_size"])
    if (min(total_steps, batch_size, stop_step) <= 0 or stop_step > total_steps
            or not 1 <= context.world_size <= 6):
        raise Pi05SourceSFTError("invalid Source-SFT runtime request")
    max_local_queries = (global_batch + context.world_size - 1) // context.world_size
    accumulation = (getattr(args, "gradient_accumulation_steps", None)
                    or (None if physical_resume else defaults.get("gradient_accumulation_steps"))
                    or (max_local_queries + batch_size - 1) // batch_size)
    source_batch_sizes(global_batch, context.world_size, batch_size, accumulation)
    if args.mode == "formal":
        expected = (int(formal["expected_world_size"]), total_steps if control else int(formal["total_steps"]),
                    int(formal["per_rank_batch_size"]), int(formal["gradient_accumulation_steps"]),
                    checkpoint_steps if control else tuple(formal["checkpoint_steps"]))
        if ((not physical_resume
             and (context.world_size, total_steps, batch_size, accumulation, checkpoint_steps) != expected)
                or (not control and stop_step not in formal["stage_stop_steps"])):
            raise Pi05SourceSFTError("formal Source-SFT launch differs from its sealed profile")
        state = git_state(REPO_ROOT)
        if not git_state_is_clean_pushed_or_frozen_authority(state):
            raise Pi05SourceSFTError("formal Source-SFT launch requires a clean pushed worktree")
        if state["branch"] != "":
            raise Pi05SourceSFTError("formal Source-SFT launch requires a detached frozen worktree")
        if context.numa_node is None or not context.cpu_affinity:
            raise Pi05SourceSFTError("formal Source-SFT launch requires GPU-local NUMA binding")
    args.stop_after_step, args.gradient_accumulation_steps = stop_step, accumulation
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


def _validate_task_files(tasks: Sequence[SourceSFTTask]) -> dict[str, Any]:
    for task in tasks:
        path = task.authority.path
        if not path.is_file() or path.stat().st_size != task.authority.expected_bytes:
            raise Pi05SourceSFTError(f"Source-SFT HDF5 size changed: {task.global_task_id}")
    return {
        "tasks_checked": len(tasks),
        "bytes_checked": sum(task.authority.expected_bytes for task in tasks),
        "full_sha256_verified": False,
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
    validate_active_training_recipe(config)
    if args.stage != "development":
        raise Pi05SourceSFTError("aligned Source-SFT reads train24 actions only")
    tasks = _target_tasks(config, args.data_root.resolve(), args.stage)
    validation = _broadcast(
        context, lambda: _validate_task_files(tasks)
    )
    first_demo, last_demo = map(int, config["data"]["demo_indices"])
    dataset = FunctionalQueryDataset(
        [task.authority for task in tasks],
        demo_indices=range(first_demo, last_demo + 1),
        action_chunk_size=int(config["data"]["action_chunk_size"]),
        action_start_offset=int(config["data"]["action_start_offset"]),
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
    if dynamic_control(config):
        return int(args.stop_after_step)
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
    recipe = config["training_recipe"]
    global_batch = int(recipe["logical_world_size"]) * int(recipe["logical_per_rank_batch_size"])
    plan = source_batch_sizes(global_batch, context.world_size, batch_size, args.gradient_accumulation_steps)
    if (len(tasks) != int(recipe["global_tasks_per_update"])
            or global_batch != len(tasks) * int(recipe["global_samples_per_task_per_update"])):
        raise Pi05SourceSFTError("Source-SFT logical task/query contract is inconsistent")
    local = {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "device": str(context.device),
        "gpu_uuid": str(torch.cuda.get_device_properties(context.local_rank).uuid),
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
        **({"training_control": dict(config["training_control"])} if dynamic_control(config) else {}),
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
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "microbatch_sizes_by_rank": [list(sizes) for sizes in plan],
            "effective_global_batch_size": global_batch,
            "global_tasks_per_update": len(tasks),
            "global_samples_per_task_per_update": int(recipe["global_samples_per_task_per_update"]),
            "logical_world_size": int(recipe["logical_world_size"]),
            "logical_per_rank_batch_size": int(recipe["logical_per_rank_batch_size"]),
            "sampler_kind": recipe["kind"],
            "loss_reduction": "query_mean_weighted_by_microbatch_size_world_over_global_batch",
            "total_steps": total_steps,
            "selected_stop_step": contract_stop_step,
            "checkpoint_steps": list(checkpoint_steps),
            "num_workers_per_rank": args.num_workers,
            "dataloader_generator_seed_base": int(config["optimization"]["seed"])
            + 0x5F7,
            "worker_random_transforms": False,
            "worker_rng_contract": "fixed DataLoader-derived worker seeds; all sample selection is a pure deterministic sampler function",
            "rank_topology": topology,
            **({"physical_packing": args.physical_packing}
               if getattr(args, "physical_packing", "contiguous") != "contiguous" else {}),
        },
        "trainable": dict(trainable),
        "software": _software_versions(),
    }


def publish_contract(
    args: argparse.Namespace,
    context: DistributedContext,
    contract: Mapping[str, Any],
    contract_sha256: str,
    requested_runtime: Mapping[str, Any] | None = None,
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
                "physical_resume": bool(getattr(args, "allow_physical_resume", False)),
                **({"requested_physical_runtime": {
                    key: requested_runtime[key] for key in (
                        "world_size", "per_rank_batch_size",
                        "gradient_accumulation_steps", "microbatch_sizes_by_rank",
                        "num_workers_per_rank", "rank_topology",
                    ) if key in requested_runtime
                } | {"physical_packing": requested_runtime.get(
                    "physical_packing", "contiguous")}}
                   if requested_runtime is not None else {}),
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
        or (not dynamic_control(existing) and candidate_stop > int(existing_runtime.get("total_steps", -1)))
    ):
        raise Pi05SourceSFTError(
            "Source-SFT resume cannot shorten or exceed its sealed stage axis"
        )

    normalized = dict(candidate)
    normalized["runtime"] = {
        **candidate_runtime,
        "selected_stop_step": existing_stop,
    }
    if dynamic_control(existing):
        control = dynamic_control(existing)
        if (candidate_stop % control["validation_interval"]
                or candidate_runtime.get("total_steps") != candidate_stop
                or candidate_runtime.get("checkpoint_steps") != list(range(
                    control["checkpoint_interval"], candidate_stop + 1, control["checkpoint_interval"]))):
            raise Pi05SourceSFTError("dynamic Source-SFT extension changed its interval schedule")
        normalized["runtime"]["total_steps"] = existing_runtime["total_steps"]
        normalized["runtime"]["checkpoint_steps"] = existing_runtime["checkpoint_steps"]
    if getattr(args, "allow_physical_resume", False):
        if args.resume is None or not dynamic_control(existing):
            raise Pi05SourceSFTError("physical resume requires a dynamic checkpoint")
        for key in (
            "world_size", "per_rank_batch_size", "gradient_accumulation_steps",
            "microbatch_sizes_by_rank", "num_workers_per_rank", "rank_topology",
            "physical_packing",
        ):
            if key in existing_runtime:
                normalized["runtime"][key] = existing_runtime[key]
            else:
                normalized["runtime"].pop(key, None)
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
    if normalized.get("tokenizer") != existing.get("tokenizer"):
        old_tokenizer = existing.get("tokenizer", {})
        new_tokenizer = normalized.get("tokenizer", {})
        old_manifest = Path(old_tokenizer.get("manifest_path", ""))
        new_manifest = Path(new_tokenizer.get("manifest_path", ""))
        if (
            not getattr(args, "allow_contract_compatible_code_resume", False)
            or {k: v for k, v in old_tokenizer.items() if k != "manifest_path"}
            != {k: v for k, v in new_tokenizer.items() if k != "manifest_path"}
            or not old_manifest.is_file()
            or not new_manifest.is_file()
            or old_manifest.read_bytes() != new_manifest.read_bytes()
        ):
            raise Pi05SourceSFTError("Source-SFT resume tokenizer authority changed")
        normalized["tokenizer"] = old_tokenizer
    if normalized != existing:
        raise Pi05SourceSFTError(
            "Source-SFT code-compatible resume changed the scientific contract"
        )
    return existing
