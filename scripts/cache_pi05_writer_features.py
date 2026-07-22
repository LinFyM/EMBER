#!/usr/bin/env python3
"""Cache frozen PI05 language and action-hidden teacher-video features."""

from __future__ import annotations

import argparse
import atexit
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.pi05_eval_contract import (
    git_state,
    inspect_source_checkpoint,
    inspect_tokenizer,
    load_evaluation_authorities,
)
from ember.pi05_processing import Pi05PureLanguageTokenizer
from ember.pi05_source_checkpoint import (
    canonical_hash,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.pi05_source_setup import initialize_distributed, load_policy
from ember.writer.data import iter_action_hidden_video_chunks
from ember.writer.feature_cache import (
    PI05_FEATURE_CACHE_MANIFEST_SCHEMA,
    PI05_TASK_FEATURE_CACHE_SCHEMA,
    FeatureCacheError,
    FeatureCacheTask,
    balanced_task_assignments,
    extraction_contract_sha256,
    load_pi05_feature_cache_config,
    load_pi05_feature_tasks,
    pool_pi05_visual_tokens,
    save_task_cache,
    select_pi05_language_tokens,
    task_cache_is_complete,
    task_cache_paths,
    write_task_record,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SCHEMA = "ember_pi05_writer_feature_cache_launch_v2"
ROLE_NAMES = ("development",)


def _destroy_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_writer_feature_cache_v2.json",
    )
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=ROLE_NAMES, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _policy_files(source: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(record["path"]): {
            "bytes": int(record["bytes"]),
            "sha256": str(record["sha256"]),
        }
        for record in source["model_files"]
    }


def runtime_versions() -> dict[str, Any]:
    packages = (
        "lerobot",
        "transformers",
        "sentencepiece",
        "safetensors",
        "h5py",
    )
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "cuda_matmul_allow_tf32": True,
    }


def build_contract(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    tasks: Sequence[FeatureCacheTask],
    assignments: Sequence[Sequence[FeatureCacheTask]],
    demo_indices: Sequence[int],
    world_size: int,
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "schema_version": RUN_SCHEMA,
        "mode": args.mode,
        "role": args.role,
        "git": {
            key: value
            for key, value in git_state(REPO_ROOT).items()
            if key in {"branch", "commit"}
        },
        "command": [
            sys.executable,
            *(argument for argument in sys.argv if argument != "--resume"),
        ],
        "config_path": str(args.config.resolve()),
        "config_sha256": config["config_sha256"],
        "authorities": {
            name: dict(value) for name, value in config["authorities"].items()
        },
        "source": dict(source),
        "policy_files": _policy_files(source),
        "tokenizer": dict(tokenizer),
        "data_root": str(args.data_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "test_video_values_read": 0,
        "task_ids": [task.task_id for task in tasks],
        "tasks": [
            {
                "global_task_id": task.task_id,
                "suite": task.suite,
                "task_id": task.local_task_id,
                "split_role": task.split_role,
                "language": task.language,
                "hdf5_sha256": task.expected_hdf5_sha256,
                "episode_lengths": list(task.episode_lengths),
            }
            for task in tasks
        ],
        "demo_indices": list(demo_indices),
        "features": dict(config["features"]),
        "runtime_versions": runtime_versions(),
        "runtime": {
            "world_size": world_size,
            "one_policy_process_per_gpu": True,
            "gpu0_extra_cuda_roles": 0,
            "assignments": [
                [task.task_id for task in rank_tasks]
                for rank_tasks in assignments
            ],
            "assigned_frames": [
                sum(
                    sum(task.episode_lengths[index] for index in demo_indices)
                    for task in rank_tasks
                )
                for rank_tasks in assignments
            ],
            "task_level_atomic_resume": True,
        },
    }
    contract["extraction_sha256"] = extraction_contract_sha256(contract)
    contract["contract_sha256"] = canonical_hash(contract)
    return contract


def _language_features(
    policy: Any,
    tokenizer: Pi05PureLanguageTokenizer,
    language: str,
    config: Mapping[str, Any],
) -> torch.Tensor:
    tokens, masks = tokenizer([language])
    with torch.inference_mode():
        embeddings = policy.model.paligemma_with_expert.embed_language_tokens(tokens)
        selected = select_pi05_language_tokens(
            embeddings,
            masks,
            expected_dim=int(config["features"]["language_feature_dim"]),
        )
    return selected.to(device="cpu", dtype=torch.bfloat16)


def _video_features(
    policy: Any,
    task: FeatureCacheTask,
    demo_indices: Sequence[int],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_tokens = int(config["features"]["vision_token_count"])
    expected_dim = int(config["features"]["vision_feature_dim"])
    chunks_by_demo: dict[int, list[torch.Tensor]] = {
        int(index): [] for index in demo_indices
    }
    next_start = {int(index): 0 for index in demo_indices}
    for demo_index, start, episode_length, frames in iter_action_hidden_video_chunks(
        task.authority,
        demo_indices,
        chunk_size=int(config["features"]["frame_batch_size_per_rank"]),
    ):
        if (
            start != next_start[demo_index]
            or episode_length != task.episode_lengths[demo_index]
        ):
            raise FeatureCacheError(
                f"PI05 video stream changed for target task {task.task_id}"
            )
        next_start[demo_index] += int(frames.shape[0])
        images = (
            torch.from_numpy(frames)
            .to(device=device, dtype=torch.float32, non_blocking=True)
            .div_(255.0)
        )
        prepared, masks = policy._preprocess_images(
            {"observation.images.base_0_rgb": images}
        )
        if (
            len(prepared) != 3
            or len(masks) != 3
            or not bool(masks[0].all())
            or any(bool(mask.any()) for mask in masks[1:])
        ):
            raise FeatureCacheError("PI05 single-camera preprocessing changed")
        with torch.inference_mode():
            embeddings = policy.model.paligemma_with_expert.embed_image(prepared[0])
            pooled = pool_pi05_visual_tokens(
                embeddings,
                expected_tokens=expected_tokens,
                expected_dim=expected_dim,
                spatial_grid_size=int(config["features"]["vision_spatial_grid_size"]),
            )
        chunks_by_demo[demo_index].append(
            pooled.to(device="cpu", dtype=torch.bfloat16)
        )

    episodes: list[torch.Tensor] = []
    offsets = [0]
    for demo_index in demo_indices:
        index = int(demo_index)
        if not chunks_by_demo[index]:
            raise FeatureCacheError(
                f"PI05 video episode produced no features: {task.task_id}/{index}"
            )
        episode = torch.cat(chunks_by_demo[index], dim=0)
        expected_length = task.episode_lengths[index]
        if episode.shape != (
            expected_length,
            int(config["features"]["vision_spatial_tokens"]),
            expected_dim,
        ):
            raise FeatureCacheError(
                f"PI05 cached episode length changed: {task.task_id}/{index}"
            )
        episodes.append(episode)
        offsets.append(offsets[-1] + expected_length)
    return torch.cat(episodes, dim=0), torch.tensor(offsets, dtype=torch.int64)


def _cache_task(
    *,
    policy: Any,
    tokenizer: Pi05PureLanguageTokenizer,
    task: FeatureCacheTask,
    demo_indices: Sequence[int],
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    rank: int,
) -> dict[str, Any]:
    started = time.monotonic()
    language = _language_features(policy, tokenizer, task.language, config)
    generic_language = _language_features(
        policy,
        tokenizer,
        str(config["features"]["generic_writer_language"]),
        config,
    )
    video, offsets = _video_features(
        policy, task, demo_indices, config, device
    )
    tensor_path, _ = task_cache_paths(output_dir, task.task_id)
    file_record = save_task_cache(
        tensor_path,
        language_features=language,
        generic_language_features=generic_language,
        video_features=video,
        episode_offsets=offsets,
        demo_indices=torch.tensor(demo_indices, dtype=torch.int64),
        metadata={
            "schema_version": PI05_TASK_FEATURE_CACHE_SCHEMA,
            "global_task_id": str(task.task_id),
            "extraction_sha256": str(contract["extraction_sha256"]),
        },
    )
    record = {
        "schema_version": PI05_TASK_FEATURE_CACHE_SCHEMA,
        "task_id": task.task_id,
        "suite": task.suite,
        "local_task_id": task.local_task_id,
        "split_role": task.split_role,
        "language": task.language,
        "expected_hdf5_sha256": task.expected_hdf5_sha256,
        "extraction_sha256": contract["extraction_sha256"],
        "rank": rank,
        "demo_indices": list(demo_indices),
        "episode_lengths": [task.episode_lengths[index] for index in demo_indices],
        "file": file_record,
        "wall_seconds": time.monotonic() - started,
    }
    write_task_record(output_dir, task.task_id, record)
    return record


def _rank_zero_contract(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    tasks: Sequence[FeatureCacheTask],
    assignments: Sequence[Sequence[FeatureCacheTask]],
    demo_indices: Sequence[int],
    world_size: int,
) -> dict[str, Any]:
    observed_git = git_state(REPO_ROOT)
    if observed_git["dirty_paths"]:
        raise FeatureCacheError("PI05 feature caching requires a clean worktree")
    if (
        args.mode == "formal"
        and not args.resume
        and observed_git["commit"] != observed_git["origin_main"]
    ):
        raise FeatureCacheError("fresh formal feature caching must launch from origin/main")
    evaluation_path = REPO_ROOT / config["authorities"]["evaluation_config"]["path"]
    authorities = load_evaluation_authorities(evaluation_path, REPO_ROOT)
    source = inspect_source_checkpoint(
        authorities,
        args.source_run,
        args.checkpoint,
        evaluation_mode="formal",
    )
    tokenizer = inspect_tokenizer(authorities, args.tokenizer_path)
    contract = build_contract(
        args=args,
        config=config,
        source=source,
        tokenizer=tokenizer,
        tasks=tasks,
        assignments=assignments,
        demo_indices=demo_indices,
        world_size=world_size,
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FeatureCacheError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "run_contract.json"
    if args.resume and not contract_path.is_file():
        raise FeatureCacheError("PI05 feature-cache resume has no run contract")
    if args.resume:
        previous = read_json(contract_path)
        if previous.get("contract_sha256") != contract["contract_sha256"]:
            raise FeatureCacheError("PI05 feature-cache resume contract changed")
        return previous
    write_json_atomic(contract_path, contract)
    return contract


def _resolve_paths(args: argparse.Namespace) -> None:
    args.config = args.config.resolve()
    args.source_run = args.source_run.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.tokenizer_path = args.tokenizer_path.resolve()
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()


def _prepare_schedule(
    args: argparse.Namespace, context: Any
) -> tuple[
    dict[str, Any],
    tuple[FeatureCacheTask, ...],
    tuple[int, ...],
    tuple[tuple[FeatureCacheTask, ...], ...],
    dict[str, Any],
]:
    config = load_pi05_feature_cache_config(args.config, REPO_ROOT)
    config["config_sha256"] = sha256_file(args.config)
    if args.mode == "formal" and config["profile"]["status"] != "sealed":
        raise FeatureCacheError("formal PI05 feature caching requires a sealed real profile")
    if context.world_size != int(config["parallel"]["world_size"]):
        raise FeatureCacheError("PI05 feature-cache launch must use all eight ranks")
    all_tasks = load_pi05_feature_tasks(
        config, REPO_ROOT, args.data_root, role=args.role
    )
    demo_indices: tuple[int, ...] = tuple(
        range(int(config["protocol"]["demo_count_per_task"]))
    )
    tasks = all_tasks
    if args.mode == "smoke":
        tasks = tasks[: int(config["smoke"]["task_count"])]
        demo_indices = demo_indices[: int(config["smoke"]["demo_count_per_task"])]
    assignments = balanced_task_assignments(tasks, context.world_size)
    payload: list[Any] = [None]
    if context.is_main:
        try:
            payload[0] = _rank_zero_contract(
                args=args,
                config=config,
                tasks=tasks,
                assignments=assignments,
                demo_indices=demo_indices,
                world_size=context.world_size,
            )
        except Exception as error:
            payload[0] = {"error": repr(error)}
    dist.broadcast_object_list(payload, src=0, device=context.device)
    if payload[0].get("error"):
        raise FeatureCacheError(payload[0]["error"])
    return config, tasks, demo_indices, assignments, payload[0]


def _load_frozen_feature_model(
    args: argparse.Namespace,
    context: Any,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[Any, Pi05PureLanguageTokenizer]:
    authorities = load_evaluation_authorities(
        REPO_ROOT / config["authorities"]["evaluation_config"]["path"],
        REPO_ROOT,
    )
    policy = load_policy(
        Path(contract["source"]["model_path"]),
        authorities.source_base_config,
        context.device,
    ).eval()
    if hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    tokenizer = Pi05PureLanguageTokenizer(
        args.tokenizer_path,
        int(config["features"]["language_max_tokens"]),
        str(context.device),
    )
    return policy, tokenizer


def _cache_assigned_tasks(
    *,
    args: argparse.Namespace,
    context: Any,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    assignments: Sequence[Sequence[FeatureCacheTask]],
    demo_indices: Sequence[int],
    policy: Any,
    tokenizer: Pi05PureLanguageTokenizer,
) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for task in assignments[context.rank]:
        if args.resume and task_cache_is_complete(
            args.output_dir,
            task.task_id,
            extraction_sha256=contract["extraction_sha256"],
            record_schema=PI05_TASK_FEATURE_CACHE_SCHEMA,
        ):
            continue
        record = _cache_task(
            policy=policy,
            tokenizer=tokenizer,
            task=task,
            demo_indices=demo_indices,
            config=config,
            contract=contract,
            output_dir=args.output_dir,
            device=context.device,
            rank=context.rank,
        )
        completed.append(record)
        print(
            json.dumps(
                {
                    "event": "task_complete",
                    "rank": context.rank,
                    "task_id": task.task_id,
                    "frames": record["file"]["frames"],
                    "wall_seconds": record["wall_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return completed


def _write_rank_record(
    args: argparse.Namespace,
    context: Any,
    assignments: Sequence[Sequence[FeatureCacheTask]],
    completed: Sequence[Mapping[str, Any]],
) -> None:
    write_json_atomic(
        args.output_dir / f"rank_{context.rank:02d}.json",
        {
            "rank": context.rank,
            "local_rank": context.local_rank,
            "device": str(context.device),
            "cuda_device_name": torch.cuda.get_device_name(context.device),
            "cuda_total_memory": torch.cuda.get_device_properties(
                context.device
            ).total_memory,
            "numa_node": context.numa_node,
            "cpu_affinity": list(context.cpu_affinity or ()),
            "task_ids": [task.task_id for task in assignments[context.rank]],
            "new": completed,
        },
    )


def _collect_records(
    output_dir: Path,
    tasks: Sequence[FeatureCacheTask],
    extraction_sha256: str,
) -> list[dict[str, Any]]:
    records = []
    for task in tasks:
        if not task_cache_is_complete(
            output_dir,
            task.task_id,
            extraction_sha256=extraction_sha256,
            record_schema=PI05_TASK_FEATURE_CACHE_SCHEMA,
        ):
            raise FeatureCacheError(f"PI05 task feature cache is incomplete: {task.task_id}")
        _, record_path = task_cache_paths(output_dir, task.task_id)
        records.append(read_json(record_path))
    return records


def _finalize_manifest(
    args: argparse.Namespace,
    tasks: Sequence[FeatureCacheTask],
    contract: Mapping[str, Any],
) -> None:
    records = _collect_records(
        args.output_dir, tasks, str(contract["extraction_sha256"])
    )
    result = {
        "schema_version": PI05_FEATURE_CACHE_MANIFEST_SCHEMA,
        "contract_sha256": contract["contract_sha256"],
        "extraction_sha256": contract["extraction_sha256"],
        "role": args.role,
        "task_count": len(records),
        "episode_count": sum(record["file"]["episodes"] for record in records),
        "frame_count": sum(record["file"]["frames"] for record in records),
        "task_records": records,
    }
    result["canonical_payload_sha256"] = canonical_hash(result)
    write_json_atomic(args.output_dir / "cache_manifest.json", result)
    print(
        json.dumps(
            {
                "event": "complete",
                "role": args.role,
                "tasks": result["task_count"],
                "episodes": result["episode_count"],
                "frames": result["frame_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _raise_rank_failures(context: Any, error: Exception | None) -> None:
    failures: list[Any] = [None] * context.world_size
    dist.all_gather_object(failures, None if error is None else repr(error))
    observed = [f"rank {rank}: {value}" for rank, value in enumerate(failures) if value]
    if observed:
        raise FeatureCacheError("PI05 feature cache rank failed; " + "; ".join(observed))


def main() -> int:
    args = parse_args()
    _resolve_paths(args)
    context = initialize_distributed(require_numa=args.mode == "formal")
    atexit.register(_destroy_process_group)
    torch.backends.cuda.matmul.allow_tf32 = True
    config, tasks, demos, assignments, contract = _prepare_schedule(args, context)
    policy, tokenizer = _load_frozen_feature_model(args, context, config, contract)
    completed: list[dict[str, Any]] = []
    failure: Exception | None = None
    try:
        completed = _cache_assigned_tasks(
            args=args,
            context=context,
            config=config,
            contract=contract,
            assignments=assignments,
            demo_indices=demos,
            policy=policy,
            tokenizer=tokenizer,
        )
        _write_rank_record(args, context, assignments, completed)
    except Exception as error:
        failure = error
    _raise_rank_failures(context, failure)
    if context.is_main:
        _finalize_manifest(args, tasks, contract)
    dist.barrier()
    _destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
