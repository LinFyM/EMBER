#!/usr/bin/env python3
"""Build eight-rank frozen SmolVLA features for full action-hidden videos."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from ember.source_base_checkpoint import (
    canonical_hash,
    git_state,
    read_json,
    sha256_file,
    write_json_atomic,
)
from ember.writer.data import iter_action_hidden_video_chunks, verify_authority
from ember.writer.feature_cache import (
    FeatureCacheError,
    balanced_task_assignments,
    extraction_contract_sha256,
    load_feature_cache_config,
    load_train_tasks,
    pool_visual_tokens,
    save_task_cache,
    select_language_tokens,
    task_cache_is_complete,
    task_cache_paths,
    write_task_record,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/writer_feature_cache_v1.json"
    )
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _build_contract(
    args: argparse.Namespace,
    config: dict[str, Any],
    tasks: tuple[Any, ...],
    assignments: tuple[tuple[Any, ...], ...],
    demo_indices: tuple[int, ...],
    world_size: int,
) -> dict[str, Any]:
    policy_files = {
        name: sha256_file(args.policy_path / name)
        for name in ("config.json", "model.safetensors")
    }
    if policy_files["config.json"] != config["model"]["expected_policy_config_sha256"]:
        raise FeatureCacheError("source policy config differs from the sealed cache contract")
    contract = {
        "schema_version": "ember_writer_feature_cache_launch_v1",
        "mode": args.mode,
        "git": git_state(),
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "policy_path": str(args.policy_path.resolve()),
        "policy_files": policy_files,
        "data_root": str(args.data_root.resolve()),
        "manifest_sha256": config["protocol"]["manifest_sha256"],
        "expected_hdf5_sha256": {
            str(task.task_id): task.expected_hdf5_sha256 for task in tasks
        },
        "task_ids": [task.task_id for task in tasks],
        "demo_indices": list(demo_indices),
        "features": config["features"],
        "runtime": {
            "world_size": world_size,
            "one_policy_process_per_gpu": True,
            "assignments": [
                [task.task_id for task in rank_tasks] for rank_tasks in assignments
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


def _load_policy(policy_path: Path, device: torch.device) -> Any:
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy_config = PreTrainedConfig.from_pretrained(policy_path)
    if not isinstance(policy_config, SmolVLAConfig):
        raise FeatureCacheError("feature cache policy is not SmolVLA")
    policy_config.device = str(device)
    policy_config.pretrained_path = policy_path
    policy_config.use_amp = False
    policy = SmolVLAPolicy.from_pretrained(policy_path, config=policy_config)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def _language_features(
    policy: Any, tokenizer: Any, language: str, config: dict[str, Any], device: torch.device
) -> torch.Tensor:
    max_tokens = int(config["features"]["language_max_tokens"])
    tokenizer.padding_side = "right"
    encoded = tokenizer(
        language.rstrip("\n") + "\n",
        max_length=max_tokens,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    token_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device=device, dtype=torch.bool)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        embeddings = policy.model.vlm_with_expert.embed_language_tokens(token_ids)
        selected = select_language_tokens(
            embeddings,
            attention_mask,
            expected_dim=int(config["features"]["language_feature_dim"]),
        )
    return selected.to(device="cpu", dtype=torch.bfloat16)


def _video_features(
    policy: Any,
    task: Any,
    demo_indices: tuple[int, ...],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_tokens = int(config["features"]["vision_token_count"])
    expected_dim = int(config["features"]["vision_feature_dim"])
    chunks_by_demo: dict[int, list[torch.Tensor]] = {index: [] for index in demo_indices}
    next_start = {index: 0 for index in demo_indices}
    for demo_index, start, episode_length, frames in iter_action_hidden_video_chunks(
        task.authority,
        demo_indices,
        chunk_size=int(config["features"]["frame_batch_size_per_rank"]),
    ):
        if start != next_start[demo_index] or episode_length != task.episode_lengths[demo_index]:
            raise FeatureCacheError(f"video stream changed for task {task.task_id}")
        next_start[demo_index] += frames.shape[0]
        images = torch.from_numpy(frames).to(device=device, dtype=torch.float32).div_(255.0)
        prepared, masks = policy.prepare_images({"observation.images.camera1": images})
        if len(prepared) != 1 or len(masks) != 1 or not bool(masks[0].all()):
            raise FeatureCacheError("SmolVLA image preprocessing changed")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            embeddings = policy.model.vlm_with_expert.embed_image(prepared[0])
            pooled = pool_visual_tokens(
                embeddings, expected_tokens=expected_tokens, expected_dim=expected_dim
            )
        chunks_by_demo[demo_index].append(
            pooled.to(device="cpu", dtype=torch.bfloat16)
        )
    episodes: list[torch.Tensor] = []
    offsets = [0]
    for demo_index in demo_indices:
        episode = torch.cat(chunks_by_demo[demo_index], dim=0)
        expected_length = task.episode_lengths[demo_index]
        if episode.shape != (expected_length, expected_dim):
            raise FeatureCacheError(f"cached episode length changed for task {task.task_id}")
        episodes.append(episode)
        offsets.append(offsets[-1] + expected_length)
    return torch.cat(episodes, dim=0), torch.tensor(offsets, dtype=torch.int64)


def _cache_task(
    *,
    policy: Any,
    tokenizer: Any,
    task: Any,
    demo_indices: tuple[int, ...],
    config: dict[str, Any],
    contract: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    rank: int,
) -> dict[str, Any]:
    verify_authority(task.authority)
    started = time.monotonic()
    language = _language_features(policy, tokenizer, task.language, config, device)
    video, offsets = _video_features(policy, task, demo_indices, config, device)
    tensor_path, _ = task_cache_paths(output_dir, task.task_id)
    file_record = save_task_cache(
        tensor_path,
        language_features=language,
        video_features=video,
        episode_offsets=offsets,
        demo_indices=torch.tensor(demo_indices, dtype=torch.int64),
        metadata={
            "schema_version": "ember_writer_task_feature_cache_v1",
            "task_id": str(task.task_id),
            "extraction_sha256": contract["extraction_sha256"],
        },
    )
    record = {
        "schema_version": "ember_writer_task_feature_cache_v1",
        "task_id": task.task_id,
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


def main() -> int:
    args = _parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available() or not 0 <= local_rank < torch.cuda.device_count():
        raise FeatureCacheError("feature-cache rank has no eligible CUDA device")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("gloo")

    config = load_feature_cache_config(args.config.resolve(), REPO_ROOT)
    if world_size != int(config["parallel"]["world_size"]):
        raise FeatureCacheError("launch world size differs from cache contract")
    all_tasks = load_train_tasks(config, REPO_ROOT, args.data_root.resolve())
    tasks = all_tasks
    demo_indices = tuple(range(int(config["protocol"]["demo_count_per_task"])))
    if args.mode == "smoke":
        tasks = tasks[: int(config["smoke"]["task_count"])]
        demo_indices = demo_indices[: int(config["smoke"]["demo_count_per_task"])]
    assignments = balanced_task_assignments(tasks, world_size)

    if rank == 0:
        missing = [
            str(args.policy_path / name)
            for name in ("config.json", "model.safetensors")
            if not (args.policy_path / name).is_file()
        ]
        if missing:
            raise FeatureCacheError(f"source policy is incomplete: {missing}")
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
            raise FeatureCacheError(f"output directory is not empty: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        contract = _build_contract(
            args, config, tasks, assignments, demo_indices, world_size
        )
        if args.mode == "formal" and contract["git"]["dirty_paths"]:
            raise FeatureCacheError("formal cache build requires a clean committed worktree")
        contract_path = args.output_dir / "run_contract.json"
        if args.resume and contract_path.is_file():
            previous = read_json(contract_path)
            if previous.get("contract_sha256") != contract["contract_sha256"]:
                raise FeatureCacheError("resume contract differs from existing cache build")
        else:
            write_json_atomic(contract_path, contract)
    dist.barrier()
    contract = read_json(args.output_dir / "run_contract.json")

    policy = _load_policy(args.policy_path.resolve(), device)
    tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
    completed: list[dict[str, Any]] = []
    for task in assignments[rank]:
        if args.resume and task_cache_is_complete(
            args.output_dir,
            task.task_id,
            extraction_sha256=contract["extraction_sha256"],
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
            device=device,
            rank=rank,
        )
        completed.append(record)
        print(
            json.dumps(
                {
                    "event": "task_complete",
                    "rank": rank,
                    "task_id": task.task_id,
                    "frames": record["file"]["frames"],
                    "wall_seconds": record["wall_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    write_json_atomic(
        args.output_dir / f"rank_{rank:02d}.json",
        {
            "rank": rank,
            "local_rank": local_rank,
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "cuda_total_memory": torch.cuda.get_device_properties(device).total_memory,
            "task_ids": [task.task_id for task in assignments[rank]],
            "new": completed,
        },
    )
    dist.barrier()

    if rank == 0:
        records = []
        for task in tasks:
            if not task_cache_is_complete(
                args.output_dir,
                task.task_id,
                extraction_sha256=contract["extraction_sha256"],
            ):
                raise FeatureCacheError(f"task cache is incomplete: {task.task_id}")
            _, record_path = task_cache_paths(args.output_dir, task.task_id)
            records.append(read_json(record_path))
        result = {
            "schema_version": "ember_writer_feature_cache_manifest_v1",
            "contract_sha256": contract["contract_sha256"],
            "extraction_sha256": contract["extraction_sha256"],
            "task_count": len(records),
            "episode_count": sum(record["file"]["episodes"] for record in records),
            "frame_count": sum(record["file"]["frames"] for record in records),
            "task_records": records,
        }
        write_json_atomic(args.output_dir / "cache_manifest.json", result)
        print(
            json.dumps(
                {
                    "event": "complete",
                    "tasks": result["task_count"],
                    "episodes": result["episode_count"],
                    "frames": result["frame_count"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
