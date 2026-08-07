"""Hashless train24 cache of action-hidden frozen PI0.5 video innovations."""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import save_file

from ember.expert_manifold.contract import (
    REPO_ROOT,
    ExpertManifoldError,
    ExpertTask,
    authority_path,
    load_expert_manifold_config,
    load_train_tasks,
    parse_task_indices,
)
from ember.expert_manifold.video_features import FrozenPi05VideoInnovationEncoder
from ember.pi05_eval_contract import (
    git_state,
    inspect_source_checkpoint,
    load_evaluation_authorities,
)
from ember.pi05_processing import Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import load_policy
from ember.writer.data import RawTeacherVideoStore


CACHE_WORKER_SCHEMA = "ember_pi05_expert_manifold_feature_worker_v1"
CACHE_TASK_SCHEMA = "ember_pi05_expert_manifold_feature_task_v1"
CACHE_MANIFEST_SCHEMA = "ember_pi05_expert_manifold_feature_cache_v1"


def _feature_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the authority that can change extracted video features."""

    return {
        "schema_version": config["schema_version"],
        "video_features": dict(config["video_features"]),
        "information_wall": dict(config["information_wall"]),
        "authorities": {
            name: dict(config["authorities"][name])
            for name in (
                "evaluation_config",
                "source_base_config",
                "target_data_manifest",
            )
        },
    }


def inspect_feature_cache(
    config_path: Path,
    cache_root: Path,
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the sealed train24 feature cache by authority, path, and size."""

    config_path = config_path.resolve()
    cache_root = cache_root.resolve()
    config = load_expert_manifold_config(config_path)
    manifest_path = cache_root / "cache_manifest.json"
    manifest = read_json(manifest_path)
    expected_source = {
        "source_run": str(Path(str(source["source_run"])).resolve()),
        "checkpoint": str(Path(str(source["checkpoint"])).resolve()),
        "model_path": str(Path(str(source["model_path"])).resolve()),
    }
    valid = (
        manifest.get("schema_version") == CACHE_MANIFEST_SCHEMA
        and manifest.get("feature_contract") == _feature_contract(config)
        and Path(str(manifest.get("cache_root", ""))).resolve() == cache_root
        and manifest.get("source") == expected_source
        and int(manifest.get("task_count", -1)) == 24
        and int(manifest.get("demo_count", -1)) == 50
        and int(manifest.get("phase_slots", -1))
        == int(config["video_features"]["phase_slots"])
        and int(manifest.get("feature_width", -1))
        == int(config["video_features"]["feature_width"])
        and manifest.get("information_wall")
        == {
            "teacher_action_reads": 0,
            "teacher_state_reads": 0,
            "reward_reads": 0,
            "terminal_reads": 0,
            "validation_video_reads": 0,
            "test_video_reads": 0,
        }
        and manifest.get("content_hash_policy") == "disabled_by_owner"
    )
    tasks = manifest.get("tasks", ())
    if not valid or len(tasks) != 24:
        raise ExpertManifoldError("video feature cache manifest changed")
    for ordinal, row in enumerate(tasks):
        path = Path(str(row.get("features", {}).get("path", "")))
        if (
            int(row.get("task_ordinal", -1)) != ordinal
            or row.get("split_role") != "train"
            or row.get("feature_shape")
            != [
                50,
                int(config["video_features"]["phase_slots"]),
                int(config["video_features"]["feature_width"]),
            ]
            or row.get("feature_dtype") != "bfloat16"
            or not path.is_file()
            or path.stat().st_size != int(row.get("features", {}).get("bytes", -1))
        ):
            raise ExpertManifoldError("video feature cache task changed")
    return manifest


def _feature_runtime(
    config: Mapping[str, Any], mode: str
) -> tuple[int, int, tuple[int, ...]]:
    video = config["video_features"]
    formal = video["formal_run"]
    if mode not in {"profile", "formal"}:
        raise ExpertManifoldError("unsupported feature-cache mode")
    if mode == "formal" and formal.get("status") != "sealed":
        raise ExpertManifoldError("formal video feature cache is not profile-sealed")
    demo_count = int(
        formal["demo_count"] if mode == "formal" else video["profile_defaults"]["demo_count"]
    )
    first, last = map(int, video["extraction"]["demo_indices"])
    demos = tuple(range(first, first + demo_count))
    if first != 0 or last != 49 or demos[-1] > last:
        raise ExpertManifoldError("video feature demo schedule changed")
    return demo_count, int(video["extraction"]["videos_per_batch"]), demos


def _worker_contract(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    mode: str,
    output_dir: Path,
    tasks: Sequence[ExpertTask],
    source: Mapping[str, Any],
    data_root: Path,
    tokenizer_path: Path,
    demos: Sequence[int],
    video_batch: int,
) -> dict[str, Any]:
    state = git_state(REPO_ROOT)
    return {
        "schema_version": CACHE_WORKER_SCHEMA,
        "mode": mode,
        "git": {"branch": state["branch"], "commit": state["commit"]},
        "config": {
            "path": str(config_path.resolve()),
            "schema": config["schema_version"],
            "bytes": config_path.stat().st_size,
        },
        "source": {
            "source_run": source["source_run"],
            "checkpoint": source["checkpoint"],
            "model_path": source["model_path"],
        },
        "tasks": [
            {
                "ordinal": task.ordinal,
                "global_task_id": task.global_task_id,
                "suite": task.suite,
                "task_id": task.task_id,
                "split_role": task.split_role,
                "language": task.language,
                "hdf5_bytes": task.authority.expected_bytes,
            }
            for task in tasks
        ],
        "extraction": {
            **dict(config["video_features"]),
            "demo_indices": list(demos),
            "videos_per_batch": video_batch,
            "teacher_action_reads": 0,
            "teacher_state_reads": 0,
            "reward_reads": 0,
            "terminal_reads": 0,
        },
        "runtime": {
            "host": socket.gethostname(),
            "cuda_visible_device": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "device": torch.cuda.get_device_name(0),
            "output_dir": str(output_dir.resolve()),
            "data_root": str(data_root.resolve()),
            "tokenizer": str(tokenizer_path.resolve()),
            "one_source_policy_per_worker": True,
        },
        "content_hash_policy": "disabled_by_owner",
    }


def _task_output_dir(output_dir: Path, task: ExpertTask) -> Path:
    return output_dir / f"task_{task.ordinal:02d}_global_{task.global_task_id:02d}"


@torch.inference_mode()
def _extract_task(
    *,
    task: ExpertTask,
    output_dir: Path,
    policy: torch.nn.Module,
    encoder: FrozenPi05VideoInnovationEncoder,
    tokenizer: Pi05TeacherPrefixTokenizer,
    frame_stride: int,
    demos: Sequence[int],
    videos_per_batch: int,
) -> dict[str, Any]:
    task_dir = _task_output_dir(output_dir, task)
    if task_dir.exists() and any(task_dir.iterdir()):
        raise ExpertManifoldError("video feature task output is not empty")
    task_dir.mkdir(parents=True, exist_ok=True)
    store = RawTeacherVideoStore(
        [task.authority], frame_stride=frame_stride, max_open_files=1
    )
    rows = []
    raw_counts = []
    sampled_counts = []
    started = time.monotonic()
    try:
        for start in range(0, len(demos), videos_per_batch):
            selected_demos = demos[start : start + videos_per_batch]
            videos = [store.load(task.global_task_id, demo) for demo in selected_demos]
            frame_batches = [torch.from_numpy(video.frames).to("cuda:0") for video in videos]
            frames = torch.cat(frame_batches)
            offsets = [0]
            for batch in frame_batches:
                offsets.append(offsets[-1] + int(batch.shape[0]))
            video_offsets = torch.tensor(offsets, dtype=torch.long, device="cuda:0")
            frame_video_ids = torch.repeat_interleave(
                torch.arange(len(videos), device="cuda:0"),
                torch.tensor([batch.shape[0] for batch in frame_batches], device="cuda:0"),
            )
            tokens, masks, spans = tokenizer([task.language] * len(videos))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features = encoder(
                    policy,
                    frames,
                    frame_video_ids,
                    video_offsets,
                    tokens,
                    masks,
                    spans,
                )
            rows.append(features.to(device="cpu", dtype=torch.bfloat16))
            raw_counts.extend(int(video.raw_frame_count) for video in videos)
            sampled_counts.extend(int(video.frames.shape[0]) for video in videos)
    finally:
        store.close()
    feature_path = task_dir / "features.safetensors"
    temporary = task_dir / f".features.tmp-{os.getpid()}.safetensors"
    save_file(
        {
            "video_innovation": torch.cat(rows),
            "demo_indices": torch.tensor(demos, dtype=torch.int64),
            "raw_frame_counts": torch.tensor(raw_counts, dtype=torch.int64),
            "sampled_frame_counts": torch.tensor(sampled_counts, dtype=torch.int64),
        },
        str(temporary),
    )
    os.replace(temporary, feature_path)
    record = {
        "schema_version": CACHE_TASK_SCHEMA,
        "task_ordinal": task.ordinal,
        "global_task_id": task.global_task_id,
        "suite": task.suite,
        "task_id": task.task_id,
        "split_role": "train",
        "language": task.language,
        "demo_indices": list(demos),
        "feature_shape": [len(demos), encoder.phase_slots, encoder.feature_width],
        "feature_dtype": "bfloat16",
        "features": {
            "path": str(feature_path.resolve()),
            "bytes": feature_path.stat().st_size,
        },
        "raw_frame_count_min": min(raw_counts),
        "raw_frame_count_max": max(raw_counts),
        "sampled_frame_count_min": min(sampled_counts),
        "sampled_frame_count_max": max(sampled_counts),
        "wall_seconds": time.monotonic() - started,
        "max_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "teacher_action_reads": 0,
        "teacher_state_reads": 0,
        "reward_reads": 0,
        "terminal_reads": 0,
        "content_hash_policy": "disabled_by_owner",
    }
    write_json_atomic(task_dir / "record.json", record)
    return record


def run_feature_worker(
    *,
    config_path: Path,
    mode: str,
    source_run: Path,
    checkpoint: Path,
    tokenizer_path: Path,
    data_root: Path,
    output_dir: Path,
    task_indices: str,
) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ExpertManifoldError("feature worker requires exactly one visible GPU")
    torch.cuda.set_device(0)
    config_path = config_path.resolve()
    config = load_expert_manifold_config(config_path)
    demo_count, video_batch, demos = _feature_runtime(config, mode)
    all_tasks = load_train_tasks(config, data_root.resolve())
    indices = parse_task_indices(task_indices, len(all_tasks))
    tasks = tuple(all_tasks[index] for index in indices)
    formal = config["video_features"]["formal_run"]
    if mode == "formal" and len(tasks) != int(formal["tasks_per_worker"]):
        raise ExpertManifoldError("formal feature worker must own four tasks")
    if mode == "profile" and len(tasks) != int(
        config["video_features"]["profile_defaults"]["task_count"]
    ):
        raise ExpertManifoldError("profile feature worker must own one task")
    if mode == "formal":
        state = git_state(REPO_ROOT)
        if state["dirty_paths"] or state["commit"] != state["upstream_commit"]:
            raise ExpertManifoldError("formal feature worker requires clean pushed code")
    authorities = load_evaluation_authorities(
        authority_path(config, "evaluation_config"), REPO_ROOT
    )
    source = inspect_source_checkpoint(
        authorities,
        source_run.resolve(),
        checkpoint.resolve(),
        evaluation_mode="formal",
    )
    contract = _worker_contract(
        config_path=config_path,
        config=config,
        mode=mode,
        output_dir=output_dir,
        tasks=tasks,
        source=source,
        data_root=data_root,
        tokenizer_path=tokenizer_path,
        demos=demos,
        video_batch=video_batch,
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ExpertManifoldError("fresh feature worker output is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "run_contract.json", contract)
    policy = load_policy(
        Path(source["model_path"]), authorities.source_base_config, torch.device("cuda:0")
    ).eval()
    if hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    video = config["video_features"]
    extraction = video["extraction"]
    encoder = FrozenPi05VideoInnovationEncoder(
        image_width=int(video["image_hidden_width"]),
        expert_width=int(video["expert_hidden_width"]),
        feature_width=int(video["feature_width"]),
        phase_slots=int(video["phase_slots"]),
        max_frames_per_encoder_call=int(extraction["max_frames_per_encoder_call"]),
        action_horizon=int(extraction["action_horizon"]),
        padded_action_dim=int(extraction["padded_action_dim"]),
        initialization_seed=int(extraction["initialization_seed"]),
    ).to("cuda:0").eval()
    tokenizer = Pi05TeacherPrefixTokenizer(
        tokenizer_path.resolve(),
        int(authorities.source_base_config["features"]["tokenizer_max_length"]),
        "cuda:0",
    )
    torch.cuda.reset_peak_memory_stats()
    records = [
        _extract_task(
            task=task,
            output_dir=output_dir,
            policy=policy,
            encoder=encoder,
            tokenizer=tokenizer,
            frame_stride=int(video["frame_stride"]),
            demos=demos,
            videos_per_batch=video_batch,
        )
        for task in tasks
    ]
    summary = {
        "schema_version": "ember_pi05_expert_manifold_feature_worker_summary_v1",
        "mode": mode,
        "demo_count": demo_count,
        "task_count": len(records),
        "tasks": records,
        "max_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "content_hash_policy": "disabled_by_owner",
    }
    write_json_atomic(output_dir / "worker_summary.json", summary)
    return summary


def seal_feature_cache(config_path: Path, cache_root: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    cache_root = cache_root.resolve()
    config = load_expert_manifold_config(config_path)
    formal = config["video_features"]["formal_run"]
    workers = tuple(sorted(path for path in cache_root.glob("worker_*") if path.is_dir()))
    if len(workers) != int(formal["allowed_worker_count"]):
        raise ExpertManifoldError("feature cache worker count is incomplete")
    tasks: dict[int, dict[str, Any]] = {}
    commits = set()
    sources: dict[tuple[str, str, str], dict[str, str]] = {}
    for worker in workers:
        contract = read_json(worker / "run_contract.json")
        summary = read_json(worker / "worker_summary.json")
        commits.add(str(contract.get("git", {}).get("commit", "")))
        declared_source = contract.get("source", {})
        source_record = {
            "source_run": str(Path(str(declared_source.get("source_run", ""))).resolve()),
            "checkpoint": str(Path(str(declared_source.get("checkpoint", ""))).resolve()),
            "model_path": str(Path(str(declared_source.get("model_path", ""))).resolve()),
        }
        sources[tuple(source_record.values())] = source_record
        if (
            contract.get("schema_version") != CACHE_WORKER_SCHEMA
            or contract.get("mode") != "formal"
            or summary.get("schema_version")
            != "ember_pi05_expert_manifold_feature_worker_summary_v1"
            or int(summary.get("task_count", -1)) != int(formal["tasks_per_worker"])
            or int(summary.get("demo_count", -1)) != int(formal["demo_count"])
        ):
            raise ExpertManifoldError("feature cache worker contract changed")
        for row in summary["tasks"]:
            ordinal = int(row["task_ordinal"])
            feature = Path(row["features"]["path"])
            if (
                ordinal in tasks
                or row.get("schema_version") != CACHE_TASK_SCHEMA
                or row.get("split_role") != "train"
                or row.get("feature_shape")
                != [
                    int(formal["demo_count"]),
                    int(config["video_features"]["phase_slots"]),
                    int(config["video_features"]["feature_width"]),
                ]
                or not feature.is_file()
                or feature.stat().st_size != int(row["features"]["bytes"])
                or any(int(row.get(name, -1)) != 0 for name in (
                    "teacher_action_reads", "teacher_state_reads", "reward_reads", "terminal_reads"
                ))
            ):
                raise ExpertManifoldError("feature cache task record changed")
            tasks[ordinal] = dict(row)
    if (
        set(tasks) != set(range(24))
        or len(commits) != 1
        or "" in commits
        or len(sources) != 1
        or any(not value for value in next(iter(sources.values())).values())
    ):
        raise ExpertManifoldError("feature cache does not cover train24 exactly")
    manifest = {
        "schema_version": CACHE_MANIFEST_SCHEMA,
        "config": {
            "path": str(config_path),
            "schema": config["schema_version"],
        },
        "feature_contract": _feature_contract(config),
        "cache_root": str(cache_root),
        "training_commit": next(iter(commits)),
        "source": next(iter(sources.values())),
        "task_count": 24,
        "demo_count": int(formal["demo_count"]),
        "phase_slots": int(config["video_features"]["phase_slots"]),
        "feature_width": int(config["video_features"]["feature_width"]),
        "tasks": [tasks[index] for index in range(24)],
        "information_wall": {
            "teacher_action_reads": 0,
            "teacher_state_reads": 0,
            "reward_reads": 0,
            "terminal_reads": 0,
            "validation_video_reads": 0,
            "test_video_reads": 0,
        },
        "content_hash_policy": "disabled_by_owner",
    }
    write_json_atomic(cache_root / "cache_manifest.json", manifest)
    return manifest
