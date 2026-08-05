#!/usr/bin/env python3
"""Seal the action-hidden condition address and its train/validation Gram audit."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from ember.pi05_eval_contract import (
    inspect_source_checkpoint,
    load_evaluation_authorities,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json, sha256_file, write_json_atomic
from ember.pi05_source_setup import initialize_distributed, load_policy
from ember.writer.as_sampling import TeacherVideoSchedule
from ember.writer.condition_kernel import (
    FactorizedConditionFeature,
    load_condition_authority,
)
from ember.writer.data import RawTeacherVideo, RawTeacherVideoStore, WriterTaskAuthority
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.video_program import Pi05FrozenConditionDescriptor


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MANIFEST = REPO_ROOT / "configs/pi05_target_data_v1/manifest.json"
EVALUATION_CONFIG = REPO_ROOT / "configs/pi05_target_evaluation_v1.json"
SOURCE_CONFIG = REPO_ROOT / "configs/pi05_source_base_v1.json"
LORA_CONFIG = REPO_ROOT / "configs/pi05_lora_v1.json"
ADDRESS_SEED = 2_026_080_501
RFF_FREQUENCIES = 16


def _wait(paths: Sequence[Path], *, minutes: int = 30) -> None:
    deadline = time.monotonic() + minutes * 60
    while True:
        if all(path.is_file() for path in paths):
            return
        if time.monotonic() >= deadline:
            missing = [path.name for path in paths if not path.is_file()]
            raise RuntimeError("timed out waiting for " + ",".join(missing))
        time.sleep(0.05)


def _tasks(rows: Iterable[dict[str, Any]], root: Path) -> tuple[WriterTaskAuthority, ...]:
    result = []
    for row in rows:
        path = (root / str(row["hdf5"]["relative_path"])).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError("target video escaped the sealed data root")
        result.append(
            WriterTaskAuthority(
                task_id=int(row["global_task_id"]),
                language=str(row["language"]),
                path=path,
                expected_bytes=int(row["hdf5"]["bytes"]),
                expected_sha256=None,
            )
        )
    return tuple(sorted(result, key=lambda item: item.task_id))


def _descriptor_batch(
    descriptor: Pi05FrozenConditionDescriptor,
    policy: torch.nn.Module,
    videos: Sequence[RawTeacherVideo],
    tokens: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = [int(video.frames.shape[0]) for video in videos]
    offsets = torch.tensor(
        [0, *torch.tensor(counts, dtype=torch.long).cumsum(0).tolist()],
        dtype=torch.long,
        device=device,
    )
    frames = torch.from_numpy(
        np.concatenate([video.frames for video in videos], axis=0)
    ).to(device, non_blocking=True)
    condition_ids = torch.repeat_interleave(
        torch.arange(len(videos), device=device),
        torch.tensor(counts, dtype=torch.long, device=device),
    )
    language, mask, span = tokens
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return descriptor(
            policy,
            frames,
            condition_ids,
            offsets,
            language.expand(len(videos), -1),
            mask.expand(len(videos), -1),
            span.expand(len(videos), -1),
        )


def _counterfactuals(
    descriptor: Pi05FrozenConditionDescriptor,
    policy: torch.nn.Module,
    video: RawTeacherVideo,
    tokens: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    task_id: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(ADDRESS_SEED + task_id)
    permutation = torch.randperm(video.frames.shape[0], generator=generator).numpy()
    variants = (
        video,
        RawTeacherVideo(
            frames=np.ascontiguousarray(video.frames[::-1]),
            frame_indices=video.frame_indices,
            raw_frame_count=video.raw_frame_count,
        ),
        RawTeacherVideo(
            frames=np.ascontiguousarray(video.frames[permutation]),
            frame_indices=video.frame_indices,
            raw_frame_count=video.raw_frame_count,
        ),
    )
    _, video_descriptors = _descriptor_batch(
        descriptor, policy, variants, tokens, device
    )
    return video_descriptors.detach().cpu().to(torch.float32)


def _encode_tasks(
    owned: Sequence[WriterTaskAuthority],
    *,
    descriptor: Pi05FrozenConditionDescriptor,
    policy: torch.nn.Module,
    tokenizer: Pi05TeacherPrefixTokenizer,
    device: torch.device,
    include_counterfactual: bool,
) -> dict[str, torch.Tensor]:
    store = RawTeacherVideoStore(owned, frame_stride=5, max_open_files=2)
    task_rows = []
    video_rows = []
    counterfactual_rows = []
    task_ids = []
    text_repeat_max = []
    try:
        for task in owned:
            tokens = tokenizer([task.language])
            observed_task = []
            observed_video = []
            first_video: RawTeacherVideo | None = None
            for left in range(0, 50, 5):
                videos = [store.load(task.task_id, index) for index in range(left, left + 5)]
                if first_video is None:
                    first_video = videos[0]
                task_descriptor, video_descriptor = _descriptor_batch(
                    descriptor, policy, videos, tokens, device
                )
                observed_task.append(task_descriptor.detach().cpu().to(torch.float32))
                observed_video.append(video_descriptor.detach().cpu().to(torch.float32))
            repeated = torch.cat(observed_task)
            task_rows.append(repeated[0])
            video_rows.append(torch.cat(observed_video))
            task_ids.append(task.task_id)
            text_repeat_max.append(float((repeated - repeated[:1]).abs().max()))
            if include_counterfactual:
                if first_video is None:
                    raise RuntimeError("counterfactual video disappeared")
                counterfactual_rows.append(
                    _counterfactuals(
                        descriptor,
                        policy,
                        first_video,
                        tokens,
                        device,
                        task.task_id,
                    )
                )
    finally:
        store.close()
    return {
        "task_ids": torch.tensor(task_ids, dtype=torch.long),
        "task_descriptors": torch.stack(task_rows),
        "video_descriptors": torch.stack(video_rows),
        "text_repeat_max": torch.tensor(text_repeat_max, dtype=torch.float32),
        **(
            {"counterfactual_video_descriptors": torch.stack(counterfactual_rows)}
            if include_counterfactual
            else {}
        ),
    }


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    save_file({name: value.contiguous() for name, value in tensors.items()}, temporary)
    os.replace(temporary, path)


def _median_positive(values: torch.Tensor) -> float:
    selected = values[values > 0]
    if selected.numel() == 0:
        raise RuntimeError("condition address bandwidth collapsed")
    return float(selected.median())


def _build_authority(
    rank_files: Sequence[Path], authority_output: Path
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[int, int]]:
    payloads = [load_file(str(path), device="cpu") for path in rank_files]
    task_ids = torch.cat([payload["task_ids"] for payload in payloads])
    order = torch.argsort(task_ids)
    task_ids = task_ids.index_select(0, order)
    if task_ids.unique().numel() != 24:
        raise RuntimeError("address audit lost train24 coverage")
    tasks = torch.cat([payload["task_descriptors"] for payload in payloads]).index_select(
        0, order
    )
    videos = torch.cat([payload["video_descriptors"] for payload in payloads]).index_select(
        0, order
    )
    counterfactual = torch.cat(
        [payload["counterfactual_video_descriptors"] for payload in payloads]
    ).index_select(0, order)
    repeats = torch.cat([payload["text_repeat_max"] for payload in payloads]).index_select(
        0, order
    )
    center = tasks.mean(dim=0)
    normalized_tasks = F.normalize(tasks - center, dim=-1, eps=1e-12)
    normalized_videos = F.normalize(videos, dim=-1, eps=1e-12)
    task_bandwidth = _median_positive(torch.pdist(normalized_tasks))
    video_bandwidth = _median_positive(
        torch.cat([torch.pdist(value) for value in normalized_videos])
    )
    generator = torch.Generator(device="cpu").manual_seed(ADDRESS_SEED)
    authority = {
        "task_center": center.to(torch.float32),
        "task_frequencies": torch.randn(
            RFF_FREQUENCIES, tasks.shape[1], generator=generator
        ).div_(task_bandwidth),
        "video_frequencies": torch.randn(
            RFF_FREQUENCIES, videos.shape[2], generator=generator
        ).div_(video_bandwidth),
    }
    _atomic_safetensors(authority_output, authority)
    feature = FactorizedConditionFeature(**authority)
    expanded_tasks = tasks[:, None, :].expand(-1, 50, -1).reshape(-1, tasks.shape[1])
    phi = feature(expanded_tasks, videos.reshape(-1, videos.shape[2])).reshape(24, 50, -1)
    schedule = TeacherVideoSchedule(
        task_ids=tuple(int(value) for value in task_ids.tolist()),
        demo_indices=range(50),
        seed=20260722,
    )
    schedule_rows = []
    for visit in range(50):
        selected = torch.stack(
            [
                phi[index, schedule.demo_for_task_visit(int(task_id), visit)]
                for index, task_id in enumerate(task_ids.tolist())
            ]
        )
        gram = selected @ selected.T
        eigenvalues = torch.linalg.eigvalsh(gram.to(torch.float64))
        schedule_rows.append(
            {
                "visit": visit,
                "rank": int(torch.linalg.matrix_rank(gram.to(torch.float64))),
                "minimum_eigenvalue": float(eigenvalues.min()),
                "maximum_off_diagonal": float(
                    gram.masked_fill(torch.eye(24, dtype=torch.bool), -1).max()
                ),
                "regularized_condition_number": float(
                    torch.linalg.cond(
                        gram.to(torch.float64)
                        + torch.eye(24, dtype=torch.float64) * 0.01
                    )
                ),
            }
        )
    counterfactual_task = tasks[:, None, :].expand(-1, 3, -1).reshape(-1, tasks.shape[1])
    counterfactual_phi = feature(
        counterfactual_task, counterfactual.reshape(-1, counterfactual.shape[-1])
    ).reshape(24, 3, -1)
    summary = {
        "schema_version": "ember_condition_kernel_address_audit_v1",
        "information_wall": {
            "train_action_reads": 0,
            "validation_action_reads": 0,
            "test_action_reads": 0,
            "reward_or_outcome_reads": 0,
            "teacher_video_camera": "obs/agentview_rgb_only",
        },
        "address_seed": ADDRESS_SEED,
        "rff_frequencies_per_factor": RFF_FREQUENCIES,
        "train_tasks": 24,
        "train_videos": 1200,
        "task_bandwidth": task_bandwidth,
        "video_bandwidth": video_bandwidth,
        "maximum_repeated_text_descriptor_abs_difference": float(repeats.max()),
        "same_task_video_feature_distance_median": float(
            torch.cat([torch.pdist(value) for value in phi]).median()
        ),
        "cross_task_demo0_feature_distance_median": float(torch.pdist(phi[:, 0]).median()),
        "reversed_demo0_feature_distance": {
            "minimum": float((counterfactual_phi[:, 0] - counterfactual_phi[:, 1]).norm(dim=1).min()),
            "median": float((counterfactual_phi[:, 0] - counterfactual_phi[:, 1]).norm(dim=1).median()),
        },
        "shuffled_demo0_feature_distance": {
            "minimum": float((counterfactual_phi[:, 0] - counterfactual_phi[:, 2]).norm(dim=1).min()),
            "median": float((counterfactual_phi[:, 0] - counterfactual_phi[:, 2]).norm(dim=1).median()),
        },
        "schedule_gram": {
            "minimum_rank": min(row["rank"] for row in schedule_rows),
            "maximum_regularized_condition_number": max(
                row["regularized_condition_number"] for row in schedule_rows
            ),
            "maximum_off_diagonal": max(row["maximum_off_diagonal"] for row in schedule_rows),
            "visits": schedule_rows,
        },
    }
    task_to_row = {int(task_id): index for index, task_id in enumerate(task_ids.tolist())}
    summary["_train_tasks"] = tasks
    summary["_train_videos"] = videos
    return authority, summary, task_to_row


def _validation_summary(
    rank_files: Sequence[Path],
    authority: dict[str, torch.Tensor],
    train_tasks: torch.Tensor,
    train_videos: torch.Tensor,
) -> dict[str, Any]:
    payloads = [load_file(str(path), device="cpu") for path in rank_files]
    ids = torch.cat([payload["task_ids"] for payload in payloads])
    order = torch.argsort(ids)
    ids = ids.index_select(0, order)
    tasks = torch.cat([payload["task_descriptors"] for payload in payloads]).index_select(0, order)
    videos = torch.cat([payload["video_descriptors"] for payload in payloads]).index_select(0, order)
    if ids.unique().numel() != 8:
        raise RuntimeError("address audit lost validation8 coverage")
    feature = FactorizedConditionFeature(**authority)
    validation_phi = feature(
        tasks[:, None, :].expand(-1, 50, -1).reshape(-1, tasks.shape[1]),
        videos.reshape(-1, videos.shape[2]),
    )
    train_phi = feature(
        train_tasks[:, None, :].expand(-1, 50, -1).reshape(-1, train_tasks.shape[1]),
        train_videos.reshape(-1, train_videos.shape[2]),
    )
    nearest = (validation_phi @ train_phi.T).max(dim=1).values
    return {
        "validation_tasks": 8,
        "validation_videos": 400,
        "all_features_finite": bool(torch.isfinite(validation_phi).all()),
        "feature_norm_max_abs_error": float(
            (validation_phi.norm(dim=1) - 1).abs().max()
        ),
        "nearest_train_kernel": {
            "minimum": float(nearest.min()),
            "median": float(nearest.median()),
            "maximum": float(nearest.max()),
        },
        "statistics_fit_from_validation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--authority-output", type=Path, required=True)
    parser.add_argument("--max-frames-per-encoder-call", type=int, default=16)
    args = parser.parse_args()
    context = initialize_distributed(defer_process_group=True)
    target = read_json(TARGET_MANIFEST)
    train = _tasks(
        (row for row in target["tasks"] if row["split_role"] == "train"),
        args.data_root.resolve(),
    )
    validation = _tasks(
        (row for row in target["tasks"] if row["split_role"] == "validation"),
        args.data_root.resolve(),
    )
    if len(train) != 24 or len(validation) != 8 or context.world_size != 6:
        raise RuntimeError("address audit requires sealed train24/validation8 on six ranks")
    source_config = read_json(SOURCE_CONFIG)
    evaluations = load_evaluation_authorities(EVALUATION_CONFIG, REPO_ROOT)
    source = inspect_source_checkpoint(
        evaluations,
        args.source_run,
        args.checkpoint,
        evaluation_mode="formal",
    )
    policy = load_policy(Path(source["model_path"]), source_config, context.device)
    prepare_frozen_writer_policy(policy, load_pi05_lora_contract(LORA_CONFIG))
    descriptor = Pi05FrozenConditionDescriptor(
        image_width=2048,
        expert_width=1024,
        max_frames_per_encoder_call=args.max_frames_per_encoder_call,
        action_horizon=50,
        padded_action_dim=32,
        initialization_seed=7,
    ).to(context.device)
    tokenizer = Pi05TeacherPrefixTokenizer(
        args.tokenizer_path,
        int(source_config["features"]["tokenizer_max_length"]),
        str(context.device),
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_file = output / f"train_rank_{context.rank:02d}.safetensors"
    _atomic_safetensors(
        train_file,
        _encode_tasks(
            train[context.rank :: context.world_size],
            descriptor=descriptor,
            policy=policy,
            tokenizer=tokenizer,
            device=context.device,
            include_counterfactual=True,
        ),
    )
    train_files = [output / f"train_rank_{rank:02d}.safetensors" for rank in range(6)]
    _wait(train_files)
    authority_ready = output / "authority_ready.json"
    train_summary: dict[str, Any] | None = None
    if context.rank == 0:
        authority, train_summary, _ = _build_authority(
            train_files, args.authority_output.resolve()
        )
        write_json_atomic(
            authority_ready,
            {
                "authority": str(args.authority_output.resolve()),
                "sha256": sha256_file(args.authority_output.resolve()),
            },
        )
    _wait([authority_ready])
    validation_file = output / f"validation_rank_{context.rank:02d}.safetensors"
    _atomic_safetensors(
        validation_file,
        _encode_tasks(
            validation[context.rank :: context.world_size],
            descriptor=descriptor,
            policy=policy,
            tokenizer=tokenizer,
            device=context.device,
            include_counterfactual=False,
        ),
    )
    validation_files = [
        output / f"validation_rank_{rank:02d}.safetensors" for rank in range(6)
    ]
    _wait(validation_files)
    if context.rank == 0:
        if train_summary is None:
            raise RuntimeError("rank zero lost the train address summary")
        authority = load_condition_authority(str(args.authority_output.resolve()))
        train_tasks = train_summary.pop("_train_tasks")
        train_videos = train_summary.pop("_train_videos")
        summary = {
            **train_summary,
            "authority": {
                "path": str(args.authority_output.resolve()),
                "sha256": sha256_file(args.authority_output.resolve()),
            },
            "validation_apply_only": _validation_summary(
                validation_files, authority, train_tasks, train_videos
            ),
            "world_size": context.world_size,
            "rank_ownership": {
                str(rank): {
                    "train_task_ids": [item.task_id for item in train[rank::6]],
                    "validation_task_ids": [item.task_id for item in validation[rank::6]],
                }
                for rank in range(6)
            },
        }
        write_json_atomic(output / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        for path in (*train_files, *validation_files, authority_ready):
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
