#!/usr/bin/env python3
"""Measure video and frame-order specificity of one sealed PI05 AS-Writer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, lora_state_sha256
from ember.pi05_evaluation import _initialize_worker
from ember.pi05_source_checkpoint import canonical_hash, sha256_file
from ember.writer.inference import FrozenWriterTaskAdapter, task_video_mapping


CONDITION_ORDER = (
    "correct_full",
    "same_task_other_demo",
    "cross_suite_wrong_full",
    "reverse_full",
    "shuffle_full",
    "single_first",
    "single_middle",
    "single_last",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def _state_row(
    state: Mapping[str, torch.Tensor], row: int
) -> dict[str, torch.Tensor]:
    return {
        name: value[row].detach().to(device="cpu", dtype=torch.float32)
        for name, value in state.items()
    }


def _tensor_metrics(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    reference_sq = 0.0
    candidate_sq = 0.0
    inner = 0.0
    for name in sorted(reference):
        left = reference[name].double()
        right = candidate[name].double()
        reference_sq += float(torch.sum(left * left))
        candidate_sq += float(torch.sum(right * right))
        inner += float(torch.sum(left * right))
    difference_sq = max(0.0, reference_sq + candidate_sq - 2.0 * inner)
    reference_norm = math.sqrt(reference_sq)
    candidate_norm = math.sqrt(candidate_sq)
    denominator = max(0.5 * (reference_norm + candidate_norm), 1e-30)
    cosine_denominator = max(reference_norm * candidate_norm, 1e-30)
    return {
        "relative_l2": math.sqrt(difference_sq) / denominator,
        "cosine_similarity": inner / cosine_denominator,
        "reference_l2": reference_norm,
        "candidate_l2": candidate_norm,
    }


def _effective_update_metrics(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    adapter: FrozenWriterTaskAdapter,
) -> dict[str, float]:
    reference_sq = 0.0
    candidate_sq = 0.0
    inner = 0.0
    scale = float(adapter.lora_contract.alpha) / float(adapter.lora_contract.rank)
    for target in adapter.lora_contract.targets:
        a_name = target.name + LORA_A_SUFFIX
        b_name = target.name + LORA_B_SUFFIX
        a_left = reference[a_name].double()
        b_left = reference[b_name].double()
        a_right = candidate[a_name].double()
        b_right = candidate[b_name].double()
        # <B1 A1, B2 A2>_F = <B1^T B2, A1 A2^T>_F.
        reference_sq += float(
            torch.sum((b_left.T @ b_left) * (a_left @ a_left.T))
        )
        candidate_sq += float(
            torch.sum((b_right.T @ b_right) * (a_right @ a_right.T))
        )
        inner += float(torch.sum((b_left.T @ b_right) * (a_left @ a_right.T)))
    reference_sq *= scale * scale
    candidate_sq *= scale * scale
    inner *= scale * scale
    difference_sq = max(0.0, reference_sq + candidate_sq - 2.0 * inner)
    reference_norm = math.sqrt(reference_sq)
    candidate_norm = math.sqrt(candidate_sq)
    denominator = max(0.5 * (reference_norm + candidate_norm), 1e-30)
    cosine_denominator = max(reference_norm * candidate_norm, 1e-30)
    return {
        "relative_l2": math.sqrt(difference_sq) / denominator,
        "cosine_similarity": inner / cosine_denominator,
        "reference_frobenius": reference_norm,
        "candidate_frobenius": candidate_norm,
    }


def _feature_metrics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    left = reference.detach().to(device="cpu", dtype=torch.float64).flatten()
    right = candidate.detach().to(device="cpu", dtype=torch.float64).flatten()
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    difference = float(torch.linalg.vector_norm(left - right))
    denominator = max(0.5 * (left_norm + right_norm), 1e-30)
    cosine_denominator = max(left_norm * right_norm, 1e-30)
    return {
        "relative_l2": difference / denominator,
        "cosine_similarity": float(torch.dot(left, right)) / cosine_denominator,
        "reference_l2": left_norm,
        "candidate_l2": right_norm,
    }


def _condition_videos(
    adapter: FrozenWriterTaskAdapter,
    *,
    language_global_task_id: int,
    wrong_video_global_task_id: int,
    demo_index: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    correct = adapter.store.load(language_global_task_id, demo_index)
    alternate_demo = (demo_index + 1) % 50
    alternate = adapter.store.load(language_global_task_id, alternate_demo)
    wrong = adapter.store.load(wrong_video_global_task_id, demo_index)
    frame_count = int(correct.frames.shape[0])
    permutation = np.random.default_rng(
        int(
            hashlib.sha256(
                f"ember-video-specificity:{language_global_task_id}:{demo_index}".encode()
            ).hexdigest()[:16],
            16,
        )
    ).permutation(frame_count)
    middle = frame_count // 2

    def record(
        frames: np.ndarray,
        indices: np.ndarray,
        *,
        video_global_task_id: int,
        selected_demo: int,
        transform: str,
        raw_frame_count: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        return (
            np.ascontiguousarray(frames),
            np.ascontiguousarray(indices, dtype=np.int64),
            {
                "video_global_task_id": video_global_task_id,
                "demo_index": selected_demo,
                "transform": transform,
                "sampled_frame_count": int(frames.shape[0]),
                "raw_frame_count": raw_frame_count,
                "frame_indices": [int(value) for value in indices.tolist()],
            },
        )

    return {
        "correct_full": record(
            correct.frames,
            correct.frame_indices,
            video_global_task_id=language_global_task_id,
            selected_demo=demo_index,
            transform="none",
            raw_frame_count=correct.raw_frame_count,
        ),
        "same_task_other_demo": record(
            alternate.frames,
            alternate.frame_indices,
            video_global_task_id=language_global_task_id,
            selected_demo=alternate_demo,
            transform="none",
            raw_frame_count=alternate.raw_frame_count,
        ),
        "cross_suite_wrong_full": record(
            wrong.frames,
            wrong.frame_indices,
            video_global_task_id=wrong_video_global_task_id,
            selected_demo=demo_index,
            transform="none",
            raw_frame_count=wrong.raw_frame_count,
        ),
        # Content is reversed or shuffled while chronological positions remain
        # ascending, so these arms isolate order rather than timestamp identity.
        "reverse_full": record(
            correct.frames[::-1],
            correct.frame_indices,
            video_global_task_id=language_global_task_id,
            selected_demo=demo_index,
            transform="reverse_content_keep_chronological_positions",
            raw_frame_count=correct.raw_frame_count,
        ),
        "shuffle_full": record(
            correct.frames[permutation],
            correct.frame_indices,
            video_global_task_id=language_global_task_id,
            selected_demo=demo_index,
            transform="deterministic_shuffle_keep_chronological_positions",
            raw_frame_count=correct.raw_frame_count,
        ),
        "single_first": record(
            correct.frames[:1],
            correct.frame_indices[:1],
            video_global_task_id=language_global_task_id,
            selected_demo=demo_index,
            transform="single_first",
            raw_frame_count=correct.raw_frame_count,
        ),
        "single_middle": record(
            correct.frames[middle : middle + 1],
            correct.frame_indices[middle : middle + 1],
            video_global_task_id=language_global_task_id,
            selected_demo=demo_index,
            transform="single_middle",
            raw_frame_count=correct.raw_frame_count,
        ),
        "single_last": record(
            correct.frames[-1:],
            correct.frame_indices[-1:],
            video_global_task_id=language_global_task_id,
            selected_demo=demo_index,
            transform="single_last",
            raw_frame_count=correct.raw_frame_count,
        ),
    }


@torch.inference_mode()
def main() -> None:
    arguments = _arguments()
    started = time.monotonic()
    runtime = _initialize_worker(
        arguments.evaluation_root.resolve(), arguments.worker_id
    )
    adapter = runtime.task_adapter
    if not isinstance(adapter, FrozenWriterTaskAdapter):
        raise RuntimeError("evaluation root does not own a frozen AS-Writer")
    correct_rows = tuple(adapter.evaluation_adapter["task_video_mapping"])
    keys = tuple((str(row["suite"]), int(row["task_id"])) for row in correct_rows)
    roles = {key: "validation" for key in keys}
    wrong_by_key = {
        (str(row["suite"]), int(row["task_id"])): row
        for row in task_video_mapping(keys, roles, "cross_suite_wrong")
    }
    video_seed = int(adapter.evaluation_adapter["video_schedule"]["seed"])

    task_results: list[dict[str, Any]] = []
    for mapping in correct_rows:
        suite = str(mapping["suite"])
        task_id = int(mapping["task_id"])
        language_id = int(mapping["language_global_task_id"])
        wrong_id = int(wrong_by_key[(suite, task_id)]["video_global_task_id"])
        # Match the exact deterministic teacher-video choice of fixed init state 0.
        from ember.writer.inference import writer_video_demo_index

        demo_index = writer_video_demo_index(
            video_seed, suite, task_id, 0, demo_count=50
        )
        conditions = _condition_videos(
            adapter,
            language_global_task_id=language_id,
            wrong_video_global_task_id=wrong_id,
            demo_index=demo_index,
        )
        frames: list[torch.Tensor] = []
        indices: list[torch.Tensor] = []
        offsets = [0]
        metadata: dict[str, Any] = {}
        for condition in CONDITION_ORDER:
            value_frames, value_indices, value_metadata = conditions[condition]
            frames.append(torch.from_numpy(value_frames))
            indices.append(torch.from_numpy(value_indices))
            offsets.append(offsets[-1] + int(value_frames.shape[0]))
            metadata[condition] = value_metadata
        frame_batch = torch.cat(frames).to(adapter.device, non_blocking=True)
        index_batch = torch.cat(indices).to(adapter.device, non_blocking=True)
        offset_batch = torch.tensor(
            offsets, dtype=torch.long, device=adapter.device
        )
        language = adapter.language_by_id[language_id]
        tokens, mask = adapter.tokenizer([language] * len(CONDITION_ORDER))
        captured: dict[str, torch.Tensor] = {}

        def capture_features(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            captured["task_features"] = output.detach()

        hook = adapter.writer.task_encoder.register_forward_hook(capture_features)
        try:
            with torch.autocast(
                device_type=adapter.device.type,
                dtype=torch.bfloat16,
                enabled=adapter.device.type == "cuda",
            ):
                state = adapter.writer(
                    frame_batch,
                    index_batch,
                    offset_batch,
                    tokens,
                    mask,
                    policy=adapter.policy,
                )
        finally:
            hook.remove()
        features = captured["task_features"]
        states = {
            condition: _state_row(state, row)
            for row, condition in enumerate(CONDITION_ORDER)
        }
        reference = states["correct_full"]
        comparisons: dict[str, Any] = {}
        for row, condition in enumerate(CONDITION_ORDER[1:], start=1):
            comparisons[condition] = {
                "condition": metadata[condition],
                "lora_parameter": _tensor_metrics(reference, states[condition]),
                "effective_lora_update": _effective_update_metrics(
                    reference, states[condition], adapter
                ),
                "temporal_task_features": _feature_metrics(
                    features[0], features[row]
                ),
                "lora_sha256": lora_state_sha256(states[condition]),
            }
        task_results.append(
            {
                "suite": suite,
                "task_id": task_id,
                "language_global_task_id": language_id,
                "language": language,
                "correct_condition": metadata["correct_full"],
                "correct_lora_sha256": lora_state_sha256(reference),
                "comparisons_to_correct_full": comparisons,
            }
        )
        print(f"completed {suite}:{task_id}", flush=True)

    aggregates: dict[str, Any] = {}
    for condition in CONDITION_ORDER[1:]:
        aggregates[condition] = {}
        for family in (
            "lora_parameter",
            "effective_lora_update",
            "temporal_task_features",
        ):
            values = [
                float(
                    task["comparisons_to_correct_full"][condition][family][
                        "relative_l2"
                    ]
                )
                for task in task_results
            ]
            aggregates[condition][family] = {
                "relative_l2_median": float(np.median(values)),
                "relative_l2_mean": float(np.mean(values)),
                "relative_l2_min": float(np.min(values)),
                "relative_l2_max": float(np.max(values)),
            }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "ember_pi05_writer_video_specificity_v1",
        "purpose": (
            "post-selection representation and effective-adapter diagnostic; "
            "no actions, rewards, policy outcomes, or checkpoint selection"
        ),
        "git_commit": _git_commit(),
        "evaluation_root": str(arguments.evaluation_root.resolve()),
        "evaluation_contract_sha256": sha256_file(
            arguments.evaluation_root / "run_contract.json"
        ),
        "evaluation_results_sha256": sha256_file(
            arguments.evaluation_root / "results.json"
        ),
        "checkpoint": dict(adapter.evaluation_adapter["checkpoint"]),
        "video_seed": video_seed,
        "reference": "correct_full",
        "conditions": list(CONDITION_ORDER),
        "task_count": len(task_results),
        "tasks": task_results,
        "aggregates": aggregates,
        "wall_clock_seconds": time.monotonic() - started,
        "cuda_max_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
    }
    result["canonical_payload_sha256"] = canonical_hash(result)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(arguments.output)
    arguments.output.with_suffix(arguments.output.suffix + ".sha256").write_text(
        f"{digest}  {arguments.output.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": digest,
                "wall_clock_seconds": result["wall_clock_seconds"],
                "aggregates": aggregates,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
