from __future__ import annotations

import numpy as np
import torch

from ember.expert_manifold.video_schedule import (
    condition_demo_indices,
    reference_demo_index,
    reference_demo_indices,
    shuffled_frame_permutation,
    task_video_mapping,
    video_selection_seed,
)
from ember.pi05_eval_contract import policy_noise_seed
from ember.pi05_eval_results import _per_task_rows
from ember.writer.data import RawTeacherVideo
from ember.writer.live_adapter import _ordered_video_tensors


def _rows() -> list[dict]:
    return [
        {
            "suite": "libero_spatial",
            "task_id": 0,
            "split_role": "train",
            "language": "task zero",
            "init_state_id": state_id,
            "env_seed": 7,
            "policy_seed_root": 7,
            "policy_noise_seeds": [
                policy_noise_seed(7, "libero_spatial", 0, state_id, 0)
            ],
            "success": state_id == 0,
            "steps": 1,
            "wall_seconds": 0.1,
            "finished_at": 0.1,
        }
        for state_id in (0, 1)
    ]


def test_per_task_rows_summarizes_teacher_video_sets() -> None:
    rows = _rows()
    for row, demos in zip(rows, ((0, 1), (2, 3)), strict=True):
        row["writer"] = {
            "condition": "correct",
            "teacher_demo_indices": list(demos),
            "writer_generation_seconds": 0.25,
        }
    tasks = {
        ("libero_spatial", 0): {"split_role": "train", "language": "task zero"}
    }

    writer = _per_task_rows(rows, tasks)[0]["writer"]

    assert writer["videos_per_condition"] == 2
    assert writer["unique_teacher_videos"] == 4
    assert writer["unique_teacher_video_sets"] == 2
    assert writer["teacher_demo_set_counts"] == {"0,1": 1, "2,3": 1}
    assert writer["generation_wall_seconds"] == 0.5


def test_video_schedule_is_nested_balanced_and_order_independent() -> None:
    request = (7, "libero_spatial", 6, 0)
    assert video_selection_seed(
        *request, sampling_mode="without_replacement"
    ) == video_selection_seed(*request, sampling_mode="without_replacement")
    assert (
        0
        <= reference_demo_index(
            *request,
            demo_count=50,
            sampling_mode="without_replacement",
        )
        < 50
    )
    k1 = reference_demo_indices(
        *request,
        demo_count=50,
        sampling_mode="without_replacement",
        video_count=1,
    )
    k4 = reference_demo_indices(
        *request,
        demo_count=50,
        sampling_mode="without_replacement",
        video_count=4,
    )
    assert k1 == k4[:1]
    assert len(set(k4)) == 4
    all_sets = [
        reference_demo_indices(
            7,
            "libero_spatial",
            6,
            state,
            demo_count=50,
            sampling_mode="without_replacement",
            video_count=4,
        )
        for state in range(50)
    ]
    assert sorted(demo for demos in all_sets for demo in demos) == sorted(
        list(range(50)) * 4
    )
    other_k4 = condition_demo_indices(
        *request,
        condition="same_task_other",
        demo_count=50,
        sampling_mode="without_replacement",
        video_count=4,
    )
    assert len(set(other_k4)) == 4
    assert set(other_k4).isdisjoint(k4)
    assert condition_demo_indices(
        *request,
        condition="reversed",
        demo_count=50,
        sampling_mode="without_replacement",
        video_count=4,
    ) == k4

    keys = (
        ("libero_spatial", 1),
        ("libero_spatial", 3),
        ("libero_object", 1),
        ("libero_object", 3),
        ("libero_goal", 3),
        ("libero_goal", 6),
        ("libero_10", 1),
        ("libero_10", 2),
    )
    roles = {key: "validation" for key in keys}
    forward = task_video_mapping(keys, roles, "cross_suite_wrong")
    reverse = task_video_mapping(tuple(reversed(keys)), roles, "cross_suite_wrong")
    same_task = task_video_mapping(keys, roles, "same_task_other")
    temporal = task_video_mapping(keys, roles, "reversed")
    assert forward == reverse
    assert same_task == temporal
    assert all(row["suite"] == row["video_suite"] for row in temporal)
    assert all(row["suite"] != row["video_suite"] for row in forward)
    assert len({row["video_global_task_id"] for row in forward}) == len(keys)


def test_shuffled_keep_first_changes_only_the_anchor_position() -> None:
    shuffled = shuffled_frame_permutation(20, 7, keep_first=False)
    keep_first = shuffled_frame_permutation(20, 7, keep_first=True)
    assert keep_first[0].item() == 0
    assert keep_first[1:].tolist() == [
        index for index in shuffled.tolist() if index != 0
    ]
    assert sorted(keep_first.tolist()) == list(range(20))


def test_temporal_controls_reorder_frames_but_keep_display_positions() -> None:
    frames = np.arange(4 * 3 * 2 * 2, dtype=np.uint8).reshape(4, 3, 2, 2)
    indices = np.asarray([0, 5, 10, 15], dtype=np.int64)
    video = RawTeacherVideo(
        frames=frames.copy(), frame_indices=indices.copy(), raw_frame_count=16
    )
    correct_frames, correct_indices = _ordered_video_tensors(
        video, condition="correct", order_seed=7, device=torch.device("cpu")
    )
    reversed_frames, reversed_indices = _ordered_video_tensors(
        video, condition="reversed", order_seed=7, device=torch.device("cpu")
    )
    shuffled_frames, shuffled_indices = _ordered_video_tensors(
        video, condition="shuffled", order_seed=7, device=torch.device("cpu")
    )
    permutation = shuffled_frame_permutation(4, 7, keep_first=False)

    assert torch.equal(correct_frames, torch.from_numpy(frames))
    assert torch.equal(reversed_frames, torch.from_numpy(frames).flip(0))
    assert torch.equal(
        shuffled_frames, torch.from_numpy(frames).index_select(0, permutation)
    )
    assert torch.equal(correct_indices, torch.from_numpy(indices))
    assert torch.equal(reversed_indices, correct_indices)
    assert torch.equal(shuffled_indices, correct_indices)
    assert np.array_equal(video.frames, frames)
    assert np.array_equal(video.frame_indices, indices)


def test_process_controls_separate_endpoints_middle_and_sparse_evidence() -> None:
    frames = np.arange(7 * 3, dtype=np.uint8).reshape(7, 3, 1, 1)
    indices = np.arange(0, 35, 5, dtype=np.int64)
    video = RawTeacherVideo(
        frames=frames.copy(), frame_indices=indices.copy(), raw_frame_count=35
    )

    first, first_positions = _ordered_video_tensors(
        video, condition="first_frame_only", order_seed=7, device=torch.device("cpu")
    )
    final, final_positions = _ordered_video_tensors(
        video, condition="final_frame_only", order_seed=7, device=torch.device("cpu")
    )
    endpoints, endpoint_positions = _ordered_video_tensors(
        video, condition="first_final", order_seed=7, device=torch.device("cpu")
    )
    shuffled_middle, shuffled_positions = _ordered_video_tensors(
        video,
        condition="endpoints_middle_shuffled",
        order_seed=7,
        device=torch.device("cpu"),
    )
    sparse, sparse_positions = _ordered_video_tensors(
        video, condition="monotone_sparse", order_seed=7, device=torch.device("cpu")
    )

    assert torch.equal(first, torch.from_numpy(frames[:1]))
    assert first_positions.tolist() == [0]
    assert torch.equal(final, torch.from_numpy(frames[-1:]))
    assert final_positions.tolist() == [30]
    assert torch.equal(endpoints, torch.from_numpy(frames[[0, -1]]))
    assert endpoint_positions.tolist() == [0, 30]
    assert torch.equal(shuffled_middle[[0, -1]], torch.from_numpy(frames[[0, -1]]))
    assert not torch.equal(shuffled_middle[1:-1], torch.from_numpy(frames[1:-1]))
    assert torch.equal(shuffled_positions, torch.from_numpy(indices))
    assert torch.equal(sparse, torch.from_numpy(frames[[0, 2, 4, 6]]))
    assert sparse_positions.tolist() == [0, 10, 20, 30]
