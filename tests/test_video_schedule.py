from __future__ import annotations

import torch

from ember.expert_manifold.video_schedule import (
    condition_demo_indices,
    reference_demo_index,
    reference_demo_indices,
    shuffled_frame_permutation,
    task_video_mapping,
    video_selection_seed,
)
from ember.video_conditions import frame_control


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

    held_keys = (
        ("libero_spatial", 0),
        ("libero_spatial", 9),
        ("libero_object", 8),
        ("libero_goal", 5),
        ("libero_10", 6),
    )
    held_roles = {key: "train" for key in held_keys}
    held_correct = task_video_mapping(held_keys, held_roles, "correct")
    assert [(row["suite"], row["task_id"]) for row in held_correct] == list(
        held_keys
    )
    assert all(row["suite"] == row["video_suite"] for row in held_correct)


def test_shuffled_keep_first_changes_only_the_anchor_position() -> None:
    shuffled = shuffled_frame_permutation(20, 7, keep_first=False)
    keep_first = shuffled_frame_permutation(20, 7, keep_first=True)
    assert keep_first[0].item() == 0
    assert keep_first[1:].tolist() == [
        index for index in shuffled.tolist() if index != 0
    ]
    assert sorted(keep_first.tolist()) == list(range(20))


def test_temporal_controls_reorder_frames_but_keep_display_positions() -> None:
    frames = torch.arange(4 * 3 * 2 * 2).reshape(4, 3, 2, 2)
    indices = torch.tensor([0, 5, 10, 15])
    correct = frame_control(4, condition="correct", order_seed=7)
    reversed_control = frame_control(4, condition="reversed", order_seed=7)
    shuffled = frame_control(4, condition="shuffled", order_seed=7)
    permutation = shuffled_frame_permutation(4, 7, keep_first=False)

    assert torch.equal(frames.index_select(0, correct.content), frames)
    assert torch.equal(frames.index_select(0, reversed_control.content), frames.flip(0))
    assert torch.equal(frames.index_select(0, shuffled.content), frames.index_select(0, permutation))
    assert torch.equal(indices.index_select(0, correct.positions), indices)
    assert torch.equal(indices.index_select(0, reversed_control.positions), indices)
    assert torch.equal(indices.index_select(0, shuffled.positions), indices)


def test_process_controls_separate_endpoints_middle_and_sparse_evidence() -> None:
    frames = torch.arange(7 * 3).reshape(7, 3, 1, 1)
    indices = torch.arange(0, 35, 5)
    controls = {
        name: frame_control(7, condition=name, order_seed=7)
        for name in (
            "first_frame_only",
            "final_frame_only",
            "first_final",
            "endpoints_middle_shuffled",
            "monotone_sparse",
            "static_first_repeated",
        )
    }
    selected = {
        name: frames.index_select(0, control.content)
        for name, control in controls.items()
    }
    positions = {
        name: indices.index_select(0, control.positions)
        for name, control in controls.items()
    }
    assert torch.equal(selected["first_frame_only"], frames[:1])
    assert positions["first_frame_only"].tolist() == [0]
    assert torch.equal(selected["final_frame_only"], frames[-1:])
    assert positions["final_frame_only"].tolist() == [30]
    assert torch.equal(selected["first_final"], frames[[0, -1]])
    assert positions["first_final"].tolist() == [0, 30]
    shuffled = selected["endpoints_middle_shuffled"]
    assert torch.equal(shuffled[[0, -1]], frames[[0, -1]])
    assert not torch.equal(shuffled[1:-1], frames[1:-1])
    assert torch.equal(positions["endpoints_middle_shuffled"], indices)
    assert torch.equal(selected["monotone_sparse"], frames[[0, 2, 4, 6]])
    assert positions["monotone_sparse"].tolist() == [0, 10, 20, 30]
    assert torch.equal(selected["static_first_repeated"], frames[[0] * 7])
    assert positions["static_first_repeated"].tolist() == indices.tolist()
