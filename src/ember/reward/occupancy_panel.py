"""Temporal stratification for successful task-expert occupancy credit."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from lerobot.utils.constants import ACTION

from ember.reward.protocol import RewardProtocolError
from ember.reward.rollout import RewardTrajectory


def _action_disagreement(
    expert: torch.Tensor,
    student: torch.Tensor,
    valid: int,
) -> float:
    if (
        expert.shape != student.shape
        or expert.ndim != 3
        or expert.shape[0] != 1
        or not 0 < valid <= expert.shape[1]
    ):
        raise RewardProtocolError("successful expert action shape changed")
    value = (
        (expert[:, :valid].float() - student[:, :valid].float())
        .square()
        .mean()
        .sqrt()
    )
    if not bool(torch.isfinite(value)):
        raise RewardProtocolError("successful expert action disagreement is nonfinite")
    return float(value)


def _stratified_indices(scores: Sequence[float], strata: int) -> tuple[int, ...]:
    count = len(scores)
    selected_count = min(strata, count)
    if strata != 8 or selected_count <= 0:
        raise RewardProtocolError("successful expert occupancy strata changed")
    selected = []
    for index in range(selected_count):
        start = index * count // selected_count
        stop = (index + 1) * count // selected_count
        selected.append(max(range(start, stop), key=lambda row: (scores[row], -row)))
    if selected != sorted(set(selected)):
        raise RewardProtocolError("successful expert strata lost temporal order")
    return tuple(selected)


def _trajectory_selection(
    trajectory_id: int,
    trajectory: RewardTrajectory,
    actions_by_policy: Mapping[str, Sequence[Sequence[torch.Tensor]]],
    strata_per_trajectory: int,
) -> tuple[
    list[tuple[dict[str, torch.Tensor], torch.Tensor, int, int]],
    tuple[int, ...],
    tuple[float, ...],
]:
    count = len(trajectory.observations)
    expert_actions = actions_by_policy["expert"][trajectory_id]
    student_actions = actions_by_policy["student"][trajectory_id]
    if (
        not trajectory.success
        or count <= 0
        or len(trajectory.valid_action_steps) != count
        or len(expert_actions) != count
        or len(student_actions) != count
    ):
        raise RewardProtocolError("successful expert trajectory contract changed")
    scores = tuple(
        _action_disagreement(expert, student, valid)
        for expert, student, valid in zip(
            expert_actions,
            student_actions,
            trajectory.valid_action_steps,
            strict=True,
        )
    )
    indices = _stratified_indices(scores, strata_per_trajectory)
    rows = [
        (
            trajectory.observations[index],
            expert_actions[index],
            trajectory.valid_action_steps[index],
            trajectory.policy_noise_seeds[index],
        )
        for index in indices
    ]
    return rows, indices, scores


def _occupancy_batch(
    rows: Sequence[tuple[dict[str, torch.Tensor], torch.Tensor, int, int]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    keys = set(rows[0][0])
    if any(set(observation) != keys for observation, _, _, _ in rows):
        raise RewardProtocolError("successful expert observation keys changed")
    valid = torch.tensor(
        [count for _, _, count, _ in rows], dtype=torch.long, device=device
    )
    noise_seeds = torch.tensor(
        [seed for _, _, _, seed in rows], dtype=torch.long, device=device
    )
    actions = torch.cat([action for _, action, _, _ in rows]).to(
        device=device, non_blocking=True
    )
    if (
        actions.ndim != 3
        or valid.shape != (actions.shape[0],)
        or bool((valid <= 0).any())
        or bool((valid > actions.shape[1]).any())
    ):
        raise RewardProtocolError("successful expert executed prefix is invalid")
    batch = {
        name: torch.cat([observation[name] for observation, _, _, _ in rows]).to(
            device=device, non_blocking=True
        )
        for name in sorted(keys)
    }
    batch[ACTION] = actions
    batch["executed_action_steps"] = valid
    batch["policy_noise_seed"] = noise_seeds
    batch["action_is_pad"] = (
        torch.arange(actions.shape[1], device=device)[None] >= valid[:, None]
    )
    return batch


def complete_successful_expert_occupancy_batch(
    trajectories: Sequence[RewardTrajectory],
    actions_by_policy: Mapping[str, Sequence[Sequence[torch.Tensor]]],
    *,
    strata_per_trajectory: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
    """Select one maximum expert/student disagreement per progress stratum."""

    if (
        len(trajectories) not in {1, 2}
        or set(actions_by_policy) != {"expert", "student"}
        or any(
            len(actions_by_policy[arm]) != len(trajectories)
            for arm in actions_by_policy
        )
    ):
        raise RewardProtocolError("successful expert occupancy panel changed")
    rows: list[tuple[dict[str, torch.Tensor], torch.Tensor, int, int]] = []
    trajectory_ids: list[int] = []
    selected_indices, selected_scores, all_scores = [], [], []
    for trajectory_id, trajectory in enumerate(trajectories):
        trajectory_rows, indices, scores = _trajectory_selection(
            trajectory_id,
            trajectory,
            actions_by_policy,
            strata_per_trajectory,
        )
        rows.extend(trajectory_rows)
        trajectory_ids.extend([trajectory_id] * len(indices))
        selected_indices.append(list(indices))
        selected_scores.append([scores[index] for index in indices])
        all_scores.extend(scores)
    batch = _occupancy_batch(rows, device)
    flat_selected = [score for values in selected_scores for score in values]
    return (
        batch,
        torch.tensor(trajectory_ids, dtype=torch.long, device=device),
        {
            "selected_credit_states": len(trajectory_ids),
            "selected_replan_indices": selected_indices,
            "selected_action_disagreement_rms_mean": sum(flat_selected)
            / len(flat_selected),
            "selected_action_disagreement_rms_minimum": min(flat_selected),
            "complete_action_disagreement_rms_mean": sum(all_scores)
            / len(all_scores),
        },
    )


def empty_successful_expert_occupancy_credit() -> dict[str, Any]:
    """Return the inactive-task metric row for successful expert credit."""

    return {
        "objective": 0.0,
        "expert_action_distance": 0.0,
        "successful_trajectories": 0,
        "selected_credit_states": 0,
        "replay_rows": 0,
        "successful_action_steps": 0,
        "matched_expert_student_action_rms": 0.0,
        "complete_occupancy_chunks": 0,
        "matched_policy_forwards": 0,
        "matched_query_batch_sizes": [],
        "stored_expert_to_matched_requery_rms": 0.0,
        "matched_action_seconds": 0.0,
        "selected_replan_indices": [],
        "selected_action_disagreement_rms_mean": 0.0,
        "selected_action_disagreement_rms_minimum": 0.0,
        "complete_action_disagreement_rms_mean": 0.0,
        "functional_policy_forwards": 0,
        "functional_policy_backwards": 0,
        "lora_gradient_rms": 0.0,
        "credit_conditions": 0,
        "credit_unique_video_count": 0,
        "credit_view_records": [],
    }


def successful_expert_occupancy_cycle_metrics(
    records: Sequence[Mapping[str, Any]],
    active_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate successful-expert evidence for one full task cycle."""

    return {
        "successful_expert_trajectories": sum(
            int(row["successful_trajectories"]) for row in records
        ),
        "selected_credit_states": sum(
            int(row["selected_credit_states"]) for row in records
        ),
        "complete_occupancy_chunks": sum(
            int(row["complete_occupancy_chunks"]) for row in records
        ),
        "matched_policy_forwards": sum(
            int(row["matched_policy_forwards"]) for row in records
        ),
        "stored_expert_to_matched_requery_rms_mean": math.fsum(
            float(row["stored_expert_to_matched_requery_rms"])
            for row in active_records
        )
        / len(active_records),
        "matched_action_seconds": math.fsum(
            float(row["matched_action_seconds"]) for row in records
        ),
        "credit_conditions": sum(int(row["credit_conditions"]) for row in records),
        "credit_unique_video_count": sum(
            int(row["credit_unique_video_count"]) for row in records
        ),
        "expert_replay_rows": sum(int(row["replay_rows"]) for row in records),
        "successful_action_steps": sum(
            int(row["successful_action_steps"]) for row in records
        ),
        "successful_expert_objective_mean": math.fsum(
            float(row["objective"]) for row in active_records
        )
        / len(active_records),
        "expert_action_distance_mean": math.fsum(
            float(row["expert_action_distance"]) for row in active_records
        )
        / len(active_records),
    }
