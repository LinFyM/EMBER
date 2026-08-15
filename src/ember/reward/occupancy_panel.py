"""Matched-batch temporal stratification for successful-occupancy credit."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from lerobot.utils.constants import ACTION

from ember.reward.protocol import RewardProtocolError
from ember.reward.rollout import RewardTrajectory


def _action_disagreement(
    winner: torch.Tensor,
    loser: torch.Tensor,
    valid: int,
) -> float:
    if (
        winner.shape != loser.shape
        or winner.ndim != 3
        or winner.shape[0] != 1
        or not 0 < valid <= winner.shape[1]
    ):
        raise RewardProtocolError("matched occupancy action shape changed")
    value = (winner[:, :valid].float() - loser[:, :valid].float()).square().mean().sqrt()
    if not bool(torch.isfinite(value)):
        raise RewardProtocolError("matched occupancy action disagreement is nonfinite")
    return float(value)


def _stratified_indices(scores: Sequence[float], strata: int) -> tuple[int, ...]:
    count = len(scores)
    selected_count = min(strata, count)
    if strata != 8 or selected_count <= 0:
        raise RewardProtocolError("matched occupancy stratum contract changed")
    selected = []
    for index in range(selected_count):
        start = index * count // selected_count
        stop = (index + 1) * count // selected_count
        selected.append(max(range(start, stop), key=lambda row: (scores[row], -row)))
    if selected != sorted(set(selected)):
        raise RewardProtocolError("matched occupancy strata lost temporal order")
    return tuple(selected)


def _trajectory_selection(
    trajectory_id: int,
    pair: tuple[RewardTrajectory, RewardTrajectory],
    winner_arm: str,
    actions_by_arm: Mapping[str, Sequence[Sequence[torch.Tensor]]],
    strata_per_trajectory: int,
) -> tuple[
    list[tuple[dict[str, torch.Tensor], torch.Tensor, int]],
    tuple[int, ...],
    tuple[float, ...],
]:
    winner, loser = pair
    loser_arm = "reference" if winner_arm == "candidate" else "candidate"
    count = len(winner.observations)
    winner_actions = actions_by_arm[winner_arm][trajectory_id]
    loser_actions = actions_by_arm[loser_arm][trajectory_id]
    if (
        winner_arm not in {"reference", "candidate"}
        or not winner.success
        or loser.success
        or count <= 0
        or len(winner.valid_action_steps) != count
        or len(winner_actions) != count
        or len(loser_actions) != count
    ):
        raise RewardProtocolError("matched occupancy trajectory contract changed")
    scores = tuple(
        _action_disagreement(winner_action, loser_action, valid)
        for winner_action, loser_action, valid in zip(
            winner_actions,
            loser_actions,
            winner.valid_action_steps,
            strict=True,
        )
    )
    indices = _stratified_indices(scores, strata_per_trajectory)
    rows = []
    for index in indices:
        observation = winner.observations[index]
        valid = winner.valid_action_steps[index]
        rows.extend(
            (
                (observation, winner_actions[index], valid),
                (observation, loser_actions[index], valid),
            )
        )
    return rows, indices, scores


def _occupancy_batch(
    rows: Sequence[tuple[dict[str, torch.Tensor], torch.Tensor, int]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    keys = set(rows[0][0])
    if any(set(observation) != keys for observation, _, _ in rows):
        raise RewardProtocolError("matched occupancy observation keys changed")
    valid = torch.tensor([count for _, _, count in rows], dtype=torch.long, device=device)
    actions = torch.cat([action for _, action, _ in rows]).to(
        device=device, non_blocking=True
    )
    if (
        actions.ndim != 3
        or valid.shape != (actions.shape[0],)
        or bool((valid <= 0).any())
        or bool((valid > actions.shape[1]).any())
        or not torch.equal(valid[0::2], valid[1::2])
    ):
        raise RewardProtocolError("matched occupancy executed prefix is invalid")
    batch = {
        name: torch.cat([observation[name] for observation, _, _ in rows]).to(
            device=device, non_blocking=True
        )
        for name in sorted(keys)
    }
    batch[ACTION] = actions
    batch["executed_action_steps"] = valid
    batch["action_is_pad"] = (
        torch.arange(actions.shape[1], device=device)[None] >= valid[:, None]
    )
    return batch


def complete_matched_stratified_occupancy_batch(
    pairs: Sequence[tuple[RewardTrajectory, RewardTrajectory]],
    active_labels: Sequence[str],
    actions_by_arm: Mapping[str, Sequence[Sequence[torch.Tensor]]],
    *,
    strata_per_trajectory: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
    """Select one maximum-disagreement state per temporal stratum and pair actions."""

    if (
        len(pairs) not in {1, 2}
        or len(active_labels) != len(pairs)
        or any(label not in {"reference", "candidate"} for label in active_labels)
        or set(actions_by_arm) != {"reference", "candidate"}
        or any(len(actions_by_arm[arm]) != len(pairs) for arm in actions_by_arm)
    ):
        raise RewardProtocolError("matched stratified occupancy panel changed")
    rows: list[tuple[dict[str, torch.Tensor], torch.Tensor, int]] = []
    trajectory_ids: list[int] = []
    selected_indices, selected_scores, all_scores = [], [], []
    for trajectory_id, (pair, winner_arm) in enumerate(
        zip(pairs, active_labels, strict=True)
    ):
        trajectory_rows, indices, scores = _trajectory_selection(
            trajectory_id,
            pair,
            winner_arm,
            actions_by_arm,
            strata_per_trajectory,
        )
        rows.extend(trajectory_rows)
        trajectory_ids.extend([trajectory_id] * len(indices))
        selected_indices.append(list(indices))
        selected_scores.append([scores[index] for index in indices])
        all_scores.extend(scores)
    batch = _occupancy_batch(rows, device)
    flat_selected = [score for values in selected_scores for score in values]
    return batch, torch.tensor(trajectory_ids, dtype=torch.long, device=device), {
        "selected_credit_pairs": len(trajectory_ids),
        "selected_replan_indices": selected_indices,
        "selected_action_disagreement_rms_mean": sum(flat_selected) / len(flat_selected),
        "selected_action_disagreement_rms_minimum": min(flat_selected),
        "complete_action_disagreement_rms_mean": sum(all_scores) / len(all_scores),
    }


def empty_matched_occupancy_credit() -> dict[str, Any]:
    """Return the inactive-task metric row for the canonical occupancy panel."""

    return {
        "objective": 0.0,
        "preference_margin": 0.0,
        "winner_flow_loss": 0.0,
        "loser_flow_loss": 0.0,
        "discordant_trajectories": 0,
        "selected_credit_pairs": 0,
        "replay_rows": 0,
        "successful_action_steps": 0,
        "matched_winner_loser_action_rms": 0.0,
        "complete_occupancy_chunks": 0,
        "matched_policy_forwards": 0,
        "matched_query_batch_sizes": [],
        "stored_winner_to_matched_requery_rms": 0.0,
        "stored_loser_to_matched_first_requery_rms": 0.0,
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


def occupancy_cycle_metrics(
    records: Sequence[Mapping[str, Any]],
    active_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate the canonical matched-occupancy evidence for one cycle."""

    return {
        "discordant_preference_trajectories": sum(
            int(row["discordant_trajectories"]) for row in records
        ),
        "selected_credit_pairs": sum(
            int(row["selected_credit_pairs"]) for row in records
        ),
        "complete_occupancy_chunks": sum(
            int(row["complete_occupancy_chunks"]) for row in records
        ),
        "matched_policy_forwards": sum(
            int(row["matched_policy_forwards"]) for row in records
        ),
        "stored_winner_to_matched_requery_rms_mean": math.fsum(
            float(row["stored_winner_to_matched_requery_rms"])
            for row in active_records
        )
        / len(active_records),
        "stored_loser_to_matched_first_requery_rms_mean": math.fsum(
            float(row["stored_loser_to_matched_first_requery_rms"])
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
        "preference_replay_rows": sum(int(row["replay_rows"]) for row in records),
        "successful_action_steps": sum(
            int(row["successful_action_steps"]) for row in records
        ),
        "matched_stratified_occupancy_objective_mean": math.fsum(
            float(row["objective"]) for row in active_records
        )
        / len(active_records),
        "matched_stratified_occupancy_margin_mean": math.fsum(
            float(row["preference_margin"]) for row in active_records
        )
        / len(active_records),
    }
