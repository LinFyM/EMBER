"""Frozen task-grounded visual progress credit for one-shot Writer RL."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ember.reward.protocol import RewardProtocolError
from ember.rl_writer.flow_credit import leave_one_out_binary_advantages


PROGRESS_DIAGNOSTIC_ROW_SCHEMA = (
    "ember_pi05_task_grounded_progress_credit_diagnostic_row_v1"
)
PROGRESS_DIAGNOSTIC_RESULT_SCHEMA = (
    "ember_pi05_task_grounded_progress_credit_diagnostic_result_v1"
)


def normalized_progress_components(
    grounded_evidence: torch.Tensor,
    interactions: torch.Tensor,
    valid_task_tokens: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Pack RMS-normalized task-token evidence plus one interaction component."""

    if (
        grounded_evidence.ndim != 3
        or interactions.ndim != 2
        or grounded_evidence.shape[0] != interactions.shape[0]
        or grounded_evidence.shape[-1] != interactions.shape[-1]
        or valid_task_tokens.ndim != 2
        or valid_task_tokens.shape[0] != 1
        or valid_task_tokens.shape[1] != grounded_evidence.shape[1]
        or valid_task_tokens.dtype != torch.bool
        or not bool(valid_task_tokens.any())
        or epsilon <= 0
    ):
        raise RewardProtocolError("invalid task-grounded progress evidence")
    selected = grounded_evidence[:, valid_task_tokens[0]].float()
    packed = torch.cat((selected, interactions[:, None].float()), dim=1)
    denominator = packed.square().mean(dim=-1, keepdim=True).add(epsilon).sqrt()
    normalized = packed / denominator
    if not bool(torch.isfinite(normalized).all()):
        raise RewardProtocolError("non-finite task-grounded progress components")
    return normalized


def semantic_progress_utilities(
    teacher_start: torch.Tensor,
    teacher_goal: torch.Tensor,
    rollout_starts: torch.Tensor,
    rollout_terminals: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project start-relative rollout changes onto the teacher content change."""

    if (
        teacher_start.ndim != 2
        or teacher_goal.shape != teacher_start.shape
        or rollout_starts.ndim != 3
        or rollout_terminals.shape != rollout_starts.shape
        or rollout_starts.shape[1:] != teacher_start.shape
        or rollout_starts.shape[0] <= 0
        or epsilon <= 0
    ):
        raise RewardProtocolError("invalid semantic progress utility inputs")
    values = (teacher_start, teacher_goal, rollout_starts, rollout_terminals)
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise RewardProtocolError("semantic progress utility received non-finite input")

    direction = teacher_goal.float() - teacher_start.float()
    displacement = rollout_terminals.float() - rollout_starts.float()
    component_energy = direction.square().sum(dim=-1)
    total_energy = component_energy.sum()
    if float(total_energy) <= 0:
        return displacement.new_zeros(displacement.shape[0]), component_energy
    weights = component_energy / (total_energy + epsilon)
    direction_norm = direction.square().sum(dim=-1).sqrt()
    displacement_norm = displacement.square().sum(dim=-1).sqrt()
    alignment = (displacement * direction[None]).sum(dim=-1) / (
        displacement_norm * direction_norm[None] + epsilon
    )
    magnitude = (displacement_norm / (direction_norm[None] + epsilon)).clamp(max=1.0)
    utilities = (weights[None] * alignment * magnitude).sum(dim=-1)
    if not bool(torch.isfinite(utilities).all()) or bool((utilities.abs() > 1.00001).any()):
        raise RewardProtocolError("semantic progress utility escaped its finite bound")
    return utilities, component_energy


def binary_first_progress_advantages(
    successes: torch.Tensor,
    utilities: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    """Keep binary precedence and use semantic LOO only for all-failure groups."""

    if (
        successes.ndim != 1
        or utilities.shape != successes.shape
        or successes.numel() < 2
        or not bool(torch.isfinite(utilities).all())
    ):
        raise RewardProtocolError("invalid binary-first progress credit group")
    binary = leave_one_out_binary_advantages(successes)
    count = int(successes.sum())
    if 0 < count < successes.numel():
        return binary.to(utilities), "mixed_binary"
    if count == successes.numel():
        return torch.zeros_like(utilities), "all_success_zero"
    semantic = (successes.numel() * utilities - utilities.sum()) / (
        successes.numel() - 1
    )
    return semantic, "all_failure_semantic"


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    left = 0
    while left < array.size:
        right = left + 1
        while right < array.size and array[order[right]] == array[order[left]]:
            right += 1
        ranks[order[left:right]] = (left + right - 1) / 2.0
        left = right
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise RewardProtocolError("invalid progress-credit nuisance correlation")
    x = _average_ranks(left)
    y = _average_ranks(right)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.sqrt(np.square(x).sum() * np.square(y).sum()))
    return float(np.dot(x, y) / denominator) if denominator else 0.0


def _group_diagnostic_rows(
    rows: Sequence[Mapping[str, Any]], gates: Mapping[str, Any]
) -> dict[int, list[Mapping[str, Any]]]:
    expected_rows = int(gates["task_count"]) * int(gates["rollouts_per_task"])
    if len(rows) != expected_rows:
        raise RewardProtocolError("progress diagnostic lost Cartesian rollout rows")
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    identities = set()
    for row in rows:
        if row.get("schema_version") != PROGRESS_DIAGNOSTIC_ROW_SCHEMA:
            raise RewardProtocolError("progress diagnostic row schema changed")
        task_id = int(row["global_task_id"])
        cursor = int(row["rollout_cursor"])
        identity = (task_id, cursor)
        if identity in identities:
            raise RewardProtocolError("progress diagnostic duplicated a rollout")
        identities.add(identity)
        grouped[task_id].append(row)
    if (
        len(grouped) != int(gates["task_count"])
        or any(len(values) != int(gates["rollouts_per_task"]) for values in grouped.values())
    ):
        raise RewardProtocolError("progress diagnostic lost a task condition")
    return dict(grouped)


def _outcome_summary(
    grouped: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    task_rows = []
    mixed = []
    all_failure = []
    all_success = []
    pair_wins = 0.0
    pair_count = 0
    mixed_agree = 0
    for task_id, values in sorted(grouped.items()):
        positive = [float(row["utility_correct"]) for row in values if row["success"]]
        negative = [float(row["utility_correct"]) for row in values if not row["success"]]
        utilities = [float(row["utility_correct"]) for row in values]
        observed = {
            "global_task_id": task_id,
            "successes": len(positive),
            "utility_min": min(utilities),
            "utility_max": max(utilities),
            "utility_range": max(utilities) - min(utilities),
            "teacher_change_energy": float(values[0]["teacher_change_energy"]),
            "observer_repeat_max_abs": max(
                float(row["observer_repeat_max_abs"]) for row in values
            ),
        }
        if positive and negative:
            mixed.append(observed)
            success_mean = float(np.mean(positive))
            failure_mean = float(np.mean(negative))
            observed.update(
                success_utility_mean=success_mean,
                failure_utility_mean=failure_mean,
                success_mean_higher=success_mean > failure_mean,
            )
            mixed_agree += int(success_mean > failure_mean)
            comparisons = [
                float(success > failure) + 0.5 * float(success == failure)
                for success in positive
                for failure in negative
            ]
            pair_wins += sum(comparisons)
            pair_count += len(comparisons)
        elif positive:
            all_success.append(observed)
        else:
            all_failure.append(observed)
        task_rows.append(observed)
    return {
        "tasks": task_rows,
        "mixed": mixed,
        "all_failure": all_failure,
        "all_success": all_success,
        "mixed_agree": mixed_agree,
        "auc": pair_wins / pair_count if pair_count else 0.0,
    }


def _dispersion_summary(
    all_failure: Sequence[Mapping[str, Any]], gates: Mapping[str, Any]
) -> tuple[list[float], int, bool]:
    ranges = [float(row["utility_range"]) for row in all_failure]
    persistent = set(int(value) for value in gates["persistent_all_failure_task_ids"])
    minimum = float(gates["minimum_all_failure_range"])
    persistent_passes = sum(
        row["global_task_id"] in persistent and row["utility_range"] >= minimum
        for row in all_failure
    )
    passed = (
        sum(value >= minimum for value in ranges)
        >= int(gates["minimum_all_failure_tasks_with_range"])
        and float(np.median(ranges)) >= float(gates["minimum_all_failure_median_range"])
        and persistent_passes >= int(gates["minimum_persistent_tasks_with_range"])
    )
    return ranges, persistent_passes, passed


def _counterfactual_summary(
    rows: Sequence[Mapping[str, Any]], gates: Mapping[str, Any]
) -> tuple[dict[str, Any], bool]:
    successful = [row for row in rows if bool(row["success"])]
    result = {}
    for name in ("wrong", "shuffled", "reversed"):
        margins = [
            float(row["utility_correct"]) - float(row[f"utility_{name}"])
            for row in successful
        ]
        greater_fraction = sum(value > 0 for value in margins) / len(margins)
        median_margin = float(np.median(margins))
        result[name] = {
            "correct_greater_fraction": greater_fraction,
            "median_margin": median_margin,
            "passed": (
                greater_fraction
                >= float(gates["minimum_counterfactual_win_fraction"])
                and median_margin > 0
            ),
        }
    return result, all(value["passed"] for value in result.values())


def summarize_progress_diagnostic(
    rows: Sequence[Mapping[str, Any]],
    *,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the pre-registered read-only mechanism gates from 24 x K4 rows."""

    grouped = _group_diagnostic_rows(rows, gates)
    outcome = _outcome_summary(grouped)
    task_rows = outcome["tasks"]
    mixed = outcome["mixed"]
    all_failure = outcome["all_failure"]
    all_success = outcome["all_success"]
    successes = sum(bool(row["success"]) for row in rows)

    identity_gate = (
        successes == int(gates["expected_successes"])
        and len(mixed) == int(gates["expected_mixed_tasks"])
        and len(all_success) == int(gates["expected_all_success_tasks"])
        and len(all_failure) == int(gates["expected_all_failure_tasks"])
    )
    content_gate = all(
        row["teacher_change_energy"] > float(gates["minimum_teacher_change_energy"])
        and row["observer_repeat_max_abs"] <= float(gates["maximum_repeat_abs"])
        for row in task_rows
    )
    auc = float(outcome["auc"])
    mixed_agree = int(outcome["mixed_agree"])
    binary_gate = (
        mixed_agree >= int(gates["minimum_mixed_task_agreements"])
        and auc >= float(gates["minimum_success_failure_auc"])
    )
    ranges, persistent_passes, dispersion_gate = _dispersion_summary(
        all_failure, gates
    )
    counterfactuals, counterfactual_gate = _counterfactual_summary(rows, gates)

    all_failure_task_ids = {int(row["global_task_id"]) for row in all_failure}
    failure_rows = [
        row for row in rows if int(row["global_task_id"]) in all_failure_task_ids
    ]
    pixel_spearman = _spearman(
        [float(row["utility_correct"]) for row in failure_rows],
        [float(row["pixel_change_rms"]) for row in failure_rows],
    )
    nuisance_gate = abs(pixel_spearman) < float(gates["maximum_abs_pixel_spearman"])
    gate_rows = {
        "paired_k4_identity": identity_gate,
        "finite_content": content_gate,
        "binary_agreement": binary_gate,
        "all_failure_dispersion": dispersion_gate,
        "video_counterfactual": counterfactual_gate,
        "non_pixel_shortcut": nuisance_gate,
    }
    return {
        "schema_version": PROGRESS_DIAGNOSTIC_RESULT_SCHEMA,
        "row_count": len(rows),
        "task_count": len(grouped),
        "successes": successes,
        "mixed_tasks": len(mixed),
        "all_success_tasks": len(all_success),
        "all_failure_tasks": len(all_failure),
        "mixed_task_success_mean_higher": mixed_agree,
        "success_failure_pair_auc": auc,
        "all_failure_ranges": ranges,
        "persistent_all_failure_range_passes": persistent_passes,
        "counterfactuals": counterfactuals,
        "failure_utility_pixel_spearman": pixel_spearman,
        "gates": gate_rows,
        "passed": all(gate_rows.values()),
        "tasks": task_rows,
    }
