"""Registered seven independent parent-state task lookahead cases."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ember.pi05_source_checkpoint import read_json


REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = Path("configs/metatask_lookahead_credit_v1/experiment_spec.json")


def authority() -> dict[str, Any]:
    spec = read_json(REPO_ROOT / SPEC_PATH)
    if (spec.get("schema_version") != "ember_metatask_lookahead_credit_v1"
            or spec.get("study_id") != "metatask_lookahead_credit_20260925"
            or spec.get("status") != "registered_phase0_only"
            or spec["parent"]["macro"] != 1155 or spec["parent"]["world_size"] != 2
            or [row["source_macro"] for row in spec["cases"]] != list(range(1156, 1163))
            or len(spec["cases"]) != 7
            or spec["readouts"]["parameter_states"] !=
            ["P", "TASK_A", "TASK_B", "MIX_A", "MIX_B", "BASE", "TASK", "MIX"]
            or spec["readouts"]["random_tau_full_H_fm_rows"] != 3584
            or spec["readouts"]["true_flow_predictions"] != 448):
        raise ValueError("metatask lookahead phase0 registration changed")
    return spec


def inspect_cases(spec: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[tuple[dict, ...], ...]:
    """Resolve actual source events, including later macros, without moving the sampler."""
    plan = read_json(Path(spec["parent"]["event_plan"]))
    run = read_json(Path(spec["parent"]["checkpoint"]).parent.parent / "run_contract.json")
    if (run["git"]["commit"] != spec["parent"]["training_commit"]
            or run["config"] != config or run["topology"]["world_size"] != 2
            or plan["task_ids"] != sorted(spec["data"]["gradient_tasks"])
            or len(spec["data"]["gradient_tasks"]) != 28):
        raise ValueError("lookahead parent training or gradient task authority changed")
    all_cases = []
    for case in spec["cases"]:
        indices = plan["groups"][case["source_macro"] - 1]
        if indices != case["event_indices"] or len(indices) != 4:
            raise ValueError("lookahead source macro event indices changed")
        draws = []
        for position, index in enumerate(indices):
            event = plan["events"][index]
            if (event["task"] != case["tasks"][position]
                    or event["teacher_demo"] != case["teacher_demos"][position]
                    or event["occurrence"] != case["occurrences"][position]
                    or not set(event["action_demos"]) <= set(range(46))
                    or not set(event["teaching"]["action_demos"]) <= set(range(46))
                    or len(event["action_demos"]) != 21
                    or len(event["teaching"]["action_demos"]) != 7):
                raise ValueError("lookahead source event or gradient wall changed")
            draws.append({"job_id": position, "condition_index": 0, "task": event["task"],
                          "occurrence": event["occurrence"],
                          "video_demos": (event["teacher_demo"],),
                          "query_seed": event["query_seed"], "query_offset": 0,
                          "query_count": 21, "teaching_offset": 0,
                          "teaching_count": 7, "frames": event["frames"],
                          "source_macro": case["source_macro"], "source_event_index": index,
                          "source_event": event})
        all_cases.append(tuple(draws))
    flattened = [draw["task"] for group in all_cases for draw in group]
    if sorted(flattened) != sorted(spec["data"]["gradient_tasks"]) or len(set(flattened)) != 28:
        raise ValueError("seven source cases must cover the fit28 tasks exactly once")
    return tuple(all_cases)


def query_weights(group: str, side: str, position: int) -> tuple[float, ...]:
    if group not in ("TASK", "MIX") or side not in ("A", "B") or position not in range(4):
        raise ValueError("unregistered lookahead query group")
    chosen = 0 if side == "A" else 1
    if group == "TASK":
        value = 2.0 if position % 2 == chosen else 0.0
        return (value,) * 28
    return tuple(2.0 if (ordinal + position) % 2 == chosen else 0.0
                 for ordinal in range(28))


def weight_audit() -> dict[str, Any]:
    result = {}
    for group in ("TASK", "MIX"):
        a = [query_weights(group, "A", position) for position in range(4)]
        b = [query_weights(group, "B", position) for position in range(4)]
        if (sum(sum(row) for row in a) != 112 or sum(sum(row) for row in b) != 112
                or any(x + y != 2 for left, right in zip(a, b, strict=True)
                       for x, y in zip(left, right, strict=True))):
            raise ValueError("lookahead groups do not partition the 112 original queries")
        result[group] = {"A_selected": 56, "B_selected": 56,
                         "per_query_group_coefficient": 1 / 42,
                         "original_query_coefficient": 1 / 84}
    return result


def tensor_norm(values: Sequence[torch.Tensor]) -> float:
    return math.sqrt(sum(float(value.double().square().sum()) for value in values))


def preconditioner(parameters: Sequence[torch.nn.Parameter], optimizer, *, step: int) -> tuple[torch.Tensor, ...]:
    beta2 = float(optimizer.param_groups[0]["betas"][1])
    eps = float(optimizer.param_groups[0]["eps"])
    result = []
    for parameter in parameters:
        state = optimizer.state[parameter]
        if int(state["step"]) != step or state["exp_avg_sq"].shape != parameter.shape:
            raise ValueError("parent Adam second moment or step changed")
        variance = state["exp_avg_sq"].detach().float() / (1 - beta2 ** step)
        result.append((variance.sqrt() + eps).reciprocal())
    return tuple(result)


def virtual_alpha(displacement, preconditioner_values, grad_a, grad_b) -> tuple[float, float, bool]:
    length = tensor_norm(displacement)
    ga = tuple(p * g for p, g in zip(preconditioner_values, grad_a, strict=True))
    gb = tuple(p * g for p, g in zip(preconditioner_values, grad_b, strict=True))
    denominator = math.sqrt((tensor_norm(ga) ** 2 + tensor_norm(gb) ** 2) / 2)
    if not math.isfinite(length) or not math.isfinite(denominator):
        raise ValueError("nonfinite lookahead displacement or parent preconditioner")
    zero = length == 0 or denominator == 0
    return (0.0 if zero else length / denominator), denominator, zero
