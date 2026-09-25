"""Registered score direction and independent one-step Writer candidates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ember.pi05_eval.exploration import exploration_covariance
from ember.pi05_source_checkpoint import read_json


def inspect_parent_events(spec: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Bind the eight FM events to original occurrence165, without resampling."""
    plan = read_json(Path(spec["parent"]["event_plan"]))
    run = read_json(Path(spec["parent"]["checkpoint"]).parent.parent / "run_contract.json")
    if (run["git"]["commit"] != spec["parent"]["training_commit"]
            or plan["task_ids"] != sorted(run["config"]["data"]["task_ids"])
            or len(plan["events"]) <= max(row["event_index"] for row in spec["data"]["FM_events"])):
        raise ValueError("return-credit parent event or train-task authority changed")
    resolved = []
    for registration in spec["data"]["FM_events"]:
        event = plan["events"][registration["event_index"]]
        expected = {key: registration[key] for key in
                    ("task", "occurrence", "teacher_demo", "query_seed")}
        if (any(event[key] != value for key, value in expected.items())
                or len(event["action_demos"]) != 21
                or len(event["teaching"]["action_demos"]) != 7
                or not set(event["action_demos"] + event["teaching"]["action_demos"]) <= set(range(46))):
            raise ValueError("return-credit FM event differs from original 21+7 legal query stream")
        resolved.append(event)
    if [row["task"] for row in resolved] != spec["data"]["gradient_tasks"]:
        raise ValueError("return-credit FM task set or ordering changed")
    return tuple(resolved)


def loo_advantages(rewards: Sequence[int | bool]) -> torch.Tensor:
    if len(rewards) != 4 or any(type(value) not in (bool, int) or value not in (0, 1)
                                for value in rewards):
        raise ValueError("return-credit LOO requires four official binary rewards")
    values = torch.tensor(rewards, dtype=torch.float32)
    return values - (values.sum() - values) / 3.0


def score_cotangent(latent: torch.Tensor, old_mean: torch.Tensor,
                     advantage: float, *, Q: int, M: int,
                     precision: torch.Tensor | None = None) -> torch.Tensor:
    """Positive-return VJP, including Q/M and the full 128 episode divisor."""
    if (latent.shape != (35,) or old_mean.shape != (35,)
            or not 0 < M <= min(4, Q) or not math.isfinite(float(advantage))):
        raise ValueError("return-credit saved decision or weight is invalid")
    precision = (torch.linalg.inv(exploration_covariance(dtype=torch.float32))
                 if precision is None else precision)
    if precision.shape != (35, 35):
        raise ValueError("return-credit Sigma inverse must be 35x35")
    delta = latent.detach().float() - old_mean.detach().float()
    return float(advantage) * (Q / M) / 128.0 * (precision @ delta)


def norm(values: Sequence[torch.Tensor]) -> float:
    return math.sqrt(sum(float(value.double().square().sum()) for value in values))


def dot(left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]) -> float:
    return sum(float((a.double() * b.double()).sum()) for a, b in zip(left, right, strict=True))


def candidate_step(parameters: Sequence[tuple[str, torch.nn.Parameter]],
                   parent: Mapping[str, torch.Tensor], direction: Mapping[str, torch.Tensor],
                   *, sign: int, radius: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply exactly one fresh zero-momentum/decay/clipping SGD step from P."""
    if sign not in (-1, 1) or set(direction) != set(parent) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("return-credit candidate direction is incomplete")
    ordered = [direction[name].detach().float() for name, _ in parameters]
    length = norm(ordered)
    if length == 0 or not math.isfinite(length):
        raise ValueError("return-credit candidate direction has zero or nonfinite norm")
    lr = radius / length
    for name, parameter in parameters:
        parameter.data.copy_(parent[name].to(parameter))
    optimizer = torch.optim.SGD([parameter for _, parameter in parameters],
                                lr=lr, momentum=0, weight_decay=0)
    optimizer.zero_grad(set_to_none=True)
    for name, parameter in parameters:
        parameter.grad = (-sign * direction[name]).to(parameter).clone()
    optimizer.step()
    displacement = {name: parameter.detach().float() - parent[name].float()
                    for name, parameter in parameters}
    actual = norm(tuple(displacement.values()))
    if not math.isfinite(actual) or abs(actual - radius) > max(1e-4, radius * 0.002):
        raise ValueError("fresh SGD candidate lost its registered parameter step length")
    return {"optimizer": "fresh_SGD_step1", "lr": lr, "momentum": 0,
            "weight_decay": 0, "clipping": False, "direction_sign": sign,
            "direction_norm": length, "actual_parameter_step_norm": actual,
            "parent_macro": 1155, "optimizer_history_inherited": False}, optimizer.state_dict()
