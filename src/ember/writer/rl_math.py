"""Gaussian action credit and one-direction AdamW trust proposals.

Rollout, differentiated flow replay, distributed reduction and accepted-update
scheduling belong to the trainer. These helpers never query an environment or
differentiate a second time at a candidate parameter version.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Generic, Hashable, Mapping, Sequence, TypeVar

import torch
from torch import Tensor


def exploration_covariance(*, device=None, dtype=torch.float32) -> Tensor:
    """Fixed temporal-major covariance for five normalized seven-axis actions."""
    positions = torch.arange(5, device=device)
    temporal = 0.8 ** (positions[:, None] - positions[None, :]).abs().to(dtype)
    variances = torch.tensor([0.05 ** 2] * 6 + [0.10 ** 2], device=device, dtype=dtype)
    return torch.kron(temporal, torch.diag(variances))


_T = TypeVar("_T")


class DecisionReservoir(Generic[_T]):
    """Uniform decisions without replacement, independent of terminal reward.

    The caller supplies a dedicated RNG and stores its state in checkpoints.
    Items should already contain detached CPU observations and action evidence.
    """

    def __init__(self, rng: random.Random, capacity: int = 16) -> None:
        if capacity <= 0:
            raise ValueError("reservoir capacity must be positive")
        self.rng = rng
        self.capacity = capacity
        self.items: list[_T] = []
        self.total_seen = 0

    def add(self, item: _T) -> None:
        self.total_seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            index = self.rng.randrange(self.total_seen)
            if index < self.capacity:
                self.items[index] = item

    def sample(self, max_items: int = 4, *, rng: random.Random) -> list[_T]:
        """Choose trust evidence once; reuse that exact subset at all candidates."""
        if not 0 < max_items <= 4:
            raise ValueError("trust sample size must be between one and four")
        return rng.sample(self.items, min(max_items, len(self.items)))


def loo_advantages(rewards: Tensor) -> Tensor:
    """Four independent binary episode rewards; no advantage normalization."""
    if rewards.ndim == 0 or rewards.shape[-1] != 4:
        raise ValueError("LOO requires exactly four episodes per condition")
    rewards = rewards.detach().to(dtype=torch.float32)
    if not torch.all((rewards == 0) | (rewards == 1)):
        raise ValueError("only official binary success rewards are supported")
    return rewards - (rewards.sum(dim=-1, keepdim=True) - rewards) / 3


def _precision_for(reference: Tensor, precision: Tensor | None) -> Tensor:
    if reference.shape[-1] != 35:
        raise ValueError("action evidence must retain all 35 temporal-major coordinates")
    if precision is None:
        precision = torch.linalg.inv(exploration_covariance(device=reference.device, dtype=reference.dtype))
    if precision.shape != (35, 35):
        raise ValueError("precision must have shape [35,35]")
    return precision.to(device=reference.device, dtype=reference.dtype)


def rl_mean_cotangent(
    z: Tensor, old_mean: Tensor, advantage: Tensor | float,
    total_decisions: int, saved_decisions: int, *, precision: Tensor | None = None,
) -> Tensor:
    """Loss cotangent per saved decision, including RL/task/episode weights.

    Sum these cotangents through flow VJPs; do not average by Q or M again.
    z and old_mean are the *unclipped* Gaussian action and collection mean.
    """
    if z.shape != old_mean.shape or z.ndim == 0:
        raise ValueError("action and collection mean shapes must match")
    if not 0 < saved_decisions <= total_decisions:
        raise ValueError("decision sampling requires 0 < M <= Q")
    dtype = torch.float64 if z.dtype == torch.float64 else torch.float32
    delta = z.detach().to(dtype=dtype) - old_mean.detach().to(dtype=dtype)
    score = delta @ _precision_for(delta, precision).T
    weight = torch.as_tensor(advantage, device=delta.device, dtype=delta.dtype).detach()
    return (-0.1 / 16) * (total_decisions / saved_decisions) * weight[..., None] * score


def task_trust_kl(
    new_means: Tensor, old_means: Tensor, task_ids: Sequence[Hashable],
    episode_ids: Sequence[Hashable], *, precision: Tensor | None = None,
) -> dict[Hashable, float]:
    """Full-action KL: mean within episode, then equal episode mean per task.

    Inputs contain only the fixed uniformly selected trust decisions. Missing
    episodes are not invented as zero KL; caller checks global task coverage.
    """
    if new_means.ndim != 2 or new_means.shape != old_means.shape:
        raise ValueError("trust means must have matching [decisions,35] shapes")
    if len(task_ids) != len(new_means) or len(episode_ids) != len(new_means):
        raise ValueError("each trust decision needs task and episode identity")
    dtype = torch.float64 if new_means.dtype == torch.float64 else torch.float32
    delta = new_means.detach().to(dtype=dtype) - old_means.detach().to(dtype=dtype)
    values = (0.5 * (delta @ _precision_for(delta, precision)) * delta).sum(-1).cpu().tolist()
    episodes: dict[tuple[Hashable, Hashable], list[float]] = {}
    for task, episode, value in zip(task_ids, episode_ids, values, strict=True):
        episodes.setdefault((task, episode), []).append(value)
    tasks: dict[Hashable, list[float]] = {}
    for (task, _), episode_values in episodes.items():
        tasks.setdefault(task, []).append(sum(episode_values) / len(episode_values))
    return {task: sum(values) / len(values) for task, values in tasks.items()}


@dataclass(frozen=True)
class TrustAttempt:
    alpha: float
    task_kl: dict[Hashable, float]


@dataclass(frozen=True)
class TrustStepResult:
    accepted: bool
    alpha: float | None
    attempts: tuple[TrustAttempt, ...]


def _cpu_snapshot(value):
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return value


@torch.no_grad()
def adamw_trust_step(
    optimizer: torch.optim.AdamW,
    score_candidate: Callable[[], Mapping[Hashable, float]],
    *, max_task_kl: float = 0.02,
) -> TrustStepResult:
    """One AdamW direction; test alpha=1,.5,.25,.125 by fresh forward callbacks.

    Caller has already SUM-reduced and globally clipped the joint gradient.
    score_candidate sees current candidate parameters and must recompute its
    observer/Writer/flow means, using one fixed trust subset. It returns global
    task means (distributed gathering belongs to the caller), without gradients
    or optimizer/sampler mutations. Nonfinite scores reject that candidate.

    Old parameters, moments and the direction are shadowed on CPU. AdamW runs
    exactly once, and accepted scaled candidates retain those proposed moments
    and the single step. Rejection or an exception restores old optimizer state;
    exceptions are re-raised. Scheduler and sampler/RNG are never advanced or
    rolled back here: caller advances scheduling only on acceptance, but consumes
    its sampling RNG even on rejection. Gradients remain available for logging.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("trust proposal requires AdamW")
    if not math.isfinite(max_task_kl) or max_task_kl < 0:
        raise ValueError("trust bound must be finite and nonnegative")
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    old_parameters = [_cpu_snapshot(p) for p in parameters]
    old_optimizer = _cpu_snapshot(optimizer.state_dict())

    def restore() -> None:
        for parameter, old in zip(parameters, old_parameters, strict=True):
            parameter.copy_(old)
        optimizer.load_state_dict(old_optimizer)

    attempts: list[TrustAttempt] = []
    try:
        optimizer.step()
        direction = [p.detach().cpu() - old for p, old in zip(parameters, old_parameters, strict=True)]
        if not all(bool(torch.isfinite(d).all()) for d in direction):
            restore()
            return TrustStepResult(False, None, ())
        for alpha in (1.0, 0.5, 0.25, 0.125):
            if alpha != 1.0:
                for parameter, old, delta in zip(parameters, old_parameters, direction, strict=True):
                    parameter.copy_(old + alpha * delta)
            scores = {task: float(value) for task, value in score_candidate().items()}
            attempts.append(TrustAttempt(alpha, scores))
            if scores and all(math.isfinite(value) and 0 <= value <= max_task_kl for value in scores.values()):
                return TrustStepResult(True, alpha, tuple(attempts))
    except BaseException:
        restore()
        raise
    restore()
    return TrustStepResult(False, None, tuple(attempts))
