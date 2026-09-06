"""Exact functional credit with separate policy and Writer activation lifetimes.

The caller supplies one deterministic, complete-LoRA materialization at the
current optimizer step. This retains the existing leaf/VJP/replay mechanism;
it does not implement a Writer graph, cross-condition batching, or Meta caches.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.lora import LoRAContract
from ember.writer.functional import (
    functional_lora_loss_gradient,
    writer_chain_rule_surrogate,
)


def functional_objective(
    *, generated_loss: float, normalizer: float, task_weight: float,
) -> dict[str, float]:
    """Normalize positive functional credit while preserving global task mass."""

    if (
        not all(math.isfinite(float(value)) for value in (
            generated_loss, normalizer, task_weight,
        ))
        or normalizer <= 0
        or task_weight <= 0
    ):
        raise ValueError("shared Writer functional objective changed")
    return {
        "functional_normalized": generated_loss / normalizer,
        "gradient_mass": task_weight / normalizer,
    }


def functional_writer_backward(
    materialize: Callable[[], Mapping[str, torch.Tensor]],
    policy: torch.nn.Module,
    contract: LoRAContract,
    *,
    batch: Mapping[str, Any],
    normalizer: float,
    task_weight: float,
    policy_rng_seed: int | None = None,
    policy_rng_device: torch.device | str | None = None,
    flow_time_sampling_scheme: str | None = None,
    flow_noise_sampling_scheme: str | None = None,
    policy_microbatch_size: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Accumulate one condition's exact, globally weighted Writer gradient.

    Both materializations must use the same parameters and evidence with no
    intervening optimizer update. The callback owns its autocast/chunk replay
    context and must be deterministic, as in the retained original trainer.
    Learned observer responses must be recomputed with gradients during replay;
    this helper does not make a detached response cache differentiable.
    """

    objective = functional_objective(
        generated_loss=0.0, normalizer=normalizer, task_weight=task_weight,
    )
    with torch.no_grad():
        leaf_state = materialize()
    loss, details, gradients = functional_lora_loss_gradient(
        policy, leaf_state, contract, batch=batch,
        policy_rng_seed=policy_rng_seed,
        policy_rng_device=policy_rng_device,
        flow_time_sampling_scheme=flow_time_sampling_scheme,
        flow_noise_sampling_scheme=flow_noise_sampling_scheme,
        policy_microbatch_size=policy_microbatch_size,
        collect_policy_details=False,
    )
    if details or not bool(torch.isfinite(loss)):
        raise RuntimeError("shared Writer functional derivative changed")
    del leaf_state
    generated_state = materialize()
    surrogate = writer_chain_rule_surrogate(generated_state, gradients)
    (surrogate * objective["gradient_mass"]).backward()
    objective["functional_normalized"] = float(loss) / normalizer
    return loss, objective


def sum_writer_gradients(
    parameters: Sequence[torch.nn.Parameter], *, world_size: int,
) -> None:
    """SUM already globally weighted task gradients, including idle ranks.

    A task's weight is determined before device assignment. No extra world-size
    division is applied. All ranks must pass parameters in the same order.
    """

    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        elif not bool(torch.isfinite(parameter.grad).all()):
            raise RuntimeError("shared Writer produced a non-finite gradient")
        if world_size > 1:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
