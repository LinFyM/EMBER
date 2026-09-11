"""Native paired FM predictions and distinct compiled/teacher cotangents.

Every source and compiled prediction uses the official PI05 model.forward and
the same action/noise/time sample. No query feature enters video compilation.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from ember.lora import validate_lora_state
from ember.writer.functional import (
    INDEPENDENT_BETA_TIME_SAMPLING_SCHEME, INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    functional_microbatch_contract, scoped_policy_flow_noise_sampling,
    scoped_policy_flow_time_sampling, scoped_policy_randomness,
)


@dataclass
class FlowSample:
    arguments: tuple
    target: Tensor
    action_width: int


def flow_sample(policy, batch, *, seed: int, device, random_batch: int, offset: int) -> FlowSample:
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, masks = policy._preprocess_images(dict(batch))
    actions = policy.prepare_action(batch)
    with ExitStack() as stack:
        stack.enter_context(scoped_policy_randomness(seed, device))
        stack.enter_context(scoped_policy_flow_noise_sampling(
            policy, INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
            logical_batch_size=random_batch, batch_offset=offset))
        stack.enter_context(scoped_policy_flow_time_sampling(
            policy, INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
            logical_batch_size=random_batch, batch_offset=offset))
        noise = policy.model.sample_noise(actions.shape, actions.device)
        time = policy.model.sample_time(len(actions), actions.device)
    return FlowSample(
        (images, masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], actions, noise, time),
        noise - actions, int(policy.config.output_features[ACTION].shape[0]),
    )


def mean_velocity_loss(prediction: Tensor, target: Tensor, width: int) -> Tensor:
    """Official PI05 mean over the complete horizon and actual action dimensions."""
    if prediction.shape != target.shape or prediction.ndim != 3 or not 0 < width <= prediction.shape[-1]:
        raise ValueError("paired flow output lost native horizon/action dimensions")
    return (prediction[..., :width].float() - target[..., :width].float()).square().mean()


class NativeFlowPrediction(nn.Module):
    """Expose velocity and its actual normalized input from one native forward."""

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, sample: FlowSample) -> tuple[Tensor, Tensor]:
        captured = []

        def capture(module, args, output):
            captured.append((output, args[0]))

        handle = self.policy.model.action_out_proj.register_forward_hook(capture)
        try:
            self.policy.model(*sample.arguments)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("paired flow requires one actual action_out_proj forward")
        velocity, query = captured[0]
        if velocity.shape != sample.target.shape or query.shape[:2] != velocity.shape[:2]:
            raise RuntimeError("native flow capture changed its actual action horizon")
        return velocity, query


def _add(destination: dict[str, Tensor], values: Mapping[str, Tensor], weight: float) -> None:
    for name, value in values.items():
        if name not in destination:
            destination[name] = value.detach().float().mul(weight)
        else:
            destination[name].add_(value.detach().float(), alpha=weight)


def paired_functional_credit(policy, state, contract, batch, *, reader, memory, prior,
                             seed: int, device, random_batch: int, offset: int, microbatch: int,
                             condition_weight: float, auxiliary_weight: float, rho: float,
                             backward: bool = True) -> dict[str, Any]:
    """Return L_C/L_D LoRA cotangents and L_R memory cotangent separately.

    Auxiliary parameter gradients accumulate here. Video memory is an ephemeral
    leaf; its L_R cotangent is returned for the one shared encoder replay.
    Source features and teacher targets have no policy/teacher gradient route
    through distillation. Pure-FM uses no reader and performs no source forward.
    """
    validate_lora_state(state, contract)
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise ValueError("functional credit requires a frozen physical policy")
    total, chunk = functional_microbatch_contract(
        batch, microbatch, policy_rng_seed=seed,
        flow_time_sampling_scheme=INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
        flow_noise_sampling_scheme=INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
        policy_random_batch_size=random_batch, policy_batch_offset=offset,
    )
    owner, losses = NativeFlowPrediction(policy), {"flow_loss": 0., "reader_loss": 0., "distill_loss": 0., "source_loss": 0.}
    gradient_c, gradient_d = {}, {}
    memory_leaf = None if reader is None else memory.detach().requires_grad_(backward)
    calls = 0
    for start in range(0, total, chunk):
        stop = min(total, start + chunk)
        sliced = {name: value[start:stop] if isinstance(value, Tensor) and value.ndim and len(value) == total else value
                  for name, value in batch.items()}
        sample = flow_sample(policy, sliced, seed=seed, device=device, random_batch=random_batch, offset=offset + start)
        weight = (stop - start) / total
        teacher = None
        if reader is not None:
            with torch.no_grad():
                base, query = owner(sample)
            calls += 1
            with torch.set_grad_enabled(backward):
                teacher = reader(query.detach(), base.detach(), memory_leaf, prior.detach())
                loss_r = mean_velocity_loss(teacher, sample.target, sample.action_width)
            if backward:
                (loss_r * (weight * condition_weight * auxiliary_weight)).backward()
            teacher = teacher.detach()
            losses["reader_loss"] += float(loss_r.detach()) * weight
            losses["source_loss"] += float(mean_velocity_loss(base, sample.target, sample.action_width)) * weight
            del loss_r, base, query
        leaves = {name: value.detach().requires_grad_(backward) for name, value in state.items()}
        with torch.set_grad_enabled(backward):
            prediction, _ = torch.func.functional_call(
                owner, {"policy." + name: value for name, value in leaves.items()}, (sample,), strict=False)
            calls += 1
            loss_c = mean_velocity_loss(prediction, sample.target, sample.action_width)
            loss_d = None if teacher is None else mean_velocity_loss(prediction, teacher, sample.action_width)
            if backward:
                grad_c = torch.autograd.grad(loss_c, tuple(leaves.values()), retain_graph=rho > 0)
                _add(gradient_c, dict(zip(leaves, grad_c, strict=True)), weight * condition_weight)
                if rho > 0:
                    grad_d = torch.autograd.grad(loss_d, tuple(leaves.values()))
                    _add(gradient_d, dict(zip(leaves, grad_d, strict=True)), weight * condition_weight)
        losses["flow_loss"] += float(loss_c.detach()) * weight
        if loss_d is not None:
            losses["distill_loss"] += float(loss_d.detach()) * weight
        del prediction, leaves, sample, teacher, loss_c, loss_d
    return {**losses, "lora_cotangent": gradient_c, "distill_cotangent": gradient_d,
            "memory_cotangent": None if memory_leaf is None else memory_leaf.grad,
            "source_forward_calls": 0 if reader is None else calls // 2,
            "compiled_forward_calls": calls if reader is None else calls // 2,
            "rho": rho}
