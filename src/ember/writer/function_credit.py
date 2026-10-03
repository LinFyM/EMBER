"""Main FM and endpoint video-teaching credit through one complete LoRA."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from ember.ecp.policy_effects import ExecutionPolicyPrefix, prepare_prefix_kv_cache
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


def flow_sample(policy, batch, *, seed: int, device, random_batch: int, offset: int,
                noise_endpoint: bool = False) -> FlowSample:
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
        time = (torch.ones(len(actions), device=actions.device) if noise_endpoint
                else policy.model.sample_time(len(actions), actions.device))
    return FlowSample(
        (images, masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], actions, noise, time),
        noise - actions, int(policy.config.output_features[ACTION].shape[0]),
    )


def mean_velocity_loss(prediction: Tensor, target: Tensor, width: int, *, prefix_steps: int | None = None) -> Tensor:
    """Mean over real action dimensions and the full or explicitly taught horizon."""
    if prediction.shape != target.shape or prediction.ndim != 3 or not 0 < width <= prediction.shape[-1]:
        raise ValueError("paired flow output lost native horizon/action dimensions")
    if prefix_steps is not None and (type(prefix_steps) is not int or not 0 < prefix_steps <= prediction.shape[1]):
        raise ValueError("teaching prefix must fit the actual native horizon")
    return (prediction[:, :prefix_steps, :width].float()
            - target[:, :prefix_steps, :width].float()).square().mean()


def effect_velocity_loss(prediction: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """Registered query mean of valid displacement3; no padding coordinates."""
    residual = (prediction[..., 7:10].float() - target[..., 7:10].float()).square()
    return ((residual * valid[..., None]).sum((1, 2)) / (valid.sum(1) * 3).clamp_min(1)).mean()


class NativeFlowPrediction(nn.Module):
    """Official FM velocity with frozen prefix KV and a differentiable action suffix."""

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def prepare(self, sample: FlowSample) -> tuple[Tensor, Any, Tensor, Tensor]:
        """Prepare one frozen query prefix and noisy action for both LoRA suffixes."""
        images, masks, tokens, token_masks, actions, noise, time = sample.arguments
        core = self.policy.model
        # Official prefix queries cannot attend to action tokens. All execution
        # LoRAs belong to the suffix, so this KV has no LoRA gradient to retain.
        with torch.no_grad():
            embeddings, padding, _ = core.embed_prefix(images, masks, tokens, token_masks)
            prefix = ExecutionPolicyPrefix(embeddings, padding)
            cache = prepare_prefix_kv_cache(
                self.policy, prefix,
                native_precision=not torch.is_autocast_enabled(embeddings.device.type),
            )
        time_expanded = time[:, None, None]
        noisy_actions = time_expanded * noise + (1 - time_expanded) * actions
        return padding, cache, noisy_actions, time

    def forward(self, sample: FlowSample, prepared=None) -> Tensor:
        padding, cache, noisy_actions, time = self.prepare(sample) if prepared is None else prepared
        velocity = self.policy.model.denoise_step(padding, cache, noisy_actions, time)
        if velocity.shape != sample.target.shape:
            raise RuntimeError("native flow suffix changed its actual action horizon")
        return velocity


def _add(destination: dict[str, Tensor], values: Mapping[str, Tensor], weight: float) -> None:
    for name, value in values.items():
        if name not in destination:
            destination[name] = value.detach().float().mul(weight)
        else:
            destination[name].add_(value.detach().float(), alpha=weight)


def paired_functional_credit(policy, state, contract, batch, *,
                             seed: int, device, random_batch: int, offset: int, microbatch: int,
                             condition_weight: float, backward: bool = True,
                             noise_endpoint: bool = False, prefix_steps: int | None = None,
                             query_weights: Sequence[float] | None = None,
                             effect_target: Tensor | None = None,
                             effect_valid: Tensor | None = None) -> dict[str, Any]:
    """Return one separately normalized loss and weighted complete-LoRA cotangent."""
    return _functional_credit(policy, state, contract, batch, seed=seed, device=device,
                              random_batch=random_batch, offset=offset, microbatch=microbatch,
                              condition_weight=condition_weight, backward=backward,
                              noise_endpoint=noise_endpoint, prefix_steps=prefix_steps,
                              query_weights=query_weights, effect_target=effect_target,
                              effect_valid=effect_valid)


def dual_functional_credit(policy, state, public_state, contract, batch, *,
                           seed: int, device, random_batch: int, offset: int, microbatch: int,
                           condition_weight: float, backward: bool = True) -> tuple[dict, dict]:
    """Full/public cotangents share one FM sample and frozen prefix per microbatch."""
    full = _functional_credit(policy, state, contract, batch, seed=seed, device=device,
                              random_batch=random_batch, offset=offset, microbatch=microbatch,
                              condition_weight=condition_weight, backward=backward,
                              public_state=public_state)
    return full, full.pop("public_credit")


def _functional_credit(policy, state, contract, batch, *,
                             seed: int, device, random_batch: int, offset: int, microbatch: int,
                             condition_weight: float, backward: bool = True,
                             noise_endpoint: bool = False, prefix_steps: int | None = None,
                             query_weights: Sequence[float] | None = None,
                             public_state: Mapping[str, Tensor] | None = None,
                             effect_target: Tensor | None = None,
                             effect_valid: Tensor | None = None) -> dict[str, Any]:
    """Return full credit, optionally public credit on the same sample/prefix KV."""
    states = (state,) if public_state is None else (state, public_state)
    for values in states:
        validate_lora_state(values, contract)
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise ValueError("functional credit requires a frozen physical policy")
    total, chunk = functional_microbatch_contract(
        batch, microbatch, policy_rng_seed=seed,
        flow_time_sampling_scheme=INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
        flow_noise_sampling_scheme=INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
        policy_random_batch_size=random_batch, policy_batch_offset=offset,
    )
    if query_weights is not None and (len(query_weights) != total or
            any(not math.isfinite(float(weight)) or float(weight) < 0 for weight in query_weights)):
        raise ValueError("functional query weights must cover the unchanged logical batch")
    owner = NativeFlowPrediction(policy)
    if effect_target is not None and (effect_target.shape != (total, 50, 3)
            or effect_valid is None or effect_valid.shape != (total, 50)
            or public_state is not None or query_weights is not None or prefix_steps is not None
            or not torch.isfinite(effect_target).all()):
        raise ValueError('registered joint-effect target/mask consumer changed')
    credits = [{"flow_loss": 0., "lora_cotangent": {}, "source_forward_calls": 0,
                "compiled_forward_calls": 0} for _ in states]
    for start in range(0, total, chunk):
        stop = min(total, start + chunk)
        sliced = {name: value[start:stop] if isinstance(value, Tensor) and value.ndim and len(value) == total else value
                  for name, value in batch.items()}
        sample = flow_sample(policy, sliced, seed=seed, device=device, random_batch=random_batch,
                             offset=offset + start, noise_endpoint=noise_endpoint)
        valid = None
        if effect_target is not None:
            actions = sample.arguments[4].clone()
            actions[..., 7:10] = effect_target[start:stop].to(actions)
            sample = FlowSample((*sample.arguments[:4], actions, *sample.arguments[5:]),
                                sample.arguments[5] - actions, sample.action_width)
            valid = effect_valid[start:stop].to(device=device, dtype=torch.bool)
        prepared = owner.prepare(sample)
        weight = (stop - start) / total
        for values, credit in zip(states, credits, strict=True):
            leaves = {name: value.detach().requires_grad_(backward) for name, value in values.items()}
            with torch.set_grad_enabled(backward):
                prediction = torch.func.functional_call(
                    owner, {"policy." + name: value for name, value in leaves.items()},
                    (sample, prepared), strict=False)
                credit["compiled_forward_calls"] += 1
                if query_weights is None:
                    value = mean_velocity_loss(prediction, sample.target, sample.action_width,
                                               prefix_steps=prefix_steps)
                else:
                    residual = (prediction[:, :prefix_steps, :sample.action_width].float()
                                - sample.target[:, :prefix_steps, :sample.action_width].float()).square()
                    weights = torch.as_tensor(query_weights[start:stop], device=residual.device,
                                              dtype=residual.dtype)
                    value = (residual.mean(dim=(1, 2)) * weights).mean()
                action_value = value
                effect_value = value.new_zeros(())
                if valid is not None:
                    effect_value = effect_velocity_loss(prediction, sample.target, valid)
                    value = value + effect_value
                if backward:
                    gradients = torch.autograd.grad(value, tuple(leaves.values()))
                    _add(credit["lora_cotangent"], dict(zip(leaves, gradients, strict=True)),
                         weight * condition_weight)
            credit["flow_loss"] += float(value.detach()) * weight
            credit['action_flow_loss'] = credit.get('action_flow_loss', 0.) + float(action_value.detach()) * weight
            credit['effect_flow_loss'] = credit.get('effect_flow_loss', 0.) + float(effect_value.detach()) * weight
    if public_state is not None:
        credits[0]["public_credit"] = credits[1]
    return credits[0]
