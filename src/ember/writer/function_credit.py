"""Main FM and endpoint video-teaching credit through one complete LoRA."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Mapping

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


class NativeFlowPrediction(nn.Module):
    """Official FM velocity with frozen prefix KV and a differentiable action suffix."""

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, sample: FlowSample) -> Tensor:
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
        velocity = core.denoise_step(padding, cache, noisy_actions, time)
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
                             noise_endpoint: bool = False, prefix_steps: int | None = None) -> dict[str, Any]:
    """Return a separately normalized loss and its weighted complete-LoRA cotangent."""
    validate_lora_state(state, contract)
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise ValueError("functional credit requires a frozen physical policy")
    total, chunk = functional_microbatch_contract(
        batch, microbatch, policy_rng_seed=seed,
        flow_time_sampling_scheme=INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
        flow_noise_sampling_scheme=INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
        policy_random_batch_size=random_batch, policy_batch_offset=offset,
    )
    owner, loss, gradient, calls = NativeFlowPrediction(policy), 0., {}, 0
    for start in range(0, total, chunk):
        stop = min(total, start + chunk)
        sliced = {name: value[start:stop] if isinstance(value, Tensor) and value.ndim and len(value) == total else value
                  for name, value in batch.items()}
        sample = flow_sample(policy, sliced, seed=seed, device=device, random_batch=random_batch,
                             offset=offset + start, noise_endpoint=noise_endpoint)
        weight = (stop - start) / total
        leaves = {name: value.detach().requires_grad_(backward) for name, value in state.items()}
        with torch.set_grad_enabled(backward):
            prediction = torch.func.functional_call(
                owner, {"policy." + name: value for name, value in leaves.items()}, (sample,), strict=False)
            calls += 1
            value = mean_velocity_loss(prediction, sample.target, sample.action_width, prefix_steps=prefix_steps)
            if backward:
                gradients = torch.autograd.grad(value, tuple(leaves.values()))
                _add(gradient, dict(zip(leaves, gradients, strict=True)), weight * condition_weight)
        loss += float(value.detach()) * weight
    return {"flow_loss": loss, "lora_cotangent": gradient,
            "source_forward_calls": 0, "compiled_forward_calls": calls}
