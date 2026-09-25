"""Differentiable native ten-step PI05 action flow for a complete LoRA state.

The frozen prefix is reused across Euler steps. Functional LoRA substitution
stays inside the checkpointed denoiser so backward sees the same adapter.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch.utils.checkpoint import checkpoint

from ember.ecp.policy_effects import ExecutionPolicyPrefix, prepare_prefix_kv_cache
from ember.lora import validate_lora_state


class _Denoiser(torch.nn.Module):
    def __init__(self, policy: torch.nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, padding, cache, action, time):
        return self.policy.model.denoise_step(padding, cache, action, time)


def flow_actions(policy: torch.nn.Module, state: Mapping[str, torch.Tensor], contract,
                 batch: Mapping[str, torch.Tensor], noise: torch.Tensor,
                 *, checkpoint_steps: bool = True) -> torch.Tensor:
    """Return all 50x32 coordinates after ten native Euler denoising steps."""
    validate_lora_state(state, contract)
    if noise.ndim != 3 or noise.shape[1:] != (50, 32):
        raise ValueError("native action flow requires complete 50x32 noise")
    with torch.autocast(noise.device.type, enabled=False):
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        images, masks = policy._preprocess_images(dict(batch))
        with torch.no_grad():
            embeddings, padding, _ = policy.model.embed_prefix(
                images, masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK])
        prefix = ExecutionPolicyPrefix(embeddings, padding)
        cache = prepare_prefix_kv_cache(policy, prefix, native_precision=True)
        owner = _Denoiser(policy)
        names = tuple(state)
        values = tuple(state[name].to(dtype=policy.get_parameter(name).dtype,
                                      memory_format=torch.contiguous_format)
                       for name in names)
        action = noise
        for step in range(10):
            time = torch.full((len(noise),), 1.0 - step * 0.1, device=noise.device)

            def denoise(current, when, *parameters):
                return torch.func.functional_call(
                    owner, {f"policy.{name}": value for name, value in
                            zip(names, parameters, strict=True)},
                    (prefix.padding, cache, current, when), strict=False)

            velocity = (checkpoint(denoise, action, time, *values, use_reentrant=False,
                                   preserve_rng_state=False)
                        if checkpoint_steps and torch.is_grad_enabled()
                        else denoise(action, time, *values))
            action = action - 0.1 * velocity
        return action


def flow_mean_lora_gradient(policy, state, contract, batch, noise, cotangent):
    """VJP from the actual first-five action means through all ten steps."""
    leaves = {name: value.detach().requires_grad_(True) for name, value in state.items()}
    output = flow_actions(policy, leaves, contract, batch, noise)[:, :5, :7]
    if cotangent.shape != (len(noise), 35):
        raise ValueError("score cotangent needs the full 35-vector")
    gradients = torch.autograd.grad(
        output, tuple(leaves.values()), grad_outputs=cotangent.detach().reshape_as(output))
    return {name: gradient.detach() for name, gradient in zip(leaves, gradients, strict=True)}, output.detach()
