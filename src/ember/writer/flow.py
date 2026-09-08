"""Differentiable canonical ten-step PI05 flow with task-local LoRA leaves."""

from __future__ import annotations

from typing import Mapping

import torch
from torch.utils.checkpoint import checkpoint

from ember.ecp.policy_effects import prepare_execution_policy_prefix, prepare_prefix_kv_cache
from ember.lora import validate_lora_state


class _Denoise(torch.nn.Module):
    def __init__(self, policy: torch.nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, padding, cache, x, time):
        return self.policy.model.denoise_step(padding, cache, x, time)


def flow_actions(
    policy: torch.nn.Module, state: Mapping[str, torch.Tensor], contract,
    batch: Mapping[str, torch.Tensor], noise: torch.Tensor, *, checkpoint_steps: bool = True,
) -> torch.Tensor:
    """Keep all 50x32 coordinates connected through all ten Euler steps.

    Functional substitution happens inside each replayed step, so activation
    checkpointing never observes the physical identity adapter during backward.
    Prefix is frozen and shared across the ten steps of this observation batch.
    """
    validate_lora_state(state, contract)
    if noise.ndim != 3 or noise.shape[1:] != (50, 32):
        raise ValueError("execution flow requires complete 50x32 noise")
    # Match evaluator predict_action_chunk: native mixed parameter dtypes,
    # including FP32 action/time heads and materialized LoRA values.
    with torch.autocast(noise.device.type, enabled=False):
        prefix = prepare_execution_policy_prefix(policy, batch, native_precision=True)
        cache = prepare_prefix_kv_cache(policy, prefix, native_precision=True)
        owner = _Denoise(policy)
        names = tuple(state)
        values = tuple(state[name].to(dtype=policy.get_parameter(name).dtype,
                                      memory_format=torch.contiguous_format) for name in names)
        x = noise
        for step in range(10):
            timestep = torch.full((len(noise),), 1.0 - step * 0.1, device=noise.device)

            def denoise(current, time, *parameters):
                return torch.func.functional_call(
                    owner, {f"policy.{name}": value for name, value in zip(names, parameters, strict=True)},
                    (prefix.padding, cache, current, time), strict=False,
                )

            velocity = (
                checkpoint(denoise, x, timestep, *values, use_reentrant=False, preserve_rng_state=False)
                if checkpoint_steps and torch.is_grad_enabled()
                else denoise(x, timestep, *values)
            )
            x = x - 0.1 * velocity
        return x


def flow_mean_lora_gradient(policy, state, contract, batch, noise, mean_cotangent):
    """Exact score-mean VJP into both A/B sides; exploration and credit are fixed."""
    with torch.autocast(noise.device.type, enabled=False):
        leaves = {name: value.detach().requires_grad_(True) for name, value in state.items()}
        output = flow_actions(policy, leaves, contract, batch, noise)[:, :5, :7]
        if mean_cotangent.shape != (len(noise), 35):
            raise ValueError("Gaussian score needs the complete unclipped 35-vector")
        gradients = torch.autograd.grad(
            output, tuple(leaves.values()), grad_outputs=mean_cotangent.detach().reshape_as(output),
        )
        return {name: gradient.detach() for name, gradient in zip(leaves, gradients, strict=True)}
