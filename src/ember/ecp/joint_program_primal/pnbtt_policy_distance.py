"""Paired frozen-policy velocity distance for PNBTT preservation."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Mapping

import torch

from ember.lora import LoRAContract, functional_lora_call
from ember.writer.errors import WriterModelError
from ember.writer.functional import (
    _scoped_policy_detail_collection,
    functional_microbatch_contract,
    scoped_policy_flow_noise_sampling,
    scoped_policy_flow_time_sampling,
    scoped_policy_randomness,
)


def _batch_slice(
    batch: Mapping[str, Any], *, start: int, stop: int, logical_batch_size: int
) -> dict[str, Any]:
    return {
        name: (
            value[start:stop]
            if isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == logical_batch_size
            else value
        )
        for name, value in batch.items()
    }


def _action_width(policy: torch.nn.Module) -> int:
    from lerobot.utils.constants import ACTION

    try:
        width = int(policy.config.output_features[ACTION].shape[0])
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise WriterModelError("PNBTT policy-distance action width changed") from error
    if width <= 0:
        raise WriterModelError("PNBTT policy-distance action width is invalid")
    return width


def _velocity_call(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, Any],
    *,
    policy_rng_seed: int,
    policy_rng_device: torch.device,
    flow_time_sampling_scheme: str,
    flow_noise_sampling_scheme: str,
    logical_batch_size: int,
    batch_offset: int,
    physical_microbatching: bool,
    track_gradient: bool,
) -> torch.Tensor:
    model = getattr(policy, "model", None)
    action_out = getattr(model, "action_out_proj", None)
    if not isinstance(action_out, torch.nn.Module):
        raise WriterModelError("PNBTT policy-distance action projection changed")
    captured: list[torch.Tensor] = []

    def capture(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        if not isinstance(output, torch.Tensor):
            raise WriterModelError("PNBTT policy velocity is not a tensor")
        captured.append(output)

    handle = action_out.register_forward_hook(capture)
    random_batch_size = logical_batch_size if physical_microbatching else None
    gradient_scope = nullcontext() if track_gradient else torch.no_grad()
    try:
        with gradient_scope:
            with scoped_policy_randomness(policy_rng_seed, policy_rng_device):
                with scoped_policy_flow_noise_sampling(
                    policy,
                    flow_noise_sampling_scheme,
                    logical_batch_size=random_batch_size,
                    batch_offset=batch_offset,
                ):
                    with scoped_policy_flow_time_sampling(
                        policy,
                        flow_time_sampling_scheme,
                        logical_batch_size=random_batch_size,
                        batch_offset=batch_offset,
                    ):
                        with _scoped_policy_detail_collection(policy, False):
                            functional_lora_call(policy, state, contract, batch)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise WriterModelError(
            f"PNBTT policy-distance captured {len(captured)} velocities"
        )
    velocity = captured[0]
    width = _action_width(policy)
    if velocity.ndim != 3 or velocity.shape[-1] < width:
        raise WriterModelError("PNBTT policy velocity shape changed")
    return velocity[..., :width]


def paired_policy_velocity_distance_gradient(
    policy: torch.nn.Module,
    generated_state: Mapping[str, torch.Tensor],
    carrier_state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    *,
    batch: Mapping[str, Any],
    policy_rng_seed: int,
    policy_rng_device: torch.device,
    flow_time_sampling_scheme: str,
    flow_noise_sampling_scheme: str,
    policy_microbatch_size: int,
    cached_carrier_velocity: torch.Tensor | None = None,
) -> tuple[float, dict[str, torch.Tensor], torch.Tensor]:
    """Differentiate paired velocity MSE only through generated LoRA leaves.

    Carrier and generated calls replay identical keyed PI05 flow time/noise.  A
    retained carrier velocity avoids an otherwise redundant frozen forward on
    later visits; it is an output cache, never a deployment input.
    """

    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise WriterModelError("PNBTT policy-distance policy must remain frozen")
    if set(generated_state) != set(carrier_state) or not generated_state:
        raise WriterModelError("PNBTT policy-distance LoRA state changed")
    logical_batch_size, microbatch_size = functional_microbatch_contract(
        batch,
        policy_microbatch_size,
        policy_rng_seed=policy_rng_seed,
        flow_time_sampling_scheme=flow_time_sampling_scheme,
        flow_noise_sampling_scheme=flow_noise_sampling_scheme,
    )
    if cached_carrier_velocity is not None and (
        cached_carrier_velocity.ndim != 3
        or cached_carrier_velocity.shape[0] != logical_batch_size
        or cached_carrier_velocity.shape[-1] != _action_width(policy)
        or not bool(torch.isfinite(cached_carrier_velocity).all())
    ):
        raise WriterModelError("PNBTT cached carrier velocity changed")

    names = tuple(generated_state)
    gradient_sum = {
        name: torch.zeros_like(generated_state[name]) for name in names
    }
    carrier_chunks: list[torch.Tensor] = []
    distance_sum = 0.0
    physical = microbatch_size < logical_batch_size
    for start in range(0, logical_batch_size, microbatch_size):
        stop = min(start + microbatch_size, logical_batch_size)
        microbatch = _batch_slice(
            batch,
            start=start,
            stop=stop,
            logical_batch_size=logical_batch_size,
        )
        if cached_carrier_velocity is None:
            carrier = _velocity_call(
                policy,
                {name: value.detach() for name, value in carrier_state.items()},
                contract,
                microbatch,
                policy_rng_seed=policy_rng_seed,
                policy_rng_device=policy_rng_device,
                flow_time_sampling_scheme=flow_time_sampling_scheme,
                flow_noise_sampling_scheme=flow_noise_sampling_scheme,
                logical_batch_size=logical_batch_size,
                batch_offset=start,
                physical_microbatching=physical,
                track_gradient=False,
            ).detach()
            carrier_chunks.append(carrier)
        else:
            carrier = cached_carrier_velocity[start:stop]
        leaves = {
            name: value.detach().requires_grad_(True)
            for name, value in generated_state.items()
        }
        generated = _velocity_call(
            policy,
            leaves,
            contract,
            microbatch,
            policy_rng_seed=policy_rng_seed,
            policy_rng_device=policy_rng_device,
            flow_time_sampling_scheme=flow_time_sampling_scheme,
            flow_noise_sampling_scheme=flow_noise_sampling_scheme,
            logical_batch_size=logical_batch_size,
            batch_offset=start,
            physical_microbatching=physical,
            track_gradient=True,
        )
        if generated.shape != carrier.shape:
            raise WriterModelError("PNBTT paired policy velocity shape changed")
        distance = (generated.float() - carrier.float()).square().mean()
        gradients = torch.autograd.grad(
            distance, tuple(leaves[name] for name in names)
        )
        weight = (stop - start) / logical_batch_size
        distance_sum += float(distance.detach()) * weight
        for name, gradient in zip(names, gradients, strict=True):
            gradient_sum[name].add_(gradient.to(gradient_sum[name]), alpha=weight)

    carrier_velocity = (
        torch.cat(carrier_chunks, dim=0)
        if cached_carrier_velocity is None
        else cached_carrier_velocity
    )
    if (
        carrier_velocity.shape[0] != logical_batch_size
        or not math.isfinite(distance_sum)
        or not bool(torch.isfinite(carrier_velocity).all())
        or not all(bool(torch.isfinite(value).all()) for value in gradient_sum.values())
    ):
        raise WriterModelError("PNBTT policy-distance became non-finite")
    return distance_sum, gradient_sum, carrier_velocity.detach()
