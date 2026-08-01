"""Functional action loss from a Writer-generated task-local LoRA."""

from __future__ import annotations

import hashlib
import struct
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

import torch

from ember.lora import (
    LoRAContract,
    functional_lora_call,
    inject_task_lora,
    task_lora_state_dict,
)
from ember.writer.model import CompleteLoRAWriter, WriterModelError


TASK_QUERY_POLICY_RNG_SCHEME = "task_query_keyed_stateless_policy_cuda_v1"


def task_query_policy_rng_seed(
    *,
    optimization_seed: int,
    task_id: int,
    task_visit: int,
    demo_indices: Sequence[int],
    frame_indices: Sequence[int],
) -> int:
    """Derive one rank/phase-independent policy-noise seed from query identity."""

    demos = tuple(int(value) for value in demo_indices)
    frames = tuple(int(value) for value in frame_indices)
    if (
        optimization_seed < 0
        or task_id < 0
        or task_visit < 0
        or not demos
        or len(demos) != len(frames)
        or any(value < 0 for value in (*demos, *frames))
    ):
        raise WriterModelError("invalid task-query policy randomness identity")
    digest = hashlib.sha256()
    digest.update(TASK_QUERY_POLICY_RNG_SCHEME.encode("ascii"))
    digest.update(
        struct.pack(
            ">4Q",
            int(optimization_seed),
            int(task_id),
            int(task_visit),
            len(demos),
        )
    )
    for demo_index, frame_index in zip(demos, frames, strict=True):
        digest.update(struct.pack(">2Q", demo_index, frame_index))
    return int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)


@contextmanager
def scoped_policy_randomness(
    seed: int | None,
    device: torch.device | str | None,
) -> Iterator[None]:
    """Seed and restore only the policy forward's local CPU/CUDA generator."""

    if seed is None:
        yield
        return
    if seed < 0 or device is None:
        raise WriterModelError("invalid scoped policy randomness request")
    selected = torch.device(device)
    if selected.type == "cuda":
        index = selected.index
        if index is None:
            index = torch.cuda.current_device()
        with torch.random.fork_rng(devices=[index], device_type="cuda"):
            torch.cuda.default_generators[index].manual_seed(seed)
            yield
        return
    if selected.type != "cpu":
        raise WriterModelError("policy randomness device must be CPU or CUDA")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        yield


def prepare_frozen_writer_policy(
    policy: torch.nn.Module,
    contract: LoRAContract,
) -> dict[str, torch.Tensor]:
    """Inject the sealed identity LoRA and freeze all physical policy state."""

    inject_task_lora(policy, contract)
    template = task_lora_state_dict(policy, clone=True)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    policy.eval()
    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise WriterModelError("Writer functional policy must be fully frozen")
    return template


def writer_functional_action_loss(
    writer: CompleteLoRAWriter,
    policy: torch.nn.Module,
    contract: LoRAContract,
    *,
    language_features: torch.Tensor,
    video_features: torch.Tensor,
    episode_offsets: torch.Tensor,
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    """Run the frozen policy with the Writer's differentiable complete LoRA."""

    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise WriterModelError("functional action loss received a trainable policy")
    generated = writer(language_features, video_features, episode_offsets)
    output = functional_lora_call(
        policy,
        generated,
        contract,
        dict(batch),
    )
    if (
        not isinstance(output, tuple)
        or len(output) != 2
        or not isinstance(output[0], torch.Tensor)
        or output[0].ndim != 0
        or not isinstance(output[1], Mapping)
    ):
        raise WriterModelError("functional policy did not return a scalar loss")
    return output


def functional_lora_loss_gradient(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    *,
    batch: Mapping[str, Any],
    policy_rng_seed: int | None = None,
    policy_rng_device: torch.device | str | None = None,
) -> tuple[torch.Tensor, Mapping[str, Any], dict[str, torch.Tensor]]:
    """Differentiate one policy loss only through detached LoRA leaf tensors.

    Backpropagating the returned leaf gradients through the one-video Writer is
    the exact first derivative by the chain rule; no policy parameter is
    trainable or accumulated.
    """

    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise WriterModelError("functional LoRA gradient received a trainable policy")
    leaves = {
        name: value.detach().requires_grad_(True) for name, value in state.items()
    }
    with scoped_policy_randomness(policy_rng_seed, policy_rng_device):
        output = functional_lora_call(policy, leaves, contract, dict(batch))
    if (
        not isinstance(output, tuple)
        or len(output) != 2
        or not isinstance(output[0], torch.Tensor)
        or output[0].ndim != 0
        or not isinstance(output[1], Mapping)
    ):
        raise WriterModelError("functional policy did not return a scalar loss")
    names = tuple(leaves)
    gradients = torch.autograd.grad(output[0], tuple(leaves[name] for name in names))
    if any(not bool(torch.isfinite(value).all()) for value in gradients):
        raise WriterModelError("functional policy produced non-finite LoRA gradients")
    return (
        output[0].detach(),
        output[1],
        {name: gradient.detach() for name, gradient in zip(names, gradients, strict=True)},
    )

def writer_success_weighted_flow_loss(
    writer: CompleteLoRAWriter,
    policy: torch.nn.Module,
    contract: LoRAContract,
    *,
    language_features: torch.Tensor,
    video_features: torch.Tensor,
    episode_offsets: torch.Tensor,
    batch: Mapping[str, Any],
    rollout_episode_ids: torch.Tensor,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    """Sum equal-episode flow losses for successful on-policy trajectories."""

    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise WriterModelError("reward-weighted loss received a trainable policy")
    generated = writer(language_features, video_features, episode_offsets)
    output = functional_lora_call(
        policy,
        generated,
        contract,
        dict(batch),
        reduction="none",
    )
    if (
        not isinstance(output, tuple)
        or len(output) != 2
        or not isinstance(output[0], torch.Tensor)
        or output[0].ndim != 1
        or not isinstance(output[1], Mapping)
        or rollout_episode_ids.ndim != 1
        or rollout_episode_ids.shape != output[0].shape
    ):
        raise WriterModelError("reward-weighted policy did not return per-chunk loss")
    episode_ids = rollout_episode_ids.to(device=output[0].device, dtype=torch.long)
    unique = torch.unique(episode_ids, sorted=True)
    if unique.numel() == 0 or not torch.equal(
        unique, torch.arange(unique.numel(), device=unique.device)
    ):
        raise WriterModelError("reward-weighted episode IDs must be contiguous")
    episode_losses = torch.stack(
        [output[0][episode_ids == episode_id].mean() for episode_id in unique]
    )
    details = {
        **output[1],
        "successful_episodes": int(unique.numel()),
        "successful_chunks": int(output[0].numel()),
    }
    return episode_losses.sum(), details
