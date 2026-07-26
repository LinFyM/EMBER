"""Functional action loss from a Writer-generated task-local LoRA."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from ember.batched_lora import BatchedLoRAInference
from ember.lora import (
    LoRAContract,
    functional_lora_call,
    inject_task_lora,
    task_lora_state_dict,
)
from ember.writer.model import CompleteLoRAWriter, WriterModelError


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
) -> tuple[torch.Tensor, Mapping[str, Any], dict[str, torch.Tensor]]:
    """Differentiate one policy loss only through detached LoRA leaf tensors.

    This lets the caller evaluate several adapters sequentially without keeping
    several PI05 activation graphs resident. Backpropagating the returned leaf
    gradients through the Writer is the exact first derivative by the chain
    rule; no policy parameter is trainable or accumulated.
    """

    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise WriterModelError("functional LoRA gradient received a trainable policy")
    leaves = {
        name: value.detach().requires_grad_(True) for name, value in state.items()
    }
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


def batched_functional_lora_loss_gradients(
    policy: torch.nn.Module,
    generated: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    *,
    batch: Mapping[str, Any],
    pair_to_generated: torch.Tensor,
    executor: BatchedLoRAInference,
) -> tuple[torch.Tensor, Mapping[str, Any], dict[str, torch.Tensor]]:
    """Evaluate N per-sample adapter views with shared policy noise and targets.

    ``pair_to_generated`` has shape ``[action_batch, views]``.  Each column is
    one ordinary policy forward with a distinct adapter for every action row.
    Exact duplicate videos may point at the same generated adapter row; their
    logical losses remain separate and their gradients are summed.
    """

    if (
        any(parameter.requires_grad for parameter in policy.parameters())
        or not generated
        or pair_to_generated.ndim != 2
        or pair_to_generated.dtype != torch.long
        or pair_to_generated.numel() == 0
    ):
        raise WriterModelError("invalid batched functional LoRA request")
    batch_size, views = pair_to_generated.shape
    unique_count = next(iter(generated.values())).shape[0]
    if (
        unique_count <= 0
        or int(pair_to_generated.min()) < 0
        or int(pair_to_generated.max()) >= unique_count
        or any(value.shape[0] != unique_count for value in generated.values())
    ):
        raise WriterModelError("generated LoRA batch and pair map differ")

    gradients = {
        name: torch.zeros_like(value)
        for name, value in generated.items()
    }
    losses = []
    details = []
    cpu_before = torch.get_rng_state()
    device = next(iter(generated.values())).device
    cuda_before = (
        torch.cuda.get_rng_state(device)
        if device.type == "cuda"
        else None
    )
    cpu_after: torch.Tensor | None = None
    cuda_after: torch.Tensor | None = None
    for view in range(views):
        torch.set_rng_state(cpu_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state(cuda_before, device)
        indices = pair_to_generated[:, view]
        leaves = {
            name: value.index_select(0, indices).detach().requires_grad_(True)
            for name, value in generated.items()
        }
        per_sample_states = tuple(
            {name: value[row] for name, value in leaves.items()}
            for row in range(batch_size)
        )
        with executor.activate(per_sample_states):
            output = policy(dict(batch))
        if (
            not isinstance(output, tuple)
            or len(output) != 2
            or not isinstance(output[0], torch.Tensor)
            or output[0].ndim != 0
            or not isinstance(output[1], Mapping)
        ):
            raise WriterModelError("batched functional policy returned invalid loss")
        names = tuple(leaves)
        leaf_gradients = torch.autograd.grad(
            output[0],
            tuple(leaves[name] for name in names),
        )
        for name, gradient in zip(names, leaf_gradients, strict=True):
            if not bool(torch.isfinite(gradient).all()):
                raise WriterModelError(
                    "batched functional policy produced non-finite LoRA gradients"
                )
            gradients[name].index_add_(
                0,
                indices,
                gradient.detach() / views,
            )
        losses.append(output[0].detach())
        details.append(dict(output[1]))
        if cpu_after is None:
            cpu_after = torch.get_rng_state()
            if cuda_before is not None:
                cuda_after = torch.cuda.get_rng_state(device)
    if cpu_after is None:
        raise WriterModelError("batched functional policy executed no views")
    torch.set_rng_state(cpu_after)
    if cuda_after is not None:
        torch.cuda.set_rng_state(cuda_after, device)

    mean_loss = torch.stack(losses).mean()
    loss_per_dim = [
        row.get("loss_per_dim")
        for row in details
        if isinstance(row.get("loss_per_dim"), list)
    ]
    detail: dict[str, Any] = {
        "loss": float(mean_loss),
        "view_losses": [float(value) for value in losses],
    }
    if len(loss_per_dim) == views:
        detail["loss_per_dim"] = (
            torch.tensor(loss_per_dim, dtype=torch.float32)
            .mean(dim=0)
            .tolist()
        )
    return mean_loss, detail, gradients


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
