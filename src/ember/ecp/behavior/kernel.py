"""Decoder-free policy-behavior topology for the deployed Natural Program."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F


PROGRAM_BLOCKS = (
    "P_lang",
    "P_scene",
    "sqrt_rho_P_process",
    "sqrt_rho_sigma",
    "rho",
    "tau",
)


def _unit_block(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.float().flatten(start_dim=2), dim=-1, eps=1e-6)


def program_behavior_features(
    program: Any, selected_targets: Sequence[int]
) -> torch.Tensor:
    """Return one fixed block-equal feature per condition and selected owner."""

    targets = torch.as_tensor(
        tuple(map(int, selected_targets)),
        dtype=torch.long,
        device=program.p_process.device,
    )
    if targets.shape != (8,) or int(targets.min()) < 0 or int(targets.max()) >= 38:
        raise ValueError("behavior-kernel target contract changed")
    rho = program.rho.float().clamp(0.0, 1.0)
    event_weight = rho.sqrt()[:, :, None, None]
    process = (
        program.p_process.float().index_select(2, targets) * event_weight
    ).transpose(1, 2)
    sigma = (
        program.sigma.float().index_select(2, targets) * event_weight
    ).transpose(1, 2)
    owner_count = targets.numel()
    rho_block = rho[:, None].expand(-1, owner_count, -1)
    tau_block = program.tau.float()[:, None].expand(-1, owner_count, -1, -1)
    blocks = (
        _unit_block(program.p_lang.float().index_select(1, targets)),
        _unit_block(program.p_scene.float().index_select(1, targets)),
        _unit_block(process),
        _unit_block(sigma),
        _unit_block(rho_block),
        _unit_block(tau_block),
    )
    feature = torch.cat(blocks, dim=-1) / math.sqrt(len(blocks))
    if feature.shape[:2] != (program.p_process.shape[0], 8):
        raise ValueError("behavior-kernel Program feature shape changed")
    return F.normalize(feature, dim=-1, eps=1e-6)


def program_gram(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 3 or features.shape[1] != 8:
        raise ValueError("behavior-kernel feature batch changed")
    return torch.einsum("ntd,mtd->tnm", features.float(), features.float())


def normalized_centered_kernel(kernel: torch.Tensor) -> torch.Tensor:
    if kernel.ndim != 3 or kernel.shape[-1] != kernel.shape[-2]:
        raise ValueError("behavior kernel must be [target, task, task]")
    count = kernel.shape[-1]
    if count < 3:
        raise ValueError("behavior kernel requires at least three tasks")
    centered = (
        kernel
        - kernel.mean(-1, keepdim=True)
        - kernel.mean(-2, keepdim=True)
        + kernel.mean(dim=(-1, -2), keepdim=True)
    )
    mask = ~torch.eye(count, dtype=torch.bool, device=kernel.device)
    centered = centered.masked_fill(~mask[None], 0.0)
    norm = centered.square().sum(dim=(-1, -2), keepdim=True).sqrt().clamp_min(1e-6)
    return centered / norm


def kernel_correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (normalized_centered_kernel(left) * normalized_centered_kernel(right)).sum(
        dim=(-1, -2)
    )


def _gather_features(
    local_features: torch.Tensor,
    local_task_ids: torch.Tensor,
    *,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if local_features.shape[0] != local_task_ids.numel():
        raise ValueError("behavior-kernel local task ownership changed")
    if world_size == 1:
        return local_features, local_task_ids
    from torch.distributed.nn.functional import all_gather

    gathered_features = torch.cat(tuple(all_gather(local_features)), dim=0)
    gathered_ids = [torch.empty_like(local_task_ids) for _ in range(world_size)]
    dist.all_gather(gathered_ids, local_task_ids)
    return gathered_features, torch.cat(gathered_ids)


def distributed_behavior_kernel_loss(
    *,
    local_features: torch.Tensor,
    local_task_ids: torch.Tensor,
    authority: Any,
    world_size: int,
    cross_view_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align Program and behavior relations while keeping credit in Program."""

    if local_features.ndim != 4 or local_features.shape[1:3] != (2, 8):
        raise ValueError("behavior-kernel views must be [task,2,8,width]")
    features, task_ids = _gather_features(
        local_features, local_task_ids, world_size=world_size
    )
    losses = []
    view_correlations: dict[str, list[torch.Tensor]] = {"a": [], "b": []}
    cross_view = []
    for role, role_ids in (
        ("meta", authority.meta_gradient_task_ids),
        ("target", authority.target_gradient_task_ids),
    ):
        del role
        selection = torch.tensor(
            [int(value) in role_ids for value in task_ids.detach().cpu().tolist()],
            dtype=torch.bool,
            device=features.device,
        )
        if int(selection.sum()) < 3:
            raise ValueError("behavior-kernel optimizer step lost a task role")
        ids = task_ids[selection]
        teacher_a = authority.kernel(ids, kind="panel_a")
        teacher_consensus = authority.kernel(ids, kind="consensus")
        normalized_teachers = tuple(
            normalized_centered_kernel(value) for value in (teacher_a, teacher_consensus)
        )
        programs = []
        for view, name in enumerate(("a", "b")):
            program = program_gram(features[selection, view])
            normalized = normalized_centered_kernel(program)
            programs.append(normalized)
            for teacher in normalized_teachers:
                losses.append((normalized - teacher).square().sum(dim=(-1, -2)).mean())
            view_correlations[name].append(
                (normalized * normalized_teachers[-1]).sum(dim=(-1, -2)).mean()
            )
        cross_view.append(
            (programs[0] - programs[1]).square().sum(dim=(-1, -2)).mean()
        )
    alignment = torch.stack(losses).mean()
    view_loss = torch.stack(cross_view).mean()
    total = alignment + float(cross_view_weight) * view_loss
    metrics = {
        "behavior_kernel_alignment_loss": float(alignment.detach()),
        "behavior_kernel_cross_view_loss": float(view_loss.detach()),
        "behavior_kernel_correlation_a": float(
            torch.stack(view_correlations["a"]).mean().detach()
        ),
        "behavior_kernel_correlation_b": float(
            torch.stack(view_correlations["b"]).mean().detach()
        ),
    }
    return total, metrics


def kernel_ridge_predictions(
    *,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    query_features: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Fixed evaluator-only readout; no held target enters the solve."""

    if (
        train_features.ndim != 3
        or train_targets.ndim != 3
        or query_features.ndim != 3
        or train_features.shape[:2] != train_targets.shape[:2]
        or train_features.shape[1] != query_features.shape[1]
        or ridge <= 0
    ):
        raise ValueError("behavior-kernel ridge contract changed")
    predictions = []
    for target in range(train_features.shape[1]):
        train = train_features[:, target].float()
        query = query_features[:, target].float()
        gram = train @ train.T
        scale = gram.diag().mean().clamp_min(1e-6)
        system = gram + float(ridge) * scale * torch.eye(
            gram.shape[0], device=gram.device, dtype=gram.dtype
        )
        coefficient = torch.linalg.solve(system, train_targets[:, target].float())
        predictions.append((query @ train.T) @ coefficient)
    return torch.stack(predictions, dim=1)


def topology_summary(
    *,
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    task_ids: torch.Tensor,
    authority: Any,
    roles: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"by_role": {}}
    for role, ids in roles.items():
        selection = torch.tensor(
            [int(value) in set(map(int, ids)) for value in task_ids.tolist()],
            dtype=torch.bool,
            device=features_a.device,
        )
        selected_ids = task_ids[selection]
        teacher = authority.kernel(selected_ids, kind="panel_b")
        first = program_gram(features_a[selection])
        second = program_gram(features_b[selection])
        result["by_role"][role] = {
            "program_to_behavior_a": float(kernel_correlation(first, teacher).mean()),
            "program_to_behavior_b": float(kernel_correlation(second, teacher).mean()),
            "cross_view": float(kernel_correlation(first, second).mean()),
            "task_count": int(selection.sum()),
        }
    for metric in ("program_to_behavior_a", "program_to_behavior_b", "cross_view"):
        result[f"role_equal_{metric}"] = sum(
            float(value[metric]) for value in result["by_role"].values()
        ) / len(result["by_role"])
    return result
