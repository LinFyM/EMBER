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


def _off_diagonal(kernel: torch.Tensor) -> torch.Tensor:
    count = kernel.shape[-1]
    mask = ~torch.eye(count, dtype=torch.bool, device=kernel.device)
    return kernel[:, mask]


def lifted_behavior_kernel(kernel: torch.Tensor) -> torch.Tensor:
    """Give every task a fixed common axis without erasing behavior scale."""

    return 0.5 * (kernel.float().clamp(-1.0, 1.0) + 1.0)


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


def _topology_scope_loss(
    *,
    features: torch.Tensor,
    task_ids: torch.Tensor,
    selection: torch.Tensor,
    authority: Any,
    teacher_scales: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if int(selection.sum()) < 3:
        raise ValueError("behavior-kernel optimizer step lost a topology scope")
    ids = task_ids[selection]
    raw_teachers = tuple(
        authority.kernel(ids, kind=kind).float()
        for kind in ("panel_a", "consensus")
    )
    calibrated_teachers = tuple(
        lifted_behavior_kernel(value) for value in raw_teachers
    )
    programs = []
    losses = []
    correlations = {}
    for view, name in enumerate(("a", "b")):
        program = program_gram(features[selection, view])
        programs.append(program)
        losses.extend(
            (
                (
                    _off_diagonal(program) - _off_diagonal(teacher)
                ) / scale[:, None]
            ).square().mean()
            for teacher, scale in zip(
                calibrated_teachers, teacher_scales, strict=True
            )
        )
        correlations[name] = kernel_correlation(program, raw_teachers[-1]).mean()
        correlations[f"program_std_{name}"] = _off_diagonal(program).std(-1).mean()
    correlations["teacher_std"] = teacher_scales[-1].mean()
    cross_view = (
        (
            _off_diagonal(programs[0]) - _off_diagonal(programs[1])
        ) / teacher_scales[-1][:, None]
    ).square().mean()
    return torch.stack(losses).mean(), cross_view, correlations


def distributed_behavior_kernel_loss(
    *,
    local_features: torch.Tensor,
    local_task_ids: torch.Tensor,
    authority: Any,
    world_size: int,
    cross_view_weight: float,
    scope_weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align Program and behavior relations while keeping credit in Program."""

    if local_features.ndim != 4 or local_features.shape[1:3] != (2, 8):
        raise ValueError("behavior-kernel views must be [task,2,8,width]")
    features, task_ids = _gather_features(
        local_features, local_task_ids, world_size=world_size
    )
    task_id_rows = task_ids.detach().cpu().tolist()
    selections = {
        "joint": torch.ones(
            task_ids.shape, dtype=torch.bool, device=features.device
        ),
        "meta": torch.tensor(
            [int(value) in authority.meta_gradient_task_ids for value in task_id_rows],
            dtype=torch.bool,
            device=features.device,
        ),
        "target": torch.tensor(
            [int(value) in authority.target_gradient_task_ids for value in task_id_rows],
            dtype=torch.bool,
            device=features.device,
        ),
    }
    scope_weights = {name: float(value) for name, value in scope_weights.items()}
    if set(scope_weights) != set(selections) or not math.isclose(
        sum(scope_weights.values()), 1.0
    ) or any(value <= 0.0 for value in scope_weights.values()):
        raise ValueError("behavior-kernel topology scope weights changed")
    alignments: dict[str, torch.Tensor] = {}
    cross_views: dict[str, torch.Tensor] = {}
    scope_correlations: dict[str, dict[str, torch.Tensor]] = {}
    global_scope_ids = {
        "joint": authority.fit_task_ids,
        "meta": tuple(sorted(authority.meta_gradient_task_ids)),
        "target": tuple(sorted(authority.target_gradient_task_ids)),
    }
    for scope, selection in selections.items():
        scale_ids = torch.tensor(
            global_scope_ids[scope], dtype=torch.long, device=features.device
        )
        teacher_scales = tuple(
            _off_diagonal(
                lifted_behavior_kernel(authority.kernel(scale_ids, kind=kind))
            ).std(-1).clamp_min(1e-3)
            for kind in ("panel_a", "consensus")
        )
        (
            alignments[scope],
            cross_views[scope],
            scope_correlations[scope],
        ) = _topology_scope_loss(
            features=features,
            task_ids=task_ids,
            selection=selection,
            authority=authority,
            teacher_scales=teacher_scales,
        )
    alignment = sum(
        scope_weights[scope] * value for scope, value in alignments.items()
    )
    view_loss = sum(
        scope_weights[scope] * value for scope, value in cross_views.items()
    )
    total = alignment + float(cross_view_weight) * view_loss
    metrics = {
        "behavior_kernel_alignment_loss": float(alignment.detach()),
        "behavior_kernel_cross_view_loss": float(view_loss.detach()),
        "behavior_kernel_correlation_a": float(
            sum(
                scope_weights[scope] * values["a"]
                for scope, values in scope_correlations.items()
            )
            .detach()
        ),
        "behavior_kernel_correlation_b": float(
            sum(
                scope_weights[scope] * values["b"]
                for scope, values in scope_correlations.items()
            )
            .detach()
        ),
    }
    for scope, correlations in scope_correlations.items():
        metrics[f"behavior_kernel_{scope}_correlation_a"] = float(
            correlations["a"].detach()
        )
        metrics[f"behavior_kernel_{scope}_correlation_b"] = float(
            correlations["b"].detach()
        )
        metrics[f"behavior_kernel_{scope}_program_std_a"] = float(
            correlations["program_std_a"].detach()
        )
        metrics[f"behavior_kernel_{scope}_program_std_b"] = float(
            correlations["program_std_b"].detach()
        )
        metrics[f"behavior_kernel_{scope}_teacher_std"] = float(
            correlations["teacher_std"].detach()
        )
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
