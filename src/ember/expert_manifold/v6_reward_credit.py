"""Success-only on-policy Program guards for the active OSG-PC Writer."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from lerobot.utils.constants import ACTION

from ember.expert_manifold.contract import ExpertManifoldError
from ember.lora import LoRAContract
from ember.reward.loss import functional_executed_prefix_flow_loss
from ember.reward.protocol import RewardProtocolError, flow_sample_seed


_PROGRAM_SHAPE = (320, 256)


@dataclass(frozen=True)
class SuccessRetentionCreditSummary:
    """Per-task evidence for successful on-policy executed-prefix cotangents."""

    successes: int
    failures: int
    success_episode_ids: tuple[int, ...]
    replay_chunks: int
    flow_panel_chunks: int
    flow_panel_row_indices: tuple[int, ...]
    executed_action_steps: int
    mc_samples: int
    functional_policy_forwards: int
    episode_objectives: tuple[float, ...]
    lora_gradient_rms: tuple[float, ...]
    program_cotangent_rms: tuple[float, ...] = ()


@dataclass(frozen=True)
class SuccessGuardProjectionSummary:
    """KKT and source-descent evidence for one task-local cone projection."""

    constraint_count: int
    active_constraint_count: int
    active_constraint_ordinals: tuple[int, ...]
    raw_feasible: bool
    changed: bool
    blind_direction_rms: float
    safe_direction_rms: float
    safe_to_blind_norm_ratio: float
    blind_safe_cosine: float
    source_descent_ratio: float
    maximum_constraint_value: float


def _flow_sample_panel(
    policy: torch.nn.Module,
    *,
    count: int,
    mc_samples: int,
    seed_root: int,
    cycle: int,
    global_task_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate the complete keyed task panel before physical slicing."""

    model_config = getattr(getattr(policy, "model", None), "config", None)
    if (
        count <= 0
        or mc_samples != 4
        or model_config is None
        or float(model_config.time_sampling_beta_alpha) != 1.5
        or float(model_config.time_sampling_beta_beta) != 1.0
        or float(model_config.time_sampling_scale) != 0.999
        or float(model_config.time_sampling_offset) != 0.001
    ):
        raise RewardProtocolError("OSG-PC flow sample panel changed")
    shape = (
        count,
        int(policy.config.chunk_size),
        int(policy.config.max_action_dim),
    )
    noises = []
    times = []
    for mc_index in range(mc_samples):
        generator = torch.Generator(device=device).manual_seed(
            flow_sample_seed(
                seed_root,
                cycle=cycle,
                global_task_id=global_task_id,
                mc_index=mc_index,
            )
        )
        noises.append(
            torch.randn(
                shape,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
        )
        uniform = torch.rand(
            count,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        times.append(uniform.pow(2.0 / 3.0).mul_(0.999).add_(0.001))
    return torch.stack(noises), torch.stack(times)


def _batch_slice_to_device(
    batch: Mapping[str, torch.Tensor],
    start: int,
    stop: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: value[start:stop].to(device=device, non_blocking=True)
        for name, value in batch.items()
    }


def _validated_success_panel(
    batch: Mapping[str, torch.Tensor],
    episode_ids: torch.Tensor,
    successes: torch.Tensor,
    panel_row_indices: torch.Tensor,
    panel_total_chunks: int,
) -> tuple[torch.Tensor | None, tuple[int, ...]]:
    if (
        successes.ndim != 1
        or successes.shape != (4,)
        or not bool(torch.isfinite(successes).all())
        or bool(((successes != 0) & (successes != 1)).any())
    ):
        raise RewardProtocolError("OSG-PC requires four binary outcomes")
    success_ids = tuple(
        int(value) for value in torch.nonzero(successes, as_tuple=False).flatten()
    )
    if (
        episode_ids.ndim != 1
        or episode_ids.dtype != torch.long
        or panel_row_indices.ndim != 1
        or panel_row_indices.dtype != torch.long
        or panel_row_indices.shape != episode_ids.shape
        or type(panel_total_chunks) is not int
        or panel_total_chunks <= 0
        or bool((panel_row_indices < 0).any())
        or bool((panel_row_indices >= panel_total_chunks).any())
        or panel_row_indices.unique().numel() != panel_row_indices.numel()
    ):
        raise RewardProtocolError("OSG-PC replay episode IDs changed")
    ids = episode_ids
    if not success_ids:
        if ids.numel() or batch:
            raise RewardProtocolError("all-failure OSG-PC replay was retained")
        return None, success_ids
    action = batch.get(ACTION)
    valid = batch.get("executed_action_steps")
    if (
        not isinstance(action, torch.Tensor)
        or action.ndim != 3
        or not isinstance(valid, torch.Tensor)
        or valid.shape != (action.shape[0],)
        or ids.shape != valid.shape
        or ids.numel() == 0
        or bool((valid <= 0).any())
        or bool((valid > action.shape[1]).any())
        or set(ids.detach().cpu().tolist()) != set(success_ids)
        or bool((ids < 0).any())
        or bool((ids >= 4).any())
    ):
        raise RewardProtocolError("OSG-PC success-only replay changed")
    return valid, success_ids


def functional_success_lora_gradients(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    batch: Mapping[str, torch.Tensor],
    episode_ids: torch.Tensor,
    successes: torch.Tensor,
    *,
    panel_row_indices: torch.Tensor,
    panel_total_chunks: int,
    mc_samples: int,
    physical_microbatch_size: int,
    flow_seed_root: int,
    cycle: int,
    global_task_id: int,
    device: torch.device,
) -> tuple[tuple[dict[str, torch.Tensor], ...], SuccessRetentionCreditSummary]:
    """Differentiate one independent keep loss for every successful K4 episode."""

    if any(parameter.requires_grad for parameter in policy.parameters()):
        raise RewardProtocolError("OSG-PC functional policy must remain frozen")
    if physical_microbatch_size <= 0 or mc_samples != 4:
        raise RewardProtocolError("invalid OSG-PC retention batch contract")
    valid, success_ids = _validated_success_panel(
        batch,
        episode_ids,
        successes,
        panel_row_indices,
        panel_total_chunks,
    )
    if not success_ids:
        return (), SuccessRetentionCreditSummary(
            successes=0,
            failures=4,
            success_episode_ids=(),
            replay_chunks=0,
            flow_panel_chunks=panel_total_chunks,
            flow_panel_row_indices=(),
            executed_action_steps=0,
            mc_samples=mc_samples,
            functional_policy_forwards=0,
            episode_objectives=(),
            lora_gradient_rms=(),
            program_cotangent_rms=(),
        )
    if valid is None:
        raise RewardProtocolError("invalid OSG-PC retention batch contract")
    action = batch[ACTION]
    count = int(action.shape[0])
    ids = episode_ids.to(dtype=torch.long)
    counts = torch.bincount(ids, minlength=4)
    if any(int(counts[episode]) <= 0 for episode in success_ids):
        raise RewardProtocolError("OSG-PC lost a successful episode")
    complete_noises, complete_times = _flow_sample_panel(
        policy,
        count=panel_total_chunks,
        mc_samples=mc_samples,
        seed_root=flow_seed_root,
        cycle=cycle,
        global_task_id=global_task_id,
        device=device,
    )
    panel_rows = panel_row_indices.to(device=device, non_blocking=True)
    noises = complete_noises.index_select(1, panel_rows)
    times = complete_times.index_select(1, panel_rows)
    names = tuple(state)
    leaves = {
        name: value.detach().requires_grad_(True) for name, value in state.items()
    }
    gradient_sums = {
        episode: {
            name: torch.zeros_like(value, dtype=torch.float32)
            for name, value in state.items()
        }
        for episode in success_ids
    }
    objectives = {
        episode: torch.zeros((), dtype=torch.float32, device=device)
        for episode in success_ids
    }
    forwards = 0
    for start in range(0, count, physical_microbatch_size):
        stop = min(start + physical_microbatch_size, count)
        sliced = _batch_slice_to_device(batch, start, stop, device)
        sliced_ids = ids[start:stop].to(device=device, non_blocking=True)
        present = tuple(
            episode
            for episode in success_ids
            if bool((sliced_ids == episode).any())
        )
        for mc_index in range(mc_samples):
            per_chunk, _ = functional_executed_prefix_flow_loss(
                policy,
                leaves,
                contract,
                sliced,
                noise=noises[mc_index, start:stop],
                time=times[mc_index, start:stop],
                validate_prefix_values=False,
                collect_details=False,
            )
            per_chunk = per_chunk.to(dtype=torch.float32)
            for present_index, episode in enumerate(present):
                scalar = per_chunk[sliced_ids == episode].sum() / (
                    mc_samples * int(counts[episode])
                )
                gradients = torch.autograd.grad(
                    scalar,
                    tuple(leaves[name] for name in names),
                    retain_graph=present_index + 1 < len(present),
                )
                objectives[episode].add_(scalar.detach())
                for name, gradient in zip(names, gradients, strict=True):
                    gradient_sums[episode][name].add_(
                        gradient.to(dtype=torch.float32)
                    )
            forwards += 1
    gradient_rows = tuple(gradient_sums[episode] for episode in success_ids)
    gradient_rms = []
    objective_values = []
    for episode, gradients in zip(success_ids, gradient_rows, strict=True):
        squared = torch.stack(
            [value.square().sum() for value in gradients.values()]
        ).sum()
        count_values = sum(value.numel() for value in gradients.values())
        rms = float((squared / count_values).sqrt().detach().cpu())
        objective = float(objectives[episode].detach().cpu())
        if not math.isfinite(rms) or rms <= 0 or not math.isfinite(objective):
            raise ExpertManifoldError(
                "successful OSG-PC replay produced invalid LoRA credit"
            )
        gradient_rms.append(rms)
        objective_values.append(objective)
    return gradient_rows, SuccessRetentionCreditSummary(
        successes=len(success_ids),
        failures=4 - len(success_ids),
        success_episode_ids=success_ids,
        replay_chunks=count,
        flow_panel_chunks=panel_total_chunks,
        flow_panel_row_indices=tuple(
            int(value) for value in panel_row_indices.detach().cpu().tolist()
        ),
        executed_action_steps=int(valid.sum()),
        mc_samples=mc_samples,
        functional_policy_forwards=forwards,
        episode_objectives=tuple(objective_values),
        lora_gradient_rms=tuple(gradient_rms),
        program_cotangent_rms=(),
    )


def _validated_program_direction(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != _PROGRAM_SHAPE
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all())
    ):
        raise ExpertManifoldError(f"invalid OSG-PC {name}")
    return value.detach()


def _projection_summary(
    blind: torch.Tensor,
    safe: torch.Tensor,
    rows: torch.Tensor | None,
    *,
    active: tuple[int, ...],
    raw_feasible: bool,
    changed: bool,
) -> SuccessGuardProjectionSummary:
    blind64 = blind.flatten().to(dtype=torch.float64)
    safe64 = safe.flatten().to(dtype=torch.float64)
    blind_norm = torch.linalg.vector_norm(blind64)
    safe_norm = torch.linalg.vector_norm(safe64)
    blind_energy = blind_norm.square()
    inner = torch.dot(blind64, safe64)
    if float(blind_norm) == 0:
        norm_ratio = 1.0
        cosine = 1.0
        descent_ratio = 1.0
    else:
        norm_ratio = float(safe_norm / blind_norm)
        cosine = (
            float(inner / (blind_norm * safe_norm))
            if float(safe_norm) > 0
            else 0.0
        )
        descent_ratio = float(inner / blind_energy)
    maximum = 0.0 if rows is None else float((rows @ safe64).max())
    count = 0 if rows is None else int(rows.shape[0])
    return SuccessGuardProjectionSummary(
        constraint_count=count,
        active_constraint_count=len(active),
        active_constraint_ordinals=active,
        raw_feasible=raw_feasible,
        changed=changed,
        blind_direction_rms=float(blind64.square().mean().sqrt()),
        safe_direction_rms=float(safe64.square().mean().sqrt()),
        safe_to_blind_norm_ratio=norm_ratio,
        blind_safe_cosine=cosine,
        source_descent_ratio=descent_ratio,
        maximum_constraint_value=maximum,
    )


@torch.no_grad()
def project_blind_program_direction(
    blind_direction: torch.Tensor,
    retention_cotangents: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, SuccessGuardProjectionSummary]:
    """Project a blind Program descent into all successful keep half-spaces."""

    blind = _validated_program_direction(
        blind_direction, name="blind Program direction"
    )
    if len(retention_cotangents) > 4:
        raise ExpertManifoldError("OSG-PC has more than four success guards")
    if not retention_cotangents:
        safe = blind.clone()
        return safe, _projection_summary(
            blind, safe, None, active=(), raw_feasible=True, changed=False
        )
    normalized_rows = []
    for ordinal, value in enumerate(retention_cotangents):
        row = _validated_program_direction(
            value, name=f"retention cotangent {ordinal}"
        ).flatten().to(dtype=torch.float64)
        norm = torch.linalg.vector_norm(row)
        if not bool(torch.isfinite(norm)) or float(norm) <= 0:
            raise ExpertManifoldError("OSG-PC success guard has zero energy")
        normalized_rows.append(row / norm)
    rows = torch.stack(normalized_rows)
    blind64 = blind.flatten().to(dtype=torch.float64)
    blind_norm = torch.linalg.vector_norm(blind64)
    direction_scale = max(
        float(blind_norm), torch.finfo(torch.float64).tiny
    )
    raw_values = rows @ blind64
    fp64_tolerance = (
        262144
        * torch.finfo(torch.float64).eps
        * direction_scale
    )
    distance_tolerance = (
        262144
        * torch.finfo(torch.float64).eps
        * direction_scale
        * direction_scale
    )
    if bool((raw_values <= fp64_tolerance).all()):
        safe = blind.clone()
        return safe, _projection_summary(
            blind, safe, rows, active=(), raw_feasible=True, changed=False
        )

    best: torch.Tensor | None = None
    best_active: tuple[int, ...] = ()
    best_distance = math.inf
    for active_count in range(1, rows.shape[0] + 1):
        for active in itertools.combinations(range(rows.shape[0]), active_count):
            indices = torch.tensor(active, dtype=torch.long, device=rows.device)
            selected = rows.index_select(0, indices)
            gram = selected @ selected.transpose(0, 1)
            rhs = selected @ blind64
            multipliers = torch.linalg.pinv(gram, hermitian=True) @ rhs
            candidate = blind64 - selected.transpose(0, 1) @ multipliers
            active_residual = selected @ candidate
            all_residual = rows @ candidate
            if (
                not bool(torch.isfinite(candidate).all())
                or bool((multipliers < -fp64_tolerance).any())
                or bool((active_residual.abs() > 32 * fp64_tolerance).any())
                or bool((all_residual > 32 * fp64_tolerance).any())
            ):
                continue
            distance = float((candidate - blind64).square().sum())
            if distance < best_distance - distance_tolerance:
                best = candidate
                best_active = active
                best_distance = distance
    if best is None:
        raise ExpertManifoldError("OSG-PC cone projection found no KKT solution")
    safe = best.reshape(_PROGRAM_SHAPE).to(dtype=torch.float32)
    output64 = safe.flatten().to(dtype=torch.float64)
    constraint_tolerance = (
        64
        * torch.finfo(torch.float32).eps
        * direction_scale
    )
    residual = rows @ output64
    inner = torch.dot(blind64, output64)
    safe_energy = torch.dot(output64, output64)
    safe_norm = torch.linalg.vector_norm(output64)
    dot_scale = max(
        float(blind_norm * safe_norm),
        float(safe_energy),
        torch.finfo(torch.float64).tiny,
    )
    dot_tolerance = 128 * torch.finfo(torch.float32).eps * dot_scale
    if (
        bool((residual > constraint_tolerance).any())
        or float(inner) < -dot_tolerance
        or float(inner + dot_tolerance) < float(safe_energy)
        or not bool(torch.isfinite(safe).all())
    ):
        raise ExpertManifoldError("OSG-PC FP32 projection violates its KKT contract")
    return safe, _projection_summary(
        blind,
        safe,
        rows,
        active=best_active,
        raw_feasible=False,
        changed=not torch.equal(blind, safe),
    )


def success_retention_is_finite(summary: SuccessRetentionCreditSummary) -> bool:
    return all(
        math.isfinite(value)
        for value in (
            *summary.episode_objectives,
            *summary.lora_gradient_rms,
            *summary.program_cotangent_rms,
        )
    )


def success_projection_is_finite(summary: SuccessGuardProjectionSummary) -> bool:
    return all(
        math.isfinite(value)
        for value in (
            summary.blind_direction_rms,
            summary.safe_direction_rms,
            summary.safe_to_blind_norm_ratio,
            summary.blind_safe_cosine,
            summary.source_descent_ratio,
            summary.maximum_constraint_value,
        )
    )
