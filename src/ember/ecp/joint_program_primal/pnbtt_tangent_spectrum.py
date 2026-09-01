"""Read-only functional-gradient spectrum diagnostic for PNBTT E1."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.ecp.bank_conditioning.key_value_replay import (
    differentiable_key_moments,
    safe_rms_normalize,
    signed_key_value_pool,
    whiten_queries,
)
from ember.ecp.bank_conditioning.tangent_transport import (
    PNBTT_FAMILIES,
    TangentTransportVideo,
    pnbtt_event_weights,
)
from ember.ecp.joint_program_primal.pnbtt_runtime import (
    PNBTT_E1_STAGE,
    PNBTTTaskLocalRuntime,
    prepare_pnbtt_tasklocal_runtime,
)
from ember.ecp.joint_program_primal.pnbtt_tasklocal import (
    PreparedPNBTTArm,
    _prepare_arm,
    local_tasks,
)
from ember.ecp.joint_program_primal.pnbtt_training import _functional_derivative
from ember.ecp.joint_program_primal.train_step import functional_panel_batch
from ember.ecp.native_factors import (
    G1_RESIDUAL_RANK,
    OUTPUT_BANK_TYPES,
    native_output_group_count,
)
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import initialize_distributed


PNBTT_TANGENT_SPECTRUM_SCHEMA = "ember_ecp_pnbtt_tangent_spectrum_v1"
_ARMS = ("correct_fit0", "correct_fit1", "wrong_fit0")


@dataclass(frozen=True)
class ScopeLinearization:
    """Small-score bank operator and exact current signed directions."""

    operator: torch.Tensor  # [scope, value, event * key]
    direction: torch.Tensor  # [scope, rank, event, value]


@dataclass(frozen=True)
class ArmLinearization:
    input_operator: torch.Tensor  # [input, event * key]
    output_operator: torch.Tensor  # [type, group, value, event * key]
    raw_input: torch.Tensor  # [rank, input]
    raw_output_by_type: torch.Tensor  # [type, group, rank, value]


def safe_rms_vjp(
    value: torch.Tensor, gradient: torch.Tensor, *, epsilon: float
) -> torch.Tensor:
    """VJP of ``safe_rms_normalize`` without retaining an autograd graph."""

    if value.shape[-1] != gradient.shape[-1] or epsilon <= 0:
        raise ValueError("safe-RMS VJP topology changed")
    compute = value.float()
    upstream = gradient.float()
    width = compute.shape[-1]
    scale = (compute.square().mean(-1, keepdim=True) + epsilon**2).sqrt()
    radial = (upstream * compute).sum(-1, keepdim=True)
    return upstream / scale - compute * radial / (width * scale.pow(3))


def weighted_cross_covariance(
    values: torch.Tensor,
    keys: torch.Tensor,
    normalized_mass: torch.Tensor,
) -> torch.Tensor:
    """Compute event-wise ``Cov_mu(v, k)`` with bounded temporary storage."""

    if (
        values.ndim != 3
        or keys.ndim != 3
        or normalized_mass.ndim != 3
        or values.shape[:2] != keys.shape[:2]
        or normalized_mass.shape[0] != values.shape[0]
        or normalized_mass.shape[2] != values.shape[1]
    ):
        raise ValueError("cross-covariance topology changed")
    compute_values = values.float()
    compute_keys = keys.float()
    mass = normalized_mass.float()
    mean_value = torch.einsum("sen,snd->sed", mass, compute_values)
    mean_key = torch.einsum("sen,snm->sem", mass, compute_keys)
    blocks = []
    for event in range(mass.shape[1]):
        weighted_keys = mass[:, event, :, None] * compute_keys
        raw = torch.bmm(compute_values.transpose(1, 2), weighted_keys)
        blocks.append(
            raw
            - mean_value[:, event, :, None] * mean_key[:, event, None, :]
        )
    return torch.stack(blocks, dim=1)


def whitened_cross_covariance(
    cross_covariance: torch.Tensor, cholesky: torch.Tensor
) -> torch.Tensor:
    """Return ``T L^-T`` for the runtime's ``C = L L^T`` convention."""

    if (
        cross_covariance.ndim != 4
        or cholesky.ndim != 4
        or cross_covariance.shape[:2] != cholesky.shape[:2]
        or cross_covariance.shape[-1] != cholesky.shape[-1]
        or cholesky.shape[-1] != cholesky.shape[-2]
    ):
        raise ValueError("whitened cross-covariance topology changed")
    solved = torch.linalg.solve_triangular(
        cholesky,
        cross_covariance.transpose(-1, -2),
        upper=False,
    )
    return solved.transpose(-1, -2)


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    tensor = torch.tensor(tuple(values), dtype=torch.float64)
    if tensor.numel() == 0 or not bool(torch.isfinite(tensor).all()):
        raise ValueError("spectrum summary received no finite values")
    return {
        "count": int(tensor.numel()),
        "minimum": float(tensor.amin()),
        "p10": float(torch.quantile(tensor, 0.10)),
        "median": float(torch.quantile(tensor, 0.50)),
        "mean": float(tensor.mean()),
        "p90": float(torch.quantile(tensor, 0.90)),
        "maximum": float(tensor.amax()),
    }


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.float().norm() * right.float().norm()
    if float(denominator) <= 0:
        return 0.0
    return float(torch.dot(left.float().flatten(), right.float().flatten()) / denominator)


def _projection_coefficients(
    matrix: torch.Tensor, gradients: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return singular values, valid-mode coefficients and residual norms.

    This uses the smaller Gram matrix, avoiding a retained full U/V factor for
    the largest q-family matrices.
    """

    rows, columns = matrix.shape
    if gradients.ndim != 2 or gradients.shape[1] != rows:
        raise ValueError("functional-gradient projection topology changed")
    compute = matrix.float()
    batch = gradients.float()
    if rows >= columns:
        gram = compute.T @ compute
        eigenvalues, vectors = torch.linalg.eigh(0.5 * (gram + gram.T))
        eigenvalues = eigenvalues.flip(0).clamp_min(0)
        vectors = vectors.flip(1)
        maximum = eigenvalues[0].clamp_min(torch.finfo(eigenvalues.dtype).tiny)
        valid = eigenvalues > maximum * 1e-6
        singular = eigenvalues.sqrt()
        right_hand = batch @ compute
        coefficients = (right_hand @ vectors[:, valid]) / singular[valid][None]
    else:
        gram = compute @ compute.T
        eigenvalues, vectors = torch.linalg.eigh(0.5 * (gram + gram.T))
        eigenvalues = eigenvalues.flip(0).clamp_min(0)
        vectors = vectors.flip(1)
        maximum = eigenvalues[0].clamp_min(torch.finfo(eigenvalues.dtype).tiny)
        valid = eigenvalues > maximum * 1e-6
        singular = eigenvalues.sqrt()
        coefficients = batch @ vectors[:, valid]
    total = batch.square().sum(-1)
    projected = coefficients.square().sum(-1)
    residual = (total - projected).clamp_min(0)
    return singular, coefficients, residual


def projection_spectrum(
    matrix: torch.Tensor, gradient_sets: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    """Summarize operator spectrum and functional-gradient retention."""

    if matrix.ndim != 2 or not gradient_sets:
        raise ValueError("projection spectrum requires one matrix and gradients")
    packed = []
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, value in gradient_sets.items():
        if value.ndim != 2 or value.shape[1] != matrix.shape[0]:
            raise ValueError("projection gradient set changed")
        packed.append(value)
        spans[name] = (cursor, cursor + value.shape[0])
        cursor += value.shape[0]
    singular, coefficients, residual = _projection_coefficients(
        matrix, torch.cat(packed, dim=0)
    )
    square = singular.square()
    total_spectral = square.sum().clamp_min(torch.finfo(square.dtype).tiny)
    effective = {
        threshold: int((singular > singular[0] * float(threshold)).sum())
        for threshold in (1e-2, 1e-3, 1e-4, 1e-6)
    }
    cumulative = square.cumsum(0) / total_spectral

    def energy_rank(level: float) -> int:
        return int(torch.searchsorted(cumulative, level).item()) + 1

    reports: dict[str, Any] = {}
    combined = torch.cat(packed, dim=0).float()
    total_gradient = combined.square().sum(-1)
    projected_gradient = coefficients.square().sum(-1)
    retention = projected_gradient / total_gradient.clamp_min(1e-30)
    for name, (start, stop) in spans.items():
        selected = retention[start:stop]
        selected_residual = residual[start:stop]
        reports[name] = {
            "retention": _summary(selected.detach().cpu().tolist()),
            "mean_residual_norm": float(selected_residual.mean().sqrt()),
        }
    tail = max(1, math.ceil(square.numel() * 0.1))
    return {
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "effective_rank_by_relative_singular_value": {
            str(key): value for key, value in effective.items()
        },
        "effective_rank_fraction_1e-3": effective[1e-3]
        / min(matrix.shape),
        "spectral_energy_rank": {
            "90pct": energy_rank(0.90),
            "95pct": energy_rank(0.95),
            "99pct": energy_rank(0.99),
        },
        "tail_10pct_spectral_energy_fraction": float(square[-tail:].sum() / total_spectral),
        "largest_singular_values": singular[:16].detach().cpu().tolist(),
        "smallest_singular_values": singular[-16:].detach().cpu().tolist(),
        "functional_gradient": reports,
    }


def _scope_linearization(
    runtime: PNBTTTaskLocalRuntime,
    *,
    scope: Any,
    target: int,
    queries: torch.Tensor,
    event_weights: torch.Tensor,
) -> ScopeLinearization:
    transport = runtime.compiler.tangent_transport
    mean = torch.einsum("sn,snd->sd", scope.base_mass, scope.values.float())
    centered = scope.values.float() - mean[:, None]
    rms = (
        torch.einsum("sn,snd->s", scope.base_mass, centered.square())
        .div(centered.shape[-1])
        .clamp_min(0)
        .sqrt()
    )
    normalized = centered / rms.clamp_min(transport.native_rms_epsilon)[:, None, None]
    key_blocks = []
    start = 0
    while start < len(scope.side_indices):
        side = scope.side_indices[start]
        stop = start + 1
        while stop < len(scope.side_indices) and scope.side_indices[stop] == side:
            stop += 1
        key_blocks.append(
            transport.key_encoder(
                target=target,
                side=side,
                normalized_values=normalized[start:stop],
                metadata=scope.metadata[start:stop],
            )
        )
        start = stop
    keys = torch.cat(key_blocks)
    moments = differentiable_key_moments(
        keys, scope.event_mass, ridge=transport.covariance_ridge
    )
    result = signed_key_value_pool(
        keys=keys,
        values=scope.values,
        moments=moments,
        whitened_queries=whiten_queries(queries, moments),
        temperature=torch.stack(
            tuple(transport.temperature_by_side[side] for side in scope.side_indices)
        ),
        score_epsilon=transport.score_epsilon,
        chunk_size=transport.replay_chunk_size,
    )
    cross = weighted_cross_covariance(scope.values, keys, moments.normalized_mass)
    whitened_cross = whitened_cross_covariance(cross, moments.cholesky)
    rho = event_weights.float().clamp_min(0)
    rho = rho / rho.sum().clamp_min(1e-30)
    operator = torch.cat(
        tuple(whitened_cross[:, event] * rho[event] for event in range(rho.numel())),
        dim=-1,
    )
    return ScopeLinearization(
        operator=operator.detach(), direction=result.direction.detach()
    )


def _arm_linearization(
    runtime: PNBTTTaskLocalRuntime,
    *,
    task: int,
    target: int,
    arm: PreparedPNBTTArm,
) -> ArmLinearization:
    transport = runtime.compiler.tangent_transport
    videos = tuple(
        TangentTransportVideo(native=video, context=context)
        for video, context in zip(arm.videos, arm.bank_contexts, strict=True)
    )
    query = runtime.free_query.target(task, target)
    rho = pnbtt_event_weights(arm.program)
    input_scope = transport._input_scope(videos, target=target)
    input_linear = _scope_linearization(
        runtime,
        scope=input_scope,
        target=target,
        queries=query[:, :, 0][None],
        event_weights=rho,
    )
    groups = native_output_group_count(runtime.owners[target])
    output_scope = transport._output_scope(videos, target=target)
    output_queries = torch.cat(
        tuple(
            query[:, :, side][None].expand(groups, -1, -1, -1)
            for side in range(1, len(OUTPUT_BANK_TYPES) + 1)
        ),
        dim=0,
    )
    output_linear = _scope_linearization(
        runtime,
        scope=output_scope,
        target=target,
        queries=output_queries,
        event_weights=rho,
    )
    normalized_rho = rho.float().clamp_min(0)
    normalized_rho = normalized_rho / normalized_rho.sum().clamp_min(1e-30)
    raw_input = torch.einsum(
        "e,sred->srd", normalized_rho, input_linear.direction
    )[0]
    types = len(OUTPUT_BANK_TYPES)
    width = runtime.owners[target].out_features // groups
    output_by_type = torch.einsum(
        "e,sred->srd", normalized_rho, output_linear.direction
    ).reshape(types, groups, G1_RESIDUAL_RANK, width)
    return ArmLinearization(
        input_operator=input_linear.operator[0],
        output_operator=output_linear.operator.reshape(
            types, groups, width, -1
        ),
        raw_input=raw_input,
        raw_output_by_type=output_by_type,
    )


def _output_type_vjp(
    runtime: PNBTTTaskLocalRuntime,
    *,
    target: int,
    output_by_type: torch.Tensor,
    leaf_gradient: torch.Tensor,
) -> torch.Tensor:
    """Map batched rank4 B leaf gradients to each pre-normalized Y type."""

    transport = runtime.compiler.tangent_transport
    family = PNBTT_FAMILIES.index(runtime.owners[target].family)
    normalized = safe_rms_normalize(
        output_by_type, epsilon=transport.direction_epsilon
    )
    combined = torch.einsum(
        "t,tgrd->grd", transport.type_balance[family], normalized
    )
    raw = combined.permute(1, 0, 2).reshape(G1_RESIDUAL_RANK, -1)
    scale = runtime.ranks.s_ref[target].to(transport.scale_prior_ratio)
    scale = scale * transport.scale_prior_ratio[target]
    upstream_raw = safe_rms_vjp(
        raw[None], leaf_gradient.float() * scale[None, :, None],
        epsilon=transport.direction_epsilon,
    )
    groups = output_by_type.shape[1]
    width = output_by_type.shape[-1]
    upstream_combined = upstream_raw.reshape(
        leaf_gradient.shape[0], G1_RESIDUAL_RANK, groups, width
    ).permute(0, 2, 1, 3)
    upstream_types = (
        upstream_combined[:, None]
        * transport.type_balance[family][None, :, None, None, None]
    )
    return safe_rms_vjp(
        output_by_type[None], upstream_types,
        epsilon=transport.direction_epsilon,
    )


def _prepare_gradient_arms(
    runtime: PNBTTTaskLocalRuntime, task: int
) -> dict[str, PreparedPNBTTArm]:
    wrong_task = int(runtime.config["task_local"]["wrong_task_by_task"][str(task)])
    return {
        "correct_fit0": _prepare_arm(
            runtime,
            name="correct_fit0",
            program_task=task,
            bank_task=task,
            condition=runtime.task_conditions[task].fit_views[0],
            receives_gradient=True,
        ),
        "correct_fit1": _prepare_arm(
            runtime,
            name="correct_fit1",
            program_task=task,
            bank_task=task,
            condition=runtime.task_conditions[task].fit_views[1],
            receives_gradient=True,
        ),
        "wrong_fit0": _prepare_arm(
            runtime,
            name="wrong_fit0",
            program_task=task,
            bank_task=wrong_task,
            condition=runtime.task_conditions[wrong_task].fit_views[0],
            receives_gradient=True,
        ),
    }


def _load_writer_checkpoint(
    runtime: PNBTTTaskLocalRuntime, checkpoint: Path
) -> dict[str, Any]:
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    weights = checkpoint / "ecp.safetensors"
    if (
        manifest.get("stage") != PNBTT_E1_STAGE
        or int(manifest.get("next_macro", -1)) != 110
        or int(manifest.get("files", {}).get(weights.name, {}).get("bytes", -1))
        != weights.stat().st_size
    ):
        raise ValueError("PNBTT tangent diagnostic checkpoint authority changed")
    state = load_file(str(weights), device=str(runtime.context.device))
    runtime.writer_state.load_state_dict(state, strict=True)
    runtime.writer_state.requires_grad_(False).eval()
    return {
        "path": str(checkpoint),
        "macro": 110,
        "weights": str(weights),
        "weight_bytes": weights.stat().st_size,
    }


def _collect_leaf_gradients(
    runtime: PNBTTTaskLocalRuntime,
    *,
    task: int,
    arms: Mapping[str, PreparedPNBTTArm],
    visits: int,
) -> tuple[dict[str, dict[int, dict[str, torch.Tensor]]], list[dict[str, Any]]]:
    storage: dict[str, dict[int, dict[str, list[torch.Tensor]]]] = {
        arm: {
            target: {"a": [], "b": []}
            for target in range(len(runtime.owners))
        }
        for arm in _ARMS
    }
    losses = []
    for visit in range(visits):
        batch, panel = functional_panel_batch(
            runtime, task_id=task, panel_name="a", visit_index=visit
        )
        row: dict[str, Any] = {
            "panel_visit": visit,
            "policy_rng_seed": int(panel.policy_rng_seed),
            "loss": {},
        }
        for arm_name in _ARMS:
            loss, gradients, _ = _functional_derivative(
                runtime,
                task=task,
                arm=arms[arm_name],
                batch=batch,
                seed=panel.policy_rng_seed,
            )
            row["loss"][arm_name] = float(loss)
            for target, contract_target in enumerate(runtime.ranks.contract.targets):
                storage[arm_name][target]["a"].append(
                    gradients[contract_target.name + LORA_A_SUFFIX][12:]
                    .detach()
                    .float()
                    .cpu()
                )
                storage[arm_name][target]["b"].append(
                    gradients[contract_target.name + LORA_B_SUFFIX][:, 12:]
                    .transpose(0, 1)
                    .detach()
                    .float()
                    .cpu()
                )
        losses.append(row)
        print(
            {
                "task": task,
                "completed_panel_visits": visit + 1,
                "total_panel_visits": visits,
            },
            flush=True,
        )
    stacked: dict[str, dict[int, dict[str, torch.Tensor]]] = {
        arm: {
            target: {
                name: torch.stack(values)
                for name, values in tensors.items()
            }
            for target, tensors in targets.items()
        }
        for arm, targets in storage.items()
    }
    return stacked, losses


def _gradient_sets(
    arm_gradients: Sequence[torch.Tensor], *, groups: int
) -> dict[str, torch.Tensor]:
    """Build raw and arm-balanced shared-query diagnostic objectives."""

    if len(arm_gradients) != len(_ARMS):
        raise ValueError("PNBTT diagnostic arm count changed")
    flattened = [value.flatten(2) for value in arm_gradients]
    zeros = torch.zeros_like(flattened[2])

    def pack(parts: Sequence[torch.Tensor]) -> torch.Tensor:
        # [arm, visit, rank, group*value] -> [visit*rank, arm*group*value]
        joined = torch.cat(tuple(parts), dim=-1)
        return joined.permute(0, 1, 2).reshape(-1, joined.shape[-1])

    correct_preserve_wrong = pack((flattened[0], flattened[1], zeros))
    contrast = pack((flattened[0], flattened[1], -flattened[2]))
    all_raw = pack(flattened)

    balanced = []
    for value in flattened:
        norm = value.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        balanced.append(value / norm)
    balanced_zeros = torch.zeros_like(balanced[2])
    return {
        "correct_preserve_wrong_raw": correct_preserve_wrong,
        "correct_wrong_contrast_raw": contrast,
        "all_arm_raw": all_raw,
        "correct_preserve_wrong_arm_balanced": pack(
            (balanced[0], balanced[1], balanced_zeros)
        ),
        "correct_wrong_contrast_arm_balanced": pack(
            (balanced[0], balanced[1], -balanced[2])
        ),
    }


def _query_gradient_cosines(
    operators: Sequence[torch.Tensor], gradients: Sequence[torch.Tensor]
) -> dict[str, Any]:
    query_gradients = [
        torch.einsum("dc,vrd->vrc", operator.float(), gradient.float().flatten(2))
        for operator, gradient in zip(operators, gradients, strict=True)
    ]
    correct = query_gradients[0] + query_gradients[1]
    pairs = {
        "correct_fit_pair": (query_gradients[0], query_gradients[1]),
        "correct_vs_wrong": (correct, query_gradients[2]),
    }
    result = {}
    for name, (left, right) in pairs.items():
        numerator = (left * right).sum(-1)
        denominator = left.norm(dim=-1) * right.norm(dim=-1)
        cosine = numerator / denominator.clamp_min(1e-30)
        result[name] = _summary(cosine.detach().cpu().flatten().tolist())
    return result


def _analyze_target(
    runtime: PNBTTTaskLocalRuntime,
    *,
    task: int,
    target: int,
    arms: Mapping[str, PreparedPNBTTArm],
    leaf: Mapping[str, Mapping[int, Mapping[str, torch.Tensor]]],
) -> list[dict[str, Any]]:
    device = runtime.context.device
    with torch.no_grad(), runtime.compiler.bank_operator.ieee_matmul(device):
        linear = {
            name: _arm_linearization(
                runtime, task=task, target=target, arm=arms[name]
            )
            for name in _ARMS
        }
    owner = runtime.owners[target]
    transport = runtime.compiler.tangent_transport
    input_gradients = []
    output_gradients = []
    for name in _ARMS:
        a_leaf = leaf[name][target]["a"].to(device)
        b_leaf = leaf[name][target]["b"].to(device)
        input_gradients.append(
            safe_rms_vjp(
                linear[name].raw_input[None],
                a_leaf,
                epsilon=transport.direction_epsilon,
            )
        )
        output_gradients.append(
            _output_type_vjp(
                runtime,
                target=target,
                output_by_type=linear[name].raw_output_by_type,
                leaf_gradient=b_leaf,
            )
        )

    records = []
    input_operators = [linear[name].input_operator for name in _ARMS]
    input_matrix = torch.cat(input_operators, dim=0)
    input_sets = _gradient_sets(input_gradients, groups=1)
    input_record = {
        "task": task,
        "target": target,
        "target_name": owner.target_name,
        "family": owner.family.value,
        "side": "input",
        "groups": 1,
        "value_width": owner.in_features,
        "operator_cosine": {
            "correct_fit_pair": _cosine(input_operators[0], input_operators[1]),
            "correct0_vs_wrong": _cosine(input_operators[0], input_operators[2]),
            "correct1_vs_wrong": _cosine(input_operators[1], input_operators[2]),
        },
        "query_gradient_cosine": _query_gradient_cosines(
            input_operators, input_gradients
        ),
        "spectrum": projection_spectrum(input_matrix, input_sets),
    }
    records.append(input_record)

    groups = native_output_group_count(owner)
    width = owner.out_features // groups
    for bank_type, type_name in enumerate(OUTPUT_BANK_TYPES):
        operators = [
            linear[name].output_operator[bank_type].reshape(groups * width, -1)
            for name in _ARMS
        ]
        gradients = [
            value[:, bank_type].permute(0, 2, 1, 3)
            for value in output_gradients
        ]
        matrix = torch.cat(operators, dim=0)
        records.append(
            {
                "task": task,
                "target": target,
                "target_name": owner.target_name,
                "family": owner.family.value,
                "side": type_name,
                "groups": groups,
                "value_width": width,
                "operator_cosine": {
                    "correct_fit_pair": _cosine(operators[0], operators[1]),
                    "correct0_vs_wrong": _cosine(operators[0], operators[2]),
                    "correct1_vs_wrong": _cosine(operators[1], operators[2]),
                },
                "query_gradient_cosine": _query_gradient_cosines(
                    operators, gradients
                ),
                "spectrum": projection_spectrum(
                    matrix, _gradient_sets(gradients, groups=groups)
                ),
            }
        )
    return records


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in records:
        groups.setdefault((str(row["family"]), str(row["side"])), []).append(row)
    output = {}
    for (family, side), selected in sorted(groups.items()):
        key = f"{family}:{side}"
        output[key] = {
            "targets": len(selected),
            "effective_rank_fraction_1e-3": _summary(
                [
                    float(row["spectrum"]["effective_rank_fraction_1e-3"])
                    for row in selected
                ]
            ),
            "tail_10pct_spectral_energy_fraction": _summary(
                [
                    float(row["spectrum"]["tail_10pct_spectral_energy_fraction"])
                    for row in selected
                ]
            ),
            "correct_preserve_wrong_raw_retention": _summary(
                [
                    float(
                        row["spectrum"]["functional_gradient"]
                        ["correct_preserve_wrong_raw"]["retention"]["mean"]
                    )
                    for row in selected
                ]
            ),
            "correct_wrong_contrast_raw_retention": _summary(
                [
                    float(
                        row["spectrum"]["functional_gradient"]
                        ["correct_wrong_contrast_raw"]["retention"]["mean"]
                    )
                    for row in selected
                ]
            ),
            "correct_vs_wrong_query_gradient_cosine": _summary(
                [
                    float(row["query_gradient_cosine"]["correct_vs_wrong"]["mean"])
                    for row in selected
                ]
            ),
            "correct_vs_wrong_operator_cosine": _summary(
                [
                    0.5
                    * (
                        float(row["operator_cosine"]["correct0_vs_wrong"])
                        + float(row["operator_cosine"]["correct1_vs_wrong"])
                    )
                    for row in selected
                ]
            ),
        }
    return output


def analyze_pnbtt_tangent_spectrum(args: Any) -> None:
    context = initialize_distributed(require_numa=False, defer_process_group=True)
    runtime: PNBTTTaskLocalRuntime | None = None
    started = time.monotonic()
    try:
        runtime = prepare_pnbtt_tasklocal_runtime(args, context)
        checkpoint = _load_writer_checkpoint(runtime, args.writer_checkpoint)
        assigned = local_tasks(runtime)
        if len(assigned) != 1:
            raise ValueError("tangent spectrum requires one task per GPU rank")
        task = assigned[0]
        arms = _prepare_gradient_arms(runtime, task)
        visits = int(args.panel_visits)
        if not 1 <= visits <= int(runtime.config["data"]["panel_visits"]):
            raise ValueError("tangent diagnostic panel visit count changed")
        leaf, losses = _collect_leaf_gradients(
            runtime, task=task, arms=arms, visits=visits
        )
        records = []
        for target in range(len(runtime.owners)):
            records.extend(
                _analyze_target(
                    runtime,
                    task=task,
                    target=target,
                    arms=arms,
                    leaf=leaf,
                )
            )
            print(
                {
                    "task": task,
                    "completed_targets": target + 1,
                    "total_targets": len(runtime.owners),
                },
                flush=True,
            )
        shard = {
            "schema_version": PNBTT_TANGENT_SPECTRUM_SCHEMA,
            "task": task,
            "checkpoint": checkpoint,
            "panel": "a",
            "panel_visits": visits,
            "gradient_arms": list(_ARMS),
            "held_or_panel_b_used": False,
            "losses": losses,
            "aggregate": _aggregate(records),
            "records": records,
            "elapsed_seconds": time.monotonic() - started,
        }
        shard_dir = args.output_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(shard_dir / f"task_{task:03d}.json", shard)
        if dist.is_initialized():
            dist.barrier()
        if context.is_main:
            task_ids = tuple(map(int, runtime.config["task_local"]["task_ids"]))
            shards = [
                read_json(shard_dir / f"task_{task_id:03d}.json")
                for task_id in task_ids
            ]
            all_records = [row for shard_row in shards for row in shard_row["records"]]
            result = {
                "schema_version": PNBTT_TANGENT_SPECTRUM_SCHEMA,
                "checkpoint": checkpoint,
                "tasks": list(task_ids),
                "panel": "a",
                "panel_visits": visits,
                "gradient_arms": list(_ARMS),
                "held_or_panel_b_used": False,
                "formula": "T=Cov_mu(v,k); operator=concat_e rho_e T_e L_e^-T",
                "interpretation_contract": {
                    "continuous_diagnostic_not_a_new_gate": True,
                    "increase_key_width_only_if_effective_spectrum_is_truncated": True,
                    "otherwise_use_family_trunk_and_target_specific_low_rank_projection": True,
                    "rank_split_not_reopened": True,
                },
                "aggregate": _aggregate(all_records),
                "shards": [
                    {
                        "task": int(row["task"]),
                        "path": str(shard_dir / f"task_{int(row['task']):03d}.json"),
                        "elapsed_seconds": float(row["elapsed_seconds"]),
                    }
                    for row in shards
                ],
                "elapsed_seconds": time.monotonic() - started,
            }
            write_json_atomic(args.output_dir / "result.json", result)
            write_json_atomic(
                args.output_dir / "completion.json",
                {
                    "schema_version": PNBTT_TANGENT_SPECTRUM_SCHEMA,
                    "completed": True,
                    "tasks": list(task_ids),
                    "panel_visits": visits,
                },
            )
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
