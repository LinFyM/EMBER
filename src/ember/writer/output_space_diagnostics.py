"""Frozen measurements and interventions for Writer q/v-B output spaces."""

from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from safetensors import safe_open

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX
from ember.writer.runtime import autocast


FAMILIES = ("q_b", "v_b")
HEAD_KEYS = {
    "q_b": "writer.factor_heads.q_b.network.2.weight",
    "v_b": "writer.factor_heads.v_b.network.2.weight",
}


def _effective_bf16(value: torch.Tensor) -> torch.Tensor:
    """Reproduce the stored matrix after BF16 forward quantization, then use FP64."""
    return value.detach().to(device="cpu", dtype=torch.bfloat16).to(torch.float64)


def checkpoint_head_matrix(checkpoint: Path, family: str) -> torch.Tensor:
    if family not in FAMILIES:
        raise ValueError(f"unsupported Writer output family: {family}")
    with safe_open(str(checkpoint / "ecp.safetensors"), framework="pt", device="cpu") as handle:
        value = handle.get_tensor(HEAD_KEYS[family])
    expected = (2048, 216) if family == "q_b" else (256, 216)
    if tuple(value.shape) != expected or value.dtype != torch.float32:
        raise ValueError(f"Writer {family} head changed shape or storage dtype")
    return _effective_bf16(value)


def checkpoint_b_templates(checkpoint: Path) -> dict[str, Any]:
    """Inspect all 38 physical B templates without constructing the policy."""
    with safe_open(str(checkpoint / "ecp.safetensors"), framework="pt", device="cpu") as handle:
        names = sorted(name for name in handle.keys() if name.startswith("writer.template_"))
        if len(names) != 76:
            raise ValueError("Writer template tensor count changed")
        b_names = names[1::2]
        nonzero = sum(int(torch.count_nonzero(handle.get_tensor(name))) for name in b_names)
    return {"template_tensors": len(names), "b_templates": len(b_names), "b_nonzero_values": nonzero}


def column_space(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return the complete numerical column space using one fixed library tolerance."""
    if matrix.ndim != 2 or matrix.shape[0] < matrix.shape[1]:
        raise ValueError("column-space matrix must be tall")
    left, singular, _right = torch.linalg.svd(matrix.to(torch.float64), full_matrices=False)
    tolerance = float(max(matrix.shape) * torch.finfo(torch.float64).eps * singular[0])
    rank = int((singular > tolerance).sum())
    if rank <= 0:
        raise ValueError("Writer output head has empty numerical column space")
    return left[:, :rank].contiguous(), singular, tolerance


def space_summary(asset: str, checkpoint: Path, family: str) -> tuple[dict[str, Any], torch.Tensor]:
    matrix = checkpoint_head_matrix(checkpoint, family)
    basis, singular, tolerance = column_space(matrix)
    record = {
        "asset": asset,
        "family": family,
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "effective_weight_precision": "bf16_then_fp64_analysis",
        "rank_tolerance": tolerance,
        "numerical_rank": int(basis.shape[1]),
        "singular_values": singular.tolist(),
        "frobenius_norm": float(torch.linalg.vector_norm(matrix)),
        "spectral_condition": float(singular[0] / singular[-1]),
    }
    return record, basis


def space_overlap(
    left_asset: str, left: torch.Tensor, right_asset: str, right: torch.Tensor, family: str
) -> dict[str, Any]:
    if left.shape[0] != right.shape[0]:
        raise ValueError("output spaces have different native widths")
    cosines = torch.linalg.svdvals(left.transpose(0, 1) @ right).clamp(0.0, 1.0)
    shared_energy = float(cosines.square().sum())
    return {
        "left_asset": left_asset,
        "right_asset": right_asset,
        "family": family,
        "left_rank": int(left.shape[1]),
        "right_rank": int(right.shape[1]),
        "principal_cosines": cosines.tolist(),
        "minimum_principal_cosine": float(cosines.min()),
        "mean_squared_principal_cosine": float(cosines.square().mean()),
        "shared_projection_energy": shared_energy,
        "chordal_distance": math.sqrt(max(0.0, min(left.shape[1], right.shape[1]) - shared_energy)),
    }


@contextmanager
def capture_factor_codes(writer: torch.nn.Module) -> Iterator[dict[str, list[torch.Tensor]]]:
    """Capture the exact 216-dimensional inputs to q/v-B final linears."""
    captured: dict[str, list[torch.Tensor]] = {family: [] for family in FAMILIES}
    handles = []
    for family in FAMILIES:
        linear = writer.factor_heads[family].network[-1]

        def hook(_module, values, *, owner=family):
            if len(values) != 1 or not isinstance(values[0], torch.Tensor):
                raise ValueError(f"{owner} final head lost its tensor input")
            captured[owner].append(values[0].detach().to(device="cpu", dtype=torch.float32))

        handles.append(linear.register_forward_pre_hook(hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def compile_with_codes(loaded: Any, condition: tuple) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    writer = loaded.runtime.state.writer
    with capture_factor_codes(writer) as captured:
        with autocast(loaded.runtime.device):
            state = loaded.runtime.compile(condition)
    codes = {}
    for family, rows in captured.items():
        if len(rows) != 18 or any(tuple(row.shape) != (1, 16, 216) for row in rows):
            raise ValueError(f"{family} code capture changed call count or shape")
        codes[family] = torch.stack([row[0] for row in rows], dim=0).contiguous()
    return ({name: value.detach().to(device="cpu", dtype=torch.float32) for name, value in state.items()}, codes)


def target_name(layer: int, family: str) -> str:
    projection = {"q_b": "q_proj", "v_b": "v_proj"}.get(family)
    if projection is None or not 0 <= layer < 18:
        raise ValueError("invalid expert target coordinate")
    return (
        f"model.paligemma_with_expert.gemma_expert.model.layers.{layer}."
        f"self_attn.{projection}"
    )


def factors(state: Mapping[str, torch.Tensor], layer: int, family: str) -> tuple[torch.Tensor, torch.Tensor]:
    owner = target_name(layer, family)
    a, b = state[owner + LORA_A_SUFFIX], state[owner + LORA_B_SUFFIX]
    expected_b = 2048 if family == "q_b" else 256
    if tuple(a.shape) != (16, 1024) or tuple(b.shape) != (expected_b, 16):
        raise ValueError("generated expert LoRA factor shape changed")
    return a.to(torch.float64), b.to(torch.float64)


def effective_frobenius(b: torch.Tensor, a: torch.Tensor) -> float:
    gram_b = b.transpose(0, 1) @ b
    gram_a = a @ a.transpose(0, 1)
    squared = (gram_b * gram_a.transpose(0, 1)).sum().clamp_min(0)
    return float(squared.sqrt())


def effective_singular_values(b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    q_b, r_b = torch.linalg.qr(b, mode="reduced")
    q_a, r_a = torch.linalg.qr(a.transpose(0, 1), mode="reduced")
    del q_b, q_a
    return torch.linalg.svdvals(r_b @ r_a.transpose(0, 1))


def _cosine_summary(vectors: torch.Tensor) -> dict[str, float]:
    flat = vectors.reshape(vectors.shape[0], -1).to(torch.float64)
    normalized = torch.nn.functional.normalize(flat, dim=1)
    gram = normalized @ normalized.transpose(0, 1)
    off = gram[~torch.eye(len(gram), dtype=torch.bool)]
    return {
        "off_diagonal_mean": float(off.mean()),
        "off_diagonal_min": float(off.min()),
        "off_diagonal_max": float(off.max()),
    }


def code_geometry(codes: torch.Tensor) -> dict[str, Any]:
    if tuple(codes.shape) != (18, 16, 216):
        raise ValueError("captured code tensor has unexpected shape")
    value = codes.to(torch.float64)
    energy = value.square().sum()
    rank_centered = value - value.mean(dim=1, keepdim=True)
    layer_centered = value - value.mean(dim=0, keepdim=True)
    return {
        "total_energy": float(energy),
        "rank_centered_energy_ratio": float(rank_centered.square().sum() / energy),
        "layer_centered_energy_ratio": float(layer_centered.square().sum() / energy),
        "rank_gram": _cosine_summary(value.mean(dim=0)),
        "layer_gram": _cosine_summary(value),
    }


def condition_geometry(
    *, asset: str, task: int, demo: int, state: Mapping[str, torch.Tensor],
    codes: Mapping[str, torch.Tensor], heads: Mapping[str, torch.Tensor], scale: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    code_rows, factor_rows = [], []
    for family in FAMILIES:
        code_rows.append({"asset": asset, "global_task_id": task, "teacher_demo": demo,
                          "family": family, **code_geometry(codes[family])})
        for layer in range(18):
            a, b = factors(state, layer, family)
            singular_b = torch.linalg.svdvals(b)
            singular_delta = effective_singular_values(b, a) * scale
            reconstructed = heads[family] @ codes[family][layer].to(torch.float64).transpose(0, 1)
            denominator = float(torch.linalg.vector_norm(b))
            residual = torch.linalg.vector_norm(reconstructed - b)
            factor_rows.append({
                "asset": asset, "global_task_id": task, "teacher_demo": demo,
                "family": family, "layer": layer,
                "b_frobenius_norm": denominator,
                "b_singular_values": singular_b.tolist(),
                "delta_frobenius_norm": effective_frobenius(b, a) * scale,
                "delta_singular_values": singular_delta.tolist(),
                "reconstruction_max_abs": float((reconstructed - b).abs().max()),
                "reconstruction_relative_l2": float(residual / max(denominator, 1e-30)),
            })
    return code_rows, factor_rows


def cross_video_geometry(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left64, right64 = left.to(torch.float64), right.to(torch.float64)
    flat_left, flat_right = left64.flatten(), right64.flatten()
    cosine = torch.nn.functional.cosine_similarity(flat_left, flat_right, dim=0)
    difference = torch.linalg.vector_norm(flat_left - flat_right)
    return {
        "flattened_cosine": float(cosine),
        "relative_l2": float(difference / torch.linalg.vector_norm(flat_left).clamp_min(1e-30)),
        "centered_energy_ratio": float(
            torch.stack((left64, right64)).sub((left64 + right64) / 2).square().sum()
            / torch.stack((left64, right64)).square().sum().clamp_min(1e-30)
        ),
    }


def transform_state(
    state: Mapping[str, torch.Tensor], *, arm: str,
    old_bases: Mapping[str, torch.Tensor], new_bases: Mapping[str, torch.Tensor], scale: float,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    if arm not in {"SELF", "NEWSPACE", "SHRINK"}:
        raise ValueError(f"unsupported output-space arm: {arm}")
    transformed = {name: value.detach().to(device="cpu", dtype=torch.float32).clone()
                   for name, value in state.items()}
    rows = []
    for family in FAMILIES:
        basis = old_bases[family] if arm == "SELF" else new_bases[family]
        for layer in range(18):
            a, b = factors(state, layer, family)
            projected = basis @ (basis.transpose(0, 1) @ b)
            original_norm = effective_frobenius(b, a) * scale
            projected_norm = effective_frobenius(projected, a) * scale
            alpha = projected_norm / max(original_norm, 1e-30)
            candidate = b * alpha if arm == "SHRINK" else projected
            candidate_norm = effective_frobenius(candidate, a) * scale
            error_norm = effective_frobenius(candidate - b, a) * scale
            name = target_name(layer, family) + LORA_B_SUFFIX
            transformed[name].copy_(candidate.to(dtype=transformed[name].dtype))
            rows.append({
                "arm": arm, "family": family, "layer": layer,
                "original_delta_norm": original_norm,
                "projected_delta_norm": projected_norm,
                "candidate_delta_norm": candidate_norm,
                "candidate_to_original_norm_ratio": candidate_norm / max(original_norm, 1e-30),
                "rho": error_norm / max(original_norm, 1e-30),
                "alpha": alpha,
                "b_projection_relative_l2": float(
                    torch.linalg.vector_norm(projected - b) / torch.linalg.vector_norm(b).clamp_min(1e-30)
                ),
                "finite": bool(torch.isfinite(candidate).all()),
            })
    if not all(row["finite"] for row in rows):
        raise FloatingPointError("output-space intervention produced nonfinite factors")
    return transformed, rows
