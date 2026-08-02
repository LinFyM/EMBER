"""Gauge-robust metrics for the canonical AS-Writer internal analysis."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ember.writer.model import CompleteLoRAWriter, WriterModelError


CONDITIONS = (
    "correct",
    "same_task_other",
    "cross_suite_wrong",
    "shuffled",
    "reversed",
)


def relative_metrics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    """Compare paired equal-shape tensors without hiding their absolute scale."""

    left = reference.detach().float().reshape(-1)
    right = candidate.detach().float().reshape(-1)
    if left.shape != right.shape or not left.numel():
        raise WriterModelError("paired internal-analysis tensors changed shape")
    left_energy = float(left.square().sum())
    right_energy = float(right.square().sum())
    difference = float((right - left).square().sum())
    dot = float((left * right).sum())
    count = left.numel()
    return {
        "relative_l2": math.sqrt(difference / max(left_energy, 1e-24)),
        "cosine": dot / max(math.sqrt(left_energy * right_energy), 1e-24),
        "reference_rms": math.sqrt(left_energy / count),
        "candidate_rms": math.sqrt(right_energy / count),
    }


def mapping_metrics(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    *,
    select: str = "all",
) -> dict[str, float]:
    """Compare a complete public/factor mapping in its current gauge."""

    names = sorted(reference)
    if names != sorted(candidate) or select not in {"all", "a", "b"}:
        raise WriterModelError("paired LoRA mapping changed")
    if select != "all":
        marker = f".lora_{select.upper()}.default.weight"
        names = [name for name in names if name.endswith(marker)]
    left_energy = right_energy = difference = dot = count = 0.0
    for name in names:
        left = reference[name].detach().float()
        right = candidate[name].detach().float()
        if left.shape != right.shape:
            raise WriterModelError("paired LoRA tensor shape changed")
        left_energy += float(left.square().sum())
        right_energy += float(right.square().sum())
        difference += float((right - left).square().sum())
        dot += float((left * right).sum())
        count += left.numel()
    if not names:
        raise WriterModelError("empty selected LoRA mapping")
    return {
        "relative_l2": math.sqrt(difference / max(left_energy, 1e-24)),
        "cosine": dot / max(math.sqrt(left_energy * right_energy), 1e-24),
        "reference_rms": math.sqrt(left_energy / count),
        "candidate_rms": math.sqrt(right_energy / count),
    }


def fixed_sequence(
    value: torch.Tensor, valid: torch.Tensor, *, bins: int = 16
) -> torch.Tensor:
    """Create a deterministic fixed-length signature for ragged video stages."""

    selected = value[valid].detach().float()
    if selected.ndim < 2 or selected.shape[0] == 0 or bins <= 0:
        raise WriterModelError("empty internal-analysis sequence signature")
    flat = selected.reshape(selected.shape[0], -1)
    if flat.shape[0] != bins:
        flat = F.interpolate(
            flat.T[None], size=bins, mode="linear", align_corners=True
        )[0].T
    return flat


def variance_metrics(values: Sequence[torch.Tensor]) -> dict[str, Any]:
    tensor = torch.stack([value.detach().float().reshape(-1) for value in values])
    sample = float(tensor.square().sum(dim=1).mean())
    mean = float(tensor.mean(dim=0).square().sum())
    if len(values) < 2:
        return {
            "videos": len(values),
            "estimable": False,
            "sample_energy": sample,
            "task_mean_energy": mean,
            "centered_variance": None,
            "centered_variance_over_sample_energy": None,
            "centered_variance_over_task_mean_energy": None,
        }
    centered = max(sample - mean, 0.0)
    return {
        "videos": len(values),
        "estimable": True,
        "sample_energy": sample,
        "task_mean_energy": mean,
        "centered_variance": centered,
        "centered_variance_over_sample_energy": centered / max(sample, 1e-24),
        "centered_variance_over_task_mean_energy": centered / max(mean, 1e-24),
    }


def _lora_pairs(writer: CompleteLoRAWriter) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = defaultdict(dict)
    for spec in writer.tensor_specs:
        result[spec.module]["a" if spec.factor_index == 0 else "b"] = spec.name
    if len(result) != 38 or any(set(pair) != {"a", "b"} for pair in result.values()):
        raise WriterModelError("public LoRA A/B pairing changed")
    return dict(result)


def _module_kind(module: str) -> str:
    if module.endswith("q_proj"):
        return "q"
    if module.endswith("v_proj"):
        return "v"
    if module.endswith("action_in_proj") or module.endswith("action_out_proj"):
        return "action"
    raise WriterModelError("unknown public LoRA target")


def effective_inner(
    writer: CompleteLoRAWriter,
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    *,
    kind: str | None = None,
) -> float:
    """Compute the Frobenius inner product of complete effective ``BA`` maps."""

    total = 0.0
    for module, names in _lora_pairs(writer).items():
        if kind is not None and _module_kind(module) != kind:
            continue
        la, lb = left[names["a"]].double(), left[names["b"]].double()
        ra, rb = right[names["a"]].double(), right[names["b"]].double()
        total += float(((lb.T @ rb) * (la @ ra.T)).sum())
    return total


def effective_metrics(
    writer: CompleteLoRAWriter,
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    left = effective_inner(writer, reference, reference)
    right = effective_inner(writer, candidate, candidate)
    dot = effective_inner(writer, reference, candidate)
    return {
        "relative_l2": math.sqrt(
            max(left + right - 2.0 * dot, 0.0) / max(left, 1e-24)
        ),
        "cosine": dot / max(math.sqrt(left * right), 1e-24),
        "reference_l2": math.sqrt(max(left, 0.0)),
        "candidate_l2": math.sqrt(max(right, 0.0)),
    }


def effective_ba_error(
    writer: CompleteLoRAWriter,
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    difference = reference_energy = maximum = 0.0
    count = 0
    for names in _lora_pairs(writer).values():
        left = reference[names["b"]].float() @ reference[names["a"]].float()
        right = candidate[names["b"]].float() @ candidate[names["a"]].float()
        delta = right - left
        difference += float(delta.square().sum())
        reference_energy += float(left.square().sum())
        maximum = max(maximum, float(delta.abs().max()))
        count += delta.numel()
    return {
        "relative_l2": math.sqrt(difference / max(reference_energy, 1e-24)),
        "difference_rms": math.sqrt(difference / max(count, 1)),
        "max_absolute_error": maximum,
    }


def effective_variance(
    writer: CompleteLoRAWriter,
    states: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    """Split same-task video variation into scale-like and orthogonal parts."""

    size = len(states)
    gram = np.empty((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(left, size):
            gram[left, right] = gram[right, left] = effective_inner(
                writer, states[left], states[right]
            )
    sample = float(np.diag(gram).mean())
    mean = float(gram.mean())
    if size < 2:
        return {
            "videos": size,
            "estimable": False,
            "sample_energy": sample,
            "task_mean_energy": mean,
            "centered_variance": None,
            "centered_variance_over_sample_energy": None,
            "centered_variance_over_task_mean_energy": None,
            "scale_like_video_variance_fraction": None,
            "orthogonal_direction_video_variance_fraction": None,
        }
    centered = max(sample - mean, 0.0)
    row_mean = gram.mean(axis=1)
    delta_energy = np.diag(gram) - 2.0 * row_mean + mean
    scale_energy = np.square(row_mean - mean) / max(mean, 1e-24)
    mean_scale = float(np.maximum(scale_energy, 0.0).mean())
    orthogonal = float(np.maximum(delta_energy - scale_energy, 0.0).mean())
    return {
        "videos": size,
        "estimable": True,
        "sample_energy": sample,
        "task_mean_energy": mean,
        "centered_variance": centered,
        "centered_variance_over_sample_energy": centered / max(sample, 1e-24),
        "centered_variance_over_task_mean_energy": centered / max(mean, 1e-24),
        "scale_like_video_variance_fraction": mean_scale / max(centered, 1e-24),
        "orthogonal_direction_video_variance_fraction": orthogonal
        / max(centered, 1e-24),
    }


def _spectrum_row(energy: torch.Tensor) -> dict[str, float | int]:
    total = float(energy.sum())
    if total <= 0.0 or not bool(torch.isfinite(energy).all()):
        raise WriterModelError("effective-BA singular spectrum is degenerate")
    probability = energy / total
    cumulative = probability.cumsum(0)
    return {
        "effective_ba_energy": total,
        "effective_ba_norm": math.sqrt(total),
        "stable_rank": total / max(float(energy[0]), 1e-24),
        "entropy_effective_rank": float(
            torch.exp(-(probability * probability.clamp_min(1e-30).log()).sum())
        ),
        "top_singular_energy": float(probability[0]),
        "rank90": int(
            torch.searchsorted(
                cumulative, torch.tensor(0.9, dtype=cumulative.dtype)
            )
        )
        + 1,
        "rank99": int(
            torch.searchsorted(
                cumulative, torch.tensor(0.99, dtype=cumulative.dtype)
            )
        )
        + 1,
    }


def _factor_spectra(
    factors: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[tuple[int, int, int], list[str]] = defaultdict(list)
    for name, (a, b) in factors.items():
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise WriterModelError("invalid effective-BA factor shapes")
        grouped[(a.shape[1], b.shape[0], a.shape[0])].append(name)
    result: dict[str, dict[str, float | int]] = {}
    for names in grouped.values():
        a = torch.stack([factors[name][0].double().cpu() for name in names])
        b = torch.stack([factors[name][1].double().cpu() for name in names])
        right_b = torch.linalg.qr(b, mode="r").R
        right_a = torch.linalg.qr(a.transpose(-2, -1), mode="r").R
        energy = torch.linalg.svdvals(right_b @ right_a.transpose(-2, -1)).square()
        result.update(
            {name: _spectrum_row(energy[index]) for index, name in enumerate(names)}
        )
    if not result:
        raise WriterModelError("cannot analyze empty effective-BA factors")
    return result


def _aggregate_spectra(
    spectra: Sequence[Mapping[str, float | int]],
) -> dict[str, float | int]:
    if not spectra:
        raise WriterModelError("cannot aggregate empty effective-BA spectra")
    energy = float(sum(float(value["effective_ba_energy"]) for value in spectra))
    return {
        "targets": len(spectra),
        "effective_ba_energy": energy,
        "effective_lora_norm": math.sqrt(energy),
        "stable_rank_mean": float(
            np.mean([float(value["stable_rank"]) for value in spectra])
        ),
        "entropy_effective_rank_mean": float(
            np.mean([float(value["entropy_effective_rank"]) for value in spectra])
        ),
        "top_singular_energy_mean": float(
            np.mean([float(value["top_singular_energy"]) for value in spectra])
        ),
        "rank90_mean": float(np.mean([float(value["rank90"]) for value in spectra])),
        "rank99_mean": float(np.mean([float(value["rank99"]) for value in spectra])),
    }


def _component_geometry(
    gram: torch.Tensor,
    b_gram: torch.Tensor,
    layer_energy: Sequence[float],
) -> dict[str, Any]:
    rank = gram.shape[0]
    layers = np.asarray(layer_energy, dtype=np.float64)
    diagonal = gram.diag().clamp_min(0.0)
    if (
        gram.shape != (rank, rank)
        or b_gram.shape != gram.shape
        or not layers.size
        or not np.isfinite(layers).all()
        or float(diagonal.sum()) <= 0.0
    ):
        raise WriterModelError("rank-coordinate component geometry is degenerate")
    component = gram / torch.sqrt(
        diagonal[:, None] * diagonal[None]
    ).clamp_min(1e-24)
    b_diagonal = b_gram.diag().clamp_min(0.0)
    b_cosine = b_gram / torch.sqrt(
        b_diagonal[:, None] * b_diagonal[None]
    ).clamp_min(1e-24)
    upper = torch.triu(torch.ones(rank, rank, dtype=torch.bool), diagonal=1)
    participation = diagonal / diagonal.sum()
    return {
        "coordinate_energy_participation": participation.tolist(),
        "active_coordinates_1e6": int((participation > 1e-6).sum()),
        "component_pair_cosine_mean": float(component[upper].mean()),
        "component_negative_pair_fraction": float(
            (component[upper] < 0.0).float().mean()
        ),
        "b_column_cosine_mean": float(b_cosine[upper].mean()),
        "b_column_negative_fraction": float((b_cosine[upper] < 0.0).float().mean()),
        "layer_energy_cv": float(layers.std() / max(layers.mean(), 1e-24)),
    }


def _cross_layer_cosine(
    writer: CompleteLoRAWriter, state: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    pairs = _lora_pairs(writer)
    result = {}
    for kind in ("q", "v"):
        modules = [name for name in sorted(pairs) if _module_kind(name) == kind]
        values = []
        for left_index, left in enumerate(modules):
            for right in modules[left_index + 1 :]:
                la = state[pairs[left]["a"]].double()
                lb = state[pairs[left]["b"]].double()
                ra = state[pairs[right]["a"]].double()
                rb = state[pairs[right]["b"]].double()
                dot = float(((lb.T @ rb) * (la @ ra.T)).sum())
                left_energy = float(((lb.T @ lb) * (la @ la.T)).sum())
                right_energy = float(((rb.T @ rb) * (ra @ ra.T)).sum())
                values.append(
                    dot / max(math.sqrt(left_energy * right_energy), 1e-24)
                )
        result[kind] = float(np.mean(values))
    return result


def lora_geometry(
    writer: CompleteLoRAWriter, state: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    """Report true BA spectra plus explicitly gauge-dependent component geometry."""

    rank = writer.PUBLIC_LORA_RANK
    gram = torch.zeros(rank, rank, dtype=torch.float64)
    b_gram = torch.zeros_like(gram)
    energy_by_kind = {name: 0.0 for name in ("q", "v", "action")}
    layers_by_kind: dict[str, list[float]] = {name: [] for name in energy_by_kind}
    factors = {
        module: (
            state[names["a"]].double().cpu(),
            state[names["b"]].double().cpu(),
        )
        for module, names in _lora_pairs(writer).items()
    }
    spectra = _factor_spectra(factors)
    for module, (a, b) in factors.items():
        component = (b.T @ b) * (a @ a.T)
        gram += component
        b_gram += b.T @ b
        energy = float(spectra[module]["effective_ba_energy"])
        kind = _module_kind(module)
        energy_by_kind[kind] += energy
        layers_by_kind[kind].append(energy)
    total = sum(energy_by_kind.values())
    return {
        **_aggregate_spectra(list(spectra.values())),
        "q_v_action_energy_ratio": {
            key: value / max(total, 1e-24) for key, value in energy_by_kind.items()
        },
        "per_target_effective_ba_spectrum": spectra,
        "rank_coordinate_component_gram": _component_geometry(
            gram,
            b_gram,
            [value for group in layers_by_kind.values() for value in group],
        ),
        "per_layer_energy_cv": float(
            np.std([value for group in layers_by_kind.values() for value in group])
            / max(
                np.mean(
                    [value for group in layers_by_kind.values() for value in group]
                ),
                1e-24,
            )
        ),
        "cross_layer_effective_ba_cosine": _cross_layer_cosine(writer, state),
        "public_a_rms": mapping_metrics(state, state, select="a")["reference_rms"],
        "public_b_rms": mapping_metrics(state, state, select="b")["reference_rms"],
    }


def rank_gauge_permute(
    writer: CompleteLoRAWriter,
    state: Mapping[str, torch.Tensor],
    permutation: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Permute every A row and matching B column while preserving complete BA."""

    rank = writer.PUBLIC_LORA_RANK
    if (
        permutation.shape != (rank,)
        or permutation.dtype != torch.long
        or sorted(permutation.cpu().tolist()) != list(range(rank))
    ):
        raise WriterModelError("invalid public LoRA rank gauge permutation")
    result = {name: value.detach().clone() for name, value in state.items()}
    changes = {}
    for module, names in _lora_pairs(writer).items():
        a = state[names["a"]]
        b = state[names["b"]]
        index = permutation.to(a.device)
        result[names["a"]] = a.index_select(0, index)
        result[names["b"]] = b.index_select(1, index.to(b.device))
        changes[module] = {
            "public_a": relative_metrics(a, result[names["a"]]),
            "public_b": relative_metrics(b, result[names["b"]]),
        }
    return result, changes


def attention_summary(
    weights: torch.Tensor,
    valid_intervals: torch.Tensor,
    valid_semantics: torch.Tensor,
) -> dict[str, float]:
    """Summarize target/rank Program attention over outgoing A/E/D columns."""

    if weights.ndim != 4 or weights.shape[0] != valid_intervals.shape[0]:
        raise WriterModelError("Program reader attention topology changed")
    columns = valid_semantics.shape[1]
    task_tokens = (columns - 1) // 2
    if columns != 1 + 2 * task_tokens:
        raise WriterModelError("Program A/E/D column topology changed")
    counts = (
        valid_intervals[:, :, None] & valid_semantics[:, None]
    ).sum(dim=(1, 2)).clamp_min(2)
    probabilities = weights.detach().float()
    entropy = -(
        probabilities * probabilities.clamp_min(1e-30).log()
    ).sum(dim=-1) / counts.log()[:, None, None]
    grid = probabilities.reshape(
        weights.shape[0],
        weights.shape[1],
        weights.shape[2],
        valid_intervals.shape[1],
        columns,
    )
    mass = grid.sum(dim=3).mean(dim=(0, 1, 2))
    return {
        "normalized_entropy_mean": float(entropy.mean()),
        "top_mass_mean": float(probabilities.max(dim=-1).values.mean()),
        "action_mass": float(mass[0]),
        "effect_mass": float(mass[1 : 1 + task_tokens].sum()),
        "change_mass": float(mass[1 + task_tokens :].sum()),
    }


def probability_summary(weights: torch.Tensor, valid_queries: torch.Tensor) -> dict[str, float]:
    """Summarize an explicit attention softmax without retaining its large grid."""

    if weights.ndim != 4 or valid_queries.shape != (weights.shape[0], weights.shape[2]):
        raise WriterModelError("internal-analysis attention query topology changed")
    selected = weights.detach().float().permute(0, 2, 1, 3)[valid_queries]
    if not selected.numel() or not bool(torch.isfinite(selected).all()):
        raise WriterModelError("internal-analysis attention is empty or non-finite")
    entropy = -(selected * selected.clamp_min(1e-30).log()).sum(dim=-1)
    support = (selected > 0).sum(dim=-1).clamp_min(2).log()
    return {
        "normalized_entropy_mean": float((entropy / support).mean()),
        "top_mass_mean": float(selected.max(dim=-1).values.mean()),
        "probability_sum_error_max": float((selected.sum(dim=-1) - 1.0).abs().max()),
    }


def routing_centered_energy(
    weights: torch.Tensor, target_count: int, rank: int
) -> dict[str, float]:
    """Measure target/rank differentiation in ProgramReader probabilities."""

    if weights.ndim != 4 or weights.shape[2] != target_count * rank:
        raise WriterModelError("internal-analysis ProgramReader routing changed")
    grid = weights.detach().float().reshape(
        weights.shape[0], weights.shape[1], target_count, rank, weights.shape[-1]
    )
    total = float(grid.square().mean())
    target = float((grid - grid.mean(dim=2, keepdim=True)).square().mean())
    coordinate = float((grid - grid.mean(dim=3, keepdim=True)).square().mean())
    joint = grid - grid.mean(dim=2, keepdim=True) - grid.mean(dim=3, keepdim=True) + grid.mean(dim=(2, 3), keepdim=True)
    interaction = float(joint.square().mean())
    return {
        "total_energy": total,
        "target_centered_energy": target,
        "rank_centered_energy": coordinate,
        "target_rank_interaction_energy": interaction,
        "target_centered_fraction": target / max(total, 1e-30),
        "rank_centered_fraction": coordinate / max(total, 1e-30),
    }


def change_retention(input_change: float, output_change: float) -> float:
    if not math.isfinite(input_change) or not math.isfinite(output_change):
        raise WriterModelError("non-finite internal-analysis change retention")
    return output_change / max(input_change, 1e-12)


def validate_finite_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            validate_finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise WriterModelError(f"non-finite internal-analysis value at {path}")
