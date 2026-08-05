"""Gauge-aware and gauge-invariant metrics for generated PI05 LoRA states."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ember.writer.model import CompleteLoRAWriter
from ember.writer.policy_dictionary import PolicyCoordinateComposer


def state_row(state: Mapping[str, torch.Tensor], row: int) -> dict[str, torch.Tensor]:
    return {
        name: value[row].detach().to(device="cpu", dtype=torch.float32)
        for name, value in state.items()
    }


def lora_pairs(writer: CompleteLoRAWriter) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for spec in writer.tensor_specs:
        result.setdefault(spec.module, {})[
            "a" if spec.factor_index == 0 else "b"
        ] = spec.name
    return result


def _kind(module: str) -> str:
    if module.endswith("q_proj"):
        return "q"
    if module.endswith("v_proj"):
        return "v"
    return "action"


def effective_inner(
    pairs: Mapping[str, Mapping[str, str]],
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> float:
    total = 0.0
    for names in pairs.values():
        la, lb = left[names["a"]].double(), left[names["b"]].double()
        ra, rb = right[names["a"]].double(), right[names["b"]].double()
        total += float(((lb.T @ rb) * (la @ ra.T)).sum())
    return total


def effective_metrics(
    pairs: Mapping[str, Mapping[str, str]],
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    left = effective_inner(pairs, reference, reference)
    right = effective_inner(pairs, candidate, candidate)
    dot = effective_inner(pairs, reference, candidate)
    return {
        "relative_l2": math.sqrt(max(left + right - 2.0 * dot, 0.0) / max(left, 1e-24)),
        "cosine": dot / max(math.sqrt(left * right), 1e-24),
        "reference_l2": math.sqrt(max(left, 0.0)),
        "candidate_l2": math.sqrt(max(right, 0.0)),
    }


def effective_delta_metrics(
    pairs: Mapping[str, Mapping[str, str]],
    reference: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Compare a candidate effective-BA update with a target update."""

    rr = effective_inner(pairs, reference, reference)
    tt = effective_inner(pairs, target, target)
    cc = effective_inner(pairs, candidate, candidate)
    rt = effective_inner(pairs, reference, target)
    rc = effective_inner(pairs, reference, candidate)
    tc = effective_inner(pairs, target, candidate)
    target_sq = max(tt + rr - 2.0 * rt, 0.0)
    candidate_sq = max(cc + rr - 2.0 * rc, 0.0)
    delta_dot = tc - rt - rc + rr
    residual_sq = max(tt + cc - 2.0 * tc, 0.0)
    return {
        "target_delta_l2": math.sqrt(target_sq),
        "candidate_delta_l2": math.sqrt(candidate_sq),
        "candidate_over_target_delta_l2": math.sqrt(
            candidate_sq / max(target_sq, 1e-24)
        ),
        "delta_cosine": delta_dot / max(math.sqrt(target_sq * candidate_sq), 1e-24),
        "residual_over_target_delta_l2": math.sqrt(residual_sq / max(target_sq, 1e-24)),
    }


def _off_diagonal_mean(value: torch.Tensor) -> float:
    mask = ~torch.eye(value.shape[0], dtype=torch.bool)
    return float(value[mask].mean())


def _component_summary(gram: torch.Tensor) -> dict[str, float | int]:
    diagonal = gram.diag().clamp_min(0)
    total = float(diagonal.sum())
    normalized = gram / torch.sqrt(
        diagonal[:, None].clamp_min(1e-30) * diagonal[None].clamp_min(1e-30)
    )
    return {
        "active_coordinates": int((diagonal > max(total, 1e-30) * 1e-10).sum()),
        "top4_coordinate_energy_fraction": float(
            diagonal.topk(min(4, diagonal.numel())).values.sum() / max(total, 1e-30)
        ),
        "mean_absolute_offdiagonal_component_cosine": _off_diagonal_mean(
            normalized.abs()
        ),
    }


def _energy_profile(energy: torch.Tensor, *, label: str) -> dict[str, float | int]:
    energy = energy.detach().double().clamp_min(0).reshape(-1)
    total = float(energy.sum())
    probability = energy / max(total, 1e-30)
    return {
        f"active_{label}": int((energy > max(total, 1e-30) * 1e-10).sum()),
        f"effective_{label}": float(1.0 / probability.square().sum().clamp_min(1e-30)),
        f"top4_{label}_energy_fraction": float(
            probability.topk(min(4, probability.numel())).values.sum()
        ),
        f"max_{label}_energy_fraction": float(probability.max()),
        "energy": total,
    }


def adapter_geometry(
    writer: CompleteLoRAWriter,
    pairs: Mapping[str, Mapping[str, str]],
    state: Mapping[str, torch.Tensor],
    scale: float,
) -> dict[str, Any]:
    spectra = []
    by_kind: dict[str, list[dict[str, float]]] = {"q": [], "v": [], "action": []}
    component = torch.zeros(
        writer.PUBLIC_LORA_RANK, writer.PUBLIC_LORA_RANK, dtype=torch.float64
    )
    component_by_kind = {name: torch.zeros_like(component) for name in by_kind}
    b_cosines: dict[str, list[float]] = {name: [] for name in by_kind}
    a_sq = b_sq = a_count = b_count = 0.0
    for module, names in pairs.items():
        a = state[names["a"]].double()
        b = state[names["b"]].double()
        right_b = torch.linalg.qr(b, mode="r").R
        right_a = torch.linalg.qr(a.T, mode="r").R
        energy = torch.linalg.svdvals(right_b @ right_a.T).square()
        probability = energy / energy.sum().clamp_min(1e-30)
        spectrum = {
            "energy": float(energy.sum()),
            "stable_rank": float(energy.sum() / energy[0].clamp_min(1e-30)),
            "top_singular_energy": float(probability[0]),
            "rank90": float(
                torch.searchsorted(
                    probability.cumsum(0), torch.tensor(0.9, dtype=probability.dtype)
                )
                + 1
            ),
            "rank99": float(
                torch.searchsorted(
                    probability.cumsum(0), torch.tensor(0.99, dtype=probability.dtype)
                )
                + 1
            ),
        }
        kind = _kind(module)
        spectra.append(spectrum)
        by_kind[kind].append(spectrum)
        value = (b.T @ b) * (a @ a.T)
        component += value
        component_by_kind[kind] += value
        columns = b / torch.linalg.vector_norm(b, dim=0, keepdim=True).clamp_min(1e-30)
        b_cosines[kind].append(_off_diagonal_mean((columns.T @ columns).abs()))
        a_sq += float(a.square().sum())
        b_sq += float(b.square().sum())
        a_count += a.numel()
        b_count += b.numel()

    total_energy = sum(row["energy"] for row in spectra)
    target_energy = torch.tensor(
        [row["energy"] for row in spectra], dtype=torch.float64
    )

    def aggregate(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
        return {
            "targets": float(len(rows)),
            "energy_fraction": float(sum(row["energy"] for row in rows) / total_energy),
            "stable_rank_mean": float(np.mean([row["stable_rank"] for row in rows])),
            "top_singular_energy_mean": float(
                np.mean([row["top_singular_energy"] for row in rows])
            ),
            "rank90_mean": float(np.mean([row["rank90"] for row in rows])),
            "rank99_mean": float(np.mean([row["rank99"] for row in rows])),
        }

    return {
        "effective_ba_energy_unscaled": total_energy,
        "effective_lora_norm_unscaled": math.sqrt(total_energy),
        "effective_lora_norm_scaled": math.sqrt(total_energy) * scale,
        "stable_rank_mean": float(np.mean([row["stable_rank"] for row in spectra])),
        "top_singular_energy_mean": float(
            np.mean([row["top_singular_energy"] for row in spectra])
        ),
        "rank90_mean": float(np.mean([row["rank90"] for row in spectra])),
        "rank99_mean": float(np.mean([row["rank99"] for row in spectra])),
        "target_energy_profile": _energy_profile(target_energy, label="targets"),
        "public_a_rms": math.sqrt(a_sq / a_count),
        "public_b_rms": math.sqrt(b_sq / b_count),
        "rank_coordinate_geometry_gauge_dependent": {
            "all": _component_summary(component),
            **{
                name: {
                    **_component_summary(component_by_kind[name]),
                    "mean_absolute_b_column_cosine": float(np.mean(b_cosines[name])),
                }
                for name in by_kind
            },
        },
        "by_kind": {name: aggregate(rows) for name, rows in by_kind.items()},
    }


@contextmanager
def capture_policy_dictionary_mixing(
    writer: CompleteLoRAWriter,
) -> Any:
    """Capture the one conditioned atom-mixing batch without changing Writer output."""

    record: dict[str, torch.Tensor] = {}
    if not isinstance(writer.composer, PolicyCoordinateComposer):
        yield record
        return

    def hook(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        mix_a, mix_b = output
        record["mix_a"] = mix_a.detach().to(device="cpu", dtype=torch.float32)
        record["mix_b"] = mix_b.detach().to(device="cpu", dtype=torch.float32)

    handle = writer.composer.register_forward_hook(hook)
    try:
        yield record
    finally:
        handle.remove()


def _mixing_matrix_metrics(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach().double()
    singular_energy = torch.linalg.svdvals(value).square()
    atom_energy = value.square().sum(dim=0)
    return {
        **_energy_profile(atom_energy, label="atoms"),
        "stable_row_rank": float(
            singular_energy.sum() / singular_energy[0].clamp_min(1e-30)
        ),
        "top_singular_energy_fraction": float(
            singular_energy[0] / singular_energy.sum().clamp_min(1e-30)
        ),
    }


def _dictionary_atom_energy(
    writer: CompleteLoRAWriter,
) -> tuple[torch.Tensor, torch.Tensor]:
    a_energy = torch.stack(
        [value.detach().double().square().sum(dim=1).cpu() for value in writer.policy_atoms.a_atoms]
    ).sum(dim=0)
    b_energy = torch.stack(
        [value.detach().double().square().sum(dim=0).cpu() for value in writer.policy_atoms.b_atoms]
    ).sum(dim=0)
    return a_energy, b_energy


def policy_dictionary_batch_records(
    writer: CompleteLoRAWriter,
    capture: Mapping[str, torch.Tensor],
    names: Sequence[str],
) -> dict[str, Any] | None:
    """Serialize raw mixing and storage-aware participation for one condition batch."""

    if "mix_a" not in capture or "mix_b" not in capture:
        return None
    mix_a, mix_b = capture["mix_a"], capture["mix_b"]
    if mix_a.shape != mix_b.shape or mix_a.shape[0] != len(names):
        raise ValueError("captured policy dictionary mixing batch changed shape")
    a_storage, b_storage = _dictionary_atom_energy(writer)
    dictionary = {
        "a": _energy_profile(a_storage, label="atoms"),
        "b": _energy_profile(b_storage, label="atoms"),
        "combined": _energy_profile(a_storage + b_storage, label="atoms"),
    }
    conditions = {}
    for index, name in enumerate(names):
        current_a, current_b = mix_a[index], mix_b[index]
        a_energy = current_a.double().square().sum(dim=0)
        b_energy = current_b.double().square().sum(dim=0)
        conditions[name] = {
            "mix_a": current_a.tolist(),
            "mix_b": current_b.tolist(),
            "a": _mixing_matrix_metrics(current_a),
            "b": _mixing_matrix_metrics(current_b),
            "combined": _energy_profile(a_energy + b_energy, label="atoms"),
            "storage_norm_weighted": _energy_profile(
                a_energy * a_storage + b_energy * b_storage,
                label="atoms",
            ),
        }
    return {"dictionary_storage": dictionary, "conditions": conditions}


def _mixing_vector(condition: Mapping[str, Any]) -> torch.Tensor:
    return torch.cat(
        (
            torch.tensor(condition["mix_a"], dtype=torch.float64).reshape(-1),
            torch.tensor(condition["mix_b"], dtype=torch.float64).reshape(-1),
        )
    )


def _vector_set_metrics(values: Sequence[torch.Tensor]) -> dict[str, float]:
    stacked = torch.stack([value.double().reshape(-1) for value in values])
    gram = stacked @ stacked.T
    sample = float(gram.diag().mean())
    mean = float(gram.mean())
    diagonal = gram.diag().clamp_min(1e-30)
    cosine = gram / torch.sqrt(diagonal[:, None] * diagonal[None, :])
    mask = ~torch.eye(len(values), dtype=torch.bool)
    return {
        "sample_energy": sample,
        "mean_energy": mean,
        "centered_variance_over_sample_energy": max(sample - mean, 0.0)
        / max(sample, 1e-30),
        "mean_pairwise_cosine": float(cosine[mask].mean()),
        "mean_absolute_pairwise_cosine": float(cosine[mask].abs().mean()),
    }


def _same_task_video_mixing_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    video_names = ("demo_0", "demo_1", "demo_2", "demo_3", "demo_4")
    keys = (
        "centered_variance_over_sample_energy",
        "mean_pairwise_cosine",
        "mean_absolute_pairwise_cosine",
    )
    metrics = [
        _vector_set_metrics(
            [_mixing_vector(record["conditions"][name]) for name in video_names]
        )
        for record in records
    ]
    return {
        key: distribution([float(value[key]) for value in metrics]) for key in keys
    }


def _condition_relative_l2_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = {}
    for name in ("demo_1", "reversed_0", "shuffled_0"):
        selected = [record for record in records if name in record["conditions"]]
        result[name] = distribution(
            [
                tensor_metrics(
                    _mixing_vector(record["conditions"]["demo_0"]),
                    _mixing_vector(record["conditions"][name]),
                )["relative_l2"]
                for record in selected
            ]
        )
    return result


def policy_dictionary_checkpoint_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Summarize atom use across tasks and same-task video conditions."""

    records = [row.get("policy_dictionary") for row in rows]
    records = [record for record in records if record is not None]
    if not records:
        return None
    demo_zero = [record["conditions"]["demo_0"] for record in records]
    scalar_keys = (
        "effective_atoms",
        "top4_atoms_energy_fraction",
        "max_atoms_energy_fraction",
    )
    result: dict[str, Any] = {
        "dictionary_storage": records[0]["dictionary_storage"],
        "demo_0_atom_participation": {
            group: {
                key: distribution([float(value[group][key]) for value in demo_zero])
                for key in scalar_keys
            }
            for group in ("a", "b", "combined", "storage_norm_weighted")
        },
        "demo_0_mixing_row_geometry": {
            group: {
                key: distribution([float(value[group][key]) for value in demo_zero])
                for key in ("stable_row_rank", "top_singular_energy_fraction")
            }
            for group in ("a", "b")
        },
        "cross_task_demo_0_mixing": _vector_set_metrics(
            [_mixing_vector(value) for value in demo_zero]
        ),
    }
    result["same_task_video_mixing"] = _same_task_video_mixing_summary(records)
    result["demo_0_condition_relative_l2"] = _condition_relative_l2_summary(records)
    return result


def effective_variance(
    pairs: Mapping[str, Mapping[str, str]],
    states: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, float]:
    size = len(states)
    gram = np.empty((size, size), dtype=np.float64)
    for left in range(size):
        for right in range(left, size):
            gram[left, right] = gram[right, left] = effective_inner(
                pairs, states[left], states[right]
            )
    sample = float(np.diag(gram).mean())
    mean = float(gram.mean())
    centered = max(sample - mean, 0.0)
    row_mean = gram.mean(axis=1)
    delta_energy = np.diag(gram) - 2.0 * row_mean + mean
    scale_energy = np.square(row_mean - mean) / max(mean, 1e-24)
    scale_like = float(np.maximum(scale_energy, 0.0).mean())
    orthogonal = float(np.maximum(delta_energy - scale_energy, 0.0).mean())
    return {
        "sample_energy": sample,
        "task_mean_energy": mean,
        "centered_variance_over_sample_energy": centered / max(sample, 1e-24),
        "scale_like_video_variance_fraction": scale_like / max(centered, 1e-24),
        "orthogonal_direction_video_variance_fraction": orthogonal
        / max(centered, 1e-24),
    }


def tensor_metrics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float]:
    left = reference.detach().double().reshape(-1)
    right = candidate.detach().double().reshape(-1)
    left_sq = float(left.square().sum())
    right_sq = float(right.square().sum())
    dot = float((left * right).sum())
    return {
        "relative_l2": math.sqrt(
            max(left_sq + right_sq - 2.0 * dot, 0.0) / max(left_sq, 1e-24)
        ),
        "cosine": dot / max(math.sqrt(left_sq * right_sq), 1e-24),
        "reference_l2": math.sqrt(left_sq),
        "candidate_l2": math.sqrt(right_sq),
    }


def tensor_delta_metrics(
    reference: torch.Tensor,
    target: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    """Compare a candidate tensor update with a target tensor update."""

    reference = reference.detach().double().reshape(-1)
    target_delta = target.detach().double().reshape(-1) - reference
    candidate_delta = candidate.detach().double().reshape(-1) - reference
    target_sq = float(target_delta.square().sum())
    candidate_sq = float(candidate_delta.square().sum())
    dot = float((target_delta * candidate_delta).sum())
    residual_sq = float((target_delta - candidate_delta).square().sum())
    return {
        "target_delta_l2": math.sqrt(target_sq),
        "candidate_delta_l2": math.sqrt(candidate_sq),
        "candidate_over_target_delta_l2": math.sqrt(
            candidate_sq / max(target_sq, 1e-24)
        ),
        "delta_cosine": dot / max(math.sqrt(target_sq * candidate_sq), 1e-24),
        "residual_over_target_delta_l2": math.sqrt(residual_sq / max(target_sq, 1e-24)),
    }


def distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }
