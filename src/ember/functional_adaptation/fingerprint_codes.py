"""Fixed policy-functional coordinates shared by decoder fit and task inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file

from ember.pi05_source_checkpoint import read_json


FINGERPRINT_CODE_SCHEMA = "ember_pi05_functional_fingerprint_codes_v1"


@dataclass(frozen=True)
class FunctionalFingerprintCodeSpace:
    train_codes: torch.Tensor
    held_codes: torch.Tensor
    mean: torch.Tensor
    components: torch.Tensor
    scales: torch.Tensor
    explained_variance_fraction: float


@dataclass(frozen=True)
class FunctionalFingerprintCodeTargets:
    root: Path
    train_task_ids: tuple[int, ...]
    held_task_ids: tuple[int, ...]
    train_codes: torch.Tensor
    held_codes: torch.Tensor


def uniformly_spaced_task_ids(
    task_ids: Sequence[int], count: int
) -> tuple[int, ...]:
    """Select fixed anchors from task order without consulting outcomes."""

    values = tuple(int(value) for value in task_ids)
    if not values or len(set(values)) != len(values) or not 0 < count <= len(values):
        raise ValueError("invalid functional-fingerprint anchor request")
    if count == 1:
        return (values[len(values) // 2],)
    indices = tuple(
        round(index * (len(values) - 1) / (count - 1)) for index in range(count)
    )
    if len(set(indices)) != count:
        raise ValueError("functional-fingerprint anchors are not unique")
    return tuple(values[index] for index in indices)


def _orient_components(components: torch.Tensor) -> torch.Tensor:
    """Remove the otherwise arbitrary PCA sign convention."""

    pivots = components.abs().argmax(dim=1)
    signs = components[
        torch.arange(components.shape[0], device=components.device), pivots
    ].sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return components * signs[:, None]


def whiten_functional_fingerprints(
    train_fingerprints: torch.Tensor,
    held_fingerprints: torch.Tensor,
    *,
    code_width: int,
) -> FunctionalFingerprintCodeSpace:
    """Fit PCA only on meta-train responses and transform both task roles."""

    if (
        train_fingerprints.ndim != 2
        or held_fingerprints.ndim != 2
        or train_fingerprints.shape[1] != held_fingerprints.shape[1]
        or not 0 < code_width < train_fingerprints.shape[0]
        or train_fingerprints.shape[1] < code_width
        or not torch.isfinite(train_fingerprints).all()
        or not torch.isfinite(held_fingerprints).all()
    ):
        raise ValueError("invalid policy-functional fingerprint matrix")
    train = train_fingerprints.float()
    held = held_fingerprints.float()
    mean = train.mean(dim=0)
    centered = train - mean
    _, singular_values, right = torch.linalg.svd(centered, full_matrices=False)
    selected = singular_values[:code_width]
    if float(selected[-1]) <= torch.finfo(selected.dtype).eps:
        raise ValueError("policy-functional fingerprints are rank deficient")
    components = _orient_components(right[:code_width])
    scales = selected / (train.shape[0] - 1) ** 0.5
    train_codes = (centered @ components.transpose(0, 1)) / scales
    held_codes = ((held - mean) @ components.transpose(0, 1)) / scales
    explained = float(
        selected.square().sum() / singular_values.square().sum().clamp_min(1e-12)
    )
    return FunctionalFingerprintCodeSpace(
        train_codes=train_codes.contiguous(),
        held_codes=held_codes.contiguous(),
        mean=mean.contiguous(),
        components=components.contiguous(),
        scales=scales.contiguous(),
        explained_variance_fraction=explained,
    )


def load_functional_fingerprint_code_targets(
    root: Path,
    *,
    expected_train_task_ids: Sequence[int],
    expected_held_task_ids: Sequence[int],
    code_width: int,
    device: torch.device | str,
    expected_fit_surface: str = "meta_train_only_pca_whitening",
) -> FunctionalFingerprintCodeTargets:
    """Load one clean train-only-whitened code authority for a fixed task fold."""

    resolved = root.resolve()
    result_path = resolved / "result.json"
    codes_path = resolved / "fingerprint_codes.safetensors"
    result = read_json(result_path)
    train_ids = tuple(int(value) for value in result.get("train_global_task_ids", ()))
    held_ids = tuple(int(value) for value in result.get("held_global_task_ids", ()))
    if (
        result.get("schema_version") != FINGERPRINT_CODE_SCHEMA
        or result.get("formal_authority") is not True
        or result.get("repository", {}).get("dirty_paths") != []
        or result.get("fit_surface") != expected_fit_surface
        or train_ids != tuple(int(value) for value in expected_train_task_ids)
        or held_ids != tuple(int(value) for value in expected_held_task_ids)
        or not codes_path.is_file()
        or codes_path.stat().st_size
        != int(result.get("files", {}).get("fingerprint_codes.safetensors", -1))
    ):
        raise ValueError("functional-fingerprint code authority changed")
    state = load_file(str(codes_path), device=str(device))
    train_codes = state.get("train_codes")
    held_codes = state.get("held_codes")
    if (
        train_codes is None
        or held_codes is None
        or tuple(train_codes.shape) != (len(train_ids), code_width)
        or tuple(held_codes.shape) != (len(held_ids), code_width)
        or not torch.isfinite(train_codes).all()
        or not torch.isfinite(held_codes).all()
    ):
        raise ValueError("functional-fingerprint codes changed shape")
    return FunctionalFingerprintCodeTargets(
        root=resolved,
        train_task_ids=train_ids,
        held_task_ids=held_ids,
        train_codes=train_codes,
        held_codes=held_codes,
    )
