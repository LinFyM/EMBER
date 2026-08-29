"""Fit-only policy-behavior authority for Natural Program topology.

The coordinate authority is produced from two disjoint cross-episode
flow-gradient panels.  An internal task fold owns gradients while both the
internal and official held tasks remain zero-gradient qualifications.  None of
these tensors is a deployment input.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file


BEHAVIOR_CODE_SCHEMA = "ember_ecp_g2_behavior_authority_v3"


def fixed_internal_behavior_fold(
    fit_tasks: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the pre-registered role-stratified 60/15 fold zero."""

    meta = tuple(task for task in fit_tasks if task < 71)
    target = tuple(task for task in fit_tasks if task >= 71)
    internal = tuple(meta[index] for index in range(4, len(meta), 5)) + tuple(
        target[index] for index in (3, 8, 13, 18)
    )
    gradient = tuple(task for task in fit_tasks if task not in set(internal))
    if len(gradient) != 60 or len(internal) != 15:
        raise ValueError("behavior internal fold zero changed")
    return gradient, internal


@dataclass(frozen=True)
class BehaviorCodeAuthority:
    task_ids: tuple[int, ...]
    fit_task_ids: tuple[int, ...]
    held_task_ids: tuple[int, ...]
    official_held_task_ids: tuple[int, ...]
    selected_targets: tuple[int, ...]
    coordinates: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor
    eigenvectors: torch.Tensor
    eigenvalues: torch.Tensor
    norms: torch.Tensor
    train_sqrt_weights: torch.Tensor
    panel_a_gram: torch.Tensor
    panel_b_gram: torch.Tensor
    consensus_gram: torch.Tensor
    factor_roots: tuple[Path, ...]
    manifest: Mapping[str, Any]

    @property
    def dimension(self) -> int:
        return int(self.coordinates.shape[-1])

    @property
    def meta_gradient_task_ids(self) -> frozenset[int]:
        return frozenset(task for task in self.fit_task_ids if task < 71)

    @property
    def target_gradient_task_ids(self) -> frozenset[int]:
        return frozenset(task for task in self.fit_task_ids if task >= 71)

    def _row(self, authority_id: int) -> int:
        try:
            return self.task_ids.index(int(authority_id))
        except ValueError as error:
            raise KeyError(f"unknown behavior-code task: {authority_id}") from error

    def target(self, authority_id: int, *, standardized: bool) -> torch.Tensor:
        value = self.coordinates[self._row(authority_id)]
        return (value - self.mean) / self.scale if standardized else value

    def decode(self, standardized: torch.Tensor) -> torch.Tensor:
        return standardized.float() * self.scale + self.mean

    def kernel(self, task_ids: torch.Tensor, *, kind: str) -> torch.Tensor:
        values = {
            "panel_a": self.panel_a_gram,
            "panel_b": self.panel_b_gram,
            "consensus": self.consensus_gram,
        }
        if kind not in values:
            raise ValueError(f"unsupported behavior kernel: {kind}")
        rows = torch.tensor(
            [self._row(int(task)) for task in task_ids.detach().cpu().tolist()],
            dtype=torch.long,
            device=values[kind].device,
        )
        return values[kind].index_select(1, rows).index_select(2, rows)


def _tensor(handle: Any, name: str, device: torch.device) -> torch.Tensor:
    value = handle.get_tensor(name).to(device=device)
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"behavior-code tensor is non-finite: {name}")
    return value


def _validate_behavior_tensor_contract(
    *,
    task_ids: tuple[int, ...],
    fit: tuple[int, ...],
    held: tuple[int, ...],
    official: tuple[int, ...],
    selected: tuple[int, ...],
    dimension: int,
    tensors: Mapping[str, torch.Tensor],
) -> None:
    if len(task_ids) != 75 or tuple(sorted(task_ids)) != task_ids:
        raise ValueError("behavior-code task authority changed")
    if (len(fit), len(held), len(official), len(selected)) != (60, 15, 20, 8):
        raise ValueError("behavior-code fold changed")
    if (
        set(fit) & set(held)
        or (set(fit) | set(held)) & set(official)
        or set(fit) | set(held) != set(task_ids)
        or set(task_ids) | set(official) != set(range(95))
    ):
        raise ValueError("behavior-code fold overlap changed")
    expected_shapes = {
        "coordinates": (75, 8, dimension),
        "mean": (8, dimension),
        "scale": (8, dimension),
        "eigenvectors": (8, 60, dimension),
        "eigenvalues": (8, dimension),
        "norms": (8, 75),
        "train_sqrt_weights": (60,),
        "panel_a_gram": (8, 75, 75),
        "panel_b_gram": (8, 75, 75),
        "consensus_gram": (8, 75, 75),
    }
    if any(tensors[name].shape != shape for name, shape in expected_shapes.items()):
        raise ValueError("behavior-code tensor shape changed")
    for name in ("panel_a_gram", "panel_b_gram", "consensus_gram"):
        diagonal = tensors[name].diagonal(dim1=-2, dim2=-1)
        if not torch.allclose(
            diagonal, torch.ones_like(diagonal), atol=1e-3, rtol=1e-3
        ):
            raise ValueError("behavior-code factor-cosine kernel changed")
    if bool((tensors["scale"] <= 0).any()) or bool(
        (tensors["eigenvalues"] <= 0).any()
    ):
        raise ValueError("behavior-code basis scale changed")
    if not math.isclose(
        float(tensors["train_sqrt_weights"].square().sum()), 1.0, rel_tol=1e-5
    ):
        raise ValueError("behavior-code role weights changed")


def load_behavior_code_authority(
    manifest_path: Path, *, asset_root: Path, device: torch.device
) -> BehaviorCodeAuthority:
    import json

    manifest = json.loads(manifest_path.read_text())
    tensor_path = manifest_path.parent / str(manifest.get("tensor_file", ""))
    if (
        manifest.get("schema_version") != BEHAVIOR_CODE_SCHEMA
        or manifest.get("status") != "complete"
        or not tensor_path.is_file()
        or tensor_path.stat().st_size != int(manifest.get("tensor_bytes", -1))
    ):
        raise ValueError("behavior-code authority changed")
    factor_roots = tuple(
        (asset_root / str(value)).resolve()
        for value in manifest.get("factor_roots", ())
    )
    if len(factor_roots) != 2 or any(not root.is_dir() for root in factor_roots):
        raise ValueError("behavior-code factor provenance changed")
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        task_ids_tensor = _tensor(handle, "task_ids", torch.device("cpu"))
        fit_tensor = _tensor(handle, "fit_task_ids", torch.device("cpu"))
        held_tensor = _tensor(handle, "held_task_ids", torch.device("cpu"))
        official_tensor = _tensor(
            handle, "official_held_task_ids", torch.device("cpu")
        )
        targets_tensor = _tensor(handle, "selected_targets", torch.device("cpu"))
        coordinates = _tensor(handle, "coordinates", device).float()
        mean = _tensor(handle, "mean", device).float()
        scale = _tensor(handle, "scale", device).float()
        eigenvectors = _tensor(handle, "eigenvectors", device).float()
        eigenvalues = _tensor(handle, "eigenvalues", device).float()
        norms = _tensor(handle, "norms", device).float()
        train_sqrt_weights = _tensor(
            handle, "train_sqrt_weights", device
        ).float()
        panel_a_gram = _tensor(handle, "panel_a_gram", device).float()
        panel_b_gram = _tensor(handle, "panel_b_gram", device).float()
        consensus_gram = _tensor(handle, "consensus_gram", device).float()
    task_ids = tuple(map(int, task_ids_tensor.tolist()))
    fit = tuple(map(int, fit_tensor.tolist()))
    held = tuple(map(int, held_tensor.tolist()))
    official = tuple(map(int, official_tensor.tolist()))
    selected = tuple(map(int, targets_tensor.tolist()))
    dimension = int(manifest.get("dimension", -1))
    tensors = {
        "coordinates": coordinates,
        "mean": mean,
        "scale": scale,
        "eigenvectors": eigenvectors,
        "eigenvalues": eigenvalues,
        "norms": norms,
        "train_sqrt_weights": train_sqrt_weights,
        "panel_a_gram": panel_a_gram,
        "panel_b_gram": panel_b_gram,
        "consensus_gram": consensus_gram,
    }
    _validate_behavior_tensor_contract(
        task_ids=task_ids,
        fit=fit,
        held=held,
        official=official,
        selected=selected,
        dimension=dimension,
        tensors=tensors,
    )
    return BehaviorCodeAuthority(
        task_ids=task_ids,
        fit_task_ids=fit,
        held_task_ids=held,
        official_held_task_ids=official,
        selected_targets=selected,
        coordinates=coordinates,
        mean=mean,
        scale=scale,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        norms=norms,
        train_sqrt_weights=train_sqrt_weights,
        panel_a_gram=panel_a_gram,
        panel_b_gram=panel_b_gram,
        consensus_gram=consensus_gram,
        factor_roots=factor_roots,
        manifest=manifest,
    )


def load_program_model_initialization(
    model: torch.nn.Module,
    checkpoint: Path,
    *,
    device: torch.device,
    allowed_new_prefix: str | None,
    expected_macro: int,
) -> dict[str, Any]:
    import json

    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
    weights = checkpoint / "ecp.safetensors"
    if (
        manifest.get("schema_version") != "ember_ecp_checkpoint_v1"
        or manifest.get("stage") != "g2_natural_program"
        or int(manifest.get("next_macro", -1)) != expected_macro
        or manifest.get("run_contract_schema")
        != "ember_ecp_natural_program_g2_run_v2"
        or int(manifest.get("world_size", -1)) != 4
        or not weights.is_file()
        or weights.stat().st_size
        != int(manifest.get("files", {}).get("ecp.safetensors", {}).get("bytes", -1))
    ):
        raise ValueError("qualified G2 initialization authority changed")
    source = load_file(str(weights), device=str(device))
    current = model.state_dict()
    missing = sorted(set(current) - set(source))
    unexpected = sorted(set(source) - set(current))
    mismatched = sorted(
        name
        for name in set(current) & set(source)
        if current[name].shape != source[name].shape
    )
    if (
        (missing and allowed_new_prefix is None)
        or any(
            allowed_new_prefix is None or not name.startswith(allowed_new_prefix)
            for name in missing
        )
        or unexpected
        or mismatched
    ):
        raise ValueError("G2 model-only initialization topology changed")
    result = model.load_state_dict(source, strict=False)
    if sorted(result.missing_keys) != missing or result.unexpected_keys:
        raise ValueError("G2 model-only initialization load changed")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_macro": expected_macro,
        "loaded_tensors": len(source),
        "fresh_tensors": len(missing),
        "fresh_prefix": allowed_new_prefix,
        "optimizer_loaded": False,
    }
