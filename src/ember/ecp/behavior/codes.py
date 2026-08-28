"""Fit-only policy-behavior coordinates for Natural Program alignment.

The coordinate authority is produced from two disjoint cross-episode
flow-gradient panels on the 75 fit tasks.  Held-task coordinates are sealed for
zero-gradient qualification only.  None of these tensors is a deployment
input: the shared decoder reads only the deployed Natural Program fields.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file


BEHAVIOR_CODE_SCHEMA = "ember_ecp_g2_behavior_codes_v1"


@dataclass(frozen=True)
class BehaviorCodeAuthority:
    task_ids: tuple[int, ...]
    fit_task_ids: tuple[int, ...]
    held_task_ids: tuple[int, ...]
    selected_targets: tuple[int, ...]
    coordinates: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor
    eigenvectors: torch.Tensor
    eigenvalues: torch.Tensor
    norms: torch.Tensor
    train_sqrt_weights: torch.Tensor
    factor_roots: tuple[Path, ...]
    manifest: Mapping[str, Any]

    @property
    def dimension(self) -> int:
        return int(self.coordinates.shape[-1])

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


def _tensor(handle: Any, name: str, device: torch.device) -> torch.Tensor:
    value = handle.get_tensor(name).to(device=device)
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"behavior-code tensor is non-finite: {name}")
    return value


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
    task_ids = tuple(map(int, task_ids_tensor.tolist()))
    fit = tuple(map(int, fit_tensor.tolist()))
    held = tuple(map(int, held_tensor.tolist()))
    selected = tuple(map(int, targets_tensor.tolist()))
    dimension = int(manifest.get("dimension", -1))
    if (
        task_ids != tuple(range(95))
        or len(fit) != 75
        or len(held) != 20
        or set(fit) & set(held)
        or set(fit) | set(held) != set(task_ids)
        or len(selected) != 8
        or coordinates.shape != (95, 8, dimension)
        or mean.shape != (8, dimension)
        or scale.shape != (8, dimension)
        or eigenvectors.shape != (8, 75, dimension)
        or eigenvalues.shape != (8, dimension)
        or norms.shape != (8, 95)
        or train_sqrt_weights.shape != (75,)
        or bool((scale <= 0).any())
        or bool((eigenvalues <= 0).any())
        or not math.isclose(
            float(train_sqrt_weights.square().sum()), 1.0, rel_tol=1e-5
        )
    ):
        raise ValueError("behavior-code tensor contract changed")
    return BehaviorCodeAuthority(
        task_ids=task_ids,
        fit_task_ids=fit,
        held_task_ids=held,
        selected_targets=selected,
        coordinates=coordinates,
        mean=mean,
        scale=scale,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        norms=norms,
        train_sqrt_weights=train_sqrt_weights,
        factor_roots=factor_roots,
        manifest=manifest,
    )


def load_program_model_initialization(
    model: torch.nn.Module,
    checkpoint: Path,
    *,
    device: torch.device,
    allowed_new_prefix: str,
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
        not missing
        or any(not name.startswith(allowed_new_prefix) for name in missing)
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


class BehaviorCodeDecoder(torch.nn.Module):
    """Training-only shared reader of event-bearing Program fields.

    The decoder has no task lookup and deliberately cannot read ``P_lang`` or
    ``P_scene``.  Its only route is the video-derived process, uncertainty,
    event mass, and aligned time.  It is not called by deployment extraction.
    """

    def __init__(
        self,
        *,
        program_width: int,
        hidden_width: int,
        event_slots: int,
        selected_targets: Sequence[int],
        family_ids: Sequence[int],
        family_count: int,
        dimension: int,
    ) -> None:
        super().__init__()
        targets = tuple(map(int, selected_targets))
        families = tuple(map(int, family_ids))
        if (
            len(targets) != 8
            or len(set(targets)) != len(targets)
            or len(families) != len(targets)
            or min(targets) < 0
            or max(targets) >= 38
            or min(families) < 0
            or max(families) >= family_count
            or dimension <= 0
        ):
            raise ValueError("invalid behavior-code decoder contract")
        self.register_buffer(
            "selected_targets", torch.tensor(targets, dtype=torch.long)
        )
        self.register_buffer("family_ids", torch.tensor(families, dtype=torch.long))
        self.field_context = torch.nn.Sequential(
            torch.nn.LayerNorm(2 * program_width),
            torch.nn.Linear(2 * program_width, 2 * hidden_width),
            torch.nn.GELU(),
            torch.nn.Linear(2 * hidden_width, hidden_width),
            torch.nn.LayerNorm(hidden_width),
        )
        self.scalar_context = torch.nn.Linear(3, hidden_width, bias=False)
        self.target_embedding = torch.nn.Parameter(
            torch.empty(len(targets), hidden_width)
        )
        self.family_embedding = torch.nn.Parameter(
            torch.empty(family_count, hidden_width)
        )
        self.event_embedding = torch.nn.Parameter(
            torch.empty(event_slots, hidden_width)
        )
        self.event_score = torch.nn.Linear(hidden_width, 1)
        self.output_trunk = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_width),
            torch.nn.Linear(hidden_width, hidden_width),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_width),
        )
        self.output_heads = torch.nn.ModuleList(
            torch.nn.Linear(hidden_width, dimension, bias=False)
            for _ in targets
        )
        for value in (
            self.target_embedding,
            self.family_embedding,
            self.event_embedding,
        ):
            torch.nn.init.normal_(value, std=hidden_width**-0.5)
        torch.nn.init.zeros_(self.event_score.weight)
        torch.nn.init.zeros_(self.event_score.bias)

    def forward(self, program: Any) -> torch.Tensor:
        targets = self.selected_targets.to(program.p_process.device)
        process = program.p_process.float().index_select(-2, targets)
        sigma = program.sigma.float().index_select(-2, targets)
        fields = torch.cat((process, sigma), dim=-1)
        rho = program.rho.float().clamp_min(1e-8)
        rho = rho / rho.sum(-1, keepdim=True).clamp_min(1e-8)
        scalar = self.scalar_context(
            torch.cat((rho[..., None], program.tau.float()), dim=-1)
        )
        context = (
            self.field_context(fields)
            + scalar[:, :, None]
            + self.event_embedding[None, :, None]
            + self.target_embedding[None, None]
            + self.family_embedding[self.family_ids][None, None]
        )
        weights = (
            self.event_score(context).squeeze(-1) + rho[:, :, None].log()
        ).softmax(1)
        pooled = torch.einsum("cej,cejw->cjw", weights, context)
        hidden = self.output_trunk(pooled)
        return torch.stack(
            [head(hidden[:, index]) for index, head in enumerate(self.output_heads)],
            dim=1,
        )


def behavior_alignment_loss(
    prediction: torch.Tensor,
    robust_prediction: torch.Tensor,
    standardized_target: torch.Tensor,
) -> torch.Tensor:
    target = standardized_target.float()[None]
    if prediction.shape != target.shape or robust_prediction.shape != target.shape:
        raise ValueError("behavior-code prediction shape changed")
    return 0.5 * (
        F.mse_loss(prediction.float(), target)
        + F.mse_loss(robust_prediction.float(), target)
    )
