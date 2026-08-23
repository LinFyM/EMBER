"""Fit-only frozen effect-code coordinates for the ECP realizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

from ember.ecp.realizer_evidence import (
    EFFECT_PARTICLE_AUTHORITY_SCHEMA,
    load_effect_member,
    resolve_asset,
)
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic


EFFECT_CODE_AUTHORITY_SCHEMA = "ember_ecp_fixed_effect_code_authority_v1"


def held_global_ids(config: Mapping[str, Any], fold: int) -> tuple[int, ...]:
    ordered = tuple(int(value) for value in config["target_train_global_ids_ordered"])
    if len(ordered) != 24 or not 0 <= fold < 5:
        raise ValueError("effect realizer fold contract changed")
    return tuple(value for index, value in enumerate(ordered) if index % 5 == fold)


def member_fold_role(row: Mapping[str, Any], held: Sequence[int]) -> str:
    return (
        "held_transform_only"
        if str(row["domain"]) == "target_train"
        and int(row["global_task_id"]) in set(held)
        else "fit"
    )


def task_equal_member_weights(
    rows: Sequence[Mapping[str, Any]], selected: Sequence[int]
) -> torch.Tensor:
    groups: dict[str, list[int]] = {}
    for index in selected:
        groups.setdefault(str(rows[index]["asset_key"]), []).append(index)
    if not groups:
        raise ValueError("effect code fit split is empty")
    weights = torch.zeros(len(rows), dtype=torch.float64)
    for indices in groups.values():
        for index in indices:
            weights[index] = 1.0 / (len(groups) * len(indices))
    if not torch.allclose(weights.sum(), torch.tensor(1.0, dtype=torch.float64)):
        raise ValueError("effect code task weights changed")
    return weights


def fit_weighted_owner_pca(
    values: torch.Tensor, weights: torch.Tensor, width: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit independent owner PCA coordinates from weighted observations."""

    if (
        values.ndim != 3
        or weights.shape != (values.shape[0],)
        or not 0 < width <= values.shape[-1]
        or not torch.isfinite(values).all()
        or not torch.isfinite(weights).all()
        or torch.any(weights < 0)
    ):
        raise ValueError("effect-code PCA inputs changed")
    weights = weights.float()
    weights = weights / weights.sum()
    mean = torch.einsum("n,noi->oi", weights, values.float())
    centered = values.float() - mean[None]
    covariance = torch.einsum("n,noi,noj->oij", weights, centered, centered)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(-1)
    components = eigenvectors.flip(-1).transpose(-1, -2)[:, :width]
    selected = eigenvalues[:, :width].clamp_min(
        eigenvalues[:, :1].clamp_min(1e-12) * 1e-6
    )
    scales = selected.sqrt()
    explained = selected.sum(-1) / eigenvalues.clamp_min(0).sum(-1).clamp_min(1e-12)
    return mean, components, scales, explained


def transform_owner_particles(
    owner_delta: torch.Tensor,
    *,
    mean: torch.Tensor,
    components: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    if owner_delta.ndim != 5 or owner_delta.shape[1:] != (8, 38, 4, 128):
        raise ValueError("effect particles changed before coordinate transform")
    flat = owner_delta.float().flatten(-2)
    centered = flat - mean[None, None]
    code = torch.einsum("peoi,ori->peor", centered, components)
    return code / scales[None, None]


def _target_scales(
    rows: Sequence[Mapping[str, Any]],
    fit_indices: Sequence[int],
    weights: torch.Tensor,
    *,
    contract: LoRAContract,
) -> tuple[torch.Tensor, torch.Tensor]:
    a_energy = torch.zeros(len(contract.targets), dtype=torch.float64)
    b_energy = torch.zeros_like(a_energy)
    for index in fit_indices:
        _, residual, _, _ = load_effect_member(
            Path(str(rows[index]["tensor_path"])), contract=contract
        )
        weight = float(weights[index])
        for target_index, target in enumerate(contract.targets):
            a = residual[target.name + LORA_A_SUFFIX].double()
            b = residual[target.name + LORA_B_SUFFIX].double()
            a_energy[target_index] += weight * a.square().mean()
            b_energy[target_index] += weight * b.square().mean()
    return (
        a_energy.clamp_min(1e-12).sqrt().float(),
        b_energy.clamp_min(1e-12).sqrt().float(),
    )


def fit_effect_code_authority(
    *,
    config_path: Path,
    particle_manifest: Path,
    output_dir: Path,
    fold: int,
    asset_root: Path,
    device: torch.device,
    formal: bool,
) -> Path:
    config_path = config_path.resolve()
    config = read_json(config_path)
    particles = read_json(particle_manifest.resolve())
    repository = git_state(Path(__file__).resolve().parents[3])
    if (
        config.get("schema_version") != "ember_ecp_fixed_effect_realizer_v1"
        or particles.get("schema_version") != EFFECT_PARTICLE_AUTHORITY_SCHEMA
        or particles.get("status") != "complete_effect_particle_authority"
        or len(particles.get("members", ())) != 118
        or (
            formal
            and not git_state_is_clean_pushed_or_frozen_authority(repository)
        )
    ):
        raise ValueError("effect-code authority inputs changed")
    rows = tuple(dict(row) for row in particles["members"])
    held = held_global_ids(config, fold)
    roles = tuple(member_fold_role(row, held) for row in rows)
    fit_indices = tuple(index for index, role in enumerate(roles) if role == "fit")
    held_indices = tuple(
        index for index, role in enumerate(roles) if role == "held_transform_only"
    )
    weights = task_equal_member_weights(rows, fit_indices)
    contract = load_pi05_lora_contract(
        resolve_asset(asset_root, config["authorities"]["lora_contract"])
    )

    observations, observation_weights = [], []
    for index in fit_indices:
        owner, _, _, _ = load_effect_member(
            Path(str(rows[index]["tensor_path"])), contract=contract, device=device
        )
        flattened = owner.float().flatten(-2).reshape(-1, 38, 512)
        observations.append(flattened)
        observation_weights.append(
            torch.full(
                (flattened.shape[0],),
                float(weights[index]) / flattened.shape[0],
                device=device,
            )
        )
    values = torch.cat(observations)
    value_weights = torch.cat(observation_weights)
    width = int(config["effect_code"]["output_width"])
    mean, components, scales, explained = fit_weighted_owner_pca(
        values, value_weights, width
    )
    a_scales, b_scales = _target_scales(
        rows, fit_indices, weights, contract=contract
    )

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    transform_path = output_dir / "transform.safetensors"
    save_file(
        {
            "mean": mean.cpu().contiguous(),
            "components": components.cpu().contiguous(),
            "scales": scales.cpu().contiguous(),
            "explained_variance_fraction": explained.cpu().contiguous(),
            "target_a_scales": a_scales,
            "target_b_scales": b_scales,
        },
        str(transform_path),
    )
    code_root = output_dir / "members"
    code_root.mkdir()
    output_rows = []
    for index, row in enumerate(rows):
        owner, _, trajectory_ids, probe_signs = load_effect_member(
            Path(str(row["tensor_path"])), contract=contract, device=device
        )
        code = transform_owner_particles(
            owner,
            mean=mean,
            components=components,
            scales=scales,
        )
        path = code_root / f"member_{index:03d}.safetensors"
        save_file(
            {
                "effect_code": code.cpu().to(torch.bfloat16).contiguous(),
                "particle_trajectory_ids": trajectory_ids.cpu(),
                "particle_probe_signs": probe_signs.cpu(),
            },
            str(path),
        )
        output_rows.append(
            {
                **row,
                "fold_role": roles[index],
                "effect_code_path": str(path),
                "effect_code_bytes": path.stat().st_size,
            }
        )
    manifest = output_dir / "manifest.json"
    write_json_atomic(
        manifest,
        {
            "schema_version": EFFECT_CODE_AUTHORITY_SCHEMA,
            "status": "complete_fit_only_effect_code_coordinate",
            "repository": repository,
            "config": {"path": str(config_path), "bytes": config_path.stat().st_size},
            "particle_authority": str(particle_manifest.resolve()),
            "fold": fold,
            "held_global_task_ids": list(held),
            "fit_tasks": len({rows[index]["asset_key"] for index in fit_indices}),
            "fit_members": len(fit_indices),
            "held_tasks": len({rows[index]["asset_key"] for index in held_indices}),
            "held_members": len(held_indices),
            "coordinate": {
                "input_shape": [38, 4, 128],
                "output_shape": [38, width],
                "fit_weighting": "equal task then equal member then event and particle",
                "held_updates": 0,
                "explained_variance_fraction_min": float(explained.min()),
                "explained_variance_fraction_mean": float(explained.mean()),
                "transform_path": str(transform_path),
                "transform_bytes": transform_path.stat().st_size,
            },
            "members": output_rows,
            "information_wall": {
                "task_id_model_input": False,
                "validation_action_or_reward_reads": 0,
                "test_action_or_reward_reads": 0,
                "held_optimizer_steps": 0,
            },
        },
    )
    return manifest
