"""Fit-only centered two-sided functional coordinates for ECP corrections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import save_file

from ember.ecp.low_rank import canonicalize_low_rank_factors
from ember.ecp.realizer_code import (
    held_global_ids,
    member_fold_role,
    task_equal_member_weights,
)
from ember.ecp.realizer_evidence import (
    EFFECT_PARTICLE_AUTHORITY_SCHEMA,
    load_effect_member,
    resolve_asset,
)
from ember.ecp.stage1_parameterization import effective_inner_product
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRAContract
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic


TWO_SIDED_COORDINATE_SCHEMA = "ember_ecp_centered_two_sided_authority_v1"


@dataclass(frozen=True)
class CoordinateAuthority:
    config_path: Path
    particle_manifest: Path
    config: Mapping[str, Any]
    repository: Mapping[str, Any]
    rows: tuple[dict[str, Any], ...]
    fold: int
    held: tuple[int, ...]
    roles: tuple[str, ...]
    fit_indices: tuple[int, ...]
    held_indices: tuple[int, ...]
    weights: torch.Tensor
    contract: LoRAContract


def fixed_two_sided_probes(
    *,
    in_features: int,
    out_features: int,
    width: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic task-independent input and output probes."""

    if width <= 0 or width > min(in_features, out_features):
        raise ValueError("two-sided probe width is incompatible with target shape")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    omega, _ = torch.linalg.qr(
        torch.randn(in_features, width, generator=generator), mode="reduced"
    )
    psi, _ = torch.linalg.qr(
        torch.randn(out_features, width, generator=generator), mode="reduced"
    )
    return omega.to(device), psi.to(device)


def two_sided_sketch(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    omega: torch.Tensor,
    psi: torch.Tensor,
) -> torch.Tensor:
    """Flatten ``[deltaW Omega, Psi^T deltaW]`` without forming deltaW."""

    if (
        a.ndim != 2
        or b.ndim != 2
        or a.shape[0] != b.shape[1]
        or omega.shape[0] != a.shape[1]
        or psi.shape[0] != b.shape[0]
        or omega.shape[1] != psi.shape[1]
    ):
        raise ValueError("two-sided sketch factor or probe shapes changed")
    a = a.float()
    b = b.float()
    y = b @ (a @ omega.float())
    z = (psi.float().T @ b) @ a
    return torch.cat((y.flatten(), z.flatten()))


def fit_weighted_sketch_basis(
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    width: int,
    relative_eigenvalue_floor: float,
    scale_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a whitened feature basis through the weighted sample Gram matrix."""

    if (
        values.ndim != 2
        or weights.shape != (values.shape[0],)
        or not 0 < width <= values.shape[1]
        or not 0 < relative_eigenvalue_floor < 1
        or not 0 < scale_floor < 1
    ):
        raise ValueError("two-sided coordinate fit inputs changed")
    weights = weights.double()
    weights = weights / weights.sum()
    values = values.double()
    mean = torch.einsum("n,nd->d", weights, values)
    centered = values - mean
    weighted = centered * weights.sqrt()[:, None]
    gram = weighted @ weighted.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    eigenvectors = eigenvectors[:, order]
    largest = eigenvalues[0].clamp_min(torch.finfo(eigenvalues.dtype).eps)
    active = min(
        width,
        int(torch.count_nonzero(eigenvalues > largest * relative_eigenvalue_floor)),
    )
    if active == 0:
        raise ValueError("two-sided coordinate has no active fit direction")
    selected = eigenvalues[:active]
    components_active = (eigenvectors[:, :active].T @ weighted) / selected.sqrt()[:, None]
    scales_active = selected.sqrt().clamp_min(selected[0].sqrt() * scale_floor)
    components = torch.zeros(width, values.shape[1], dtype=torch.float32)
    scales = torch.ones(width, dtype=torch.float32)
    mask = torch.zeros(width, dtype=torch.uint8)
    components[:active] = components_active.float()
    scales[:active] = scales_active.float()
    mask[:active] = 1
    return mean.float(), components, scales, mask, eigenvalues[:active].float()


def transform_two_sided_sketch(
    sketch: torch.Tensor,
    *,
    mean: torch.Tensor,
    components: torch.Tensor,
    scales: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    code = ((sketch.float() - mean.float()) @ components.float().T) / scales.float()
    return code * active_mask.to(code)


def decode_two_sided_code(
    code: torch.Tensor,
    *,
    mean: torch.Tensor,
    components: torch.Tensor,
    scales: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    weighted = code.float() * scales.float() * active_mask.to(code)
    return mean.float() + weighted @ components.float()


def reconstruct_rank4_factors(
    sketch: torch.Tensor,
    *,
    omega: torch.Tensor,
    psi: torch.Tensor,
    out_features: int,
    in_features: int,
    rank: int = 4,
    relative_singular_floor: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically reconstruct a rank-four update from decoded sketches."""

    probe_width = int(omega.shape[1])
    y_size = out_features * probe_width
    if sketch.numel() != y_size + probe_width * in_features:
        raise ValueError("decoded two-sided sketch shape changed")
    y = sketch[:y_size].reshape(out_features, probe_width).float()
    z = sketch[y_size:].reshape(probe_width, in_features).float()
    core = 0.5 * (psi.float().T @ y + z @ omega.float())
    u, singular, vh = torch.linalg.svd(core, full_matrices=False)
    if rank > singular.numel():
        raise ValueError("two-sided reconstruction rank exceeds probe width")
    selected = singular[:rank]
    floor = selected[0].clamp_min(torch.finfo(selected.dtype).eps) * relative_singular_floor
    inverse_root = selected.clamp_min(floor).rsqrt()
    b = (y @ vh[:rank].T) * inverse_root[None]
    a = inverse_root[:, None] * (u[:, :rank].T @ z)
    return canonicalize_low_rank_factors(a, b, output_rank=rank)


def _target_transform(
    tensors: Mapping[str, torch.Tensor], target_index: int
) -> dict[str, torch.Tensor]:
    prefix = f"target_{target_index:02d}_"
    return {
        name: tensors[prefix + name]
        for name in ("omega", "psi", "mean", "components", "scales", "active_mask")
    }


def _geometry(
    predicted: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted_a, predicted_b = predicted
    expected_a, expected_b = expected
    predicted_energy = effective_inner_product(
        predicted_b, predicted_a, predicted_b, predicted_a
    )
    expected_energy = effective_inner_product(
        expected_b, expected_a, expected_b, expected_a
    )
    cross = effective_inner_product(predicted_b, predicted_a, expected_b, expected_a)
    return predicted_energy, expected_energy, cross


def _coordinate_authority(args: Any) -> CoordinateAuthority:
    config_path = args.config.resolve()
    particle_manifest = args.particle_manifest.resolve()
    asset_root = args.asset_root.resolve()
    config = read_json(config_path)
    particles = read_json(particle_manifest)
    repository = git_state(Path(__file__).resolve().parents[3])
    formal = args.mode == "formal"
    fold = int(config["fold"]["index"])
    if (
        config.get("schema_version") != "ember_ecp_centered_two_sided_coordinate_v1"
        or particles.get("schema_version") != EFFECT_PARTICLE_AUTHORITY_SCHEMA
        or particles.get("status") != "complete_effect_particle_authority"
        or len(particles.get("members", ())) != 118
        or (formal and not git_state_is_clean_pushed_or_frozen_authority(repository))
    ):
        raise ValueError("two-sided coordinate authority inputs changed")
    rows = tuple(dict(row) for row in particles["members"])
    held = held_global_ids(config, fold)
    if tuple(config["fold"]["held_global_ids"]) != held:
        raise ValueError("two-sided held fold declaration changed")
    roles = tuple(member_fold_role(row, held) for row in rows)
    fit_indices = tuple(index for index, role in enumerate(roles) if role == "fit")
    held_indices = tuple(index for index, role in enumerate(roles) if role != "fit")
    weights = task_equal_member_weights(rows, fit_indices)
    if (
        len(fit_indices) != 108
        or len(held_indices) != 10
        or len({rows[index]["asset_key"] for index in fit_indices}) != 90
        or len({rows[index]["asset_key"] for index in held_indices}) != 5
    ):
        raise ValueError("two-sided fold membership changed")
    contract = load_pi05_lora_contract(
        resolve_asset(asset_root, config["authorities"]["lora_contract"])
    )
    return CoordinateAuthority(
        config_path=config_path,
        particle_manifest=particle_manifest,
        config=config,
        repository=repository,
        rows=rows,
        fold=fold,
        held=held,
        roles=roles,
        fit_indices=fit_indices,
        held_indices=held_indices,
        weights=weights,
        contract=contract,
    )


def _fit_transform(
    authority: CoordinateAuthority, transform_path: Path
) -> tuple[dict[str, torch.Tensor], list[int]]:
    coordinate = authority.config["coordinate"]
    probe_width = int(coordinate["probe_width"])
    output_width = int(coordinate["output_width"])
    probes = []
    fit_values: list[list[torch.Tensor]] = [[] for _ in authority.contract.targets]
    for target_index, target in enumerate(authority.contract.targets):
        probes.append(
            fixed_two_sided_probes(
                in_features=target.in_features,
                out_features=target.out_features,
                width=probe_width,
                seed=int(coordinate["probe_seed"])
                + int(coordinate["probe_seed_stride"]) * target_index,
            )
        )
    for member_index in authority.fit_indices:
        _, residual, _, _ = load_effect_member(
            Path(authority.rows[member_index]["tensor_path"]),
            contract=authority.contract,
        )
        for target_index, target in enumerate(authority.contract.targets):
            omega, psi = probes[target_index]
            fit_values[target_index].append(
                two_sided_sketch(
                    residual[target.name + LORA_A_SUFFIX],
                    residual[target.name + LORA_B_SUFFIX],
                    omega=omega,
                    psi=psi,
                )
            )

    transform_tensors: dict[str, torch.Tensor] = {}
    active_ranks = []
    fit_weights = authority.weights[list(authority.fit_indices)]
    for target_index, values in enumerate(fit_values):
        mean, components, scales, mask, eigenvalues = fit_weighted_sketch_basis(
            torch.stack(values),
            fit_weights,
            width=output_width,
            relative_eigenvalue_floor=float(
                coordinate["relative_eigenvalue_floor"]
            ),
            scale_floor=float(
                coordinate["whitening_scale_floor_relative_to_largest"]
            ),
        )
        omega, psi = probes[target_index]
        prefix = f"target_{target_index:02d}_"
        transform_tensors.update(
            {
                prefix + "omega": omega.contiguous(),
                prefix + "psi": psi.contiguous(),
                prefix + "mean": mean.contiguous(),
                prefix + "components": components.contiguous(),
                prefix + "scales": scales.contiguous(),
                prefix + "active_mask": mask.contiguous(),
                prefix + "eigenvalues": eigenvalues.contiguous(),
            }
        )
        active_ranks.append(int(mask.sum()))
    save_file(
        transform_tensors,
        str(transform_path),
        metadata={"schema_version": TWO_SIDED_COORDINATE_SCHEMA},
    )
    return transform_tensors, active_ranks


def _member_code_and_geometry(
    residual: Mapping[str, torch.Tensor],
    *,
    contract: LoRAContract,
    coordinate: Mapping[str, Any],
    transform_tensors: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    codes = []
    predicted_energy = torch.tensor(0.0)
    expected_energy = torch.tensor(0.0)
    cross = torch.tensor(0.0)
    coordinate_keys = ("mean", "components", "scales", "active_mask")
    for target_index, target in enumerate(contract.targets):
        tensors = _target_transform(transform_tensors, target_index)
        sketch = two_sided_sketch(
            residual[target.name + LORA_A_SUFFIX],
            residual[target.name + LORA_B_SUFFIX],
            omega=tensors["omega"],
            psi=tensors["psi"],
        )
        code = transform_two_sided_sketch(
            sketch,
            **{
                name: tensors[name]
                for name in coordinate_keys
            },
        )
        decoded = decode_two_sided_code(
            code,
            **{
                name: tensors[name]
                for name in coordinate_keys
            },
        )
        prediction = reconstruct_rank4_factors(
            decoded,
            omega=tensors["omega"],
            psi=tensors["psi"],
            out_features=target.out_features,
            in_features=target.in_features,
            rank=int(coordinate["reconstruction_rank"]),
            relative_singular_floor=float(
                coordinate["core_inverse_relative_singular_floor"]
            ),
        )
        geometry = _geometry(
            prediction,
            (
                residual[target.name + LORA_A_SUFFIX].float(),
                residual[target.name + LORA_B_SUFFIX].float(),
            ),
        )
        predicted_energy += geometry[0].cpu()
        expected_energy += geometry[1].cpu()
        cross += geometry[2].cpu()
        codes.append(code.cpu())
    error = (predicted_energy + expected_energy - 2 * cross).clamp_min(0)
    return torch.stack(codes), {
        "effective_cosine": float(
            cross / (predicted_energy * expected_energy).clamp_min(1e-20).sqrt()
        ),
        "relative_error": float((error / expected_energy).sqrt()),
        "energy_ratio": float(predicted_energy / expected_energy),
    }


def _transform_members(
    authority: CoordinateAuthority,
    *,
    transform_tensors: Mapping[str, torch.Tensor],
    member_root: Path,
) -> list[dict[str, Any]]:
    output_rows = []
    for member_index, row in enumerate(authority.rows):
        _, residual, _, _ = load_effect_member(
            Path(row["tensor_path"]), contract=authority.contract
        )
        code, geometry = _member_code_and_geometry(
            residual,
            contract=authority.contract,
            coordinate=authority.config["coordinate"],
            transform_tensors=transform_tensors,
        )
        code_path = member_root / f"member_{member_index:03d}.safetensors"
        save_file({"effect_code": code}, str(code_path))
        output_rows.append(
            {
                **row,
                "fold_role": authority.roles[member_index],
                "effect_code_path": str(code_path),
                "effect_code_bytes": code_path.stat().st_size,
                "coordinate_geometry": geometry,
            }
        )
    return output_rows


def _write_manifest(
    authority: CoordinateAuthority,
    *,
    output_dir: Path,
    transform_path: Path,
    active_ranks: list[int],
    output_rows: list[dict[str, Any]],
) -> Path:
    coordinate = authority.config["coordinate"]
    manifest = output_dir / "manifest.json"
    write_json_atomic(
        manifest,
        {
            "schema_version": TWO_SIDED_COORDINATE_SCHEMA,
            "status": "complete_fit_only_centered_two_sided_coordinate",
            "repository": authority.repository,
            "config": {
                "path": str(authority.config_path),
                "bytes": authority.config_path.stat().st_size,
            },
            "particle_authority": str(authority.particle_manifest),
            "fold": authority.fold,
            "held_global_task_ids": list(authority.held),
            "fit_tasks": 90,
            "fit_members": len(authority.fit_indices),
            "held_tasks": 5,
            "held_members": len(authority.held_indices),
            "coordinate": {
                "sketch_shape": [38, int(coordinate["output_width"])],
                "probe_width": int(coordinate["probe_width"]),
                "active_rank_min": min(active_ranks),
                "active_rank_max": max(active_ranks),
                "fit_weighting": "equal task then equal member",
                "held_updates": 0,
                "transform_path": str(transform_path),
                "transform_bytes": transform_path.stat().st_size,
            },
            "members": output_rows,
            "information_wall": {
                "held_target_residual_reads": len(authority.held_indices),
                "held_coordinate_updates": 0,
                "held_optimizer_steps": 0,
                "validation_action_or_reward_reads": 0,
                "test_action_or_reward_reads": 0,
                "task_id_model_input": False,
            },
        },
    )
    return manifest


def build_centered_two_sided_coordinate(args: Any) -> Path:
    """Fit the fold coordinate, then transform all members without held updates."""

    authority = _coordinate_authority(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    transform_path = output_dir / "transform.safetensors"
    transform_tensors, active_ranks = _fit_transform(authority, transform_path)
    member_root = output_dir / "members"
    member_root.mkdir()
    output_rows = _transform_members(
        authority, transform_tensors=transform_tensors, member_root=member_root
    )
    return _write_manifest(
        authority,
        output_dir=output_dir,
        transform_path=transform_path,
        active_ranks=active_ranks,
        output_rows=output_rows,
    )
