"""Materialize the held-latest centered two-sided coordinate oracle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from ember.ecp.realizer_evidence import resolve_asset
from ember.ecp.realizer_materialization import (
    build_static_lora_projection_manifest,
    merge_carrier_residual,
)
from ember.ecp.two_sided_coordinate import (
    TWO_SIDED_COORDINATE_SCHEMA,
    decode_two_sided_code,
    reconstruct_rank4_factors,
)
from ember.lora import LoRAContract
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic


TWO_SIDED_ORACLE_SCHEMA = "ember_ecp_centered_two_sided_oracle_v1"


@dataclass(frozen=True)
class OracleAuthority:
    repository: Mapping[str, Any]
    config_path: Path
    coordinate_manifest_path: Path
    config: Mapping[str, Any]
    coordinate_manifest: Mapping[str, Any]
    held_ids: tuple[int, ...]
    contract: LoRAContract
    transform: Mapping[str, torch.Tensor]
    latest: Mapping[int, Mapping[str, Any]]
    carrier_path: Path
    carrier: Mapping[str, torch.Tensor]
    base_projection_path: Path
    base_projection: Mapping[str, Any]


def _target_transform(
    tensors: Mapping[str, torch.Tensor], target_index: int
) -> dict[str, torch.Tensor]:
    prefix = f"target_{target_index:02d}_"
    return {
        name: tensors[prefix + name]
        for name in ("omega", "psi", "mean", "components", "scales", "active_mask")
    }


def _oracle_authority(args: Any) -> OracleAuthority:
    repository = git_state(Path(__file__).resolve().parents[3])
    if not git_state_is_clean_pushed_or_frozen_authority(repository):
        raise ValueError("formal two-sided materialization requires clean authority")
    config_path = args.config.resolve()
    coordinate_manifest_path = args.coordinate_manifest.resolve()
    asset_root = args.asset_root.resolve()
    config = read_json(config_path)
    coordinate_manifest = read_json(coordinate_manifest_path)
    held_ids = tuple(int(value) for value in config["fold"]["held_global_ids"])
    if (
        config.get("schema_version") != "ember_ecp_centered_two_sided_coordinate_v1"
        or coordinate_manifest.get("schema_version") != TWO_SIDED_COORDINATE_SCHEMA
        or coordinate_manifest.get("status")
        != "complete_fit_only_centered_two_sided_coordinate"
        or tuple(coordinate_manifest.get("held_global_task_ids", ())) != held_ids
        or coordinate_manifest.get("fit_tasks") != 90
        or coordinate_manifest.get("fit_members") != 108
        or coordinate_manifest.get("information_wall", {}).get(
            "held_coordinate_updates"
        )
        != 0
    ):
        raise ValueError("two-sided materialization authority changed")

    contract = load_pi05_lora_contract(
        resolve_asset(asset_root, config["authorities"]["lora_contract"])
    )
    transform = load_file(
        str(Path(coordinate_manifest["coordinate"]["transform_path"]).resolve())
    )
    latest = {
        int(row["global_task_id"]): dict(row)
        for row in coordinate_manifest["members"]
        if row["fold_role"] == "held_transform_only" and row["member"] == "latest"
    }
    if tuple(sorted(latest)) != tuple(sorted(held_ids)):
        raise ValueError("two-sided held-latest code panel changed")

    carrier_path = resolve_asset(
        asset_root, config["authorities"]["stable_carrier"]
    )
    carrier = load_file(str(carrier_path), device="cpu")
    base_projection_path = resolve_asset(
        asset_root, config["authorities"]["base_projection_manifest"]
    )
    base_projection = read_json(base_projection_path)
    return OracleAuthority(
        repository=repository,
        config_path=config_path,
        coordinate_manifest_path=coordinate_manifest_path,
        config=config,
        coordinate_manifest=coordinate_manifest,
        held_ids=held_ids,
        contract=contract,
        transform=transform,
        latest=latest,
        carrier_path=carrier_path,
        carrier=carrier,
        base_projection_path=base_projection_path,
        base_projection=base_projection,
    )


def _decode_task_residual(
    authority: OracleAuthority, global_id: int
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    row = authority.latest[global_id]
    coordinate = authority.config["coordinate"]
    code = load_file(str(Path(row["effect_code_path"]).resolve()))["effect_code"]
    if tuple(code.shape) != (38, int(coordinate["output_width"])):
        raise ValueError("two-sided held code shape changed")
    residual = []
    for target_index, target in enumerate(authority.contract.targets):
        tensors = _target_transform(authority.transform, target_index)
        decoded = decode_two_sided_code(
            code[target_index],
            mean=tensors["mean"],
            components=tensors["components"],
            scales=tensors["scales"],
            active_mask=tensors["active_mask"],
        )
        residual.append(
            reconstruct_rank4_factors(
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
        )
    return tuple(residual)


def _materialize_tasks(
    authority: OracleAuthority, adapter_root: Path
) -> list[dict[str, Any]]:
    base_rows = {
        int(row["global_task_id"]): dict(row)
        for row in authority.base_projection.get("tasks", ())
    }
    task_rows = []
    for global_id in authority.held_ids:
        row = authority.latest[global_id]
        base = base_rows.get(global_id)
        if base is None:
            raise ValueError("two-sided base projection is missing a held task")
        state = merge_carrier_residual(
            carrier=authority.carrier,
            residual=_decode_task_residual(authority, global_id),
            contract=authority.contract,
            carrier_rank=int(authority.config["oracle"]["carrier_rank"]),
        )
        adapter_path = adapter_root / f"task_global_{global_id:02d}.safetensors"
        save_file(state, str(adapter_path))
        task_rows.append(
            {
                "suite": str(base["suite"]),
                "task_id": int(base["task_id"]),
                "global_task_id": global_id,
                "code_member": "latest",
                "coordinate_geometry": row["coordinate_geometry"],
                "adapter_path": str(adapter_path),
                "adapter_bytes": adapter_path.stat().st_size,
            }
        )
    return task_rows


def _write_oracle_manifests(
    authority: OracleAuthority,
    *,
    output_dir: Path,
    task_rows: list[dict[str, Any]],
) -> Path:
    purpose = "stage1b_centered_two_sided_coordinate_latest"
    projection = build_static_lora_projection_manifest(
        repository=authority.repository,
        purpose=purpose,
        base_projection_path=authority.base_projection_path,
        base_projection=authority.base_projection,
        tasks=task_rows,
    )
    projection_path = output_dir / "projection_manifest.json"
    write_json_atomic(projection_path, projection)
    manifest = output_dir / "manifest.json"
    write_json_atomic(
        manifest,
        {
            "schema_version": TWO_SIDED_ORACLE_SCHEMA,
            "status": "complete",
            "repository": authority.repository,
            "config": {
                "path": str(authority.config_path),
                "bytes": authority.config_path.stat().st_size,
            },
            "coordinate_authority": {
                "path": str(authority.coordinate_manifest_path),
                "bytes": authority.coordinate_manifest_path.stat().st_size,
            },
            "carrier": {
                "path": str(authority.carrier_path),
                "bytes": authority.carrier_path.stat().st_size,
            },
            "projection_manifest": {
                "path": str(projection_path),
                "bytes": projection_path.stat().st_size,
            },
            "tasks": task_rows,
            "information_wall": {
                "held_target_residual_reads_during_materialization": 0,
                "held_optimizer_steps": 0,
                "validation_action_or_reward_reads": 0,
                "test_action_or_reward_reads": 0,
                "single_complete_lora": True,
                "second_adapter_deployed": False,
            },
        },
    )
    return manifest


def materialize_centered_two_sided_oracle(args: Any) -> Path:
    authority = _oracle_authority(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    adapter_root = output_dir / "adapters"
    adapter_root.mkdir()
    task_rows = _materialize_tasks(authority, adapter_root)
    return _write_oracle_manifests(
        authority, output_dir=output_dir, task_rows=task_rows
    )
