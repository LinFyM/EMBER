#!/usr/bin/env python3
"""Project Stage 1B members into an additive carrier12 + mobile residual4."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from safetensors.torch import load_file, save_file

from ember.ecp.stage1_realization import (
    RankReservedProjectionTarget,
    project_expert_onto_rank_reserved_residual,
)
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "ember_ecp_stage1b_rank4_residual_member_projection_v1"
MEMBERS = ("latest", "independent", "earliest")
RESIDUAL_RANK = 4


def _authority(config: Mapping[str, Any], name: str, asset_root: Path) -> Path:
    path = Path(str(config["authorities"][name]))
    return path.resolve() if path.is_absolute() else (asset_root / path).resolve()


def _adapter_file(value: str) -> Path:
    path = Path(value).resolve()
    return path / "adapter.safetensors" if path.is_dir() else path


def _family(name: str) -> str:
    if name.endswith(".self_attn.q_proj"):
        return "q"
    if name.endswith(".self_attn.v_proj"):
        return "v"
    if name.endswith(".action_in_proj"):
        return "action_in"
    if name.endswith(".action_out_proj"):
        return "action_out"
    raise ValueError(f"unknown ECP LoRA target family: {name}")


def _aggregate(rows: Iterable[RankReservedProjectionTarget]) -> dict[str, float]:
    values = tuple(rows)
    expert = math.fsum(row.expert_energy for row in values)
    correction = math.fsum(row.required_correction_energy for row in values)
    residual = math.fsum(row.residual_energy for row in values)
    return {
        "expert_energy_coverage": 1.0 - residual / max(expert, 1e-24),
        "correction_energy_coverage": 1.0 - residual / max(correction, 1e-24),
        "best_relative_error_to_expert": math.sqrt(residual / max(expert, 1e-24)),
        "best_relative_error_to_required_correction": math.sqrt(
            residual / max(correction, 1e-24)
        ),
        "carrier_relative_distance_to_expert": math.sqrt(
            correction / max(expert, 1e-24)
        ),
        "projected_correction_energy": math.fsum(
            row.projected_correction_energy for row in values
        ),
        "projected_effective_update_energy": math.fsum(
            row.projected_effective_update_energy for row in values
        ),
    }


def _metric_groups(
    metrics: tuple[RankReservedProjectionTarget, ...],
) -> dict[str, dict[str, float]]:
    result = {"all": _aggregate(metrics)}
    for family in ("q", "v", "action_in", "action_out"):
        result[family] = _aggregate(
            row for row in metrics if _family(row.target) == family
        )
    return result


def _task_manifests(root: Path) -> tuple[Path, ...]:
    manifests = tuple(sorted(root.resolve().glob("task_*/manifest.json")))
    if len(manifests) != 5:
        raise ValueError("rank4 residual diagnostic requires the registered held5 banks")
    return manifests


def _project_member(
    *,
    task: Mapping[str, Any],
    member_name: str,
    member: Mapping[str, Any],
    carrier: Mapping[str, Any],
    contract: Any,
    output_dir: Path,
) -> dict[str, Any]:
    expert_path = _adapter_file(str(member["adapter"]))
    expert = load_file(str(expert_path), device="cpu")
    projected, metrics = project_expert_onto_rank_reserved_residual(
        carrier=carrier,
        expert=expert,
        contract=contract,
        carrier_rank=int(contract.rank) - RESIDUAL_RANK,
    )
    adapter_path = (
        output_dir
        / member_name
        / (
            f"task_{int(task['ordinal']):03d}_global_"
            f"{int(task['global_task_id']):02d}.safetensors"
        )
    )
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in projected.items()
        },
        str(adapter_path),
    )
    return {
        "task": dict(task),
        "member": member_name,
        "member_reliability": float(member["reliability"]),
        "expert_adapter": str(expert_path),
        "projected_adapter": str(adapter_path),
        "projected_adapter_bytes": adapter_path.stat().st_size,
        "metrics": _metric_groups(metrics),
        "targets": [
            {**asdict(row), "family": _family(row.target)} for row in metrics
        ],
    }


def _result(
    *,
    repository: Mapping[str, Any],
    args: argparse.Namespace,
    carrier_path: Path,
    contract_path: Path,
    contract: Any,
    bank_files: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "mode": "formal",
        "scientific_role": "held5_mobile_rank4_residual_capacity_diagnostic",
        "repository": {
            "commit": repository["commit"],
            "dirty_paths": repository["dirty_paths"],
        },
        "config": str(args.config.resolve()),
        "carrier": {"path": str(carrier_path), "bytes": carrier_path.stat().st_size},
        "lora_contract": {
            "path": str(contract_path),
            "bytes": contract_path.stat().st_size,
            "rank": contract.rank,
            "target_count": len(contract.targets),
        },
        "effect_bank_manifests": bank_files,
        "projection": {
            "equation": (
                "carrier_rank12 + argmin_rank(X)<=4 "
                "||(W_expert - W_carrier) - X||_F"
            ),
            "optimizer_steps": 0,
            "interpolation_or_checkpoint_selection": False,
            "single_complete_lora": True,
            "effective_update_addition": True,
            "carrier_rank": int(contract.rank) - RESIDUAL_RANK,
            "mobile_residual_rank": RESIDUAL_RANK,
            "raw_factor_cross_terms": False,
        },
        "records": records,
        "information_wall": {
            "role": "development_train_privileged_diagnostic_only",
            "held_shared_gradient_steps": 0,
            "validation_action_or_reward_reads": 0,
            "test_action_or_reward_reads": 0,
            "second_adapter_deployed": False,
        },
        "content_hash_policy": "disabled_by_owner",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repository = git_state(REPO_ROOT)
    if not git_state_is_clean_pushed_or_frozen_authority(repository):
        raise ValueError("formal rank4 residual projection requires clean pushed authority")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ValueError("rank4 residual projection output already exists")
    config = read_json(args.config.resolve())
    asset_root = args.asset_root.resolve()
    carrier_path = _authority(config, "stable_carrier", asset_root)
    contract_path = _authority(config, "lora_contract", asset_root)
    carrier = load_file(str(carrier_path), device="cpu")
    contract = load_pi05_lora_contract(contract_path)

    records = []
    bank_files = []
    for manifest_path in _task_manifests(args.effect_bank_root):
        bank_files.append(
            {"path": str(manifest_path), "bytes": manifest_path.stat().st_size}
        )
        metadata = read_json(manifest_path)["metadata"]
        task = dict(metadata["task"])
        members = {str(row["name"]): dict(row) for row in metadata["members"]}
        if set(members) != set(MEMBERS):
            raise ValueError("rank4 residual diagnostic member panel changed")
        for member_name in MEMBERS:
            records.append(
                _project_member(
                    task=task,
                    member_name=member_name,
                    member=members[member_name],
                    carrier=carrier,
                    contract=contract,
                    output_dir=output_dir,
                )
            )
    result = _result(
        repository=repository,
        args=args,
        carrier_path=carrier_path,
        contract_path=contract_path,
        contract=contract,
        bank_files=bank_files,
        records=records,
    )
    write_json_atomic(output_dir / "projection_analysis.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_stage1b_occupancy_oracle_v1.json",
    )
    parser.add_argument("--effect-bank-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(result["records"][0]["projected_adapter"], flush=True)


if __name__ == "__main__":
    main()
