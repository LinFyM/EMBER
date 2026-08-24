#!/usr/bin/env python3
"""Measure the output-space ceiling of G1's native Y banks.

For a frozen linear target, every signed combination of absolute or temporal
output activations lies in the column space of the target's base weight.  This
diagnostic projects the known-success mobile-rank4 residuals into that space.
It never optimizes parameters; optionally it seals one projected member bank
for a paired closed-loop response diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ember.ecp.contracts import build_target_owners
from ember.ecp.g1_runtime import G1_CHECKPOINT_SCHEMA
from ember.ecp.native_materialization import (
    compose_rank12_plus_rank4,
    extract_rank12_carrier,
    extract_rank4_residual,
    small_core_balanced_svd,
)
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.pi05_eval_contract import git_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic


SCHEMA = "ember_ecp_native_factor_g1_output_span_v1"
STATIC_BANK_SCHEMA = "ember_pi05_static_task_lora_bank_v1"


def _energy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((b.transpose(0, 1) @ b) * (a @ a.transpose(0, 1))).sum()


def _column_basis(weight: torch.Tensor) -> tuple[torch.Tensor, int, float]:
    out_features, in_features = weight.shape
    if out_features <= in_features:
        _q, r = torch.linalg.qr(weight.transpose(0, 1), mode="reduced")
        diagonal = r.diagonal().abs()
        threshold = (
            torch.finfo(weight.dtype).eps
            * max(out_features, in_features)
            * diagonal.max()
        )
        rank = int((diagonal > threshold).sum())
        if rank != out_features:
            raise ValueError("wide native target base weight lost full row rank")
        basis = torch.eye(out_features, device=weight.device, dtype=weight.dtype)
    else:
        basis, r = torch.linalg.qr(weight, mode="reduced")
        diagonal = r.diagonal().abs()
        threshold = (
            torch.finfo(weight.dtype).eps
            * max(out_features, in_features)
            * diagonal.max()
        )
        rank = int((diagonal > threshold).sum())
        if rank != in_features:
            raise ValueError("tall native target base weight lost full column rank")
    condition_proxy = float((diagonal.max() / diagonal.min()).detach().cpu())
    return basis, rank, condition_proxy


def _record_key(record: Mapping[str, Any]) -> tuple[int, str]:
    return int(record["task"]["ordinal"]), str(record["member"])


def _materialize_projection_bank(
    *,
    args: argparse.Namespace,
    contract: Any,
    carrier_rank12: Mapping[str, torch.Tensor],
    records: tuple[Mapping[str, Any], ...],
    projected_states: Mapping[tuple[int, str], Mapping[str, torch.Tensor]],
    repository: Mapping[str, Any],
) -> dict[str, Any] | None:
    if args.materialize_member is None:
        return None
    if args.materialized_root is None:
        raise ValueError("--materialized-root is required with --materialize-member")
    selected = sorted(
        (row for row in records if row["member"] == args.materialize_member),
        key=lambda row: int(row["task"]["ordinal"]),
    )
    if len(selected) != 5:
        raise ValueError("materialized span projection requires the fixed held5 panel")
    root = args.materialized_root.resolve()
    if root.exists():
        raise ValueError(f"materialized output already exists: {root}")
    rows = []
    for record in selected:
        task = record["task"]
        raw = projected_states[_record_key(record)]
        residual = {}
        for target in contract.targets:
            a_name = target.name + LORA_A_SUFFIX
            b_name = target.name + LORA_B_SUFFIX
            a, b = small_core_balanced_svd(raw[a_name], raw[b_name])
            residual[a_name] = a.cpu()
            residual[b_name] = b.cpu()
        complete = compose_rank12_plus_rank4(
            carrier_state=carrier_rank12,
            residual_state=residual,
            rank16_contract=contract,
        )
        validate_lora_state(complete, contract)
        ordinal = int(task["ordinal"])
        task_root = root / f"task_{ordinal}"
        checkpoint = task_root / "checkpoints" / "step_00000000"
        checkpoint.mkdir(parents=True)
        adapter_path = checkpoint / "adapter.safetensors"
        save_file(
            {
                name: value.to(dtype=torch.bfloat16).contiguous()
                for name, value in complete.items()
            },
            str(adapter_path),
        )
        checkpoint_manifest = checkpoint / "manifest.json"
        write_json_atomic(
            checkpoint_manifest,
            {
                "schema_version": G1_CHECKPOINT_SCHEMA,
                "step": 0,
                "task_ordinal": ordinal,
                "global_task_id": int(task["global_task_id"]),
                "rank_partition": {"carrier": [0, 12], "task": [12, 16]},
                "single_complete_rank16": True,
                "state_tensor_count": len(complete),
                "files": {"adapter.safetensors": adapter_path.stat().st_size},
                "diagnostic": "native_y_base_output_space_projection",
                "reference_member": args.materialize_member,
                "content_hash_policy": "disabled_by_owner",
            },
        )
        rows.append(
            {
                **task,
                "step": 0,
                "run_root": str(task_root),
                "checkpoint": str(checkpoint),
                "checkpoint_manifest_bytes": checkpoint_manifest.stat().st_size,
                "adapter_path": str(adapter_path),
                "adapter_bytes": adapter_path.stat().st_size,
                "single_complete_rank16": True,
            }
        )
    source_model_path = args.source_model.resolve().parent
    source_checkpoint = source_model_path.parent
    bank_path = root / "evaluation_bank.json"
    write_json_atomic(
        bank_path,
        {
            "schema_version": STATIC_BANK_SCHEMA,
            "status": "sealed",
            "arm": "ecp_native_factor_g1_output_span_projection",
            "source": {
                "source_run": str(source_checkpoint.parent.parent),
                "checkpoint": str(source_checkpoint),
                "model_path": str(source_model_path),
            },
            "lora_contract": {
                "path": str(args.lora_contract.resolve()),
                "bytes": args.lora_contract.resolve().stat().st_size,
            },
            "rank_partition": {"carrier": [0, 12], "task": [12, 16]},
            "single_complete_rank16": True,
            "training_commit": repository["commit"],
            "tasks": rows,
            "diagnostic": {
                "scientific_role": "read_only_g1_response_projection",
                "reference_member": args.materialize_member,
                "projection": (
                    "left_orthogonal_projection_into_base_weight_column_space"
                ),
                "claim_boundary": (
                    "tests one known-success rank4 response after imposing the "
                    "current native-Y output-space ceiling; it is not a trained "
                    "compiler"
                ),
            },
            "information_wall": {
                "task_local_free_code_capacity_oracle": True,
                "shared_program_attention_claim": False,
                "action_meta_installed": False,
                "teacher_video_runtime_reads": 0,
                "second_adapter_deployed": False,
                "validation_action_or_reward_reads": 0,
                "test_action_or_reward_reads": 0,
            },
            "content_hash_policy": "disabled_by_owner",
        },
    )
    return {
        "root": str(root),
        "evaluation_bank": str(bank_path),
        "reference_member": args.materialize_member,
        "task_count": len(rows),
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_pi05_lora_contract(args.lora_contract.resolve())
    owners = build_target_owners(contract)
    carrier = load_file(str(args.carrier.resolve()), device="cpu")
    carrier_rank12 = extract_rank12_carrier(carrier, contract)
    projection = read_json(args.mobile_projection.resolve())
    records = tuple(projection.get("records", ()))
    if len(records) != 15:
        raise ValueError("G1 mobile-rank4 authority is not the fixed 5x3 panel")

    residuals = {}
    for record in records:
        complete = load_file(str(Path(record["projected_adapter"])), device="cpu")
        residuals[_record_key(record)] = extract_rank4_residual(
            complete,
            contract,
            carrier_state=carrier,
        )

    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("output-span analysis requires an explicit CUDA device")
    torch.cuda.set_device(device)
    target_rows: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    projected_states: dict[tuple[int, str], dict[str, torch.Tensor]] = {
        key: {} for key in residuals
    }
    weight_rows = []
    with safe_open(
        str(args.source_model.resolve()), framework="pt", device="cpu"
    ) as source:
        for target, owner in zip(contract.targets, owners, strict=True):
            weight_name = target.name + ".weight"
            weight = source.get_tensor(weight_name).to(
                device=device, dtype=torch.float32
            )
            if weight.shape != (target.out_features, target.in_features):
                raise ValueError(f"native base weight changed shape: {target.name}")
            basis, rank, condition_proxy = _column_basis(weight)
            weight_rows.append(
                {
                    "target": target.name,
                    "family": owner.family.value,
                    "out_features": target.out_features,
                    "in_features": target.in_features,
                    "base_output_rank": rank,
                    "base_output_rank_fraction": rank / target.out_features,
                    "qr_diagonal_condition_proxy": condition_proxy,
                }
            )
            for record in records:
                key = _record_key(record)
                state = residuals[key]
                a = state[target.name + LORA_A_SUFFIX].to(
                    device=device, dtype=torch.float32
                )
                b = state[target.name + LORA_B_SUFFIX].to(
                    device=device, dtype=torch.float32
                )
                projected_b = basis @ (basis.transpose(0, 1) @ b)
                projected_states[key][target.name + LORA_A_SUFFIX] = (
                    a.detach().cpu()
                )
                projected_states[key][target.name + LORA_B_SUFFIX] = (
                    projected_b.detach().cpu()
                )
                energy = _energy(a, b).clamp_min(0)
                projected_energy = torch.minimum(
                    _energy(a, projected_b).clamp_min(0), energy
                )
                capture = float(
                    (projected_energy / energy.clamp_min(1e-20)).detach().cpu()
                )
                target_rows[key].append(
                    {
                        "target": target.name,
                        "family": owner.family.value,
                        "energy": float(energy.detach().cpu()),
                        "projected_energy": float(projected_energy.detach().cpu()),
                        "output_energy_capture": capture,
                        "excluded_relative_error": math.sqrt(max(0.0, 1.0 - capture)),
                    }
                )
            del weight, basis

    summaries = []
    record_by_key = {_record_key(record): record for record in records}
    for key, rows in sorted(target_rows.items()):
        task_ordinal, member = key
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_family[row["family"]].append(row)

        def summarize(selected: list[dict[str, Any]]) -> dict[str, float]:
            energy = sum(row["energy"] for row in selected)
            projected = sum(row["projected_energy"] for row in selected)
            capture = projected / max(energy, 1e-20)
            return {
                "energy": energy,
                "projected_energy": projected,
                "output_energy_capture": capture,
                "excluded_relative_error": math.sqrt(max(0.0, 1.0 - capture)),
            }

        authority = record_by_key[key]
        summaries.append(
            {
                "task": authority["task"],
                "member": member,
                "member_reliability": authority["member_reliability"],
                "all": summarize(rows),
                "by_family": {
                    family: summarize(selected)
                    for family, selected in sorted(by_family.items())
                },
                "targets": rows,
            }
        )
    repository = git_state(Path(__file__).resolve().parents[1])
    materialized = _materialize_projection_bank(
        args=args,
        contract=contract,
        carrier_rank12=carrier_rank12,
        records=records,
        projected_states=projected_states,
        repository=repository,
    )
    return {
        "schema_version": SCHEMA,
        "scientific_role": "read_only_g1_earliest_interface_diagnostic",
        "question": (
            "how much known-success mobile-rank4 update energy can any signed "
            "pooling of the current native Y banks represent before video-span, "
            "softmax, scale, or optimization constraints"
        ),
        "derivation": (
            "signed pooling has zero total coefficient mass, so biases cancel; "
            "Y=W X and abs/adj/init/goal differences all remain in column_space(W)"
        ),
        "repository": repository,
        "authorities": {
            "source_model": str(args.source_model.resolve()),
            "lora_contract": str(args.lora_contract.resolve()),
            "carrier": str(args.carrier.resolve()),
            "mobile_projection": str(args.mobile_projection.resolve()),
        },
        "device": str(device),
        "content_hash_policy": "disabled_by_owner",
        "target_base_spaces": weight_rows,
        "records": summaries,
        "materialized_projection_bank": materialized,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--lora-contract", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--mobile-projection", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--materialize-member", choices=("latest", "independent", "earliest")
    )
    parser.add_argument("--materialized-root", type=Path)
    args = parser.parse_args()
    result = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, result)
    print(json.dumps({"output": str(args.output), "records": len(result["records"])}))


if __name__ == "__main__":
    main()
