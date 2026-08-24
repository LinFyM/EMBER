#!/usr/bin/env python3
"""Measure the output-space ceiling of G1's native Y banks.

For a frozen linear target, every signed combination of absolute or temporal
output activations lies in the column space of the target's base weight.  This
diagnostic projects the known-success mobile-rank4 residuals into that space.
It is intentionally read-only and does not optimize or materialize adapters.
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
from safetensors.torch import load_file

from ember.ecp.contracts import build_target_owners
from ember.ecp.native_materialization import (
    extract_rank12_carrier,
    extract_rank4_residual,
)
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX
from ember.pi05_eval_contract import git_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic


SCHEMA = "ember_ecp_native_factor_g1_output_span_v1"


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


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_pi05_lora_contract(args.lora_contract.resolve())
    owners = build_target_owners(contract)
    carrier = load_file(str(args.carrier.resolve()), device="cpu")
    extract_rank12_carrier(carrier, contract)
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
        "repository": git_state(Path(__file__).resolve().parents[1]),
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--lora-contract", type=Path, required=True)
    parser.add_argument("--carrier", type=Path, required=True)
    parser.add_argument("--mobile-projection", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, result)
    print(json.dumps({"output": str(args.output), "records": len(result["records"])}))


if __name__ == "__main__":
    main()
