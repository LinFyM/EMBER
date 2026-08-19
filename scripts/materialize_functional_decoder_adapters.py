#!/usr/bin/env python3
"""Materialize one complete train24 LoRA bank from a functional decoder profile."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file

from ember.expert_manifold.evaluation import (
    FUNCTIONAL_DECODER_TASK_EXPERT_MANIFEST_SCHEMA,
)
from ember.functional_adaptation.decoder_training import (
    FunctionalDecoderSystem,
    authority_path,
    decoder_task_split,
    expert_records,
    inspect_train24_expert_bank,
    load_functional_adapter_config,
)
from ember.lora import identity_lora_state, validate_lora_state
from ember.pi05_eval_contract import git_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCHEMA = "ember_pi05_functional_flow_profile_v1"


def _asset(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"missing functional-decoder asset: {resolved}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size}


def _loss_by_ordinal(
    profile: Mapping[str, Any], role: str
) -> dict[int, float]:
    ordinals = profile[f"active_{role}_ordinals"]
    losses = profile[f"final_{role}_per_task"]
    if len(ordinals) != len(losses):
        raise ValueError("functional profile task losses changed")
    return {int(ordinal): float(loss) for ordinal, loss in zip(ordinals, losses)}


def _materialize_rows(
    *,
    output_dir: Path,
    records: Any,
    split: Any,
    bank: Mapping[str, Any],
    system: FunctionalDecoderSystem,
    held_codes: torch.Tensor,
    contract: Any,
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    fit_codes = {
        row.ordinal: system.codebook.weight[index]
        for index, row in enumerate(split.fit)
    }
    held_code_map = {
        row.ordinal: held_codes[index] for index, row in enumerate(split.held)
    }
    losses = {
        **_loss_by_ordinal(profile, "fit"),
        **_loss_by_ordinal(profile, "held"),
    }
    bank_by_ordinal = {int(row["ordinal"]): row for row in bank["tasks"]}
    rows = []
    with torch.no_grad():
        for record in records:
            held = record.ordinal in held_code_map
            code = held_code_map[record.ordinal] if held else fit_codes[record.ordinal]
            candidate = system.decoder(code)
            reference = load_file(
                str(record.checkpoint / "adapter.safetensors"), device="cpu"
            )
            stored = {
                name: value.detach().cpu().to(reference[name].dtype).contiguous()
                for name, value in candidate.items()
            }
            validate_lora_state(stored, contract)
            path = (
                output_dir
                / f"task_{record.ordinal:02d}_global_{record.global_task_id:02d}.safetensors"
            )
            save_file(stored, str(path))
            bank_row = bank_by_ordinal[record.ordinal]
            rows.append(
                {
                    "suite": bank_row["suite"],
                    "task_id": int(bank_row["task_id"]),
                    "ordinal": record.ordinal,
                    "global_task_id": record.global_task_id,
                    "expert_checkpoint": str(record.checkpoint),
                    "projected_adapter": str(path.resolve()),
                    "projected_adapter_bytes": path.stat().st_size,
                    "code_role": "held_free_code" if held else "fit_codebook",
                    "functional_flow_eval_relative_loss": losses[record.ordinal],
                }
            )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_functional_adapter_config(args.config, REPO_ROOT)
    mechanism = config["train24_mechanism"]
    profile_root = args.profile_root.resolve()
    decoder_path = profile_root / "decoder.safetensors"
    held_codes_path = profile_root / "held_codes.safetensors"
    profile_result_path = profile_root / "result.json"
    profile = read_json(profile_result_path)
    if (
        profile.get("schema_version") != PROFILE_SCHEMA
        or profile.get("mode") != "informative"
        or profile.get("repository", {}).get("dirty_paths") != []
    ):
        raise ValueError("functional flow profile is not clean informative evidence")
    repository = git_state(REPO_ROOT)
    if repository["dirty_paths"]:
        raise ValueError("functional adapter materialization requires a clean worktree")

    bank = inspect_train24_expert_bank(
        config,
        REPO_ROOT,
        source_run=args.source_run,
        checkpoint=args.checkpoint,
        bank_root=args.expert_bank_root,
    )
    records = expert_records(bank)
    split = decoder_task_split(
        records,
        fold_count=int(mechanism["fold_count"]),
        held_out_fold=int(mechanism["held_out_fold"]),
    )
    fit_ordinals = tuple(row.ordinal for row in split.fit)
    held_ordinals = tuple(row.ordinal for row in split.held)
    if (
        tuple(int(value) for value in profile["active_fit_ordinals"])
        != fit_ordinals
        or tuple(int(value) for value in profile["active_held_ordinals"])
        != held_ordinals
    ):
        raise ValueError("functional profile no longer covers the declared train24 fold")

    contract = load_pi05_lora_contract(
        authority_path(config, "lora_contract", REPO_ROOT)
    )
    decoder_config = config["decoder"]
    system = FunctionalDecoderSystem(
        contract,
        identity_lora_state(contract),
        task_count=len(split.fit),
        code_width=int(decoder_config["train24_smoke_code_width"]),
        address_width=int(decoder_config["address_width"]),
        hidden_width=int(decoder_config["hidden_width"]),
        seed=int(decoder_config["initialization_seed"]),
    )
    system.load_state_dict(load_file(str(decoder_path), device="cpu"), strict=True)
    system.eval().requires_grad_(False)
    held_codes = load_file(str(held_codes_path), device="cpu")["held_codes"]
    if tuple(held_codes.shape) != (len(split.held), system.decoder.code_width):
        raise ValueError("held functional codes changed shape")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = _materialize_rows(
        output_dir=args.output_dir,
        records=records,
        split=split,
        bank=bank,
        system=system,
        held_codes=held_codes,
        contract=contract,
        profile=profile,
    )

    result = {
        "schema_version": FUNCTIONAL_DECODER_TASK_EXPERT_MANIFEST_SCHEMA,
        "projection_kind": "fixed_functional_decoder_code_projection",
        "repository": {
            "commit": repository["commit"],
            "dirty_paths": repository["dirty_paths"],
        },
        "functional_config": _asset(args.config),
        "decoder_checkpoint": _asset(decoder_path),
        "held_codes": _asset(held_codes_path),
        "profile_result": _asset(profile_result_path),
        "expert_bank_root": str(args.expert_bank_root.resolve()),
        "expert_step": int(mechanism["expert_step"]),
        "optimization": {
            "fit_task_count": len(split.fit),
            "held_task_count": len(split.held),
            "decoder_frozen_for_held_code_fit": True,
            "decoder_steps": int(profile["steps"]["decoder"]),
            "held_code_steps": int(profile["steps"]["held_code"]),
            "fold_count": int(mechanism["fold_count"]),
            "held_out_fold": int(mechanism["held_out_fold"]),
        },
        "information_wall": {
            "role": "development_train_oracle_only",
            "validation_experts": 0,
            "test_experts": 0,
            "deployment_carrier": False,
        },
        "tasks": rows,
        "content_hash_policy": "disabled_by_owner",
    }
    write_json_atomic(args.output_dir / "projection_manifest.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_functional_adapter_v1.json",
    )
    result.add_argument("--source-run", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--expert-bank-root", type=Path, required=True)
    result.add_argument("--profile-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
