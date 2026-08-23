#!/usr/bin/env python3
"""Build one formal occupancy-complete ECP Stage 1 policy-effect bank."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ember.ecp.stage1_bank_building import BankBuildInputs, build_effect_bank
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_source_checkpoint import read_json


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/pi05_ecp_stage1b_effective_update_oracle_v1.json",
    )
    parser.add_argument("--task-ordinal", type=int, required=True)
    parser.add_argument("--source-support-run", type=Path, required=True)
    parser.add_argument("--independent-fixed50-run", type=Path, required=True)
    parser.add_argument("--independent-occupancy-run", type=Path, required=True)
    parser.add_argument("--candidate-occupancy-run", type=Path, required=True)
    parser.add_argument("--independent-adapter-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = git_state(REPO_ROOT)
    if not git_state_is_clean_pushed_or_frozen_authority(repository):
        raise ValueError("formal ECP effect bank requires clean pushed authority")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    config = read_json(args.config.resolve())
    if (
        config.get("schema_version")
        != "ember_ecp_stage1b_effective_update_oracle_v1"
        or config.get("status")
        != "preregistered_effective_update_privileged_realization"
    ):
        raise ValueError("ECP Stage 1B effect-bank config changed")
    manifest = build_effect_bank(
        BankBuildInputs(
            config=config,
            task_ordinal=args.task_ordinal,
            source_support_run=args.source_support_run.resolve(),
            independent_fixed50_run=args.independent_fixed50_run.resolve(),
            independent_occupancy_run=args.independent_occupancy_run.resolve(),
            candidate_occupancy_run=args.candidate_occupancy_run.resolve(),
            independent_adapter_root=args.independent_adapter_root.resolve(),
            source_run=args.source_run.resolve(),
            checkpoint=args.checkpoint.resolve(),
            data_root=args.data_root.resolve(),
            output_dir=args.output_dir.resolve(),
            device=device,
        ),
        args.asset_root.resolve(),
    )
    print(manifest, flush=True)


if __name__ == "__main__":
    main()
