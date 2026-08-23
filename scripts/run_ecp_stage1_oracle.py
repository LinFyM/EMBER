#!/usr/bin/env python3
"""Run one formal ECP Stage 1B mobile-rank4 realization task."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ember.ecp.stage1_oracle import solve_stage1_task
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
        default=REPO_ROOT / "configs/pi05_ecp_stage1b_mobile_rank4_oracle_v1.json",
    )
    parser.add_argument("--task-ordinal", type=int, required=True)
    parser.add_argument("--effect-bank-manifest", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = git_state(REPO_ROOT)
    if not git_state_is_clean_pushed_or_frozen_authority(repository):
        raise ValueError("formal ECP Stage 1B oracle requires clean pushed authority")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    config = read_json(args.config.resolve())
    if config.get("schema_version") != "ember_ecp_stage1b_mobile_rank4_oracle_v1":
        raise ValueError("ECP Stage 1B config changed")
    result = solve_stage1_task(
        args=args,
        config=config,
        effect_bank_manifest=args.effect_bank_manifest.resolve(),
        asset_root=args.asset_root.resolve(),
        output_dir=args.output_dir.resolve(),
        device=device,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()
