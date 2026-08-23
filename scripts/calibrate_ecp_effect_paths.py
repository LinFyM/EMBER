#!/usr/bin/env python3
"""Run or aggregate formal ECP known-success effect-path calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ember.ecp.effect_path_calibration import aggregate_results
from ember.ecp.effect_path_capture import run_task
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
        default=REPO_ROOT / "configs/pi05_ecp_effect_path_calibration.json",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task-ordinal", type=int)
    mode.add_argument("--aggregate-root", type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = git_state(REPO_ROOT)
    if not git_state_is_clean_pushed_or_frozen_authority(repository):
        raise ValueError("formal effect-path calibration requires clean authority")
    config = read_json(args.config.resolve())
    if config.get("schema_version") != "ember_ecp_effect_path_calibration_v1":
        raise ValueError("effect-path calibration config changed")
    if args.aggregate_root is not None:
        result = aggregate_results(
            config=config,
            root=args.aggregate_root.resolve(),
            output=args.output.resolve(),
        )
    else:
        if args.source_run is None or args.checkpoint is None:
            raise ValueError("effect-path task capture requires source authorities")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        result = run_task(
            args=args,
            config=config,
            asset_root=args.asset_root.resolve(),
            output=args.output.resolve(),
            device=device,
        )
    print(result, flush=True)


if __name__ == "__main__":
    main()
