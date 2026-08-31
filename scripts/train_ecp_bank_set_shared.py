#!/usr/bin/env python3
"""Train one shared fixed-route EBSRI S2 interaction."""

from __future__ import annotations

import argparse
from pathlib import Path

from ember.ecp.joint_program_primal.bank_set_shared_training import run
from ember.ecp.joint_program_primal.runtime import REPO_ROOT


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--base-config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_shared_compiler_g3_v5.json",
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition-cache-root", type=Path, required=True)
    parser.add_argument("--program-bank-condition-cache-root", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.log_every <= 0:
        raise ValueError("S2 log interval must be positive")
    args.phase = "shared_loto"
    return args


if __name__ == "__main__":
    run(_arguments())
