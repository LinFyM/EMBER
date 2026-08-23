#!/usr/bin/env python3
"""Train the held-free fixed ECP effect realizer."""

from __future__ import annotations

import argparse
from pathlib import Path

from ember.ecp.realizer_training import run_fixed_effect_realizer


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_fixed_effect_realizer.json",
    )
    parser.add_argument("--effect-code-manifest", type=Path, required=True)
    parser.add_argument(
        "--lora-contract",
        type=Path,
        default=REPO_ROOT / "configs/pi05_lora_v1.json",
    )
    parser.add_argument("--mode", choices=("profile", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    print(run_fixed_effect_realizer(parse_args()), flush=True)


if __name__ == "__main__":
    main()
