#!/usr/bin/env python3
"""Materialize one formal fixed-effect realizer checkpoint on fold0 held codes."""

from __future__ import annotations

import argparse
from pathlib import Path

from ember.ecp.realizer_materialization import materialize_fixed_effect_realizer


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_fixed_effect_realizer.json",
    )
    parser.add_argument("--effect-code-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-projection-manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    print(materialize_fixed_effect_realizer(parse_args()), flush=True)


if __name__ == "__main__":
    main()
