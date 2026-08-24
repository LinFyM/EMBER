#!/usr/bin/env python3
"""Seal target-40 teacher HDF5 metadata without reading held values."""

from __future__ import annotations

import argparse
from pathlib import Path

from ember.pi05_target_data import seal_target_data


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "configs" / "pi05_target_data_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPO_ROOT / "configs/libero_24_8_8_v1/protocol.json",
    )
    parser.add_argument(
        "--overlap-audit",
        type=Path,
        default=REPO_ROOT / "configs/pi05_source_corpus_v1/overlap_audit.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR / "manifest.json")
    parser.add_argument("--sealed-utc", default="2026-07-21")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seal_target_data(
        protocol_path=args.protocol.resolve(),
        overlap_audit_path=args.overlap_audit.resolve(),
        data_root=args.data_root.resolve(),
        output_path=args.output.resolve(),
        sealed_utc=args.sealed_utc,
    )


if __name__ == "__main__":
    main()
