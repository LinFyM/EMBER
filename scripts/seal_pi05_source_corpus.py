#!/usr/bin/env python3
"""Seal the filtered LIBERO-90 corpus used by the pi0.5 source base."""

from __future__ import annotations

import argparse
from pathlib import Path

from ember.pi05_source_corpus import (
    seal_overlap_audit,
    seal_source_data,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "pi05_source_corpus_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    overlap = subparsers.add_parser("overlap")
    overlap.add_argument("--output", type=Path, default=DEFAULT_CONFIG_DIR / "overlap_audit.json")
    overlap.add_argument("--sealed-utc", default="2026-07-21")
    data = subparsers.add_parser("data")
    data.add_argument("--overlap", type=Path, default=DEFAULT_CONFIG_DIR / "overlap_audit.json")
    data.add_argument(
        "--prior-manifest",
        type=Path,
        default=REPO_ROOT / "configs/libero90_70_10_10/data_manifest.json",
    )
    data.add_argument("--data-root", type=Path, required=True)
    data.add_argument("--manifest", type=Path, default=DEFAULT_CONFIG_DIR / "source_manifest.json")
    data.add_argument(
        "--normalization", type=Path, default=DEFAULT_CONFIG_DIR / "source_normalization.json"
    )
    data.add_argument("--sealed-utc", default="2026-07-21")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "overlap":
        seal_overlap_audit(args.output, sealed_utc=args.sealed_utc)
    elif args.command == "data":
        seal_source_data(
            overlap_path=args.overlap,
            prior_manifest_path=args.prior_manifest,
            data_root=args.data_root,
            manifest_path=args.manifest,
            normalization_path=args.normalization,
            sealed_utc=args.sealed_utc,
        )


if __name__ == "__main__":
    main()
