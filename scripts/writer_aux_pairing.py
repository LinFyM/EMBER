#!/usr/bin/env python3
"""Run the one CPU event preflight for the matched auxiliary-pairing candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.auxiliary_pairing import REFERENCE_CONFIG, run_preflight


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/libero_24_8_8_coverage_v1/writer_aux_cross_episode.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--reference-exposures", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from ember.writer.training import _config

    result = run_preflight(
        asset_root=args.asset_root,
        reference_exposures=args.reference_exposures,
        candidate=_config(args.config),
        reference=read_json(ROOT / REFERENCE_CONFIG),
        updates=args.updates,
    )
    result["config"] = str(args.config.resolve())
    write_json_atomic(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
