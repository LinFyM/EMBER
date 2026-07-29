#!/usr/bin/env python3
"""Derive one inference-only uniform average of raw AS-Writer checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ember.writer.derived_checkpoint import (
    derive_uniform_writer_average_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        action="append",
        dest="source_checkpoints",
        required=True,
        help="Repeat once per raw checkpoints/step_* source.",
    )
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint, manifest = derive_uniform_writer_average_checkpoint(
        source_run=args.source_run,
        source_checkpoints=args.source_checkpoints,
        output_name=args.name,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "schema_version": manifest["schema_version"],
                "manifest_payload_sha256": manifest[
                    "canonical_payload_sha256"
                ],
                "writer_state_sha256": manifest["files"][
                    "writer.safetensors"
                ]["sha256"],
                "source_cursors": [
                    int(row["cursor"])
                    for row in manifest["derivation"]["source_checkpoints"]
                ],
                "parameter_count": int(
                    manifest["derivation"]["parameter_count"]
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
