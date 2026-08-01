#!/usr/bin/env python3
"""Apply the pre-registered endpoint10 association gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ember.writer.endpoint_association import write_endpoint_association


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_endpoint_association(
        args.endpoint_root, args.preregistration, args.output
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
