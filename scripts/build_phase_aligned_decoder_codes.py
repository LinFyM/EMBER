#!/usr/bin/env python3
"""Build the fixed task-consensus codes for the phase-aligned decoder."""

import argparse
from pathlib import Path

from ember.functional_adaptation.phase_code_building import run


REPO_ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    run(args.config, args.output_dir, repo_root=REPO_ROOT)
