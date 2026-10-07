#!/usr/bin/env python
"""Thin entrypoint for the sole registered finite teacher batch."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--task', type=int, choices=(12, 29, 32, 38), required=True)
    parser.add_argument('--kind', choices=('formal', 'smoke', 'profile'), required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--stop-after', type=int, default=480)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--microbatch', type=int, default=28)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.microbatch <= 112 or '/' in args.attempt or not args.attempt:
        parser.error('invalid physical microbatch or task-owned attempt')
    from ember.aligned_teacher_recovery.training import run
    run(args)


if __name__ == '__main__':
    main()
