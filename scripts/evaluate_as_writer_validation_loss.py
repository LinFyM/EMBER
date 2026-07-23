#!/usr/bin/env python3
"""Evaluate AS-Writer checkpoints on the sealed validation functional-loss panel."""

from ember.writer.validation import build_parser, evaluate, finalize_args


if __name__ == "__main__":
    evaluate(finalize_args(build_parser().parse_args()))
