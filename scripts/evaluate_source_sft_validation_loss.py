#!/usr/bin/env python3
"""Evaluate Source-SFT checkpoints on the sealed validation-loss panel."""

from ember.source_sft.validation import build_parser, evaluate, finalize_args


if __name__ == "__main__":
    evaluate(finalize_args(build_parser().parse_args()))
