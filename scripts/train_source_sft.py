#!/usr/bin/env python3
"""Train the canonical shared PI05 Source-SFT LoRA."""

from ember.source_sft.training import build_parser, finalize_args, train


if __name__ == "__main__":
    train(finalize_args(build_parser().parse_args()))
