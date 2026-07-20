#!/usr/bin/env python3
"""Canonical eight-rank source-only reward training of the shared Writer."""

from ember.writer_rl import build_parser, finalize_args, train


if __name__ == "__main__":
    train(finalize_args(build_parser().parse_args()))
