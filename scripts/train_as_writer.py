#!/usr/bin/env python3
"""Train the canonical dynamic-K Backbone-Memory AS-Writer."""

from ember.writer.training import build_parser, finalize_args, train


if __name__ == "__main__":
    train(finalize_args(build_parser().parse_args()))
