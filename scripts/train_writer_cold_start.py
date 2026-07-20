#!/usr/bin/env python3
"""Train the canonical shared EMBER Writer with functional source action loss."""

from ember.writer.training import build_parser, finalize_args, train


if __name__ == "__main__":
    train(finalize_args(build_parser().parse_args()))
