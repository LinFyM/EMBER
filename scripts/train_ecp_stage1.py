#!/usr/bin/env python3
"""Train the privileged EMBER-ECP Stage 1 teacher and compiler."""

from ember.ecp.stage1_training import build_parser, finalize_args, train


if __name__ == "__main__":
    train(finalize_args(build_parser().parse_args()))
