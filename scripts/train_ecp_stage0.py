#!/usr/bin/env python3
"""Train EMBER-ECP Stage 0 native observer."""

from ember.ecp.stage0_training import build_parser, finalize_args, train


if __name__ == "__main__":
    train(finalize_args(build_parser().parse_args()))
