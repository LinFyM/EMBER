#!/usr/bin/env python3
"""Train ECP G2 Natural Program."""

from ember.ecp.natural_program_training import build_parser, finalize_args, train


if __name__ == "__main__":
    train(finalize_args(build_parser().parse_args()))
