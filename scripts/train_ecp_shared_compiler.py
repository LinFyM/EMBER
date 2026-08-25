#!/usr/bin/env python3
"""Train ECP G3 frozen-Program shared compiler."""

from ember.ecp.shared_compiler_training import build_parser, finalize_args, train


if __name__ == "__main__":
    train(finalize_args(build_parser().parse_args()))
