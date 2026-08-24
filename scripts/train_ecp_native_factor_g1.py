#!/usr/bin/env python3
"""Optimize one held task of the ECP Native-Factor G1 capacity oracle."""

from ember.ecp.g1_training import build_parser, finalize_args, train


if __name__ == "__main__":
    train(finalize_args(build_parser().parse_args()))
