#!/usr/bin/env python3
"""Train EMBER-ECP Stage 1 outcome-calibrated Program--policy binding."""

from ember.ecp.stage1_outcome_training import (
    build_parser,
    finalize_args,
    train,
)


if __name__ == "__main__":
    train(finalize_args(build_parser().parse_args()))
