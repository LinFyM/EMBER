#!/usr/bin/env python3
"""Train or profile the canonical v6-prior Expert-Manifold Writer."""

from ember.expert_manifold.v6_prior_training import (
    build_parser,
    finalize_args,
    train,
)


if __name__ == "__main__":
    train(finalize_args(build_parser().parse_args()))
