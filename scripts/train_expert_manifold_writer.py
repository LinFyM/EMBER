#!/usr/bin/env python3
"""Train the video-conditioned expert-manifold topological Writer."""

from ember.expert_manifold.writer_training import build_parser, finalize_args, train


if __name__ == "__main__":
    train(finalize_args(build_parser().parse_args()))
