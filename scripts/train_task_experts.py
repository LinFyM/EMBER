#!/usr/bin/env python3
"""Train the video-expert-manifold task-local PI0.5 experts."""

from ember.expert_manifold.expert_training import build_parser, finalize_args, train


if __name__ == "__main__":
    train(finalize_args(build_parser().parse_args()))
