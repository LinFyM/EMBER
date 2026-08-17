#!/usr/bin/env python3
"""Historical GOMQ trainer; the canonical terminal config fails closed."""

from ember.writer.reward_training import build_parser, finalize_args, train


if __name__ == "__main__":
    train(finalize_args(build_parser().parse_args()))
