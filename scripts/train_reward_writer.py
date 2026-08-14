#!/usr/bin/env python3
"""Train the ordered-Procedure Writer from train24 on-policy reward."""

from ember.writer.reward_training import build_parser, finalize_args, train


if __name__ == "__main__":
    train(finalize_args(build_parser().parse_args()))
