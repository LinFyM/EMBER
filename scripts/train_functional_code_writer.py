#!/usr/bin/env python3
"""Train the fixed-decoder language/video functional-code Writer."""

from ember.functional_adaptation.code_training import (
    finalize_args,
    parser,
    prepare_runtime,
)
from ember.functional_adaptation.code_train_step import train


if __name__ == "__main__":
    train(prepare_runtime(finalize_args(parser().parse_args())))
