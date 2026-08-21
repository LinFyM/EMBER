#!/usr/bin/env python3
"""Launch fixed-decoder Writer training with train-task closed-loop credit."""

from ember.functional_adaptation.outer_credit_training import (
    finalize_args,
    parser,
    prepare_runtime,
)
from ember.functional_adaptation.outer_credit_train_step import train


def main() -> None:
    args = finalize_args(parser().parse_args())
    train(prepare_runtime(args))


if __name__ == "__main__":
    main()
