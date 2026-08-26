#!/usr/bin/env python3
"""Train the active ECP G3 bank-conditioned mapping compiler."""

from ember.ecp.bank_conditioning.mapping_training import (
    build_parser,
    finalize_args,
    train,
)


if __name__ == "__main__":
    train(finalize_args(build_parser().parse_args()))
