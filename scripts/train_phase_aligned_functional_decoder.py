#!/usr/bin/env python3
"""Train the phase-aligned multi-success fixed functional decoder."""

from ember.functional_adaptation.phase_decoder_training import parser, run


if __name__ == "__main__":
    run(parser().parse_args())
