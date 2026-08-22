#!/usr/bin/env python3
"""Run or assemble the exact-effect EMBER-PECS oracle."""

from ember.ecp.effect_oracle import build_parser, finalize_args, main


if __name__ == "__main__":
    main(finalize_args(build_parser().parse_args()))
