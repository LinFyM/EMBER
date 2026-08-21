#!/usr/bin/env python3
"""Materialize privileged EMBER-ECP Stage 1 adapters."""

from ember.ecp.stage1_materialization import (
    build_parser,
    finalize_materialization_args,
    materialize,
)


if __name__ == "__main__":
    materialize(finalize_materialization_args(build_parser().parse_args()))
