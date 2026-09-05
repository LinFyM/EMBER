#!/usr/bin/env python3
"""Train or smoke the active unified Policy-Native Factor Writer."""

from ember.ecp.policy_response_writer.training import build_parser, finalize_args, run


if __name__ == "__main__":
    run(finalize_args(build_parser().parse_args()))
