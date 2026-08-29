#!/usr/bin/env python3
"""Run the J2 functional positive control or joint Program--primal training."""

from ember.ecp.joint_program_primal.gate import run_positive_control
from ember.ecp.joint_program_primal.training import (
    build_parser,
    finalize_args,
    train,
)


if __name__ == "__main__":
    arguments = finalize_args(build_parser().parse_args())
    if arguments.phase == "positive-control":
        run_positive_control(arguments)
    else:
        train(arguments)
