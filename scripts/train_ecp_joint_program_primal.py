#!/usr/bin/env python3
"""Run the retained positive control, J3, or routing boundary control."""

from ember.ecp.joint_program_primal.routing_control import (
    ROUTING_CONTROL_SCHEMA,
)
from ember.ecp.joint_program_primal.routing_control_training import (
    train_routing_control,
)
from ember.ecp.joint_program_primal.gate import run_positive_control
from ember.ecp.joint_program_primal.training import (
    build_parser,
    finalize_args,
    train,
)
from ember.pi05_source_checkpoint import read_json


if __name__ == "__main__":
    arguments = finalize_args(build_parser().parse_args())
    if arguments.phase == "positive-control":
        run_positive_control(arguments)
    elif read_json(arguments.config).get("schema_version") == ROUTING_CONTROL_SCHEMA:
        train_routing_control(arguments)
    else:
        train(arguments)
