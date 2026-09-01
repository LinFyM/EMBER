#!/usr/bin/env python3
"""Run the canonical PNBTT trainer or a retained sealed control."""

from ember.ecp.joint_program_primal.routing_control import (
    PROGRAM_BANK_INTERACTION_SCHEMA,
    ROUTING_CONTROL_SCHEMA,
)
from ember.ecp.joint_program_primal.routing_control_training import (
    train_routing_control,
)
from ember.ecp.joint_program_primal.gate import run_positive_control
from ember.ecp.joint_program_primal.pnbtt_runtime import is_pnbtt_tasklocal_config
from ember.ecp.joint_program_primal.pnbtt_evaluation import (
    evaluate_pnbtt_tasklocal,
)
from ember.ecp.joint_program_primal.pnbtt_training import train_pnbtt_tasklocal
from ember.ecp.joint_program_primal.training import (
    build_parser,
    finalize_args,
)
from ember.pi05_source_checkpoint import read_json


if __name__ == "__main__":
    arguments = finalize_args(build_parser().parse_args())
    selected = read_json(arguments.config)
    if is_pnbtt_tasklocal_config(selected):
        if arguments.phase != "joint" or arguments.task is not None:
            raise ValueError("PNBTT task-local training uses the joint phase")
        if arguments.evaluate_checkpoint is not None:
            evaluate_pnbtt_tasklocal(arguments)
        else:
            train_pnbtt_tasklocal(arguments)
    elif arguments.phase == "positive-control":
        run_positive_control(arguments)
    elif (
        selected.get("schema_version") == PROGRAM_BANK_INTERACTION_SCHEMA
    ):
        raise ValueError(
            "retired pointwise Program-bank interaction config is not executable"
        )
    elif selected.get("schema_version") == ROUTING_CONTROL_SCHEMA:
        train_routing_control(arguments)
    else:
        raise ValueError(
            "retired joint Program-primal config is retained for audit only"
        )
