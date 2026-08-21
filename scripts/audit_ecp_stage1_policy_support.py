#!/usr/bin/env python3
"""Audit a frozen ECP Stage 1 projection on the complete support bank."""

from ember.ecp.stage1_support_audit import (
    assemble_audit,
    build_audit_shard,
    build_parser,
    finalize_args,
)


if __name__ == "__main__":
    arguments = finalize_args(build_parser().parse_args())
    if arguments.mode == "build":
        build_audit_shard(arguments)
    else:
        assemble_audit(arguments)
