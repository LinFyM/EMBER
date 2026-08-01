#!/usr/bin/env python3
"""Evaluate AS-Writer diagnostics on the sealed validation-loss panel."""

from ember.writer.validation import build_parser, evaluate, finalize_args


if __name__ == "__main__":
    arguments = finalize_args(build_parser().parse_args())
    if arguments.diagnostic == "endpoint10":
        from ember.writer.endpoint_runtime import evaluate_endpoint

        evaluate_endpoint(arguments)
    else:
        evaluate(arguments)
