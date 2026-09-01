#!/usr/bin/env python3
"""Analyze PNBTT E1 current-bank tangent and functional-gradient spectra."""

from pathlib import Path

from ember.ecp.joint_program_primal.pnbtt_tangent_spectrum import (
    analyze_pnbtt_tangent_spectrum,
)
from ember.ecp.joint_program_primal.training import build_parser, finalize_args


if __name__ == "__main__":
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--writer-checkpoint", type=Path, required=True)
    parser.add_argument("--panel-visits", type=int, default=16)
    arguments = finalize_args(parser.parse_args())
    arguments.writer_checkpoint = arguments.writer_checkpoint.resolve()
    if arguments.mode != "profile" or arguments.phase != "joint":
        raise ValueError("PNBTT tangent spectrum is a read-only joint profile")
    if arguments.resume is not None or arguments.evaluate_checkpoint is not None:
        raise ValueError("PNBTT tangent spectrum does not resume or evaluate")
    analyze_pnbtt_tangent_spectrum(arguments)
