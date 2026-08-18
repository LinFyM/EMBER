#!/usr/bin/env python3
"""Combine clean functional-gradient audits into remote-safe review evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


AUDIT_SCHEMA = "ember_writer_functional_gradient_audit_v1"
EVIDENCE_SCHEMA = "ember_external_review_gradient_credit_evidence_v1"


def _audit_argument(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("audit must be LABEL=PATH")
    return label, Path(raw_path)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _summary(audit: Mapping[str, Any]) -> dict[str, Any]:
    repository = audit.get("repository", {})
    if (
        audit.get("schema_version") != AUDIT_SCHEMA
        or repository.get("dirty_paths") != []
    ):
        raise ValueError("gradient evidence requires a clean canonical audit")
    states = list(audit["states"])
    return {
        "repository": repository,
        "first_observed_nonzero_state": audit["first_observed_nonzero_state"],
        "source_policy_zero_all_states": all(
            int(state["source_policy_nonzero_gradient_tensors"]) == 0
            for state in states
        ),
        "unclassified_empty_all_states": all(
            state["unclassified_parameter_names"] == [] for state in states
        ),
        "states": states,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="append", type=_audit_argument, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audits: dict[str, dict[str, Any]] = {}
    for label, path in args.audit:
        if label in audits:
            raise ValueError(f"duplicate audit label: {label}")
        audits[label] = _summary(_read(path))
    evidence = {
        "schema_version": EVIDENCE_SCHEMA,
        "audit_order": list(audits),
        "audits": audits,
        "comparison_boundary": (
            "A is the pre-fix architecture, B removes Text/VL Meta-LoRA while "
            "retaining the detach, and C changes only the three returned frontend "
            "tensors from detached to differentiable relative to B. Fresh zero-head "
            "credit ordering is reported separately from post-update credit."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
