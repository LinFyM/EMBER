#!/usr/bin/env python3
"""Report retained/gained/lost behavior under the same PI0.5 rollout contract."""

import argparse
import json
import re
import subprocess
from pathlib import Path

from ember.pi05_eval_results import AGGREGATE_SCHEMA, paired_success_comparison
from ember.pi05_source_checkpoint import read_json, write_json_atomic


def normalization_evidence(contract: dict) -> tuple[dict, dict]:
    """Read live evidence or reconstruct its recorded clean Git version."""
    path = Path(contract["normalization"]["path"])
    if path.is_file():
        return read_json(path), {"path": str(path), "source": "recorded_file"}
    relative = "configs/pi05_source_corpus_v1/source_normalization.json"
    state = contract.get("git", {})
    commit = str(state.get("commit", ""))
    if (not str(path).endswith("/" + relative) or state.get("dirty_paths") != []
            or not re.fullmatch(r"[0-9a-f]{40}", commit)):
        raise ValueError("missing normalization has no unambiguous clean Git provenance")
    raw = subprocess.run(
        ["git", "show", f"{commit}:{relative}"], cwd=Path(__file__).resolve().parents[1],
        check=True, stdout=subprocess.PIPE,
    ).stdout
    if len(raw) != contract["normalization"]["bytes"]:
        raise ValueError("archived normalization size differs from the recorded artifact")
    return json.loads(raw), {"path": str(path), "source": "recorded_clean_git", "git_blob": f"{commit}:{relative}"}


def compare(reference: Path, candidate: Path) -> dict:
    panels, contracts = [], []
    for root in (reference, candidate):
        panel = read_json(root / "results.json")
        contract = read_json(root / "run_contract.json")
        completion = read_json(root / "launcher_completion.json")
        if (panel["schema_version"] != AGGREGATE_SCHEMA
                or panel["contract_reference"] != contract["contract_reference"]
                or completion["contract_reference"] != contract["contract_reference"]
                or any(code != 0 for code in completion["return_codes"].values())):
            raise ValueError("paired panel lacks successful complete evaluator evidence")
        panels.append(panel)
        contracts.append(contract)
    for field in ("policy", "environment", "rng"):
        if contracts[0][field] != contracts[1][field]:
            raise ValueError(f"paired rollout contract changed: {field}")
    if contracts[0]["model"]["checkpoint"] != contracts[1]["model"]["checkpoint"]:
        raise ValueError("paired reference uses a different frozen source")
    normalizers = [normalization_evidence(contract) for contract in contracts]
    if normalizers[0][0] != normalizers[1][0]:
        raise ValueError("paired rollout uses different source normalization")
    return {"reference": str(reference.resolve()), "candidate": str(candidate.resolve()),
            "reference_arm": panels[0]["arm"], "candidate_arm": panels[1]["arm"],
            "normalization_evidence": [item[1] for item in normalizers],
            "comparison": paired_success_comparison(*panels)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json_atomic(args.output, compare(args.reference, args.candidate))
