#!/usr/bin/env python3
"""Read complete low-LR Writer Validation400 nodes and apply phase-only stopping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ember.early_stopping import phase_validation_decision
from ember.pi05_eval_results import AGGREGATE_SCHEMA, paired_success_comparison


PARENT_STEP = 1800
INTERVAL = 100
REFERENCE_BEST = 117


def read_phase(study: Path, protocol_path: Path, endpoint: int) -> dict:
    if endpoint < PARENT_STEP + INTERVAL or (endpoint - PARENT_STEP) % INTERVAL:
        raise ValueError("endpoint must be a registered low-LR phase validation node")
    protocol = json.loads(protocol_path.read_text())
    expected = {
        (suite, task, state)
        for suite, roles in protocol["split"]["suites"].items()
        for task in roles["validation"]
        for state in range(50)
    }
    history, comparisons, previous = [], [], None
    for step in range(PARENT_STEP + INTERVAL, endpoint + 1, INTERVAL):
        directory = study / "evaluation" / f"writer_{step:08d}"
        result, contract, done = [
            json.loads((directory / name).read_text())
            for name in ("results.json", "run_contract.json", "launcher_completion.json")
        ]
        if result["schema_version"] != AGGREGATE_SCHEMA:
            raise ValueError("unexpected aggregate schema")
        if len({result["contract_reference"], contract["contract_reference"],
                done["contract_reference"]}) != 1:
            raise ValueError("evaluation completion contract reference changed")
        if contract["role"] != "validation" or contract["mode"] != "formal":
            raise ValueError("low-LR phase accepts only formal validation panels")
        if not done["return_codes"] or any(code != 0 for code in done["return_codes"].values()):
            raise ValueError("evaluation worker did not exit successfully")
        rows = result["rows"]
        if {(row["suite"], row["task_id"], row["init_state_id"]) for row in rows} != expected:
            raise ValueError("validation task/state coverage changed")
        if not len(rows) == result["overall"]["episodes"] == done["queue"]["completed_rows"]:
            raise ValueError("validation panel is incomplete")
        if len(rows) != 400:
            raise ValueError("low-LR phase requires exactly 400 paired rows")
        successes = sum(bool(row["success"]) for row in rows)
        if successes != result["overall"]["successes"] or successes != done["queue"]["successes"]:
            raise ValueError("validation success aggregate changed")
        history.append({
            "step": step,
            "global_step": step,
            "phase_step": step - PARENT_STEP,
            "complete": True,
            "episodes": 400,
            "successes": successes,
            "results": str(directory / "results.json"),
        })
        if previous is not None:
            comparisons.append({"step": step, **paired_success_comparison(previous, result)})
        previous = result
    decision = phase_validation_decision(history, parent_step=PARENT_STEP, interval=INTERVAL)
    return {
        "schema_version": "ember_writer_low_lr_phase_history_v1",
        "method": "writer_low_lr_repair",
        "parent": {"step": PARENT_STEP, "successes": 92},
        "selection_reference": {"step": 1000, "successes": REFERENCE_BEST},
        "mtbc_reference": {"step": 300, "successes": 155},
        "history": history,
        "decision": decision,
        "adjacent_comparisons": comparisons,
        "beats_original_best": decision["best_successes"] > REFERENCE_BEST,
        "selection_finalized": False,
        "next": "stop for phase selection" if decision["stop"] else "next registered 100-update segment",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--endpoint", type=int, required=True)
    args = parser.parse_args()
    result = read_phase(args.study, args.protocol, args.endpoint)
    path = args.study / "writer_phase_validation_history.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps({key: value for key, value in result.items() if key != "adjacent_comparisons"}, indent=2))


if __name__ == "__main__":
    main()
