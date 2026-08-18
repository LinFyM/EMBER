#!/usr/bin/env python3
"""Export remote-safe FactorHead reachability and projected-rollout evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from ember.pi05_eval.paired_metrics import index_rows, paired_transition_summary


RESULT_SCHEMA = "ember_pi05_target_eval_results_v2"
PROJECTION_SCHEMA = "ember_writer_fixed_head_reachability_oracle_v1"
EVIDENCE_SCHEMA = "ember_external_review_reachability_evidence_v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _validate_panel(result: Mapping[str, Any]) -> None:
    if (
        result.get("schema_version") != RESULT_SCHEMA
        or result.get("role") != "development_train"
        or result.get("mode") != "formal"
        or len(result.get("rows", ())) != 1200
        or int(result.get("overall", {}).get("episodes", 0)) != 1200
    ):
        raise ValueError("reachability rollout is not a formal train24x50 panel")


def _contract_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    git = contract.get("git", {})
    if contract.get("mode") != "formal" or git.get("dirty_paths") != []:
        raise ValueError("reachability rollout contract is not clean formal evidence")
    return {
        "contract_reference": contract.get("contract_reference"),
        "git_commit": git.get("commit"),
        "git_dirty_paths": git.get("dirty_paths"),
        "host": contract.get("host"),
        "mode": contract.get("mode"),
    }


def _rng_prefix_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_seeds = list(left.get("policy_noise_seeds", ()))
    right_seeds = list(right.get("policy_noise_seeds", ()))
    common = min(len(left_seeds), len(right_seeds))
    return left_seeds[:common] == right_seeds[:common]


def _per_task(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["suite"]), int(row["task_id"]))].append(row)
    return [
        {
            "suite": suite,
            "task_id": task_id,
            "episodes": len(selected),
            "successes": sum(bool(row["success"]) for row in selected),
        }
        for (suite, task_id), selected in sorted(grouped.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-manifest", type=Path, required=True)
    parser.add_argument("--direct-results", type=Path, required=True)
    parser.add_argument("--direct-contract", type=Path, required=True)
    parser.add_argument("--projected-results", type=Path, required=True)
    parser.add_argument("--projected-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    projection = _read(args.projection_manifest)
    direct = _read(args.direct_results)
    projected = _read(args.projected_results)
    if (
        projection.get("schema_version") != PROJECTION_SCHEMA
        or projection.get("repository", {}).get("dirty_paths") != []
        or len(projection.get("tasks", ())) != 24
    ):
        raise ValueError("fixed-head projection manifest changed")
    _validate_panel(direct)
    _validate_panel(projected)
    direct_index = index_rows(direct["rows"])
    projected_index = index_rows(projected["rows"])
    if set(direct_index) != set(projected_index) or len(direct_index) != 1200:
        raise ValueError("direct and projected expert panels are not paired")
    mismatch = {
        "environment_seed": 0,
        "policy_seed_root": 0,
        "policy_noise_common_prefix": 0,
    }
    remote_rows = []
    for key in sorted(direct_index):
        left, right = direct_index[key], projected_index[key]
        mismatch["environment_seed"] += left.get("env_seed") != right.get("env_seed")
        mismatch["policy_seed_root"] += left.get("policy_seed_root") != right.get(
            "policy_seed_root"
        )
        mismatch["policy_noise_common_prefix"] += not _rng_prefix_matches(left, right)
        remote_rows.append(
            {
                "suite": key[0],
                "task_id": key[1],
                "init_state_id": key[2],
                "env_seed": left.get("env_seed"),
                "policy_seed_root": left.get("policy_seed_root"),
                "direct_success": bool(left["success"]),
                "projected_success": bool(right["success"]),
                "direct_noise_seed_count": len(left.get("policy_noise_seeds", ())),
                "projected_noise_seed_count": len(
                    right.get("policy_noise_seeds", ())
                ),
            }
        )

    direct_successes = int(direct["overall"]["successes"])
    projected_successes = int(projected["overall"]["successes"])
    transition = paired_transition_summary(direct["rows"], projected["rows"])
    evidence = {
        "schema_version": EVIDENCE_SCHEMA,
        "provenance": {
            "projection_repository": projection["repository"],
            "direct_rollout": _contract_summary(_read(args.direct_contract)),
            "projected_rollout": _contract_summary(_read(args.projected_contract)),
        },
        "projection_optimization": projection["optimization"],
        "projection_information_wall": projection["information_wall"],
        "projection_elapsed_seconds": projection["elapsed_seconds"],
        "projection_curve": projection["curve"],
        "final_projection_curve_point": projection["curve"][-1],
        "task_projection_metrics": [
            {
                "ordinal": task["ordinal"],
                "global_task_id": task["global_task_id"],
                "suite": task["suite"],
                "task_id": task["task_id"],
                "metrics": task["metrics"],
                "projected_adapter_bytes": task["projected_adapter_bytes"],
            }
            for task in projection["tasks"]
        ],
        "pairing_audit": {
            "row_count": len(remote_rows),
            "mismatch_counts": mismatch,
            "strict_pairing_verified": not any(mismatch.values()),
            "rng_note": (
                "Noise sequences use common-prefix comparison because successful "
                "episodes terminate early."
            ),
        },
        "direct_expert": {
            "overall": direct["overall"],
            "per_task": _per_task(direct["rows"]),
        },
        "fixed_head_projection": {
            "overall": projected["overall"],
            "per_task": _per_task(projected["rows"]),
        },
        "paired_transition": transition,
        "advisory_threshold": {
            "definition": "projected total successes retain at least 90 percent of direct expert total successes",
            "direct_successes": direct_successes,
            "required_successes": 0.9 * direct_successes,
            "projected_successes": projected_successes,
            "projected_over_direct": projected_successes / direct_successes,
            "passes": projected_successes >= 0.9 * direct_successes,
        },
        "rows": remote_rows,
        "deployment_boundary": (
            "The free per-task Program and projected adapters are privileged "
            "train24 diagnostics only and are not a deployment route or held score."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
