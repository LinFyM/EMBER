"""Aggregate the sharded G3 fit-task native-span diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ember.ecp.shared_compiler_span import RESULT_SCHEMA
from ember.pi05_source_checkpoint import read_json, write_json_atomic


REPORT_SCHEMA = "ember_ecp_g3_fit_native_span_report_v1"
FAMILIES = ("q", "v", "action_in", "action_out")
ROLES = ("meta_fit", "target_fit")


def _distribution(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(map(float, values))
    if not ordered:
        raise ValueError("G3 fit-span distribution is empty")

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "minimum": ordered[0],
        "p10": percentile(0.10),
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "maximum": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _member_summary(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("G3 fit-span member panel is empty")
    return {
        "member_count": len(pairs),
        "overall_update_cosine": _distribution(
            member["geometry"]["overall"]["update_cosine"]
            for _, member in pairs
        ),
        "overall_relative_update_error": _distribution(
            member["geometry"]["overall"]["relative_update_error"]
            for _, member in pairs
        ),
        "candidate_to_reference_norm": _distribution(
            member["geometry"]["overall"]["candidate_to_reference_norm"]
            for _, member in pairs
        ),
        "reference_scale_above_cap_fraction": _distribution(
            member["reference_scale_pressure"]["overall"][
                "above_cap_fraction"
            ]
            for _, member in pairs
        ),
        "projected_scale_cap_fraction": _distribution(
            member["projected_scale_cap_fraction"] for _, member in pairs
        ),
        "minimum_signed_input_realization_cosine": _distribution(
            member["minimum_signed_input_realization_cosine"]
            for _, member in pairs
        ),
        "minimum_signed_output_realization_cosine": _distribution(
            member["minimum_signed_output_realization_cosine"]
            for _, member in pairs
        ),
        "counts": {
            "update_cosine_at_least_0_90": sum(
                member["geometry"]["overall"]["update_cosine"] >= 0.90
                for _, member in pairs
            ),
            "update_cosine_at_least_0_95": sum(
                member["geometry"]["overall"]["update_cosine"] >= 0.95
                for _, member in pairs
            ),
            "relative_update_error_at_most_0_50": sum(
                member["geometry"]["overall"]["relative_update_error"] <= 0.50
                for _, member in pairs
            ),
        },
    }


def _load_shards(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payloads = []
    contracts = []
    for index in range(6):
        shard = root / f"shard_{index}"
        payload = read_json(shard / "span_results.json")
        contract = read_json(shard / "run_contract.json")
        if (
            payload.get("schema_version") != RESULT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("shard")
            != {
                "index": index,
                "count": 6,
                "task_count": len(payload.get("tasks", ())),
            }
            or contract.get("git", {}).get("branch") != ""
            or contract.get("git", {}).get("commit")
            != contract.get("git", {}).get("authority_commit")
        ):
            raise ValueError(f"G3 fit-span shard {index} authority changed")
        payloads.append(payload)
        contracts.append(contract)
    commits = {contract["git"]["commit"] for contract in contracts}
    if len(commits) != 1:
        raise ValueError("G3 fit-span shards used different commits")
    return payloads, contracts


def build_fit_span_report(*, shard_root: Path, output_path: Path) -> dict[str, Any]:
    shard_root = shard_root.resolve()
    payloads, contracts = _load_shards(shard_root)
    tasks = [task for payload in payloads for task in payload["tasks"]]
    task_ids = [int(task["authority_id"]) for task in tasks]
    role_counts = {
        role: sum(task["role"] == role for task in tasks) for role in ROLES
    }
    if (
        len(tasks) != 75
        or len(set(task_ids)) != 75
        or role_counts != {"meta_fit": 56, "target_fit": 19}
        or any(
            payload.get("information_wall")
            != {
                "roles": ["meta_fit", "target_fit"],
                "held_tasks": 0,
                "validation_or_test_reads": 0,
                "action_meta_installed": False,
                "shuffled_or_reversed_use": False,
            }
            for payload in payloads
        )
    ):
        raise ValueError("G3 fit-span task or information-wall authority changed")
    pairs = [(task, member) for task in tasks for member in task["members"]]
    if len(pairs) != 93:
        raise ValueError("G3 fit-span verified-member count changed")
    best_members = [
        min(
            task["members"],
            key=lambda member: member["geometry"]["overall"][
                "relative_update_error"
            ],
        )
        for task in tasks
    ]
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "complete",
        "question": (
            "whether current fit-task verified mobile-rank4 parameter targets "
            "are directly compatible with their real K1 native X/Y output "
            "manifolds under the bounded signed-pooling form"
        ),
        "claim_boundary": (
            "direct stable-span projection of the current verified rank4 "
            "factorizations through one deterministic K1 video per fit task; "
            "this does not optimize a native free-code functional equivalent, "
            "prove the shared Program-to-attention mapping, or prove closed-loop"
        ),
        "authority": {
            "shard_root": str(shard_root),
            "shard_count": 6,
            "run_commit": contracts[0]["git"]["commit"],
            "task_count": len(tasks),
            "verified_member_count": len(pairs),
            "roles": role_counts,
            "task_ids": sorted(task_ids),
            "max_cuda_allocated_bytes": max(
                int(payload["max_cuda_allocated_bytes"])
                for payload in payloads
            ),
        },
        "information_wall": payloads[0]["information_wall"],
        "metrics": {
            "all_members": _member_summary(pairs),
            "by_role": {
                role: _member_summary(
                    [
                        (task, member)
                        for task, member in pairs
                        if task["role"] == role
                    ]
                )
                for role in ROLES
            },
            "task_best": {
                "update_cosine": _distribution(
                    member["geometry"]["overall"]["update_cosine"]
                    for member in best_members
                ),
                "relative_update_error": _distribution(
                    member["geometry"]["overall"]["relative_update_error"]
                    for member in best_members
                ),
            },
            "families": {
                family: {
                    "update_cosine": _distribution(
                        member["geometry"]["families"][family]["update_cosine"]
                        for _, member in pairs
                    ),
                    "relative_update_error": _distribution(
                        member["geometry"]["families"][family][
                            "relative_update_error"
                        ]
                        for _, member in pairs
                    ),
                }
                for family in FAMILIES
            },
        },
        "lowest_coverage_tasks": [
            {
                "authority_id": int(task["authority_id"]),
                "role": task["role"],
                "language": task["language"],
                "best_relative_update_error": float(
                    task["best_relative_update_error"]
                ),
            }
            for task in sorted(
                tasks,
                key=lambda row: float(row["best_relative_update_error"]),
                reverse=True,
            )[:10]
        ],
    }
    write_json_atomic(output_path.resolve(), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_fit_span_report(
        shard_root=args.shard_root, output_path=args.output
    )
    metrics = report["metrics"]["all_members"]
    print(
        "completed G3 fit-span report: "
        f"{report['authority']['task_count']} tasks, "
        f"median cosine {metrics['overall_update_cosine']['median']:.6f}",
        flush=True,
    )
    return 0
