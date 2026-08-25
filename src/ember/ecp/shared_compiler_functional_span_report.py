"""Aggregate the sharded G3 full-to-mobile-to-native functional diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ember.ecp.shared_compiler_functional_span import (
    DEFAULT_SCREEN_TASK_IDS,
    RESULT_SCHEMA,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic


REPORT_SCHEMA = "ember_ecp_g3_functional_native_span_report_v1"
ROLES = ("meta_fit", "target_fit")


def _distribution(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(map(float, values))
    if not ordered:
        raise ValueError("G3 functional-span distribution is empty")

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
        raise ValueError("G3 functional-span member panel is empty")
    positive_full = [
        (task, member)
        for task, member in pairs
        if float(member["action_flow"]["full_benefit_over_carrier"]) > 0
    ]
    summary = {
        "member_count": len(pairs),
        "mobile_named_effect_retention": _distribution(
            member["effect"]["mobile_rank4"]["named_member_retention"]
            for _, member in pairs
        ),
        "native_named_effect_retention": _distribution(
            member["effect"]["native_projected"]["named_member_retention"]
            for _, member in pairs
        ),
        "mobile_global_set_retention": _distribution(
            member["effect"]["mobile_rank4"]["global_set_retention"]
            for _, member in pairs
        ),
        "native_global_set_retention": _distribution(
            member["effect"]["native_projected"]["global_set_retention"]
            for _, member in pairs
        ),
        "full_action_benefit_over_carrier": _distribution(
            member["action_flow"]["full_benefit_over_carrier"] for _, member in pairs
        ),
        "mobile_action_benefit_over_carrier": _distribution(
            member["action_flow"]["mobile_benefit_over_carrier"] for _, member in pairs
        ),
        "native_action_benefit_over_carrier": _distribution(
            member["action_flow"]["native_benefit_over_carrier"] for _, member in pairs
        ),
        "full_to_mobile_update_cosine": _distribution(
            member["geometry"]["full_to_mobile"]["overall"]["update_cosine"]
            for _, member in pairs
        ),
        "mobile_to_native_update_cosine": _distribution(
            member["geometry"]["mobile_to_native"]["overall"]["update_cosine"]
            for _, member in pairs
        ),
        "counts": {
            "full_action_benefit_positive": len(positive_full),
            "mobile_action_benefit_positive": sum(
                member["action_flow"]["mobile_benefit_over_carrier"] > 0
                for _, member in pairs
            ),
            "native_action_benefit_positive": sum(
                member["action_flow"]["native_benefit_over_carrier"] > 0
                for _, member in pairs
            ),
            "mobile_named_effect_retention_at_least_0_80": sum(
                member["effect"]["mobile_rank4"]["named_member_retention"] >= 0.80
                for _, member in pairs
            ),
            "native_named_effect_retention_at_least_0_80": sum(
                member["effect"]["native_projected"]["named_member_retention"] >= 0.80
                for _, member in pairs
            ),
        },
    }
    if positive_full:
        summary["positive_full_action_panel"] = {
            "member_count": len(positive_full),
            "mobile_retention": _distribution(
                member["action_flow"]["mobile_retention"] for _, member in positive_full
            ),
            "native_retention": _distribution(
                member["action_flow"]["native_retention"] for _, member in positive_full
            ),
        }
    return summary


def _load_shards(
    root: Path, shard_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payloads = []
    contracts = []
    for index in range(shard_count):
        shard = root / f"shard_{index}"
        payload = read_json(shard / "functional_span_results.json")
        contract = read_json(shard / "run_contract.json")
        wall = payload.get("information_wall", {})
        if (
            payload.get("schema_version") != RESULT_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("shard")
            != {
                "index": index,
                "count": shard_count,
                "task_count": len(payload.get("tasks", ())),
            }
            or wall.get("held_tasks") != 0
            or wall.get("validation_or_test_reads") != 0
            or wall.get("action_meta_installed") is not False
            or wall.get("action_meta_named_modules") != []
            or wall.get("shuffled_or_reversed_use") is not False
            or wall.get("video_action_cross_episode") is not True
            or contract.get("git", {}).get("branch") != ""
            or contract.get("git", {}).get("commit")
            != contract.get("git", {}).get("authority_commit")
        ):
            raise ValueError(f"G3 functional-span shard {index} authority changed")
        payloads.append(payload)
        contracts.append(contract)
    commits = {contract["git"]["commit"] for contract in contracts}
    if len(commits) != 1:
        raise ValueError("G3 functional-span shards used different commits")
    return payloads, contracts


def build_functional_span_report(
    *, shard_root: Path, shard_count: int, output_path: Path
) -> dict[str, Any]:
    shard_root = shard_root.resolve()
    payloads, contracts = _load_shards(shard_root, shard_count)
    tasks = [task for payload in payloads for task in payload["tasks"]]
    task_ids = [int(task["authority_id"]) for task in tasks]
    role_counts = {role: sum(task["role"] == role for task in tasks) for role in ROLES}
    if (
        len(tasks) != len(DEFAULT_SCREEN_TASK_IDS)
        or set(task_ids) != set(DEFAULT_SCREEN_TASK_IDS)
        or role_counts != {"meta_fit": 3, "target_fit": 3}
    ):
        raise ValueError("G3 functional-span screen task authority changed")
    pairs = [(task, member) for task in tasks for member in task["members"]]
    if len(pairs) != 9:
        raise ValueError("G3 functional-span screen member count changed")
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "complete",
        "question": (
            "whether function is first lost in full-expert to mobile-rank4 "
            "compression, mobile-rank4 to real K1-native-span projection, or "
            "only later in the learned shared mapping"
        ),
        "claim_boundary": payloads[0]["claim_boundary"],
        "authority": {
            "shard_root": str(shard_root),
            "shard_count": shard_count,
            "run_commit": contracts[0]["git"]["commit"],
            "task_count": len(tasks),
            "verified_member_count": len(pairs),
            "roles": role_counts,
            "task_ids": sorted(task_ids),
            "max_cuda_allocated_bytes": max(
                int(payload["max_cuda_allocated_bytes"]) for payload in payloads
            ),
        },
        "information_wall": {
            "roles": list(ROLES),
            "held_tasks": 0,
            "validation_or_test_reads": 0,
            "action_meta_installed": False,
            "action_meta_named_modules": [],
            "shuffled_or_reversed_use": False,
            "video_action_cross_episode": True,
        },
        "metrics": {
            "all_members": _member_summary(pairs),
            "by_role": {
                role: _member_summary(
                    [(task, member) for task, member in pairs if task["role"] == role]
                )
                for role in ROLES
            },
        },
        "members": [
            {
                "authority_id": int(task["authority_id"]),
                "role": task["role"],
                "member": member["member"],
                "step": int(member["step"]),
                "mobile_named_effect_retention": member["effect"]["mobile_rank4"][
                    "named_member_retention"
                ],
                "native_named_effect_retention": member["effect"]["native_projected"][
                    "named_member_retention"
                ],
                "full_action_benefit_over_carrier": member["action_flow"][
                    "full_benefit_over_carrier"
                ],
                "mobile_action_benefit_over_carrier": member["action_flow"][
                    "mobile_benefit_over_carrier"
                ],
                "native_action_benefit_over_carrier": member["action_flow"][
                    "native_benefit_over_carrier"
                ],
                "mobile_action_retention": member["action_flow"]["mobile_retention"],
                "native_action_retention": member["action_flow"]["native_retention"],
                "full_to_mobile_update_cosine": member["geometry"]["full_to_mobile"][
                    "overall"
                ]["update_cosine"],
                "mobile_to_native_update_cosine": member["geometry"][
                    "mobile_to_native"
                ]["overall"]["update_cosine"],
            }
            for task, member in sorted(
                pairs,
                key=lambda pair: (
                    int(pair[0]["authority_id"]),
                    int(pair[1]["step"]),
                ),
            )
        ],
    }
    write_json_atomic(output_path.resolve(), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_functional_span_report(
        shard_root=args.shard_root,
        shard_count=args.shard_count,
        output_path=args.output,
    )
    metrics = report["metrics"]["all_members"]
    print(
        "completed G3 functional-span report: "
        f"{report['authority']['task_count']} tasks, "
        "median mobile/native named-effect retention "
        f"{metrics['mobile_named_effect_retention']['median']:.6f}/"
        f"{metrics['native_named_effect_retention']['median']:.6f}",
        flush=True,
    )
    return 0
