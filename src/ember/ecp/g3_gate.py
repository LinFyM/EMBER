"""Build the paired five-arm G3 shared-compiler closed-loop Gate report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from ember.ecp.g3_gate_results import load_paired_g3_arms
from ember.ecp.shared_compiler_assets import (
    authority_path,
    load_shared_compiler_config,
)
from ember.ecp.shared_compiler_evaluation_runtime import (
    REPO_ROOT,
    load_g3_gate_config,
)
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.static_task_lora import STATIC_TASK_LORA_KIND


G3_GATE_SCHEMA = "ember_ecp_shared_compiler_g3_gate_report_v1"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
VIDEO_ARMS = {
    "correct_full": "ecp_shared_compiler_g3_correct_full",
    "first_final": "ecp_shared_compiler_g3_first_final",
    "same_task_other": "ecp_shared_compiler_g3_same_task_other",
}
LANGUAGE_ARM = "ecp_shared_compiler_g3_learned_language_only"


def _static_bank_valid(
    arm: Mapping[str, Any], *, expected_arm: str, expected_condition: str
) -> bool:
    adapter = arm["adapter"]
    tasks = tuple(adapter.get("tasks", ()))
    wall = adapter.get("information_wall", {})
    condition = adapter.get("condition", {})
    return (
        arm["arm"] == expected_arm
        and adapter.get("kind") == STATIC_TASK_LORA_KIND
        and adapter.get("single_complete_rank16") is True
        and adapter.get("rank_partition")
        == {"carrier": [0, 12], "task": [12, 16]}
        and len(tasks) == 5
        and len({str(row.get("adapter_path")) for row in tasks}) == 5
        and all(row.get("single_complete_rank16") is True for row in tasks)
        and condition.get("name") == expected_condition
        and wall.get("action_meta_installed") is False
        and wall.get("second_adapter_deployed") is False
        and int(wall.get("teacher_video_runtime_reads", -1)) == 0
        and wall.get("shuffled_or_reversed_use") is False
    )


def _g3_adapter_authority(
    arms: Mapping[str, Mapping[str, Any]], gate: Mapping[str, Any]
) -> tuple[dict[str, bool], dict[str, Any]]:
    language = arms["learned_language_only"]
    language_adapter = language["adapter"]
    language_contract = language_adapter.get("shared_run_contract", {})
    language_valid = (
        _static_bank_valid(
            language,
            expected_arm=LANGUAGE_ARM,
            expected_condition="learned_language_only",
        )
        and language_adapter.get("condition", {}).get("video_demos") == []
        and int(language_adapter.get("condition", {}).get("K", -1)) == 0
        and language_contract.get("schema_version")
        == "ember_ecp_g3_language_only_baseline_v1"
        and language_contract.get("stage") == "g3_learned_language_only"
        and language_contract.get("mode") == "formal"
        and language_contract.get("method") == gate["language_baseline"]
        and int(language_contract.get("held_video_reads", -1)) == 0
        and int(language_contract.get("held_action_or_reward_reads", -1)) == 0
    )
    video_valid = {}
    checkpoint_authorities = []
    for name, expected_arm in VIDEO_ARMS.items():
        adapter = arms[name]["adapter"]
        video_valid[name] = (
            _static_bank_valid(
                arms[name],
                expected_arm=expected_arm,
                expected_condition=name,
            )
            and int(adapter.get("condition", {}).get("K", -1)) == 4
            and adapter.get("condition", {}).get("video_demos")
            == gate["video_panel"][name]
            and isinstance(adapter.get("training_commit"), str)
            and len(adapter["training_commit"]) == 40
            and isinstance(adapter.get("materialization_commit"), str)
            and len(adapter["materialization_commit"]) == 40
        )
        checkpoint = adapter.get("compiler_checkpoint", {})
        checkpoint_authorities.append(
            (
                adapter.get("training_commit"),
                checkpoint.get("path"),
                int(checkpoint.get("macro", -1)),
            )
        )
    single_checkpoint = (
        len(set(checkpoint_authorities)) == 1
        and checkpoint_authorities[0][2] > 0
        and isinstance(checkpoint_authorities[0][1], str)
    )
    checks = {
        "language_bank": language_valid,
        **{f"{name}_bank": value for name, value in video_valid.items()},
        "single_compiler_checkpoint": single_checkpoint,
    }
    return checks, {
        "training_commit": checkpoint_authorities[0][0],
        "compiler_checkpoint": checkpoint_authorities[0][1],
        "compiler_macro": checkpoint_authorities[0][2],
    }


def _success_metrics(
    arms: Mapping[str, Mapping[str, Any]], held_keys: set[tuple[str, int]]
) -> dict[str, Any]:
    carrier = arms["carrier"]
    full = arms["correct_full"]
    endpoints = arms["first_final"]
    language = arms["learned_language_only"]
    other = arms["same_task_other"]
    retained = sum(
        bool(carrier["rows"][key]["success"])
        and bool(full["rows"][key]["success"])
        for key in full["rows"]
    )
    gained = sum(
        not bool(carrier["rows"][key]["success"])
        and bool(full["rows"][key]["success"])
        for key in full["rows"]
    )
    lost = sum(
        bool(carrier["rows"][key]["success"])
        and not bool(full["rows"][key]["success"])
        for key in full["rows"]
    )
    same_retained = sum(
        bool(full["rows"][key]["success"])
        and bool(other["rows"][key]["success"])
        for key in full["rows"]
    )
    suite_successes = {
        suite: sum(
            value
            for (task_suite, _), value in full["per_task"].items()
            if task_suite == suite
        )
        for suite in SUITES
    }
    return {
        "full_successes": int(full["successes"]),
        "breadth": int(full["breadth"]),
        "carrier_successes_retained": retained,
        "gained": gained,
        "lost": lost,
        "churn": gained + lost,
        "full_over_language": int(full["successes"])
        - int(language["successes"]),
        "full_over_first_final": int(full["successes"])
        - int(endpoints["successes"]),
        "same_task_retained": same_retained,
        "same_task_retention": same_retained / max(int(full["successes"]), 1),
        "suite_successes": suite_successes,
        "tasks_above_carrier": sum(
            full["per_task"][key] > carrier["per_task"][key] for key in held_keys
        ),
    }


def build_g3_gate_report(
    *,
    config_path: Path,
    asset_root: Path,
    language_results: Path,
    full_results: Path,
    first_final_results: Path,
    same_task_results: Path,
    output_path: Path,
) -> dict[str, Any]:
    gate = load_g3_gate_config(config_path.resolve())
    training = load_shared_compiler_config(
        (REPO_ROOT / gate["training_config"]).resolve()
    )
    target_manifest = read_json(
        authority_path(training, "target_manifest", asset_root=asset_root)
    )
    held_globals = set(map(int, training["fold"]["target_held_task_ids"]))
    held_rows = [
        row
        for row in target_manifest.get("tasks", ())
        if int(row["global_task_id"]) in held_globals
    ]
    held_keys = {(str(row["suite"]), int(row["task_id"])) for row in held_rows}
    if len(held_rows) != 5 or len(held_keys) != 5:
        raise ValueError("G3 Gate held5 authority changed")
    paths = {
        "carrier": authority_path(
            gate, "carrier_strict250", asset_root=asset_root
        ),
        "learned_language_only": language_results.resolve(),
        "correct_full": full_results.resolve(),
        "first_final": first_final_results.resolve(),
        "same_task_other": same_task_results.resolve(),
    }
    arms = load_paired_g3_arms(paths, held_keys)
    adapter_checks, checkpoint = _g3_adapter_authority(arms, gate)
    metrics = _success_metrics(arms, held_keys)
    thresholds = gate["gate"]
    checks = {
        "full_successes": metrics["full_successes"]
        >= int(thresholds["full_successes_minimum"]),
        "breadth": metrics["breadth"] >= int(thresholds["breadth_minimum"]),
        "carrier_successes_retained": metrics["carrier_successes_retained"]
        >= int(thresholds["carrier_retained_minimum"]),
        "goal_or_long_nonzero": (
            metrics["suite_successes"]["libero_goal"] > 0
            or metrics["suite_successes"]["libero_10"] > 0
        ),
        "full_over_language": metrics["full_over_language"]
        >= int(thresholds["full_over_language_minimum"]),
        "full_over_first_final": metrics["full_over_first_final"]
        >= int(thresholds["full_over_first_final_minimum"]),
        "same_task_retention": metrics["same_task_retention"]
        >= float(thresholds["same_task_retention_minimum"]),
        **adapter_checks,
    }
    payload = {
        "schema_version": G3_GATE_SCHEMA,
        "status": "pass" if all(checks.values()) else "non_pass",
        "question": (
            "whether a frozen natural Program can drive one shared native-content "
            "attention compiler to a strong, video-necessary rank4 residual"
        ),
        "claim_boundary": (
            "held5 shared mapping Gate only; joint Writer and final validation8 "
            "performance remain unproven"
        ),
        "strict250": {
            name: {
                "path": arm["path"],
                "bytes": arm["bytes"],
                "contract_reference": arm["contract_reference"],
                "arm": arm["arm"],
                "successes": arm["successes"],
                "breadth": arm["breadth"],
            }
            for name, arm in arms.items()
        },
        "per_task": [
            {
                "suite": key[0],
                "task_id": key[1],
                **{name: arm["per_task"][key] for name, arm in arms.items()},
            }
            for key in sorted(held_keys)
        ],
        "per_suite": [
            {
                "suite": suite,
                **{
                    name: sum(
                        value
                        for (task_suite, _), value in arm["per_task"].items()
                        if task_suite == suite
                    )
                    for name, arm in arms.items()
                },
            }
            for suite in SUITES
        ],
        "metrics": metrics,
        "checks": checks,
        "paired_authority": {
            "task_state_keys": 250,
            "source_model": True,
            "tokenizer": True,
            "source_normalization": True,
            "environment_and_policy_rng": True,
        },
        "compiler_checkpoint": checkpoint,
        "single_complete_rank16": all(adapter_checks.values()),
        "shuffled_or_reversed_use": False,
    }
    output_path = output_path.resolve()
    if output_path.exists():
        if read_json(output_path) != payload:
            raise ValueError("existing G3 Gate report differs")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/pi05_ecp_shared_compiler_g3_gate_v1.json",
    )
    parser.add_argument("--asset-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--language-results", type=Path, required=True)
    parser.add_argument("--full-results", type=Path, required=True)
    parser.add_argument("--first-final-results", type=Path, required=True)
    parser.add_argument("--same-task-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_g3_gate_report(
        config_path=args.config,
        asset_root=args.asset_root.resolve(),
        language_results=args.language_results,
        full_results=args.full_results,
        first_final_results=args.first_final_results,
        same_task_results=args.same_task_results,
        output_path=args.output,
    )
    print(f"G3 Gate: {report['status']} {report['metrics']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
