#!/usr/bin/env python3
"""Adjudicate the fixed 100-row ECP recovery-teacher Gate A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ember.ecp.process_meta import ProcessMetaError, load_process_meta_authority
from ember.pi05_assets import write_json_atomic


REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_SCHEMA = "ember_ecp_process_meta_privileged_episode_v1"
PUBLIC_SCHEMA = "ember_ecp_action_hidden_video_v1"
A3_BASELINE_ROOT = (
    REPO_ROOT
    / "runs/outputs/pi05_ecp_process_separate_plates_gate_a3_4bf5039_gpu01p123457_20260824"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _authority(manifest: Path) -> Any:
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    source_root = REPO_ROOT / str(raw["source_policy_authority"])
    source_contract = json.loads(
        (source_root / "run_contract.json").read_text(encoding="utf-8")
    )
    return load_process_meta_authority(
        manifest,
        repo_root=REPO_ROOT,
        libero_init_root=Path(source_contract["libero_paths"]["init_states"]),
    )


def _worker_rows(root: Path) -> list[dict[str, Any]]:
    summaries = sorted((root / "shards").glob("*.json"))
    if not summaries:
        summaries = sorted((root / "workers").glob("*.json"))
    if not summaries:
        raise ProcessMetaError("recovery Gate has no completed worker summaries")
    rows = []
    for path in summaries:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("schema_version")
            != "ember_ecp_process_meta_teacher_collection_v1"
        ):
            raise ProcessMetaError("recovery Gate worker schema changed")
        rows.extend(value["rows"])
    return rows


def _public_video(
    path: Path, *, language: str, action_count: int, frame_stride: int
) -> bool:
    if not path.is_file():
        return False
    with np.load(path, allow_pickle=False) as value:
        if set(value.files) != {
            "schema_version",
            "language",
            "camera1",
            "camera2",
            "source_steps",
            "model_frame_stride",
        }:
            return False
        return bool(
            str(value["schema_version"].item()) == PUBLIC_SCHEMA
            and str(value["language"].item()) == language
            and int(value["model_frame_stride"].item()) == frame_stride
            and value["camera1"].shape == value["camera2"].shape
            and value["camera1"].shape[0] == action_count + 1
            and np.array_equal(
                value["source_steps"], np.arange(action_count + 1, dtype=np.int32)
            )
        )


def _inspect_row(root: Path, authority: Any, row: dict[str, Any]) -> dict[str, Any]:
    variant = authority.family.variant(str(row["variant_name"]))
    expected = authority.variant_phase_experts[variant.name]
    ledger_path = root / str(row["privileged_ledger"])
    ledger = torch.load(ledger_path, map_location="cpu", weights_only=False)
    phases = tuple(str(value) for value in ledger["replan_phase_keys"])
    task_ids = tuple(int(value) for value in ledger["replan_teacher_task_ids"])
    roles = tuple(str(value) for value in ledger["replan_teacher_roles"])
    checkpoints = tuple(str(value) for value in ledger["replan_teacher_checkpoints"])
    route_mismatch = int(
        ledger.get("schema_version") != LEDGER_SCHEMA
        or ledger.get("privileged_teacher_kind") != "variant_phase_recovery_rank16_lora"
        or not phases
        or not (len(phases) == len(task_ids) == len(roles) == len(checkpoints))
        or any(
            phase not in expected
            or task_id != expected[phase].task_id
            or role != expected[phase].role
            or checkpoint != str(expected[phase].checkpoint)
            for phase, task_id, role, checkpoint in zip(
                phases, task_ids, roles, checkpoints, strict=True
            )
        )
    )
    drops = {
        str(name): tuple(int(step) for step in steps)
        for name, steps in ledger["post_completion_drop_steps"].items()
    }
    completion = {
        str(name): int(step) for name, step in ledger["completion_steps"].items()
    }
    environment_success = bool(ledger["environment_success"])
    strict_success = bool(ledger["success"])
    success_mismatch = int(
        strict_success != bool(row["success"])
        or strict_success
        != bool(
            environment_success
            and not drops[variant.required_order[0]]
            and tuple(completion) == variant.required_order
            and completion[variant.required_order[0]]
            < completion[variant.required_order[1]]
            and all(phase in phases for phase in variant.required_order)
            and all(bool(value) for value in ledger["predicate_values"].values())
        )
    )
    public = ledger.get("public_video")
    public_mismatch = int(
        (strict_success and not isinstance(public, str))
        or (not strict_success and public is not None)
        or row.get("public_video") != public
        or (
            strict_success
            and not _public_video(
                root / str(public),
                language=authority.family.exact_language,
                action_count=int(ledger["teacher_actions"].shape[0]),
                frame_stride=int(authority.rollout["frame_stride"]),
            )
        )
    )
    return {
        "variant_name": variant.name,
        "state_id": int(row["state_id"]),
        "success": strict_success,
        "environment_success": environment_success,
        "first_event_completed": variant.required_order[0] in completion,
        "second_event_completed": variant.required_order[1] in completion,
        "first_event_dropped": bool(drops[variant.required_order[0]]),
        "invalid": bool(ledger["invalid"]),
        "route_mismatch": route_mismatch,
        "success_mismatch": success_mismatch,
        "public_mismatch": public_mismatch,
        "policy_noise_seeds": tuple(
            int(value) for value in ledger["policy_noise_seeds"]
        ),
    }


def _a3_success_keys(authority: Any) -> set[tuple[str, int]]:
    result = set()
    for variant in authority.family.variants:
        for state_id in authority.family.init_state_ids:
            episode_id = (
                f"{authority.family.family_id}-{variant.name}-state{state_id:03d}"
            )
            ledger = torch.load(
                A3_BASELINE_ROOT / "privileged_ledgers" / f"{episode_id}.pt",
                map_location="cpu",
                weights_only=False,
            )
            if (
                ledger.get("variant_name") != variant.name
                or int(ledger.get("state_id", -1)) != state_id
            ):
                raise ProcessMetaError("A3 matched-row baseline authority changed")
            if bool(ledger["success"]):
                result.add((variant.name, state_id))
    return result


def adjudicate(manifest: Path, root: Path) -> dict[str, Any]:
    authority = _authority(manifest)
    if authority.privileged_teacher_kind != "variant_phase_recovery_rank16_lora":
        raise ProcessMetaError("recovery Gate manifest has the wrong teacher kind")
    inspected = [_inspect_row(root, authority, row) for row in _worker_rows(root)]
    variants = tuple(variant.name for variant in authority.family.variants)
    by_key = {(row["variant_name"], row["state_id"]): row for row in inspected}
    expected_keys = {
        (variant, state_id)
        for variant in variants
        for state_id in authority.family.init_state_ids
    }
    completeness_mismatch = len(expected_keys.symmetric_difference(by_key)) + (
        len(inspected) - len(by_key)
    )
    pairing_mismatch = 0
    for state_id in authority.family.init_state_ids:
        left = by_key.get((variants[0], state_id))
        right = by_key.get((variants[1], state_id))
        if left is None or right is None:
            continue
        count = min(len(left["policy_noise_seeds"]), len(right["policy_noise_seeds"]))
        pairing_mismatch += int(
            left["policy_noise_seeds"][:count] != right["policy_noise_seeds"][:count]
        )
    successes = {
        variant: sum(
            row["success"] for row in inspected if row["variant_name"] == variant
        )
        for variant in variants
    }
    stage_results = {
        variant: {
            "first_event_completed": sum(
                row["first_event_completed"]
                for row in inspected
                if row["variant_name"] == variant
            ),
            "second_event_completed": sum(
                row["second_event_completed"]
                for row in inspected
                if row["variant_name"] == variant
            ),
            "environment_successes": sum(
                row["environment_success"]
                for row in inspected
                if row["variant_name"] == variant
            ),
            "first_event_drops": sum(
                row["first_event_dropped"]
                for row in inspected
                if row["variant_name"] == variant
            ),
        }
        for variant in variants
    }
    baseline_success = _a3_success_keys(authority)
    current_success = {
        (row["variant_name"], row["state_id"]) for row in inspected if row["success"]
    }
    matched_change = {
        "baseline_root": str(A3_BASELINE_ROOT),
        "baseline_successes": len(baseline_success),
        "retained": len(current_success & baseline_success),
        "gained": len(current_success - baseline_success),
        "lost": len(baseline_success - current_success),
        "by_variant": {
            variant: {
                "baseline_successes": sum(
                    key[0] == variant for key in baseline_success
                ),
                "retained": sum(
                    key[0] == variant for key in current_success & baseline_success
                ),
                "gained": sum(
                    key[0] == variant for key in current_success - baseline_success
                ),
                "lost": sum(
                    key[0] == variant for key in baseline_success - current_success
                ),
            }
            for variant in variants
        },
    }
    checks = {
        "each_direction_at_least_20": all(value >= 20 for value in successes.values()),
        "total_at_least_50": sum(successes.values()) >= 50,
        "completeness_mismatch": completeness_mismatch,
        "invalid": sum(row["invalid"] for row in inspected),
        "route_mismatch": sum(row["route_mismatch"] for row in inspected),
        "success_mismatch": sum(row["success_mismatch"] for row in inspected),
        "public_mismatch": sum(row["public_mismatch"] for row in inspected),
        "pairing_mismatch": pairing_mismatch,
    }
    passed = bool(
        checks["each_direction_at_least_20"]
        and checks["total_at_least_50"]
        and all(
            checks[name] == 0
            for name in (
                "completeness_mismatch",
                "invalid",
                "route_mismatch",
                "success_mismatch",
                "public_mismatch",
                "pairing_mismatch",
            )
        )
    )
    return {
        "schema_version": "ember_ecp_composite_recovery_teacher_gate_v1",
        "status": "pass" if passed else "non_pass",
        "manifest": str(manifest),
        "collection_root": str(root),
        "episodes": len(inspected),
        "successes": successes,
        "stage_results": stage_results,
        "matched_a3_change": matched_change,
        "total_successes": sum(successes.values()),
        "environment_successes": sum(row["environment_success"] for row in inspected),
        "checks": checks,
        "content_hash_policy": "disabled_by_owner",
    }


def main() -> None:
    args = build_parser().parse_args()
    result = adjudicate(args.manifest.resolve(), args.collection_root.resolve())
    write_json_atomic(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
