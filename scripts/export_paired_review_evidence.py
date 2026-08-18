#!/usr/bin/env python3
"""Export compact, remote-safe evidence from strict paired rollout panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from ember.expert_manifold.video_schedule import VIDEO_CONDITIONS
from ember.pi05_eval.paired_metrics import (
    control_outcome_summary,
    episode_key,
    index_rows,
    paired_transition_summary,
    suite_sort_key,
    summarize_panel,
)


def _panel_argument(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("panel must be LABEL=RESULTS_JSON")
    return label, Path(raw_path)


def _comparison_argument(value: str) -> tuple[str, str]:
    left, separator, right = value.partition(":")
    if not separator or not left or not right:
        raise argparse.ArgumentTypeError("comparison must be LEFT:RIGHT")
    return left, right


def _load_panel(path: Path, *, allow_video_controls: bool) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    condition = result.get("adapter", {}).get("video_condition")
    if (
        result.get("mode") != "formal"
        or result.get("role") != "validation"
        or (
            condition not in VIDEO_CONDITIONS
            if allow_video_controls
            else condition != "correct"
        )
        or result.get("paired_control", {}).get("git", {}).get("dirty_paths") != []
    ):
        expected = "video-control" if allow_video_controls else "correct-video"
        raise ValueError(f"not a clean formal {expected} panel: {path}")
    rows = list(result.get("rows", []))
    indexed = index_rows(rows)
    if len(indexed) != 400:
        raise ValueError(f"panel does not contain 400 unique rows: {path}")
    expected_states = set(range(50))
    tasks: dict[tuple[str, int], set[int]] = {}
    for suite, task_id, state in indexed:
        tasks.setdefault((suite, task_id), set()).add(state)
    if len(tasks) != 8 or any(states != expected_states for states in tasks.values()):
        raise ValueError(f"panel is not the canonical 8x50 shape: {path}")
    return result


def _rng_prefix_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_seeds = list(left.get("policy_noise_seeds", []))
    right_seeds = list(right.get("policy_noise_seeds", []))
    common = min(len(left_seeds), len(right_seeds))
    return left_seeds[:common] == right_seeds[:common]


def _pairing_audit(
    indexes: Mapping[str, Mapping[tuple[str, int, int], Mapping[str, Any]]]
) -> dict[str, Any]:
    labels = list(indexes)
    reference = indexes[labels[0]]
    expected_keys = set(reference)
    mismatches = {
        "episode_keys": 0,
        "environment_seed": 0,
        "policy_seed_root": 0,
        "policy_noise_common_prefix": 0,
        "teacher_reference_videos": 0,
    }
    for label in labels[1:]:
        candidate = indexes[label]
        if set(candidate) != expected_keys:
            mismatches["episode_keys"] += 1
            continue
        for key in expected_keys:
            left, right = reference[key], candidate[key]
            mismatches["environment_seed"] += left.get("env_seed") != right.get(
                "env_seed"
            )
            mismatches["policy_seed_root"] += left.get(
                "policy_seed_root"
            ) != right.get("policy_seed_root")
            mismatches["policy_noise_common_prefix"] += not _rng_prefix_matches(
                left, right
            )
            mismatches["teacher_reference_videos"] += left.get("writer", {}).get(
                "teacher_reference_demo_indices"
            ) != right.get("writer", {}).get("teacher_reference_demo_indices")
    return {
        "reference_panel": labels[0],
        "compared_panel_count": len(labels),
        "mismatch_counts": mismatches,
        "strict_pairing_verified": not any(mismatches.values()),
        "rng_note": (
            "Noise sequences are compared by common prefix because successful "
            "episodes terminate early and therefore record fewer replans."
        ),
    }


def _provenance(result: Mapping[str, Any]) -> dict[str, Any]:
    adapter = result["adapter"]
    paired = result["paired_control"]
    asset = adapter.get("writer_asset", {})
    checkpoint = asset.get("checkpoint_manifest", {})
    training = asset.get("training_run_contract", {})
    checkpoint_body = json.loads(
        Path(str(checkpoint["path"])).read_text(encoding="utf-8")
    )
    training_body = json.loads(
        Path(str(training["path"])).read_text(encoding="utf-8")
    )
    return {
        "evaluation_contract_reference": result.get("contract_reference"),
        "evaluation_git_commit": paired.get("git", {}).get("commit"),
        "evaluation_git_dirty_paths": paired.get("git", {}).get("dirty_paths"),
        "training_git_commit": training_body.get("git", {}).get("commit"),
        "method_arm": result.get("arm"),
        "writer_kind": adapter.get("kind"),
        "writer_architecture": asset.get("architecture"),
        "writer_checkpoint_kind": asset.get("kind"),
        "writer_checkpoint_macro": asset.get("method_macro"),
        "writer_checkpoint_reference": asset.get("reference"),
        "checkpoint_manifest": {
            "schema": checkpoint_body.get("schema_version"),
            "bytes": checkpoint.get("bytes"),
            "world_size": checkpoint_body.get("world_size"),
            "next_macro": checkpoint_body.get("next_macro"),
            "next_cycle": checkpoint_body.get("next_cycle"),
            "writer_state_bytes": checkpoint_body.get("files", {})
            .get("writer.safetensors", {})
            .get("bytes"),
        },
        "training_contract": {
            "schema": training_body.get("schema_version"),
            "bytes": training.get("bytes"),
            "mode": training_body.get("mode"),
            "writer": training_body.get("writer"),
        },
        "config_schema": adapter.get("config", {}).get("schema"),
        "lora_contract": adapter.get("lora_contract"),
        "video_schedule": adapter.get("video_schedule"),
        "information_wall": adapter.get("information_wall"),
    }


def _remote_rows(
    indexes: Mapping[str, Mapping[tuple[str, int, int], Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    labels = list(indexes)
    reference = indexes[labels[0]]
    output = []
    for key in sorted(
        reference, key=lambda item: (*suite_sort_key(item[0]), item[1], item[2])
    ):
        row = reference[key]
        writer = row.get("writer", {})
        output.append(
            {
                "suite": key[0],
                "task_id": key[1],
                "init_state_id": key[2],
                "env_seed": row.get("env_seed"),
                "policy_seed_root": row.get("policy_seed_root"),
                "policy_noise_first_seed": next(
                    iter(row.get("policy_noise_seeds", [])), None
                ),
                "teacher_reference_demo_indices": writer.get(
                    "teacher_reference_demo_indices"
                ),
                "teacher_video_order_seeds": writer.get("teacher_video_order_seeds"),
                "teacher_video_selection_seed": writer.get(
                    "teacher_video_selection_seed"
                ),
                "success": {
                    label: bool(indexes[label][key]["success"]) for label in labels
                },
                "video_evidence": {
                    label: {
                        "condition": indexes[label][key]
                        .get("writer", {})
                        .get("condition"),
                        "teacher_demo_indices": indexes[label][key]
                        .get("writer", {})
                        .get("teacher_demo_indices"),
                        "video_suite": indexes[label][key]
                        .get("writer", {})
                        .get("video_suite"),
                        "video_task_id": indexes[label][key]
                        .get("writer", {})
                        .get("video_task_id"),
                        "teacher_video_frames_used": indexes[label][key]
                        .get("writer", {})
                        .get("teacher_video_frames_used"),
                    }
                    for label in labels
                },
                "policy_noise_seed_count": {
                    label: len(indexes[label][key].get("policy_noise_seeds", []))
                    for label in labels
                },
            }
        )
    return output


def build_evidence(
    panels: Mapping[str, Mapping[str, Any]],
    comparisons: list[tuple[str, str]],
    *,
    reference_panel: str | None = None,
) -> dict[str, Any]:
    indexes = {label: index_rows(result["rows"]) for label, result in panels.items()}
    for left, right in comparisons:
        if left not in panels or right not in panels:
            raise ValueError(f"comparison references an unknown panel: {left}:{right}")
    if reference_panel is not None and reference_panel not in panels:
        raise ValueError(f"unknown reference panel: {reference_panel}")
    evidence = {
        "schema_version": "ember_external_review_paired_evidence_v1",
        "panel_order": list(panels),
        "pairing_audit": _pairing_audit(indexes),
        "provenance": {
            label: _provenance(result) for label, result in panels.items()
        },
        "panels": {
            label: summarize_panel(result["rows"])
            for label, result in panels.items()
        },
        "comparisons": {
            f"{left}_to_{right}": paired_transition_summary(
                panels[left]["rows"], panels[right]["rows"]
            )
            for left, right in comparisons
        },
        "rows": _remote_rows(indexes),
    }
    if reference_panel is not None:
        evidence["control_comparisons_to_reference"] = {
            label: control_outcome_summary(
                panels[reference_panel]["rows"], result["rows"]
            )
            for label, result in panels.items()
            if label != reference_panel
        }
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", action="append", type=_panel_argument, required=True)
    parser.add_argument(
        "--comparison", action="append", type=_comparison_argument, default=[]
    )
    parser.add_argument(
        "--allow-video-controls",
        action="store_true",
        help="accept any registered formal video condition instead of correct only",
    )
    parser.add_argument(
        "--reference-panel",
        help="emit correct-vs-control outcome summaries against this panel label",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panels: dict[str, dict[str, Any]] = {}
    for label, path in args.panel:
        if label in panels:
            raise ValueError(f"duplicate panel label: {label}")
        panels[label] = _load_panel(
            path, allow_video_controls=args.allow_video_controls
        )
        if args.allow_video_controls:
            condition = panels[label]["adapter"]["video_condition"]
            if label != condition:
                raise ValueError(
                    "video-control panel labels must equal their video conditions: "
                    f"{label} != {condition}"
                )
    evidence = build_evidence(
        panels,
        list(args.comparison),
        reference_panel=args.reference_panel,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
