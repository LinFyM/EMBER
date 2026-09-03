#!/usr/bin/env python3
"""Audit whether the expanded positive-only Writer split has useful contrasts."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "ember_ecp_policy_response_writer_factorial_coverage_v1"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git_state(repository: Path) -> dict[str, Any]:
    def run(*command: str) -> str:
        return subprocess.run(
            command,
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "path": str(repository),
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def task_rows(
    protocol: Mapping[str, Any],
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    base: Mapping[str, Any],
) -> list[dict[str, Any]]:
    held_fold = next(
        row
        for row in protocol["task_level_cross_validation"]["folds"]
        if int(row["fold"]) == 0
    )
    held_meta = set(map(int, held_fold["task_ids"]))
    source_by_id = {
        int(row["task_index"]): row for row in source["tasks"]
    }
    rows: list[dict[str, Any]] = []
    for domain_id in map(int, protocol["active_source_task_ids"]):
        row = source_by_id[domain_id]
        rows.append(
            {
                "authority_id": len(rows),
                "domain": "meta",
                "domain_task_id": domain_id,
                "fold_role": "held" if domain_id in held_meta else "fit",
                "suite": "libero_90",
                "language": str(row["language"]),
            }
        )

    target_fit = set(map(int, base["fold"]["target_fit_task_ids"]))
    target_held = set(map(int, base["fold"]["target_held_task_ids"]))
    selected = target_fit | target_held
    target_rows = sorted(
        (
            row
            for row in target["tasks"]
            if row["split_role"] == "train"
            and int(row["global_task_id"]) in selected
        ),
        key=lambda row: int(row["global_task_id"]),
    )
    for row in target_rows:
        domain_id = int(row["global_task_id"])
        rows.append(
            {
                "authority_id": len(rows),
                "domain": "target",
                "domain_task_id": domain_id,
                "fold_role": "held" if domain_id in target_held else "fit",
                "suite": str(row["suite"]),
                "language": str(row["language"]),
            }
        )
    if len(rows) != 95 or sum(row["domain"] == "meta" for row in rows) != 71:
        raise ValueError("95-task authority changed")
    return rows


def split_sets(config: Mapping[str, Any]) -> dict[str, set[int]]:
    split = config["task_split"]
    names = (
        "gradient_meta",
        "gradient_target",
        "true_task_held_meta",
        "true_task_held_target",
    )
    result = {name: set(map(int, split[name])) for name in names}
    flat = [task for name in names for task in result[name]]
    if len(flat) != len(set(flat)):
        raise ValueError("Writer audit split overlaps")
    return result


def exact_language_groups(
    rows: Sequence[Mapping[str, Any]], gradient: set[int]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["domain"] == "meta":
            grouped[str(row["language"])].append(row)
    result = []
    for language, members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        authority_ids = [int(row["authority_id"]) for row in members]
        fit = [task for task in authority_ids if task in gradient]
        held = [task for task in authority_ids if task not in gradient]
        result.append(
            {
                "language": language,
                "authority_ids": authority_ids,
                "domain_task_ids": [int(row["domain_task_id"]) for row in members],
                "gradient_authority_ids": fit,
                "non_gradient_authority_ids": held,
                "has_train_pair": len(fit) >= 2,
                "has_gradient_to_held_bridge": bool(fit and held),
            }
        )
    return result


def contrast_coverage(
    protocol: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    gradient: set[int],
) -> list[dict[str, Any]]:
    authority = {
        int(row["domain_task_id"]): int(row["authority_id"])
        for row in rows
        if row["domain"] == "meta"
    }
    result = []
    for family in protocol["process_contrast_groups"]:
        train_pairs = 0
        bridges = 0
        groups = []
        for domain_ids in family["task_groups"]:
            ids = [authority[int(task)] for task in domain_ids]
            fit = [task for task in ids if task in gradient]
            held = [task for task in ids if task not in gradient]
            train_pairs += len(fit) >= 2
            bridges += bool(fit and held)
            groups.append(
                {
                    "domain_task_ids": list(map(int, domain_ids)),
                    "gradient_authority_ids": fit,
                    "non_gradient_authority_ids": held,
                }
            )
        result.append(
            {
                "kind": str(family["kind"]),
                "purpose": str(family["purpose"]),
                "group_count": len(groups),
                "groups_with_train_pair": train_pairs,
                "gradient_to_held_bridges": bridges,
                "groups": groups,
            }
        )
    return result


def held_checks(
    rows: Sequence[Mapping[str, Any]], split: Mapping[str, set[int]]
) -> list[dict[str, Any]]:
    by_id = {int(row["authority_id"]): row for row in rows}
    gradient = split["gradient_meta"] | split["gradient_target"]
    true_meta, = split["true_task_held_meta"]
    true_target, = split["true_task_held_target"]
    held_target = {
        int(row["domain_task_id"]): int(row["authority_id"])
        for row in rows
        if row["domain"] == "target" and row["fold_role"] == "held"
    }

    def matching(pool: set[int], predicate: Any) -> list[int]:
        return [
            task
            for task in sorted(pool)
            if predicate(str(by_id[task]["language"]))
        ]

    meta_peer = matching(
        split["gradient_meta"],
        lambda language: language == by_id[true_meta]["language"],
    )
    spatial = matching(
        split["gradient_target"],
        lambda language: language.startswith("pick up the black bowl")
        and language.endswith("and place it on the plate"),
    )
    object_peers = matching(
        split["gradient_target"],
        lambda language: language.endswith("and place it in the basket"),
    )
    long_components = {
        language: matching(
            split["gradient_meta"], lambda candidate, value=language: candidate == value
        )
        for language in (
            "put the white mug on the plate",
            "put the chocolate pudding to the right of the plate",
        )
    }
    long_components["ordered multi-action target tasks"] = matching(
        split["gradient_target"],
        lambda language: " and " in language or "both" in language,
    )
    push = matching(gradient, lambda language: language.startswith("push "))
    return [
        {
            "name": "true_held_meta_exact_language_cross_scene",
            "held_authority_ids": [true_meta],
            "gradient_peers": meta_peer,
            "covered": bool(meta_peer),
        },
        {
            "name": "true_held_target_spatial_relation_recombination",
            "held_authority_ids": [true_target],
            "gradient_peers": spatial,
            "covered": len(spatial) >= 3,
        },
        {
            "name": "held5_spatial_relation_family",
            "held_authority_ids": [held_target[0], held_target[9]],
            "gradient_peers": spatial,
            "covered": len(spatial) >= 3,
        },
        {
            "name": "held5_object_identity_recombination",
            "held_authority_ids": [held_target[18]],
            "gradient_peers": object_peers,
            "covered": len(object_peers) >= 4,
        },
        {
            "name": "held5_long_seen_components_new_ordered_composition",
            "held_authority_ids": [held_target[36]],
            "gradient_component_peers": long_components,
            "covered": all(long_components.values()),
        },
        {
            "name": "held5_goal_push_procedure",
            "held_authority_ids": [held_target[25]],
            "gradient_peers": push,
            "covered": bool(push),
            "coverage_gap": "no Writer-gradient task begins with push",
        },
    ]


def exposure_ranges(
    split: Mapping[str, set[int]], config: Mapping[str, Any]
) -> dict[str, Any]:
    cell = config["optimization"]["shared"]
    groups = (
        tuple(sorted(split["gradient_meta"])),
        tuple(sorted(split["gradient_target"])),
    )
    configured = cell["tasks_per_update_by_role"]
    counts = (int(configured["meta"]), int(configured["target"]))
    if (
        sum(counts) != int(cell["global_tasks_per_update"])
        or any(count < 0 for count in counts)
        or any(count > len(group) for count, group in zip(counts, groups, strict=True))
    ):
        raise ValueError("invalid per-role task counts")
    seed = int(config["optimization"]["seed"])
    checkpoints = [
        int(cell["warmup_updates"]) + int(step)
        for step in cell["checkpoint_effective_updates"]
    ]
    result = {}
    for checkpoint in checkpoints:
        exposures: Counter[int] = Counter()
        for step in range(checkpoint):
            for index, (group, count) in enumerate(zip(groups, counts, strict=True)):
                ordered = list(group)
                random.Random(seed + 104729 + index * 25609).shuffle(ordered)
                offset = (step * count) % len(ordered)
                exposures.update(
                    ordered[(offset + position) % len(ordered)]
                    for position in range(count)
                )
        meta_values = [exposures[task] for task in groups[0]]
        target_values = [exposures[task] for task in groups[1]]
        result[str(checkpoint)] = {
            "meta": [min(meta_values), max(meta_values)],
            "target": [min(target_values), max(target_values)],
            "all_tasks": [
                min((*meta_values, *target_values)),
                max((*meta_values, *target_values)),
            ],
        }
    return {
        "global_tasks_per_update": int(cell["global_tasks_per_update"]),
        "tasks_per_update_by_role": {"meta": counts[0], "target": counts[1]},
        "warmup_optimizer_steps": int(cell["warmup_updates"]),
        "checkpoint_optimizer_steps": checkpoints,
        "checkpoint_post_warmup_effective_steps": list(
            map(int, cell["checkpoint_effective_updates"])
        ),
        "ranges": result,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    repository = args.repository.resolve()
    protocol = read_json(args.meta_protocol.resolve())
    source = read_json(args.source_manifest.resolve())
    target = read_json(args.target_manifest.resolve())
    base = read_json(args.base_config.resolve())
    writer = read_json(args.writer_config.resolve())
    rows = task_rows(protocol, source, target, base)
    split = split_sets(writer)
    by_id = {int(row["authority_id"]): row for row in rows}
    expected_meta = {
        task
        for task, row in by_id.items()
        if row["domain"] == "meta" and row["fold_role"] == "fit"
    } - split["true_task_held_meta"]
    expected_target = {
        task
        for task, row in by_id.items()
        if row["domain"] == "target" and row["fold_role"] == "fit"
    } - split["true_task_held_target"]
    if (
        split["gradient_meta"] != expected_meta
        or split["gradient_target"] != expected_target
        or len(split["true_task_held_meta"]) != 1
        or len(split["true_task_held_target"]) != 1
    ):
        raise ValueError("expanded Writer split is not all eligible fit tasks")

    gradient = split["gradient_meta"] | split["gradient_target"]
    exact = exact_language_groups(rows, gradient)
    contrasts = contrast_coverage(protocol, rows, gradient)
    checks = held_checks(rows, split)
    return {
        "schema_version": SCHEMA,
        "status": "complete",
        "created_unix": time.time(),
        "repository": git_state(repository),
        "information_wall": {
            "reads": "metadata_language_and_manual_protocol_groups_only",
            "video_action_state_reward_model_or_outcome_reads": 0,
            "selection_uses_outcomes": False,
        },
        "task_inventory": {
            "authority_tasks": len(rows),
            "meta_tasks": sum(row["domain"] == "meta" for row in rows),
            "target_train_tasks": sum(row["domain"] == "target" for row in rows),
            "gradient_meta": len(split["gradient_meta"]),
            "gradient_target": len(split["gradient_target"]),
            "true_task_held_meta": sorted(split["true_task_held_meta"]),
            "true_task_held_target": sorted(split["true_task_held_target"]),
        },
        "exact_language_cross_scene": {
            "group_count": len(exact),
            "task_count": sum(len(row["authority_ids"]) for row in exact),
            "groups_with_train_pair": sum(row["has_train_pair"] for row in exact),
            "gradient_to_held_bridges": sum(
                row["has_gradient_to_held_bridge"] for row in exact
            ),
            "groups": exact,
        },
        "protocol_contrast_coverage": contrasts,
        "held_recombination_checks": checks,
        "exposure_contract": exposure_ranges(split, writer),
        "decision": {
            "factorial_training_signal_exists": True,
            "metadata_alone_proves_video_dependent_optimal_adapter": False,
            "expanded_mapping_experiment_is_identifiable_enough_to_run": True,
            "known_coverage_gap": "held5 Goal push procedure",
            "next_experiment": (
                "full K1 event-measure Writer on all 55 meta-fit and 18 "
                "target-fit tasks with near task-equal 9/3 sampling"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--meta-protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--writer-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    report = analyze(args)
    write_json(output / "report.json", report)
    write_json(
        output / "completion.json",
        {
            "schema_version": SCHEMA,
            "status": "complete",
            "report": "report.json",
            "task_count": report["task_inventory"]["authority_tasks"],
            "gradient_task_count": (
                report["task_inventory"]["gradient_meta"]
                + report["task_inventory"]["gradient_target"]
            ),
        },
    )
    print(json.dumps(report["decision"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
