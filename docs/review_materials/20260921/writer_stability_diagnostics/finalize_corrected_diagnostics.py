#!/usr/bin/env python3
"""Validate and export the corrected four-arm terminal diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

import torch


BRANCHES = ("O-H", "O-L", "N-H", "N-L")
EXPECTED_TASKS = {3, 11, 26, 31}
EXPECTED_TRAIN_TASKS = {
    0, 1, 2, 4, 5, 7,
    12, 13, 14, 15, 17, 19,
    20, 21, 22, 25, 28, 29,
    32, 34, 35, 36, 37, 38,
    42, 43, 51, 55, 56, 62, 64, 73, 95, 96, 97, 101,
}
FULL_CONDITIONS = {(3, 0), (31, 0)}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refuse empty export: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scrub(row: dict[str, str], study: Path, repo: Path) -> dict[str, str]:
    return {
        key: value.replace(str(study), "<STUDY>").replace(str(repo), "<REPO>")
        for key, value in row.items()
    }


def validate_branch(study: Path, branch: str) -> dict[str, list[dict[str, str]]]:
    slug = branch.replace("-", "_")
    exit_path = study / "launch" / f"corrected_{slug}.exit"
    if exit_path.read_text().strip() != "0":
        raise ValueError(f"branch did not exit successfully: {branch}")
    tables = {
        "training_steps": read_csv(study / "parts" / f"shadow_training_steps_{branch}.csv"),
        "task_events": read_csv(study / "parts" / f"shadow_task_exposures_{branch}.csv"),
        "terminal_probes": read_csv(study / "parts" / f"shadow_probe_rows_{branch}.csv"),
        "terminal_rollouts": read_csv(study / "parts" / f"shadow_rollout_rows_{branch}_036.csv"),
    }
    expected = {"training_steps": 36, "task_events": 144, "terminal_probes": 36, "terminal_rollouts": 16}
    for name, count in expected.items():
        if len(tables[name]) != count:
            raise ValueError(f"{branch} {name} count differs: {len(tables[name])} != {count}")
    steps = [int(row["local_step"]) for row in tables["training_steps"]]
    if steps != list(range(1, 37)):
        raise ValueError(f"{branch} training steps are not exactly 1..36")
    events = tables["task_events"]
    for offset in range(0, 144, 36):
        if len({int(row["global_task_id"]) for row in events[offset:offset + 36]}) != 36:
            raise ValueError(f"{branch} task events are not balanced by nine-update cycle")
    probes = tables["terminal_probes"]
    if (
        {int(row["global_task_id"]) for row in probes} != EXPECTED_TRAIN_TASKS
        or any(int(row["teacher_demo"]) != 46 or int(row["query_demo"]) != 47 for row in probes)
        or any(float(row["query_fraction"]) != 0.25 for row in probes)
    ):
        raise ValueError(f"{branch} terminal compact probe contract changed")
    rollouts = tables["terminal_rollouts"]
    keys = {(int(row["global_task_id"]), int(row["init_state_id"])) for row in rollouts}
    if keys != {(task, state) for task in EXPECTED_TASKS for state in range(4)}:
        raise ValueError(f"{branch} terminal rollout panel changed")
    completion = json.loads((study / "shadow" / branch / "completion.json").read_text())
    if completion.get("status") != "complete" or completion.get("local_updates") != 36:
        raise ValueError(f"{branch} completion record changed")
    return tables


def trajectory_manifest(study: Path) -> list[dict[str, Any]]:
    rows = []
    for branch in BRANCHES:
        root = study / "rollouts" / "e3" / branch / "local_036" / "trajectories"
        paths = sorted(root.glob("*.pt"))
        if len(paths) != 16:
            raise ValueError(f"{branch} trajectory count differs: {len(paths)} != 16")
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            global_task = {
                "libero_spatial": 0,
                "libero_object": 10,
                "libero_goal": 20,
                "libero_10": 30,
            }[payload["suite"]] + int(payload["task_id"])
            key = (global_task, int(payload["init_state_id"]))
            expected_level = "full" if key in FULL_CONDITIONS else "compact"
            if payload.get("capture_level") != expected_level:
                raise ValueError(f"{branch} trajectory capture level changed: {key}")
            has_images = "observations" in payload
            if has_images != (expected_level == "full"):
                raise ValueError(f"{branch} trajectory image retention changed: {key}")
            if len(payload["states"]) != len(payload["action_chunks"]):
                raise ValueError(f"{branch} trajectory state/action replans differ: {key}")
            rows.append({
                "branch": branch,
                "global_task_id": global_task,
                "suite": payload["suite"],
                "task_id": int(payload["task_id"]),
                "init_state_id": int(payload["init_state_id"]),
                "success": bool(payload["success"]),
                "steps": int(payload["steps"]),
                "capture_level": expected_level,
                "has_images": has_images,
                "replans": len(payload["states"]),
                "local_payload_bytes": path.stat().st_size,
                "local_payload": f"<STUDY>/rollouts/e3/{branch}/local_036/trajectories/{path.name}",
            })
            del payload
    if sum(row["has_images"] for row in rows) != 8:
        raise ValueError("full-image terminal trajectory count differs from eight")
    return rows


def export(study: Path, repo: Path, output: Path) -> None:
    tables = {name: [] for name in ("training_steps", "task_events", "terminal_probes", "terminal_rollouts")}
    for branch in BRANCHES:
        branch_tables = validate_branch(study, branch)
        for name, rows in branch_tables.items():
            tables[name].extend(scrub(row, study, repo) for row in rows)
    trajectories = trajectory_manifest(study)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for name, rows in tables.items():
        write_csv(temporary / f"{name}.csv", rows)
    write_csv(temporary / "terminal_trajectory_manifest.csv", trajectories)
    source = study / "analysis" / "e1_lightweight_trajectories"
    shutil.copytree(source, temporary / "e1_lightweight_trajectories")
    launch = json.loads((study / "launch" / "corrected_launch_contract.json").read_text())
    launch["output_root"] = "<STUDY>"
    launch["runtime_workspace"] = "<REPO>/.codex/tmp/writer-stability-runtime"
    launch["correction_of"] = "<STUDY>/launch/launch_contract.json"
    launch["existing_e1_export"]["path"] = "<STUDY>/analysis/e1_lightweight_trajectories"
    (temporary / "run_contract.json").write_text(json.dumps(launch, indent=2, sort_keys=True) + "\n")
    branch_successes = {
        branch: sum(row["success"] == "True" for row in tables["terminal_rollouts"] if row["branch"] == branch)
        for branch in BRANCHES
    }
    probe_fields = ("full_mse", "tau1_first5_mse", "inference_first5_mse", "action_delta_mse_vs_parent")
    branch_probe_means = {}
    for branch in BRANCHES:
        rows = [row for row in tables["terminal_probes"] if row["branch"] == branch]
        branch_probe_means[branch] = {
            field: sum(float(row[field]) for row in rows) / len(rows) for field in probe_fields
        }
    completion = {
        "schema_version": "ember_writer_stability_corrected_export_v1",
        "status": "complete",
        "branches": list(BRANCHES),
        "training_steps": len(tables["training_steps"]),
        "task_events": len(tables["task_events"]),
        "terminal_probes": len(tables["terminal_probes"]),
        "terminal_rollouts": len(tables["terminal_rollouts"]),
        "terminal_trajectory_rows": len(trajectories),
        "terminal_full_image_trajectories": sum(row["has_images"] for row in trajectories),
        "existing_e1_lightweight_conditions": 16,
        "branch_terminal_successes_of_16": branch_successes,
        "branch_terminal_probe_means": branch_probe_means,
        "checkpoint_selection_use": False,
        "test_use": False,
        "causal_attribution": "not_performed_by_execution_agent",
    }
    (temporary / "completion.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Writer修订终点诊断", "",
        "本包包含预登记的四臂各36更新终点诊断：原始training steps、task events、compact动作探针、"
        "四任务闭环行、本地轨迹索引，以及从既有E1轨迹只读导出的16个轻量视频。", "",
        "| 分支 | 成功 /16 | actual-flow MSE | 相对父节点动作差 |", "| --- | ---: | ---: | ---: |",
    ]
    for branch in BRANCHES:
        means = branch_probe_means[branch]
        lines.append(
            f"| {branch} | {branch_successes[branch]} | {means['inference_first5_mse']:.6f} | "
            f"{means['action_delta_mse_vs_parent']:.6f} |"
        )
    lines.extend((
        "", "大checkpoint与轨迹张量仅在本地保留。执行agent未作因果归因、未选择分支，也未继续下游实验。", "",
    ))
    (temporary / "README.md").write_text("\n".join(lines))
    if output.exists():
        shutil.rmtree(output)
    temporary.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.study.resolve(), args.repo.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
