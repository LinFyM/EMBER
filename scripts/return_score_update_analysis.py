#!/usr/bin/env python3
"""Mechanical completion and prespecified paired readouts for score-update stage1."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from ember.pi05_eval.trajectory_capture import validate_passive_trace_row
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record
from scripts.return_score_update import (ARMS, SPEC_PATH, _bank_path, _bank_record, _contract,
    _episode_path, _metadata, authority, bank_keys, episode_keys)


COMPARISONS = (("RB", "RAW"), ("RB", "P"), ("RAW", "P"))
OLD_ROOT = Path("/data0/user/ymdai/ember_runs/return_credit_direction_causality_20260925")


def _episode_header(spec, root, commit, evaluation, tasks, paths, key):
    arm, task_id, teacher, state = key
    output = _episode_path(root, key)
    payload = read_json(output / "completion.json")
    row = payload["row"]
    task = tasks[task_id]
    full = task_id in spec["capture"]["full_cases"]["tasks"] and state == 32 and teacher == 46
    contract = _contract(evaluation, paths, task, output, full=full)
    if (payload["schema_version"] != "ember_return_score_update_episode_v1"
            or payload["implementation_commit"] != commit
            or tuple(payload["key"]) != key
            or (row["arm"], row["global_task_id"], row["teacher_demo"], row["init_state_id"]) != key
            or (row["suite"], row["task_id"], row["language"]) !=
               (task.suite, task.task_id, task.language)
            or row["split_role"] != "train"
            or type(row["success"]) is not bool
            or row["bank_record"] != file_record(_bank_path(root, arm, task_id, teacher) / "bank_record.json")
            or row["occupancy_trajectory"]["capture_level"] != ("full" if full else "compact")
            or not row.get("stage_predicates", {}).get("predicates")
            or int(row["steps"]) not in range(1, task.horizon + 1)):
        raise ValueError(f"score-update episode identity or outcome incomplete: {key}")
    validate_passive_trace_row(row, contract, asdict(task))
    return row, output, full


def _first_replan(row, output, full, key):
    trajectory_path = Path(row["occupancy_trajectory"]["path"])
    if (not trajectory_path.is_relative_to(output / "trajectories")
            or file_record(trajectory_path) != {"path": str(trajectory_path),
                                                 "bytes": row["occupancy_trajectory"]["bytes"]}):
        raise ValueError(f"score-update trajectory asset changed: {key}")
    trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=True)
    chunks = trajectory["action_chunks"]
    actions = trajectory["executed_action_prefixes"]
    if (trajectory["capture_level"] != ("full" if full else "compact")
            or trajectory["steps"] != row["steps"]
            or trajectory["policy_noise_seeds"] != tuple(row["policy_noise_seeds"])
            or len(chunks) != len(actions)
            or len(chunks) != len(row["policy_noise_seeds"])
            or len(chunks) != (row["steps"] + 4) // 5
            or not chunks or chunks[0].shape != (1, 50, 7)
            or actions[0].shape != (5, 7)
            or trajectory["states"][0].shape != (8,)
            or not torch.isfinite(chunks[0]).all()
            or not torch.isfinite(actions[0]).all()):
        raise ValueError(f"score-update first real policy replan incomplete: {key}")
    with np.load(row["continuous_control_trace"]["trace"]["path"], allow_pickle=False) as trace:
        applied = trace["actions"][:5].copy()
    if (applied.shape != (min(5, row["steps"]), 7)
            or not np.allclose(applied, actions[0].numpy()[:len(applied)], atol=1e-6, rtol=0)):
        raise ValueError(f"score-update planned and actually applied first commands differ: {key}")
    return trajectory["states"][0].numpy(), chunks[0][0].numpy(), applied


def _read_episode(spec, root, commit, evaluation, tasks, paths, key):
    row, output, full = _episode_header(spec, root, commit, evaluation, tasks, paths, key)
    return row, _first_replan(row, output, full, key)


def _compare(success: dict, tasks: list[int], teachers: tuple[int, ...],
             target: str, reference: str) -> dict:
    cases = tuple((task, state, teacher) for task in tasks
                  for state in (32, 33) for teacher in teachers)
    gained = [list(key) for key in cases if success[key, target] and not success[key, reference]]
    lost = [list(key) for key in cases if success[key, reference] and not success[key, target]]
    retained = [list(key) for key in cases if success[key, reference] and success[key, target]]
    union = sum(success[key, reference] or success[key, target] for key in cases)
    return {"target": target, "reference": reference, "rows": len(cases),
            "target_successes": sum(success[key, target] for key in cases),
            "reference_successes": sum(success[key, reference] for key in cases),
            "delta_successes": len(gained) - len(lost),
            "retained": retained, "gained": gained, "lost": lost,
            "churn": len(gained) + len(lost),
            "jaccard": len(retained) / union if union else 1.0}


def _bootstrap(spec, success):
    tasks = spec["evaluation"]["task_ids"]
    cube = np.asarray([[[[int(success[(task, state, teacher), arm]) for arm in ARMS]
                         for teacher in (46, 47)] for state in (32, 33)] for task in tasks],
                      dtype=np.float64)
    rng = np.random.default_rng(spec["analysis"]["bootstrap"]["seed"])
    count = spec["analysis"]["bootstrap"]["draws"]
    task_draw = rng.integers(0, 8, size=(count, 8))
    state_draw = rng.integers(0, 2, size=(count, 8, 2))
    sampled = cube[task_draw[:, :, None], state_draw]
    if sampled.shape != (count, 8, 2, 2, 3):
        raise ValueError("joint task/state bootstrap lost paired teachers or arms")
    by_teacher = sampled.mean(axis=(1, 2))
    summary = {}
    for target, reference in COMPARISONS:
        t, r = ARMS.index(target), ARMS.index(reference)
        differences = by_teacher[:, :, t] - by_teacher[:, :, r]
        summary[f"{target}-{reference}"] = {}
        for name, value in (("46", differences[:, 0]), ("47", differences[:, 1]),
                            ("equal_two_teachers", differences.mean(axis=1))):
            summary[f"{target}-{reference}"][name] = {
                "point": float((cube[..., t] - cube[..., r]).mean(axis=(0, 1))[int(name) - 46]
                               if name in ("46", "47") else (cube[..., t] - cube[..., r]).mean()),
                "ci95": np.quantile(value, [.025, .975]).tolist()}
    return {"draws": count, "seed": spec["analysis"]["bootstrap"]["seed"],
            "joint_resampling": "8 tasks then 2 states per drawn task; both teachers and all arms paired",
            "comparisons": summary}


def _function_pairs(spec, first):
    records = []
    for task in spec["evaluation"]["task_ids"]:
        for state in (32, 33):
            for teacher in (46, 47):
                data = {arm: first[arm, task, teacher, state] for arm in ARMS}
                baseline = data["P"][0]
                if any(np.max(np.abs(value[0] - baseline)) > 1e-4 for value in data.values()):
                    raise ValueError("score-update same-state robot policy input diverged")
                for target, reference in COMPARISONS:
                    normalized = data[target][1] - data[reference][1]
                    common_steps = min(len(data[target][2]), len(data[reference][2]))
                    commands = data[target][2][:common_steps] - data[reference][2][:common_steps]
                    groups = {}
                    for label, sl in (("xyz", slice(0, 3)), ("rotation", slice(3, 6)),
                                      ("gripper", slice(6, 7))):
                        groups[label] = {
                            "full50_normalized_rms": float(np.sqrt(np.mean(normalized[:, sl] ** 2))),
                            "full50_normalized_signed_mean": np.mean(normalized[:, sl], axis=0).tolist(),
                            "executed_first_replan_common_steps": common_steps,
                            "first5_environment_rms": float(np.sqrt(np.mean(commands[:, sl] ** 2))),
                            "first5_environment_signed_mean": np.mean(commands[:, sl], axis=0).tolist()}
                    records.append({"task": task, "state": state, "teacher": teacher,
                                    "target": target, "reference": reference, "readout": groups,
                                    "first5_gripper_sign_changes": int(np.sum(
                                        np.sign(data[target][2][:common_steps, 6]) !=
                                        np.sign(data[reference][2][:common_steps, 6]))),
                                    "policy_noise_seed": int(data[target][3])})
    return records


def _old_parent_self(spec, success):
    compared, changed = 0, []
    for task in spec["evaluation"]["task_ids"]:
        for teacher in (46, 47):
            path = OLD_ROOT / "evaluation" / "P" / f"task_{task:03d}_teacher_{teacher:02d}" / "completion.json"
            payload = read_json(path)
            for row in payload["rows"]:
                state = row["init_state_id"]
                if state not in (32, 33):
                    continue
                compared += 1
                if row["success"] != success[(task, state, teacher), "P"]:
                    changed.append([task, state, teacher])
    if compared != 32:
        raise ValueError("historical parent self subset incomplete")
    return {"rows": compared, "changed_success_keys": changed,
            "role": "descriptive old E2 P versus new P; excluded from primary comparisons"}


def _check_exits(root, commit):
    evidence = []
    for stage, expected in (("materialize", 48), ("pilot", 3), ("evaluate", 93)):
        receipts = sorted((root / "launch").glob(f"{stage}_exit_*.json"))
        if len(receipts) != 1:
            raise ValueError(f"score-update {stage} needs one complete worker-exit receipt")
        result = read_json(receipts[0])
        if (result["implementation_commit"] != commit or result["jobs"] != expected
                or any(worker["exit_code"] != 0 for worker in result["workers"])):
            raise ValueError(f"score-update {stage} worker exit or scope failed")
        evidence.append({"stage": stage, "receipt": file_record(receipts[0]),
                         "gpu_hours_conservative": result["gpu_hours_conservative"]})
    if not (root / "launch" / "pilot_acceptance.json").is_file():
        raise ValueError("score-update score-blind formal pilot acceptance missing")
    return evidence


def _collect_rows(spec, root, config, commit):
    for key in bank_keys(spec):
        _bank_record(spec, root, commit, key)
    evaluation, tasks, paths = _metadata(spec, root, config)
    rows, first, success, cases = [], {}, {}, []
    for key in episode_keys(spec):
        row, (state, normalized, commands) = _read_episode(
            spec, root, commit, evaluation, tasks, paths, key)
        arm, task, teacher, init = key
        rows.append(row)
        success[(task, init, teacher), arm] = int(row["success"])
        first[key] = (state, normalized, commands, row["policy_noise_seeds"][0])
        if row["occupancy_trajectory"]["capture_level"] == "full":
            cases.append({"key": list(key), "trajectory": row["occupancy_trajectory"],
                          "continuous_control_trace": row["continuous_control_trace"]["trace"]})
    if len(rows) != 96 or len(cases) != 12:
        raise ValueError("score-update required 96 unique rows or 12 full cases missing")
    for task in spec["evaluation"]["task_ids"]:
        for state in (32, 33):
            paired = [first[arm, task, teacher, state] for arm in ARMS for teacher in (46, 47)]
            if (len({item[3] for item in paired}) != 1
                    or any(np.max(np.abs(item[0] - paired[0][0])) > 1e-4 for item in paired[1:])):
                raise ValueError("score-update same-state initial observation or first policy RNG differs")
    return rows, first, success, cases


def _summaries(spec, success):
    summaries = {}
    for name, tasks_group in [("all8", list(spec["evaluation"]["task_ids"])),
                              *((suite, [task for task in spec["evaluation"]["task_ids"]
                                          if task // 10 == index]) for index, suite in enumerate(
                                              ("spatial", "object", "goal", "long"))),
                              *((f"task_{task}", [task]) for task in spec["evaluation"]["task_ids"])]:
        summaries[name] = {teacher_name: [_compare(success, tasks_group, teachers, *pair)
                                         for pair in COMPARISONS]
                           for teacher_name, teachers in (("46", (46,)), ("47", (47,)),
                                                          ("equal_two_teachers", (46, 47)))}
    return summaries


def _write_outputs(spec, root, commit, rows, first, success, cases, receipts):
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    with (analysis / "raw_rows.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (analysis / "first_replan_pairs.jsonl").open("x", encoding="utf-8") as handle:
        for record in _function_pairs(spec, first):
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    write_json_atomic(analysis / "case_index.json", {"cases": cases, "count": len(cases)})
    breadth = {}
    for arm in ARMS:
        won = {task for task in spec["evaluation"]["task_ids"]
               if any(success[(task, state, teacher), arm]
                      for state in (32, 33) for teacher in (46, 47))}
        breadth[arm] = {"tasks_with_any_success": sorted(won), "task_count": len(won),
                        "suites_with_any_success": sorted({task // 10 for task in won})}
    write_json_atomic(analysis / "comparisons.json", {
        "arm_successes": {arm: sum(success[case, arm] for case in (
            (task, state, teacher) for task in spec["evaluation"]["task_ids"]
            for state in (32, 33) for teacher in (46, 47))) for arm in ARMS},
        "breadth": breadth, "summaries": _summaries(spec, success),
        "bootstrap": _bootstrap(spec, success),
        "historical_parent_self": _old_parent_self(spec, success)})
    write_json_atomic(analysis / "completion.json", {
        "schema_version": "ember_return_score_update_stage1_completion_v1",
        "implementation_commit": commit, "study_spec": str(SPEC_PATH),
        "banks": 48, "unique_episodes": 96, "continuous_traces": 96,
        "full_cases": 12, "first_replan_comparisons": 96,
        "raw_rows": file_record(analysis / "raw_rows.jsonl"),
        "comparisons": file_record(analysis / "comparisons.json"),
        "first_replan": file_record(analysis / "first_replan_pairs.jsonl"),
        "case_index": file_record(analysis / "case_index.json"),
        "worker_exit_receipts": receipts,
        "gpu_hours_conservative_total": sum(row["gpu_hours_conservative"] for row in receipts),
        "missing": []})


def main() -> None:
    spec, root, config, git = authority(formal=True)
    commit = git["commit"]
    receipts = _check_exits(root, commit)
    rows, first, success, cases = _collect_rows(spec, root, config, commit)
    _write_outputs(spec, root, commit, rows, first, success, cases, receipts)


if __name__ == "__main__":
    main()
