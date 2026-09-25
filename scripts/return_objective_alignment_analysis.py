#!/usr/bin/env python3
"""Mechanical acceptance and paired readouts for the frozen 128-row diagnostic."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from ember.pi05_eval.exploration import exploration_covariance, exploration_noise_seed, validate_episode_exploration
from ember.pi05_eval.trajectory_capture import validate_passive_trace_row
from ember.pi05_eval_contract import policy_noise_seed
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record
from scripts.return_objective_alignment import (CELLS, SPEC_PATH, _bank_path, _episode_path,
    _metadata, authority, bank_keys, bank_record, episode_contract, episode_keys,
    same_common_seed_prefix)


CONTRASTS = {
    "Delta_Sigma": {"RB_JS": 1, "P_JS": -1},
    "Delta0": {"RB_J0": 1, "P_J0": -1},
    "noise_at_P": {"P_JS": 1, "P_J0": -1},
    "noise_at_RB": {"RB_JS": 1, "RB_J0": -1},
    "interaction": {"RB_JS": 1, "P_JS": -1, "RB_J0": -1, "P_J0": 1},
}


def check_action_injection(spec, row, output: Path):
    """Recompute registered epsilon and compare normalized and applied commands."""
    trajectory_path = Path(row["occupancy_trajectory"]["path"])
    if (not trajectory_path.is_relative_to(output / "trajectories")
            or file_record(trajectory_path) != {"path": str(trajectory_path),
                                                 "bytes": row["occupancy_trajectory"]["bytes"]}):
        raise ValueError("objective-alignment trajectory path or size changed")
    trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=True)
    means = trajectory.get("pre_exploration_normalized_means")
    chunks = trajectory["action_chunks"]
    commands = trajectory["executed_action_prefixes"]
    seeds = row["diagnostic_exploration"]["noise_seeds"]
    replans = (int(row["steps"]) + 4) // 5
    if (means is None or len(means) != replans or len(chunks) != replans
            or len(commands) != replans or len(seeds) != replans
            or trajectory["policy_noise_seeds"] != tuple(row["policy_noise_seeds"])
            or trajectory.get("exploration_noise_seeds") != tuple(seeds)
            or trajectory["steps"] != row["steps"]
            or trajectory["capture_level"] != row["occupancy_trajectory"]["capture_level"]
            or any(mean.shape != (5, 7) or chunk.shape != (1, 50, 7)
                   or command.shape != (5, 7) for mean, chunk, command in zip(means, chunks, commands, strict=True))):
        raise ValueError("objective-alignment mean/chunk/seed/action trace incomplete")
    lower = torch.linalg.cholesky(exploration_covariance())
    maximum, noise_norm = 0., []
    for index, (mean, chunk, seed) in enumerate(zip(means, chunks, seeds, strict=True)):
        expected_seed = exploration_noise_seed(row["diagnostic_exploration"],
            suite=row["suite"], task_id=row["task_id"], state_id=row["init_state_id"], replan=index)
        if seed != expected_seed or not torch.isfinite(mean).all() or not torch.isfinite(chunk).all():
            raise ValueError("objective-alignment seed or finite policy action changed")
        eta = (torch.randn(35, generator=torch.Generator(device="cpu").manual_seed(seed),
                           dtype=torch.float32) @ lower.T).reshape(5, 7)
        if not row["diagnostic_exploration"]["enabled"]:
            eta.zero_()
        expected = (mean + eta).to(chunk.dtype)
        maximum = max(maximum, float((expected - chunk[0, :5, :7]).abs().max()))
        noise_norm.append(float(torch.linalg.vector_norm(eta)))
    if maximum > 1e-3:
        raise ValueError(f"objective-alignment normalized injection reconstruction failed: {maximum}")
    with np.load(row["continuous_control_trace"]["trace"]["path"], allow_pickle=False) as trace:
        actual = trace["actions"].copy()
    planned = np.concatenate([part.numpy() for part in commands])[:row["steps"]]
    if (actual.shape != planned.shape or not np.allclose(actual, planned, atol=1e-6, rtol=0)):
        raise ValueError("objective-alignment planned and actually applied actions diverged")
    return {"maximum_normalized_residual": maximum, "noise_l2_by_replan": noise_norm,
            "first_mean": means[0].numpy(), "first_command": actual[:min(5, len(actual))],
            "first_state": trajectory["states"][0].numpy()}


def _read_checked_episode(spec, root, commit, evaluation, tasks, paths, key):
    cell, task_id, teacher, state = key
    output = _episode_path(root, key)
    payload = read_json(output / "completion.json")
    row = payload["row"]
    task = tasks[task_id]
    full = task_id in spec["capture"]["full_cases"]["tasks"] and state == 0
    contract = episode_contract(spec, evaluation, paths, task, output, key)
    expected_seeds = [policy_noise_seed(7, task.suite, task.task_id, state, index)
                      for index in range((row["steps"] + 4) // 5)]
    if (payload["schema_version"] != "ember_return_objective_alignment_episode_v1"
            or payload["implementation_commit"] != commit or tuple(payload["key"]) != key
            or (row["cell"], row["global_task_id"], row["teacher_demo"], row["init_state_id"]) != key
            or (row["suite"], row["task_id"], row["language"]) !=
               (task.suite, task.task_id, task.language)
            or row["split_role"] != "train" or type(row["success"]) is not bool
            or not 1 <= row["steps"] <= task.horizon
            or row["bank_record"] != file_record(_bank_path(root, row["model"], task_id, teacher) / "bank_record.json")
            or row["occupancy_trajectory"]["capture_level"] != ("full" if full else "compact")
            or not row.get("stage_predicates", {}).get("predicates")
            or row["policy_noise_seeds"] != expected_seeds
            or not validate_episode_exploration(contract, row, replans=len(expected_seeds))):
        raise ValueError(f"objective-alignment episode provenance incomplete: {key}")
    validate_passive_trace_row(row, contract, asdict(task))
    return row, check_action_injection(spec, row, output), full


def _check_pairing(spec, root, rows_by_key, first):
    for condition in spec["evaluation"]["conditions"]:
        task, teacher = condition["task"], condition["teacher_demo"]
        for state in condition["init_state_ids"]:
            paired = [first[cell, task, teacher, state] for cell in CELLS]
            group = [rows_by_key[cell, task, teacher, state] for cell in CELLS]
            if (any(not same_common_seed_prefix(row["policy_noise_seeds"], group[0]["policy_noise_seeds"])
                    for row in group)
                    or any(np.max(np.abs(item["first_state"] - paired[0]["first_state"])) > 1e-4
                           for item in paired[1:])
                    or group[0]["bank_record"] != group[1]["bank_record"]
                    or group[2]["bank_record"] != group[3]["bank_record"]
                    or any(not same_common_seed_prefix(
                        row["diagnostic_exploration"]["noise_seeds"],
                        group[0]["diagnostic_exploration"]["noise_seeds"])
                        for row in group)):
                raise ValueError("objective-alignment four-cell initial query/bank/RNG pairing changed")


def _collect(spec, root, config, commit):
    for key in bank_keys(spec):
        bank_record(spec, root, commit, key)
    evaluation, tasks, paths = _metadata(spec, root, config)
    rows, rows_by_key, first, success, cases, audits = [], {}, {}, {}, [], []
    for key in episode_keys(spec):
        row, readout, full = _read_checked_episode(spec, root, commit, evaluation, tasks, paths, key)
        cell, task_id, teacher, state = key
        rows.append(row)
        rows_by_key[key] = row
        first[key] = readout
        success[(task_id, state), cell] = int(row["success"])
        audits.append({"key": list(key), "maximum_normalized_residual": readout["maximum_normalized_residual"],
                       "noise_l2_by_replan": readout["noise_l2_by_replan"], "steps": row["steps"]})
        if full:
            cases.append({"key": list(key), "trajectory": row["occupancy_trajectory"],
                          "continuous_control_trace": row["continuous_control_trace"]["trace"]})
    if len(rows) != 128 or len(cases) != 16 or len({tuple(row["key"]) for row in audits}) != 128:
        raise ValueError("objective-alignment raw matrix, continuous traces or full cases missing")
    _check_pairing(spec, root, rows_by_key, first)
    return rows, first, success, cases, audits


def _simple_comparison(success, cases, target, reference):
    retained = [list(key) for key in cases if success[key, target] and success[key, reference]]
    gained = [list(key) for key in cases if success[key, target] and not success[key, reference]]
    lost = [list(key) for key in cases if success[key, reference] and not success[key, target]]
    union = len(retained) + len(gained) + len(lost)
    return {"target": target, "reference": reference, "rows": len(cases),
            "delta_successes": len(gained) - len(lost), "retained": retained,
            "gained": gained, "lost": lost, "churn": len(gained) + len(lost),
            "jaccard": len(retained) / union if union else 1.0}


def _summaries(spec, success):
    tasks = [row["task"] for row in spec["evaluation"]["conditions"]]
    groups = {"all8": tasks, **{f"task_{task}": [task] for task in tasks},
              **{suite: [task for task in tasks if task // 10 == index]
                 for index, suite in enumerate(("spatial", "object", "goal", "long"))}}
    results = {}
    for name, members in groups.items():
        cases = [(task, state) for task in members for state in (0, 1, 2, 3)]
        totals = {cell: sum(success[key, cell] for key in cases) for cell in CELLS}
        comparisons = {}
        for label, coefficients in CONTRASTS.items():
            value = sum(coefficient * totals[cell] for cell, coefficient in coefficients.items())
            comparisons[label] = {"delta_successes": value, "mean": value / len(cases)}
        pairs = {label: _simple_comparison(success, cases, target, reference)
                 for label, target, reference in (("Delta_Sigma", "RB_JS", "P_JS"),
                                                  ("Delta0", "RB_J0", "P_J0"),
                                                  ("noise_at_P", "P_JS", "P_J0"),
                                                  ("noise_at_RB", "RB_JS", "RB_J0"))}
        results[name] = {"tasks": members, "rows_per_cell": len(cases),
                         "cell_successes": totals, "contrasts": comparisons,
                         "paired_sets": pairs}
    breadth = {}
    for cell in CELLS:
        won = sorted(task for task in tasks if any(success[(task, state), cell] for state in range(4)))
        breadth[cell] = {"tasks_with_any_success": won, "task_count": len(won),
                         "suites_with_any_success": sorted({SUITES[task // 10] for task in won})}
    return results, breadth


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _bootstrap(spec, success):
    tasks = [row["task"] for row in spec["evaluation"]["conditions"]]
    cube = np.asarray([[[success[(task, state), cell] for cell in CELLS]
                        for state in range(4)] for task in tasks], dtype=np.float64)
    config = spec["analysis"]["bootstrap"]
    rng = np.random.default_rng(config["seed"])
    count = config["draws"]
    task_draw = rng.integers(0, 8, size=(count, 8))
    state_draw = rng.integers(0, 4, size=(count, 8, 4))
    sampled = cube[task_draw[:, :, None], state_draw]
    if sampled.shape != (count, 8, 4, 4):
        raise ValueError("objective-alignment paired task/state bootstrap shape changed")
    means = sampled.mean(axis=(1, 2))
    points = cube.mean(axis=(0, 1))
    return {"draws": count, "seed": config["seed"],
            "sampling": "joint task8 then state4 per drawn task; all four cells paired",
            "contrasts": {name: {"point": float(sum(points[CELLS.index(cell)] * coefficient
                                                     for cell, coefficient in weights.items())),
                                 "ci95": np.quantile(sum(means[:, CELLS.index(cell)] * coefficient
                                                         for cell, coefficient in weights.items()),
                                                       [.025, .975]).tolist()}
                          for name, weights in CONTRASTS.items()}}


def _function_readout(spec, first):
    records = []
    for row in spec["evaluation"]["conditions"]:
        task, teacher = row["task"], row["teacher_demo"]
        for state in range(4):
            data = {cell: first[cell, task, teacher, state] for cell in CELLS}
            for name, a, b in (("model_J0", "RB_J0", "P_J0"),
                               ("model_JS", "RB_JS", "P_JS"),
                               ("noise_P", "P_JS", "P_J0"),
                               ("noise_RB", "RB_JS", "RB_J0")):
                means = data[a]["first_mean"] - data[b]["first_mean"]
                length = min(len(data[a]["first_command"]), len(data[b]["first_command"]))
                actions = data[a]["first_command"][:length] - data[b]["first_command"][:length]
                groups = {}
                for label, sl in (("xyz", slice(0, 3)), ("rotation", slice(3, 6)),
                                  ("gripper", slice(6, 7))):
                    groups[label] = {"normalized_mean_rms": float(np.sqrt(np.mean(means[:, sl] ** 2))),
                                     "normalized_signed_mean": np.mean(means[:, sl], axis=0).tolist(),
                                     "actual_first5_rms": float(np.sqrt(np.mean(actions[:, sl] ** 2))),
                                     "actual_signed_mean": np.mean(actions[:, sl], axis=0).tolist()}
                records.append({"task": task, "teacher": teacher, "state": state,
                                "comparison": name, "target": a, "reference": b,
                                "common_actual_steps": length, "groups": groups,
                                "first5_gripper_sign_changes": int(np.sum(
                                    np.sign(data[a]["first_command"][:length, 6]) !=
                                    np.sign(data[b]["first_command"][:length, 6])))})
    return records


def _receipts(root, commit):
    results = []
    for stage, count in (("materialize", 16), ("pilot", 4), ("evaluate", 124)):
        paths = sorted((root / "launch").glob(f"{stage}_exit_*.json"))
        if len(paths) != 1:
            raise ValueError(f"objective-alignment {stage} needs one exit receipt")
        receipt = read_json(paths[0])
        if (receipt["implementation_commit"] != commit or receipt["jobs"] != count
                or receipt["budget_exhausted"] or any(worker["exit_code"] != 0
                                                    for worker in receipt["workers"])):
            raise ValueError(f"objective-alignment {stage} worker exit or scope failed")
        results.append({"stage": stage, "receipt": file_record(paths[0]),
                        "gpu_hours_conservative": receipt["gpu_hours_conservative"]})
    if not (root / "launch" / "pilot_acceptance.json").is_file():
        raise ValueError("objective-alignment score-blind pilot acceptance missing")
    return results


def main():
    spec, root, config, git = authority(formal=True)
    commit = git["commit"]
    receipts = _receipts(root, commit)
    rows, first, success, cases, audits = _collect(spec, root, config, commit)
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    with (analysis / "raw_rows.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (analysis / "noise_action_audit.jsonl").open("x", encoding="utf-8") as handle:
        for row in audits:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (analysis / "first_replan_readout.jsonl").open("x", encoding="utf-8") as handle:
        for row in _function_readout(spec, first):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json_atomic(analysis / "case_index.json", {"count": len(cases), "cases": cases})
    summaries, breadth = _summaries(spec, success)
    write_json_atomic(analysis / "comparisons.json", {
        "schema_version": "ember_return_objective_alignment_comparisons_v1",
        "cell_success_sets": {cell: [[task, state] for task, state in (
            (row["task"], state) for row in spec["evaluation"]["conditions"] for state in range(4))
            if success[(task, state), cell]] for cell in CELLS},
        "breadth": breadth, "summaries": summaries, "bootstrap": _bootstrap(spec, success)})
    write_json_atomic(analysis / "completion.json", {
        "schema_version": "ember_return_objective_alignment_completion_v1",
        "implementation_commit": commit, "study_spec": str(SPEC_PATH),
        "banks": 16, "unique_episodes": 128, "continuous_traces": 128,
        "full_cases": 16, "first_replan_comparisons": 128,
        "raw_rows": file_record(analysis / "raw_rows.jsonl"),
        "noise_action_audit": file_record(analysis / "noise_action_audit.jsonl"),
        "first_replan_readout": file_record(analysis / "first_replan_readout.jsonl"),
        "comparisons": file_record(analysis / "comparisons.json"),
        "case_index": file_record(analysis / "case_index.json"),
        "worker_exit_receipts": receipts,
        "gpu_hours_conservative_total": sum(row["gpu_hours_conservative"] for row in receipts),
        "missing": []})


if __name__ == "__main__":
    main()
