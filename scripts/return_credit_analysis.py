#!/usr/bin/env python3
"""Mechanical completion and fixed paired statistics for the bounded batch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from ember.pi05_eval.return_credit import authority
from ember.pi05_eval_contract import policy_noise_seed
from ember.pi05_source_checkpoint import read_json, write_json_atomic


ARMS = ("P", "R", "NEG", "FM")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError(f"return-credit analysis output already exists: {path}")
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _trace(row):
    info = row.get("continuous_control_trace") or {}
    reference = info.get("trace") or {}
    path = Path(reference.get("path", ""))
    if not path.is_file() or path.stat().st_size != reference.get("bytes"):
        raise ValueError("return-credit continuous trace or file evidence missing")
    steps = int(row["steps"])
    with np.load(path, allow_pickle=False) as trace:
        if (trace["actions"].shape != (steps, 7)
                or trace["body_positions"].shape[0] != steps + 1
                or trace["eef_pos"].shape != (steps + 1, 3)
                or trace["eef_quat"].shape != (steps + 1, 4)
                or trace["gripper_qpos"].shape != (steps + 1, 2)
                or trace["predicates"].shape[0] != steps + 1
                or not bool(np.all(trace["predicates"][-1])) == bool(row["success"])):
            raise ValueError("return-credit T+1 or terminal predicate samples missing")
        objects = []
        for index, body in enumerate(info["body_registry"]):
            if body["kind"] != "object":
                continue
            positions = trace["body_positions"][:, index]
            distance = np.linalg.norm(positions - trace["eef_pos"], axis=1)
            objects.append({"name": body["name"], "body_id": body["body_id"],
                "initial_xyz": positions[0].tolist(), "final_xyz": positions[-1].tolist(),
                "endpoint_displacement": float(np.linalg.norm(positions[-1]-positions[0])),
                "sampled_min_eef_distance": float(distance.min())})
        return {"steps": steps, "samples": steps + 1,
                "objects": objects, "eef_endpoint_displacement": float(
                    np.linalg.norm(trace["eef_pos"][-1]-trace["eef_pos"][0])),
                "final_predicates": trace["predicates"][-1].tolist(),
                "control_step_only": True}


def _paired_native_noise(rows, task, state):
    suite, local_task = SUITES[task // 10], task % 10
    for row in rows:
        saved = row["policy_noise_seeds"]
        expected = [policy_noise_seed(7, suite, local_task, state, replan)
                    for replan in range(len(saved))]
        if saved != expected:
            raise ValueError("return-credit native policy RNG differs from registered stateless stream")


def _collection_rows(spec, root):
    collected, geometry = [], []
    for condition in spec["collection"]["conditions"]:
        task = condition["task"]
        for state in condition["init_state_ids"]:
            path = root / "collection" / "groups" / f"task_{task:03d}_state_{state:02d}" / "completion.json"
            group = read_json(path)
            if ((group["task"], group["state"], group["teacher_demo"]) !=
                    (task, state, condition["teacher_demo"]) or len(group["rows"]) != 4):
                raise ValueError("return-credit collection group completion changed")
            _paired_native_noise(group["rows"], task, state)
            for replica, row in enumerate(group["rows"]):
                if (row["global_task_id"], row["init_state_id"], row["replica"]) != (task, state, replica):
                    raise ValueError("return-credit collection row identity changed")
                decision = row.get("return_credit_collection") or {}
                if (decision.get("Q") != len(row["policy_noise_seeds"])
                        or decision.get("M") != min(4, decision["Q"])):
                    raise ValueError("return-credit Q/M selection changed")
                geometry.append({"phase": "collection", "task": task, "state": state,
                                 "replica": replica, **_trace(row)})
                collected.append(row)
    if len(collected) != 128 or len({(r["global_task_id"], r["init_state_id"], r["replica"])
                                     for r in collected}) != 128:
        raise ValueError("return-credit collection is not 128 unique episodes")
    return collected, geometry


def _evaluation_rows(spec, root):
    evaluated, cases, geometry = [], [], []
    for arm in ARMS:
        for task in spec["evaluation"]["task_ids"]:
            for teacher in spec["evaluation"]["teacher_demos"]:
                path = root / "evaluation" / arm / f"task_{task:03d}_teacher_{teacher:02d}" / "completion.json"
                panel = read_json(path)
                if ((panel["arm"], panel["task"], panel["teacher_demo"]) !=
                        (arm, task, teacher) or len(panel["rows"]) != 4):
                    raise ValueError("return-credit evaluation panel changed")
                for row in panel["rows"]:
                    if (row["arm"], row["global_task_id"], row["teacher_demo"]) != (arm, task, teacher):
                        raise ValueError("return-credit evaluation row identity changed")
                    geometry.append({"phase": "evaluation", "arm": arm, "task": task,
                                     "teacher": teacher, "state": row["init_state_id"], **_trace(row)})
                    capture = row["occupancy_trajectory"]
                    full = (task in spec["capture"]["full_cases"]["evaluation_tasks"]
                            and teacher == 46 and row["init_state_id"] == 32)
                    if capture["capture_level"] != ("full" if full else "compact"):
                        raise ValueError("return-credit fixed full-case selection changed")
                    if full:
                        cases.append({"arm": arm, "task": task, "state": 32,
                                      "teacher": teacher, "capture": capture,
                                      "trace": row["continuous_control_trace"]["trace"]})
                    evaluated.append(row)
    keys = [(r["arm"], r["global_task_id"], r["init_state_id"], r["teacher_demo"])
            for r in evaluated]
    if len(evaluated) != 256 or len(set(keys)) != 256 or len(cases) != 16:
        raise ValueError("return-credit evaluation rows or fixed cases incomplete")
    grouped = {}
    for row in evaluated:
        key = row["global_task_id"], row["init_state_id"]
        grouped.setdefault(key, []).append(row)
    for key, rows in grouped.items():
        if len(rows) != 8:
            raise ValueError(f"return-credit cross-arm/teacher policy RNG pairing changed: {key}")
        _paired_native_noise(rows, *key)
    return evaluated, cases, geometry


def _rows(spec, root, *, include_eval=True):
    collected, geometry = _collection_rows(spec, root)
    if not include_eval:
        return collected, [], [], geometry
    evaluated, cases, evaluation_geometry = _evaluation_rows(spec, root)
    return collected, evaluated, cases, geometry + evaluation_geometry


def _success_and_pairs(spec, evaluated):
    tasks = spec["evaluation"]["task_ids"]
    teachers = spec["evaluation"]["teacher_demos"]
    states = spec["evaluation"]["init_state_ids"]
    lookup = {(r["arm"], r["global_task_id"], r["init_state_id"], r["teacher_demo"]):
              int(r["success"]) for r in evaluated}
    cube = np.asarray([[[[lookup[(arm, task, state, teacher)] for teacher in teachers]
                         for state in states] for task in tasks] for arm in ARMS], dtype=np.int8)
    counts = {arm: {str(task): {str(teacher): int(cube[index, ti, :, vi].sum())
                                   for vi, teacher in enumerate(teachers)}
                         for ti, task in enumerate(tasks)} for index, arm in enumerate(ARMS)}
    suites = {arm: {suite: int(cube[index, [ti for ti, task in enumerate(tasks)
                                       if SUITES[task//10] == suite]].sum())
                    for suite in SUITES} for index, arm in enumerate(ARMS)}
    rng = np.random.default_rng(spec["evaluation"]["bootstrap"]["seed"])
    draws = int(spec["evaluation"]["bootstrap"]["replicates"])
    boot = np.empty((draws, len(ARMS), 3), dtype=np.float32)
    for sample in range(draws):
        task_ids = rng.integers(0, len(tasks), size=len(tasks))
        state_ids = rng.integers(0, len(states), size=(len(tasks), len(states)))
        picked = cube[:, task_ids[:, None], state_ids, :]
        boot[sample, :, :2] = picked.mean(axis=(1, 2))
        boot[sample, :, 2] = picked.mean(axis=(1, 2, 3))
    comparisons = []
    for left, right in spec["evaluation"]["comparisons"]:
        li, ri = ARMS.index(left), ARMS.index(right)
        a, b = cube[li], cube[ri]
        gained, lost, retained = int(((a == 1)&(b == 0)).sum()), int(((a == 0)&(b == 1)).sum()), int(((a == 1)&(b == 1)).sum())
        union = int(((a == 1)|(b == 1)).sum())
        delta = boot[:, li] - boot[:, ri]
        comparisons.append({"left": left, "right": right,
            "delta_teacher46": float((a[:, :, 0]-b[:, :, 0]).mean()),
            "delta_teacher47": float((a[:, :, 1]-b[:, :, 1]).mean()),
            "delta_equal_teachers": float((a-b).mean()),
            "ci95_teacher46": np.quantile(delta[:, 0], [.025, .975]).tolist(),
            "ci95_teacher47": np.quantile(delta[:, 1], [.025, .975]).tolist(),
            "ci95_equal_teachers": np.quantile(delta[:, 2], [.025, .975]).tolist(),
            "retained": retained, "gained": gained, "lost": lost,
            "churn": gained + lost, "jaccard": retained/union if union else 1.0,
            "per_task": {str(task): {"delta": float((a[i]-b[i]).mean()),
                "retained": int(((a[i]==1)&(b[i]==1)).sum()),
                "gained": int(((a[i]==1)&(b[i]==0)).sum()),
                "lost": int(((a[i]==0)&(b[i]==1)).sum())}
                for i, task in enumerate(tasks)}})
    return {"per_task_teacher_success": counts, "per_suite_success": suites,
            "arm_successes": {arm: int(cube[index].sum()) for index, arm in enumerate(ARMS)},
            "comparisons": comparisons, "bootstrap_draws": draws,
            "bootstrap_seed": spec["evaluation"]["bootstrap"]["seed"],
            "single_parent_and_collection_seed": True}


def _replay(spec, root, collected):
    expected = sum(row["return_credit_collection"]["M"] for row in collected)
    rows = []
    for arm in ("R", "NEG", "FM"):
        for task in spec["evaluation"]["task_ids"]:
            completion = read_json(root / "replay" / arm / f"task_{task:03d}" / "completion.json")
            record = completion["arrays"]
            path = Path(record["path"])
            if not path.is_file() or path.stat().st_size != record["bytes"]:
                raise ValueError("return-credit candidate replay array missing")
            with np.load(path, allow_pickle=False) as values:
                count = completion["predictions"]
                if (values["normalized"].shape != (count, 50, 7)
                        or values["environment"].shape != (count, 50, 7)
                        or values["osc_first5"].shape != (count, 5, 6)
                        or values["collection_mu_old"].shape != (count, 50, 7)
                        or not np.isfinite(values["normalized"]).all()):
                    raise ValueError("return-credit candidate full-flow replay invalid")
                delta = values["normalized"][:, :5] - values["collection_mu_old"][:, :5]
                rows.append({"arm": arm, "task": task, "predictions": count,
                    "normalized_first5_rms": float(np.sqrt(np.mean(delta**2))),
                    "normalized_first5_maxabs": float(np.max(np.abs(delta))),
                    "array": record})
    if sum(row["predictions"] for row in rows) != 3*expected or expected > 512:
        raise ValueError("return-credit candidate replay count exceeds registered decisions")
    return {"selected_parent_decisions": expected, "candidate_predictions": 3*expected,
            "task_arm_rows": rows}


def main():
    spec = authority()
    root = Path(spec["resources"]["study_root"])
    output = root / "analysis"
    if output.exists():
        raise ValueError("return-credit analysis already exists or has partial preserved output")
    gradient = read_json(root / "gradient" / "completion.json")
    stopped = gradient["all_advantages_zero"]
    collected, evaluated, cases, geometry = _rows(spec, root, include_eval=not stopped)
    if gradient["all_advantages_zero"]:
        output.mkdir(parents=True)
        _jsonl(output / "collection_rows.jsonl", collected)
        _jsonl(output / "geometry_rows.jsonl", geometry)
        write_json_atomic(output / "completion.json", {
            "schema_version": "ember_return_credit_completion_v1",
            "study_id": spec["study_id"], "status": "reward_credit_unidentified",
            "collection_rows": len(collected), "evaluation_rows": 0,
            "continuous_traces": len(geometry), "gradient": str(root / "gradient" / "completion.json"),
            "stop_reason": "all registered LOO advantages are zero"})
        return
    summary = _success_and_pairs(spec, evaluated)
    replay = _replay(spec, root, collected)
    output.mkdir(parents=True)
    _jsonl(output / "collection_rows.jsonl", collected)
    _jsonl(output / "evaluation_rows.jsonl", evaluated)
    _jsonl(output / "geometry_rows.jsonl", geometry)
    write_json_atomic(output / "case_index.json", {"cases": cases, "count": len(cases)})
    write_json_atomic(output / "paired_statistics.json", summary)
    write_json_atomic(output / "function_displacement.json", replay)
    write_json_atomic(output / "completion.json", {
        "schema_version": "ember_return_credit_completion_v1",
        "study_id": spec["study_id"], "collection_rows": len(collected),
        "evaluation_rows": len(evaluated), "full_cases": len(cases),
        "continuous_traces": len(geometry), "replay_predictions": replay["candidate_predictions"],
        "gradient": str(root / "gradient" / "completion.json"),
        "comparisons": len(summary["comparisons"]),
        "limits": spec["interpretation_limits"],
        "result_interpretation_owner": spec["scientific_owner_thread"]})


if __name__ == "__main__":
    main()
