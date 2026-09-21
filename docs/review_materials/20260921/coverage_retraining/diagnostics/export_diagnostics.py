"""Export portable, read-only diagnostics from the two formal Writer studies.

The CSVs contain scalar measurements and success flags, never checkpoints,
video frames, action arrays, host preflight snapshots, or raw run contracts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


SUITE_BASE = {"libero_spatial": 0, "libero_object": 10, "libero_goal": 20, "libero_10": 30}
OLD_WRITER_STEPS = (900, 1200, 1500, 1800, 2100)
NEW_WRITER_STEPS = tuple(range(200, 1801, 200))
NEW_MTBC_STEPS = tuple(range(50, 501, 50))
STEP_FIELDS = (
    "step", "lr_next", "mean_flow_loss", "mean_teaching_loss", "mean_total_loss",
    "writer_grad_norm",
    "text_meta_grad_norm", "vl_meta_grad_norm", "meta_grad_norm", "total_grad_norm",
    "seconds", "peak_reserved_gib", "supervised_queries", "teaching_queries",
)
EXPOSURE_FIELDS = (
    "step", "suite", "task", "occurrence", "condition_index", "queries", "frames",
    "flow_loss", "teaching_loss", "fm_lora_gradient_norm",
    "teaching_lora_gradient_norm", "task_weight", "seconds",
)


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def read_jsonl(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, fields: list[str], rows: list[dict]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def panels(old: Path, new: Path):
    for step in OLD_WRITER_STEPS:
        sub = ("training/evaluation" if step <= 1500 else "continuation/main/training/evaluation")
        yield ("old", "writer", "correct", step, old / sub / f"validation_correct_step{step}")
    for arm, step in (("other", 1200), ("other", 1500), ("wrong", 1500),
                      ("shuffled", 1500), ("reversed", 1500)):
        source_arm = {"other": "same_task_other", "wrong": "cross_suite_wrong"}.get(arm, arm)
        yield ("old", "writer", arm, step,
               old / "training/evaluation" / f"validation_{source_arm}_step{step}")
    for step in NEW_WRITER_STEPS:
        yield ("new", "writer", "correct", step, new / "evaluation" / f"writer_{step:08d}")
    for arm, suffix in (("other", "other"), ("wrong", "cross_suite_wrong"),
                        ("shuffled", "shuffled"), ("reversed", "reversed")):
        yield ("new", "writer", arm, 1000, new / "evaluation" / f"writer_00001000_{suffix}")
    for step in NEW_MTBC_STEPS:
        yield ("new", "mtbc", "correct", step, new / "evaluation" / f"mtbc_{step:08d}")
    yield ("new", "source", "correct", 1000, new / "evaluation/source_validation")


def panel_records(old: Path, new: Path):
    success_rows, task_rows, metadata = [], [], []
    by_panel = {}
    for study, method, arm, step, directory in panels(old, new):
        result = read_json(directory / "results.json")
        contract = read_json(directory / "run_contract.json")
        tag = (study, method, arm, step)
        assert result["mode"] == "formal" and result["role"] == "validation", tag
        assert len(result["rows"]) == result["overall"]["episodes"] == 400, tag
        seen = set()
        current = {}
        for row in result["rows"]:
            key = (row["suite"], int(row["task_id"]), int(row["init_state_id"]))
            assert key not in seen, (tag, key)
            seen.add(key)
            h = row.get("horizon_writer_lora") or {}
            demos = h.get("teacher_demo_indices", [])
            item = {
                "study": study, "method": method, "arm": arm, "step": step,
                "suite": key[0], "task_id": key[1],
                "global_task_id": SUITE_BASE[key[0]] + key[1], "init_state_id": key[2],
                "env_seed": row.get("env_seed", ""),
                "policy_seed_root": row.get("policy_seed_root", ""),
                "video_ordinal": h.get("video_ordinal", ""),
                "teacher_demo_indices": ";".join(map(str, demos)),
                "condition_id": h.get("condition_id", ""),
                "rollout_steps": row.get("steps", ""),
                "success": int(row["success"]),
            }
            success_rows.append(item)
            current[key] = item
        assert sum(x["success"] for x in current.values()) == result["overall"]["successes"], tag
        assert len(result["per_task"]) == 8, tag
        for task in result["per_task"]:
            count = sum(x["success"] for k, x in current.items()
                        if k[:2] == (task["suite"], task["task_id"]))
            assert count == task["successes"] and task["episodes"] == 50, (tag, task)
            if arm == "correct":
                task_rows.append({"study": study, "method": method, "step": step,
                                  "suite": task["suite"], "task_id": task["task_id"],
                                  "global_task_id": SUITE_BASE[task["suite"]] + task["task_id"],
                                  "language": task.get("language", ""),
                                  "episodes": 50, "successes": count})
        metadata.append({
            "study": study, "method": method, "arm": arm, "step": step,
            "episodes": 400, "successes": result["overall"]["successes"],
            "eval_git_commit": contract["git"]["commit"],
            "materialization_git_commit": (contract.get("adapter") or {}).get("materialization_git", {}).get("commit", ""),
            "eval_schema_version": result["schema_version"],
            "contract_reference": result["contract_reference"],
        })
        by_panel[tag] = current
    return success_rows, task_rows, metadata, by_panel


def paired_transition(left: dict, right: dict, *, strict: bool):
    assert left.keys() == right.keys()
    retained = gained = lost = 0
    same_mapping = True
    for key in left:
        a, b = left[key], right[key]
        mapping = ("env_seed", "policy_seed_root", "video_ordinal", "teacher_demo_indices")
        if any(a[k] != b[k] for k in mapping):
            same_mapping = False
        retained += a["success"] and b["success"]
        gained += not a["success"] and b["success"]
        lost += a["success"] and not b["success"]
    if strict:
        assert same_mapping, "formal paired panel mapping differs"
    return {"same_state_rng_video_mapping": int(same_mapping),
            "retained": retained if same_mapping else "",
            "gained": gained if same_mapping else "", "lost": lost if same_mapping else ""}


def validation_transitions(by_panel):
    rows = []
    for study, method, steps in (("old", "writer", OLD_WRITER_STEPS),
                                  ("new", "writer", NEW_WRITER_STEPS),
                                  ("new", "mtbc", NEW_MTBC_STEPS)):
        for left_step, right_step in zip(steps, steps[1:]):
            left, right = (by_panel[(study, method, "correct", s)] for s in (left_step, right_step))
            for suite, task_id in sorted({k[:2] for k in left}):
                a = {k: v for k, v in left.items() if k[:2] == (suite, task_id)}
                b = {k: v for k, v in right.items() if k[:2] == (suite, task_id)}
                paired = paired_transition(a, b, strict=(method == "writer"))
                rows.append({"study": study, "method": method, "from_step": left_step,
                             "to_step": right_step, "suite": suite, "task_id": task_id,
                             "global_task_id": SUITE_BASE[suite] + task_id,
                             "from_successes": sum(x["success"] for x in a.values()),
                             "to_successes": sum(x["success"] for x in b.values()), **paired})
    return rows


def shared_old_new(by_panel):
    old = by_panel[("old", "writer", "correct", 1200)]
    new = by_panel[("new", "writer", "correct", 1800)]
    shared = sorted({k[:2] for k in old} & {k[:2] for k in new})
    assert len(shared) == 5, shared
    rows = []
    for suite, task_id in shared:
        a = {k: v for k, v in old.items() if k[:2] == (suite, task_id)}
        b = {k: v for k, v in new.items() if k[:2] == (suite, task_id)}
        paired = paired_transition(a, b, strict=False)
        old_n, new_n = (sum(x["success"] for x in d.values()) for d in (a, b))
        rows.append({"suite": suite, "task_id": task_id,
                     "global_task_id": SUITE_BASE[suite] + task_id,
                     "old_step": 1200, "new_step": 1800,
                     "old_successes": old_n, "new_successes": new_n,
                     "delta_new_minus_old": new_n - old_n, **paired})
    return rows


def percentiles(values):
    x = sorted(values)
    return statistics.mean(x), statistics.median(x), x[math.ceil(0.95 * len(x)) - 1]


def training_records(old: Path, new: Path):
    old_metrics = read_jsonl(old / "continuation/main/training/metrics.jsonl")
    old_exposures = read_jsonl(old / "continuation/main/training/exposures.jsonl")
    new_metrics = read_jsonl(new / "training/writer/metrics.jsonl")
    new_exposures = read_jsonl(new / "training/writer/exposures.jsonl")
    originals = (("old", old_metrics, old_exposures, 2100),
                 ("new", new_metrics, new_exposures, 1800))
    task_ids = {study: {int(x["task"]) for x in exposures}
                for study, _, exposures, _ in originals}
    common = task_ids["old"] & task_ids["new"]
    assert len(task_ids["old"]) == 24 and len(task_ids["new"]) == 36 and len(common) == 20
    step_rows, exposure_rows, group_windows, occurrence_windows, task_groups = [], [], [], [], []
    metrics_by_study = {}
    exposures_by_study = {}
    for study, metrics, exposures, total in originals:
        assert len(metrics) == total and len(exposures) == total * 4
        assert [m["step"] for m in metrics] == list(range(1, total + 1))
        metrics_by_study[study] = {m["step"]: m for m in metrics}
        per_step = defaultdict(list)
        seen_tasks = set()
        for e in exposures:
            per_step[e["step"]].append(e)
            tid = int(e["task"])
            group = "common20" if tid in common else ("aux12" if tid >= 40 else f"{study}_only4")
            if tid not in seen_tasks:
                task_groups.append({"study": study, "suite": e["suite"],
                                    "global_task_id": tid, "task_group": group})
                seen_tasks.add(tid)
            exposure_rows.append({"study": study, "task_group": group,
                                  **{k: e.get(k, "") for k in EXPOSURE_FIELDS}})
        assert set(per_step) == set(range(1, total + 1))
        for step, batch in per_step.items():
            assert len(batch) == 4 and len({e["task"] for e in batch}) == 4, (study, step)
        exposures_by_study[study] = per_step
        for m in metrics:
            step_rows.append({"study": study,
                              "lr_applied": (metrics[m["step"] - 2]["lr_next"]
                                             if m["step"] > 1 else ""),
                              **{k: m.get(k, "") for k in STEP_FIELDS}})
        for bucketing, dest in (("step", group_windows), ("occurrence", occurrence_windows)):
            grouped = defaultdict(list)
            for e in exposures:
                tid = int(e["task"])
                group = "common20" if tid in common else ("aux12" if tid >= 40 else f"{study}_only4")
                value = int(e["step"]) if bucketing == "step" else int(e["occurrence"])
                start = ((value - 1) // 200) * 200 + 1 if bucketing == "step" else (value // 25) * 25
                grouped[(group, start)].append(e)
            for (group, start), items in sorted(grouped.items()):
                end = min(start + (199 if bucketing == "step" else 24),
                          total if bucketing == "step" else max(int(e["occurrence"]) for e in items))
                row = {"study": study, "task_group": group, "start": start, "end": end,
                       "exposures": len(items)}
                for name in ("flow_loss", "teaching_loss", "fm_lora_gradient_norm", "teaching_lora_gradient_norm"):
                    mean, median, p95 = percentiles([float(e[name]) for e in items])
                    row.update({f"{name}_mean": mean, f"{name}_median": median, f"{name}_p95": p95})
                dest.append(row)
    return (step_rows, exposure_rows, group_windows, occurrence_windows,
            task_groups, metrics_by_study, exposures_by_study)


def node_context(old: Path, new: Path, metadata: list[dict], metrics, exposures):
    lookup = {(x["study"], x["method"], x["arm"], x["step"]): x for x in metadata}
    contracts = {
        "old": read_json(old / "training/run_contract.json"),
        "new": read_json(new / "training/writer/run_contract.json"),
    }
    old_cont = read_json(old / "continuation/main/training/run_contract.json")
    old_tasks = {int(e["task"]) for batch in exposures["old"].values() for e in batch}
    new_tasks = {int(e["task"]) for batch in exposures["new"].values() for e in batch}
    common = old_tasks & new_tasks
    rows = []
    for study, steps, cycle in (("old", OLD_WRITER_STEPS, 6), ("new", NEW_WRITER_STEPS, 9)):
        for step in steps:
            m = metrics[study][step]
            recent = [e for s in range(max(1, step - cycle + 1), step + 1)
                      for e in exposures[study][s]]
            counts = Counter("common20" if int(e["task"]) in common else
                             ("aux12" if int(e["task"]) >= 40 else f"{study}_only4")
                             for e in recent)
            contract = old_cont if study == "old" and step > 1500 else contracts[study]
            meta = lookup[(study, "writer", "correct", step)]
            rows.append({
                "study": study, "step": step, "cycle_updates": cycle,
                "cycle_position_completed": step % cycle,
                "recent_window_updates": cycle,
                "recent_window_task_global_ids_in_order": ";".join(str(e["task"]) for e in recent),
                "last_update_task_global_ids": ";".join(str(e["task"]) for e in exposures[study][step]),
                "recent_window_common20_exposures": counts["common20"],
                "recent_window_study_only4_exposures": counts[f"{study}_only4"],
                "recent_window_aux12_exposures": counts["aux12"],
                "training_segment_start_checkpoint_step": (
                    0 if step == (900 if study == "old" else 200) else
                    (1500 if study == "old" and step > 1500 else
                     step - (300 if study == "old" else 200))),
                "lr_next": m["lr_next"], "mean_total_loss": m["mean_total_loss"],
                "lr_applied_last_update": metrics[study][step - 1]["lr_next"],
                "train_git_commit": contract["git"]["commit"],
                "eval_git_commit": meta["eval_git_commit"],
                "materialization_git_commit": meta["materialization_git_commit"],
                "frame_chunk": contract["config"]["observer"]["frame_chunk"],
                "world_size": contract["topology"]["world_size"],
                "validation_successes": meta["successes"],
            })
    return rows


def mtbc_steps(new: Path):
    rows = []
    for m in read_jsonl(new / "training/mtbc/metrics.jsonl"):
        rows.append({k: m.get(k, "") for k in (
            "optimizer_step", "applied_lr", "next_lr", "mean_action_loss",
            "gradient_norm_before_clip_max", "global_action_queries",
            "global_samples_per_task_this_step", "global_task_count_this_step",
            "step_seconds_max", "max_cuda_reserved_bytes")})
    assert len(rows) == 500 and [x["optimizer_step"] for x in rows] == list(range(1, 501))
    return rows


def writer_config_summary(old: Path, new: Path):
    configs = {
        "old": read_json(old / "training/run_contract.json"),
        "new": read_json(new / "training/writer/run_contract.json"),
    }
    fields = {
        "optimization": ("lr", "warmup_updates", "decay_updates", "decay_lr",
                         "tail_start_update", "tail_end_update", "tail_final_ratio",
                         "teaching_weight", "betas", "grad_clip"),
        "data": ("task_ids", "tasks_per_update", "queries_per_task",
                 "teaching_queries_per_task", "video_demos", "action_demos",
                 "teacher_video_seed", "sampler_seed"),
        "observer": ("frame_chunk", "camera_view", "horizon_read"),
    }
    rows = []
    for section, names in fields.items():
        for name in names:
            row = {"section": section, "field": name}
            for study, contract in configs.items():
                value = contract["config"][section][name]
                row[study] = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-study", type=Path, required=True)
    parser.add_argument("--new-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    success, tasks, metadata, by_panel = panel_records(args.old_study, args.new_study)
    steps, exposures, groups, occurrences, task_groups, metric_map, exposure_map = training_records(args.old_study, args.new_study)
    files = {
        "validation_success_rows.csv": success,
        "validation_per_task_nodes.csv": tasks,
        "validation_panel_provenance.csv": metadata,
        "validation_task_transitions.csv": validation_transitions(by_panel),
        "shared_held_old1200_new1800.csv": shared_old_new(by_panel),
        "writer_training_steps.csv": steps,
        "writer_training_task_exposures.csv": exposures,
        "writer_training_group_step_windows.csv": groups,
        "writer_training_group_occurrence_windows.csv": occurrences,
        "writer_training_task_groups.csv": task_groups,
        "writer_training_config_summary.csv": writer_config_summary(args.old_study, args.new_study),
        "writer_validation_node_context.csv": node_context(args.old_study, args.new_study, metadata, metric_map, exposure_map),
        "mtbc_training_steps.csv": mtbc_steps(args.new_study),
    }
    for name, rows in files.items():
        assert rows, name
        write_csv(args.output / name, list(rows[0]), rows)
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
