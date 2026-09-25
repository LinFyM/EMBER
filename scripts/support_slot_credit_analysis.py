#!/usr/bin/env python3
"""Mechanical completion of the registered four-model support-slot diagnostic."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

from ember.pi05_eval.crossed_video_predict import scaled_osc_actions
from ember.pi05_eval.trajectory_capture import validate_passive_trace_row
from ember.pi05_eval_contract import git_state, policy_noise_seed
from ember.writer.materialization import planned_episodes, selection_contract
from ember.writer.support_slot_credit import ARMS, authority, root


REPO = Path(__file__).resolve().parents[1]
STUDY = root()
OUT = STUDY / "analysis"
MODELS = ("P1155", *ARMS)
TASKS = {14: ("libero_object", 4), 21: ("libero_goal", 1)}
SPEC = authority()
SELECTION = selection_contract(role="development_train", task_ids=[14, 21], cardinality=1,
    arm="correct", mode="per_init_ordinal", seed=20260911,
    init_state_ids=tuple(range(50)), video_pool=tuple(range(50)))
EXPECTED = {(task, row["init_state_id"]): row for task in TASKS
            for row in planned_episodes(SELECTION, task)}
QUERY_ROOT = Path(SPEC["first_donor_function_probe"]["query_root"])


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(name: str, value: object) -> dict:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {"path": str(path), "bytes": path.stat().st_size}


def _write_lines(name: str, values: list[dict]) -> dict:
    path = OUT / name
    with path.open("x", encoding="utf-8") as handle:
        for row in values:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {"path": str(path), "bytes": path.stat().st_size, "rows": len(values)}


def _first_run(distance: np.ndarray, threshold: float) -> int | None:
    mask = (distance > threshold).astype(np.int32)
    if len(mask) < 5:
        return None
    hits = np.flatnonzero(np.convolve(mask, np.ones(5, dtype=np.int32), mode="valid") == 5)
    return int(hits[0]) if len(hits) else None


def _queries() -> dict[tuple[int, int], dict]:
    rows = [json.loads(line) for line in (QUERY_ROOT / "query_index.jsonl").read_text().splitlines()]
    selected = {(int(row["global_task_id"]), int(row["init_state_id"])): row for row in rows
                if row["global_task_id"] in TASKS and row["control_step"] == 0}
    if len(rows) != 300 or len(selected) != 100:
        raise ValueError("original t0 query authority is incomplete")
    return selected


def _training(commit: str) -> list[dict]:
    branches = []
    for arm in ARMS:
        path = STUDY / "training" / arm
        run = _json(path / "run_contract.json")
        done = _json(path / "completion.json")
        exposures = [json.loads(line) for line in (path / "exposures.jsonl").read_text().splitlines()]
        metrics = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
        if (run["mode"] != "formal" or run["git"]["commit"] != commit
                or run["support_slot_credit"]["arm"] != arm
                or run["training"]["source_trainable_parameters"] != 0
                or done["optimizer_updates"] != 1183 or done["branch_cursor"] != 28
                or len(exposures) != 112 or len(metrics) != 28
                or [row["step"] for row in metrics] != list(range(1156, 1184))
                or any(metric["total_grad_norm"] <= 0 for metric in metrics)
                or any(max(metric[name] for metric in metrics) <= 0 for name in (
                    "meta_grad_norm", "vl_meta_grad_norm", "text_meta_grad_norm"))):
            raise ValueError("support-slot full training branch or active gradient history incomplete")
        by_step = {macro: [row for row in exposures if row["step"] == macro]
                   for macro in range(1156, 1184)}
        if any(len(group) != 4 for group in by_step.values()):
            raise ValueError("support-slot training macro lost one task event")
        for macro, group in by_step.items():
            slot = [row for row in group if row["task"] in (76, 77)]
            if macro in SPEC["training"]["donor_macros"]:
                wanted_task = 76 if arm == "SWAP76" else 77
                wanted_gate = 0 if arm == "DROP77" else 1
                if (len(slot) != 1 or slot[0]["task"] != wanted_task
                        or slot[0]["credit_gate"] != wanted_gate
                        or slot[0]["effective_main_weight"] != wanted_gate * 0.25):
                    raise ValueError("support-slot donor event or current credit changed")
            elif any(row["credit_gate"] != 1 for row in group):
                raise ValueError("common event credit changed")
        for macro in (1160, 1183):
            checkpoint = path / "checkpoints" / f"macro_{macro:08d}"
            manifest = _json(checkpoint / "checkpoint_manifest.json")
            trainer = torch.load(checkpoint / "trainer_state.pt", map_location="meta",
                                 mmap=True, weights_only=True)
            if (manifest["next_macro"] != macro or manifest["world_size"] != 2
                    or set(manifest["files"]) != {"ecp.safetensors", "trainer_state.pt",
                                                 "rank_00_state.pt", "rank_01_state.pt"}
                    or trainer["next_macro"] != macro
                    or trainer["scheduler"]["last_epoch"] != macro
                    or trainer["sampler_state"]["branch_cursor"] != macro - 1155):
                raise ValueError("support-slot complete checkpoint or two-rank RNG identity missing")
        branches.append({"arm": arm, "run_contract": str(path / "run_contract.json"),
            "completion": str(path / "completion.json"),
            "checkpoints": [str(path / "checkpoints" / f"macro_{macro:08d}")
                            for macro in (1160, 1183)],
            "macro_updates": len(metrics), "events": len(exposures),
            "active_queries": 3024 if arm == "DROP77" else 3136})
    return branches


def _panels(queries: dict, commit: str) -> tuple[dict, list[dict], list[dict], list[dict], list[dict], dict]:
    rows, panels, raw_index, cases, geometry, first = {}, [], [], [], [], {}
    trace_paths = set()
    for stage, states in (("pilot", (0,)), ("remaining", tuple(range(1, 50)))):
        for model in MODELS:
            path = STUDY / "evaluation" / stage / model
            contract = _json(path / "run_contract.json")
            result = _json(path / "results.json")
            completion = _json(path / "launcher_completion.json")
            summary = _json(path / "run_summary.json")
            registration = contract["support_slot_credit"]
            if (registration["model"] != model or registration["stage"] != stage
                    or contract["git"]["commit"] != commit
                    or len(result["rows"]) != 2 * len(states)
                    or not completion["return_codes"]
                    or any(code != 0 for code in completion["return_codes"].values())
                    or summary["episodes"] != len(result["rows"])):
                raise ValueError("support-slot panel identity, row count or worker exit changed")
            panels.append({"model": model, "stage": stage, "rows": len(result["rows"]),
                "contract": str(path / "run_contract.json"), "results": str(path / "results.json"),
                "launcher_completion": str(path / "launcher_completion.json"),
                "gpu_indices": contract["parallel"]["physical_gpu_ids"],
                "worker_return_codes": completion["return_codes"],
                "wall_seconds": completion["wall_seconds"]})
            task_contracts = {(task["suite"], int(task["task_id"])): task for task in contract["tasks"]}
            for row in result["rows"]:
                identity = (row["suite"], int(row["task_id"]))
                task = next(global_id for global_id, key in TASKS.items() if key == identity)
                state = int(row["init_state_id"])
                key = (model, task, state)
                if state not in states or key in rows:
                    raise ValueError("support-slot evaluation has duplicate or unregistered row")
                validate_passive_trace_row(row, contract, task_contracts[identity])
                episode = EXPECTED[(task, state)]
                adapter = row["horizon_writer_lora"]
                if (any(adapter[field] != episode[field] for field in (
                        "condition_id", "teacher_demo_indices", "video_ordinal",
                        "paired_correct_demos", "paired_other_demos"))
                        or row["env_seed"] != 7 or row["policy_seed_root"] != 7
                        or row["policy_noise_seeds"][0] != policy_noise_seed(7, *identity, state, 0)):
                    raise ValueError("support-slot video, environment or policy pairing changed")
                capture = row["occupancy_trajectory"]
                full = state in (0, 25)
                trajectory_path = Path(capture["path"])
                if (capture["capture_level"] != ("full" if full else "compact")
                        or not trajectory_path.is_file()
                        or trajectory_path.stat().st_size != capture["bytes"]):
                    raise ValueError("registered full/compact case missing")
                trajectory = torch.load(trajectory_path, map_location="cpu", weights_only=False)
                first_chunk = trajectory["action_chunks"][0].numpy()
                applied = trajectory["executed_action_prefixes"][0].numpy()
                state8 = trajectory["states"][0].numpy()
                if (first_chunk.shape != (1, 50, 7) or applied.ndim != 2
                        or applied.shape[1] != 7 or len(applied) != min(5, row["steps"])
                        or state8.shape != (8,) or not np.isfinite(first_chunk).all()):
                    raise ValueError("first real replan action or state is incomplete")
                first[key] = {"normalized": first_chunk[0], "environment_first5": applied,
                              "state8": state8}
                if full:
                    images = trajectory["observations"][0]
                    if not {"observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb"} <= set(images):
                        raise ValueError("fixed full case lost one of its two actual cameras")
                    cases.append({"model": model, "task": task, "state": state,
                        "trajectory": capture, "trace": row["continuous_control_trace"]["trace"],
                        "predicates": row["stage_predicates"], "success": bool(row["success"])})
                trace_info = row["continuous_control_trace"]["trace"]
                trace_path = Path(trace_info["path"])
                if trace_path in trace_paths:
                    raise ValueError("passive trace path reused across episodes")
                trace_paths.add(trace_path)
                with np.load(trace_path, allow_pickle=False) as trace:
                    positions = trace["body_positions"].astype(float)
                    names = trace["body_names"].tolist()
                    eef = trace["eef_pos"].astype(float)
                    actions = trace["actions"].astype(float)
                    if (positions.shape[0] != row["steps"] + 1 or actions.shape != (row["steps"], 7)
                            or not np.allclose(actions[:len(applied)], applied)):
                        raise ValueError("passive T+1 history differs from executed action capture")
                    target = row["continuous_control_trace"]["goal_predicates"][0][1]
                    if target not in names:
                        raise ValueError("target body missing from actual BDDL registry")
                    target_index = names.index(target)
                    objects = [entry["name"] for entry in row["continuous_control_trace"]["body_registry"]
                               if entry["kind"] == "object" and entry["name"] != target]
                    point = min(25, int(row["steps"]))
                    distance = np.linalg.norm(positions - positions[:1], axis=-1)
                    item = {"model": model, "task": task, "state": state, "target": target,
                        "objects": objects, "control_samples": int(row["steps"] + 1),
                        "eef_to_target_xy_at_t25_m": float(np.linalg.norm(
                            eef[point, :2] - positions[point, target_index, :2])),
                        "target_lift_at_t25_m": float(positions[point, target_index, 2]
                                                    - positions[0, target_index, 2]),
                        "trace": trace_info}
                    if task == 21 and "plate_1" not in objects:
                        raise ValueError("Goal plate distractor is absent from actual scene")
                    for cm in (1, 2, 3):
                        onset = {name: _first_run(distance[:, names.index(name)], cm / 100)
                                 for name in (target, *objects)}
                        item[f"onset_{cm}cm"] = onset
                        if task == 21:
                            plate = onset["plate_1"]
                            item[f"plate_before_target_{cm}cm"] = (plate is not None and
                                (onset[target] is None or plate < onset[target]))
                            item[f"other_object_before_target_{cm}cm"] = [
                                name for name in objects if name != "plate_1" and onset[name] is not None
                                and (onset[target] is None or onset[name] < onset[target])]
                    geometry.append(item)
                rows[key] = row
                raw_index.append({"model": model, "task": task, "state": state,
                    "result_path": str(path / "results.json"), "condition_id": adapter["condition_id"],
                    "teacher_demo_indices": adapter["teacher_demo_indices"],
                    "trajectory": capture, "trace": trace_info})
    if len(rows) != 400 or len(trace_paths) != 400 or len(cases) != 16 or len(panels) != 8:
        raise ValueError("four-model rollout, trace, case or panel matrix incomplete")
    for task in TASKS:
        for state in range(50):
            reference = rows[(MODELS[0], task, state)]
            for model in MODELS[1:]:
                candidate = rows[(model, task, state)]
                if (reference["language"], reference["env_seed"], reference["policy_seed_root"],
                    reference["horizon_writer_lora"]["condition_id"]) != (
                    candidate["language"], candidate["env_seed"], candidate["policy_seed_root"],
                    candidate["horizon_writer_lora"]["condition_id"]):
                    raise ValueError("four-model task, language or teacher pairing changed")
                count = min(len(reference["policy_noise_seeds"]), len(candidate["policy_noise_seeds"]))
                if (reference["policy_noise_seeds"][:count] != candidate["policy_noise_seeds"][:count]
                        or np.max(np.abs(first[(model, task, state)]["state8"]
                                         - first[(MODELS[0], task, state)]["state8"])) >= 1e-3):
                    raise ValueError("four-model initial state or stateless policy noise differs")
    return rows, panels, raw_index, cases, geometry, first


def _success(rows: dict) -> tuple[list[dict], list[dict], list[dict]]:
    contrasts, pairs, sets = [], [], []
    rng = np.random.default_rng(SPEC["evaluation"]["bootstrap"]["seed"])
    for task in TASKS:
        matrix = np.asarray([[int(rows[(model, task, state)]["success"]) for model in MODELS]
                             for state in range(50)], dtype=np.int8)
        draws = rng.integers(0, 50, size=(20000, 50))
        for model, successes in zip(MODELS, matrix.T, strict=True):
            sets.append({"task": task, "model": model, "successes": int(successes.sum()),
                         "success_state_ids": np.flatnonzero(successes).tolist()})
        for contrast in SPEC["evaluation"]["contrasts"]:
            values = sum(coef * matrix[:, MODELS.index(model)]
                         for model, coef in contrast["terms"].items())
            boot = values[draws].mean(axis=1) * 100
            contrasts.append({"task": task, "id": contrast["id"], "terms": contrast["terms"],
                "point_pp": float(values.mean() * 100),
                "ci95_pp": np.quantile(boot, (0.025, 0.975)).tolist(),
                "bootstrap_resamples": 20000, "seed": 20260925,
                "training_seeds": 1, "multiplicity_adjusted": False})
        for reference, candidate in itertools.combinations(MODELS, 2):
            x, y = matrix[:, MODELS.index(reference)], matrix[:, MODELS.index(candidate)]
            kept = np.flatnonzero((x == 1) & (y == 1)).tolist()
            gained = np.flatnonzero((x == 0) & (y == 1)).tolist()
            lost = np.flatnonzero((x == 1) & (y == 0)).tolist()
            union = int(((x == 1) | (y == 1)).sum())
            pairs.append({"task": task, "reference": reference, "candidate": candidate,
                "retained": len(kept), "gained": len(gained), "lost": len(lost),
                "retained_states": kept, "gained_states": gained, "lost_states": lost,
                "churn": len(gained) + len(lost), "jaccard": len(kept) / union if union else None})
    if len(contrasts) != 12 or len(pairs) != 12:
        raise ValueError("registered twelve task-level contrasts or success sets missing")
    return contrasts, pairs, sets


def _functions(queries: dict, first: dict) -> tuple[list[dict], list[dict]]:
    final, first_slot = [], []
    for task in TASKS:
        for state in range(50):
            query = queries[(task, state)]
            values = [first[(model, task, state)]["environment_first5"] for model in MODELS]
            if any(value.shape != (5, 7) for value in values):
                raise ValueError("terminal first replan must have five real actions")
            osc = scaled_osc_actions(np.stack([
                np.concatenate((value, np.zeros((45, 7), dtype=np.float32)))
                for value in values]), query["osc_channel_audit"])
            full = {model: osc[index, :, :2].tolist() for index, model in enumerate(MODELS)}
            final.append({"task": task, "state": state, "query_id": query["query_id"],
                "first5_scaled_OSC_xy": full,
                "add_swap_minus_drop": (osc[2, :, :2] - osc[3, :, :2]).tolist(),
                "retained_keep_minus_drop": (osc[1, :, :2] - osc[3, :, :2]).tolist(),
                "replace_swap_minus_keep": (osc[2, :, :2] - osc[1, :, :2]).tolist()})
    prediction_rows = [json.loads(line) for line in
                       (STUDY / "prediction/first_slot/rows.jsonl").read_text().splitlines()]
    if len(prediction_rows) != 30 or len({(r["model"], r["query_id"]) for r in prediction_rows}) != 30:
        raise ValueError("first-slot prediction rows incomplete")
    by_query = {}
    for row in prediction_rows:
        by_query.setdefault(row["query_id"], []).append(row)
    for task in TASKS:
        for state in (0, 10, 20, 30, 40):
            query = queries[(task, state)]
            matching = by_query[query["query_id"]]
            if {row["model"] for row in matching} != set(ARMS):
                raise ValueError("first-slot model triple missing")
            path = Path(matching[0]["prediction"])
            with np.load(path, allow_pickle=False) as data:
                if (data["environment"].shape != (3, 50, 7)
                        or data["normalized"].shape != (3, 50, 7)
                        or data["osc_first5"].shape != (3, 5, 6)
                        or data["models"].tolist() != list(ARMS)):
                    raise ValueError("first-slot full50 prediction artifact changed")
                osc = data["osc_first5"][:, :, :2].astype(float)
            for row in matching:
                if row["prediction"] != str(path) or row["policy_noise_seed"] != query["policy_noise_seed"]:
                    raise ValueError("first-slot query/seed/provenance pairing changed")
            add, retained, replace = osc[1] - osc[2], osc[0] - osc[2], osc[1] - osc[0]
            residual = float(np.max(np.abs(replace - (add - retained))))
            entry = {"task": task, "state": state, "query_id": query["query_id"],
                "add_swap_minus_drop_xy": add.tolist(),
                "retained_keep_minus_drop_xy": retained.tolist(),
                "replace_swap_minus_keep_xy": replace.tolist(),
                "identity_max_abs": residual}
            if task == 21:
                with np.load(query["query_path"], allow_pickle=False) as data:
                    names = data["object_names"].tolist()
                    positions = data["object_positions"].astype(float)
                vector = positions[names.index("plate_1"), :2] - positions[names.index("akita_black_bowl_1"), :2]
                vector /= np.linalg.norm(vector)
                entry["initial_bowl_to_plate_unit_xy"] = vector.tolist()
                entry["mean_projection_add"] = float(np.mean(add @ vector))
                entry["mean_projection_retained"] = float(np.mean(retained @ vector))
                entry["mean_projection_replace"] = float(np.mean(replace @ vector))
            first_slot.append(entry)
    if len(final) != 100 or len(first_slot) != 10:
        raise ValueError("terminal or first-slot shared-query function matrix incomplete")
    return final, first_slot


def _fm() -> tuple[list[dict], list[dict]]:
    path = STUDY / "prediction/donor_fm/rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    keys = {(row["task"], row["model"], row["query_index"]) for row in rows}
    if (len(rows) != len(keys) or len(rows) != 128
            or keys != {(task, model, index) for task in (76, 77)
                        for model in MODELS for index in range(16)}):
        raise ValueError("donor FM 128-row registration incomplete")
    summary = []
    for task in (76, 77):
        common = {index: (row["query_demo"], row["query_frame"], row["policy_rng_seed"])
                  for row in rows if row["task"] == task and row["model"] == MODELS[0]
                  for index in (row["query_index"],)}
        for model in MODELS:
            chosen = [row for row in rows if row["task"] == task and row["model"] == model]
            if (len(chosen) != 16 or any(row["query_demo"] not in (47, 48, 49)
                    or (row["query_demo"], row["query_frame"], row["policy_rng_seed"])
                       != common[row["query_index"]] or row["gradients"] is not False
                    for row in chosen)):
                raise ValueError("donor FM query, RNG or no-gradient wall changed")
            summary.append({"task": task, "model": model,
                            "mean_FM": float(np.mean([row["loss"] for row in chosen]))})
    return rows, summary


def main() -> None:
    state = git_state(REPO)
    if state["branch"] or state["dirty_paths"] or OUT.exists():
        raise ValueError("registered analysis needs the clean frozen implementation and new output")
    queries = _queries()
    branches = _training(state["commit"])
    rows, panels, raw_index, cases, geometry, first = _panels(queries, state["commit"])
    contrasts, pairs, sets = _success(rows)
    terminal_function, first_slot = _functions(queries, first)
    fm_rows, fm_summary = _fm()
    outputs = {
        "training": _write("training.json", branches),
        "panels": _write("panels.json", panels),
        "raw_rows": _write_lines("raw_rows.jsonl", [
            {"model": model, "task": task, "state": state, "row": rows[(model, task, state)]}
            for model in MODELS for task in TASKS for state in range(50)]),
        "raw_index": _write_lines("raw_index.jsonl", raw_index),
        "case_index": _write("case_index.json", cases),
        "geometry": _write_lines("geometry.jsonl", geometry),
        "contrasts": _write("contrasts.json", contrasts),
        "paired_sets": _write("paired_sets.json", pairs),
        "success_sets": _write("success_sets.json", sets),
        "terminal_first_replan": _write_lines("terminal_first_replan.jsonl", terminal_function),
        "first_slot_function": _write("first_slot_function.json", first_slot),
        "donor_fm_rows": {"path": str(STUDY / "prediction/donor_fm/rows.jsonl"),
                           "rows": len(fm_rows)},
        "donor_fm_summary": _write("donor_fm_summary.json", fm_summary),
    }
    banks = {phase: {model: str(STUDY / "materialization" / phase / model / "manifest.json")
                     for model in (ARMS if phase == "first_slot" else MODELS)}
             for phase in ("final", "first_slot", "donor_fm")}
    if any(not Path(path).is_file() for phase in banks.values() for path in phase.values()):
        raise ValueError("registered final, first-slot or donor bank missing")
    _write("completion.json", {"schema_version": "ember_support_slot_completion_v1",
        "study_id": SPEC["study_id"], "implementation_commit": state["commit"],
        "status": "registered_complete", "training_macros": sum(row["macro_updates"] for row in branches),
        "rollouts": len(rows),
        "full_cases": len(cases), "continuous_traces": len(geometry),
        "first_slot_predictions": 30, "donor_fm_rows": len(fm_rows),
        "task_contrasts": len(contrasts), "pair_rows": len(pairs),
        "no_additional_training_or_evaluation_authorized": True,
        "parent_training_commit": SPEC["parent"]["training_commit"],
        "bank_manifests": banks, "outputs": outputs})


if __name__ == "__main__":
    main()
