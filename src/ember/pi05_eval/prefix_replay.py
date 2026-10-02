"""Archived prefix evidence and saved-action cases in the canonical rollout."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
from ember.pi05_assets import Pi05EvaluationError

ROW_SCHEMA = "ember_frozen_prefix_episode_v1"


def validate_cases(cases, state_ids, task, contract) -> None:
    """Duplicate physical states are legal only in an explicit frozen case matrix."""
    import numpy as np

    registration = contract.get("frozen_case_matrix") or {}
    ids = [case["evidence"]["case_id"] for case in cases]
    cut = int(registration.get("cut_control_steps", -1))
    if (registration.get("schema_version") != "ember_saved_action_case_matrix_v1"
            or ids != registration.get("case_ids") or len(set(ids)) != len(ids)
            or len(cases) != len(state_ids) or len(cases) < 2
            or cut <= 0 or cut % int(contract["policy"]["replan_steps"])
            or registration.get("task") != [task["suite"], int(task["task_id"])]
            or any(int(s) != registration.get("init_state_id") for s in state_ids)
            or task["split_role"] != "train" or contract["role"] != "development_train"):
        raise Pi05EvaluationError("saved-action case matrix scope changed")
    for case in cases:
        actions = np.asarray(case["physical_prefix"])
        if (actions.shape != (cut, 7) or not np.isfinite(actions).all()
                or len(case["archived_chunks"]) != cut // 5
                or len(case["noise_seeds"]) != (int(task["horizon"]) + 4) // 5
                or case["noise_seeds"] != cases[0]["noise_seeds"]
                or case["contract"]["policy"] != contract["policy"]
                or case["contract"]["operator_read_write_scene"] != contract["operator_read_write_scene"]):
            raise Pi05EvaluationError("saved-action prefix/clock/scene changed")


def bind_case(slot, case) -> None:
    slot["episode_contract"] = case["contract"]
    slot["episode_adapter"] = case["prepared_adapter"]
    slot["frozen_case"] = case["evidence"]
    slot["saved_prefix"] = case


def record_case_prediction(group, chunks, seeds, contract, seconds) -> None:
    cut = (contract.get("frozen_case_matrix") or {}).get("cut_control_steps")
    if not group or not all("frozen_case" in slot and slot["steps"] == cut for slot in group):
        return
    from ember.pi05_source_checkpoint import write_json_atomic

    record = {"event": "first_actual_crossover_replan", "physical_batch": len(group),
              "case_ids": [slot["frozen_case"]["case_id"] for slot in group],
              "step": cut, "replan_indices": [slot["replan_index"] for slot in group],
              "noise_seeds": list(seeds), "normalized_shape": list(chunks.shape),
              "flow_steps": contract["policy"]["num_inference_steps"],
              "seconds": seconds, "execution_git": contract.get("git")}
    write_json_atomic(Path(contract["output_dir"]) / "first_consumer.json", record)
    import json

    print(json.dumps(record), flush=True)


def plan_saved_prefix(group, raw_inputs, processed, *, task, contract) -> bool:
    """Queue archived physical commands, retaining the absolute replan clock."""
    from ember.pi05_eval_contract import policy_noise_seed
    from ember.pi05_eval.trajectory_capture import record_replan

    selected = [slot for slot in group if "saved_prefix" in slot]
    if not selected:
        return False
    cut = int(contract["frozen_case_matrix"]["cut_control_steps"])
    before = [int(slot["steps"]) < cut for slot in selected]
    if not any(before):
        return False
    if len(selected) != len(group) or not all(before):
        raise Pi05EvaluationError("saved-prefix batch mixed incompatible control clocks")
    for slot, raw, value in zip(group, raw_inputs, processed, strict=True):
        start, index = int(slot["steps"]), int(slot["replan_index"])
        case = slot["saved_prefix"]
        if start != index * 5:
            raise Pi05EvaluationError("saved-prefix absolute noise index changed")
        seed = policy_noise_seed(int(contract["rng"]["inference_seed"]),
                                 str(task["suite"]), int(task["task_id"]),
                                 int(slot["init_state_id"]), index)
        if seed != case["noise_seeds"][index]:
            raise Pi05EvaluationError("saved-prefix noise does not match its original")
        plan = case["physical_prefix"][start:start + 5]
        record_replan(slot, raw, value, None, plan, command_kind="external_saved_action")
        slot.setdefault("archived_prefix_action_chunks", []).append(case["archived_chunks"][index])
        slot["action_plan"].extend(plan)
        slot["policy_noise_seeds"].append(seed)
        slot["replan_index"] += 1
    return True

def validate_row(row: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    if contract.get("approach_channel_intervention") is not None:
        from ember.pi05_eval.approach_channel_replay import validate_channel_row

        validate_channel_row(row, contract)
        return
    intervention = contract["frozen_prefix_intervention"]
    prefix = row.get("frozen_prefix_intervention")
    if not isinstance(prefix, Mapping) or prefix.get("schema_version") != ROW_SCHEMA:
        raise Pi05EvaluationError("frozen-prefix row evidence missing")
    key = f"{row['suite']}:{int(row['task_id'])}:{int(row['init_state_id'])}"
    reference = intervention["references"].get(key)
    cut = int(intervention["cut_control_steps"])
    terminal = bool(reference["success"] and int(reference["steps"]) <= cut) if reference else None
    trace = prefix.get("trace", {})
    path = Path(str(trace.get("path", "")))
    if (reference is None or prefix.get("reference") != reference
            or prefix.get("anchor") != intervention["anchor"]
            or prefix.get("follower") != intervention["follower"]
            or prefix.get("cut_control_steps") != cut
            or prefix.get("actual_prefix_steps") != min(cut, int(reference["steps"]))
            or prefix.get("prefix_terminal") is not terminal
            or prefix.get("follower_start_replan_index") != min(cut, int(reference["steps"])) // 5
            or (terminal and (row["steps"] != reference["steps"] or row["success"] is not True))
            or (not terminal and int(row["steps"]) <= cut)
            or not path.is_relative_to(Path(intervention["trace_root"]))
            or not path.is_file() or path.stat().st_size != int(trace.get("bytes", -1))
            or int(trace.get("steps", -1)) != int(row["steps"])):
        raise Pi05EvaluationError("frozen-prefix row contract changed")
