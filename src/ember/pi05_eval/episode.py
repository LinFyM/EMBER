"""Canonical LIBERO episode initialization and BDDL predicate tracking."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np

from ember.eval_adapters import episode_adapter_fields
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.exploration import episode_exploration_fields
from ember.pi05_eval.trajectory_capture import initialize_capture, save_capture


def stage_predicate_snapshot(
    env: Any, states: Sequence[Sequence[str]] | None = None,
) -> tuple[tuple[tuple[str, ...], ...], tuple[bool, ...]]:
    owner = getattr(env, "env", env)
    raw_states = (
        tuple(tuple(str(value) for value in state) for state in states)
        if states is not None
        else tuple(tuple(str(value) for value in state)
                   for state in owner.parsed_problem["goal_state"])
    )
    if not raw_states or not callable(getattr(owner, "_eval_predicate", None)):
        raise Pi05EvaluationError("stage diagnosis requires LIBERO BDDL predicates")
    return raw_states, tuple(bool(owner._eval_predicate(state)) for state in raw_states)


def update_stage_predicates(env: Any, slot: dict[str, Any]) -> None:
    states, values = stage_predicate_snapshot(env, slot["stage_predicate_states"])
    slot["stage_predicate_ever"] = tuple(
        before or current
        for before, current in zip(slot["stage_predicate_ever"], values, strict=True)
    )
    slot["stage_predicate_peak"] = max(int(slot["stage_predicate_peak"]), sum(values))
    if values != slot["stage_predicate_last"]:
        slot["stage_predicate_transitions"].append(
            {"step": int(slot["steps"]), "satisfied": list(values)}
        )
        slot["stage_predicate_last"] = values
    slot["stage_predicate_states"] = states


def start_fixed_episode(
    *, env: Any, init_state_id: int, init_states: Any, task: Mapping[str, Any],
    contract: Mapping[str, Any], root_seed: int, dummy: np.ndarray,
    task_adapter: Any | None, capture_level: str | None,
) -> dict[str, Any]:
    env.seed(root_seed)
    env.reset()
    observation = env.set_init_state(init_states[init_state_id])
    for _ in range(int(contract["environment"]["dummy_settling_steps"])):
        observation, _, _, _ = env.step(dummy)
    prepared = None
    if task_adapter is not None:
        prepared = task_adapter.prepare_episode(
            suite=str(task["suite"]), task_id=int(task["task_id"]),
            init_state_id=init_state_id,
        )
    slot = {
        "init_state_id": init_state_id, "obs": observation, "steps": 0,
        "replan_index": 0, "policy_noise_seeds": [], "action_plan": deque(),
        "started": time.monotonic(),
    }
    if prepared is not None:
        slot["episode_adapter"] = prepared
    initialize_capture(slot, capture_level)
    stage_contract = contract.get("diagnostic_stage_predicates")
    if stage_contract is not None and (
        not stage_contract.get("full_conditions_only") or capture_level == "full"
    ):
        states, values = stage_predicate_snapshot(env)
        slot.update({
            "stage_predicate_states": states, "stage_predicate_last": values,
            "stage_predicate_ever": values, "stage_predicate_peak": sum(values),
            "stage_predicate_transitions": [{"step": 0, "satisfied": list(values)}],
        })
    return slot


def finish_episode_row(
    *, slot: Mapping[str, Any], task: Mapping[str, Any], contract: Mapping[str, Any],
    task_adapter: Any | None, worker_started: float,
) -> dict[str, Any]:
    root_seed = int(contract["rng"]["inference_seed"])
    finished = time.monotonic()
    row = {
        "suite": task["suite"], "task_id": int(task["task_id"]),
        "split_role": task["split_role"], "language": task["language"],
        "init_state_id": int(slot["init_state_id"]),
        "env_seed": root_seed, "policy_seed_root": root_seed,
        "policy_noise_seeds": list(slot["policy_noise_seeds"]),
        "success": bool(slot["episode_done"]), "steps": int(slot["steps"]),
        "wall_seconds": finished - float(slot["started"]),
        "finished_at": finished - worker_started,
    }
    row.update(episode_exploration_fields(contract, slot))
    if contract.get("frozen_prefix_intervention") is not None:
        from ember.pi05_eval.prefix_replay import finish_trace

        row["frozen_prefix_intervention"] = finish_trace(slot, task, contract)
    if "stage_predicate_states" in slot:
        row["stage_predicates"] = {
            "schema_version": "ember_pi05_stage_predicate_episode_v1",
            "predicates": [list(state) for state in slot["stage_predicate_states"]],
            "transitions": slot["stage_predicate_transitions"],
            "ever_satisfied": list(slot["stage_predicate_ever"]),
            "final_satisfied": list(slot["stage_predicate_last"]),
            "peak_satisfied_count": int(slot["stage_predicate_peak"]),
        }
    trajectory = save_capture(contract.get("diagnostic_occupancy_capture"), task, slot,
                              success=bool(slot["episode_done"]))
    if trajectory is not None:
        row["occupancy_trajectory"] = trajectory
    row.update(episode_adapter_fields(contract, task_adapter, slot.get("episode_adapter")))
    return row
