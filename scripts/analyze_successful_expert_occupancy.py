#!/usr/bin/env python3
"""Compare task-expert action and JVP responses on successful occupancy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.functional_adaptation.functional_response import (
    pi05_flow_action_jvp_response,
)
from ember.lora import identity_lora_state
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import load_policy
from ember.reward.occupancy_panel import complete_successful_expert_occupancy_batch
from ember.reward.rollout import (
    RewardTrajectory,
    query_successful_expert_occupancy_actions,
)
from ember.writer.functional import prepare_frozen_writer_policy


REPO_ROOT = Path(__file__).resolve().parents[1]
SELECTION_SCHEMA = "ember_successful_expert_occupancy_selection_v1"
CAPTURE_SCHEMA = "ember_successful_expert_occupancy_capture_v1"
SHARD_SCHEMA = "ember_successful_expert_occupancy_response_shard_v1"
ANALYSIS_SCHEMA = "ember_successful_expert_occupancy_response_analysis_v1"


def _repo_reference(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        parts = path.parts
        if "runs" in parts:
            return str(Path(*parts[parts.index("runs") :]))
    return path.name


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    indexed = {
        (str(row["suite"]), int(row["task_id"]), int(row["init_state_id"])): dict(row)
        for row in rows
    }
    if len(indexed) != len(rows):
        raise ValueError("successful occupancy rows are not unique")
    return indexed


def _valid_action_steps(steps: int, replans: int, replan_steps: int) -> tuple[int, ...]:
    values = tuple(
        min(replan_steps, max(steps - index * replan_steps, 0))
        for index in range(replans)
    )
    if not values or min(values) <= 0 or sum(values) != steps:
        raise ValueError("captured trajectory cannot reconstruct executed prefixes")
    return values


def _load_trajectory(
    selection: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    replan_steps: int,
) -> RewardTrajectory:
    record = row.get("occupancy_trajectory", {})
    path = Path(str(record.get("path", "")))
    if not path.is_file() or path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError("successful occupancy trajectory sidecar changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    observations = tuple(payload.get("observations", ()))
    actions = tuple(payload.get("action_chunks", ()))
    seeds = tuple(int(value) for value in payload.get("policy_noise_seeds", ()))
    steps = int(payload.get("steps", -1))
    if (
        payload.get("schema_version") != "ember_writer_occupancy_trajectory_v1"
        or not bool(payload.get("success"))
        or len(observations) != len(actions)
        or len(observations) != len(seeds)
        or len(observations) != int(record.get("replans", -1))
    ):
        raise ValueError("successful occupancy trajectory layout changed")
    return RewardTrajectory(
        suite=str(selection["suite"]),
        task_id=int(selection["task_id"]),
        global_task_id=int(selection["global_task_id"]),
        adaptation_seed=0,
        rollout_cursor=int(selection["init_state_id"]),
        env_seed=int(row["env_seed"]),
        policy_seed_root=int(row["policy_seed_root"]),
        success=True,
        steps=steps,
        reward_sum=1.0,
        dummy_settling_steps=10,
        policy_noise_seeds=seeds,
        observations=observations,
        action_chunks=actions,
        valid_action_steps=_valid_action_steps(steps, len(observations), replan_steps),
    )


def _stage_at_replans(
    row: Mapping[str, Any], selected_indices: Sequence[int], *, replan_steps: int
) -> tuple[list[list[bool]], list[bool]]:
    stage = row["stage_predicates"]
    transitions = tuple(stage["transitions"])
    values, full = [], []
    for index in selected_indices:
        step = int(index) * replan_steps
        current = transitions[0]["satisfied"]
        for transition in transitions[1:]:
            if int(transition["step"]) > step:
                break
            current = transition["satisfied"]
        state = [bool(value) for value in current]
        values.append(state)
        full.append(all(state))
    return values, full


def _selected_action_delta(
    actions: Mapping[str, Sequence[Sequence[torch.Tensor]]],
    selected: Sequence[Sequence[int]],
) -> torch.Tensor:
    trajectories = []
    for trajectory, indices in enumerate(selected):
        trajectories.append(
            torch.cat(
                [
                    actions["expert"][trajectory][index].float()
                    - actions["student"][trajectory][index].float()
                    for index in indices
                ],
                dim=0,
            )
        )
    result = torch.stack(trajectories)
    if result.shape != (2, 8, 50, 7) or not torch.isfinite(result).all():
        raise ValueError("selected action response shape changed")
    return result


def _selected_jvp_delta(
    policy: torch.nn.Module,
    identity: Mapping[str, torch.Tensor],
    expert: Mapping[str, torch.Tensor],
    lora: Any,
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    rows = []
    for index in range(int(batch["policy_noise_seed"].shape[0])):
        row = {
            name: value[index : index + 1]
            for name, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        seed = int(batch["policy_noise_seed"][index])
        source = pi05_flow_action_jvp_response(
            policy, identity, lora, row, policy_seed=seed
        )
        task = pi05_flow_action_jvp_response(
            policy, expert, lora, row, policy_seed=seed
        )
        rows.append((task - source).detach().to(device="cpu"))
    result = torch.cat(rows).reshape(2, 8, 50, 32)
    if not torch.isfinite(result).all():
        raise ValueError("selected JVP response is nonfinite")
    return result


def run_shard(args: argparse.Namespace) -> None:
    panel = args.panel_run.resolve()
    selection = read_json(args.selection.resolve())
    results = read_json(panel / "results.json")
    contract = read_json(panel / "run_contract.json")
    capture = contract.get("diagnostic_occupancy_capture", {})
    state = git_state(REPO_ROOT)
    if (
        selection.get("schema_version") != SELECTION_SCHEMA
        or capture.get("schema_version") != CAPTURE_SCHEMA
        or results.get("mode") != "formal"
        or results.get("role") != "nonheld_meta_train"
        or int(results.get("overall", {}).get("successes", -1)) != 8
        or int(results.get("overall", {}).get("episodes", -1)) != 8
        or contract.get("git", {}).get("dirty_paths") != []
        or not git_state_is_clean_pushed_or_frozen_authority(state)
    ):
        raise ValueError("successful occupancy formal authority changed")
    selected = [
        dict(row) for row in selection["rows"] if int(row["task_id"]) == args.task_id
    ]
    if len(selected) != 2 or {row["category"] for row in selected} != {
        "gained",
        "retained_success",
    }:
        raise ValueError("successful occupancy task selection changed")
    indexed = _index_rows(results["rows"])
    rows = [
        indexed[(str(row["suite"]), int(row["task_id"]), int(row["init_state_id"]))]
        for row in selected
    ]
    if not all(bool(row["success"]) for row in rows):
        raise ValueError("all preregistered direct rows must reproduce success")
    replan_steps = int(contract["policy"]["replan_steps"])
    trajectories = tuple(
        _load_trajectory(record, row, replan_steps=replan_steps)
        for record, row in zip(selected, rows, strict=True)
    )

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source_config = read_json(REPO_ROOT / "configs/pi05_source_base_v1.json")
    policy = load_policy(
        args.source_checkpoint.resolve() / "policy", source_config, device
    )
    lora = load_pi05_lora_contract(REPO_ROOT / "configs/pi05_lora_v1.json")
    prepare_frozen_writer_policy(policy, lora)
    identity = identity_lora_state(lora, device=device)
    adapter = {
        (str(row["suite"]), int(row["task_id"])): row
        for row in contract["adapter"]["tasks"]
    }[(str(selected[0]["suite"]), args.task_id)]
    expert = load_file(
        str(Path(adapter["checkpoint"]) / "adapter.safetensors"),
        device=str(device),
    )
    actions, requery = query_successful_expert_occupancy_actions(
        policy=policy,
        lora_contract=lora,
        identity_state=identity,
        trajectories=trajectories,
        expert_lora=expert,
        student_lora=identity,
        device=device,
        microbatch_size=args.microbatch_size,
        num_inference_steps=10,
    )
    batch, trajectory_ids, occupancy = complete_successful_expert_occupancy_batch(
        trajectories,
        actions,
        strata_per_trajectory=8,
        device=device,
    )
    selected_indices = occupancy["selected_replan_indices"]
    action_delta = _selected_action_delta(actions, selected_indices)
    jvp_delta = _selected_jvp_delta(policy, identity, expert, lora, batch)
    stage_values, stage_full = [], []
    for row, indices in zip(rows, selected_indices, strict=True):
        values, full = _stage_at_replans(row, indices, replan_steps=replan_steps)
        stage_values.append(values)
        stage_full.append(full)
    payload = {
        "schema_version": SHARD_SCHEMA,
        "analysis_git": state,
        "panel_run": _repo_reference(panel),
        "panel_contract_reference": contract["contract_reference"],
        "task_id": args.task_id,
        "suite": selected[0]["suite"],
        "trajectories": [
            {
                "init_state_id": int(record["init_state_id"]),
                "category": record["category"],
                "steps": int(row["steps"]),
                "replans": len(trajectory.observations),
                "selected_replan_indices": list(indices),
                "stage_predicates": row["stage_predicates"]["predicates"],
                "selected_stage_values": values,
                "selected_full_conjunction": full,
            }
            for record, row, trajectory, indices, values, full in zip(
                selected,
                rows,
                trajectories,
                selected_indices,
                stage_values,
                stage_full,
                strict=True,
            )
        ],
        "occupancy": occupancy,
        "requery": requery,
        "trajectory_ids": trajectory_ids.detach().to(device="cpu"),
        "action_delta": action_delta,
        "jvp_delta": jvp_delta,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        json.dumps(
            {
                "task_id": args.task_id,
                "selected_states": int(action_delta.shape[0] * action_delta.shape[1]),
                "expert_requery_rms": requery["stored_expert_to_matched_requery_rms"],
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _pairwise(values: torch.Tensor) -> dict[str, Any]:
    flat = values.double().flatten(1)
    norms = flat.square().sum(dim=1).sqrt()
    cosine = (flat @ flat.T) / (norms[:, None] * norms[None, :]).clamp_min(1e-24)
    distance = torch.cdist(flat, flat) / (
        (norms[:, None] * norms[None, :]).sqrt().clamp_min(1e-24)
    )
    cosine.fill_diagonal_(float("-inf"))
    distance.fill_diagonal_(float("inf"))
    return {
        "cosine": cosine,
        "relative_distance": distance,
        "cosine_nearest": cosine.argmax(dim=1),
        "distance_nearest": distance.argmin(dim=1),
        "rms": flat.square().mean(dim=1).sqrt(),
    }


def _family_summary(
    name: str,
    values: torch.Tensor,
    labels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    full = _pairwise(values)
    early = _pairwise(values[:, :4])
    task_rows = []
    for start in range(0, len(labels), 2):
        partner = (start + 1, start)
        full_mutual = all(
            int(full["cosine_nearest"][row]) == peer
            for row, peer in zip((start, start + 1), partner, strict=True)
        )
        early_mutual = all(
            int(early["cosine_nearest"][row]) == peer
            for row, peer in zip((start, start + 1), partner, strict=True)
        )
        early_nonfinal = all(
            not any(bool(value) for value in labels[row]["selected_full_conjunction"][:4])
            for row in (start, start + 1)
        )
        task_rows.append(
            {
                "task_id": int(labels[start]["task_id"]),
                "full_same_task_mutual_cosine_nearest": full_mutual,
                "early_same_task_mutual_cosine_nearest": early_mutual,
                "early_has_no_full_goal_conjunction": early_nonfinal,
                "passes": full_mutual and early_mutual and early_nonfinal,
            }
        )
    passed = sum(bool(row["passes"]) for row in task_rows)

    def matrix(tensor: torch.Tensor) -> list[list[float | None]]:
        return [
            [
                None if row == column else float(tensor[row, column])
                for column in range(len(labels))
            ]
            for row in range(len(labels))
        ]

    def indices(tensor: torch.Tensor) -> list[int]:
        return [int(value) for value in tensor]

    def values(tensor: torch.Tensor) -> list[float]:
        return [float(value) for value in tensor]

    return {
        "family": name,
        "primary_metric": "mutual nearest neighbor by cosine of ordered source-subtracted response",
        "early_nonfinal_rule": "the first four of eight progress strata must also be mutual nearest while neither trajectory has completed the BDDL goal conjunction",
        "passing_tasks": passed,
        "required_tasks": 3,
        "passes": passed >= 3,
        "per_task": task_rows,
        "full": {
            "cosine": matrix(full["cosine"]),
            "relative_distance": matrix(full["relative_distance"]),
            "cosine_nearest": indices(full["cosine_nearest"]),
            "distance_nearest": indices(full["distance_nearest"]),
            "response_rms": values(full["rms"]),
        },
        "early": {
            "cosine": matrix(early["cosine"]),
            "relative_distance": matrix(early["relative_distance"]),
            "cosine_nearest": indices(early["cosine_nearest"]),
            "distance_nearest": indices(early["distance_nearest"]),
            "response_rms": values(early["rms"]),
        },
    }


def aggregate(args: argparse.Namespace) -> None:
    shards = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.shard
    ]
    if len(shards) != 4 or any(
        row.get("schema_version") != SHARD_SCHEMA for row in shards
    ):
        raise ValueError("successful occupancy response shards changed")
    shards.sort(key=lambda row: int(row["task_id"]))
    if len({int(row["task_id"]) for row in shards}) != 4:
        raise ValueError("successful occupancy response tasks changed")
    labels = []
    for shard in shards:
        for trajectory in shard["trajectories"]:
            labels.append({"task_id": int(shard["task_id"]), **trajectory})
    action = torch.cat([row["action_delta"] for row in shards])
    jvp = torch.cat([row["jvp_delta"] for row in shards])
    action_summary = _family_summary("denoised_action", action, labels)
    jvp_summary = _family_summary("exact_action_jvp", jvp, labels)
    decision = (
        "advance_action_successful_on_policy_manifold"
        if action_summary["passes"]
        else "do_not_advance_response_family"
    )
    evidence = {
        "schema_version": ANALYSIS_SCHEMA,
        "panel_run": shards[0]["panel_run"],
        "panel_contract_reference": shards[0]["panel_contract_reference"],
        "analysis_git": shards[0]["analysis_git"],
        "validity": {
            "preregistered_rows": 8,
            "reproduced_successes": 8,
            "row_replacement": False,
            "held_data_use": False,
            "training_gradient_use": False,
        },
        "labels": labels,
        "action": action_summary,
        "jvp": jvp_summary,
        "expert_requery_rms_by_task": {
            str(row["task_id"]): float(
                row["requery"]["stored_expert_to_matched_requery_rms"]
            )
            for row in shards
        },
        "decision": decision,
        "decision_boundary": "Action may pass without JVP; JVP alone cannot override failed denoised-action geometry.",
        "claim_boundary": "This successful-policy occupancy diagnostic is neither an unbiased evaluation nor a task dictionary, and BDDL predicates are only a partial final-goal stage proxy.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, evidence)
    print(
        json.dumps(
            {
                "decision": decision,
                "action_passing_tasks": action_summary["passing_tasks"],
                "jvp_passing_tasks": jvp_summary["passing_tasks"],
            },
            sort_keys=True,
        )
    )


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shard = subparsers.add_parser("shard")
    shard.add_argument("--panel-run", type=Path, required=True)
    shard.add_argument("--selection", type=Path, required=True)
    shard.add_argument("--source-checkpoint", type=Path, required=True)
    shard.add_argument("--task-id", type=int, required=True)
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--device", default="cuda:0")
    shard.add_argument("--microbatch-size", type=int, default=8)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--shard", type=Path, action="append", required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.command == "shard":
        run_shard(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
