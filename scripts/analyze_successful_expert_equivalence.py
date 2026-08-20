#!/usr/bin/env python3
"""Build and gate phase-aligned functional labels from successful experts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file

from ember.functional_adaptation.phase_alignment import (
    arc_length_phase_embedding,
    fit_task_equal_whitener,
    uniform_time_embedding,
)
from ember.lora import identity_lora_state
from ember.pi05_eval_contract import (
    git_state,
    git_state_is_clean_pushed_or_frozen_authority,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import load_policy
from ember.reward.rollout import (
    RewardTrajectory,
    query_successful_expert_occupancy_actions,
)
from ember.writer.functional import prepare_frozen_writer_policy


REPO_ROOT = Path(__file__).resolve().parents[1]
SELECTION_SCHEMA = "ember_successful_expert_equivalence_selection_v1"
CAPTURE_SCHEMA = "ember_successful_expert_equivalence_capture_v1"
SHARD_SCHEMA = "ember_successful_expert_equivalence_response_shard_v1"
ANALYSIS_SCHEMA = "ember_successful_expert_equivalence_phase_analysis_v1"
TRANSFORM_SCHEMA = "ember_successful_expert_equivalence_phase_transform_v1"
EXPECTED_BY_STEP = {250: 21, 500: 2, 1000: 1, 2000: 23}


def _repo_reference(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        parts = resolved.parts
        if "runs" in parts:
            return str(Path(*parts[parts.index("runs") :]))
    return resolved.name


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
        raise ValueError("successful expert trajectory sidecar changed")
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
        raise ValueError("successful expert trajectory layout changed")
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


def _stage_timeline(row: Mapping[str, Any], *, replans: int, replan_steps: int) -> dict[str, Any]:
    stage = row["stage_predicates"]
    transitions = tuple(stage["transitions"])
    values = []
    for index in range(replans):
        step = index * replan_steps
        current = transitions[0]["satisfied"]
        for transition in transitions[1:]:
            if int(transition["step"]) > step:
                break
            current = transition["satisfied"]
        values.append([bool(value) for value in current])
    full = [all(value) for value in values]
    return {
        "predicates": stage["predicates"],
        "satisfied_at_replans": values,
        "full_conjunction_at_replans": full,
        "first_full_conjunction_replan": full.index(True) if any(full) else None,
    }


def _load_panels(paths: Sequence[Path]) -> dict[int, dict[str, Any]]:
    panels: dict[int, dict[str, Any]] = {}
    for path in paths:
        root = path.resolve()
        contract = read_json(root / "run_contract.json")
        results = read_json(root / "results.json")
        capture = contract.get("diagnostic_occupancy_capture", {})
        step = int(capture.get("selected_step", -1))
        expected = EXPECTED_BY_STEP.get(step)
        adapter = contract.get("adapter", {})
        if (
            expected is None
            or capture.get("schema_version") != CAPTURE_SCHEMA
            or results.get("mode") != "formal"
            or results.get("role") != "development_train"
            or int(results.get("overall", {}).get("episodes", -1)) != expected
            or int(results.get("overall", {}).get("successes", -1)) != expected
            or contract.get("git", {}).get("dirty_paths") != []
            or adapter.get("information_wall", {}).get("diagnostic_subset")
            != "successful_expert_equivalence_occupancy"
            or contract.get("diagnostic_stage_predicates", {}).get("schema_version")
            != "ember_pi05_stage_predicate_capture_v1"
            or step in panels
        ):
            raise ValueError("successful expert equivalence capture changed")
        rows = {
            (str(row["suite"]), int(row["task_id"]), int(row["init_state_id"])): row
            for row in results["rows"]
        }
        adapters = {
            (str(row["suite"]), int(row["task_id"])): row
            for row in adapter["tasks"]
        }
        if len(rows) != expected or len(adapters) != int(capture["selected_tasks"]):
            raise ValueError("successful expert equivalence panel is incomplete")
        panels[step] = {
            "root": root,
            "contract": contract,
            "results": results,
            "rows": rows,
            "adapters": adapters,
        }
    if set(panels) != set(EXPECTED_BY_STEP):
        raise ValueError("four successful expert equivalence captures are required")
    return panels


def _action_delta(
    actions: Mapping[str, Sequence[Sequence[torch.Tensor]]]
) -> torch.Tensor:
    expert = actions["expert"][0]
    source = actions["student"][0]
    result = torch.cat(
        [left.float() - right.float() for left, right in zip(expert, source, strict=True)]
    )
    if result.ndim != 3 or result.shape[1:] != (50, 7) or not torch.isfinite(result).all():
        raise ValueError("complete action response shape changed")
    return result


def run_shard(args: argparse.Namespace) -> None:
    selection = read_json(args.selection.resolve())
    state = git_state(REPO_ROOT)
    if (
        selection.get("schema_version") != SELECTION_SCHEMA
        or len(selection.get("rows", ())) != 47
        or not git_state_is_clean_pushed_or_frozen_authority(state)
    ):
        raise ValueError("successful expert equivalence analysis authority changed")
    panels = _load_panels(args.panel_run)
    selected = [
        dict(row)
        for row in selection["rows"]
        if int(row["ordinal"]) % args.shard_count == args.shard_index
    ]
    if not selected:
        raise ValueError("successful expert equivalence shard is empty")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    source_config = read_json(REPO_ROOT / "configs/pi05_source_base_v1.json")
    policy = load_policy(
        args.source_checkpoint.resolve() / "policy", source_config, device
    )
    lora = load_pi05_lora_contract(REPO_ROOT / "configs/pi05_lora_v1.json")
    prepare_frozen_writer_policy(policy, lora)
    identity = identity_lora_state(lora, device=device)
    members = []
    for record in selected:
        step = int(record["expert_step"])
        panel = panels[step]
        key = (
            str(record["suite"]),
            int(record["task_id"]),
            int(record["init_state_id"]),
        )
        row = panel["rows"].get(key)
        adapter = panel["adapters"].get(key[:2])
        if (
            row is None
            or adapter is None
            or not bool(row.get("success"))
            or int(row.get("task_expert", {}).get("step", -1)) != step
            or int(row.get("task_expert", {}).get("global_task_id", -1))
            != int(record["global_task_id"])
            or Path(str(adapter.get("checkpoint")))
            != Path(str(row["task_expert"]["checkpoint"]))
        ):
            raise ValueError("selected successful expert member changed")
        replan_steps = int(panel["contract"]["policy"]["replan_steps"])
        trajectory = _load_trajectory(record, row, replan_steps=replan_steps)
        expert = load_file(
            str(Path(adapter["checkpoint"]) / "adapter.safetensors"),
            device=str(device),
        )
        actions, requery = query_successful_expert_occupancy_actions(
            policy=policy,
            lora_contract=lora,
            identity_state=identity,
            trajectories=(trajectory,),
            expert_lora=expert,
            student_lora=identity,
            device=device,
            microbatch_size=args.microbatch_size,
            num_inference_steps=10,
        )
        response = _action_delta(actions)
        members.append(
            {
                **record,
                "captured_steps": int(row["steps"]),
                "replans": len(trajectory.observations),
                "panel_step": step,
                "panel_contract_reference": panel["contract"]["contract_reference"],
                "stage": _stage_timeline(
                    row, replans=len(trajectory.observations), replan_steps=replan_steps
                ),
                "requery": requery,
                "action_delta": response,
            }
        )
        print(
            json.dumps(
                {
                    "ordinal": int(record["ordinal"]),
                    "member": record["member"],
                    "step": step,
                    "replans": len(trajectory.observations),
                    "response_rms": float(response.square().mean().sqrt()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    payload = {
        "schema_version": SHARD_SCHEMA,
        "analysis_git": state,
        "selection": _repo_reference(args.selection),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "runtime": {
            "source_checkpoint": _repo_reference(args.source_checkpoint),
            "device": str(args.device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "microbatch_size": args.microbatch_size,
            "num_inference_steps": 10,
        },
        "panels": {
            str(step): _repo_reference(panel["root"])
            for step, panel in panels.items()
        },
        "members": members,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)


def _cosine_summary(
    name: str,
    values: torch.Tensor,
    labels: Sequence[Mapping[str, Any]],
    *,
    include_matrix: bool,
) -> dict[str, Any]:
    flat = values.double().flatten(1)
    norms = torch.linalg.vector_norm(flat, dim=1)
    cosine = (flat @ flat.T) / (norms[:, None] * norms[None, :]).clamp_min(1e-24)
    nearest_matrix = cosine.clone()
    nearest_matrix.fill_diagonal_(float("-inf"))
    nearest = nearest_matrix.argmax(dim=1)
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label["global_task_id"]), []).append(index)
    per_task = []
    for task_id, indices in sorted(groups.items()):
        if len(indices) != 2:
            continue
        left, right = indices
        per_task.append(
            {
                "global_task_id": task_id,
                "ordinal": int(labels[left]["ordinal"]),
                "same_task_cosine": float(cosine[left, right]),
                "mutual_cosine_nearest": (
                    int(nearest[left]) == right and int(nearest[right]) == left
                ),
            }
        )
    result = {
        "family": name,
        "members": len(labels),
        "paired_tasks": len(per_task),
        "mutual_nearest_tasks": sum(
            bool(row["mutual_cosine_nearest"]) for row in per_task
        ),
        "nearest_indices": [int(value) for value in nearest],
        "per_task": per_task,
    }
    if include_matrix:
        result["cosine"] = [
            [float(value) for value in row] for row in cosine
        ]
    return result


def _member_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["ordinal"]),
        int(row["expert_step"]),
        int(row["init_state_id"]),
    )


def aggregate(args: argparse.Namespace) -> None:
    selection = read_json(args.selection.resolve())
    shards = [
        torch.load(path, map_location="cpu", weights_only=False) for path in args.shard
    ]
    shard_count = len(shards)
    if (
        selection.get("schema_version") != SELECTION_SCHEMA
        or shard_count <= 0
        or any(row.get("schema_version") != SHARD_SCHEMA for row in shards)
        or {int(row["shard_count"]) for row in shards} != {shard_count}
        or {int(row["shard_index"]) for row in shards} != set(range(shard_count))
        or any(row["analysis_git"] != shards[0]["analysis_git"] for row in shards)
        or any(row["panels"] != shards[0]["panels"] for row in shards)
        or len(
            {
                (
                    row["runtime"]["source_checkpoint"],
                    row["runtime"]["device"],
                    int(row["runtime"]["microbatch_size"]),
                    int(row["runtime"]["num_inference_steps"]),
                )
                for row in shards
            }
        )
        != 1
    ):
        raise ValueError("successful expert equivalence shards changed")
    members = [member for shard in shards for member in shard["members"]]
    members.sort(key=_member_key)
    expected = sorted((dict(row) for row in selection["rows"]), key=_member_key)
    if [_member_key(row) for row in members] != [_member_key(row) for row in expected]:
        raise ValueError("successful expert equivalence members are incomplete")

    fit_members = [row for row in members if row["fold_role"] == "fit"]
    whitener = fit_task_equal_whitener(
        [row["action_delta"] for row in fit_members],
        [int(row["global_task_id"]) for row in fit_members],
        width=32,
    )
    for row in members:
        sequence = whitener.transform(row["action_delta"])
        row["uniform_embedding"] = uniform_time_embedding(sequence, count=8)
        row["phase_embedding"] = arc_length_phase_embedding(sequence, count=8)

    held = [row for row in members if row["fold_role"] == "held_transform_only"]
    fit = [row for row in members if row["fold_role"] == "fit"]
    held_uniform = _cosine_summary(
        "held5_uniform_time",
        torch.stack([row["uniform_embedding"] for row in held]),
        held,
        include_matrix=True,
    )
    held_phase = _cosine_summary(
        "held5_functional_arc_length",
        torch.stack([row["phase_embedding"] for row in held]),
        held,
        include_matrix=True,
    )
    uniform_by_task = {
        int(row["global_task_id"]): row for row in held_uniform["per_task"]
    }
    improvements = []
    for row in held_phase["per_task"]:
        task_id = int(row["global_task_id"])
        baseline = float(uniform_by_task[task_id]["same_task_cosine"])
        aligned = float(row["same_task_cosine"])
        improvements.append(
            {
                "global_task_id": task_id,
                "ordinal": int(row["ordinal"]),
                "uniform_time_cosine": baseline,
                "phase_aligned_cosine": aligned,
                "cosine_delta": aligned - baseline,
                "improved": aligned > baseline,
            }
        )
    improved_tasks = sum(bool(row["improved"]) for row in improvements)
    passes = held_phase["mutual_nearest_tasks"] >= 4 and improved_tasks >= 2
    fit_uniform = _cosine_summary(
        "fit19_uniform_time",
        torch.stack([row["uniform_embedding"] for row in fit]),
        fit,
        include_matrix=False,
    )
    fit_phase = _cosine_summary(
        "fit19_functional_arc_length",
        torch.stack([row["phase_embedding"] for row in fit]),
        fit,
        include_matrix=False,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    transform_path = output_dir / "phase_transform.pt"
    evidence_path = output_dir / "analysis.json"
    if transform_path.exists() or evidence_path.exists():
        raise ValueError("successful expert equivalence aggregate already exists")
    analysis_git = shards[0]["analysis_git"]
    transform = {
        "schema_version": TRANSFORM_SCHEMA,
        "analysis_git": analysis_git,
        "selection": _repo_reference(args.selection),
        "response": "expert_minus_source_full_50x7_action_chunk_at_every_replan",
        "fit_role": "fit19_task_equal_member_equal_state_equal",
        "held_role": "fold0_transform_only",
        "width": 32,
        **whitener.state_dict(),
    }
    temporary = transform_path.with_suffix(f".pt.tmp.{os.getpid()}")
    torch.save(transform, temporary)
    os.replace(temporary, transform_path)
    evidence = {
        "schema_version": ANALYSIS_SCHEMA,
        "analysis_git": analysis_git,
        "selection": _repo_reference(args.selection),
        "panels": shards[0]["panels"],
        "shards": [_repo_reference(path) for path in args.shard],
        "runtime_by_shard": [row["runtime"] for row in shards],
        "validity": {
            "preregistered_members": 47,
            "reproduced_successes": 47,
            "row_replacement": False,
            "fit_tasks": 19,
            "held_transform_only_tasks": 5,
            "development_train_action_derived_response_use": True,
            "fold0_response_use": "frozen_transform_and_gate_only_no_coordinate_fit",
            "held_data_use": False,
            "validation_data_use": False,
            "test_data_use": False,
            "training_gradient_use": False,
        },
        "functional_coordinate": {
            "input": "expert_minus_source_full_50x7_action_chunk_at_every_replan",
            "input_width": 350,
            "output_width": 32,
            "weighting": "equal_task_then_equal_member_then_equal_state",
            "fit_members": len(fit_members),
            "fit_response_states": sum(
                int(row["action_delta"].shape[0]) for row in fit_members
            ),
            "explained_variance_ratio": whitener.explained_variance_ratio,
            "transform_artifact": _repo_reference(transform_path),
        },
        "held_gate": {
            "rule": (
                "phase mutual-cosine-nearest >=4/5 and same-task cosine improves "
                "over uniform-time for >=2/5"
            ),
            "uniform_time": held_uniform,
            "functional_arc_length": held_phase,
            "per_task_improvement": improvements,
            "improved_tasks": improved_tasks,
            "passes": passes,
        },
        "fit_diagnostic": {
            "uniform_time": fit_uniform,
            "functional_arc_length": fit_phase,
        },
        "members": [
            {
                key: row[key]
                for key in (
                    "suite",
                    "task_id",
                    "global_task_id",
                    "ordinal",
                    "fold_role",
                    "member",
                    "expert_step",
                    "init_state_id",
                    "steps",
                    "captured_steps",
                    "replans",
                )
            }
            | {
                "response_rms": float(row["action_delta"].square().mean().sqrt()),
                "expert_requery_rms": float(
                    row["requery"]["stored_expert_to_matched_requery_rms"]
                ),
                "first_full_conjunction_replan": row["stage"][
                    "first_full_conjunction_replan"
                ],
            }
            for row in members
        ],
        "decision": (
            "advance_to_phase_aligned_fixed_decoder"
            if passes
            else "do_not_rebuild_decoder_from_arc_length_phase_alignment"
        ),
        "decision_boundary": (
            "Only a passing held5 gate authorizes rebuilding the fixed decoder "
            "from this representation."
        ),
        "claim_boundary": (
            "A failure closes this task-equal PCA plus monotone functional-arc-length "
            "alignment, not action responses, event-aware phase models, successful "
            "adapters, or functional decoders in general."
        ),
    }
    write_json_atomic(evidence_path, evidence)
    print(
        json.dumps(
            {
                "decision": evidence["decision"],
                "held_phase_mutual_nearest_tasks": held_phase[
                    "mutual_nearest_tasks"
                ],
                "held_improved_tasks": improved_tasks,
                "fit_phase_mutual_nearest_tasks": fit_phase[
                    "mutual_nearest_tasks"
                ],
            },
            sort_keys=True,
        )
    )


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    shard = commands.add_parser("shard")
    shard.add_argument("--selection", type=Path, required=True)
    shard.add_argument("--panel-run", type=Path, action="append", required=True)
    shard.add_argument("--source-checkpoint", type=Path, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--shard-count", type=int, required=True)
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--device", default="cuda:0")
    shard.add_argument("--microbatch-size", type=int, default=8)
    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument("--selection", type=Path, required=True)
    aggregate_parser.add_argument("--shard", type=Path, action="append", required=True)
    aggregate_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.command == "shard":
        run_shard(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
