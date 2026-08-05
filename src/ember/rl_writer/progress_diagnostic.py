"""Read-only cold-start task-grounded semantic progress diagnostic."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from ember.pi05_source_checkpoint import barrier, read_json, write_json_atomic
from ember.pi05_source_setup import reduce_max
from ember.reward.protocol import RewardProtocolError, RewardTask
from ember.rl_writer.contract import cycle_assignments
from ember.rl_writer.loop import CollectedTaskTrajectories, collect_task_trajectories
from ember.rl_writer.progress_credit import (
    PROGRESS_DIAGNOSTIC_ROW_SCHEMA,
    normalized_progress_components,
    semantic_progress_utilities,
    summarize_progress_diagnostic,
)
from ember.rl_writer.progress_observer import (
    encode_progress_components,
    rollout_endpoint_frames,
)
from ember.rl_writer.runtime import RLWriterRuntime
from ember.writer.inference import (
    task_video_mapping,
    writer_generation_seed,
    writer_shuffled_frame_permutation,
)


def _wrong_video_tasks(tasks: Sequence[RewardTask]) -> dict[int, int]:
    keys = tuple((task.suite, task.task_id) for task in tasks)
    roles = {key: "train" for key in keys}
    mapping = task_video_mapping(keys, roles, "cross_suite_wrong")
    result = {
        int(row["language_global_task_id"]): int(row["video_global_task_id"])
        for row in mapping
    }
    if set(result) != {task.global_task_id for task in tasks}:
        raise RewardProtocolError("progress diagnostic wrong-video mapping changed")
    return result


def _encode_components(
    runtime: RLWriterRuntime, task: RewardTask, frames: torch.Tensor
) -> torch.Tensor:
    return encode_progress_components(
        writer=runtime.writer,
        policy=runtime.policy,
        identity_state=runtime.identity_state,
        lora_contract=runtime.lora_contract,
        tokenizer=runtime.tokenizer,
        task=task,
        frames=frames,
        device=runtime.context.device,
        normalization_epsilon=float(
            runtime.config["progress_credit"]["normalization_epsilon"]
        ),
    )


def _pixel_change_rms(start: torch.Tensor, terminal: torch.Tensor) -> float:
    if start.shape != terminal.shape or start.dtype != torch.uint8:
        raise RewardProtocolError("progress diagnostic pixel pair changed")
    delta = terminal.float().sub(start.float()).div_(255.0)
    return float(delta.square().mean().sqrt())


def _teacher_frame_bank(
    runtime: RLWriterRuntime,
    collected: CollectedTaskTrajectories,
    wrong_task_id: int,
) -> torch.Tensor:
    task = collected.task
    correct = collected.frames
    wrong = torch.from_numpy(
        runtime.video_store.load(wrong_task_id, collected.demo_index).frames
    )
    seed = writer_generation_seed(
        int(runtime.config["progress_credit"]["counterfactual_frame_order_seed_root"]),
        task.suite,
        task.task_id,
        collected.demo_index,
        stream="frame_order",
    )
    permutation = writer_shuffled_frame_permutation(
        correct.shape[0], seed, keep_first=False
    )
    return torch.stack(
        (
            correct[0],
            correct[-1],
            correct[int(permutation[0])],
            correct[int(permutation[-1])],
            wrong[0],
            wrong[-1],
        )
    )


def _rollout_frame_bank(collected: CollectedTaskTrajectories) -> torch.Tensor:
    return rollout_endpoint_frames(collected.trajectories)


def _counterfactual_utilities(
    teacher: torch.Tensor,
    rollout: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    starts = rollout[0::2]
    terminals = rollout[1::2]
    endpoints = {
        "correct": (0, 1),
        "reversed": (1, 0),
        "shuffled": (2, 3),
        "wrong": (4, 5),
    }
    utilities = {}
    energy = None
    for name, (start, goal) in endpoints.items():
        values, observed_energy = semantic_progress_utilities(
            teacher[start],
            teacher[goal],
            starts,
            terminals,
            epsilon=epsilon,
        )
        utilities[name] = values
        if name == "correct":
            energy = observed_energy
    if energy is None:
        raise RewardProtocolError("progress diagnostic lost correct teacher energy")
    return utilities, energy


def _task_diagnostic_rows(
    runtime: RLWriterRuntime,
    collected: CollectedTaskTrajectories,
    wrong_task_id: int,
) -> list[dict[str, Any]]:
    task = collected.task
    teacher_frames = _teacher_frame_bank(runtime, collected, wrong_task_id)
    rollout_batch = _rollout_frame_bank(collected)
    teacher_components = _encode_components(runtime, task, teacher_frames)
    rollout_components = _encode_components(runtime, task, rollout_batch)
    repeated_teacher = _encode_components(runtime, task, teacher_frames)
    repeated_rollout = _encode_components(runtime, task, rollout_batch)
    repeat_max = max(
        float((teacher_components - repeated_teacher).abs().max()),
        float((rollout_components - repeated_rollout).abs().max()),
    )
    utilities, energy = _counterfactual_utilities(
        teacher_components,
        rollout_components,
        epsilon=float(runtime.config["progress_credit"]["projection_epsilon"]),
    )
    teacher_energy = float(energy.sum())
    rows = []
    for index, trajectory in enumerate(collected.trajectories):
        start = trajectory.progress_start_frame
        terminal = trajectory.progress_terminal_frame
        if start is None or terminal is None:
            raise RewardProtocolError("progress diagnostic endpoint disappeared")
        rows.append(
            {
                "schema_version": PROGRESS_DIAGNOSTIC_ROW_SCHEMA,
                "producer_rank": runtime.context.rank,
                "suite": task.suite,
                "task_id": task.task_id,
                "global_task_id": task.global_task_id,
                "outer_cycle": 0,
                "rollout_cursor": trajectory.rollout_cursor,
                "teacher_demo_index": collected.demo_index,
                "wrong_video_global_task_id": wrong_task_id,
                "success": trajectory.success,
                "steps": trajectory.steps,
                "utility_correct": float(utilities["correct"][index]),
                "utility_wrong": float(utilities["wrong"][index]),
                "utility_shuffled": float(utilities["shuffled"][index]),
                "utility_reversed": float(utilities["reversed"][index]),
                "teacher_change_energy": teacher_energy,
                "component_change_energies": [float(value) for value in energy],
                "observer_repeat_max_abs": repeat_max,
                "pixel_change_rms": _pixel_change_rms(start, terminal),
                "observer_inputs": [
                    "pure_task_language",
                    "teacher_agentview_rgb",
                    "rollout_agentview_rgb",
                ],
                "teacher_action_reads": 0,
                "rollout_action_reads_by_observer": 0,
                "proprio_reads_by_observer": 0,
                "validation_reads": 0,
                "test_reads": 0,
            }
        )
    return rows


def _rank_payload_path(root: Path, rank: int) -> Path:
    return root / "diagnostic_rows" / f"rank_{rank:03d}.json"


def run_progress_diagnostic(runtime: RLWriterRuntime) -> None:
    if (
        runtime.args.mode != "diagnostic"
        or runtime.next_cycle != 0
        or runtime.args.resume is not None
        or runtime.learning_epochs != 0
    ):
        raise RewardProtocolError("progress diagnostic must be read-only fresh cycle0")
    runtime.writer.eval()
    runtime.policy.eval()
    if any(parameter.requires_grad for parameter in runtime.writer.semantic_encoder.parameters()):
        raise RewardProtocolError("progress observer semantic encoder is not frozen")

    started = time.monotonic()
    wrong = _wrong_video_tasks(runtime.tasks)
    assigned = cycle_assignments(
        runtime.tasks,
        world_size=runtime.context.world_size,
        cycle=0,
        seed=int(runtime.config["data"]["task_schedule_seed"]),
    )[runtime.context.rank]
    rows = []
    local_actions = 0
    for task in assigned:
        collected = collect_task_trajectories(runtime, task, 0)
        rows.extend(
            _task_diagnostic_rows(
                runtime,
                collected,
                wrong[task.global_task_id],
            )
        )
        local_actions += sum(value.steps for value in collected.trajectories)
    write_json_atomic(
        _rank_payload_path(runtime.args.output_dir, runtime.context.rank),
        {
            "rank": runtime.context.rank,
            "world_size": runtime.context.world_size,
            "owned_task_ids": [task.global_task_id for task in assigned],
            "row_count": len(rows),
            "environment_actions": local_actions,
            "rows": rows,
        },
    )
    torch.cuda.synchronize(runtime.context.device)
    barrier(runtime.context)
    wall = reduce_max(time.monotonic() - started, runtime.context)
    peak = int(
        reduce_max(torch.cuda.max_memory_reserved(runtime.context.device), runtime.context)
    )
    if runtime.context.is_main:
        payloads = [
            read_json(_rank_payload_path(runtime.args.output_dir, rank))
            for rank in range(runtime.context.world_size)
        ]
        if any(
            int(payload.get("rank", -1)) != rank
            or int(payload.get("world_size", -1)) != runtime.context.world_size
            for rank, payload in enumerate(payloads)
        ):
            raise RewardProtocolError("progress diagnostic rank ownership changed")
        combined = [row for payload in payloads for row in payload["rows"]]
        result = summarize_progress_diagnostic(
            combined,
            gates=runtime.config["progress_credit"]["diagnostic_gates"],
        )
        result.update(
            {
                "contract_sha256": runtime.contract_sha256,
                "world_size": runtime.context.world_size,
                "rank_ownership": [
                    {
                        "rank": int(payload["rank"]),
                        "task_ids": list(payload["owned_task_ids"]),
                        "row_count": int(payload["row_count"]),
                    }
                    for payload in payloads
                ],
                "environment_actions": sum(
                    int(payload["environment_actions"]) for payload in payloads
                ),
                "wall_seconds_max_rank": wall,
                "max_cuda_reserved_bytes": peak,
                "optimizer_updates": 0,
                "writer_backward_calls": 0,
                "checkpoint_count": 0,
            }
        )
        write_json_atomic(runtime.args.output_dir / "progress_credit_results.json", result)
        write_json_atomic(
            runtime.args.output_dir / "completion.json",
            {
                "schema_version": "ember_pi05_task_grounded_progress_credit_completion_v1",
                "complete": True,
                "mechanism_passed": bool(result["passed"]),
                "row_count": len(combined),
                "optimizer_updates": 0,
                "writer_backward_calls": 0,
                "checkpoint_count": 0,
                "teacher_action_reads": 0,
                "validation_reads": 0,
                "test_reads": 0,
            },
        )
        print(json.dumps(result, sort_keys=True), flush=True)
    barrier(runtime.context)
