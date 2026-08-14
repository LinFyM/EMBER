"""One cross-video success-credit cycle for semantic factor commitment."""

from __future__ import annotations

import fcntl
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.lora import copy_task_lora_state_
from ember.pi05_source_checkpoint import barrier
from ember.reward.protocol import RewardTask, reward_credit_environment_seed
from ember.reward.rollout import (
    RewardTrajectory,
    collect_paired_reward_arm_trajectories,
    complete_selected_trajectory_batch,
)
from ember.writer.as_step import (
    accumulate_flat_gradient,
    assign_flat_gradient,
    gather_full24_records,
)
from ember.writer.data import pack_teacher_condition
from ember.writer.errors import WriterModelError
from ember.writer.model import WriterConditioningState, WriterProgramOutput
from ember.writer.reward_preference import (
    PairedSuccessCreditSummary,
    backpropagate_lora_cotangent,
    functional_selected_success_lora_gradient,
    mean_cross_video_task_gradient,
)

if TYPE_CHECKING:
    from ember.writer.reward_training import RewardRuntime


@dataclass(frozen=True)
class RewardProbe:
    global_task_id: int
    suite: str
    conditioning_state: WriterConditioningState
    condition_video_offsets: torch.Tensor
    before_lora: Mapping[str, torch.Tensor]
    query: Mapping[str, torch.Tensor]
    before_action: torch.Tensor
    policy_noise_seed: int


@dataclass(frozen=True)
class AppliedStep:
    active_tasks: int
    gradient_norm: float
    gradient_rms: float
    parameter_delta_rms: Mapping[str, float]


def _claim_task(queue: Path, ordered: tuple[RewardTask, ...]) -> RewardTask | None:
    with queue.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        value = int(handle.read().strip())
        if value >= len(ordered):
            return None
        handle.seek(0)
        handle.truncate()
        handle.write(str(value + 1))
        handle.flush()
    return ordered[value]


def _trajectory_row(value: RewardTrajectory) -> dict[str, Any]:
    return {
        "rollout_cursor": value.rollout_cursor,
        "environment_seed": value.env_seed,
        "success": value.success,
        "steps": value.steps,
        "reward_sum": value.reward_sum,
        "replay_chunks": len(value.valid_action_steps),
        "valid_action_steps": list(value.valid_action_steps),
    }


def _encode_candidate_condition(
    runtime: RewardRuntime, task: RewardTask, demos: Sequence[int]
) -> tuple[
    tuple[Any, ...],
    Mapping[str, Any],
    WriterConditioningState,
    WriterProgramOutput,
    Mapping[str, torch.Tensor],
]:
    packed, video_metrics = pack_teacher_condition(
        runtime.video_store,
        task_id=task.global_task_id,
        demos=demos,
        language=runtime.language_tokens[task.global_task_id],
        device=runtime.context.device,
    )
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=runtime.context.device.type,
            dtype=torch.bfloat16,
            enabled=runtime.context.device.type == "cuda",
        ),
    ):
        state = runtime.writer.encode_conditioning_state(
            *packed, policy=runtime.policy
        )
        encoded = runtime.writer.compile_conditioning_state(
            state, packed[3], use_query_delta=True
        )
        candidate = runtime.writer.decode_output(encoded)
    return packed, video_metrics, state, encoded, candidate


def _encode_pair(
    runtime: RewardRuntime, task: RewardTask, cycle: int
) -> tuple[
    int,
    tuple[int, ...],
    tuple[Any, ...],
    Mapping[str, Any],
    WriterConditioningState,
    Mapping[str, torch.Tensor],
    Mapping[str, torch.Tensor],
]:
    visit = cycle - 1
    demos = runtime.video_schedule.demos_for_task_visit(task.global_task_id, visit)
    packed, video_metrics, state, encoded, candidate = _encode_candidate_condition(
        runtime, task, demos
    )
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=runtime.context.device.type,
            dtype=torch.bfloat16,
            enabled=runtime.context.device.type == "cuda",
        ),
    ):
        if encoded.reference_program is None:
            raise WriterModelError("SFMC candidate lost its exact AS139 reference")
        reference = runtime.writer.decode_program(encoded.reference_program)
    return visit, tuple(demos), packed, video_metrics, state, reference, candidate


def _collect_arm(
    runtime: RewardRuntime, task: RewardTask, visit: int
) -> tuple[RewardTrajectory, ...]:
    rollout_cursors = tuple(visit * 2 + lane for lane in range(2))
    env_seeds = tuple(
        reward_credit_environment_seed(
            int(runtime.config["rng"]["environment_seed_root"]),
            task.suite,
            task.task_id,
            int(runtime.config["rng"]["optimizer_seed"]),
            cursor,
        )
        for cursor in rollout_cursors
    )
    environment = runtime.config["environment"]
    return collect_paired_reward_arm_trajectories(
        envs=tuple(runtime.env_pool.get(task, lane=lane) for lane in range(2)),
        policy=runtime.policy,
        preprocess=runtime.processor,
        postprocess=runtime.processor.unnormalize_action,
        suite=task.suite,
        task_id=task.task_id,
        global_task_id=task.global_task_id,
        language=task.language,
        adaptation_seed=int(runtime.config["rng"]["optimizer_seed"]),
        rollout_cursors=rollout_cursors,
        env_seeds=env_seeds,
        policy_seed_root=int(runtime.config["rng"]["policy_noise_seed_root"]),
        device=runtime.context.device,
        max_horizon=task.horizon,
        dummy_settling_steps=int(environment["dummy_settling_steps"]),
        dummy_action=environment["dummy_action"],
        action_execution_horizon=int(environment["action_execution_horizon"]),
        num_inference_steps=int(environment["num_inference_steps"]),
    )


def _collect_paired_arms(
    runtime: RewardRuntime,
    task: RewardTask,
    visit: int,
    reference_lora: Mapping[str, torch.Tensor],
    candidate_lora: Mapping[str, torch.Tensor],
) -> tuple[tuple[RewardTrajectory, ...], tuple[RewardTrajectory, ...], float, float]:
    reference_started = time.monotonic()
    try:
        copy_task_lora_state_(
            runtime.policy, reference_lora, runtime.lora_contract
        )
        reference = _collect_arm(runtime, task, visit)
        reference_seconds = time.monotonic() - reference_started
        candidate_started = time.monotonic()
        copy_task_lora_state_(
            runtime.policy, candidate_lora, runtime.lora_contract
        )
        candidate = _collect_arm(runtime, task, visit)
        candidate_seconds = time.monotonic() - candidate_started
    finally:
        copy_task_lora_state_(
            runtime.policy, runtime.identity_state, runtime.lora_contract
        )
    return reference, candidate, reference_seconds, candidate_seconds


def _same_pair_identifiers(
    reference: RewardTrajectory, candidate: RewardTrajectory
) -> bool:
    names = (
        "suite",
        "task_id",
        "global_task_id",
        "adaptation_seed",
        "rollout_cursor",
        "env_seed",
        "policy_seed_root",
        "dummy_settling_steps",
    )
    if any(getattr(reference, name) != getattr(candidate, name) for name in names):
        return False
    shared = min(
        len(reference.policy_noise_seeds), len(candidate.policy_noise_seeds)
    )
    return (
        shared > 0
        and reference.policy_noise_seeds[:shared]
        == candidate.policy_noise_seeds[:shared]
    )


def select_unique_success_trajectories(
    reference: Sequence[RewardTrajectory],
    candidate: Sequence[RewardTrajectory],
) -> tuple[tuple[RewardTrajectory, ...], tuple[str, ...]]:
    """Select the successful trajectory only when the two exact arms disagree."""

    if len(reference) != 2 or len(candidate) != 2:
        raise WriterModelError("SFMC requires two paired reset states")
    selected: list[RewardTrajectory] = []
    labels: list[str] = []
    for reference_row, candidate_row in zip(reference, candidate, strict=True):
        if not _same_pair_identifiers(reference_row, candidate_row):
            raise WriterModelError("SFMC arm pairing changed reset or policy RNG")
        if candidate_row.success and not reference_row.success:
            selected.append(candidate_row)
            labels.append("candidate")
        elif reference_row.success and not candidate_row.success:
            selected.append(reference_row)
            labels.append("reference")
        elif reference_row.success:
            labels.append("both_success")
        else:
            labels.append("both_failure")
    return tuple(selected), tuple(labels)


def _differentiate_credit_view(
    runtime: RewardRuntime,
    task: RewardTask,
    cycle: int,
    packed: tuple[Any, ...],
    state: WriterConditioningState,
    candidate_lora: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    trajectory_ids: torch.Tensor,
    gradient_template: torch.Tensor,
) -> tuple[torch.Tensor, PairedSuccessCreditSummary]:
    with torch.autocast(
        device_type=runtime.context.device.type,
        dtype=torch.bfloat16,
        enabled=runtime.context.device.type == "cuda",
    ):
        lora_gradient, summary = functional_selected_success_lora_gradient(
            runtime.policy,
            candidate_lora,
            runtime.lora_contract,
            batch,
            trajectory_ids,
            mc_samples=int(runtime.config["objective"]["flow_mc_samples"]),
            physical_microbatch_size=int(
                runtime.config["optimization"]["reward_replay_chunk_batch_size"]
            ),
            flow_seed_root=int(runtime.config["rng"]["flow_credit_seed_root"]),
            cycle=cycle,
            global_task_id=task.global_task_id,
            device=runtime.context.device,
        )
        recompiled = runtime.writer.compile_conditioning_state(
            state, packed[3], use_query_delta=True
        )
        generated = runtime.writer.decode_output(recompiled)
        backpropagate_lora_cotangent(generated, lora_gradient)
    flat = torch.zeros_like(gradient_template)
    gradients = tuple(item.parameter.grad for item in runtime.gradient_layout)
    accumulate_flat_gradient(flat, gradients, runtime.gradient_layout)
    for item in runtime.gradient_layout:
        item.parameter.grad = None
    return flat, summary


def _differentiate_task_credit(
    runtime: RewardRuntime,
    task: RewardTask,
    cycle: int,
    visit: int,
    anchor_demos: Sequence[int],
    packed: tuple[Any, ...],
    state: WriterConditioningState,
    candidate_lora: Mapping[str, torch.Tensor],
    selected: Sequence[RewardTrajectory],
    gradient_sum: torch.Tensor,
) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    batch, trajectory_ids = complete_selected_trajectory_batch(
        selected, torch.device("cpu")
    )
    demo_sets = runtime.video_schedule.cross_video_credit_demos_for_task_visit(
        task.global_task_id,
        visit,
        anchor_demos,
        view_count=int(runtime.config["data"]["credit_views_per_active_task"]),
    )
    view_gradients, view_rows = [], []
    for view_index, demos in enumerate(demo_sets):
        if view_index == 0:
            view_packed, view_state, view_lora = packed, state, candidate_lora
            view_metrics: Mapping[str, Any] = {}
        else:
            view_packed, view_metrics, view_state, _, view_lora = (
                _encode_candidate_condition(runtime, task, demos)
            )
        flat, summary = _differentiate_credit_view(
            runtime,
            task,
            cycle,
            view_packed,
            view_state,
            view_lora,
            batch,
            trajectory_ids,
            gradient_sum,
        )
        view_gradients.append(flat)
        view_rows.append(
            {
                "view_index": view_index,
                "demo_indices": list(demos),
                "factor_commitment_gradient_rms": float(
                    flat.square().mean().sqrt()
                ),
                **asdict(summary),
                **view_metrics,
            }
        )
    gradient_sum.add_(mean_cross_video_task_gradient(view_gradients))
    first = view_rows[0]
    return {
        "objective": math.fsum(float(row["objective"]) for row in view_rows) / 4,
        "target_trajectories": int(first["target_trajectories"]),
        "replay_chunks": int(first["replay_chunks"]),
        "executed_action_steps": int(first["executed_action_steps"]),
        "functional_policy_forwards": sum(
            int(row["functional_policy_forwards"]) for row in view_rows
        ),
        "functional_policy_backwards": sum(
            int(row["functional_policy_backwards"]) for row in view_rows
        ),
        "lora_gradient_rms": math.fsum(
            float(row["lora_gradient_rms"]) for row in view_rows
        )
        / 4,
        "credit_conditions": 4,
        "credit_unique_video_count": len(
            {demo for demos in demo_sets for demo in demos}
        ),
        "credit_view_records": view_rows,
    }, time.monotonic() - started


def _make_probe(
    task: RewardTask,
    packed: tuple[Any, ...],
    state: WriterConditioningState,
    candidate_lora: Mapping[str, torch.Tensor],
    candidate: Sequence[RewardTrajectory],
    labels: Sequence[str],
) -> RewardProbe:
    index = next(
        index
        for index, label in enumerate(labels)
        if label in {"candidate", "reference"}
    )
    trajectory = candidate[index]
    return RewardProbe(
        global_task_id=task.global_task_id,
        suite=task.suite,
        conditioning_state=state,
        condition_video_offsets=packed[3],
        before_lora={
            name: value.detach().clone() for name, value in candidate_lora.items()
        },
        query={name: value.clone() for name, value in trajectory.observations[0].items()},
        before_action=trajectory.action_chunks[0].clone(),
        policy_noise_seed=trajectory.policy_noise_seeds[0],
    )


def _empty_credit() -> dict[str, Any]:
    return {
        "objective": 0.0,
        "target_trajectories": 0,
        "replay_chunks": 0,
        "executed_action_steps": 0,
        "functional_policy_forwards": 0,
        "functional_policy_backwards": 0,
        "lora_gradient_rms": 0.0,
        "credit_conditions": 0,
        "credit_unique_video_count": 0,
        "credit_view_records": [],
    }


def _task_gradient(
    runtime: RewardRuntime,
    task: RewardTask,
    cycle: int,
    gradient_sum: torch.Tensor,
    probe: RewardProbe | None,
) -> tuple[dict[str, Any], RewardProbe | None, int]:
    visit, anchor_demos, packed, video_metrics, state, reference_lora, candidate_lora = (
        _encode_pair(runtime, task, cycle)
    )
    reference, candidate, reference_seconds, candidate_seconds = (
        _collect_paired_arms(
            runtime,
            task,
            visit,
            reference_lora,
            candidate_lora,
        )
    )
    selected, labels = select_unique_success_trajectories(reference, candidate)
    credit = _empty_credit()
    credit_seconds = 0.0
    active = int(bool(selected))
    if selected:
        credit, credit_seconds = _differentiate_task_credit(
            runtime,
            task,
            cycle,
            visit,
            anchor_demos,
            packed,
            state,
            candidate_lora,
            selected,
            gradient_sum,
        )
        if probe is None:
            probe = _make_probe(
                task,
                packed,
                state,
                candidate_lora,
                candidate,
                labels,
            )
    row = {
        "task_id": task.global_task_id,
        "rank": runtime.context.rank,
        "suite": task.suite,
        "local_task_id": task.task_id,
        "cycle": cycle,
        "active_credit": bool(active),
        "pair_labels": list(labels),
        "reference_successes": sum(value.success for value in reference),
        "candidate_successes": sum(value.success for value in candidate),
        "candidate_gains": labels.count("candidate"),
        "reference_gains": labels.count("reference"),
        "both_success": labels.count("both_success"),
        "both_failure": labels.count("both_failure"),
        **credit,
        **video_metrics,
        "anchor_demo_indices": list(anchor_demos),
        "paired_states": 2,
        "rollouts": 4,
        "reference_trajectory_rows": [
            _trajectory_row(value) for value in reference
        ],
        "candidate_trajectory_rows": [
            _trajectory_row(value) for value in candidate
        ],
        "reference_rollout_seconds": reference_seconds,
        "candidate_rollout_seconds": candidate_seconds,
        "credit_seconds": credit_seconds,
        "target_dataset_action_reads": 0,
        "teacher_action_reads": 0,
    }
    return row, probe, active


def _collect_cycle_tasks(
    runtime: RewardRuntime,
    cycle: int,
    gradient_sum: torch.Tensor,
) -> tuple[list[dict[str, Any]], RewardProbe | None, int]:
    if runtime.args.mode == "smoke":
        task_id = int(
            runtime.args.smoke_task_id
            or runtime.config["smoke_run"]["task_global_id"]
        )
        task = next(task for task in runtime.tasks if task.global_task_id == task_id)
        row, probe, active = _task_gradient(
            runtime, task, cycle, gradient_sum, None
        )
        return [row], probe, active
    ordered = tuple(
        sorted(runtime.tasks, key=lambda task: (-task.horizon, task.global_task_id))
    )
    queue = runtime.args.output_dir / f".cycle_{cycle:08d}_task_cursor"
    if runtime.context.is_main:
        queue.write_text("0", encoding="utf-8")
    barrier(runtime.context)
    records: list[dict[str, Any]] = []
    probe = None
    active = 0
    while task := _claim_task(queue, ordered):
        row, probe, task_active = _task_gradient(
            runtime, task, cycle, gradient_sum, probe
        )
        records.append(row)
        active += task_active
    barrier(runtime.context)
    if runtime.context.is_main:
        queue.unlink(missing_ok=True)
    return records, probe, active


def _trainable_named(
    runtime: RewardRuntime,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    return tuple(
        (name, value)
        for name, value in runtime.writer.named_parameters()
        if value.requires_grad
    )


def _apply_step(
    runtime: RewardRuntime,
    gradient_sum: torch.Tensor,
    local_active_tasks: int,
) -> AppliedStep:
    active = torch.tensor(
        local_active_tasks, dtype=torch.long, device=runtime.context.device
    )
    if runtime.context.world_size > 1:
        dist.all_reduce(gradient_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(active, op=dist.ReduceOp.SUM)
    active_tasks = int(active)
    if active_tasks <= 0:
        raise WriterModelError("SFMC cycle produced no discordant successful arm")
    gradient_sum.div_(active_tasks)
    if not bool(torch.isfinite(gradient_sum).all()) or not bool(
        torch.count_nonzero(gradient_sum)
    ):
        raise WriterModelError("SFMC cycle produced an invalid shared gradient")
    assign_flat_gradient(gradient_sum, runtime.gradient_layout)
    clip = float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"])
    grad_norm = torch.nn.utils.clip_grad_norm_(runtime.trainable_parameters, clip)
    named = _trainable_named(runtime)
    before = {name: value.detach().clone() for name, value in named}
    runtime.optimizer.step()
    delta = {
        name: float((value.detach() - before[name]).float().square().mean().sqrt())
        for name, value in named
    }
    return AppliedStep(
        active_tasks=active_tasks,
        gradient_norm=float(grad_norm),
        gradient_rms=float(gradient_sum.square().mean().sqrt()),
        parameter_delta_rms=delta,
    )


def _lora_response(
    before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    sums = {"lora_a": 0.0, "lora_b": 0.0, "effective_ba": 0.0}
    counts = {name: 0 for name in sums}
    for name, before_a in before.items():
        if not name.endswith(".lora_A.default.weight"):
            continue
        b_name = name.replace(".lora_A.default.weight", ".lora_B.default.weight")
        after_a, before_b, after_b = after[name], before[b_name], after[b_name]
        values = {
            "lora_a": after_a.float() - before_a.float(),
            "lora_b": after_b.float() - before_b.float(),
            "effective_ba": after_b.float() @ after_a.float()
            - before_b.float() @ before_a.float(),
        }
        for key, value in values.items():
            sums[key] += float(value.square().sum())
            counts[key] += value.numel()
    return {
        f"{key}_response_rms": math.sqrt(sums[key] / counts[key]) for key in sums
    }


@torch.inference_mode()
def _probe_after_update(
    runtime: RewardRuntime, probe: RewardProbe | None
) -> dict[str, Any] | None:
    if probe is None:
        return None
    with torch.autocast(
        device_type=runtime.context.device.type,
        dtype=torch.bfloat16,
        enabled=runtime.context.device.type == "cuda",
    ):
        encoded = runtime.writer.compile_conditioning_state(
            probe.conditioning_state,
            probe.condition_video_offsets,
            use_query_delta=True,
        )
        after = runtime.writer.decode_output(encoded)
    response = _lora_response(probe.before_lora, after)
    try:
        copy_task_lora_state_(runtime.policy, after, runtime.lora_contract)
        generator = torch.Generator(device="cpu").manual_seed(
            probe.policy_noise_seed
        )
        noise = torch.randn(
            (
                1,
                int(runtime.policy.config.chunk_size),
                int(runtime.policy.config.max_action_dim),
            ),
            generator=generator,
        ).to(runtime.context.device)
        query = {
            name: value.to(runtime.context.device, non_blocking=True)
            for name, value in probe.query.items()
        }
        with torch.autocast(
            device_type=runtime.context.device.type,
            dtype=torch.bfloat16,
            enabled=runtime.context.device.type == "cuda",
        ):
            action = runtime.policy.predict_action_chunk(
                query,
                noise=noise,
                num_steps=int(runtime.config["environment"]["num_inference_steps"]),
            )
    finally:
        copy_task_lora_state_(
            runtime.policy, runtime.identity_state, runtime.lora_contract
        )
    response["fixed_action_response_rms"] = float(
        (action.cpu().float() - probe.before_action.float()).square().mean().sqrt()
    )
    return {"task_id": probe.global_task_id, "suite": probe.suite, **response}


def _gather_cycle_evidence(
    runtime: RewardRuntime,
    records: list[dict[str, Any]],
    probe: RewardProbe | None,
    started: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    if runtime.args.mode == "formal":
        global_records = gather_full24_records(
            records,
            world_size=runtime.context.world_size,
            task_ids=[task.global_task_id for task in runtime.tasks],
        )
    else:
        global_records = records
    probe_row = _probe_after_update(runtime, probe)
    probes: list[Any] = [None] * runtime.context.world_size
    if runtime.context.world_size > 1:
        dist.all_gather_object(probes, probe_row)
    else:
        probes[0] = probe_row
    if runtime.context.device.type == "cuda":
        torch.cuda.synchronize(runtime.context.device)
    elapsed = torch.tensor(
        time.monotonic() - started,
        dtype=torch.float64,
        device=runtime.context.device,
    )
    if runtime.context.world_size > 1:
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return global_records, [value for value in probes if value is not None], float(elapsed)


def _cycle_metrics(
    runtime: RewardRuntime,
    cycle: int,
    records: list[dict[str, Any]],
    probes: list[dict[str, Any]],
    step: AppliedStep,
    elapsed: float,
) -> dict[str, Any]:
    active_records = [row for row in records if row["active_credit"]]
    return {
        "cycle": cycle,
        "cycle_semantics": (
            "one_complete_train24_semantic_factor_memory_commitment"
            if runtime.args.mode == "formal"
            else "one_task_semantic_factor_memory_commitment_live_smoke"
        ),
        "tasks": len(records),
        "paired_states": 2 * len(records),
        "rollouts": 4 * len(records),
        "reference_successes": sum(
            int(row["reference_successes"]) for row in records
        ),
        "candidate_successes": sum(
            int(row["candidate_successes"]) for row in records
        ),
        "candidate_gains": sum(int(row["candidate_gains"]) for row in records),
        "reference_gains": sum(int(row["reference_gains"]) for row in records),
        "both_success": sum(int(row["both_success"]) for row in records),
        "both_failure": sum(int(row["both_failure"]) for row in records),
        "discordant_pairs": sum(
            int(row["candidate_gains"]) + int(row["reference_gains"])
            for row in records
        ),
        "active_tasks": step.active_tasks,
        "active_suites": sorted({row["suite"] for row in active_records}),
        "selected_trajectories": sum(
            int(row["target_trajectories"]) for row in records
        ),
        "credit_conditions": sum(
            int(row["credit_conditions"]) for row in records
        ),
        "credit_unique_video_count": sum(
            int(row["credit_unique_video_count"]) for row in records
        ),
        "replay_chunks": sum(int(row["replay_chunks"]) for row in records),
        "executed_action_steps": sum(
            int(row["executed_action_steps"]) for row in records
        ),
        "selected_success_objective_mean": math.fsum(
            float(row["objective"]) for row in active_records
        )
        / len(active_records),
        "writer_gradient_norm_before_clip": step.gradient_norm,
        "writer_gradient_rms": step.gradient_rms,
        "parameter_delta_rms": step.parameter_delta_rms,
        "deployment_response_probes": probes,
        "task_records": records,
        "cycle_seconds": elapsed,
        "max_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(runtime.context.device)
            if runtime.context.device.type == "cuda"
            else 0
        ),
        "max_cuda_reserved_bytes": (
            torch.cuda.max_memory_reserved(runtime.context.device)
            if runtime.context.device.type == "cuda"
            else 0
        ),
        "target_dataset_action_reads": 0,
        "teacher_action_reads": 0,
        "validation_action_or_reward_reads": 0,
        "test_action_or_reward_reads": 0,
    }


def run_cycle(runtime: RewardRuntime, cycle: int) -> dict[str, Any]:
    started = time.monotonic()
    runtime.optimizer.zero_grad(set_to_none=True)
    gradient_sum = torch.zeros(
        runtime.gradient_layout[-1].stop,
        dtype=torch.float32,
        device=runtime.context.device,
    )
    records, probe, local_active = _collect_cycle_tasks(
        runtime, cycle, gradient_sum
    )
    step = _apply_step(runtime, gradient_sum, local_active)
    records, probes, elapsed = _gather_cycle_evidence(
        runtime, records, probe, started
    )
    return _cycle_metrics(runtime, cycle, records, probes, step, elapsed)
