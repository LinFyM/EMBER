"""One canonical support-projected reward cycle for the shared Writer."""

from __future__ import annotations

import fcntl
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import torch
import torch.distributed as dist

from ember.lora import copy_task_lora_state_
from ember.pi05_source_checkpoint import barrier
from ember.reward.protocol import RewardTask, reward_credit_environment_seed
from ember.reward.rollout import (
    RewardTrajectory,
    collect_randomized_reward_trajectories,
    complete_trajectory_batch,
)
from ember.writer.as_step import (
    accumulate_flat_gradient,
    assign_flat_gradient,
    gather_full24_records,
)
from ember.writer.data import pack_teacher_condition
from ember.writer.errors import WriterModelError
from ember.writer.reward_preference import (
    FinalSupportProjectionSummary,
    RewardPreferenceSummary,
    backpropagate_lora_cotangent,
    functional_reward_lora_gradients,
    project_final_parameter_delta,
)

if TYPE_CHECKING:
    from ember.writer.reward_training import RewardRuntime


@dataclass(frozen=True)
class RewardProbe:
    global_task_id: int
    suite: str
    shared_core: torch.Tensor
    per_video_procedure: torch.Tensor
    condition_video_offsets: torch.Tensor
    before_lora: Mapping[str, torch.Tensor]
    query: Mapping[str, torch.Tensor]
    before_action: torch.Tensor
    policy_noise_seed: int


@dataclass(frozen=True)
class ProjectedStep:
    gradient_norm: float
    raw_parameter_delta: Mapping[str, float]
    parameter_delta: Mapping[str, float]
    projection: FinalSupportProjectionSummary


def _claim_task(
    queue: Path, ordered: tuple[RewardTask, ...]
) -> RewardTask | None:
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


def _encode_condition(
    runtime: RewardRuntime, task: RewardTask, cycle: int
) -> tuple[int, tuple[Any, ...], Mapping[str, Any], Any, Mapping[str, torch.Tensor]]:
    visit = cycle - 1
    demos = runtime.video_schedule.demos_for_task_visit(task.global_task_id, visit)
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
        encoded = runtime.writer.encode_program(*packed, policy=runtime.policy)
        rollout_lora = runtime.writer.decode_program(encoded.program)
    return visit, packed, video_metrics, encoded, rollout_lora


def _collect_trajectories(
    runtime: RewardRuntime, task: RewardTask, visit: int
) -> tuple[tuple[RewardTrajectory, ...], float]:
    rollout_cursors = tuple(visit * 4 + lane for lane in range(4))
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
    started = time.monotonic()
    trajectories = collect_randomized_reward_trajectories(
        envs=tuple(runtime.env_pool.get(task, lane=lane) for lane in range(4)),
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
    return tuple(trajectories), time.monotonic() - started


def _empty_summary(
    successes: torch.Tensor,
    episode_ids: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
) -> RewardPreferenceSummary:
    return RewardPreferenceSummary(
        objective=0.0,
        support_objective=0.0,
        successes=int(successes.sum()),
        replay_chunks=int(episode_ids.numel()),
        executed_action_steps=int(batch["executed_action_steps"].sum()),
        functional_policy_forwards=0,
        preference_policy_backwards=0,
        support_policy_backwards=0,
        preference_lora_gradient_rms=0.0,
        support_lora_gradient_rms=0.0,
    )


def _accumulate_writer_cotangent(
    runtime: RewardRuntime,
    generated: Mapping[str, torch.Tensor],
    cotangent: Mapping[str, torch.Tensor],
    destination: torch.Tensor,
    *,
    retain_graph: bool = False,
) -> None:
    backpropagate_lora_cotangent(
        generated, cotangent, retain_graph=retain_graph
    )
    gradients = tuple(item.parameter.grad for item in runtime.gradient_layout)
    accumulate_flat_gradient(destination, gradients, runtime.gradient_layout)
    for item in runtime.gradient_layout:
        item.parameter.grad = None


def _differentiate_task_credit(
    runtime: RewardRuntime,
    task: RewardTask,
    cycle: int,
    packed: tuple[Any, ...],
    encoded: Any,
    rollout_lora: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    episode_ids: torch.Tensor,
    successes: torch.Tensor,
    preference_flat: torch.Tensor,
    support_rows: torch.Tensor,
    support_mask: torch.Tensor,
) -> tuple[RewardPreferenceSummary, bool, float]:
    started = time.monotonic()
    with torch.autocast(
        device_type=runtime.context.device.type,
        dtype=torch.bfloat16,
        enabled=runtime.context.device.type == "cuda",
    ):
        preference_lora, support_lora, summary = functional_reward_lora_gradients(
            runtime.policy,
            rollout_lora,
            runtime.lora_contract,
            batch,
            episode_ids,
            successes,
            mc_samples=int(runtime.config["objective"]["flow_mc_samples"]),
            physical_microbatch_size=int(
                runtime.config["optimization"]["reward_replay_chunk_batch_size"]
            ),
            flow_seed_root=int(runtime.config["rng"]["flow_credit_seed_root"]),
            cycle=cycle,
            global_task_id=task.global_task_id,
            device=runtime.context.device,
        )
        recompiled = runtime.writer.compile_readouts(
            encoded.diagnostics.shared_core_slots,
            encoded.diagnostics.per_video_procedure_slots,
            packed[3],
        )
        generated = runtime.writer.decode_program(recompiled.program)
        if support_lora is not None:
            task_index = next(
                index
                for index, observed in enumerate(runtime.tasks)
                if observed.global_task_id == task.global_task_id
            )
            _accumulate_writer_cotangent(
                runtime,
                generated,
                support_lora,
                support_rows[task_index],
                retain_graph=preference_lora is not None,
            )
            support_mask[task_index] = 1
        if preference_lora is not None:
            _accumulate_writer_cotangent(
                runtime, generated, preference_lora, preference_flat
            )
    return summary, preference_lora is not None, time.monotonic() - started


def _make_probe(
    task: RewardTask,
    packed: tuple[Any, ...],
    encoded: Any,
    rollout_lora: Mapping[str, torch.Tensor],
    trajectory: RewardTrajectory,
) -> RewardProbe:
    return RewardProbe(
        global_task_id=task.global_task_id,
        suite=task.suite,
        shared_core=encoded.diagnostics.shared_core_slots.detach(),
        per_video_procedure=encoded.diagnostics.per_video_procedure_slots.detach(),
        condition_video_offsets=packed[3],
        before_lora={
            name: value.detach().clone() for name, value in rollout_lora.items()
        },
        query={name: value.clone() for name, value in trajectory.observations[0].items()},
        before_action=trajectory.action_chunks[0].clone(),
        policy_noise_seed=trajectory.policy_noise_seeds[0],
    )


def _task_gradient(
    runtime: RewardRuntime,
    task: RewardTask,
    cycle: int,
    preference_flat: torch.Tensor,
    support_rows: torch.Tensor,
    support_mask: torch.Tensor,
    probe: RewardProbe | None,
) -> tuple[dict[str, Any], RewardProbe | None]:
    visit, packed, video_metrics, encoded, rollout_lora = _encode_condition(
        runtime, task, cycle
    )
    copy_task_lora_state_(runtime.policy, rollout_lora, runtime.lora_contract)
    trajectories, rollout_seconds = _collect_trajectories(runtime, task, visit)
    batch, episode_ids, successes = complete_trajectory_batch(
        trajectories, torch.device("cpu")
    )
    copy_task_lora_state_(runtime.policy, runtime.identity_state, runtime.lora_contract)
    mixed = not bool((successes == successes[0]).all())
    summary = _empty_summary(successes, episode_ids, batch)
    credit_seconds = 0.0
    has_preference = False
    if bool(successes.sum()):
        summary, has_preference, credit_seconds = _differentiate_task_credit(
            runtime,
            task,
            cycle,
            packed,
            encoded,
            rollout_lora,
            batch,
            episode_ids,
            successes,
            preference_flat,
            support_rows,
            support_mask,
        )
    if has_preference and probe is None:
        probe = _make_probe(task, packed, encoded, rollout_lora, trajectories[0])
    row = {
        "task_id": task.global_task_id,
        "suite": task.suite,
        "local_task_id": task.task_id,
        "cycle": cycle,
        "mixed": mixed,
        **asdict(summary),
        **video_metrics,
        "rollouts": 4,
        "trajectory_rows": [_trajectory_row(value) for value in trajectories],
        "rollout_seconds": rollout_seconds,
        "credit_seconds": credit_seconds,
        "target_dataset_action_reads": 0,
        "teacher_action_reads": 0,
    }
    return row, probe


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
    return {f"{key}_response_rms": math.sqrt(sums[key] / counts[key]) for key in sums}


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
        encoded = runtime.writer.compile_readouts(
            probe.shared_core,
            probe.per_video_procedure,
            probe.condition_video_offsets,
        )
        after = runtime.writer.decode_program(encoded.program)
    response = _lora_response(probe.before_lora, after)
    copy_task_lora_state_(runtime.policy, after, runtime.lora_contract)
    generator = torch.Generator(device="cpu").manual_seed(probe.policy_noise_seed)
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
        action = runtime.policy.predict_action_chunk(query, noise=noise, num_steps=10)
    copy_task_lora_state_(runtime.policy, runtime.identity_state, runtime.lora_contract)
    response["fixed_action_response_rms"] = float(
        (action.cpu().float() - probe.before_action.float()).square().mean().sqrt()
    )
    return {"task_id": probe.global_task_id, "suite": probe.suite, **response}


def _collect_cycle_tasks(
    runtime: RewardRuntime,
    cycle: int,
    preference_flat: torch.Tensor,
    support_rows: torch.Tensor,
    support_mask: torch.Tensor,
) -> tuple[list[dict[str, Any]], RewardProbe | None, int]:
    if runtime.args.mode == "smoke":
        task_id = int(
            runtime.args.smoke_task_id
            or runtime.config["smoke_run"]["task_global_id"]
        )
        task = next(task for task in runtime.tasks if task.global_task_id == task_id)
        row, probe = _task_gradient(
            runtime,
            task,
            cycle,
            preference_flat,
            support_rows,
            support_mask,
            None,
        )
        return [row], probe, 1
    ordered = tuple(
        sorted(runtime.tasks, key=lambda task: (-task.horizon, task.global_task_id))
    )
    queue = runtime.args.output_dir / f".cycle_{cycle:08d}_task_cursor"
    if runtime.context.is_main:
        queue.write_text("0", encoding="utf-8")
    barrier(runtime.context)
    records, probe = [], None
    while task := _claim_task(queue, ordered):
        row, probe = _task_gradient(
            runtime,
            task,
            cycle,
            preference_flat,
            support_rows,
            support_mask,
            probe,
        )
        records.append(row)
    barrier(runtime.context)
    if runtime.context.is_main:
        queue.unlink(missing_ok=True)
    return records, probe, 24


def _flat_parameters(runtime: RewardRuntime) -> torch.Tensor:
    flattened = torch.empty(
        runtime.gradient_layout[-1].stop,
        dtype=torch.float32,
        device=runtime.context.device,
    )
    for item in runtime.gradient_layout:
        flattened[item.start : item.stop].copy_(item.parameter.detach().reshape(-1))
    return flattened


def _named_delta_rms(
    runtime: RewardRuntime,
    before: Mapping[str, torch.Tensor],
    after_flat: torch.Tensor | None = None,
) -> dict[str, float]:
    values = runtime.writer.procedure_set.named_parameters()
    if after_flat is None:
        return {
            name: float((value.detach() - before[name]).float().square().mean().sqrt())
            for name, value in values
        }
    return {
        name: float(
            (
                after_flat[item.start : item.stop].view_as(value).float()
                - before[name].float()
            )
            .square()
            .mean()
            .sqrt()
        )
        for (name, value), item in zip(values, runtime.gradient_layout, strict=True)
    }


def _apply_projected_step(
    runtime: RewardRuntime,
    preference_flat: torch.Tensor,
    support_rows: torch.Tensor,
    support_mask: torch.Tensor,
    divisor: int,
) -> ProjectedStep:
    if runtime.context.world_size > 1:
        dist.all_reduce(preference_flat, op=dist.ReduceOp.SUM)
        dist.all_reduce(support_rows, op=dist.ReduceOp.SUM)
        dist.all_reduce(support_mask, op=dist.ReduceOp.SUM)
    preference_flat.div_(divisor)
    if not bool(torch.count_nonzero(preference_flat)):
        raise WriterModelError("reward cycle produced zero shared gradient")
    assign_flat_gradient(preference_flat, runtime.gradient_layout)
    clip = float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"])
    grad_norm = torch.nn.utils.clip_grad_norm_(runtime.trainable_parameters, clip)
    clipped_preference = torch.empty_like(preference_flat)
    for item in runtime.gradient_layout:
        if item.parameter.grad is None:
            raise WriterModelError("support projection lost clipped preference gradient")
        clipped_preference[item.start : item.stop].copy_(
            item.parameter.grad.detach().reshape(-1)
        )
    before_flat = _flat_parameters(runtime)
    before = {
        name: value.detach().clone()
        for name, value in runtime.writer.procedure_set.named_parameters()
    }
    runtime.optimizer.step()
    raw_after = _flat_parameters(runtime)
    projected_delta, projection = project_final_parameter_delta(
        raw_after - before_flat,
        support_rows,
        support_mask.bool(),
        clipped_preference,
    )
    projected_after = before_flat + projected_delta
    with torch.no_grad():
        for item in runtime.gradient_layout:
            item.parameter.copy_(
                projected_after[item.start : item.stop].view_as(item.parameter)
            )
    return ProjectedStep(
        gradient_norm=float(grad_norm),
        raw_parameter_delta=_named_delta_rms(runtime, before, raw_after),
        parameter_delta=_named_delta_rms(runtime, before),
        projection=projection,
    )


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
    torch.cuda.synchronize(runtime.context.device)
    elapsed = torch.tensor(
        time.monotonic() - started, dtype=torch.float64, device=runtime.context.device
    )
    if runtime.context.world_size > 1:
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return global_records, [value for value in probes if value is not None], float(elapsed)


def _cycle_metrics(
    runtime: RewardRuntime,
    cycle: int,
    records: list[dict[str, Any]],
    probes: list[dict[str, Any]],
    step: ProjectedStep,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "cycle": cycle,
        "cycle_semantics": (
            "one_complete_train24_reward_preference_actual_delta_support_projection"
            if runtime.args.mode == "formal"
            else "one_task_support_projection_live_smoke"
        ),
        "tasks": len(records),
        "rollouts": 4 * len(records),
        "successes": sum(int(row["successes"]) for row in records),
        "mixed_tasks": sum(bool(row["mixed"]) for row in records),
        "support_tasks": sum(int(row["successes"]) > 0 for row in records),
        "all_success_tasks": sum(int(row["successes"]) == 4 for row in records),
        "all_failure_tasks": sum(int(row["successes"]) == 0 for row in records),
        "mixed_suites": sorted({row["suite"] for row in records if row["mixed"]}),
        "replay_chunks": sum(int(row["replay_chunks"]) for row in records),
        "executed_action_steps": sum(
            int(row["executed_action_steps"]) for row in records
        ),
        "reward_objective_mean": math.fsum(
            float(row["objective"]) for row in records
        )
        / len(records),
        "support_objective_mean": math.fsum(
            float(row["support_objective"]) for row in records
        )
        / len(records),
        "writer_gradient_norm_before_clip": step.gradient_norm,
        "raw_parameter_delta_rms": step.raw_parameter_delta,
        "parameter_delta_rms": step.parameter_delta,
        "final_support_projection": asdict(step.projection),
        "deployment_response_probes": probes,
        "task_records": records,
        "cycle_seconds": elapsed,
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(
            runtime.context.device
        ),
        "max_cuda_reserved_bytes": torch.cuda.max_memory_reserved(
            runtime.context.device
        ),
        "target_dataset_action_reads": 0,
        "teacher_action_reads": 0,
        "validation_action_or_reward_reads": 0,
        "test_action_or_reward_reads": 0,
    }


def run_cycle(runtime: RewardRuntime, cycle: int) -> dict[str, Any]:
    started = time.monotonic()
    runtime.optimizer.zero_grad(set_to_none=True)
    width = runtime.gradient_layout[-1].stop
    preference_flat = torch.zeros(
        width, dtype=torch.float32, device=runtime.context.device
    )
    support_rows = torch.zeros(
        len(runtime.tasks), width, dtype=torch.float32, device=runtime.context.device
    )
    support_mask = torch.zeros(
        len(runtime.tasks), dtype=torch.int32, device=runtime.context.device
    )
    records, probe, divisor = _collect_cycle_tasks(
        runtime, cycle, preference_flat, support_rows, support_mask
    )
    step = _apply_projected_step(
        runtime, preference_flat, support_rows, support_mask, divisor
    )
    records, probes, elapsed = _gather_cycle_evidence(
        runtime, records, probe, started
    )
    return _cycle_metrics(runtime, cycle, records, probes, step, elapsed)
