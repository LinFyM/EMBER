"""Fixed-route shared EBSRI S2 functional-polish training.

The S1 task-local module remains the numerical owner for real B0/B1 replay,
rank16 materialization, and the effective-rank4 cache used only by evaluation.
Training uses exact cross-episode Panel-A LoRA-leaf VJPs with a memory-safe
no-grad bank pass followed by one fresh Writer replay.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.ecp.checkpoint import save_ecp_checkpoint
from ember.ecp.contracts import TargetFamily
from ember.ecp.joint_program_primal.bank_set_shared_contract import (
    functional_arm_objective as _functional_arm_objective,
    task_panel_a_visit,
)
from ember.ecp.joint_program_primal.bank_set_tasklocal import (
    TaskLocalArm,
    _base_output,
    _complete,
    _family_distances,
    _output,
    _prepare_bank,
    _target,
    _wrong_teacher,
)
from ember.ecp.joint_program_primal.bank_set_tasklocal_evaluation import (
    FAMILIES,
    EffectiveTarget,
)
from ember.ecp.joint_program_primal.train_step import (
    backward_functional_derivative,
    functional_loss_derivative,
    functional_panel_batch,
)
from ember.pi05_source_checkpoint import (
    capture_rng,
    restore_rng,
    write_json_atomic,
)
from ember.pi05_source_contract import append_jsonl
from ember.pi05_source_setup import initialize_distributed


GRADIENT_META_TASKS = (8, 9, 32, 52)
GRADIENT_TARGET_TASKS = (72, 73, 75, 94)
GRADIENT_TASKS = (*GRADIENT_META_TASKS, *GRADIENT_TARGET_TASKS)
HELD_INTERACTION_TASKS = (1, 93)
ALL_INTERACTION_TASKS = (1, 8, 9, 32, 52, 72, 73, 75, 93, 94)
WRONG_TASK_RING = {8: 9, 9: 32, 32: 52, 52: 8, 72: 73, 73: 75, 75: 94, 94: 72}


@dataclass(frozen=True)
class SharedTaskTargets:
    """Small CPU-only targets retained for one Program task."""

    targets: Mapping[str, EffectiveTarget]
    denominators: Mapping[TargetFamily, torch.Tensor]
    wrong_teacher_metrics: Mapping[str, Any]
    authority: Mapping[str, str]


@dataclass(frozen=True)
class SharedArmSpec:
    task: int
    role: str
    name: str
    bank_task: int
    condition: Any
    receives_gradient: bool


def _contract_module() -> Any:
    # The independently owned S2 config/contract workstream lands this module.
    from ember.ecp.joint_program_primal import bank_set_shared_contract

    return bank_set_shared_contract


def task_cursor_counts(optimizer_step: int) -> dict[int, int]:
    """Reconstruct all-task paired-functional cursors from the global macro."""

    if optimizer_step < 0:
        raise ValueError("S2 optimizer step must be non-negative")
    return {task: optimizer_step for task in GRADIENT_TASKS}


def balanced_task_assignments(
    tasks: Sequence[int], costs: Mapping[int, int], world_size: int
) -> tuple[tuple[int, ...], ...]:
    """Greedily shard tasks while retaining one live arm at a time per rank."""

    ordered = tuple(map(int, tasks))
    if not ordered or len(set(ordered)) != len(ordered) or not 1 <= world_size <= 6:
        raise ValueError("S2 task assignment contract changed")
    if set(ordered) != set(map(int, costs)):
        raise ValueError("S2 task costs do not cover the assignment")
    rows: list[list[int]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for task in sorted(ordered, key=lambda value: (-int(costs[value]), ordered.index(value))):
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        rows[rank].append(task)
        loads[rank] += int(costs[task])
    return tuple(tuple(row) for row in rows)


def _validate_shared_config(config: Mapping[str, Any]) -> None:
    shared = config["shared_training"]
    split = config["task_split"]
    rings = {int(task): int(wrong) for task, wrong in shared["wrong_task_by_task"].items()}
    profiles = shared["task_profiles"]
    checks = (
        tuple(map(int, split["gradient_meta"])) == GRADIENT_META_TASKS,
        tuple(map(int, split["gradient_target"])) == GRADIENT_TARGET_TASKS,
        tuple(map(int, shared["gradient_task_ids"])) == GRADIENT_TASKS,
        tuple(map(int, shared["held_interaction_task_ids"])) == HELD_INTERACTION_TASKS,
        tuple(shared["optimizer_step_arms"])
        == ("alternating_correct_fit0_fit1", "wrong_fit0"),
        shared["correct_view_schedule"] == "fit0_even_fit1_odd_optimizer_step",
        {task: rings[task] for task in GRADIENT_TASKS} == WRONG_TASK_RING,
        not set(HELD_INTERACTION_TASKS).intersection(
            rings[task] for task in GRADIENT_TASKS
        ),
        set(map(int, profiles)) == set(ALL_INTERACTION_TASKS),
    )
    if not all(checks):
        raise ValueError("S2 fixed-route shared task contract changed")
    for task, row in profiles.items():
        values = (
            int(row["replay_frame_chunk_size"]),
            int(row["interaction_group_batch_size"]),
            int(row["functional_policy_microbatch_size"]),
        )
        if min(values) <= 0:
            raise ValueError(f"S2 task profile is invalid for task {task}")


def _asset_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt only asset loading to the existing ten-task R5 authority."""

    result = copy.deepcopy(config)
    result["task_split"]["gradient_meta"] = [1, 8, 9, 32, 52]
    result["task_split"]["gradient_target"] = [72, 73, 75, 93, 94]
    return result


def _apply_task_profile(runtime: Any, task: int) -> dict[str, int]:
    row = runtime.config["shared_training"]["task_profiles"][str(int(task))]
    profile = {
        "replay_frame_chunk_size": int(row["replay_frame_chunk_size"]),
        "interaction_group_batch_size": int(row["interaction_group_batch_size"]),
        "functional_policy_microbatch_size": int(
            row["functional_policy_microbatch_size"]
        ),
    }
    runtime.compiler.bank_operator.covariance_frame_chunk = profile[
        "replay_frame_chunk_size"
    ]
    runtime.config["model"]["interaction_group_batch_size"] = profile[
        "interaction_group_batch_size"
    ]
    runtime.config["optimization"]["functional_policy_microbatch_size"] = profile[
        "functional_policy_microbatch_size"
    ]
    return profile


def _wrong_task(runtime: Any, task: int) -> int:
    shared = runtime.config["shared_training"]
    pairs = {
        int(left): int(right)
        for source in (
            shared.get("wrong_task_by_task", {}),
            shared.get("evaluation_wrong_task_by_task", {}),
        )
        for left, right in source.items()
    }
    if task not in pairs:
        raise ValueError(f"S2 task {task} has no registered wrong bank")
    return pairs[task]


def _arm_spec(runtime: Any, task: int, name: str) -> SharedArmSpec:
    task = int(task)
    if task not in runtime.task_conditions:
        raise ValueError(f"S2 task {task} has no mapping conditions")
    correct = runtime.task_conditions[task]
    if name == "correct_fit0":
        bank_task, condition, gradient = task, correct.fit_views[0], True
    elif name == "correct_fit1":
        bank_task, condition, gradient = task, correct.fit_views[1], True
    elif name == "correct_held":
        bank_task, condition, gradient = task, correct.held_video, False
    elif name in {"wrong_fit0", "wrong_fit1"}:
        bank_task = _wrong_task(runtime, task)
        wrong = runtime.task_conditions[bank_task]
        condition = wrong.fit_views[0 if name == "wrong_fit0" else 1]
        gradient = name == "wrong_fit0"
    else:
        raise ValueError(f"unsupported S2 arm: {name}")
    return SharedArmSpec(
        task=task,
        role=str(runtime.panels[task].role),
        name=name,
        bank_task=bank_task,
        condition=condition,
        receives_gradient=gradient and task in GRADIENT_TASKS,
    )


def _prepare_arm(runtime: Any, spec: SharedArmSpec) -> TaskLocalArm:
    _apply_task_profile(runtime, spec.task)
    bank = _prepare_bank(
        runtime,
        program_task=spec.task,
        bank_task=spec.bank_task,
        condition=spec.condition,
    )
    interaction_state = runtime.compiler._interaction_states(
        bank.program_state, (bank.context,)
    )[0]
    return TaskLocalArm(
        name=spec.name,
        bank=bank,
        interaction_state=interaction_state,
        summaries=None,
        receives_gradient=spec.receives_gradient,
    )


def _shared_interaction_output(
    runtime: Any,
    arm: TaskLocalArm,
    *,
    correction_observer: Any | None = None,
) -> Any:
    """Execute the S1-proven real B0/B1 path without the task-local stage guard."""

    pooled = runtime.compiler.bank_operator.apply_compact(
        arm.bank.video,
        arm.bank.input_primals,
        arm.bank.output_primals,
        bank_set_interaction=runtime.compiler.bank_set_interaction,
        interaction_state=arm.interaction_state,
        correction_observer=correction_observer,
        frozen_descriptors=arm.bank.frozen_descriptors,
        interaction_group_batch_size=int(
            runtime.config["model"]["interaction_group_batch_size"]
        ),
        replay_plan=arm.bank.plan,
    )
    return _output(runtime, arm.bank, pooled)


def _cpu_target(value: EffectiveTarget) -> EffectiveTarget:
    return EffectiveTarget(
        a=tuple(row.detach().to("cpu").contiguous() for row in value.a),
        b=tuple(row.detach().to("cpu").contiguous() for row in value.b),
    )


def _device_target(value: EffectiveTarget, device: torch.device) -> EffectiveTarget:
    return EffectiveTarget(
        a=tuple(row.to(device=device) for row in value.a),
        b=tuple(row.to(device=device) for row in value.b),
    )


def _clear_panel_cache(runtime: Any, task: int) -> None:
    for key in tuple(runtime.panel_batch_cache):
        if int(key[0]) == int(task):
            del runtime.panel_batch_cache[key]


def _shared_wrong_teacher(
    runtime: Any, task: int, arm: TaskLocalArm, base_output: Any
) -> tuple[Any, Mapping[str, Any]]:
    """Bridge the diagnostic-only S2 cache setting to the S1 helper."""

    joint = runtime.config["optimization"]["joint"]
    if "wrong_free_delta_teacher" in joint:
        raise ValueError("S2 wrong-teacher settings escaped their sealed location")
    joint["wrong_free_delta_teacher"] = runtime.config["evaluation"][
        "target_cache_wrong_free_delta_teacher"
    ]
    try:
        return _wrong_teacher(runtime, task, arm, base_output)
    finally:
        del joint["wrong_free_delta_teacher"]


def _build_task_targets(runtime: Any, task: int) -> SharedTaskTargets:
    targets: dict[str, EffectiveTarget] = {}
    for name in ("correct_fit0", "correct_fit1", "correct_held"):
        arm = _prepare_arm(runtime, _arm_spec(runtime, task, name))
        with torch.no_grad():
            targets[name] = _cpu_target(_target(_base_output(runtime, arm.bank)))
        del arm

    wrong_arm = _prepare_arm(runtime, _arm_spec(runtime, task, "wrong_fit0"))
    with torch.no_grad():
        wrong_base = _base_output(runtime, wrong_arm.bank)
    wrong_teacher, teacher_metrics = _shared_wrong_teacher(
        runtime, task, wrong_arm, wrong_base
    )
    wrong_target = _target(wrong_teacher)
    denominators = _family_distances(runtime, wrong_base, wrong_target)
    if any(
        not bool(torch.isfinite(value)) or float(value) <= 1e-12
        for value in denominators.values()
    ):
        raise RuntimeError("S2 wrong teacher missed an effective-rank4 family")
    targets["wrong_fit0"] = _cpu_target(wrong_target)
    result = SharedTaskTargets(
        targets=targets,
        denominators={
            family: value.detach().to("cpu").contiguous()
            for family, value in denominators.items()
        },
        wrong_teacher_metrics=teacher_metrics,
        authority={
            "correct": "each_bank_frozen_r5_base_residual",
            "wrong": (
                "task_wrong_fit0_one_round_functional_free_delta_"
                "suppressive_teacher"
            ),
            "denominator": (
                "wrong_fit0_r5_base_to_suppressive_teacher_squared_distance"
            ),
        },
    )
    del wrong_teacher, wrong_target, wrong_base, wrong_arm
    _clear_panel_cache(runtime, task)
    return result


def _target_cost(runtime: Any, task: int) -> int:
    correct = runtime.task_conditions[task]
    wrong = runtime.task_conditions[_wrong_task(runtime, task)]
    return sum(
        int(condition.sampled_frames)
        for condition in (*correct.fit_views, correct.held_video, wrong.fit_views[0])
    )


def prepare_shared_target_cache(
    runtime: Any,
    tasks: Sequence[int],
    *,
    distributed: bool,
) -> dict[int, SharedTaskTargets]:
    """Build each task once, CPU-offload it, then optionally share across ranks."""

    task_ids = tuple(map(int, tasks))
    if len(task_ids) != len(set(task_ids)) or not set(task_ids).issubset(
        ALL_INTERACTION_TASKS
    ):
        raise ValueError("S2 target-cache task coverage changed")
    world_size = runtime.context.world_size if distributed else 1
    rank = runtime.context.rank if distributed else 0
    costs = {task: _target_cost(runtime, task) for task in task_ids}
    assignments = balanced_task_assignments(task_ids, costs, world_size)
    rng = capture_rng(runtime.context)
    try:
        local = {
            task: _build_task_targets(runtime, task) for task in assignments[rank]
        }
    finally:
        restore_rng(rng, runtime.context)
        runtime.optimizer.zero_grad(set_to_none=True)
    if distributed and world_size > 1:
        rows: list[Any] = [None] * world_size
        dist.all_gather_object(rows, local)
    else:
        rows = [local]
    cache: dict[int, SharedTaskTargets] = {}
    for row in rows:
        for task, value in row.items():
            if int(task) in cache:
                raise RuntimeError("S2 target cache duplicated a task")
            cache[int(task)] = value
    if set(cache) != set(task_ids):
        raise RuntimeError("S2 target cache is incomplete")
    for value in cache.values():
        tensors = (
            *(row for target in value.targets.values() for row in (*target.a, *target.b)),
            *value.denominators.values(),
        )
        if any(row.device.type != "cpu" or row.requires_grad for row in tensors):
            raise RuntimeError("S2 target cache retained GPU/autograd state")
    return cache


def _cursor_tensor(runtime: Any) -> torch.Tensor:
    value = getattr(runtime.writer_state, "task_arm_cursors", None)
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.int64
        or value.shape != (len(GRADIENT_TASKS),)
    ):
        raise RuntimeError("S2 checkpoint cursor buffer changed")
    return value


def _validate_task_cursors(runtime: Any) -> dict[int, int]:
    expected = task_cursor_counts(runtime.optimizer_steps)
    actual = {
        task: int(value)
        for task, value in zip(
            GRADIENT_TASKS, _cursor_tensor(runtime).detach().cpu().tolist(), strict=True
        )
    }
    if actual != expected:
        raise ValueError(f"S2 task cursors disagree with macro: {actual} != {expected}")
    return actual


def _advance_task_cursors(runtime: Any, group: Sequence[int]) -> None:
    indices = {task: index for index, task in enumerate(GRADIENT_TASKS)}
    with torch.no_grad():
        cursor = _cursor_tensor(runtime)
        for task in group:
            cursor[indices[int(task)]] += 1


def _cpu_leaf_gradients(
    gradients: Mapping[str, torch.Tensor], *, active: bool
) -> dict[str, torch.Tensor]:
    result = {
        name: (
            value.detach().to(device="cpu").contiguous()
            if active
            else torch.zeros_like(
                value, device="cpu", memory_format=torch.preserve_format
            )
        )
        for name, value in gradients.items()
    }
    if not result or any(value.requires_grad for value in result.values()):
        raise RuntimeError("S2 Panel-A leaf-gradient offload changed")
    return result


def _functional_task_loss(
    runtime: Any,
    task: int,
    arm_name: str,
    *,
    task_cursor: int,
    task_weight: float,
) -> dict[str, Any]:
    """Backpropagate one exact Panel-A VJP with no bank/policy graph overlap."""

    tick = time.monotonic()
    spec = _arm_spec(runtime, task, arm_name)
    if not spec.receives_gradient or task in HELD_INTERACTION_TASKS:
        raise RuntimeError("S2 selected a zero-gradient interaction arm")
    visit = task_panel_a_visit(
        task_cursor, int(runtime.config["data"]["panel_visits"])
    )

    # First pass materializes only detached rank16 leaves.  The real bank and
    # its no-grad replay are released before the frozen policy VJP begins.
    first_arm = _prepare_arm(runtime, spec)
    with torch.no_grad():
        first_output = _shared_interaction_output(runtime, first_arm)
        detached_state = {
            name: value.detach()
            for name, value in _complete(runtime, first_output).items()
        }
    condition_metrics = dict(first_arm.bank.condition_metrics)
    del first_output, first_arm

    batch, panel = functional_panel_batch(
        runtime, task_id=task, panel_name="a", visit_index=visit
    )
    generated_loss, leaf_gradients = functional_loss_derivative(
        runtime,
        state=detached_state,
        batch=batch,
        policy_rng_seed=panel.policy_rng_seed,
    )
    settings = runtime.config["optimization"]["direct_functional"]
    objective = _functional_arm_objective(
        arm_name,
        generated_loss=generated_loss,
        carrier_loss=float(panel.flow_loss),
        correct_backward_mass=float(settings["correct_backward_mass"]),
        wrong_backward_mass=float(settings["wrong_backward_mass"]),
    )
    cpu_leaf_gradients = _cpu_leaf_gradients(
        leaf_gradients, active=objective.gradient_active
    )
    del leaf_gradients, detached_state, batch
    _clear_panel_cache(runtime, task)

    # Second pass is the only Writer graph.  Even an inactive wrong hinge uses
    # explicit zero leaf gradients, so every shared parameter has a grad tensor.
    arm = _prepare_arm(runtime, spec)
    output = _shared_interaction_output(runtime, arm)
    state = _complete(runtime, output)
    device_leaf_gradients = {
        name: value.to(device=runtime.context.device)
        for name, value in cpu_leaf_gradients.items()
    }
    backward_functional_derivative(
        state,
        device_leaf_gradients,
        weight=objective.backward_mass * float(task_weight),
    )
    row = {
        "task": task,
        "role": spec.role,
        "arm": arm_name,
        "bank_task": spec.bank_task,
        "video_demo": int(spec.condition.video_demo),
        "task_weight": float(task_weight),
        "panel": "a",
        "panel_visit": visit,
        "functional_policy_rng_seed": int(panel.policy_rng_seed),
        "action_demos": list(panel.action_demos),
        "action_frames": list(panel.action_frames),
        "carrier_flow_loss": float(panel.flow_loss),
        "generated_flow_loss": float(generated_loss),
        "benefit_over_carrier": objective.benefit_over_carrier,
        "training_objective": objective.value,
        "objective_kind": objective.kind,
        "backward_mass": objective.backward_mass,
        "applied_backward_mass": objective.backward_mass * float(task_weight),
        "gradient_active": objective.gradient_active,
        "condition_metrics": condition_metrics,
        "profile": _apply_task_profile(runtime, task),
        "memory_schedule": "no_grad_bank_leaf_vjp_cpu_offload_fresh_bank_replay",
        "task_seconds": time.monotonic() - tick,
    }
    del state, output, arm, device_leaf_gradients, cpu_leaf_gradients
    return row


def run_shared_optimizer_step(runtime: Any) -> dict[str, Any]:
    """Run the canonical all-task paired unit-gradient polish step."""

    from ember.ecp.joint_program_primal.bank_set_shared_gradient_combiner import (
        run_paired_unit_gradient_step,
    )

    return run_paired_unit_gradient_step(runtime)


def _train(runtime: Any) -> None:
    contract = _contract_module()
    while runtime.optimizer_steps < runtime.stop_after_step:
        row = run_shared_optimizer_step(runtime)
        if runtime.context.is_main:
            append_jsonl(runtime.args.output_dir / "metrics.jsonl", row)
            runtime.metrics_rows += 1
            if runtime.optimizer_steps % runtime.args.log_every == 0:
                print(row, flush=True)
        if runtime.optimizer_steps in runtime.checkpoint_steps:
            save_ecp_checkpoint(
                output_dir=runtime.args.output_dir,
                macro=runtime.optimizer_steps,
                stage=contract.BANK_SET_SHARED_STAGE,
                context=runtime.context,
                model=runtime.writer_state,
                optimizer=runtime.optimizer,
                scheduler=runtime.scheduler,
                run_contract_schema=contract.BANK_SET_SHARED_RUN_SCHEMA,
                metrics_rows=runtime.metrics_rows,
            )


def run(args: Any) -> None:
    if args.phase != "shared_loto":
        raise ValueError("S2 shared runner requires the shared_loto phase")
    from ember.ecp.joint_program_primal.bank_set_shared_runtime import (
        prepare_shared_training_runtime,
    )

    context = initialize_distributed(
        require_numa=args.mode == "formal", defer_process_group=True
    )
    runtime = None
    try:
        runtime = prepare_shared_training_runtime(args, context)
        inventory = runtime.run_contract["inventory"]
        if (
            int(inventory.get("action_meta_module_count", -1)) != 0
            or int(inventory.get("action_meta_parameter_count", -1)) != 0
        ):
            raise RuntimeError("S2 runtime loaded Action Meta")
        if runtime.context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.context.device)
        _train(runtime)
        if runtime.context.is_main:
            write_json_atomic(
                args.output_dir / "completion.json",
                {
                    "stage": _contract_module().BANK_SET_SHARED_STAGE,
                    "status": "complete",
                    "completed_optimizer_steps": runtime.optimizer_steps,
                    "task_arm_cursors": {
                        str(task): value
                        for task, value in _validate_task_cursors(runtime).items()
                    },
                    "functional_training": {
                        "panel": "a",
                        "cumulative_vjp_calls_from_polish_step0": (
                            runtime.optimizer_steps * 16
                        ),
                        "target_cache_builds": 0,
                        "target_cache_scope": "evaluation_diagnostics_gate_only",
                        "memory_schedule": (
                            "no_grad_bank_leaf_vjp_cpu_offload_fresh_bank_replay"
                        ),
                    },
                    "information_wall": {
                        "held_interaction_task_backward_calls": 0,
                        "held_as_training_wrong_bank_calls": 0,
                        "same_task_held_backward_calls": 0,
                        "wrong_fit1_backward_calls": 0,
                        "panel_b_backward_calls": 0,
                        "cumulative_result_or_action_gradient_calls_from_polish_step0": (
                            runtime.optimizer_steps * 16
                        ),
                        "validation_or_test_reads": 0,
                        "action_meta_installed": False,
                        "single_complete_rank16": True,
                        "shuffled_or_reversed_use": False,
                    },
                },
            )
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def prepare_shared_evaluation_runtime(args: Any, context: Any) -> Any:
    from ember.ecp.joint_program_primal.bank_set_shared_runtime import (
        prepare_shared_evaluation_runtime as implementation,
    )

    return implementation(args, context)


def load_shared_checkpoint(runtime: Any, checkpoint: Path) -> Mapping[str, Any]:
    from ember.ecp.joint_program_primal.bank_set_shared_runtime import (
        load_shared_checkpoint as implementation,
    )

    return implementation(runtime, checkpoint)


def evaluate_shared_job(
    runtime: Any, job: Mapping[str, Any], *,
    target_cache: Mapping[int, SharedTaskTargets],
) -> Mapping[str, Any]:
    from ember.ecp.joint_program_primal.bank_set_shared_runtime import (
        evaluate_shared_job as implementation,
    )

    return implementation(runtime, job, target_cache=target_cache)


def release_shared_job(runtime: Any, job: Mapping[str, Any]) -> Mapping[str, int]:
    from ember.ecp.joint_program_primal.bank_set_shared_runtime import (
        release_shared_job as implementation,
    )

    return implementation(runtime, job)
