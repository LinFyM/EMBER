"""One task-equal structured simulator calibration inside canonical MDCO Stage 1."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.ecp.compiler import select_compiled_state
from ember.ecp.contracts import TargetOwner
from ember.ecp.stage1_calibration_contract import (
    STRUCTURED_CALIBRATION_FILE,
    STRUCTURED_CALIBRATION_SCHEMA,
    calibration_task_count,
    successful_panel_for_visit,
)
from ember.ecp.stage1_config import stage1_asset_authority
from ember.ecp.stage1_objective import ecp_stage1_loss
from ember.ecp.stage1_support import (
    policy_support_activation_distillation_loss,
    shared_prior_response_distillation_loss,
)
from ember.ecp.stage1_train_step import (
    encode_stage1_visit,
    stage1_action_policy_gradient,
)
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, copy_task_lora_state_
from ember.pi05_assets import prepare_libero_config
from ember.pi05_source_checkpoint import barrier, write_json_atomic
from ember.reward.credit import paired_antithetic_credit
from ember.reward.protocol import (
    RewardProtocolError,
    RewardTask,
    reward_credit_environment_seed,
    reward_credit_update_seed,
)
from ember.reward.rollout import (
    RandomResetEnvironmentPool,
    RewardTrajectory,
    capture_paired_initial_states,
    collect_paired_reward_arm_trajectories,
)
from ember.writer.topology import visible_physical_cuda_index

if TYPE_CHECKING:
    from ember.ecp.stage1_training import ECPStage1Runtime


@dataclass(frozen=True)
class ActionGuidedFactorPerturbation:
    """One complete-LoRA perturbation with an owner-local A/B direction."""

    epsilon: torch.Tensor
    sigma: float
    directions: Mapping[str, torch.Tensor]
    direction_norm_sq: torch.Tensor
    plus_state: Mapping[str, torch.Tensor]
    minus_state: Mapping[str, torch.Tensor]
    active_owners: int


def _rademacher(count: int, *, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (
        torch.randint(0, 2, (1, count), generator=generator, dtype=torch.float32)
        .mul_(2.0)
        .sub_(1.0)
        .to(device)
    )


def action_guided_factor_perturbation(
    state: Mapping[str, torch.Tensor],
    action_gradients: Mapping[str, torch.Tensor],
    owners: tuple[TargetOwner, ...],
    *,
    sigma: float,
    seed: int,
) -> ActionGuidedFactorPerturbation:
    expected = {
        owner.target_name + suffix
        for owner in owners
        for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX)
    }
    if set(state) != expected or set(action_gradients) != expected or sigma <= 0:
        raise RewardProtocolError("invalid structured calibration factor surface")
    first = next(iter(state.values()))
    epsilon = _rademacher(len(owners), seed=seed, device=first.device)
    directions: dict[str, torch.Tensor] = {}
    plus: dict[str, torch.Tensor] = {}
    minus: dict[str, torch.Tensor] = {}
    norm_squares = []
    active = 0
    for owner in owners:
        name_a = owner.target_name + LORA_A_SUFFIX
        name_b = owner.target_name + LORA_B_SUFFIX
        base_a = state[name_a].detach().float()
        base_b = state[name_b].detach().float()
        gradient_a = action_gradients[name_a].detach().float()
        gradient_b = action_gradients[name_b].detach().float()
        base_norm_sq = base_a.square().sum() + base_b.square().sum()
        gradient_norm_sq = gradient_a.square().sum() + gradient_b.square().sum()
        if not bool(torch.isfinite(base_norm_sq + gradient_norm_sq)) or float(
            base_norm_sq
        ) <= 0:
            raise RewardProtocolError("non-finite structured calibration direction")
        if float(gradient_norm_sq) > 0:
            scale = (base_norm_sq / gradient_norm_sq).sqrt()
            direction_a = -gradient_a * scale
            direction_b = -gradient_b * scale
            active += 1
        else:
            direction_a = torch.zeros_like(base_a)
            direction_b = torch.zeros_like(base_b)
            epsilon[:, owner.index] = 0
        coefficient = sigma * epsilon[0, owner.index]
        directions[name_a] = direction_a
        directions[name_b] = direction_b
        plus[name_a] = (base_a + coefficient * direction_a).to(state[name_a])
        plus[name_b] = (base_b + coefficient * direction_b).to(state[name_b])
        minus[name_a] = (base_a - coefficient * direction_a).to(state[name_a])
        minus[name_b] = (base_b - coefficient * direction_b).to(state[name_b])
        norm_squares.append(base_norm_sq)
    return ActionGuidedFactorPerturbation(
        epsilon=epsilon,
        sigma=float(sigma),
        directions=directions,
        direction_norm_sq=torch.stack(norm_squares),
        plus_state=plus,
        minus_state=minus,
        active_owners=active,
    )


def action_guided_outcome_leaf_gradients(
    perturbation: ActionGuidedFactorPerturbation,
    owners: tuple[TargetOwner, ...],
    coordinate_gradient: torch.Tensor,
    *,
    weight: float,
) -> dict[str, torch.Tensor]:
    if (
        coordinate_gradient.shape != (1, len(owners))
        or perturbation.direction_norm_sq.shape != (len(owners),)
        or weight <= 0
        or not bool(torch.isfinite(coordinate_gradient).all())
    ):
        raise RewardProtocolError("invalid structured calibration outcome gradient")
    result: dict[str, torch.Tensor] = {}
    for owner in owners:
        denominator = perturbation.direction_norm_sq[owner.index].clamp_min(1e-20)
        coefficient = -weight * coordinate_gradient[0, owner.index] / denominator
        for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX):
            name = owner.target_name + suffix
            result[name] = (coefficient * perturbation.directions[name]).to(
                perturbation.plus_state[name]
            )
    return result


def _configure_egl(runtime: "ECPStage1Runtime") -> None:
    expected = {
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_EGL_DEVICE_ID": str(
            visible_physical_cuda_index(runtime.context.local_rank)
        ),
    }
    for name, value in expected.items():
        observed = os.environ.get(name)
        if observed not in {None, value}:
            raise ValueError(f"MDCO calibration {name} mapping changed")
        os.environ[name] = value


def _environment_pool(runtime: "ECPStage1Runtime") -> RandomResetEnvironmentPool:
    _configure_egl(runtime)
    assets_root = stage1_asset_authority(
        runtime.config, "libero_assets_root", runtime.args.asset_root
    ).resolve()
    if not assets_root.is_dir():
        raise ValueError(f"MDCO calibration LIBERO assets missing: {assets_root}")
    os.environ["EMBER_LIBERO_ASSETS_ROOT"] = str(assets_root)
    payload: list[Any] = [None]
    if runtime.context.is_main:
        try:
            payload[0] = prepare_libero_config(
                runtime.args.output_dir / "libero_config"
            )
        except Exception as error:
            payload[0] = {"error": repr(error)}
    if runtime.context.world_size > 1:
        dist.broadcast_object_list(payload, src=0, device=runtime.context.device)
    paths = payload[0]
    if not isinstance(paths, Mapping) or paths.get("error"):
        raise ValueError(f"MDCO calibration LIBERO preparation failed: {paths}")
    os.environ["LIBERO_CONFIG_PATH"] = str(
        (runtime.args.output_dir / "libero_config").resolve()
    )
    environment = runtime.config["environment"]
    return RandomResetEnvironmentPool(
        bddl_root=Path(str(paths["bddl_files"])),
        assets_root=Path(str(paths["assets"])),
        render_resolution=int(environment["render_resolution"]),
    )


def _arm_rollout(
    runtime: "ECPStage1Runtime",
    *,
    pool: RandomResetEnvironmentPool,
    task: RewardTask,
    adapter: Mapping[str, torch.Tensor],
    rollout_cursors: Sequence[int],
    environment_seeds: Sequence[int],
    initial_states: Sequence[Any],
) -> tuple[RewardTrajectory, ...]:
    processor = runtime.calibration.processor
    if processor is None:
        raise ValueError("MDCO calibration processor is unavailable")
    copy_task_lora_state_(runtime.policy, adapter, runtime.contract)
    environment = runtime.config["environment"]
    calibration = runtime.config["structured_calibration"]
    return collect_paired_reward_arm_trajectories(
        envs=tuple(pool.get(task, lane=lane) for lane in range(2)),
        policy=runtime.policy,
        preprocess=processor,
        postprocess=processor.unnormalize_action,
        suite=task.suite,
        task_id=task.task_id,
        global_task_id=task.global_task_id,
        language=task.language,
        adaptation_seed=int(runtime.config["optimization"]["seed"]),
        rollout_cursors=rollout_cursors,
        env_seeds=environment_seeds,
        policy_seed_root=int(calibration["policy_noise_seed_root"]),
        device=runtime.context.device,
        max_horizon=task.horizon,
        dummy_settling_steps=int(environment["dummy_settling_steps"]),
        dummy_action=environment["dummy_action"],
        action_execution_horizon=int(environment["action_execution_horizon"]),
        num_inference_steps=int(environment["num_inference_steps"]),
        initial_states=initial_states,
        capture_replay=False,
        capture_goal_progress=True,
    )


def _trajectory_record(value: RewardTrajectory) -> dict[str, Any]:
    return {
        "rollout_cursor": value.rollout_cursor,
        "environment_seed": value.env_seed,
        "success": value.success,
        "steps": value.steps,
        "goal_predicate_count": value.goal_predicate_count,
        "goal_predicate_peak": value.goal_predicate_peak,
    }


def _tensor_dict_norm(values: Mapping[str, torch.Tensor]) -> float:
    return float(
        torch.stack([value.float().square().sum() for value in values.values()])
        .sum()
        .sqrt()
    )


@dataclass(frozen=True)
class CalibrationProposal:
    packed: Any
    encoded: Any
    evidence: Any
    panel: Any
    cached: Any
    action_loss: torch.Tensor
    action_gradients: Mapping[str, torch.Tensor]
    perturbation: ActionGuidedFactorPerturbation
    owners: tuple[TargetOwner, ...]


@dataclass(frozen=True)
class CalibrationOutcome:
    plus: tuple[RewardTrajectory, ...]
    minus: tuple[RewardTrajectory, ...]
    credit: Any
    leaf_gradients: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class CalibrationAnchor:
    structural: torch.Tensor
    functional_response: torch.Tensor
    activation_effect: torch.Tensor
    prior_shared_response: torch.Tensor


def _build_action_proposal(
    runtime: "ECPStage1Runtime", *, task: Any
) -> CalibrationProposal:
    calibration = runtime.config["structured_calibration"]
    packed, encoded = encode_stage1_visit(
        runtime,
        task_ordinal=task.ordinal,
        task_visit=int(calibration["video_visit"]),
    )
    support_task = runtime.support_bank.task(task.ordinal)
    panel = successful_panel_for_visit(
        support_task, int(calibration["support_visit"])
    )
    cached = runtime.support_panels[(task.ordinal, panel.panel_id)]
    evidence = runtime.evidence_bank.evidence(task.ordinal, support_task)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        baseline = runtime.model(encoded, evidence, packed.video_group_ids)
        base_adapter = {
            name: value.detach()
            for name, value in select_compiled_state(
                baseline.consensus_compilation.state, 0
            ).items()
        }
    with torch.autocast("cuda", dtype=torch.bfloat16):
        action_loss, action_gradients, supervision = stage1_action_policy_gradient(
            runtime,
            candidate=base_adapter,
            cached=cached,
            task_ordinal=task.ordinal,
            task_visit=int(calibration["support_visit"]),
        )
    owners = tuple(runtime.model.compiler.owners)
    perturbation = action_guided_factor_perturbation(
        base_adapter,
        action_gradients,
        owners,
        sigma=float(calibration["relative_factor_sigma"]),
        seed=reward_credit_update_seed(
            int(calibration["update_seed_root"]), task.ordinal, 0
        ),
    )
    if supervision != 1.0 or perturbation.active_owners != int(
        calibration["minimum_active_owners"]
    ):
        raise RuntimeError("structured calibration action proposal is incomplete")
    return CalibrationProposal(
        packed=packed,
        encoded=encoded,
        evidence=evidence,
        panel=panel,
        cached=cached,
        action_loss=action_loss,
        action_gradients=action_gradients,
        perturbation=perturbation,
        owners=owners,
    )


def _collect_outcome_credit(
    runtime: "ECPStage1Runtime",
    *,
    pool: RandomResetEnvironmentPool,
    task: Any,
    proposal: CalibrationProposal,
) -> CalibrationOutcome:
    calibration = runtime.config["structured_calibration"]
    reward_task = runtime.calibration.reward_tasks[task.ordinal]
    rollout_cursors = (task.ordinal * 2, task.ordinal * 2 + 1)
    environment_seeds = tuple(
        reward_credit_environment_seed(
            int(calibration["environment_seed_root"]),
            reward_task.suite,
            reward_task.task_id,
            int(runtime.config["optimization"]["seed"]),
            cursor,
        )
        for cursor in rollout_cursors
    )
    environment = runtime.config["environment"]
    envs = tuple(pool.get(reward_task, lane=lane) for lane in range(2))
    initial_states = capture_paired_initial_states(
        envs,
        environment_seeds,
        dummy_action=environment["dummy_action"],
        dummy_settling_steps=int(environment["dummy_settling_steps"]),
    )
    try:
        plus = _arm_rollout(
            runtime,
            pool=pool,
            task=reward_task,
            adapter=proposal.perturbation.plus_state,
            rollout_cursors=rollout_cursors,
            environment_seeds=environment_seeds,
            initial_states=initial_states,
        )
        minus = _arm_rollout(
            runtime,
            pool=pool,
            task=reward_task,
            adapter=proposal.perturbation.minus_state,
            rollout_cursors=rollout_cursors,
            environment_seeds=environment_seeds,
            initial_states=initial_states,
        )
    finally:
        copy_task_lora_state_(runtime.policy, runtime.identity_state, runtime.contract)
    credit = paired_antithetic_credit(
        plus,
        minus,
        proposal.perturbation.epsilon,
        sigma=proposal.perturbation.sigma,
        success_weight=float(calibration["success_weight"]),
        progress_weight=float(calibration["progress_weight"]),
        success_efficiency_weight=float(calibration["success_efficiency_weight"]),
    )
    return CalibrationOutcome(
        plus=plus,
        minus=minus,
        credit=credit,
        leaf_gradients=action_guided_outcome_leaf_gradients(
            proposal.perturbation,
            proposal.owners,
            credit.gradient,
            weight=float(calibration["outcome_leaf_weight"]),
        ),
    )


def _backward_calibration_anchor(
    runtime: "ECPStage1Runtime",
    *,
    proposal: CalibrationProposal,
    outcome: CalibrationOutcome,
    task_count: int,
) -> CalibrationAnchor:
    objective = runtime.config["objective"]
    anchor_weight = float(
        runtime.config["structured_calibration"]["dense_anchor_weight"]
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = runtime.model(
            proposal.encoded, proposal.evidence, proposal.packed.video_group_ids
        )
        candidate = select_compiled_state(output.consensus_compilation.state, 0)
        prior_candidate = select_compiled_state(output.prior_compilation.state, 0)
        support_loss, activation_effect = policy_support_activation_distillation_loss(
            policy=runtime.policy,
            candidate_state=candidate,
            contract=runtime.contract,
            cached=proposal.cached,
            preservation=str(objective["support_preservation"]),
        )
        prior_response = shared_prior_response_distillation_loss(
            policy=runtime.policy,
            candidate_state=prior_candidate,
            contract=runtime.contract,
            cached=proposal.cached,
        )
        functional = ecp_stage1_loss(
            member=output.member_compilation,
            consensus=output.consensus_compilation,
            prior=output.prior_compilation,
            expert_states=proposal.evidence.member_states,
            prior_target=runtime.prior_state,
            contract=runtime.contract,
            policy_support=support_loss,
            weights={
                name: float(value)
                for name, value in objective["weights"].items()
            },
        )
        structural = (
            functional.total
            + float(
                objective["activation_effect_distillation_weight"]
            )
            * activation_effect.loss
            + float(objective["prior_shared_response_weight"])
            * prior_response
        )
    leaf_gradients = {
        name: anchor_weight * proposal.action_gradients[name]
        + outcome.leaf_gradients[name]
        for name in candidate
    }
    active_names = tuple(name for name, value in candidate.items() if value.requires_grad)
    torch.autograd.backward(
        tuple(candidate[name] for name in active_names),
        grad_tensors=tuple(leaf_gradients[name] / task_count for name in active_names),
        retain_graph=True,
    )
    (anchor_weight * structural / task_count).backward()
    finite = structural.detach().float() + sum(
        value.detach().float().square().sum() for value in leaf_gradients.values()
    )
    if not bool(torch.isfinite(finite)):
        raise RuntimeError("non-finite MDCO structured calibration task")
    return CalibrationAnchor(
        structural=structural,
        functional_response=functional.functional_response,
        activation_effect=activation_effect.loss,
        prior_shared_response=prior_response,
    )


def _calibrate_task(
    runtime: "ECPStage1Runtime",
    *,
    pool: RandomResetEnvironmentPool,
    task: Any,
    task_count: int,
) -> dict[str, Any]:
    started = time.monotonic()
    proposal = _build_action_proposal(runtime, task=task)
    outcome = _collect_outcome_credit(
        runtime, pool=pool, task=task, proposal=proposal
    )
    anchor = _backward_calibration_anchor(
        runtime, proposal=proposal, outcome=outcome, task_count=task_count
    )
    credit = outcome.credit
    return {
        "rank": runtime.context.rank,
        "task_ordinal": task.ordinal,
        "asset_key": task.asset_key,
        "suite": task.suite,
        "task_id": task.task_id,
        "demo_indices": list(proposal.packed.demo_indices),
        "support_panel_id": proposal.panel.panel_id,
        "active_owners": proposal.perturbation.active_owners,
        "relative_factor_sigma": proposal.perturbation.sigma,
        "action_policy_loss": float(proposal.action_loss.detach()),
        "action_lora_gradient_norm": _tensor_dict_norm(proposal.action_gradients),
        "outcome_leaf_gradient_norm": _tensor_dict_norm(outcome.leaf_gradients),
        "mean_abs_coordinate_gradient": float(credit.gradient.abs().mean()),
        "mean_advantage": credit.mean_advantage,
        "plus_successes": credit.plus_successes,
        "minus_successes": credit.minus_successes,
        "plus_progress_mean": credit.plus_progress_mean,
        "minus_progress_mean": credit.minus_progress_mean,
        "structural_anchor": float(anchor.structural.detach()),
        "functional_response": float(anchor.functional_response.detach()),
        "activation_effect": float(anchor.activation_effect.detach()),
        "prior_shared_response": float(anchor.prior_shared_response.detach()),
        "task_seconds": time.monotonic() - started,
        "plus": [_trajectory_record(value) for value in outcome.plus],
        "minus": [_trajectory_record(value) for value in outcome.minus],
    }


def _module_gradient_norm(module: torch.nn.Module) -> torch.Tensor:
    values = [
        parameter.grad.float().square().sum()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not values:
        return next(module.parameters()).new_zeros((), dtype=torch.float32)
    return torch.stack(values).sum().sqrt()


def _sync_gradients(runtime: "ECPStage1Runtime") -> dict[str, float]:
    for parameter in runtime.trainable_parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if runtime.context.world_size > 1:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    gradients = {
        "policy_teacher": _module_gradient_norm(runtime.model.policy_teacher),
        "compiler": _module_gradient_norm(runtime.model.compiler),
        "factor_heads": sum(
            parameter.grad.float().square().sum()
            for heads in (runtime.model.compiler.factor_a, runtime.model.compiler.factor_b)
            for parameter in heads.parameters()
        ).sqrt(),
        "visible_program": _module_gradient_norm(runtime.model.visible_program),
    }
    gradients["total"] = torch.nn.utils.clip_grad_norm_(
        runtime.trainable_parameters,
        float(runtime.config["optimization"]["optimizer"]["gradient_clip_norm"]),
    )
    finite = torch.stack([value.float() for value in gradients.values()]).sum()
    if not bool(torch.isfinite(finite)) or float(gradients["visible_program"]) != 0.0:
        raise RuntimeError("invalid MDCO structured calibration gradient")
    return {name: float(value) for name, value in gradients.items()}


def _gather_records(
    runtime: "ECPStage1Runtime", local: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if runtime.context.world_size == 1:
        return [dict(value) for value in local]
    gathered: list[Any] | None = (
        [None] * runtime.context.world_size if runtime.context.is_main else None
    )
    dist.gather_object(list(local), gathered, dst=0)
    if not runtime.context.is_main:
        return []
    return [dict(row) for group in gathered or () for row in group]


def run_structured_calibration(
    runtime: "ECPStage1Runtime", *, task_visits: int, run_started: float
) -> dict[str, Any]:
    settings = runtime.config["structured_calibration"]
    task_count = calibration_task_count(runtime.config, mode=runtime.args.mode)
    if runtime.args.mode == "formal" and task_visits != int(
        settings["after_task_visits"]
    ):
        raise ValueError("formal MDCO structured calibration cursor changed")
    artifact = runtime.args.output_dir / STRUCTURED_CALIBRATION_FILE
    if artifact.exists():
        raise ValueError("MDCO structured calibration artifact already exists")
    started = time.monotonic()
    pool = _environment_pool(runtime)
    runtime.optimizer.zero_grad(set_to_none=True)
    try:
        local = []
        for ordinal in runtime.calibration.assignments[runtime.context.rank]:
            try:
                local.append(
                    _calibrate_task(
                        runtime,
                        pool=pool,
                        task=runtime.task_by_ordinal[ordinal],
                        task_count=task_count,
                    )
                )
            finally:
                # Every fit mapping is visited once, so retaining its two EGL
                # environments only accumulates memory without enabling reuse.
                pool.close()
    finally:
        copy_task_lora_state_(runtime.policy, runtime.identity_state, runtime.contract)
        pool.close()
    gradients = _sync_gradients(runtime)
    runtime.optimizer.step()
    runtime.scheduler.step()
    records = _gather_records(runtime, local)
    if runtime.context.is_main:
        records.sort(key=lambda row: int(row["task_ordinal"]))
        if [int(row["task_ordinal"]) for row in records] != list(range(task_count)):
            raise ValueError("MDCO structured calibration lost task-equal coverage")
        result = {
            "schema_version": STRUCTURED_CALIBRATION_SCHEMA,
            "status": (
                "complete_fit90_structured_calibration"
                if runtime.args.mode == "formal"
                else "complete_profile_structured_calibration"
            ),
            "mode": runtime.args.mode,
            "applied_after_task_visits": task_visits,
            "task_count": task_count,
            "task_weight": "equal",
            "assignments": [list(values) for values in runtime.calibration.assignments],
            "structured_surface": str(settings["surface"]),
            "global_16d_estimator": False,
            "optimizer_updates": 1,
            "plus_successes": sum(int(row["plus_successes"]) for row in records),
            "minus_successes": sum(int(row["minus_successes"]) for row in records),
            "nonzero_advantage_tasks": sum(
                abs(float(row["mean_advantage"])) > 0 for row in records
            ),
            "mean_advantage": sum(float(row["mean_advantage"]) for row in records)
            / len(records),
            "mean_plus_progress": sum(
                float(row["plus_progress_mean"]) for row in records
            )
            / len(records),
            "mean_minus_progress": sum(
                float(row["minus_progress_mean"]) for row in records
            )
            / len(records),
            "gradient_norms_before_clip": gradients,
            "next_lr": float(runtime.optimizer.param_groups[0]["lr"]),
            "calibration_seconds": time.monotonic() - started,
            "elapsed_seconds": time.monotonic() - run_started,
            "max_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(runtime.context.device)
            ),
            "information_wall": {
                "fit_reward_tasks": task_count,
                "held5_reward_reads": 0,
                "validation_reward_reads": 0,
                "test_reward_reads": 0,
                "deployment_reward_reads": 0,
            },
            "tasks": records,
        }
        write_json_atomic(artifact, result)
    else:
        result = {"mode": runtime.args.mode, "task_count": task_count}
    barrier(runtime.context)
    return result
