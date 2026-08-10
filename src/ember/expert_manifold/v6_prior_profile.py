"""Discarded full24 Reward-Credit profile gates and mechanism probes."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.distributed as dist

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_runtime import V6PriorRuntime
from ember.lora import copy_task_lora_state_
from ember.reward.rollout import RewardTrajectory


@dataclass(frozen=True)
class FixedActionProfilePanel:
    query: Mapping[str, torch.Tensor]
    noise_seeds: tuple[int, ...]


def profile_action_panel(
    trajectories: Sequence[RewardTrajectory],
) -> FixedActionProfilePanel | None:
    """Retain the exact K4 first-replan panel only when reward credit is mixed."""

    if len(trajectories) != 4:
        raise ExpertManifoldError("Reward-Credit profile action panel is not K4")
    if {bool(value.success) for value in trajectories} != {False, True}:
        return None
    if any(
        not value.observations or not value.policy_noise_seeds for value in trajectories
    ):
        raise ExpertManifoldError("Reward-Credit profile action panel is incomplete")
    observations = [value.observations[0] for value in trajectories]
    keys = set(observations[0])
    if any(set(value) != keys for value in observations):
        raise ExpertManifoldError("Reward-Credit profile action query keys changed")
    query = {
        name: torch.cat([value[name] for value in observations], dim=0)
        for name in sorted(keys)
    }
    if len(query) != 4 or any(
        value.ndim == 0 or value.shape[0] != 4 for value in query.values()
    ):
        raise ExpertManifoldError("Reward-Credit profile action query is not K4")
    seeds = tuple(int(value.policy_noise_seeds[0]) for value in trajectories)
    if len(set(seeds)) != 4:
        raise ExpertManifoldError("Reward-Credit profile action noise is not K4")
    return FixedActionProfilePanel(query, seeds)


@contextmanager
def _policy_attention_state(policy: torch.nn.Module) -> Iterator[None]:
    bridge = policy.model.paligemma_with_expert
    language = bridge.paligemma.model.language_model.config
    expert = bridge.gemma_expert.model.config
    before = (language._attn_implementation, expert._attn_implementation)
    try:
        yield
    finally:
        language._attn_implementation, expert._attn_implementation = before


@torch.inference_mode()
def _fixed_action(
    runtime: V6PriorRuntime,
    query: Mapping[str, torch.Tensor],
    state: Mapping[str, torch.Tensor],
    *,
    seeds: Sequence[int],
) -> torch.Tensor:
    copy_task_lora_state_(runtime.policy, state, runtime.lora_contract)
    batch_sizes = {
        int(value.shape[0])
        for value in query.values()
        if isinstance(value, torch.Tensor) and value.ndim > 0
    }
    if (
        batch_sizes != {4}
        or len(seeds) != 4
        or len(set(int(value) for value in seeds)) != 4
    ):
        raise ExpertManifoldError("fixed-action Reward-Credit probe is not K4")
    noises = []
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noises.append(
            torch.randn(
                1,
                int(runtime.policy.config.chunk_size),
                int(runtime.policy.config.max_action_dim),
                generator=generator,
                dtype=torch.float32,
            )
        )
    noise = torch.cat(noises, dim=0).to(runtime.context.device)
    moved = {
        name: value.to(runtime.context.device, non_blocking=True)
        for name, value in query.items()
    }
    with (
        _policy_attention_state(runtime.policy),
        torch.autocast(
            device_type=runtime.context.device.type,
            dtype=torch.bfloat16,
            enabled=runtime.context.device.type == "cuda",
        ),
    ):
        action = runtime.policy.predict_action_chunk(moved, noise=noise, num_steps=10)
    if action.ndim != 3 or action.shape[0] != 4:
        raise ExpertManifoldError("fixed-action Reward-Credit probe changed")
    return action.detach()


def _decode_profile_lora(
    runtime: V6PriorRuntime, objective: Any, motion: torch.Tensor
) -> Mapping[str, torch.Tensor]:
    if objective.program_before is None or objective.correct_lora_before is None:
        raise ExpertManifoldError("Reward-Credit profile lost before-state")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        after_program = (
            objective.program_before + motion.to(dtype=objective.program_before.dtype)
        ).to(dtype=torch.float32)
        return runtime.writer.base_writer.decode_slots(after_program[None])


def _record_lora_evidence(
    row: torch.Tensor,
    before_state: Mapping[str, torch.Tensor],
    after_state: Mapping[str, torch.Tensor],
) -> None:
    for name, before in before_state.items():
        difference = after_state[name] - before
        columns = (2, 3) if name.endswith(".lora_A.default.weight") else (4, 5)
        row[columns[0]].add_(difference.float().square().sum())
        row[columns[1]].add_(difference.numel())


def _record_action_evidence(
    runtime: V6PriorRuntime,
    row: torch.Tensor,
    objective: Any,
    after_state: Mapping[str, torch.Tensor],
) -> None:
    panel = objective.fixed_policy_panel
    if (panel is not None) is not bool(objective.credit.mixed):
        raise ExpertManifoldError(
            "Reward-Credit mixed task action probe coverage changed"
        )
    if panel is None:
        return
    row[1] = 1
    before = _fixed_action(
        runtime, panel.query, objective.correct_lora_before, seeds=panel.noise_seeds
    )
    after = _fixed_action(runtime, panel.query, after_state, seeds=panel.noise_seeds)
    difference = after - before
    row[6].add_(difference.float().square().sum())
    row[7] = difference.numel()
    row[8] = 4
    row[9] = 2


def _gather_response_evidence(
    runtime: V6PriorRuntime, local_evidence: torch.Tensor
) -> torch.Tensor:
    gathered = local_evidence
    if runtime.context.world_size > 1:
        gathered = torch.empty(
            runtime.context.world_size * local_evidence.shape[0],
            local_evidence.shape[1],
            dtype=local_evidence.dtype,
            device=local_evidence.device,
        )
        dist.all_gather_into_tensor(gathered, local_evidence.contiguous())
    order = gathered[:, 0].to(dtype=torch.long).argsort()
    gathered = gathered.index_select(0, order)
    if not torch.equal(
        gathered[:, 0].to(dtype=torch.long),
        torch.arange(24, dtype=torch.long, device=gathered.device),
    ):
        raise ExpertManifoldError("Reward-Credit action probe task order changed")
    return gathered


def _probe_response_rows(
    evidence_rows: Sequence[Sequence[float]],
) -> list[dict[str, Any]]:
    probe_rows = []
    for values in evidence_rows:
        if values[1] == 0:
            continue
        if values[1] != 1:
            raise ExpertManifoldError("Reward-Credit action probe flag changed")
        probe_rows.append(
            {
                "task_ordinal": int(values[0]),
                "query_count": int(values[8]),
                "policy_forwards": int(values[9]),
                "lora_a_value_count": int(values[3]),
                "lora_a_response_rms": math.sqrt(values[2] / max(values[3], 1.0)),
                "lora_b_value_count": int(values[5]),
                "lora_b_response_rms": math.sqrt(values[4] / max(values[5], 1.0)),
                "fixed_action_value_count": int(values[7]),
                "fixed_action_response_rms": math.sqrt(values[6] / max(values[7], 1.0)),
            }
        )
    return probe_rows


def _summarize_response_evidence(gathered: torch.Tensor) -> dict[str, Any]:
    lora_values = gathered[:, 2:6].sum(dim=0).detach().cpu().tolist()
    probe_rows = _probe_response_rows(gathered.detach().cpu().tolist())
    action_square = math.fsum(
        float(row["fixed_action_response_rms"]) ** 2
        * int(row["fixed_action_value_count"])
        for row in probe_rows
    )
    action_count = sum(int(row["fixed_action_value_count"]) for row in probe_rows)
    return {
        "lora_a_response_rms": math.sqrt(lora_values[0] / max(lora_values[1], 1.0)),
        "lora_b_response_rms": math.sqrt(lora_values[2] / max(lora_values[3], 1.0)),
        "fixed_action_response_rms": math.sqrt(action_square / max(action_count, 1)),
        "fixed_action_probe_task_count": len(probe_rows),
        "fixed_action_probe_query_count": sum(
            int(row["query_count"]) for row in probe_rows
        ),
        "fixed_action_probe_policy_forwards": sum(
            int(row["policy_forwards"]) for row in probe_rows
        ),
        "fixed_action_passing_task_count": sum(
            float(row["fixed_action_response_rms"]) > 0 for row in probe_rows
        ),
        "fixed_action_task_rows": probe_rows,
    }


def profile_lora_response(
    runtime: V6PriorRuntime,
    local: Sequence[Any],
    correct_motion: torch.Tensor,
) -> dict[str, Any]:
    if len(local) != 24 // runtime.context.world_size:
        raise ExpertManifoldError("Reward-Credit profile local coverage changed")
    # ordinal, probe, A square/count, B square/count, action square/count,
    # K4 query count, policy invocation count.
    evidence = torch.zeros(
        len(local), 10, dtype=torch.float64, device=runtime.context.device
    )
    try:
        for row, objective in zip(evidence, local, strict=True):
            row[0] = objective.task.ordinal
            after = _decode_profile_lora(
                runtime, objective, correct_motion[objective.task.ordinal]
            )
            _record_lora_evidence(row, objective.correct_lora_before, after)
            _record_action_evidence(runtime, row, objective, after)
    finally:
        copy_task_lora_state_(
            runtime.policy, runtime.identity_state, runtime.lora_contract
        )
    return _summarize_response_evidence(_gather_response_evidence(runtime, evidence))


@torch.no_grad()
def profile_credit_motion(
    cotangents: torch.Tensor,
    correct_motion: torch.Tensor,
) -> dict[str, float | int]:
    """Measure shared-solve motion injected into exact-zero-credit task rows."""

    if cotangents.shape != correct_motion.shape or cotangents.ndim != 3:
        raise ExpertManifoldError("Reward-Credit profile motion topology changed")
    active = cotangents.flatten(1).square().sum(dim=1) > 0
    mixed_count = int(active.sum())
    homogeneous_count = int((~active).sum())
    if mixed_count == 0:
        mixed_rms = torch.zeros((), device=correct_motion.device)
    else:
        mixed_rms = correct_motion[active].square().mean().sqrt()
    if homogeneous_count == 0:
        homogeneous_rms = torch.zeros((), device=correct_motion.device)
        moving = 0
    else:
        homogeneous_rows = correct_motion[~active].flatten(1).square().mean(1).sqrt()
        homogeneous_rms = homogeneous_rows.square().mean().sqrt()
        moving = int((homogeneous_rows > 0).sum())
    values = torch.stack((mixed_rms, homogeneous_rms)).cpu().tolist()
    return {
        "mixed_correct_motion_rms": float(values[0]),
        "homogeneous_correct_motion_rms": float(values[1]),
        "homogeneous_to_mixed_motion_ratio": float(
            values[1] / values[0] if values[0] > 0 else math.inf
        ),
        "homogeneous_task_count": homogeneous_count,
        "homogeneous_moving_task_count": moving,
    }


def base_versions(runtime: V6PriorRuntime) -> tuple[tuple[str, int], ...]:
    base = runtime.writer.base_writer
    return tuple(
        (name, value._version)
        for name, value in (*base.named_parameters(), *base.named_buffers())
    )


def _profile_sections(
    row: Mapping[str, Any],
) -> tuple[
    Sequence[Mapping[str, Any]],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    records = row.get("task_records", ())
    update = row.get("update", {})
    application = row.get("application", {})
    response = row.get("lora_response", {})
    if (
        not isinstance(records, Sequence)
        or len(records) != 24
        or not all(isinstance(value, Mapping) for value in records)
        or not isinstance(update, Mapping)
        or not isinstance(application, Mapping)
        or not isinstance(response, Mapping)
    ):
        raise ExpertManifoldError("Reward-Credit profile evidence is incomplete")
    return records, update, application, response


def _workload_checks(
    row: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
    *,
    mixed_task_count: int,
    homogeneous_task_count: int,
) -> dict[str, bool]:
    return {
        "full24_tasks": int(row.get("tasks", -1)) == int(gates["tasks"]),
        "k4_rollouts": int(row.get("rollouts", -1)) == int(gates["rollouts"])
        and all(
            int(value.get("rollouts", -1)) == int(gates["rollouts_per_task"])
            for value in records
        ),
        "one_video_per_task": int(row.get("videos", -1)) == int(gates["videos"])
        and all(
            int(value.get("videos", -1)) == int(gates["videos_per_task"])
            for value in records
        ),
        "mixed_task_coverage": mixed_task_count >= int(gates["mixed_tasks_min"]),
        "homogeneous_task_coverage": homogeneous_task_count
        >= int(gates["homogeneous_tasks_min"]),
    }


def _credit_checks(
    mixed: Sequence[Mapping[str, Any]],
    homogeneous: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    return {
        "mixed_cotangents": all(
            math.isfinite(float(value.get("program_cotangent_rms", math.nan)))
            and float(value["program_cotangent_rms"]) > 0
            and int(value.get("functional_policy_forwards", 0)) > 0
            for value in mixed
        ),
        "homogeneous_exact_zero": all(
            float(value.get("program_cotangent_rms", math.nan)) == 0.0
            and int(value.get("functional_policy_forwards", -1)) == 0
            for value in homogeneous
        ),
    }


def _finite_positive(row: Mapping[str, Any], name: str) -> bool:
    value = float(row.get(name, math.nan))
    return math.isfinite(value) and value > 0


def _valid_action_response_row(
    row: Mapping[str, Any], *, queries: int, forwards: int
) -> bool:
    return (
        int(row.get("query_count", -1)) == queries
        and int(row.get("policy_forwards", -1)) == forwards
        and all(
            int(row.get(name, 0)) > 0
            for name in (
                "lora_a_value_count",
                "lora_b_value_count",
                "fixed_action_value_count",
            )
        )
        and all(
            _finite_positive(row, name)
            for name in (
                "lora_a_response_rms",
                "lora_b_response_rms",
                "fixed_action_response_rms",
            )
        )
    )


def _all_mixed_action_rows_pass(
    records: Sequence[Mapping[str, Any]],
    response: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> bool:
    mixed_records = [value for value in records if value.get("mixed") is True]
    expected_ordinals = sorted(int(value["task_ordinal"]) for value in mixed_records)
    action_rows = response.get("fixed_action_task_rows", ())
    if not isinstance(action_rows, Sequence) or isinstance(action_rows, (str, bytes)):
        return False
    if not all(isinstance(value, Mapping) for value in action_rows):
        return False
    observed_ordinals = [int(value.get("task_ordinal", -1)) for value in action_rows]
    required_queries = int(gates["fixed_action_queries_per_mixed_task"])
    required_forwards = int(gates["fixed_action_policy_forwards_per_mixed_task"])
    coverage = (
        gates.get("mixed_action_probe_scope") == "all_mixed_tasks"
        and len({str(value.get("suite")) for value in mixed_records})
        == int(gates["mixed_suite_count"])
        and observed_ordinals == expected_ordinals
        and len(set(observed_ordinals)) == len(expected_ordinals)
    )
    summaries = (
        int(response.get("fixed_action_probe_task_count", -1)) == len(expected_ordinals)
        and int(response.get("fixed_action_probe_query_count", -1))
        == required_queries * len(expected_ordinals)
        and int(response.get("fixed_action_probe_policy_forwards", -1))
        == required_forwards * len(expected_ordinals)
        and int(response.get("fixed_action_passing_task_count", -1))
        == len(expected_ordinals)
    )
    rows_pass = all(
        _valid_action_response_row(
            value, queries=required_queries, forwards=required_forwards
        )
        for value in action_rows
    )
    return coverage and summaries and rows_pass


def _mechanism_checks(
    records: Sequence[Mapping[str, Any]],
    update: Mapping[str, Any],
    application: Mapping[str, Any],
    response: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "full48_feature_rank": int(update.get("feature_rank", -1))
        >= int(gates["full48_feature_rank_min"]),
        "negative_null": math.isfinite(
            float(update.get("predicted_negative_to_correct_ratio", math.nan))
        )
        and float(update["predicted_negative_to_correct_ratio"])
        <= float(gates["negative_null_motion_ratio_max"]),
        "predicted_observed_closure": math.isfinite(
            float(application.get("predicted_observed_relative_rms", math.nan))
        )
        and float(application["predicted_observed_relative_rms"])
        <= float(gates["predicted_observed_relative_rms_max"]),
        "program_to_lora": math.isfinite(
            float(response.get("lora_a_response_rms", math.nan))
        )
        and math.isfinite(float(response.get("lora_b_response_rms", math.nan)))
        and float(response["lora_a_response_rms"]) > 0
        and float(response["lora_b_response_rms"]) > 0,
        "program_to_all_mixed_actions": math.isfinite(
            float(response.get("fixed_action_response_rms", math.nan))
        )
        and float(response["fixed_action_response_rms"]) > 0
        and _all_mixed_action_rows_pass(records, response, gates),
    }


def _runtime_checks(
    row: Mapping[str, Any], gates: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        "no_retired_forwards": int(row.get("negative_policy_forwards", -1))
        == int(gates["extra_negative_policy_forwards"])
        and int(row.get("old_policy_forwards", -1))
        == int(gates["old_policy_forwards"]),
        "runtime_health": int(row.get("oom_count", -1)) == int(gates["oom_count"])
        and int(row.get("nonfinite_count", -1)) == int(gates["nonfinite_count"])
        and int(row.get("watchdog_count", -1)) == int(gates["watchdog_count"]),
    }


def profile_passes(
    config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[bool, dict[str, Any]]:
    if len(rows) != 1:
        raise ExpertManifoldError("Reward-Credit profile must contain one full cycle")
    row = rows[0]
    gates = config["profile_run"]["gates"]
    records, update, application, response = _profile_sections(row)
    if any(type(value.get("mixed")) is not bool for value in records):
        raise ExpertManifoldError(
            "Reward-Credit profile mixed/homogeneous partition changed"
        )
    mixed = [value for value in records if value.get("mixed") is True]
    homogeneous = [value for value in records if value.get("mixed") is False]
    if len(mixed) + len(homogeneous) != len(records):
        raise ExpertManifoldError(
            "Reward-Credit profile mixed/homogeneous partition changed"
        )
    checks = _workload_checks(
        row,
        records,
        gates,
        mixed_task_count=len(mixed),
        homogeneous_task_count=len(homogeneous),
    )
    checks.update(_credit_checks(mixed, homogeneous))
    checks.update(_mechanism_checks(records, update, application, response, gates))
    checks.update(_runtime_checks(row, gates))
    return all(checks.values()), {
        "checks": checks,
        "mixed_tasks": len(mixed),
        "homogeneous_tasks": len(homogeneous),
        "successes": int(row.get("successes", -1)),
        "step_seconds": float(row.get("step_seconds", math.nan)),
        "max_cuda_allocated_bytes": int(row.get("max_cuda_allocated_bytes", -1)),
        "max_cuda_reserved_bytes": int(row.get("max_cuda_reserved_bytes", -1)),
    }


def _raw_replay_recipe_matches(
    records: Sequence[Mapping[str, Any]],
    *,
    task_count: int,
    physical_batch: int,
    mc_samples: int,
) -> bool:
    mixed = [record for record in records if record.get("mixed") is True]
    homogeneous = [record for record in records if record.get("mixed") is False]
    if physical_batch <= 0 or mc_samples <= 0 or not mixed:
        return False
    candidates = [
        candidate
        for candidate in range(
            1, max(int(record["replay_chunks"]) for record in mixed) + 1
        )
        if all(
            int(record["functional_policy_forwards"])
            == mc_samples
            * ((int(record["replay_chunks"]) + candidate - 1) // candidate)
            for record in mixed
        )
    ]
    return all(
        (
            len(records) == task_count,
            sorted(int(record.get("task_ordinal", -1)) for record in records)
            == list(range(task_count)),
            len(mixed) + len(homogeneous) == task_count,
            all(int(record.get("mc_samples", -1)) == mc_samples for record in records),
            all(int(record.get("replay_chunks", 0)) > 0 for record in records),
            all(
                int(record.get("functional_policy_forwards", -1)) == 0
                for record in homogeneous
            ),
            candidates == [physical_batch],
        )
    )


def profile_runtime_matches_config(
    run: Mapping[str, Any],
    result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> bool:
    """Bind retained raw profile counts to the sealed physical runtime recipe."""
    try:
        rows = result["macros"]
        if not isinstance(rows, list) or len(rows) != 1:
            return False
        records = rows[0]["task_records"]
        optimization = config["optimization"]
        objective = config["objective"]
        profile = config["profile_run"]
        data = config["data"]
        distributed = optimization["distributed_update"]
        runtime = run["runtime"]
        run_data = run["data"]
        mappings = (
            optimization,
            objective,
            profile,
            data,
            distributed,
            runtime,
            run_data,
        )
        if not isinstance(records, list) or not all(
            isinstance(value, Mapping) for value in mappings
        ):
            return False
        if not all(isinstance(record, Mapping) for record in records):
            return False
        topology = runtime["rank_topology"]
        if not isinstance(topology, list) or not all(
            isinstance(row, Mapping) for row in topology
        ):
            return False
        physical_batch = int(optimization["reward_replay_chunk_batch_size"])
        mc_samples = int(objective["flow_mc_samples"])
        task_count = int(data["task_count"])
    except (KeyError, TypeError, ValueError):
        return False
    runtime_checks = (
        run.get("mode") == "mechanism-profile",
        run.get("optimization") == optimization,
        run.get("objective") == objective,
        all(run_data.get(name) == value for name, value in data.items()),
        int(runtime.get("world_size", -1)) == int(profile["expected_world_size"]),
        int(runtime.get("tasks_per_rank", -1)) == int(profile["tasks_per_rank"]),
        int(runtime.get("num_workers_per_rank", -1))
        == int(profile["num_workers_per_rank"]),
        int(runtime.get("total_macros", -1)) == int(profile["diagnostic_macros"]),
        int(runtime.get("schedule_origin", -1)) == int(profile["schedule_macro"]),
        runtime.get("checkpoint_macros") == [],
        int(runtime.get("rollout_policy_batch_size", -1))
        == int(optimization["rollout_policy_batch_size"]),
        int(runtime.get("reward_replay_chunk_batch_size", -1)) == physical_batch,
        int(runtime.get("flow_mc_samples", -1)) == mc_samples,
        int(runtime.get("old_policy_forwards", -1))
        == int(objective["old_policy_forwards"]),
        int(runtime.get("negative_policy_forwards", -1))
        == int(objective["negative_policy_forwards"]),
        runtime.get("deferred_process_group")
        is bool(distributed["deferred_process_group"]),
        runtime.get("nccl_p2p_disable") == str(distributed["nccl_p2p_disable"]),
        runtime.get("nccl_algo") == distributed["nccl_algo"],
        runtime.get("nccl_proto") == distributed["nccl_proto"],
        runtime.get("device") == "NVIDIA A40",
        len(topology) == int(profile["expected_world_size"]),
        all(row.get("device_name") == "NVIDIA A40" for row in topology),
    )
    return all(runtime_checks) and _raw_replay_recipe_matches(
        records,
        task_count=task_count,
        physical_batch=physical_batch,
        mc_samples=mc_samples,
    )


def profile_seal_payload_matches(
    *,
    config: Mapping[str, Any],
    result: object,
    run: object,
    completion: object,
    invocations: object,
    profile_schema: str,
    completion_schema: str,
    profile_gates: Mapping[str, Any],
) -> bool:
    """Recompute all retained payload claims without trusting self-reported status."""
    if not all(isinstance(value, Mapping) for value in (result, run, completion)):
        return False
    if not isinstance(invocations, list) or len(invocations) != 1:
        return False
    invocation = invocations[0]
    if not isinstance(invocation, Mapping):
        return False
    try:
        passed, evidence = profile_passes(config, result["macros"])
    except (ExpertManifoldError, KeyError, TypeError, ValueError):
        return False
    expected_completion = {
        "schema_version": completion_schema,
        "mode": "mechanism-profile",
        "completed_diagnostic_macros": 1,
        "passed": True,
        "retained_checkpoint": False,
        "content_hash_policy": "disabled_by_owner",
    }
    return all(
        (
            result.get("schema_version") == profile_schema,
            result.get("passed") is True,
            result.get("schedule_macro") == 0,
            result.get("retain_weight") is False,
            result.get("gates") == profile_gates,
            result.get("gate_evidence") == evidence,
            passed is True,
            profile_runtime_matches_config(run, result, config),
            completion == expected_completion,
            invocation.get("resume") is None,
            invocation.get("requested_stop_after_macro") == 1,
        )
    )
