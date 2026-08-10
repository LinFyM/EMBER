"""Rank-reserved five-arm vertical mechanism and action closure."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ember.expert_manifold.inference import load_expert_manifold_deployment_config
from ember.expert_manifold.v6_prior_contract import REPO_ROOT
from ember.pi05_eval_contract import git_state_is_clean_pushed_or_frozen_authority
from ember.pi05_source_checkpoint import write_json_atomic
from ember.writer.evaluation_cache import (
    lora_state_storage,
    stage_writer_lora_states_to_cpu,
    validate_writer_cache_manifest,
    writer_cache_manifest_path,
)
from ember.writer.errors import WriterModelError


RANK_RESERVED_VERTICAL_SCHEMA = "ember_pi05_v6_qv_rank_reserved_native_vertical_v1"
RANK_RESERVED_VERTICAL_PREFLIGHT = "rank_reserved_vertical_preflight.json"


def _factor_product_inner(
    left_b: torch.Tensor,
    left_a: torch.Tensor,
    right_b: torch.Tensor,
    right_a: torch.Tensor,
) -> torch.Tensor:
    return (
        torch.matmul(left_b.float().transpose(-1, -2), right_b.float())
        * torch.matmul(left_a.float(), right_a.float().transpose(-1, -2))
    ).sum()


def _effective_delta_moments(
    before_left: Mapping[str, torch.Tensor],
    after_left: Mapping[str, torch.Tensor],
    before_right: Mapping[str, torch.Tensor],
    after_right: Mapping[str, torch.Tensor],
) -> tuple[float, float, float, int, int, int]:
    dot = torch.zeros(
        (), dtype=torch.float32, device=next(iter(before_left.values())).device
    )
    left_square = torch.zeros_like(dot)
    right_square = torch.zeros_like(dot)
    left_nonzero = 0
    right_nonzero = 0
    a_names = sorted(
        name
        for name in before_left
        if ".self_attn." in name and name.endswith(".lora_A.default.weight")
    )
    if len(a_names) != 36:
        raise WriterModelError("rank-reserved vertical q/v topology changed")
    for a_name in a_names:
        b_name = a_name.replace(".lora_A.default.weight", ".lora_B.default.weight")

        def delta_factors(
            before: Mapping[str, torch.Tensor],
            after: Mapping[str, torch.Tensor],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            before_a = before[a_name].float()
            before_b = before[b_name].float()
            after_a = after[a_name].float()
            after_b = after[b_name].float()
            return (
                torch.cat((after_b - before_b, before_b), dim=-1),
                torch.cat((after_a, after_a - before_a), dim=-2),
            )

        left_b, left_a = delta_factors(before_left, after_left)
        right_b, right_a = delta_factors(before_right, after_right)
        target_dot = _factor_product_inner(left_b, left_a, right_b, right_a)
        target_left_square = _factor_product_inner(
            left_b, left_a, left_b, left_a
        ).clamp_min(0)
        target_right_square = _factor_product_inner(
            right_b, right_a, right_b, right_a
        ).clamp_min(0)
        dot = dot + target_dot
        left_square = left_square + target_left_square
        right_square = right_square + target_right_square
        left_nonzero += int(float(target_left_square.detach().cpu()) > 0.0)
        right_nonzero += int(float(target_right_square.detach().cpu()) > 0.0)
    values = torch.stack((dot, left_square.clamp_min(0), right_square.clamp_min(0)))
    moments = tuple(float(value) for value in values.detach().cpu())
    return (*moments, left_nonzero, right_nonzero, len(a_names))


def _vector_delta_moments(
    old_before: torch.Tensor,
    old_after: torch.Tensor,
    new_before: torch.Tensor,
    new_after: torch.Tensor,
) -> tuple[float, float, float]:
    old = old_after.float() - old_before.float()
    new = new_after.float() - new_before.float()
    values = torch.stack(
        (
            (old * new).sum(),
            old.square().sum(),
            new.square().sum(),
        )
    )
    return tuple(float(value) for value in values.detach().cpu())


def _moment_summary(
    values: Sequence[tuple[float, ...]],
) -> dict[str, float | int]:
    dot = sum(value[0] for value in values)
    left_square = max(sum(value[1] for value in values), 0.0)
    right_square = max(sum(value[2] for value in values), 0.0)
    denominator = math.sqrt(left_square * right_square)
    return {
        "dot": dot,
        "old_delta_l2_rms_across_panel": math.sqrt(left_square / max(len(values), 1)),
        "new_delta_l2_rms_across_panel": math.sqrt(right_square / max(len(values), 1)),
        "cosine": dot / denominator if denominator > 0 else 0.0,
        **(
            {
                "old_nonzero_targets": sum(int(value[3]) for value in values),
                "new_nonzero_targets": sum(int(value[4]) for value in values),
                "target_comparisons": sum(int(value[5]) for value in values),
            }
            if values and len(values[0]) >= 6
            else {}
        ),
    }


def is_rank_reserved_vertical_contract(
    contract: Mapping[str, Any], output_dir: Path
) -> bool:
    """Identify the single pre-registered live vertical smoke."""

    try:
        adapter = contract["adapter"]
        asset = adapter["writer_asset"]
        config = load_expert_manifold_deployment_config(Path(adapter["config"]["path"]))
        registered = (
            REPO_ROOT / config["evaluation"]["registered_roots"]["vertical"]
        ).resolve()
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return (
        output_dir.resolve() == registered
        and contract.get("mode") == "smoke"
        and contract.get("role") == "validation"
        and adapter.get("schema_version")
        == "ember_pi05_v6_qv_rank_reserved_native_reward_eval_adapter_v9"
        and adapter.get("video_condition") == "correct"
        and asset.get("kind") == "v6_qv_rank14_plus2_reward_program_load_only"
        and int(asset.get("method_macro", -1)) == 1
        and asset.get("enable_program_residual") is True
        and int(contract["parallel"]["physical_gpu_count"]) == 1
        and int(contract["parallel"]["replicas_per_gpu"]) == 1
        and int(contract["parallel"]["writer_generators_per_gpu"]) == 1
        and all(
            tuple(task.get("init_state_ids", ())) == (0,) for task in contract["tasks"]
        )
        and git_state_is_clean_pushed_or_frozen_authority(contract.get("git", {}))
    )


def _rank_reserved_vertical_tasks(runtime: Any) -> tuple[dict[str, Any], ...]:
    suite_order = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    selected = []
    for suite in suite_order:
        matches = sorted(
            (task for task in runtime.tasks.values() if task["suite"] == suite),
            key=lambda task: int(task["task_id"]),
        )
        if not matches or tuple(matches[0]["init_state_ids"]) != (0,):
            raise WriterModelError("rank-reserved vertical panel changed")
        selected.append(matches[0])
    return tuple(selected)


def _rank_reserved_zero_slots(
    states: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, int | bool]:
    checked = nonzero = 0
    for state in states:
        a_names = sorted(
            name
            for name in state
            if ".self_attn." in name and name.endswith(".lora_A.default.weight")
        )
        if len(a_names) != 36:
            raise WriterModelError("rank-reserved zero-slot topology changed")
        for a_name in a_names:
            b_name = a_name.replace(".lora_A.default.weight", ".lora_B.default.weight")
            values = (state[a_name][-2:], state[b_name][:, -2:])
            checked += sum(value.numel() for value in values)
            nonzero += sum(int(torch.count_nonzero(value)) for value in values)
    return {
        "values_checked": checked,
        "nonzero_values": nonzero,
        "exact_zero": checked > 0 and nonzero == 0,
    }


def _rank_reserved_qv_only_state(
    base: Mapping[str, torch.Tensor],
    reward: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Use cached reward q/v tensors while retaining rank14-base action tensors."""

    if set(base) != set(reward):
        raise WriterModelError("rank-reserved q/v-only state topology changed")
    result = {
        name: (reward[name] if ".self_attn." in name else value)
        for name, value in base.items()
    }
    qv_tensors = sum(".self_attn." in name for name in result)
    action_tensors = len(result) - qv_tensors
    if (qv_tensors, action_tensors) != (72, 4):
        raise WriterModelError("rank-reserved q/v-only tensor ownership changed")
    return result


def _rank_reserved_cached_base_state(
    action_base: Mapping[str, torch.Tensor],
    cached_reward: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove cached q/v residual slots without crossing Writer batch shapes."""

    if set(action_base) != set(cached_reward):
        raise WriterModelError("rank-reserved cached-base topology changed")
    result: dict[str, torch.Tensor] = {}
    qv_tensors = action_tensors = 0
    for name, action_value in action_base.items():
        if ".self_attn." not in name:
            result[name] = action_value
            action_tensors += 1
            continue
        value = cached_reward[name].clone()
        if name.endswith(".lora_A.default.weight"):
            value[-2:] = 0
        elif name.endswith(".lora_B.default.weight"):
            value[:, -2:] = 0
        else:
            raise WriterModelError("rank-reserved cached-base tensor name changed")
        result[name] = value
        qv_tensors += 1
    if (qv_tensors, action_tensors) != (72, 4):
        raise WriterModelError("rank-reserved cached-base tensor ownership changed")
    return result


def _rank_reserved_video_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    keys = (
        "condition",
        "language_global_task_id",
        "teacher_video_kind",
        "teacher_video_frames_used",
        "teacher_video_count",
        "teacher_video_seed_root",
        "teacher_video_selection_seed",
        "teacher_video_sampling_mode",
        "video_suite",
        "video_task_id",
        "video_global_task_id",
        "video_split_role",
        "teacher_demo_indices",
        "teacher_reference_demo_indices",
        "task_video_mapping_reference",
        "pairing_reference",
        "teacher_video_order_seeds",
    )
    return tuple(row.get(key) for key in keys)


@torch.inference_mode()
def prepare_rank_reserved_vertical(
    runtime: Any,
    *,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Generate and stage the five diagnostic arms before Writer release."""

    contract = runtime.contract
    task_adapter = runtime.task_adapter
    if (
        not is_rank_reserved_vertical_contract(contract, runtime.output_dir)
        or not callable(getattr(task_adapter, "prepare_diagnostic_five_arms", None))
        or not callable(getattr(task_adapter, "last_diagnostic_five_arm_profile", None))
        or torch.cuda.get_device_name(0) != "NVIDIA A40"
        or preflight.get("compute_applications") != []
        or preflight.get("device_names") != ["NVIDIA A40"]
        or preflight.get("physical_gpu_ids") != [runtime.gpu_index]
    ):
        raise WriterModelError("rank-reserved vertical prepare contract changed")
    selected = _rank_reserved_vertical_tasks(runtime)
    identities = tuple(
        {
            "suite": task["suite"],
            "task_id": int(task["task_id"]),
            "init_state_id": 0,
        }
        for task in selected
    )
    arm_order = (
        "old_full_rank_base",
        "old_full_rank_reward",
        "rank14_base",
        "rank14_plus2_qv_only",
        "rank14_plus2_reward",
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    generated = task_adapter.prepare_diagnostic_five_arms(identities)
    if tuple(generated) != arm_order or any(
        len(generated[arm]) != len(selected) for arm in arm_order
    ):
        raise WriterModelError("rank-reserved vertical arms changed")
    flattened = tuple(
        generated[arm][index] for index in range(len(selected)) for arm in arm_order
    )
    staged_all = stage_writer_lora_states_to_cpu(
        (*flattened, task_adapter.identity_state)
    )
    staged_flat = staged_all[:-1]
    identity_state = staged_all[-1]
    generation_plus_stage_seconds = time.monotonic() - started
    staged = {
        arm: tuple(
            staged_flat[index * len(arm_order) + arm_order.index(arm)]
            for index in range(len(selected))
        )
        for arm in arm_order
    }
    expected_storage = contract["writer_lora_cache"]["lora_storage_per_entry"]
    storage_valid = all(
        lora_state_storage(state) == expected_storage
        for state in (*staged_flat, identity_state)
    )
    if expected_storage.get("dtype_tensor_counts") != {"BF16": 72, "F32": 4}:
        storage_valid = False
    profile = task_adapter.last_diagnostic_five_arm_profile()
    if (
        len(profile) != len(selected)
        or tuple((row["suite"], int(row["task_id"])) for row in profile)
        != tuple((task["suite"], int(task["task_id"])) for task in selected)
        or any(int(row["sampled_frames"]) <= 0 for row in profile)
    ):
        raise WriterModelError("rank-reserved diagnostic video evidence changed")
    return {
        "arm_order": arm_order,
        "states": staged,
        "identity_state": identity_state,
        "selected_tasks": selected,
        "video_evidence": tuple(dict(row) for row in profile),
        "native_storage_valid": storage_valid,
        "macro0_zero_slots": _rank_reserved_zero_slots(staged["rank14_base"]),
        "diagnostic_generation_plus_stage_seconds": generation_plus_stage_seconds,
        "diagnostic_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "diagnostic_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _load_vertical_cache(runtime: Any, prepared: Mapping[str, Any]) -> dict[str, Any]:
    contract = runtime.contract
    states = prepared["states"]
    selected = tuple(prepared["selected_tasks"])
    cached = tuple(
        runtime.task_adapter.prepare_episode(
            suite=str(task["suite"]),
            task_id=int(task["task_id"]),
            init_state_id=0,
        )
        for task in selected
    )
    cached_states = stage_writer_lora_states_to_cpu(
        tuple(item.state for item in cached)
    )
    expected_storage = contract["writer_lora_cache"]["lora_storage_per_entry"]
    cached_bases = tuple(
        _rank_reserved_cached_base_state(
            states["rank14_base"][index], cached[index].state
        )
        for index in range(len(selected))
    )
    cached_bases_cpu = tuple(
        _rank_reserved_cached_base_state(
            states["rank14_base"][index], cached_states[index]
        )
        for index in range(len(selected))
    )
    arm_order = tuple(prepared["arm_order"])
    execution_states = tuple(
        {
            "old_full_rank_base": states["old_full_rank_base"][index],
            "old_full_rank_reward": states["old_full_rank_reward"][index],
            "rank14_base": cached_bases[index],
            "rank14_plus2_qv_only": _rank_reserved_qv_only_state(
                cached_bases[index], cached[index].state
            ),
            "rank14_plus2_reward": cached[index].state,
        }
        for index in range(len(selected))
    )
    return {
        "contract": contract,
        "states": states,
        "selected": selected,
        "cached_states": cached_states,
        "cached_bases_cpu": cached_bases_cpu,
        "execution_states": execution_states,
        "arm_order": arm_order,
        "manifest": validate_writer_cache_manifest(
            contract,
            verify_entry_files=True,
        ),
        "expected_storage": expected_storage,
        "cache_storage_valid": all(
            lora_state_storage(state) == expected_storage for state in cached_states
        ),
        "cache_video_identity_exact": all(
            _rank_reserved_video_identity(item.evidence)
            == _rank_reserved_video_identity(prepared["video_evidence"][index])
            for index, item in enumerate(cached)
        ),
    }


def _run_vertical_task_probe(
    runtime: Any,
    prepared: Mapping[str, Any],
    loaded: Mapping[str, Any],
    *,
    index: int,
    root_seed: int,
    dummy: Any,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], dict[str, Any]]:
    from ember.pi05_evaluation import _start_fixed_episode, make_policy_noise
    from ember.pi05_processing import libero_policy_input
    from ember.writer.lora_rollout import PreparedWriterLoRA

    task = loaded["selected"][index]
    states = loaded["states"]
    cached_states = loaded["cached_states"]
    cached_bases_cpu = loaded["cached_bases_cpu"]
    qv = _effective_delta_moments(
        states["old_full_rank_base"][index],
        states["old_full_rank_reward"][index],
        cached_bases_cpu[index],
        cached_states[index],
    )
    envs, init_states = runtime.pool.switch(task)
    slot = _start_fixed_episode(
        env=envs[0],
        init_state_id=0,
        init_states=init_states,
        task=task,
        contract=loaded["contract"],
        root_seed=root_seed,
        dummy=dummy,
        task_adapter=None,
    )
    processed = runtime.preprocess(
        libero_policy_input(slot["obs"], str(task["language"]))
    )
    arm_order = loaded["arm_order"]
    batch = {
        name: torch.cat([value] * len(arm_order), dim=0)
        for name, value in processed.items()
        if isinstance(value, torch.Tensor)
    }
    planning = tuple(dict(slot) for _ in arm_order)
    noise, seeds = make_policy_noise(
        planning,
        root_seed=root_seed,
        suite=str(task["suite"]),
        task_id=int(task["task_id"]),
        chunk_size=int(runtime.policy.config.chunk_size),
        max_action_dim=int(runtime.policy.config.max_action_dim),
        device=batch[next(iter(batch))].device,
    )
    if len(set(seeds)) != 1:
        raise WriterModelError("rank-reserved diagnostic policy noise changed")
    runtime.policy.reset()
    chunks = runtime.task_adapter.predict_action_chunk(
        tuple(
            PreparedWriterLoRA(
                state=loaded["execution_states"][index][arm],
                evidence={},
            )
            for arm in arm_order
        ),
        batch,
        noise=noise,
        num_steps=int(loaded["contract"]["policy"]["num_inference_steps"]),
    )
    actions = runtime.postprocess(chunks).detach()
    action_by_arm = {
        arm: actions[position : position + 1]
        for position, arm in enumerate(arm_order)
    }
    identity_batch = {
        name: value
        for name, value in processed.items()
        if isinstance(value, torch.Tensor)
    }
    identity = PreparedWriterLoRA(state=prepared["identity_state"], evidence={})
    runtime.policy.reset()
    identity_chunks = runtime.task_adapter.predict_action_chunk(
        (identity,),
        identity_batch,
        noise=noise[:1],
        num_steps=int(loaded["contract"]["policy"]["num_inference_steps"]),
    )
    identity_action = runtime.postprocess(identity_chunks).detach()
    runtime.policy.reset()
    source_chunks = runtime.policy.predict_action_chunk(
        dict(identity_batch),
        noise=noise[:1],
        num_steps=int(loaded["contract"]["policy"]["num_inference_steps"]),
    )
    source_action = runtime.postprocess(source_chunks).detach()
    full_action = _vector_delta_moments(
        action_by_arm["old_full_rank_base"],
        action_by_arm["old_full_rank_reward"],
        action_by_arm["rank14_base"],
        action_by_arm["rank14_plus2_reward"],
    )
    qv_only_action = _vector_delta_moments(
        action_by_arm["rank14_base"],
        action_by_arm["rank14_plus2_qv_only"],
        action_by_arm["rank14_base"],
        action_by_arm["rank14_plus2_reward"],
    )
    row = {
        "suite": task["suite"],
        "task_id": int(task["task_id"]),
        "init_state_id": 0,
        "policy_noise_seed": int(seeds[0]),
        "native_qv_effective": _moment_summary((qv,)),
        "full_lora_policy_action": _moment_summary((full_action,)),
        "qv_only_vs_full_policy_action": _moment_summary((qv_only_action,)),
        "source_identity_action_exact": bool(
            torch.equal(identity_action, source_action)
        ),
    }
    return qv, full_action, qv_only_action, row


def _vertical_passes(
    prepared: Mapping[str, Any],
    loaded: Mapping[str, Any],
    summaries: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    cached_base_zero_slots: Mapping[str, Any],
) -> bool:
    return bool(
        prepared["native_storage_valid"]
        and loaded["cache_storage_valid"]
        and loaded["cache_video_identity_exact"]
        and prepared["macro0_zero_slots"]["exact_zero"]
        and cached_base_zero_slots["exact_zero"]
        and summaries["qv"]["new_nonzero_targets"]
        == summaries["qv"]["target_comparisons"]
        and all(
            row["native_qv_effective"]["new_nonzero_targets"] == 36
            and row["native_qv_effective"]["new_delta_l2_rms_across_panel"] > 0
            and row["qv_only_vs_full_policy_action"][
                "old_delta_l2_rms_across_panel"
            ]
            > 0
            and row["qv_only_vs_full_policy_action"][
                "new_delta_l2_rms_across_panel"
            ]
            > 0
            and row["source_identity_action_exact"]
            for row in rows
        )
    )


def _build_vertical_result(
    runtime: Any,
    prepared: Mapping[str, Any],
    loaded: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    summaries: Mapping[str, Any],
    cached_base_zero_slots: Mapping[str, Any],
    action_seconds: float,
    passed: bool,
) -> dict[str, Any]:
    contract = loaded["contract"]
    selected = loaded["selected"]
    arm_order = loaded["arm_order"]
    manifest = loaded["manifest"]
    manifest_path = writer_cache_manifest_path(contract)
    configured_batch = int(contract["parallel"]["writer_generation_batch_size"])
    actual_cache_batch = min(configured_batch, len(manifest["entry_ids"]))
    return {
        "schema_version": RANK_RESERVED_VERTICAL_SCHEMA,
        "root": str(runtime.output_dir.resolve()),
        "passed": passed,
        "git": dict(contract["git"]),
        "contract_reference": contract["contract_reference"],
        "preflight": dict(preflight),
        "writer_asset_reference": contract["adapter"]["writer_asset"]["reference"],
        "panel": "first_sealed_validation_task_per_suite_init_state0",
        "diagnostic_video_batch_size": len(selected),
        "diagnostic_video_encoder_forwards": 1,
        "diagnostic_policy_action_forwards": len(selected) * 3,
        "diagnostic_policy_action_batch_sizes": [
            [len(arm_order), 1, 1] for _ in selected
        ],
        "diagnostic_policy_action_samples": len(selected) * (len(arm_order) + 2),
        "teacher_video_evidence": list(prepared["video_evidence"]),
        "native_lora_storage": dict(loaded["expected_storage"]),
        "native_storage_valid": prepared["native_storage_valid"],
        "cache_storage_valid": loaded["cache_storage_valid"],
        "cache_video_identity_exact": loaded["cache_video_identity_exact"],
        "canonical_cache_used_for_full_reward_action": True,
        "canonical_cache_used_for_qv_only_action": True,
        "cached_reward_paired_base_zeroed_from_same_state": True,
        "configured_writer_generation_batch_size": configured_batch,
        "expected_actual_cache_generation_batch_size": actual_cache_batch,
        "cache_manifest": {
            "path": str(manifest_path.resolve()),
            "bytes": manifest_path.stat().st_size,
            "entry_count": len(manifest["entry_ids"]),
        },
        "writer_modules_released_before_actions": True,
        "source_policy_reused": True,
        "macro0_qv_residual_slots": dict(prepared["macro0_zero_slots"]),
        "cached_paired_base_qv_residual_slots": dict(cached_base_zero_slots),
        "native_qv_effective": summaries["qv"],
        "full_lora_policy_action": summaries["full_action"],
        "qv_only_vs_full_policy_action": summaries["qv_only_action"],
        "source_identity_action_exact": all(
            row["source_identity_action_exact"] for row in rows
        ),
        "diagnostic_generation_plus_stage_seconds": prepared[
            "diagnostic_generation_plus_stage_seconds"
        ],
        "diagnostic_action_seconds": action_seconds,
        "diagnostic_peak_allocated_bytes": prepared["diagnostic_peak_allocated_bytes"],
        "diagnostic_peak_reserved_bytes": prepared["diagnostic_peak_reserved_bytes"],
        "diagnostic_action_peak_allocated_bytes": int(
            torch.cuda.max_memory_allocated()
        ),
        "diagnostic_action_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "post_release_allocated_bytes": int(torch.cuda.memory_allocated()),
        "post_release_reserved_bytes": int(torch.cuda.memory_reserved()),
        "teacher_action_reads": 0,
        "teacher_state_reads": 0,
        "reward_reads": 0,
        "terminal_reads": 0,
        "rows": list(rows),
        "interpretation": "mechanism_and_deployment_closure_only_not_method_selection",
    }


@torch.inference_mode()
def complete_rank_reserved_vertical(
    runtime: Any,
    prepared: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Close q/v through native cache, released Writer, and actual policy actions."""

    import numpy as np

    if (
        not is_rank_reserved_vertical_contract(runtime.contract, runtime.output_dir)
        or hasattr(runtime.task_adapter, "writer")
        or not callable(getattr(runtime.task_adapter, "prepare_episode", None))
    ):
        raise WriterModelError("rank-reserved vertical handoff changed")
    loaded = _load_vertical_cache(runtime, prepared)
    root_seed = int(loaded["contract"]["rng"]["inference_seed"])
    dummy = np.asarray(
        loaded["contract"]["environment"]["dummy_action"],
        dtype=np.float32,
    )
    moments: dict[str, list[tuple[float, ...]]] = {
        "qv": [],
        "full_action": [],
        "qv_only_action": [],
    }
    rows = []
    torch.cuda.reset_peak_memory_stats()
    action_started = time.monotonic()
    for index in range(len(loaded["selected"])):
        qv, full_action, qv_only_action, row = _run_vertical_task_probe(
            runtime,
            prepared,
            loaded,
            index=index,
            root_seed=root_seed,
            dummy=dummy,
        )
        moments["qv"].append(qv)
        moments["full_action"].append(full_action)
        moments["qv_only_action"].append(qv_only_action)
        rows.append(row)
    summaries = {name: _moment_summary(value) for name, value in moments.items()}
    zero_slots = _rank_reserved_zero_slots(loaded["cached_bases_cpu"])
    result = _build_vertical_result(
        runtime,
        prepared,
        loaded,
        preflight=preflight,
        rows=rows,
        summaries=summaries,
        cached_base_zero_slots=zero_slots,
        action_seconds=time.monotonic() - action_started,
        passed=_vertical_passes(prepared, loaded, summaries, rows, zero_slots),
    )
    write_json_atomic(runtime.output_dir / "rank_reserved_vertical.json", result)
    return result
