"""Synthetic sealed PI05 panels shared by evaluator-analysis tests."""

from __future__ import annotations

from typing import Callable

from ember.eval_adapters import paired_writer_identity
from ember.expert_manifold.video_schedule import (
    SAME_TASK_OTHER_OFFSET,
    task_video_mapping,
)
from ember.pi05_eval_results import AGGREGATE_SCHEMA


TASKS = (
    ("libero_spatial", 1),
    ("libero_spatial", 3),
    ("libero_object", 1),
    ("libero_object", 3),
    ("libero_goal", 3),
    ("libero_goal", 6),
    ("libero_10", 1),
    ("libero_10", 2),
)

TASK_LANGUAGES = {
    ("libero_spatial", 1): "pick up the black bowl next to the ramekin and place it on the plate",
    ("libero_spatial", 3): "pick up the black bowl on the cookie box and place it on the plate",
    ("libero_object", 1): "pick up the cream cheese and place it in the basket",
    ("libero_object", 3): "pick up the bbq sauce and place it in the basket",
    ("libero_goal", 3): "open the top drawer and put the bowl inside",
    ("libero_goal", 6): "put the cream cheese in the bowl",
    ("libero_10", 1): "put both the cream cheese box and the butter in the basket",
    ("libero_10", 2): "turn on the stove and put the moka pot on it",
}

FAMILY_CONTRACTS = {
    "legacy": {
        "adapter_schema": "ember_pi05_v6_prior_eval_adapter_v5",
        "episode_schema": "ember_pi05_v6_prior_episode_v5",
        "config_schema": "ember_pi05_v6_prior_policy_effective_writer_v1",
        "arm_prefix": "expert_manifold_v6_prior_",
        "trained_checkpoint_kind": "v6_prior_trained_checkpoint",
    },
    "ecp": {
        "adapter_schema": "ember_pi05_v6_ecp_eval_adapter_v6",
        "episode_schema": "ember_pi05_v6_ecp_episode_v6",
        "config_schema": "ember_pi05_v6_ecp_policy_effective_writer_v2",
        "arm_prefix": "expert_manifold_v6_ecp_",
        "trained_checkpoint_kind": "v6_ecp_trained_checkpoint",
    },
    "tangent": {
        "adapter_schema": "ember_pi05_v6_tangent_tube_eval_adapter_v7",
        "episode_schema": "ember_pi05_v6_tangent_tube_episode_v7",
        "config_schema": "ember_pi05_v6_condition_local_tangent_tube_writer_v3",
        "arm_prefix": "expert_manifold_v6_tangent_tube_",
        "trained_checkpoint_kind": "v6_tangent_tube_trained_checkpoint",
    },
    "residual": {
        "adapter_schema": "ember_pi05_v6_condition_program_residual_eval_adapter_v8",
        "episode_schema": "ember_pi05_v6_condition_program_residual_episode_v8",
        "config_schema": (
            "ember_pi05_v6_counterfactual_null_condition_kernel_program_residual_v2"
        ),
        "arm_prefix": "expert_manifold_v6_condition_residual_",
        "trained_checkpoint_kind": "v6_condition_program_residual_checkpoint",
    },
    "reconciliation": {
        "adapter_schema": "ember_pi05_v6_condition_program_residual_eval_adapter_v8",
        "episode_schema": "ember_pi05_v6_condition_program_residual_episode_v8",
        "config_schema": (
            "ember_pi05_v6_exact_anchored_reconciliation_program_residual_v3"
        ),
        "arm_prefix": "expert_manifold_v6_condition_residual_",
        "trained_checkpoint_kind": "v6_condition_program_residual_checkpoint",
    },
    "reward": {
        "adapter_schema": "ember_pi05_v6_condition_program_residual_eval_adapter_v8",
        "episode_schema": "ember_pi05_v6_condition_program_residual_episode_v8",
        "config_schema": "ember_pi05_v6_reward_credit_program_cotangent_v1",
        "arm_prefix": "expert_manifold_v6_condition_residual_",
        "trained_checkpoint_kind": "v6_condition_program_residual_checkpoint",
    },
    "osg": {
        "adapter_schema": "ember_pi05_v6_condition_program_residual_eval_adapter_v9",
        "episode_schema": "ember_pi05_v6_condition_program_residual_episode_v9",
        "config_schema": (
            "ember_pi05_v6_on_policy_success_guarded_program_credit_v1"
        ),
        "arm_prefix": "expert_manifold_v6_condition_residual_",
        "trained_checkpoint_kind": "v6_condition_program_residual_checkpoint",
    },
    "sknc": {
        "adapter_schema": "ember_pi05_v6_condition_program_residual_eval_adapter_v9",
        "episode_schema": "ember_pi05_v6_condition_program_residual_episode_v9",
        "config_schema": (
            "ember_pi05_v6_success_key_nullspace_consolidation_v1"
        ),
        "arm_prefix": "expert_manifold_v6_condition_residual_",
        "trained_checkpoint_kind": "v6_condition_program_residual_checkpoint",
    },
    "cgik": {
        "adapter_schema": (
            "ember_pi05_v6_causal_goal_interaction_joint_credit_eval_adapter_v10"
        ),
        "episode_schema": (
            "ember_pi05_v6_causal_goal_interaction_joint_credit_episode_v10"
        ),
        "config_schema": (
            "ember_pi05_v6_causal_goal_interaction_key_joint_credit_v1"
        ),
        "arm_prefix": "expert_manifold_v6_cgik_jc_",
        "trained_checkpoint_kind": "v6_condition_program_residual_checkpoint",
    },
}


def success_keys(
    predicate: Callable[[str, int, int], bool],
) -> set[tuple[str, int, int]]:
    return {
        (suite, task_id, state)
        for suite, task_id in TASKS
        for state in range(50)
        if predicate(suite, task_id, state)
    }


def tasks() -> list[dict]:
    return [
        {
            "suite": suite,
            "task_id": task_id,
            "split_role": "validation",
            "language": TASK_LANGUAGES[(suite, task_id)],
            "init_state_ids": list(range(50)),
        }
        for suite, task_id in TASKS
    ]


def _adapter(macro: int, condition: str, *, family: str = "ecp") -> dict:
    contract = FAMILY_CONTRACTS[family]
    roles = {key: "validation" for key in TASKS}
    mapping = list(task_video_mapping(TASKS, roles, condition))
    checkpoint_kind = (
        "historical_v6_macro400_load_only"
        if macro == 0
        else contract["trained_checkpoint_kind"]
    )
    manifest_schema = None
    if family == "reconciliation" and macro > 0:
        manifest_schema = (
            "ember_pi05_v6_anchored_reconciliation_program_residual_checkpoint_v3"
        )
    elif family == "reward" and macro > 0:
        manifest_schema = (
            "ember_pi05_v6_reward_credit_program_cotangent_checkpoint_v4"
        )
    writer_asset = {
        "reference": f"writer:m{macro}",
        "kind": checkpoint_kind,
        "training_mode": "historical_v6_task_complete" if macro == 0 else "formal",
        "source_macro": 400,
        "method_macro": macro,
        "checkpoint": f"/writer/macro_{macro}",
        "manifest": {
            "path": f"/writer/macro_{macro}/manifest.json",
            **({"schema": manifest_schema} if manifest_schema else {}),
        },
        "architecture": "v6-prior",
        "writer_parameter_count": 10_775_296,
        "deployment_trainable_parameter_count": 0,
        "generated_lora_tensor_count": 76,
        "writer_state": {
            "path": f"/writer/macro_{macro}/writer.safetensors",
            "bytes": 45_818_648,
            "state_tensor_count": 600,
            "template_lora_storage": {"tensor_count": 76, "rank": 16},
        },
    }
    if family in {
        "residual",
        "reconciliation",
        "reward",
        "osg",
        "sknc",
        "cgik",
    }:
        writer_asset.update(
            {
                "program_residual_value_count": 20_971_520,
                "residual_state": {
                    "kind": (
                        "fresh_elementwise_zero"
                        if macro == 0
                        else "memory_only_checkpoint"
                    ),
                    "path": (
                        None
                        if macro == 0
                        else f"/writer/macro_{macro}/program_memory.safetensors"
                    ),
                    "bytes": 0 if macro == 0 else 83_886_200,
                    "tensor_count": 0 if macro == 0 else 1,
                    "dtype": "torch.float32",
                    "shape": [256, 320, 256],
                    "value_count": 20_971_520,
                },
            }
        )
    formal_status = "sealed"
    if family in {"residual", "reconciliation"}:
        formal_status = "sealed_from_live_residual_deployment_profile"
    elif family == "reward":
        formal_status = "sealed_from_unchanged_v6_residual_deployment_graph"
    elif family == "osg":
        formal_status = "sealed_from_live_osg_pc_deployment_smoke"
    elif family == "sknc":
        formal_status = "sealed_from_live_sknc_deployment_smoke"
    elif family == "cgik":
        formal_status = "sealed_from_live_cgik_full96_profile"
    return {
        "schema_version": contract["adapter_schema"],
        "kind": "expert_manifold_writer",
        "arm": f"{contract['arm_prefix']}{condition}",
        "execution_backend": "online_writer_then_episode_cache",
        "config": {"schema": contract["config_schema"]},
        "writer_asset": writer_asset,
        "evaluation_authority": {"formal_status": formal_status},
        "video_data": {"root": "/videos", "tasks": "sealed-validation-8"},
        "lora_contract": {"reference": "lora-v1", "rank": 16, "target_count": 38},
        "video_schedule": {
            "seed": 7,
            "demo_count": 50,
            "sampling_mode": "without_replacement",
            "videos_per_condition": 1,
            "paired_between_all_video_conditions": True,
            "queue_order_independent": True,
        },
        "pairing_reference": "one-shot-pairing-v1",
        "video_condition": condition,
        "task_video_mapping": mapping,
        "information_wall": {
            "writer_input": "exact task language plus one action-hidden teacher video",
            "video_is_only_dynamic_value": True,
            "no_video_counterfactual": condition == "no_video",
            "teacher_action_reads": 0,
            "teacher_state_reads": 0,
            "reward_reads": 0,
            "terminal_reads": 0,
            "language_only_lora_path": False,
            "deployment_expert_bank_read": False,
        },
    }


def _rows(
    macro: int,
    condition: str,
    successes: set[tuple[str, int, int]],
    adapter: dict,
) -> list[dict]:
    writer_asset = adapter["writer_asset"]
    checkpoint_kind = writer_asset["kind"]
    mapping_by_task = {
        (row["suite"], int(row["task_id"])): row
        for row in adapter["task_video_mapping"]
    }
    episode_schema = next(
        contract["episode_schema"]
        for contract in FAMILY_CONTRACTS.values()
        if adapter["config"]["schema"] == contract["config_schema"]
    )
    residual_configs = {
        FAMILY_CONTRACTS[name]["config_schema"]
        for name in (
            "residual",
            "reconciliation",
            "reward",
            "osg",
            "sknc",
            "cgik",
        )
    }
    rows = []
    for suite, task_id in TASKS:
        task_mapping = mapping_by_task[(suite, task_id)]
        for state in range(50):
            reference = (state + task_id) % 50
            selected = (
                (reference + SAME_TASK_OTHER_OFFSET) % 50
                if condition == "same_task_other"
                else reference
            )
            writer = {
                "schema_version": episode_schema,
                "condition": condition,
                "teacher_video_kind": condition,
                "method_arm": adapter["arm"],
                "writer_asset_reference": writer_asset["reference"],
                "writer_method_macro": macro,
                "writer_checkpoint_kind": checkpoint_kind,
                "lora_contract_reference": "lora-v1",
                "pairing_reference": "one-shot-pairing-v1",
                "language_global_task_id": task_mapping["language_global_task_id"],
                "teacher_reference_demo_indices": [reference],
                "teacher_demo_indices": [selected],
                "teacher_video_seed_root": 7,
                "teacher_video_selection_seed": state * 100 + task_id,
                "teacher_video_sampling_mode": "without_replacement",
                "teacher_video_order_seeds": [state * 1000 + task_id],
                "writer_generation_seed_schedule": "one-shot-seed-v1",
                "teacher_video_frames_used": condition != "no_video",
                "teacher_video_count": int(condition != "no_video"),
                "video_suite": task_mapping["video_suite"],
                "video_task_id": task_mapping["video_task_id"],
                "video_global_task_id": task_mapping["video_global_task_id"],
                "video_split_role": task_mapping["video_split_role"],
            }
            if condition == "same_task_other":
                writer["teacher_demo_offset"] = SAME_TASK_OTHER_OFFSET
            if adapter["config"]["schema"] in residual_configs:
                writer.update(
                    {
                        "writer_parameter_count": 10_775_296,
                        "writer_deployment_trainable_parameter_count": 0,
                        "writer_program_residual_value_count": 20_971_520,
                        "generated_lora_tensor_count": 76,
                    }
                )
            key = (suite, task_id, state)
            rows.append(
                {
                    "suite": suite,
                    "task_id": task_id,
                    "init_state_id": state,
                    "split_role": "validation",
                    "language": TASK_LANGUAGES[(suite, task_id)],
                    "env_seed": 7,
                    "policy_seed_root": 7,
                    "policy_noise_seeds": [state + 1, state + 1001],
                    "success": key in successes,
                    "writer": writer,
                }
            )
    return rows


def result(
    macro: int,
    condition: str,
    successes: set[tuple[str, int, int]],
    *,
    physical_gpu_ids: tuple[int, ...] = (0, 1),
    family: str = "ecp",
) -> dict:
    adapter = _adapter(macro, condition, family=family)
    parallel = {
        "physical_gpu_ids": list(physical_gpu_ids),
        "physical_gpu_count": len(physical_gpu_ids),
        "worker_count": len(physical_gpu_ids) * 2,
        "replicas_per_gpu": 2,
        "envs_per_replica": 8,
        "writer_generation_batch_size": 16,
    }
    paired = {
        "schema_version": "ember_pi05_writer_paired_control_v2",
        "mode": "formal",
        "role": "validation",
        "git": {"commit": "formal-commit", "dirty_paths": []},
        "model": {"checkpoint": "/source"},
        "tokenizer": {"path": "/tokenizer"},
        "normalization": {"path": "/normalization"},
        "tasks": tasks(),
        "environment": {"fixed_init_state_count": 50},
        "policy": {"replan_steps": 5, "precision": "bfloat16"},
        "rng": {"inference_seed": 7},
        "parallel": parallel,
        "writer": paired_writer_identity(adapter),
    }
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "contract_reference": f"contract:m{macro}:{condition}:{physical_gpu_ids}",
        "arm": adapter["arm"],
        "role": "validation",
        "mode": "formal",
        "adapter": adapter,
        "paired_control": paired,
        "rows": _rows(macro, condition, successes, adapter),
    }
