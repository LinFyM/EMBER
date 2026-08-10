"""Sealed Writer-family identities used by paired evaluation analysis."""

from __future__ import annotations


CHECKPOINT_MACROS = (0, 10, 25, 50)
HISTORICAL_TRANSITION_CANDIDATE_MACROS = {
    "v6_ecp_v2": (10, 25, 50),
    "v6_tangent_tube_v3": (10, 25, 50),
    "v6_condition_residual_v2": (10, 25, 50),
    "v6_anchored_reconciliation_v3": (10, 25),
    "v6_reward_credit_program_v1": (1, 2),
}
HISTORICAL_TRANSITION_BASELINE_FAMILIES = {
    "v6_ecp_v2": "legacy_v6_prior_v1",
    "v6_tangent_tube_v3": "legacy_v6_prior_v1",
    "v6_condition_residual_v2": "legacy_v6_prior_v1",
    "v6_anchored_reconciliation_v3": "v6_condition_residual_v2",
    "v6_reward_credit_program_v1": "v6_condition_residual_v2",
}

WRITER_FAMILIES = {
    "legacy_v6_prior_v1": {
        "adapter_schema": "ember_pi05_v6_prior_eval_adapter_v5",
        "episode_schema": "ember_pi05_v6_prior_episode_v5",
        "config_schema": "ember_pi05_v6_prior_policy_effective_writer_v1",
        "arm_prefix": "expert_manifold_v6_prior_",
        "trained_checkpoint_kind": "v6_prior_trained_checkpoint",
        "formal_statuses": ("sealed",),
    },
    "v6_ecp_v2": {
        "adapter_schema": "ember_pi05_v6_ecp_eval_adapter_v6",
        "episode_schema": "ember_pi05_v6_ecp_episode_v6",
        "config_schema": "ember_pi05_v6_ecp_policy_effective_writer_v2",
        "arm_prefix": "expert_manifold_v6_ecp_",
        "trained_checkpoint_kind": "v6_ecp_trained_checkpoint",
        "formal_statuses": ("sealed",),
    },
    "v6_tangent_tube_v3": {
        "adapter_schema": "ember_pi05_v6_tangent_tube_eval_adapter_v7",
        "episode_schema": "ember_pi05_v6_tangent_tube_episode_v7",
        "config_schema": "ember_pi05_v6_condition_local_tangent_tube_writer_v3",
        "arm_prefix": "expert_manifold_v6_tangent_tube_",
        "trained_checkpoint_kind": "v6_tangent_tube_trained_checkpoint",
        "formal_statuses": (
            "sealed",
            "sealed_from_unchanged_v6_deployment_graph",
        ),
    },
    "v6_condition_residual_v2": {
        "adapter_schema": "ember_pi05_v6_condition_program_residual_eval_adapter_v8",
        "episode_schema": "ember_pi05_v6_condition_program_residual_episode_v8",
        "config_schema": (
            "ember_pi05_v6_counterfactual_null_condition_kernel_program_residual_v2"
        ),
        "arm_prefix": "expert_manifold_v6_condition_residual_",
        "trained_checkpoint_kind": "v6_condition_program_residual_checkpoint",
        "formal_statuses": ("sealed_from_live_residual_deployment_profile",),
    },
    "v6_anchored_reconciliation_v3": {
        "adapter_schema": "ember_pi05_v6_condition_program_residual_eval_adapter_v8",
        "episode_schema": "ember_pi05_v6_condition_program_residual_episode_v8",
        "config_schema": (
            "ember_pi05_v6_exact_anchored_reconciliation_program_residual_v3"
        ),
        "arm_prefix": "expert_manifold_v6_condition_residual_",
        "trained_checkpoint_kind": "v6_condition_program_residual_checkpoint",
        "formal_statuses": ("sealed_from_live_residual_deployment_profile",),
    },
    "v6_reward_credit_program_v1": {
        "adapter_schema": "ember_pi05_v6_condition_program_residual_eval_adapter_v8",
        "episode_schema": "ember_pi05_v6_condition_program_residual_episode_v8",
        "config_schema": "ember_pi05_v6_reward_credit_program_cotangent_v1",
        "arm_prefix": "expert_manifold_v6_condition_residual_",
        "trained_checkpoint_kind": "v6_condition_program_residual_checkpoint",
        "formal_statuses": ("sealed_from_unchanged_v6_residual_deployment_graph",),
    },
    "v6_qv_rank_reserved_native_reward_v1": {
        "adapter_schema": (
            "ember_pi05_v6_qv_rank_reserved_native_reward_eval_adapter_v9"
        ),
        "episode_schema": ("ember_pi05_v6_qv_rank_reserved_native_reward_episode_v9"),
        "config_schema": "ember_pi05_v6_qv_rank_reserved_native_reward_v1",
        "arm_prefix": "expert_manifold_v6_qv_rank_reserved_native_reward_",
        "trained_checkpoint_kind": ("v6_qv_rank14_plus2_reward_program_load_only"),
        "checkpoint_kinds_by_macro": {
            0: "v6_qv_rank14_zero_program_load_only",
            1: "v6_qv_rank14_plus2_reward_program_load_only",
        },
        "formal_statuses": ("sealed_from_live_a40_rank_reserved_deployment_profile",),
    },
}

PROGRAM_RESIDUAL_WRITER_FAMILIES = frozenset(
    {
        "v6_condition_residual_v2",
        "v6_anchored_reconciliation_v3",
        "v6_reward_credit_program_v1",
        "v6_qv_rank_reserved_native_reward_v1",
    }
)
