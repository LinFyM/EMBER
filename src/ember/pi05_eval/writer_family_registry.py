"""Sealed Writer-family identities used by paired evaluation analysis."""

from __future__ import annotations


CHECKPOINT_MACROS = (0, 10, 25, 50)
HISTORICAL_TRANSITION_CANDIDATE_MACROS = {
    "v6_ecp_v2": (10, 25, 50),
    "v6_tangent_tube_v3": (10, 25, 50),
    "v6_condition_residual_v2": (10, 25, 50),
    "v6_anchored_reconciliation_v3": (10, 25),
    "v6_reward_credit_program_v1": (1, 2),
    "v6_pick_gc_v1": (10, 25),
    "v6_sknc_v1": (5, 10),
    "v6_srtp_v1": (5, 10),
    "v6_cveg_v1": (5, 10),
    "v6_pvjfc_v1": (5, 10),
    "v6_cgik_jc_v1": (5, 10),
    "v6_mgci_jc_v1": (5, 10),
}
HISTORICAL_TRANSITION_BASELINE_FAMILIES = {
    "v6_ecp_v2": "legacy_v6_prior_v1",
    "v6_tangent_tube_v3": "legacy_v6_prior_v1",
    "v6_condition_residual_v2": "legacy_v6_prior_v1",
    "v6_anchored_reconciliation_v3": "v6_condition_residual_v2",
    "v6_reward_credit_program_v1": "v6_condition_residual_v2",
    "v6_pick_gc_v1": "v6_condition_residual_v2",
    "v6_sknc_v1": "v6_condition_residual_v2",
    "v6_srtp_v1": "v6_condition_residual_v2",
    "v6_cveg_v1": "v6_condition_residual_v2",
    "v6_pvjfc_v1": "v6_condition_residual_v2",
    "v6_cgik_jc_v1": "v6_condition_residual_v2",
    "v6_mgci_jc_v1": "v6_condition_residual_v2",
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
    "v6_pick_gc_v1": {
        "adapter_schema": "ember_pi05_v6_condition_program_residual_eval_adapter_v8",
        "episode_schema": "ember_pi05_v6_condition_program_residual_episode_v8",
        "config_schema": "ember_pi05_v6_policy_innovation_goal_causal_key_v1",
        "arm_prefix": "expert_manifold_v6_condition_residual_",
        "trained_checkpoint_kind": "v6_condition_program_residual_checkpoint",
        "formal_statuses": ("sealed_from_live_pick_gc_deployment_profile",),
    },
    "v6_osg_pc_v1": {
        "adapter_schema": "ember_pi05_v6_condition_program_residual_eval_adapter_v9",
        "episode_schema": "ember_pi05_v6_condition_program_residual_episode_v9",
        "config_schema": ("ember_pi05_v6_on_policy_success_guarded_program_credit_v1"),
        "arm_prefix": "expert_manifold_v6_condition_residual_",
        "trained_checkpoint_kind": "v6_condition_program_residual_checkpoint",
        "formal_statuses": ("sealed_from_live_osg_pc_deployment_smoke",),
        "checkpoint_curve_macros": (0, 5, 10),
    },
    "v6_sknc_v1": {
        "adapter_schema": "ember_pi05_v6_condition_program_residual_eval_adapter_v9",
        "episode_schema": "ember_pi05_v6_condition_program_residual_episode_v9",
        "config_schema": "ember_pi05_v6_success_key_nullspace_consolidation_v1",
        "arm_prefix": "expert_manifold_v6_condition_residual_",
        "trained_checkpoint_kind": "v6_condition_program_residual_checkpoint",
        "formal_statuses": ("sealed_from_live_sknc_deployment_smoke",),
        "checkpoint_curve_macros": (0, 5, 10),
    },
    "v6_srtp_v1": {
        "adapter_schema": "ember_pi05_v6_condition_program_residual_eval_adapter_v9",
        "episode_schema": "ember_pi05_v6_condition_program_residual_episode_v9",
        "config_schema": "ember_pi05_v6_shared_reward_tangent_projection_v1",
        "arm_prefix": "expert_manifold_v6_condition_residual_",
        "trained_checkpoint_kind": "v6_condition_program_residual_checkpoint",
        "formal_statuses": ("sealed_from_live_srtp_deployment_smoke",),
        "checkpoint_curve_macros": (0, 5, 10),
    },
    "v6_cveg_v1": {
        "adapter_schema": "ember_pi05_v6_condition_program_residual_eval_adapter_v9",
        "episode_schema": "ember_pi05_v6_condition_program_residual_episode_v9",
        "config_schema": ("ember_pi05_v6_cross_video_equivariant_candidate_guard_v1"),
        "arm_prefix": "expert_manifold_v6_condition_residual_",
        "trained_checkpoint_kind": "v6_condition_program_residual_checkpoint",
        "formal_statuses": ("sealed_from_live_cveg_deployment_smoke",),
        "checkpoint_curve_macros": (0, 5, 10),
    },
    "v6_pvjfc_v1": {
        "adapter_schema": "ember_pi05_v6_condition_program_residual_eval_adapter_v9",
        "episode_schema": "ember_pi05_v6_condition_program_residual_episode_v9",
        "config_schema": ("ember_pi05_v6_paired_video_joint_functional_credit_v1"),
        "arm_prefix": "expert_manifold_v6_condition_residual_",
        "trained_checkpoint_kind": "v6_condition_program_residual_checkpoint",
        "formal_statuses": ("sealed_from_unchanged_v6_residual_deployment_graph",),
        "checkpoint_curve_macros": (0, 5, 10),
    },
    "v6_cgik_jc_v1": {
        "adapter_schema": (
            "ember_pi05_v6_causal_goal_interaction_joint_credit_eval_adapter_v10"
        ),
        "episode_schema": (
            "ember_pi05_v6_causal_goal_interaction_joint_credit_episode_v10"
        ),
        "config_schema": ("ember_pi05_v6_causal_goal_interaction_key_joint_credit_v1"),
        "arm_prefix": "expert_manifold_v6_cgik_jc_",
        "trained_checkpoint_kind": "v6_condition_program_residual_checkpoint",
        "formal_statuses": ("sealed_from_live_cgik_full96_profile",),
        "checkpoint_curve_macros": (0, 5, 10),
    },
    "v6_mgci_jc_v1": {
        "adapter_schema": (
            "ember_pi05_v6_magnitude_gated_causal_interaction_joint_credit_"
            "eval_adapter_v11"
        ),
        "episode_schema": (
            "ember_pi05_v6_magnitude_gated_causal_interaction_joint_credit_"
            "episode_v11"
        ),
        "config_schema": (
            "ember_pi05_v6_magnitude_gated_causal_interaction_key_joint_credit_v1"
        ),
        "arm_prefix": "expert_manifold_v6_mgci_jc_",
        "trained_checkpoint_kind": "v6_condition_program_residual_checkpoint",
        "formal_statuses": ("sealed_from_live_mgci_full96_profile",),
        "checkpoint_curve_macros": (0, 5, 10),
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
    "dynamic_k_backbone_memory_rank8_v1": {
        "adapter_schema": (
            "ember_pi05_dynamic_k_backbone_memory_rank8_eval_adapter_v1"
        ),
        "episode_schema": ("ember_pi05_dynamic_k_backbone_memory_rank8_episode_v1"),
        "config_schema": ("ember_pi05_dynamic_k_backbone_memory_rank8_as_writer_v1"),
        "writer_kind": "dynamic_k_backbone_memory_writer",
        "arm_prefix": "dynamic_k_backbone_memory_rank8_",
        "trained_checkpoint_kind": "dynamic_k_writer_macro_checkpoint",
        "formal_statuses": ("sealed",),
        "lora_rank": 8,
        "lora_target_count": 38,
        "videos_per_condition": 1,
        "writer_input": (
            "exact task language plus one action-hidden teacher video through "
            "the dynamic-K graph"
        ),
        "episode_validator": "dynamic_k",
        "checkpoint_curve_allowed_macros": (
            50,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
        ),
    },
    "dynamic_k_semantic_address_rank8_v1": {
        "adapter_schema": (
            "ember_pi05_dynamic_k_semantic_address_rank8_eval_adapter_v1"
        ),
        "episode_schema": ("ember_pi05_dynamic_k_semantic_address_rank8_episode_v1"),
        "config_schema": (
            "ember_pi05_dynamic_k_semantic_address_backbone_memory_rank8_"
            "as_writer_v1"
        ),
        "writer_kind": "dynamic_k_backbone_memory_writer",
        "arm_prefix": "dynamic_k_semantic_address_rank8_",
        "trained_checkpoint_kind": "dynamic_k_writer_macro_checkpoint",
        "formal_statuses": ("sealed",),
        "lora_rank": 8,
        "lora_target_count": 38,
        "videos_per_condition": 1,
        "writer_input": (
            "exact task language plus one action-hidden teacher video through "
            "the dynamic-K graph"
        ),
        "episode_validator": "dynamic_k",
        "checkpoint_curve_allowed_macros": (
            50,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
        ),
    },
    "dynamic_k_semantic_address_direct_family_b_rank8_v1": {
        "adapter_schema": (
            "ember_pi05_dynamic_k_semantic_address_direct_family_b_rank8_"
            "eval_adapter_v1"
        ),
        "episode_schema": (
            "ember_pi05_dynamic_k_semantic_address_direct_family_b_rank8_" "episode_v1"
        ),
        "config_schema": (
            "ember_pi05_dynamic_k_semantic_address_direct_family_b_rank8_"
            "as_writer_v1"
        ),
        "writer_kind": "dynamic_k_backbone_memory_writer",
        "arm_prefix": "dynamic_k_semantic_address_direct_family_b_rank8_",
        "trained_checkpoint_kind": (
            "dynamic_k_semantic_address_direct_family_b_rank8_macro_checkpoint"
        ),
        "formal_statuses": ("sealed",),
        "lora_rank": 8,
        "lora_target_count": 38,
        "videos_per_condition": 1,
        "supported_videos_per_condition": (1, 2, 3, 4),
        "writer_input": (
            "exact task language plus one action-hidden teacher video through "
            "the dynamic-K graph"
        ),
        "multi_video_writer_input_template": (
            "exact task language plus {evaluation_k} action-hidden teacher videos "
            "through the dynamic-K graph"
        ),
        "episode_validator": "dynamic_k",
        "checkpoint_curve_allowed_macros": (
            50,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
        ),
    },
    "dynamic_k_task_grounded_visual_value_rank8_v1": {
        "adapter_schema": (
            "ember_pi05_dynamic_k_task_grounded_visual_value_rank8_" "eval_adapter_v1"
        ),
        "episode_schema": (
            "ember_pi05_dynamic_k_task_grounded_visual_value_rank8_episode_v1"
        ),
        "config_schema": (
            "ember_pi05_dynamic_k_task_grounded_visual_value_rank8_as_writer_v1"
        ),
        "writer_kind": "dynamic_k_backbone_memory_writer",
        "arm_prefix": "dynamic_k_task_grounded_visual_value_rank8_",
        "trained_checkpoint_kind": (
            "dynamic_k_task_grounded_visual_value_rank8_macro_checkpoint"
        ),
        "formal_statuses": ("sealed",),
        "lora_rank": 8,
        "lora_target_count": 38,
        "videos_per_condition": 1,
        "supported_videos_per_condition": (1, 2, 3, 4),
        "writer_input": (
            "exact task language plus one action-hidden teacher video through "
            "the dynamic-K graph"
        ),
        "multi_video_writer_input_template": (
            "exact task language plus {evaluation_k} action-hidden teacher videos "
            "through the dynamic-K graph"
        ),
        "episode_validator": "dynamic_k",
        "checkpoint_curve_allowed_macros": (
            50,
            100,
            150,
            200,
            250,
            300,
            350,
            400,
        ),
    },
    "v6_shared_core_procedure_common_value_bridge_v1": {
        "adapter_schema": (
            "ember_pi05_v6_shared_core_procedure_common_value_bridge_eval_adapter_v1"
        ),
        "episode_schema": (
            "ember_pi05_v6_shared_core_procedure_common_value_bridge_episode_v1"
        ),
        "config_schema": (
            "ember_pi05_v6_shared_core_procedure_common_value_bridge_as_writer_v1"
        ),
        "writer_kind": "v6_shared_core_procedure_common_value_writer",
        "arm_prefix": "v6_shared_core_procedure_common_value_bridge_",
        "trained_checkpoint_kind": (
            "v6_shared_core_procedure_common_value_bridge_macro_checkpoint"
        ),
        "formal_statuses": ("sealed",),
        "lora_rank": 16,
        "lora_target_count": 38,
        "videos_per_condition": 1,
        "supported_videos_per_condition": (1, 2, 3, 4),
        "writer_input": (
            "exact task language plus one action-hidden teacher video through "
            "the dynamic-K graph"
        ),
        "multi_video_writer_input_template": (
            "exact task language plus {evaluation_k} action-hidden teacher videos "
            "through the dynamic-K graph"
        ),
        "episode_validator": "dynamic_k",
        "checkpoint_curve_allowed_macros": (
            25,
            50,
        ),
    },
    "v6_layerwise_probe_conditioned_procedure_v1": {
        "adapter_schema": (
            "ember_pi05_v6_layerwise_probe_conditioned_procedure_" "eval_adapter_v1"
        ),
        "episode_schema": (
            "ember_pi05_v6_layerwise_probe_conditioned_procedure_episode_v1"
        ),
        "config_schema": (
            "ember_pi05_v6_layerwise_probe_conditioned_procedure_as_writer_v1"
        ),
        "writer_kind": "v6_layerwise_probe_conditioned_procedure_writer",
        "arm_prefix": "v6_layerwise_probe_conditioned_procedure_",
        "trained_checkpoint_kind": (
            "v6_layerwise_probe_conditioned_procedure_macro_checkpoint"
        ),
        "formal_statuses": ("sealed",),
        "lora_rank": 16,
        "lora_target_count": 38,
        "videos_per_condition": 1,
        "supported_videos_per_condition": (1, 2, 3, 4),
        "writer_input": (
            "exact task language plus one action-hidden teacher video through "
            "the dynamic-K graph"
        ),
        "multi_video_writer_input_template": (
            "exact task language plus {evaluation_k} action-hidden teacher videos "
            "through the dynamic-K graph"
        ),
        "episode_validator": "dynamic_k",
        "checkpoint_curve_allowed_macros": (25, 50),
    },
    "v6_lpcp_cross_video_causal_success_distillation_v1": {
        "adapter_schema": (
            "ember_pi05_v6_layerwise_probe_conditioned_procedure_" "eval_adapter_v1"
        ),
        "episode_schema": (
            "ember_pi05_v6_layerwise_probe_conditioned_procedure_episode_v1"
        ),
        "config_schema": (
            "ember_pi05_v6_layerwise_probe_conditioned_procedure_as_writer_v1"
        ),
        "writer_kind": "v6_layerwise_probe_conditioned_procedure_writer",
        "arm_prefix": "v6_lpcp_cross_video_causal_success_distillation_",
        "trained_checkpoint_kind": (
            "v6_lpcp_cross_video_causal_success_distillation_cycle_checkpoint"
        ),
        "formal_statuses": ("sealed",),
        "lora_rank": 16,
        "lora_target_count": 38,
        "videos_per_condition": 1,
        "supported_videos_per_condition": (1, 2, 3, 4),
        "writer_input": (
            "exact task language plus one action-hidden teacher video through "
            "the dynamic-K graph"
        ),
        "multi_video_writer_input_template": (
            "exact task language plus {evaluation_k} action-hidden teacher videos "
            "through the dynamic-K graph"
        ),
        "episode_validator": "dynamic_k",
        "checkpoint_curve_allowed_macros": (1, 2),
    },
    "v6_lpcp_causal_coefficient_transport_v1": {
        "adapter_schema": (
            "ember_pi05_v6_layerwise_probe_conditioned_procedure_" "eval_adapter_v1"
        ),
        "episode_schema": (
            "ember_pi05_v6_layerwise_probe_conditioned_procedure_episode_v1"
        ),
        "config_schema": (
            "ember_pi05_v6_layerwise_probe_conditioned_procedure_as_writer_v1"
        ),
        "writer_kind": "v6_layerwise_probe_conditioned_procedure_writer",
        "arm_prefix": "v6_lpcp_causal_coefficient_transport_",
        "trained_checkpoint_kind": (
            "v6_lpcp_causal_coefficient_transport_cycle_checkpoint"
        ),
        "formal_statuses": ("sealed",),
        "lora_rank": 16,
        "lora_target_count": 38,
        "videos_per_condition": 4,
        "supported_videos_per_condition": (1, 2, 3, 4),
        "writer_input": (
            "exact task language plus four action-hidden teacher videos through "
            "the causal coefficient transport graph"
        ),
        "multi_video_writer_input_template": (
            "exact task language plus {evaluation_k} action-hidden teacher videos "
            "through the causal coefficient transport graph"
        ),
        "episode_validator": "dynamic_k",
        "checkpoint_curve_allowed_macros": (1, 2),
    },
    "v6_lpcp_cfmg_gradient_open_memory_query_v1": {
        "adapter_schema": (
            "ember_pi05_v6_layerwise_probe_conditioned_procedure_" "eval_adapter_v1"
        ),
        "episode_schema": (
            "ember_pi05_v6_layerwise_probe_conditioned_procedure_episode_v1"
        ),
        "config_schema": (
            "ember_pi05_v6_layerwise_probe_conditioned_procedure_as_writer_v1"
        ),
        "writer_kind": "v6_layerwise_probe_conditioned_procedure_writer",
        "arm_prefix": "v6_lpcp_cfmg_gradient_open_memory_query_",
        "trained_checkpoint_kind": (
            "v6_lpcp_cfmg_gradient_open_memory_query_cycle_checkpoint"
        ),
        "formal_statuses": ("sealed",),
        "lora_rank": 32,
        "lora_target_count": 38,
        "videos_per_condition": 4,
        "supported_videos_per_condition": (1, 2, 3, 4),
        "writer_input": (
            "exact task language plus four action-hidden teacher videos through "
            "the dynamic-K graph"
        ),
        "multi_video_writer_input_template": (
            "exact task language plus {evaluation_k} action-hidden teacher videos "
            "through the dynamic-K graph"
        ),
        "episode_validator": "dynamic_k",
        "checkpoint_curve_allowed_macros": (1, 2, 3, 4),
    },
    "pi05_layer_matched_memory_program_compiler_v2": {
        "adapter_schema": (
            "ember_pi05_layer_matched_memory_program_compiler_"
            "eval_adapter_v2"
        ),
        "episode_schema": (
            "ember_pi05_layer_matched_memory_program_compiler_episode_v2"
        ),
        "config_schema": (
            "ember_pi05_layer_matched_memory_program_compiler_writer_v2"
        ),
        "writer_kind": "layer_matched_memory_program_compiler_writer",
        "arm_prefix": "layer_matched_memory_program_compiler_",
        "trained_checkpoint_kind": (
            "layer_matched_memory_program_compiler_v2_macro_checkpoint"
        ),
        "formal_statuses": ("sealed",),
        "lora_rank": 16,
        "lora_target_count": 38,
        "videos_per_condition": 4,
        "supported_videos_per_condition": (1, 2, 3, 4),
        "writer_input": (
            "exact task language plus four action-hidden teacher videos through "
            "the layer-matched memory program compiler"
        ),
        "multi_video_writer_input_template": (
            "exact task language plus {evaluation_k} action-hidden teacher videos "
            "through the layer-matched memory program compiler"
        ),
        "episode_validator": "dynamic_k",
        "checkpoint_curve_allowed_macros": (25, 50, 75, 100),
    },
}

PROGRAM_RESIDUAL_WRITER_FAMILIES = frozenset(
    {
        "v6_condition_residual_v2",
        "v6_anchored_reconciliation_v3",
        "v6_reward_credit_program_v1",
        "v6_pick_gc_v1",
        "v6_osg_pc_v1",
        "v6_sknc_v1",
        "v6_srtp_v1",
        "v6_cveg_v1",
        "v6_pvjfc_v1",
        "v6_cgik_jc_v1",
        "v6_mgci_jc_v1",
        "v6_qv_rank_reserved_native_reward_v1",
    }
)
