"""Immutable scientific configuration seal for the active frozen-v6 method."""

from __future__ import annotations

from ember.expert_manifold.v6_prior_policy_batch import (
    LOGICAL_POLICY_BATCH_SIZE,
    POSITIVE_POLICY_RANDOMNESS,
)


EXPECTED_WRITER = {
    "architecture": "frozen_pi05_v6_plus_video_keyed_program_residual_v2",
    "base_architecture": (
        "pi05_task_grounded_semantic_set_visual_transition_causal_procedure_"
        "slot_fusion_v6"
    ),
    "frame_stride": 5,
    "max_frames_per_encoder_call": 32,
    "image_width": 2048,
    "expert_width": 1024,
    "program_width": 256,
    "text_meta_lora_rank": 4,
    "vl_meta_lora_rank": 4,
    "action_meta_lora_rank": 4,
    "patch_grounding_heads": 8,
    "action_horizon": 50,
    "padded_action_dim": 32,
    "semantic_core_heads": 8,
    "semantic_core_blocks": 2,
    "procedure_heads": 8,
    "procedure_blocks": 2,
    "visual_transition_heads": 8,
    "fusion_heads": 8,
    "factor_hidden_width": 256,
    "initialization_seed": 7,
    "activation_checkpointing": True,
    "effective_runtime_activation_checkpointing": False,
    "all_600_state_tensors_frozen": True,
}

EXPECTED_CONDITION_FEATURE = {
    "kind": "zero_preserving_balanced_static_causal_visual_innovation_jl_v2",
    "input": "mean_valid_task_tokens_frame_evidence_minus_text_queries",
    "static_block": "video_dc_mean_visual_innovation",
    "dynamic_block": (
        "uniform_pool_sqrt_normalized_causal_prefix_of_video_centered_innovation"
    ),
    "descriptor_width": 512,
    "descriptor_block_width": 256,
    "feature_width": 256,
    "projection_seed": 20260810,
    "projection_shape": [2, 128, 256],
    "projection": "two_fixed_no_bias_row_normalized_fp32_nonpersistent_blocks",
    "block_normalization": "independent_zero_preserving_l2",
    "fusion": "concatenate_then_zero_preserving_l2",
    "frame_content_reordered_before_dynamic_block": True,
}

EXPECTED_PROGRAM_RESIDUAL = {
    "program_slots": 320,
    "program_width": 256,
    "feature_width": 256,
    "value_count": 20_971_520,
    "dtype": "float32",
    "initialization": "elementwise_zero",
    "fusion": "single_add_before_frozen_historical_factor_heads",
    "deployment_checkpoint_tensor_count": 1,
}

EXPECTED_RECONCILIATION = {
    "kind": "exact_anchored_recursive_least_squares",
    "feature_width": 256,
    "precision_shape": [256, 256],
    "precision_dtype": "float64",
    "initialization": "identity",
    "assimilated_rows_initial": 0,
    "rows_per_macro": 48,
    "history_storage": False,
    "checkpoint_owned": True,
    "deployment_owned": False,
}

EXPECTED_UPDATE = {
    "kind": "full48_exact_anchored_reconciliation",
    "correct_conditions": 24,
    "negative_conditions": 24,
    "ordering": "correct_task_ordinal_0_to_23_then_negative_task_ordinal_0_to_23",
    "negative_schedule": (
        "task_ordinal_plus_task_visit_modulo_reversed_shuffled_wrong"
    ),
    "negative_counts_per_macro": {"reversed": 8, "shuffled": 8, "wrong": 8},
    "anchored_target": "previous_condition_output_plus_current_descent_motion",
    "negative_rhs": "exact_zero_incremental_program_motion",
    "step_size": 1.0,
    "relative_damping": 0.01,
    "small_solve_dtype": "float64",
    "large_rhs_and_memory_write_dtype": "float32",
    "precision_transition": "lambda_inverse_feature_outer_product_add",
    "optimizer": "none_joint_manual_memory_and_precision_write",
    "momentum": False,
    "weight_decay": False,
    "gradient_clip": False,
    "global_scale_or_cap": False,
}

EXPECTED_DATA = {
    "task_count": 24,
    "episodes_per_task": 50,
    "demo_indices": [0, 49],
    "action_chunk_size": 50,
    "action_queries_per_task": LOGICAL_POLICY_BATCH_SIZE,
    "videos_per_task_per_macro": 1,
    "teacher_video_schedule": "deterministic_no_replacement_cycles",
    "teacher_action_episode_overlap": False,
    "task_aggregation": "task_local_B20_mean_with_no_cross_task_rescale",
    "sampler_seed": 20260721,
    "teacher_video_seed": 20260722,
    "counterfactual_seed": 20260809,
    "wrong_video_schedule": (
        "deterministic_cross_suite_cycle_with_current_task_language"
    ),
}

EXPECTED_OBJECTIVE = {
    "name": "correct_condition_policy_functional_flow_loss_only",
    "positive_policy_randomness": POSITIVE_POLICY_RANDOMNESS,
}

EXPECTED_OPTIMIZATION = {
    "precision": "bfloat16",
    "seed": 7,
    "functional_policy_microbatch_size": 10,
    "physical_policy_forwards_per_task": 2,
    "extra_negative_policy_forwards_per_task": 0,
    "distributed_update": {
        "kind": (
            "all_gather_local4_features_and_cotangents_then_identical_local_"
            "manual_write"
        ),
        "world_size": 6,
        "tasks_per_rank": 4,
        "memory_allreduce": False,
        "nccl_p2p_disable": "1",
        "nccl_algo": "Ring",
        "nccl_proto": "Simple",
        "deferred_process_group": True,
    },
}
