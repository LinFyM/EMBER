"""Authority for CFMG successful-expert occupancy distillation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.pi05_source_checkpoint import read_json
from ember.writer.as_config import REPO_ROOT, load_writer_config
from ember.writer.errors import WriterModelError


REWARD_CONFIG_SCHEMA = (
    "ember_pi05_v6_lpcp_cfmg_successful_expert_occupancy_distillation_v1"
)
REWARD_LAUNCH_SCHEMA = (
    "ember_pi05_v6_lpcp_cfmg_successful_expert_occupancy_distillation_launch_v1"
)
REWARD_CONFIG = REPO_ROOT / (
    "configs/pi05_writer_v6_lpcp_cfmg_successful_expert_occupancy_distillation_v1.json"
)
_INITIALIZATION_CONTRACT = {
    "kind": "writer_weights_only_fresh_reward_optimizer",
    "as_macro": 25,
    "writer": "frozen_v6_lpcp_plus_cfmg_successful_expert_distillation",
}
_DEPLOYMENT_CONTRACT = {
    "kind": "one_complete_38_target_rank32_content_first_memory_grid_lora",
    "carrier_rank": 16,
    "residual_bank_rank": 16,
    "public_rank": 32,
    "alpha": 32,
    "scale": 1.0,
    "factorization": "A_concat_A0_A0_and_B_concat_B0_deltaB",
    "carrier_compression": False,
    "second_adapter": False,
    "task_expert_at_deployment": False,
}
_TEACHER_CONTRACT = {
    "kind": "train24_task_local_expert_successful_occupancy_only",
    "step": 2000,
    "rank": 16,
    "public_rank_padding": "rank16_factors_plus_exact_zero_second_rank16_bank",
    "failed_trajectory_credit": "zero",
    "deployment": False,
}
_DATA_CONTRACT = {
    "task_count": 24,
    "videos_per_task": 4,
    "credit_views_per_active_task": 4,
    "videos_per_credit_view": 4,
    "unique_credit_videos_per_active_task": 16,
    "credit_view_schedule": (
        "anchor_visit_then_three_subsequent_visits_with_cumulative_exclusion"
    ),
    "demo_indices": [0, 49],
}
_OBJECTIVE_CONTRACT = {
    "kind": "cross_video_successful_expert_occupancy_unit_residual_distillation",
    "success_filter": "expert_closed_loop_binary_success_only",
    "replay_scope": (
        "eight_equal_progress_strata_per_successful_expert_trajectory_with_one_"
        "maximum_expert_student_action_disagreement_state_per_stratum"
    ),
    "matched_action_panel": (
        "expert_and_writer_requeried_in_identical_observation_noise_precision_"
        "order_and_physical_batch_shape"
    ),
    "state_objective": (
        "executed_prefix_mse_divided_by_detached_current_expert_student_rms"
    ),
    "cross_video_credit": (
        "same_successful_expert_action_panel_exact_gradient_in_four_disjoint_"
        "correct_k4_conditions"
    ),
    "view_aggregation": "equal_mean_of_four_view_writer_gradients_with_unit_task_weight",
    "task_aggregation": (
        "equal_mean_after_parameter_free_median_upper_norm_cap_without_small_"
        "task_amplification"
    ),
    "occupancy_strata_per_trajectory": 8,
    "endpoint_solver_steps": 10,
    "endpoint_action_scope": "executed_prefix_only",
}
_COMMITMENT_CONTRACT = {
    "kind": "median_capped_successful_expert_task_tangent_actual_adam_single_step",
    "direction": (
        "actual_adamw_candidate_delta_from_equal_view_then_median_upper_norm_"
        "capped_task_mean_gradient"
    ),
    "direction_preconditioning": (
        "same_adamw_lr_betas_eps_weight_decay_clip_and_optimizer_state"
    ),
    "radius_schedule": "exact_actual_adam_candidate_only",
    "acceptance": (
        "finite_nonzero_rank_synchronized_actual_adam_candidate_with_all_active_"
        "task_view_objectives_recorded_as_diagnostic"
    ),
    "max_backtracks": 0,
    "failure_action": "terminal_only_on_nonfinite_zero_or_rank_inconsistent_update",
    "optimizer_state": "adam_moments_and_step_are_retained",
    "task_weighting": (
        "cap_only_norms_above_active_panel_median_never_amplify_small_tasks_then_"
        "equal_mean"
    ),
    "view_weighting": "equal_mean_over_four_correct_video_gradients_before_optimizer",
    "fixed_scale_or_checkpoint_selection": False,
    "video_or_environment_recompute": False,
    "rank_state_contract": "one_identical_parameter_delta_on_every_rank",
}
_SMOKE_CONTRACT = {
    "cycle": 1,
    "shared_anchor_task_ids": [2, 12, 21, 35],
    "required_world_size": 4,
    "assignment": "one_fixed_suite_anchor_per_local_rank_in_list_order",
}


def _contains(values: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return {key: values.get(key) for key in expected} == dict(expected)


def _contract_is_valid(config: Mapping[str, Any], cold_start: str) -> bool:
    optimization = config.get("optimization", {})
    teacher = config.get("privileged_teacher", {})
    return all(
        (
            _contains(config.get("initialization", {}), _INITIALIZATION_CONTRACT),
            cold_start.startswith("runs/outputs/"),
            _contains(config.get("deployment", {}), _DEPLOYMENT_CONTRACT),
            _contains(teacher, _TEACHER_CONTRACT),
            str(teacher.get("config", "")).startswith("configs/"),
            str(teacher.get("bank_root", "")).startswith("runs/outputs/"),
            _contains(config.get("data", {}), _DATA_CONTRACT),
            _contains(
                config.get("environment", {}),
                {
                    "expert_states_per_task": 2,
                    "rollouts_per_task": 2,
                    "persistent_lanes_per_task": 2,
                },
            ),
            _contains(config.get("objective", {}), _OBJECTIVE_CONTRACT),
            _contains(
                optimization,
                {
                    "trainable": "content_first_backbone_memory_grid_2828928_parameters",
                    "matched_action_batch_size": 8,
                    "endpoint_action_batch_size": 8,
                },
            ),
            _contains(config.get("commitment", {}), _COMMITMENT_CONTRACT),
            _contains(config.get("smoke_run", {}), _SMOKE_CONTRACT),
            _contains(
                optimization.get("distributed", {}),
                {
                    "fresh_world_sizes": [1, 2, 3, 4, 5, 6],
                    "collective_timeout_minutes": 30,
                },
            ),
            _contains(
                config.get("formal_run", {}),
                {
                    "allowed_world_sizes": [1, 2, 3, 4, 5, 6],
                    "total_cycles": 4,
                    "checkpoint_cycles": [1, 2, 3, 4],
                    "stage_stop_cycles": [1, 2, 3, 4],
                    "strict_paired400_cycles": [1, 2, 3, 4],
                    "owner_training_volume_extension": {
                        "parent_cycle": 3,
                        "parent_strict_curve": [129, 135, 143],
                        "cycle3_breadth": 5,
                        "cycle2_to_cycle3_retained_gained_lost": [120, 23, 15],
                        "extension_cycle": 4,
                        "scientific_variables_changed": False,
                        "decision": "one_bounded_adjacent_stability_checkpoint",
                    },
                },
            ),
        )
    )


def load_reward_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    config = read_json(path)
    if config.get("schema_version") != REWARD_CONFIG_SCHEMA:
        raise WriterModelError("unsupported successful-expert endpoint config")
    config_repo_root = path.parent.parent
    base_path = (config_repo_root / str(config.get("base_as_config", ""))).resolve()
    base = load_writer_config(base_path)
    initialization = config.get("initialization", {})
    cold_start = str(initialization.get("as_checkpoint", ""))
    if not _contract_is_valid(config, cold_start):
        raise WriterModelError("successful-expert endpoint contract changed")
    teacher = config["privileged_teacher"]
    config["resolved_base_as_config"] = str(base_path)
    config["cold_start_relative"] = cold_start
    config["resolved_task_expert_config"] = str(
        (config_repo_root / str(teacher["config"])).resolve()
    )
    return config, base


def require_reward_mode(config: dict[str, Any], mode: str) -> None:
    if mode not in {"smoke", "formal"}:
        raise WriterModelError("invalid successful-expert endpoint mode")
    if mode == "formal" and config["formal_run"]["status"] not in {"ready", "sealed"}:
        raise WriterModelError("formal successful-expert training is not authorized")
