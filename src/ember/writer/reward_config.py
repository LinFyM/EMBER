"""Authority for the V6-LPCP capacity-matched backbone-memory grid."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.pi05_source_checkpoint import read_json
from ember.writer.as_config import REPO_ROOT, load_writer_config
from ember.writer.errors import WriterModelError


REWARD_CONFIG_SCHEMA = "ember_pi05_v6_lpcp_capacity_matched_backbone_memory_grid_v1"
REWARD_LAUNCH_SCHEMA = (
    "ember_pi05_v6_lpcp_capacity_matched_backbone_memory_grid_launch_v1"
)
REWARD_CONFIG = REPO_ROOT / (
    "configs/pi05_writer_v6_lpcp_capacity_matched_backbone_memory_grid_v1.json"
)
_INITIALIZATION_CONTRACT = {
    "kind": "writer_weights_only_fresh_reward_optimizer",
    "as_macro": 25,
    "reference_arm": "same_cached_conditioning_with_query_delta_disabled_exact_as139",
    "candidate_arm": ("frozen_v6_lpcp_plus_capacity_matched_backbone_memory_grid"),
}
_DEPLOYMENT_CONTRACT = {
    "kind": "one_complete_38_target_rank32_capacity_matched_grid_lora",
    "carrier_rank": 16,
    "residual_bank_rank": 16,
    "public_rank": 32,
    "alpha": 32,
    "scale": 1.0,
    "factorization": "A_concat_A0_A0_and_B_concat_B0_deltaB",
    "carrier_compression": False,
    "second_adapter": False,
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
    "kind": (
        "cross_video_matched_batch_stratified_occupancy_endpoint_action_preference"
    ),
    "discordant_credit": (
        "softplus_deployed_endpoint_winner_distance_minus_loser_distance_at_"
        "one_maximum_disagreement_state_per_equal_progress_stratum"
    ),
    "tie_credit": "zero_for_both_success_and_both_failure",
    "replay_scope": (
        "eight_equal_progress_strata_per_successful_trajectory_with_one_maximum_"
        "matched_action_disagreement_state_per_stratum_and_equal_weight_inside_"
        "each_equal_weight_discordant_trajectory"
    ),
    "winner_loser_action_panel": (
        "both_arms_requeried_in_identical_observation_noise_precision_order_and_"
        "physical_batch_shape"
    ),
    "generated_endpoint_panel": (
        "one_complete_ten_step_policy_action_per_selected_observation_and_"
        "condition_with_the_same_rollout_policy_noise_as_both_matched_targets"
    ),
    "cross_video_credit": (
        "same_matched_batch_stratified_occupancy_endpoint_action_panel_exact_"
        "gradient_in_four_disjoint_correct_k4_conditions"
    ),
    "view_aggregation": (
        "equal_mean_of_four_view_writer_gradients_with_unit_task_weight"
    ),
    "occupancy_strata_per_trajectory": 8,
    "endpoint_solver_steps": 10,
    "endpoint_action_scope": "executed_prefix_only",
}
_COMMITMENT_CONTRACT = {
    "kind": (
        "actual_adam_candidate_first_global_task_complete_all_view_monotone_"
        "power_of_two_backtracking"
    ),
    "direction": (
        "actual_adamw_candidate_delta_from_equal_view_then_equal_task_mean_gradient"
    ),
    "direction_preconditioning": (
        "same_adamw_lr_betas_eps_weight_decay_clip_and_optimizer_state"
    ),
    "radius_schedule": (
        "actual_adam_candidate_times_two_to_the_negative_backtrack_index"
    ),
    "acceptance": (
        "first_candidate_with_strictly_lower_deployed_endpoint_margin_for_every_"
        "active_task_and_all_four_correct_video_views_on_the_same_panels_and_"
        "policy_noises_globally_synchronized_across_ranks"
    ),
    "max_backtracks": 10,
    "failure_action": "restore_step0_parameters_and_terminal_non_pass",
    "optimizer_state": "adam_moments_and_step_from_raw_gradient_are_retained",
    "task_weighting": "equal_mean_over_active_tasks_before_commitment",
    "view_weighting": ("equal_mean_over_four_correct_video_gradients_before_optimizer"),
    "fixed_scale_or_checkpoint_selection": False,
    "video_or_environment_recompute": False,
    "global_preference_evidence": "all_active_task_view_scalar_rows_only",
    "rank_state_contract": (
        "one_identical_accepted_scale_and_parameter_delta_on_every_rank"
    ),
    "formal_extension_status": "blocked_until_world3_shared_anchor_gate_passes",
}
_SMOKE_CONTRACT = {
    "cycle": 1,
    "shared_anchor_task_ids": [9, 15, 18],
    "required_world_size": 3,
    "assignment": "one_fixed_anchor_per_local_rank_in_list_order",
}


def _contains(values: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return {key: values.get(key) for key in expected} == dict(expected)


def _contract_is_valid(config: Mapping[str, Any], cold_start: str) -> bool:
    optimization = config.get("optimization", {})
    return all(
        (
            _contains(config.get("initialization", {}), _INITIALIZATION_CONTRACT),
            cold_start.startswith("runs/outputs/"),
            _contains(config.get("deployment", {}), _DEPLOYMENT_CONTRACT),
            _contains(config.get("data", {}), _DATA_CONTRACT),
            _contains(
                config.get("environment", {}),
                {
                    "paired_states_per_task": 2,
                    "arms_per_state": 2,
                    "rollouts_per_task": 4,
                    "persistent_lanes_per_task": 2,
                },
            ),
            _contains(config.get("objective", {}), _OBJECTIVE_CONTRACT),
            _contains(
                optimization,
                {
                    "trainable": (
                        "capacity_matched_backbone_memory_grid_2828928_parameters"
                    ),
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
                    "checkpoint_cycles": [1, 2],
                    "stage_stop_cycles": [1, 2],
                },
            ),
        )
    )


def load_reward_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    config = read_json(path)
    if config.get("schema_version") != REWARD_CONFIG_SCHEMA:
        raise WriterModelError("unsupported task-complete endpoint config")
    config_repo_root = path.parent.parent
    base_path = (config_repo_root / str(config.get("base_as_config", ""))).resolve()
    base = load_writer_config(base_path)
    initialization = config.get("initialization", {})
    cold_start = str(initialization.get("as_checkpoint", ""))
    if not _contract_is_valid(config, cold_start):
        raise WriterModelError("task-complete endpoint contract changed")
    config["resolved_base_as_config"] = str(base_path)
    config["cold_start_relative"] = cold_start
    return config, base


def require_reward_mode(config: dict[str, Any], mode: str) -> None:
    if mode not in {"smoke", "formal"}:
        raise WriterModelError("invalid task-complete endpoint mode")
    if mode == "formal" and config["formal_run"]["status"] not in {
        "ready",
        "sealed",
    }:
        raise WriterModelError(
            "formal task-complete endpoint training is not authorized"
        )
