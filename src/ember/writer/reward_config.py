"""Authority for V6-LPCP Adam-radius Euclidean commitment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.pi05_source_checkpoint import read_json
from ember.writer.as_config import REPO_ROOT, load_writer_config
from ember.writer.errors import WriterModelError


REWARD_CONFIG_SCHEMA = (
    "ember_pi05_v6_lpcp_direct_factor_adam_radius_euclidean_commitment_v1"
)
REWARD_LAUNCH_SCHEMA = (
    "ember_pi05_v6_lpcp_direct_factor_adam_radius_euclidean_commitment_launch_v1"
)
REWARD_CONFIG = REPO_ROOT / (
    "configs/pi05_writer_v6_lpcp_direct_factor_adam_radius_euclidean_commitment_v1.json"
)
_INITIALIZATION_CONTRACT = {
    "kind": "writer_weights_only_fresh_reward_optimizer",
    "as_macro": 25,
    "reference_arm": "same_cached_conditioning_with_query_delta_disabled_exact_as139",
    "candidate_arm": (
        "frozen_v6_lpcp_plus_direct_factor_adam_radius_euclidean_commitment"
    ),
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
    "kind": "cross_video_matched_batch_stratified_occupancy_flow_preference",
    "discordant_credit": (
        "softplus_matched_batch_winner_action_minus_loser_action_flow_loss_at_"
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
    "winner_loser_flow_panel": (
        "identical_beta_times_and_gaussian_noises_within_each_selected_pair"
    ),
    "cross_video_credit": (
        "same_matched_batch_stratified_occupancy_panel_exact_gradient_in_four_"
        "disjoint_correct_k4_conditions"
    ),
    "view_aggregation": (
        "equal_mean_of_four_view_writer_gradients_with_unit_task_weight"
    ),
    "flow_mc_samples": 4,
    "occupancy_strata_per_trajectory": 8,
}
_COMMITMENT_CONTRACT = {
    "kind": (
        "adam_candidate_global_l2_radius_with_negative_raw_shared_gradient_direction"
    ),
    "adam_candidate": (
        "same_raw_gradient_optimizer_state_lr_betas_eps_weight_decay_and_clip"
    ),
    "final_delta": (
        "negative_raw_shared_gradient_rescaled_to_exact_adam_candidate_global_l2"
    ),
    "optimizer_state": "adam_moments_and_step_from_raw_gradient_are_retained",
    "task_weighting": "equal_mean_over_active_tasks_before_commitment",
    "view_weighting": (
        "equal_mean_over_four_correct_video_gradients_before_commitment"
    ),
    "new_scale_or_solver": False,
}


def _contains(values: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return {key: values.get(key) for key in expected} == dict(expected)


def _contract_is_valid(config: Mapping[str, Any], cold_start: str) -> bool:
    optimization = config.get("optimization", {})
    return all(
        (
            _contains(config.get("initialization", {}), _INITIALIZATION_CONTRACT),
            cold_start.startswith("runs/outputs/"),
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
                        "eight_zero_init_direct_native_factor_heads_1654784_parameters"
                    ),
                    "matched_action_batch_size": 8,
                    "reward_replay_chunk_batch_size": 8,
                },
            ),
            _contains(config.get("commitment", {}), _COMMITMENT_CONTRACT),
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
        raise WriterModelError("unsupported Adam-radius commitment config")
    config_repo_root = path.parent.parent
    base_path = (config_repo_root / str(config.get("base_as_config", ""))).resolve()
    base = load_writer_config(base_path)
    initialization = config.get("initialization", {})
    cold_start = str(initialization.get("as_checkpoint", ""))
    if not _contract_is_valid(config, cold_start):
        raise WriterModelError("Adam-radius commitment contract changed")
    config["resolved_base_as_config"] = str(base_path)
    config["cold_start_relative"] = cold_start
    return config, base


def require_reward_mode(config: dict[str, Any], mode: str) -> None:
    if mode not in {"smoke", "formal"}:
        raise WriterModelError("invalid Adam-radius commitment mode")
    if mode == "formal" and config["formal_run"]["status"] not in {
        "ready",
        "sealed",
    }:
        raise WriterModelError(
            "formal Adam-radius commitment training is not authorized"
        )
