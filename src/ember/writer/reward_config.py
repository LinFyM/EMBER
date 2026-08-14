"""Authority for V6-LPCP semantic factor-memory commitment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.pi05_source_checkpoint import read_json
from ember.writer.as_config import REPO_ROOT, load_writer_config
from ember.writer.errors import WriterModelError


REWARD_CONFIG_SCHEMA = (
    "ember_pi05_v6_lpcp_semantic_factor_memory_commitment_v1"
)
REWARD_LAUNCH_SCHEMA = (
    "ember_pi05_v6_lpcp_semantic_factor_memory_commitment_launch_v1"
)
REWARD_CONFIG = REPO_ROOT / (
    "configs/pi05_writer_v6_lpcp_semantic_factor_memory_commitment_v1.json"
)


def _contains(values: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return {key: values.get(key) for key in expected} == dict(expected)


def load_reward_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    config = read_json(path)
    if config.get("schema_version") != REWARD_CONFIG_SCHEMA:
        raise WriterModelError("unsupported SFMC config")
    config_repo_root = path.parent.parent
    base_path = (config_repo_root / str(config.get("base_as_config", ""))).resolve()
    base = load_writer_config(base_path)
    initialization = config.get("initialization", {})
    cold_start = str(initialization.get("as_checkpoint", ""))
    data = config.get("data", {})
    environment = config.get("environment", {})
    objective = config.get("objective", {})
    optimization = config.get("optimization", {})
    distributed = optimization.get("distributed", {})
    formal = config.get("formal_run", {})
    valid = (
        _contains(
            initialization,
            {
                "kind": "writer_weights_only_fresh_reward_optimizer",
                "as_macro": 25,
                "reference_arm": (
                    "same_cached_conditioning_with_query_delta_disabled_exact_as139"
                ),
                "candidate_arm": (
                    "frozen_v6_lpcp_plus_semantic_factor_memory_commitment"
                ),
            },
        ),
        cold_start.startswith("runs/outputs/"),
        _contains(
            data,
            {
                "task_count": 24,
                "videos_per_task": 4,
                "credit_views_per_active_task": 4,
                "videos_per_credit_view": 4,
                "unique_credit_videos_per_active_task": 16,
                "credit_view_schedule": (
                    "anchor_visit_then_three_subsequent_visits_with_"
                    "cumulative_exclusion"
                ),
                "demo_indices": [0, 49],
            },
        ),
        _contains(
            environment,
            {
                "paired_states_per_task": 2,
                "arms_per_state": 2,
                "rollouts_per_task": 4,
                "persistent_lanes_per_task": 2,
            },
        ),
        _contains(
            objective,
            {
                "kind": "cross_video_causal_selected_success_flow_distillation",
                "discordant_credit": "imitate_only_the_uniquely_successful_arm",
                "tie_credit": "zero_for_both_success_and_both_failure",
                "cross_video_credit": (
                    "same_selected_success_replay_exact_gradient_in_four_"
                    "disjoint_correct_k4_conditions"
                ),
                "view_aggregation": (
                    "equal_mean_of_four_view_writer_gradients_with_unit_task_weight"
                ),
                "flow_panel_across_views": (
                    "identical_replay_beta_times_and_gaussian_noises"
                ),
                "flow_mc_samples": 4,
            },
        ),
        _contains(
            optimization,
            {
                "trainable": (
                    "semantic_factor_memory_commitment_only_2164224_parameters"
                ),
                "reward_replay_chunk_batch_size": 8,
            },
        ),
        _contains(
            distributed,
            {
                "fresh_world_sizes": [1, 2, 3, 4, 5, 6],
                "collective_timeout_minutes": 30,
            },
        ),
        _contains(
            formal,
            {
                "allowed_world_sizes": [1, 2, 3, 4, 5, 6],
                "checkpoint_cycles": [1, 2],
                "stage_stop_cycles": [1, 2],
            },
        ),
    )
    if not all(valid):
        raise WriterModelError("SFMC scientific contract changed")
    config["resolved_base_as_config"] = str(base_path)
    config["cold_start_relative"] = cold_start
    return config, base


def require_reward_mode(config: dict[str, Any], mode: str) -> None:
    if mode not in {"smoke", "formal"}:
        raise WriterModelError("invalid SFMC mode")
    if mode == "formal" and config["formal_run"]["status"] not in {
        "ready",
        "sealed",
    }:
        raise WriterModelError("formal SFMC is not authorized")
