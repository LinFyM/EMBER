"""Authority for V6-LPCP paired causal success distillation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ember.pi05_source_checkpoint import read_json
from ember.writer.as_config import REPO_ROOT, load_writer_config
from ember.writer.errors import WriterModelError


REWARD_CONFIG_SCHEMA = "ember_pi05_v6_lpcp_paired_causal_success_distillation_v1"
REWARD_LAUNCH_SCHEMA = (
    "ember_pi05_v6_lpcp_paired_causal_success_distillation_launch_v1"
)
REWARD_CONFIG = REPO_ROOT / (
    "configs/pi05_writer_v6_lpcp_paired_causal_success_distillation_v1.json"
)


def load_reward_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    config = read_json(path)
    if config.get("schema_version") != REWARD_CONFIG_SCHEMA:
        raise WriterModelError("unsupported PCSD config")
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
    if (
        initialization.get("kind")
        != "writer_weights_only_fresh_reward_optimizer"
        or int(initialization.get("as_macro", -1)) != 25
        or initialization.get("reference_arm")
        != "same_cached_conditioning_with_query_delta_disabled_exact_as139"
        or initialization.get("candidate_arm") != "current_v6_lpcp_query_delta"
        or not cold_start.startswith("runs/outputs/")
        or data.get("task_count") != 24
        or data.get("videos_per_task") != 4
        or data.get("demo_indices") != [0, 49]
        or environment.get("paired_states_per_task") != 2
        or environment.get("arms_per_state") != 2
        or environment.get("rollouts_per_task") != 4
        or environment.get("persistent_lanes_per_task") != 2
        or objective.get("kind")
        != "paired_causal_selected_success_flow_distillation"
        or objective.get("discordant_credit")
        != "imitate_only_the_uniquely_successful_arm"
        or objective.get("tie_credit")
        != "zero_for_both_success_and_both_failure"
        or objective.get("flow_mc_samples") != 4
        or optimization.get("trainable")
        != "query_delta_weight_only_65536_parameters"
        or optimization.get("reward_replay_chunk_batch_size") != 8
        or distributed.get("fresh_world_sizes") != [1, 2, 3, 4, 5, 6]
        or distributed.get("collective_timeout_minutes") != 30
        or formal.get("allowed_world_sizes") != [1, 2, 3, 4, 5, 6]
        or formal.get("checkpoint_cycles") != [1, 2]
        or formal.get("stage_stop_cycles") != [1, 2]
    ):
        raise WriterModelError("PCSD scientific contract changed")
    config["resolved_base_as_config"] = str(base_path)
    config["cold_start_relative"] = cold_start
    return config, base


def require_reward_mode(config: dict[str, Any], mode: str) -> None:
    if mode not in {"smoke", "formal"}:
        raise WriterModelError("invalid PCSD mode")
    if mode == "formal" and config["formal_run"]["status"] not in {
        "ready",
        "sealed",
    }:
        raise WriterModelError("formal PCSD is not authorized")
