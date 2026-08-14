"""Authority for ordered-Procedure on-policy preference training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ember.pi05_source_checkpoint import read_json
from ember.writer.as_config import REPO_ROOT, load_writer_config
from ember.writer.errors import WriterModelError


REWARD_CONFIG_SCHEMA = "ember_pi05_v6_ordered_procedure_on_policy_preference_v1"
REWARD_LAUNCH_SCHEMA = "ember_pi05_v6_ordered_procedure_on_policy_preference_launch_v1"
REWARD_CONFIG = REPO_ROOT / (
    "configs/pi05_writer_v6_ordered_procedure_on_policy_preference_v1.json"
)


def load_reward_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = read_json(path)
    if config.get("schema_version") != REWARD_CONFIG_SCHEMA:
        raise WriterModelError("unsupported reward preference config")
    base_path = (REPO_ROOT / str(config.get("base_as_config", ""))).resolve()
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
        initialization.get("kind") != "writer_weights_only_fresh_reward_optimizer"
        or int(initialization.get("as_macro", -1)) != 25
        or not cold_start.startswith("runs/outputs/")
        or data.get("task_count") != 24
        or data.get("videos_per_task") != 4
        or data.get("demo_indices") != [0, 49]
        or environment.get("rollouts_per_task") != 4
        or environment.get("persistent_lanes_per_task") != 4
        or objective.get("flow_mc_samples") != 4
        or objective.get("homogeneous_task_credit")
        != "exact_zero_without_reward_cfm_forward"
        or optimization.get("reward_replay_chunk_batch_size") != 8
        or distributed.get("fresh_world_sizes") != [1, 2, 3, 4, 5, 6]
        or formal.get("allowed_world_sizes") != [1, 2, 3, 4, 5, 6]
        or formal.get("checkpoint_cycles") != [1, 2]
        or formal.get("stage_stop_cycles") != [1, 2]
    ):
        raise WriterModelError("reward preference scientific contract changed")
    config["resolved_base_as_config"] = str(base_path)
    config["cold_start_relative"] = cold_start
    return config, base


def require_reward_mode(config: dict[str, Any], mode: str) -> None:
    if mode not in {"smoke", "formal"}:
        raise WriterModelError("invalid reward preference mode")
    if mode == "formal" and config["formal_run"]["status"] != "sealed":
        raise WriterModelError("formal reward preference awaits its live smoke seal")
