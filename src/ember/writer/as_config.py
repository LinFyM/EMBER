"""Configuration authority for the dynamic-K Backbone-Memory Writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json
from ember.writer.errors import WriterModelError


REPO_ROOT = Path(__file__).resolve().parents[3]
AS_WRITER_CONFIG_SCHEMA = (
    "ember_pi05_dynamic_k_backbone_memory_rank8_as_writer_v1"
)
AS_WRITER_LAUNCH_SCHEMA = (
    "ember_pi05_dynamic_k_backbone_memory_rank8_as_writer_launch_v1"
)


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    return REPO_ROOT / str(config["authorities"][name]["path"])


def writer_stage(config: Mapping[str, Any]) -> str:
    stage = str(config.get("sealed_stage", ""))
    if stage != "development":
        raise WriterModelError("dynamic-K Writer currently supports development train24 only")
    return stage


def parse_macro_boundaries(value: str | list[int], total: int) -> tuple[int, ...]:
    if isinstance(value, str) and value.startswith("every:"):
        try:
            interval = int(value.removeprefix("every:"))
        except ValueError as error:
            raise WriterModelError("invalid Writer checkpoint interval") from error
        if interval <= 0 or total % interval:
            raise WriterModelError("Writer checkpoint interval must divide total macros")
        return tuple(range(interval, total + 1, interval))
    raw = value.split(",") if isinstance(value, str) else value
    try:
        result = tuple(sorted({int(item) for item in raw}))
    except (TypeError, ValueError) as error:
        raise WriterModelError("invalid Writer checkpoint macros") from error
    if not result or result[-1] != total or any(item <= 0 for item in result):
        raise WriterModelError("Writer checkpoints must end at total macros")
    return result


def _validate_authorities(config: Mapping[str, Any]) -> None:
    required = {
        "target_data_manifest",
        "evaluation_config",
        "lora_contract",
        "source_base_config",
        "tokenizer_manifest",
    }
    if set(config.get("authorities", {})) != required:
        raise WriterModelError("dynamic-K Writer authority set changed")
    for name in required:
        if not authority_path(config, name).is_file():
            raise WriterModelError(f"missing dynamic-K Writer authority: {name}")
    lora = load_pi05_lora_contract(authority_path(config, "lora_contract"))
    if (
        lora.rank != 8
        or lora.alpha != 8
        or lora.parameter_count != 643_584
        or lora.state_tensor_count != 76
    ):
        raise WriterModelError("dynamic-K Writer rank-8 LoRA contract changed")


def _validate_method(config: Mapping[str, Any]) -> None:
    writer = config.get("writer", {})
    data = config.get("data", {})
    training = config.get("conditioning_training", {})
    distributed = config.get("optimization", {}).get("distributed", {})
    if (
        writer.get("architecture") != "pi05_dynamic_k_backbone_memory_rank8_v1"
        or int(writer.get("frame_stride", 0)) != 5
        or int(writer.get("backbone_memory_tokens", 0)) != 8
        or int(writer.get("backbone_layers", 0)) != 18
        or int(data.get("task_count", 0)) != 24
        or int(data.get("episodes_per_task", 0)) != 50
        or data.get("demo_indices") != [0, 49]
        or int(data.get("action_queries_per_task", 0)) != 20
        or int(data.get("dynamic_k_max", 0)) != 4
        or training.get("global_tasks_per_optimizer_update") != 24
        or training.get("task_assignment")
        != "cost_balanced_long_first_dynamic_uneven"
        or float(
            training.get("singleton_to_full_consistency", {}).get("weight", -1)
        )
        != 0.05
        or distributed.get("fresh_world_sizes") != [1, 2, 3, 4, 5, 6]
        or distributed.get("gradient_communication")
        != "one_flat_writer_gradient_sum_all_reduce_per_macro_then_divide_by_24"
    ):
        raise WriterModelError("dynamic-K Writer scientific contract changed")


def _validate_runtime(config: Mapping[str, Any]) -> None:
    for key in ("profile_defaults", "formal_run"):
        cell = config.get(key, {})
        total = int(cell.get("total_macros", 0))
        batch = int(cell.get("per_task_action_batch_size", 0))
        if (
            cell.get("allowed_world_sizes") != [1, 2, 3, 4, 5, 6]
            or total <= 0
            or batch != 20
        ):
            raise WriterModelError("dynamic-K Writer runtime contract changed")
        parse_macro_boundaries(cell["checkpoint_macros"], total)
    if config["formal_run"].get("status") not in {
        "unsealed_pending_live_profile",
        "sealed",
    }:
        raise WriterModelError("dynamic-K Writer formal status changed")


def load_writer_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != AS_WRITER_CONFIG_SCHEMA:
        raise WriterModelError("unsupported dynamic-K Writer config schema")
    writer_stage(config)
    _validate_authorities(config)
    _validate_method(config)
    _validate_runtime(config)
    return config


def resolve_mode_config(config: Mapping[str, Any], mode: str) -> dict[str, Any]:
    if mode not in {"profile", "formal"}:
        raise WriterModelError("unsupported dynamic-K Writer runtime mode")
    if mode == "formal" and config["formal_run"]["status"] != "sealed":
        raise WriterModelError("formal dynamic-K training awaits a sealed live profile")
    return dict(config)
