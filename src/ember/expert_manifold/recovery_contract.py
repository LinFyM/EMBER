"""Scientific contract for composite-context recovery experts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ember.ecp.composite_recovery_data import CompositeRecoverySpec
from ember.pi05_source_checkpoint import read_json


REPO_ROOT = Path(__file__).resolve().parents[3]
RECOVERY_EXPERT_CONFIG_SCHEMA = "ember_pi05_ecp_composite_recovery_expert_v1"
COMPOSITE_DATASET_SCHEMA = "ember_ecp_composite_teacher_dataset_v1"


def _authority_path(config: Mapping[str, Any], name: str) -> Path:
    return REPO_ROOT / str(config["authorities"][name]["path"])


def _composite_rows(config: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    manifest = read_json(_authority_path(config, "composite_dataset_manifest"))
    recovery = config["task_experts"]["recovery"]
    first_phase = str(recovery["first_phase"])
    query_count = sum(
        int(row["steps"]) - int(row["completion_steps"][first_phase])
        for row in manifest["rows"]
    )
    return manifest, query_count


def _primitive_row(config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = read_json(_authority_path(config, "source_manifest"))
    task_id = int(config["task_experts"]["recovery"]["primitive_task_id"])
    rows = [
        row for row in manifest.get("tasks", []) if int(row["task_index"]) == task_id
    ]
    if len(rows) != 1:
        raise ValueError("recovery primitive authority is ambiguous")
    return rows[0]


def _data_matches(config: Mapping[str, Any]) -> bool:
    try:
        experts = config["task_experts"]
        recovery = experts["recovery"]
        manifest, composite_queries = _composite_rows(config)
        primitive = _primitive_row(config)
        required_order = list(experts["required_order"])
        return (
            manifest.get("schema_version") == COMPOSITE_DATASET_SCHEMA
            and manifest.get("status") == "completed_privileged_composite_bootstrap"
            and manifest.get("variant_name") == experts.get("variant_name")
            and manifest.get("required_order") == required_order
            and required_order
            == [recovery.get("first_phase"), recovery.get("phase_key")]
            and recovery.get("phase_language") == primitive.get("language")
            and int(experts["initialization"]["global_task_id"])
            == int(recovery["primitive_task_id"])
            and int(recovery.get("composite_success_episodes", -1))
            == int(manifest.get("episodes", -2))
            and int(recovery.get("composite_query_count", -1)) == composite_queries
            and int(recovery.get("primitive_query_count", -1))
            == int(primitive["demonstrations"]["steps"])
            and recovery.get("primitive_demo_indices") == [0, 49]
            and recovery.get("mix")
            == {"composite_second_phase": 0.5, "primitive_success": 0.5}
        )
    except (KeyError, TypeError, ValueError):
        return False


def recovery_expert_config_is_valid(config: Mapping[str, Any]) -> bool:
    experts = config.get("task_experts", {})
    formal = experts.get("formal_run", {})
    initialization = experts.get("initialization", {})
    optimization = experts.get("optimization", {})
    optimizer = optimization.get("optimizer", {})
    scheduler = optimization.get("scheduler", {})
    sampler = experts.get("sampler", {})
    profile = formal.get("profile_evidence", {})
    authorities = config.get("authorities", {})
    initial_path = REPO_ROOT / str(initialization.get("adapter", ""))
    try:
        assets_exist = all(
            _authority_path(config, name).is_file() for name in authorities
        )
    except (KeyError, TypeError):
        return False
    return (
        config.get("schema_version") == RECOVERY_EXPERT_CONFIG_SCHEMA
        and config.get("status") == "sealed_composite_context_recovery_training"
        and config.get("paired_contract_id") == "ecp_composite_context_recovery_v1"
        and set(authorities)
        == {
            "composite_dataset_manifest",
            "source_manifest",
            "process_manifest",
            "evaluation_config",
            "lora_contract",
            "source_base_config",
        }
        and assets_exist
        and _data_matches(config)
        and config.get("information_wall", {}).get("policy_condition")
        == "phase_specific_language_only"
        and int(config.get("information_wall", {}).get("target40_action_reads", -1))
        == 0
        and experts.get("objective")
        == "successful_second_phase_plus_primitive_flow_sft"
        and int(experts.get("task_count", -1)) == 1
        and int(experts.get("sampler_task_id", -1)) == 0
        and int(experts.get("action_chunk_size", -1)) == 50
        and int(experts.get("model_image_size", -1)) == 224
        and experts.get("mask_action_padding") is True
        and experts.get("lora_topology") == "configs/pi05_lora_v1.json:38targets:rank16"
        and experts.get("task_parameter_sharing") == "none"
        and sampler.get("kind") == "deterministic_balanced_two_domain"
        and sampler.get("batch_domain_rows")
        == {"composite_second_phase": 8, "primitive_success": 8}
        and initialization.get("kind")
        == "fixed_step1000_primitive_adapter_no_optimizer_reuse"
        and int(initialization.get("step", -1)) == 1000
        and initial_path.is_file()
        and initial_path.stat().st_size == int(initialization.get("adapter_bytes", -1))
        and optimization.get("precision") == "bfloat16"
        and optimization.get("gradient_checkpointing") is True
        and optimizer.get("name") == "AdamW"
        and optimizer.get("betas") == [0.9, 0.999]
        and float(optimizer.get("eps", -1)) == 1e-8
        and float(optimizer.get("weight_decay", -1)) == 1e-4
        and float(optimizer.get("gradient_clip_norm", -1)) == 1.0
        and scheduler.get("kind") == "cosine_decay_with_warmup"
        and float(scheduler.get("peak_lr", -1)) == 1e-5
        and int(scheduler.get("warmup_steps", -1)) == 25
        and float(scheduler.get("decay_lr", -1)) == 1e-7
        and int(optimization.get("seed", -1)) == 7
        and formal.get("status") == "sealed"
        and int(formal.get("total_steps", -1)) == 1000
        and int(formal.get("per_task_batch_size", -1)) == 16
        and formal.get("checkpoint_steps") == [1000]
        and formal.get("stage_stop_steps") == [1000]
        and int(formal.get("default_stop_step", -1)) == 1000
        and formal.get("checkpoint_policy")
        == "one_fixed_step1000_checkpoint_no_selection"
        and profile.get("device") == "NVIDIA A40"
        and int(profile.get("per_task_batch_size", -1)) == 16
        and profile.get("inherited_same_pi05_rank16_forward") is True
        and int(profile.get("max_cuda_reserved_bytes", -1)) > 0
        and int(profile.get("oom_count", -1)) == 0
        and int(profile.get("nonfinite_count", -1)) == 0
        and config.get("content_hash_policy") == "disabled_by_owner"
    )


def load_recovery_spec(
    config: Mapping[str, Any], data_root: Path
) -> CompositeRecoverySpec:
    if not recovery_expert_config_is_valid(config):
        raise ValueError("recovery expert config changed")
    experts = config["task_experts"]
    recovery = experts["recovery"]
    manifest, _ = _composite_rows(config)
    primitive = _primitive_row(config)
    composite_manifest = _authority_path(config, "composite_dataset_manifest")
    composite_path = composite_manifest.parent / str(manifest["hdf5"]["filename"])
    primitive_path = data_root / str(primitive["hdf5"]["filename"])
    initialization = experts["initialization"]
    initial_path = REPO_ROOT / str(initialization["adapter"])
    if (
        not composite_path.is_file()
        or composite_path.stat().st_size != int(manifest["hdf5"]["bytes"])
        or not primitive_path.is_file()
        or primitive_path.stat().st_size != int(primitive["hdf5"]["bytes"])
    ):
        raise ValueError("recovery expert data authority changed")
    first_phase = str(recovery["first_phase"])
    segments = tuple(
        (
            int(row["demo_index"]),
            int(row["completion_steps"][first_phase]),
            int(row["steps"]),
        )
        for row in manifest["rows"]
    )
    return CompositeRecoverySpec(
        variant_name=str(experts["variant_name"]),
        phase_key=str(recovery["phase_key"]),
        language=str(recovery["phase_language"]),
        composite_path=composite_path,
        composite_bytes=int(manifest["hdf5"]["bytes"]),
        composite_segments=segments,
        primitive_path=primitive_path,
        primitive_bytes=int(primitive["hdf5"]["bytes"]),
        primitive_demo_indices=tuple(range(50)),
        composite_query_count=int(recovery["composite_query_count"]),
        primitive_query_count=int(recovery["primitive_query_count"]),
        model_image_size=int(experts["model_image_size"]),
        initial_adapter_path=initial_path,
        initial_adapter_bytes=int(initialization["adapter_bytes"]),
    )
