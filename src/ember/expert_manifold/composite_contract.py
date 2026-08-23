"""Order-specific composite privileged expert training authority."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ember.pi05_source_checkpoint import read_json


REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSITE_EXPERT_CONFIG_SCHEMA = "ember_pi05_ecp_composite_teacher_expert_v1"
COMPOSITE_DATASET_SCHEMA = "ember_ecp_composite_teacher_dataset_v1"


@dataclass(frozen=True)
class CompositeExpertSpec:
    sampler_task_id: int
    variant_name: str
    language: str
    path: Path
    expected_bytes: int


def _authority_path(config: Mapping[str, Any], name: str) -> Path:
    return REPO_ROOT / str(config["authorities"][name]["path"])


def _dataset_matches(config: Mapping[str, Any]) -> bool:
    try:
        manifest = read_json(_authority_path(config, "composite_dataset_manifest"))
        experts = config["task_experts"]
        episodes = int(manifest["episodes"])
        return (
            manifest.get("schema_version") == COMPOSITE_DATASET_SCHEMA
            and manifest.get("status") == "completed_privileged_composite_bootstrap"
            and manifest.get("variant_name") == experts.get("variant_name")
            and manifest.get("required_order") == experts.get("required_order")
            and manifest.get("exact_language") == experts.get("exact_language")
            and episodes == int(experts.get("episodes_per_task", -1))
            and manifest.get("demo_indices") == experts.get("demo_indices")
            and manifest.get("demo_indices") == [0, episodes - 1]
            and len(manifest.get("source_state_ids", [])) == episodes
            and int(manifest.get("hdf5", {}).get("bytes", -1)) > 0
            and int(manifest.get("total_action_steps", -1)) > 0
            and manifest.get("replay")
            == {
                "render_resolution": 32,
                "matched_successes": episodes,
                "divergences": 0,
            }
            and manifest.get("information_wall", {}).get("target40_action_reads")
            == 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def _information_matches(information: Mapping[str, Any]) -> bool:
    return (
        information.get("expert_action_roles") == ["nonheld_process_bootstrap"]
        and int(information.get("target40_action_reads", -1)) == 0
        and information.get("deployment_uses_privileged_expert") is False
        and information.get("task_identity_role") == "sampler_ownership_only"
        and information.get("policy_condition")
        == "exact_unified_composite_language_only"
    )


def _expert_contract_matches(experts: Mapping[str, Any]) -> bool:
    return (
        int(experts.get("task_count", -1)) == 1
        and int(experts.get("sampler_task_id", -1)) == 0
        and int(experts.get("action_chunk_size", -1)) == 50
        and experts.get("lora_topology")
        == "configs/pi05_lora_v1.json:38targets:rank16"
        and experts.get("task_parameter_sharing") == "none"
        and experts.get("checkpoint_selection")
        == "fixed_step1000_for_both_variants_no_selection"
    )


def _formal_contract_matches(formal: Mapping[str, Any]) -> bool:
    profile = formal.get("profile_evidence", {})
    return (
        formal.get("status") == "sealed"
        and int(formal.get("total_steps", -1)) == 1000
        and int(formal.get("per_task_batch_size", -1)) == 16
        and formal.get("checkpoint_steps") == [1000]
        and int(formal.get("allowed_worker_count", -1)) == 1
        and int(formal.get("tasks_per_worker", -1)) == 1
        and int(formal.get("default_stop_step", -1)) == 1000
        and formal.get("stage_stop_steps") == [1000]
        and formal.get("checkpoint_policy")
        == "fixed_step1000_no_variant_specific_selection"
        and profile.get("device") == "NVIDIA A40"
        and int(profile.get("per_task_batch_size", -1)) == 16
        and profile.get("same_pi05_rank16_forward_as_existing_experts") is True
        and int(profile.get("composite_data_profile_steps", -1)) == 3
        and 0 < int(profile.get("max_cuda_allocated_bytes", -1))
        <= int(profile.get("max_cuda_reserved_bytes", -1))
        < 46_000_000_000
        and len(profile.get("steady_step_seconds", [])) == 2
        and all(float(value) > 0 for value in profile["steady_step_seconds"])
        and int(profile.get("oom_count", -1)) == 0
        and int(profile.get("nonfinite_count", -1)) == 0
    )


def composite_expert_config_is_valid(config: Mapping[str, Any]) -> bool:
    experts = config.get("task_experts", {})
    formal = experts.get("formal_run", {})
    authorities = config.get("authorities", {})
    return (
        config.get("schema_version") == COMPOSITE_EXPERT_CONFIG_SCHEMA
        and config.get("status")
        == "sealed_ecp_composite_teacher_expert_training_contract"
        and set(authorities)
        == {
            "composite_dataset_manifest",
            "process_manifest",
            "evaluation_config",
            "lora_contract",
            "source_base_config",
        }
        and all(_authority_path(config, name).is_file() for name in authorities)
        and _dataset_matches(config)
        and _information_matches(config.get("information_wall", {}))
        and _expert_contract_matches(experts)
        and _formal_contract_matches(formal)
        and config.get("content_hash_policy") == "disabled_by_owner"
    )


def load_composite_expert_spec(
    config: Mapping[str, Any], data_root: Path
) -> CompositeExpertSpec:
    manifest = read_json(_authority_path(config, "composite_dataset_manifest"))
    filename = str(manifest["hdf5"]["filename"])
    expected_bytes = int(manifest["hdf5"]["bytes"])
    path = data_root / filename
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise ValueError("composite expert HDF5 path or size changed")
    experts = config["task_experts"]
    return CompositeExpertSpec(
        sampler_task_id=int(experts["sampler_task_id"]),
        variant_name=str(experts["variant_name"]),
        language=str(experts["exact_language"]),
        path=path,
        expected_bytes=expected_bytes,
    )
