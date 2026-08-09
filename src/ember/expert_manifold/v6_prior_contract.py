"""Canonical config and state gates for the frozen-v6 Program residual."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_policy_batch import (
    LOGICAL_POLICY_BATCH_SIZE,
    POSITIVE_POLICY_RANDOMNESS,
)
from ember.pi05_eval_contract import git_state_is_clean_pushed_or_frozen_authority


REPO_ROOT = Path(__file__).resolve().parents[3]
V6_PRIOR_CANONICAL_CONFIG = (
    REPO_ROOT
    / "configs/pi05_v6_counterfactual_null_condition_kernel_program_residual_v1.json"
)
V6_PRIOR_CONFIG_SCHEMA = (
    "ember_pi05_v6_counterfactual_null_condition_kernel_program_residual_v1"
)
V6_PRIOR_RUN_SCHEMA = (
    "ember_pi05_v6_counterfactual_null_condition_kernel_program_residual_launch_v1"
)
V6_PRIOR_PROFILE_SCHEMA = (
    "ember_pi05_v6_counterfactual_null_condition_kernel_program_residual_profile_v1"
)
V6_PRIOR_COMPLETION_SCHEMA = (
    "ember_pi05_v6_counterfactual_null_condition_kernel_program_residual_completion_v1"
)
V6_PRIOR_MODES = ("mechanism-profile", "formal")
_WRITER_GENERATION_PROFILE_SCHEMA = "ember_pi05_writer_generation_profile_v1"
_EVALUATION_RUN_SCHEMA = "ember_pi05_target_eval_launch_v2"
_ACTIVE_AUTHORITY_REF = "origin/codex/bci-continuation"

_EXPECTED_WRITER = {
    "architecture": "frozen_pi05_v6_plus_video_keyed_program_residual_v1",
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

_EXPECTED_CONDITION_FEATURE = {
    "kind": "zero_preserving_temporal_visual_innovation_jl_v1",
    "input": "mean_valid_task_tokens_frame_evidence_minus_text_queries",
    "temporal_basis": ["one", "tau", "cos_pi_tau", "sin_pi_tau"],
    "tau_range": [-1.0, 1.0],
    "descriptor_width": 1024,
    "feature_width": 256,
    "projection_seed": 20260810,
    "projection": "fixed_no_bias_row_normalized_fp32_nonpersistent",
    "normalization": "zero_preserving_l2",
    "frame_content_reordered_but_display_ordinals_fixed": True,
}

_EXPECTED_PROGRAM_RESIDUAL = {
    "program_slots": 320,
    "program_width": 256,
    "feature_width": 256,
    "value_count": 20_971_520,
    "dtype": "float32",
    "initialization": "elementwise_zero",
    "fusion": "single_add_before_frozen_historical_factor_heads",
    "checkpoint_tensor_count": 1,
}

_EXPECTED_UPDATE = {
    "kind": "full48_counterfactual_null_explicit_condition_kernel",
    "correct_conditions": 24,
    "negative_conditions": 24,
    "ordering": "correct_task_ordinal_0_to_23_then_negative_task_ordinal_0_to_23",
    "negative_schedule": (
        "task_ordinal_plus_task_visit_modulo_reversed_shuffled_wrong"
    ),
    "negative_counts_per_macro": {"reversed": 8, "shuffled": 8, "wrong": 8},
    "negative_rhs": "exact_zero_program_motion",
    "step_size": 1.0,
    "relative_damping": 0.01,
    "small_solve_dtype": "float64",
    "large_rhs_and_memory_write_dtype": "float32",
    "optimizer": "none_manual_add",
    "momentum": False,
    "weight_decay": False,
    "gradient_clip": False,
    "global_scale_or_cap": False,
}

_EXPECTED_DATA = {
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

_EXPECTED_OBJECTIVE = {
    "name": "correct_condition_policy_functional_flow_loss_only",
    "positive_policy_randomness": POSITIVE_POLICY_RANDOMNESS,
}

_EXPECTED_OPTIMIZATION = {
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


def authority_path(config: Mapping[str, Any], name: str) -> Path:
    try:
        row = config["authorities"][name]
        path = (REPO_ROOT / str(row["path"])).resolve()
    except (KeyError, TypeError, ValueError) as error:
        raise ExpertManifoldError(f"missing residual Writer authority: {name}") from error
    if not path.is_file():
        raise ExpertManifoldError(f"residual Writer authority is missing: {name}")
    return path


def _writer_matches(value: Mapping[str, Any]) -> bool:
    return value == _EXPECTED_WRITER


def _feature_and_update_match(config: Mapping[str, Any]) -> bool:
    return (
        config.get("condition_feature") == _EXPECTED_CONDITION_FEATURE
        and config.get("program_residual") == _EXPECTED_PROGRAM_RESIDUAL
        and config.get("update") == _EXPECTED_UPDATE
    )


def _data_and_runtime_match(config: Mapping[str, Any]) -> bool:
    return (
        config.get("data") == _EXPECTED_DATA
        and config.get("objective") == _EXPECTED_OBJECTIVE
        and config.get("optimization") == _EXPECTED_OPTIMIZATION
    )


_PROFILE_CHECKS = {
    "feature_rank",
    "correct_motion_retained",
    "counterfactual_null",
    "predicted_observed_closure",
    "production_wall_overhead",
    "lora_a_response",
    "lora_b_response",
    "fixed_action_response",
    "fixed_action_breadth",
    "task_local_motion_evidence",
    "functional_policy_program_credit",
    "negative_policy_forwards",
    "oom_and_nonfinite",
}


def _is_ancestor(left: str, right: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", left, right],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    ).returncode == 0


def git_commit_in_active_authority_lineage(commit: str) -> bool:
    """Accept only commits shared by this checkout and the active remote authority."""

    if (
        len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        return False
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        authority = subprocess.run(
            ["git", "rev-parse", _ACTIVE_AUTHORITY_REF],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return False
    return (
        _is_ancestor(commit, head)
        and _is_ancestor(commit, authority)
        and (_is_ancestor(head, authority) or _is_ancestor(authority, head))
    )


def _profile_result_matches(
    config: Mapping[str, Any],
    result: Mapping[str, Any],
) -> bool:
    macro = result.get("macro")
    if not isinstance(macro, Mapping):
        return False
    try:
        from ember.expert_manifold.v6_prior_profile import profile_passes

        passed, gate_evidence = profile_passes(config, macro)
    except (ExpertManifoldError, KeyError, TypeError, ValueError, OverflowError):
        return False
    expected = {
        "schema_version": V6_PRIOR_PROFILE_SCHEMA,
        "passed": True,
        "schedule_macro": 49,
        "retain_weight": False,
        "gates": config.get("profile_run", {}).get("gates"),
        "content_hash_policy": "disabled_by_owner",
    }
    return (
        passed is True
        and all(result.get(name) == value for name, value in expected.items())
        and result.get("gate_evidence") == gate_evidence
        and gate_evidence.get("checks")
        == {name: True for name in _PROFILE_CHECKS}
    )


_RUN_KEYS = {
    "schema_version",
    "mode",
    "git",
    "config",
    "source",
    "tokenizer",
    "initialization",
    "data",
    "method",
    "information_wall",
    "writer",
    "condition_feature",
    "program_residual",
    "update",
    "objective",
    "optimization",
    "ownership",
    "runtime",
    "content_hash_policy",
}

_EXPECTED_OWNERSHIP = {
    "historical_v6_base": {
        "state_tensor_count": 600,
        "parameter_tensor_count": 523,
        "parameter_count": 10_775_296,
        "trainable_parameter_count": 0,
        "checkpoint_owned": False,
        "deployment_owned": True,
    },
    "fixed_projection": {
        "shape": [256, 1024],
        "dtype": "torch.float32",
        "trainable": False,
        "persistent": False,
        "checkpoint_owned": False,
    },
    "program_residual_memory": {
        "shape": [256, 320, 256],
        "dtype": "torch.float32",
        "value_count": 20_971_520,
        "trainable": False,
        "manual_update": True,
        "checkpoint_owned": True,
        "deployment_owned": True,
    },
    "source_policy_trainable_parameter_count": 0,
    "optimizer": "not_instantiated",
    "scheduler": "not_instantiated",
    "scaler": "not_instantiated",
}


def _source_identity(value: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        return {
            "optimizer_step": int(value["optimizer_step"]),
            "source_training_commit": str(value["source_training_commit"]),
            "model_files": [
                {"path": str(row["path"]), "bytes": int(row["bytes"])}
                for row in value["model_files"]
            ],
        }
    except (KeyError, TypeError, ValueError):
        return None


def _source_and_tokenizer_match(
    config: Mapping[str, Any],
    source: object,
    tokenizer: object,
) -> bool:
    if not isinstance(source, Mapping) or not isinstance(tokenizer, Mapping):
        return False
    initialization = (
        REPO_ROOT / str(config["initialization"]["checkpoint"])
    ).resolve()
    try:
        historical = json.loads(
            (initialization.parent.parent / "run_contract.json").read_text(
                encoding="utf-8"
            )
        )
        tokenizer_path = Path(str(tokenizer["path"])).resolve()
        tokenizer_bytes = int(tokenizer["bytes"])
        manifest_name = Path(str(tokenizer["manifest_path"])).name
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    return (
        _source_identity(source) == _source_identity(historical.get("source", {}))
        and set(tokenizer) == {"path", "bytes", "manifest_path"}
        and tokenizer_bytes == 4_264_023
        and tokenizer_path.is_file()
        and tokenizer_path.stat().st_size == tokenizer_bytes
        and manifest_name == "pi05_tokenizer_manifest.json"
    )


def _run_data_matches(
    config: Mapping[str, Any],
    data: object,
    *,
    start: int,
    stop: int,
) -> bool:
    if not isinstance(data, Mapping):
        return False
    expected_keys = set(config["data"]) | {"root", "tasks", "consumed_schedule"}
    if set(data) != expected_keys or any(
        data.get(name) != value for name, value in config["data"].items()
    ):
        return False
    try:
        from ember.expert_manifold.contract import (
            load_task_expert_config,
            load_train_tasks,
        )

        expert = load_task_expert_config(authority_path(config, "task_expert_config"))
        tasks = load_train_tasks(expert, Path(str(data["root"])).resolve())
        expected_tasks = [
            {
                "ordinal": task.ordinal,
                "global_task_id": task.global_task_id,
                "suite": task.suite,
                "task_id": task.task_id,
                "language": task.language,
                "path": str(task.authority.path),
                "bytes": task.authority.expected_bytes,
            }
            for task in tasks
        ]
        consumed = data["consumed_schedule"]
        query = consumed["query"]
        visits = stop - start
        global_examples = visits * 24 * LOGICAL_POLICY_BATCH_SIZE
        unique_rows = int(query["unique_query_rows"])
    except (ExpertManifoldError, KeyError, OSError, TypeError, ValueError):
        return False
    expected_query = {
        "start_step": start,
        "stop_step": stop,
        "global_examples": global_examples,
        "min_examples_per_task": visits * LOGICAL_POLICY_BATCH_SIZE,
        "max_examples_per_task": visits * LOGICAL_POLICY_BATCH_SIZE,
        "identity_evidence": "cursor_counts_and_dataset_row_coverage",
    }
    expected_consumed = {
        "teacher_video_seed": int(config["data"]["teacher_video_seed"]),
        "videos_per_task_visit": 1,
        "min_video_visits_per_task": visits,
        "max_video_visits_per_task": visits,
        "min_unique_videos_per_task": visits,
        "max_unique_videos_per_task": visits,
    }
    return (
        data["tasks"] == expected_tasks
        and isinstance(consumed, Mapping)
        and set(consumed) == set(expected_consumed) | {"query"}
        and all(consumed.get(name) == value for name, value in expected_consumed.items())
        and isinstance(query, Mapping)
        and set(query) == set(expected_query) | {"unique_query_rows"}
        and all(query.get(name) == value for name, value in expected_query.items())
        and 0 < unique_rows <= global_examples
    )


def _run_science_matches(
    config: Mapping[str, Any],
    run: Mapping[str, Any],
    commit: str,
    *,
    mode: str,
    schedule_start: int,
    schedule_stop: int,
) -> bool:
    git = run.get("git", {})
    record = run.get("config", {})
    initialization = run.get("initialization", {})
    configured_writer = (
        REPO_ROOT / str(config["initialization"]["checkpoint"]) / "writer.safetensors"
    ).resolve()
    expected_initialization = {
        "mode": "strict_historical_v6_macro400_all_frozen",
        "checkpoint": str(configured_writer),
        "writer_state_tensor_count": 600,
        "writer_state_value_count": 12_064_064,
        "residual_memory": "fresh_zero_then_memory_only_exact_resume",
    }
    fixed = {
        "schema_version": V6_PRIOR_RUN_SCHEMA,
        "mode": mode,
        "method": config.get("method"),
        "information_wall": config.get("information_wall"),
        "writer": config.get("writer"),
        "condition_feature": config.get("condition_feature"),
        "program_residual": config.get("program_residual"),
        "update": config.get("update"),
        "objective": config.get("objective"),
        "optimization": config.get("optimization"),
        "ownership": _EXPECTED_OWNERSHIP,
        "content_hash_policy": "disabled_by_owner",
    }
    try:
        config_bytes = int(record.get("bytes", -1))
    except (AttributeError, TypeError, ValueError):
        return False
    return (
        set(run) == _RUN_KEYS
        and all(run.get(name) == value for name, value in fixed.items())
        and isinstance(git, Mapping)
        and set(git) == {"branch", "commit", "authority_ref", "dirty_paths"}
        and git.get("commit") == commit
        and git.get("authority_ref") == _ACTIVE_AUTHORITY_REF
        and git.get("dirty_paths") == []
        and isinstance(record, Mapping)
        and set(record) == {"path", "schema", "bytes"}
        and Path(str(record.get("path", ""))).name
        == V6_PRIOR_CANONICAL_CONFIG.name
        and record.get("schema") == V6_PRIOR_CONFIG_SCHEMA
        and config_bytes > 0
        and initialization == expected_initialization
        and _source_and_tokenizer_match(
            config, run.get("source"), run.get("tokenizer")
        )
        and _run_data_matches(
            config,
            run.get("data"),
            start=schedule_start,
            stop=schedule_stop,
        )
    )


def _runtime_matches(
    runtime: object,
    *,
    total_macros: int,
    schedule_origin: int,
    checkpoint_macros: list[int],
) -> bool:
    if not isinstance(runtime, Mapping):
        return False
    topology = runtime.get("rank_topology", [])
    expected = {
        "world_size": 6,
        "tasks_per_rank": 4,
        "total_macros": total_macros,
        "schedule_origin": schedule_origin,
        "checkpoint_macros": checkpoint_macros,
        "num_workers_per_rank": 2,
        "action_loader_prefetch_factor": 2,
        "action_loader_persistent_workers": True,
        "logical_policy_batch_size": 20,
        "functional_policy_microbatch_size": 10,
        "physical_policy_forwards_per_task": 2,
        "negative_policy_forwards_per_task": 0,
        "policy_gradient_checkpointing": False,
        "writer_activation_checkpointing_effective": False,
        "distributed_model_wrapper": "none",
        "collectives": "two_all_gathers_no_memory_allreduce",
        "deferred_process_group": True,
        "nccl_p2p_disable": "1",
        "nccl_algo": "Ring",
        "nccl_proto": "Simple",
    }
    dynamic = {"host", "device", "cuda_visible_devices", "rank_topology", "cuda_allocator_conf_observed"}
    return (
        set(runtime) == set(expected) | dynamic
        and all(runtime.get(name) == value for name, value in expected.items())
        and bool(runtime.get("host"))
        and runtime.get("device") == "NVIDIA A40"
        and bool(runtime.get("cuda_visible_devices"))
        and isinstance(topology, list)
        and len(topology) == 6
        and sorted(row.get("rank") for row in topology) == list(range(6))
        and all(row.get("device_name") == "NVIDIA A40" for row in topology)
        and len(
            {(row.get("host"), row.get("physical_gpu")) for row in topology}
        )
        == 6
        and all(isinstance(row.get("numa_node"), int) for row in topology)
        and all(bool(row.get("cpu_affinity")) for row in topology)
    )


def _profile_run_matches(
    config: Mapping[str, Any],
    run: Mapping[str, Any],
    commit: str,
) -> bool:
    return _run_science_matches(
        config,
        run,
        commit,
        mode="mechanism-profile",
        schedule_start=49,
        schedule_stop=50,
    ) and _runtime_matches(
        run.get("runtime"),
        total_macros=1,
        schedule_origin=49,
        checkpoint_macros=[],
    )


def _profile_completion_matches(completion: Mapping[str, Any]) -> bool:
    return completion == {
        "schema_version": V6_PRIOR_COMPLETION_SCHEMA,
        "mode": "mechanism-profile",
        "completed_diagnostic_macros": 1,
        "passed": True,
        "retained_checkpoint": False,
        "content_hash_policy": "disabled_by_owner",
    }


def _profile_artifact_matches(config: Mapping[str, Any]) -> bool:
    profile = config.get("profile_run", {})
    if profile.get("status") == "awaiting_live_a40_macro49_profile":
        return profile.get("artifact_evidence") is None
    artifact = profile.get("artifact_evidence")
    required_fields = {
        "path",
        "bytes",
        "schema",
        "passed",
        "schedule_macro",
        "run_commit",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != required_fields:
        return False
    try:
        path = (REPO_ROOT / str(artifact["path"])).resolve()
        result = json.loads(path.read_text(encoding="utf-8"))
        run = json.loads(
            (path.parent / "run_contract.json").read_text(encoding="utf-8")
        )
        completion = json.loads(
            (path.parent / "completion.json").read_text(encoding="utf-8")
        )
        commit = str(artifact["run_commit"])
        evidence = {
            "schema": V6_PRIOR_PROFILE_SCHEMA,
            "passed": True,
            "schedule_macro": 49,
        }
        return (
            path.stat().st_size == int(artifact["bytes"])
            and all(artifact.get(name) == value for name, value in evidence.items())
            and bool(commit)
            and _profile_result_matches(config, result)
            and _profile_run_matches(config, run, commit)
            and _profile_completion_matches(completion)
            and git_commit_in_active_authority_lineage(commit)
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False

def _throughput_baseline_matches(config: Mapping[str, Any]) -> bool:
    baseline = config.get("profile_run", {}).get("throughput_baseline", {})
    expected = {
        "path": (
            "runs/outputs/pi05_v6_prior_gradient_profile_macro49_r6_lb20_"
            "mb10_9c814ff_20260809/gradient_profile.json"
        ),
        "schema": "ember_pi05_v6_prior_gradient_profile_seal_v1",
        "schedule_macro": 49,
        "task_count": 24,
        "action_queries_per_task": 20,
        "step_seconds": 21.09510959603358,
    }
    if baseline != expected:
        return False
    try:
        observed = json.loads(
            (REPO_ROOT / expected["path"]).read_text(encoding="utf-8")
        )
        return (
            observed.get("schema_version") == expected["schema"]
            and all(
                observed.get(name) == expected[name]
                for name in (
                    "schedule_macro",
                    "task_count",
                    "action_queries_per_task",
                    "step_seconds",
                )
            )
        )
    except (OSError, json.JSONDecodeError):
        return False


def _stable_generation_rows(
    rows: object,
) -> list[Mapping[str, Any]]:
    if not isinstance(rows, list):
        return []
    return [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("stable") is True
        and math.isfinite(float(row.get("loras_per_second", math.nan)))
        and float(row["loras_per_second"]) > 0
    ]


def _evaluation_run_matches(
    config: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    run: Mapping[str, Any],
    commit: str,
) -> bool:
    adapter = run.get("adapter", {})
    git = run.get("git", {})
    parallel = run.get("parallel", {})
    writer_asset = adapter.get("writer_asset", {})
    required_sizes = evaluation.get("required_writer_model_batch_sizes", [])
    expected_adapter = {
        "schema_version": (
            "ember_pi05_v6_condition_program_residual_eval_adapter_v8"
        ),
        "kind": "expert_manifold_writer",
        "arm": "expert_manifold_v6_condition_residual_correct",
        "video_condition": "correct",
    }
    expected_parallel = {
        "physical_gpu_count": 1,
        "replicas_per_gpu": 1,
        "writer_generators_per_gpu": 1,
        "writer_generation_batch_size": max(required_sizes),
    }
    return (
        run.get("schema_version") == _EVALUATION_RUN_SCHEMA
        and run.get("mode") == "smoke"
        and run.get("role") == "validation"
        and run.get("content_hash_policy") == "disabled_by_owner"
        and git.get("commit") == commit
        and git_state_is_clean_pushed_or_frozen_authority(git)
        and all(
            adapter.get(name) == value
            for name, value in expected_adapter.items()
        )
        and adapter.get("config", {}).get("schema") == V6_PRIOR_CONFIG_SCHEMA
        and writer_asset.get("architecture")
        == config.get("writer", {}).get("architecture")
        and writer_asset.get("program_residual_value_count")
        == config.get("program_residual", {}).get("value_count")
        and writer_asset.get("deployment_trainable_parameter_count") == 0
        and writer_asset.get("generated_lora_tensor_count") == 76
        and adapter.get("evaluation_authority", {}).get("throughput_policy")
        == evaluation.get("throughput_policy")
        and adapter.get("information_wall", {}).get(
            "video_is_only_dynamic_value"
        )
        is True
        and all(
            parallel.get(name) == value
            for name, value in expected_parallel.items()
        )
    )


def _generation_profile_matches(
    evaluation: Mapping[str, Any],
    evidence: Mapping[str, Any],
    result: Mapping[str, Any],
    run: Mapping[str, Any],
) -> bool:
    rows = result.get("writer_generation_measurements")
    git = result.get("git", {})
    stable = _stable_generation_rows(rows)
    if not stable or not isinstance(rows, list):
        return False
    selected = max(
        stable,
        key=lambda row: (
            float(row["loras_per_second"]),
            int(row["batch_size"]),
        ),
    )
    required_sizes = evaluation.get("required_writer_model_batch_sizes")
    selected_batch = int(evidence["writer_model_batch_size"])
    expected = {
        "schema_version": _WRITER_GENERATION_PROFILE_SCHEMA,
        "contract_reference": run.get("contract_reference"),
        "device": "NVIDIA A40",
        "profiled_writer_model_batch_sizes": required_sizes,
        "selected_writer_model_batch_size": selected_batch,
        "selection_rule": (
            "highest_measured_fixed_panel_loras_per_second_with_stable_"
            "longest_video_batch"
        ),
        "writer_modules_released": True,
        "source_policy_reused": True,
        "content_hash_policy": "disabled_by_owner",
    }
    zero_fields = (
        "teacher_action_reads",
        "teacher_state_reads",
        "reward_reads",
        "terminal_reads",
        "oom_count",
        "nonfinite_count",
    )
    return (
        all(result.get(name) == value for name, value in expected.items())
        and git.get("commit") == evidence["run_commit"]
        and git_state_is_clean_pushed_or_frozen_authority(git)
        and sorted(row.get("batch_size") for row in rows) == required_sizes
        and selected.get("batch_size") == selected_batch
        and selected_batch
        >= int(evaluation.get("minimum_smoke_writer_model_batch_size", -1))
        and all(result.get(name) == 0 for name in zero_fields)
    )


def _evaluation_artifact_matches(config: Mapping[str, Any]) -> bool:
    evaluation = config.get("evaluation", {})
    status = evaluation.get("formal_status")
    evidence = evaluation.get("online_smoke_evidence")
    if status == "blocked_until_new_residual_deployment_graph_live_profile":
        return evidence is None
    if status != "sealed_from_live_residual_deployment_profile":
        return False
    required_fields = {
        "path",
        "bytes",
        "schema",
        "run_commit",
        "writer_model_batch_size",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != required_fields:
        return False
    try:
        path = (REPO_ROOT / str(evidence["path"])).resolve()
        result = json.loads(path.read_text(encoding="utf-8"))
        run = json.loads(
            (path.parent / "run_contract.json").read_text(encoding="utf-8")
        )
        commit = str(evidence["run_commit"])
        return (
            path.stat().st_size == int(evidence["bytes"])
            and evidence["schema"] == _WRITER_GENERATION_PROFILE_SCHEMA
            and bool(commit)
            and _evaluation_run_matches(config, evaluation, run, commit)
            and _generation_profile_matches(evaluation, evidence, result, run)
            and git_commit_in_active_authority_lineage(commit)
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False

_COHERENT_STATES = {
    (
        "active_implementation_awaiting_cpu_and_live_a40_profile",
        "awaiting_live_a40_macro49_profile",
        "blocked_until_live_profile_passes_and_is_sealed",
        "blocked_until_new_residual_deployment_graph_live_profile",
    ),
    (
        "active_profile_sealed_formal_ready",
        "sealed_from_live_a40_macro49_profile",
        "ready_after_live_profile_seal",
        "blocked_until_new_residual_deployment_graph_live_profile",
    ),
    (
        "active_profile_sealed_formal_ready",
        "sealed_from_live_a40_macro49_profile",
        "ready_after_live_profile_seal",
        "sealed_from_live_residual_deployment_profile",
    ),
    (
        "formal_result_sealed",
        "sealed_from_live_a40_macro49_profile",
        "formal_result_sealed",
        "blocked_until_new_residual_deployment_graph_live_profile",
    ),
    (
        "formal_result_sealed",
        "sealed_from_live_a40_macro49_profile",
        "formal_result_sealed",
        "sealed_from_live_residual_deployment_profile",
    ),
}

_EXPECTED_PROFILE_STATIC = {
    "expected_world_size": 6,
    "tasks_per_rank": 4,
    "schedule_macro": 49,
    "diagnostic_macros": 1,
    "num_workers_per_rank": 2,
    "retain_weight": False,
}

_EXPECTED_PROFILE_GATES = {
    "feature_rank_min": 24,
    "correct_motion_to_cotangent_rms_min": 0.25,
    "negative_to_correct_motion_rms_max": 0.25,
    "predicted_observed_relative_rms_max": 0.005,
    "production_wall_ratio_max": 1.1,
    "fixed_action_response_rms_min": 0.0,
    "fixed_action_probe_task_count": 4,
    "fixed_action_passing_task_count_min": 4,
    "correct_retained_task_count_min": 18,
    "negative_null_task_count_min": 18,
    "extra_negative_policy_forwards": 0,
    "oom_count": 0,
    "nonfinite_count": 0,
}

_EXPECTED_FORMAL_STATIC = {
    "expected_world_size": 6,
    "tasks_per_rank": 4,
    "num_workers_per_rank": 2,
    "total_macros": 50,
    "checkpoint_macros": [10, 25, 50],
    "strict400_checkpoints": [0, 10, 25, 50],
    "decision_gates": {
        "macro10_broad_loss_stop_correct_max": 129,
        "macro10_conditional_continue_correct_range": [130, 134],
        "macro10_conditional_continue_requires_breadth_churn_and_mechanics": True,
        "macro10_continue_correct_min": 135,
        "first_full_six_arm_correct_min": 144,
        "goal_full_six_arm_correct_min": 151,
        "macro25_or_50_retire_if_not_above_baseline_correct": 134,
    },
}


def _formal_completion_matches(completion: Mapping[str, Any]) -> bool:
    return completion == {
        "schema_version": V6_PRIOR_COMPLETION_SCHEMA,
        "mode": "formal",
        "completed_macro": 50,
        "metrics_rows": 50,
        "content_hash_policy": "disabled_by_owner",
    }


def _formal_metrics_match(path: Path) -> bool:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return False
    return (
        len(rows) == 50
        and all(isinstance(row, Mapping) for row in rows)
        and [row.get("macro") for row in rows] == list(range(50))
        and [row.get("schedule_macro") for row in rows] == list(range(50))
    )


def _formal_run_matches(
    config: Mapping[str, Any],
    run: Mapping[str, Any],
    commit: str,
) -> bool:
    return _run_science_matches(
        config,
        run,
        commit,
        mode="formal",
        schedule_start=0,
        schedule_stop=50,
    ) and _runtime_matches(
        run.get("runtime"),
        total_macros=50,
        schedule_origin=0,
        checkpoint_macros=[10, 25, 50],
    )


def _formal_artifact_matches(config: Mapping[str, Any]) -> bool:
    formal = config.get("formal_run", {})
    status = formal.get("status")
    evidence = formal.get("artifact_evidence")
    if status in {
        "blocked_until_live_profile_passes_and_is_sealed",
        "ready_after_live_profile_seal",
    }:
        return evidence is None
    if status != "formal_result_sealed" or not isinstance(evidence, Mapping):
        return False
    required = {
        "path",
        "bytes",
        "schema",
        "run_commit",
        "metrics_bytes",
        "checkpoint_manifests",
    }
    if set(evidence) != required:
        return False
    try:
        completion_relative = Path(str(evidence["path"]))
        if completion_relative.is_absolute():
            return False
        completion_path = (REPO_ROOT / completion_relative).resolve()
        root = completion_path.parent
        if completion_path.name != "completion.json":
            return False
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        run = json.loads((root / "run_contract.json").read_text(encoding="utf-8"))
        metrics_path = root / "metrics.jsonl"
        commit = str(evidence["run_commit"])
        manifests = evidence["checkpoint_manifests"]
        if not isinstance(manifests, list):
            return False
        from ember.expert_manifold.v6_prior_checkpoint import (
            V6_PRIOR_CHECKPOINT_SCHEMA,
            inspect_v6_prior_checkpoint,
        )
        from ember.expert_manifold.v6_prior_run_contract import (
            checkpoint_contract,
            cursor_contract,
        )

        expected_checkpoint_contract = checkpoint_contract(run)
        checkpoint_rows = []
        for macro, row in zip((10, 25, 50), manifests, strict=True):
            if not isinstance(row, Mapping):
                return False
            relative = Path(str(row.get("path", "")))
            manifest_path = (REPO_ROOT / relative).resolve()
            expected_path = (
                root / "checkpoints" / f"macro_{macro:08d}" / "manifest.json"
            ).resolve()
            if (
                relative.is_absolute()
                or set(row) != {"macro", "path", "bytes", "schema"}
                or row.get("macro") != macro
                or row.get("schema") != V6_PRIOR_CHECKPOINT_SCHEMA
                or manifest_path != expected_path
                or manifest_path.stat().st_size != int(row.get("bytes", -1))
            ):
                return False
            inspection = inspect_v6_prior_checkpoint(
                manifest_path.parent,
                expected_cursor_contract=cursor_contract(config, macro),
                expected_checkpoint_contract=expected_checkpoint_contract,
                validate_payload_values=False,
            )
            checkpoint_rows.append(int(inspection["next_macro"]))
        completion_bytes = completion_path.stat().st_size
        metrics_bytes = metrics_path.stat().st_size
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        ExpertManifoldError,
    ):
        return False
    return (
        completion_bytes == int(evidence["bytes"])
        and evidence["schema"] == V6_PRIOR_COMPLETION_SCHEMA
        and metrics_bytes == int(evidence["metrics_bytes"])
        and checkpoint_rows == [10, 25, 50]
        and _formal_completion_matches(completion)
        and _formal_metrics_match(metrics_path)
        and _formal_run_matches(config, run, commit)
        and git_commit_in_active_authority_lineage(commit)
    )

_EXPECTED_EVALUATION_STATIC = {
    "throughput_policy": (
        "highest_measured_batch_throughput_with_device_memory_headroom"
    ),
    "required_writer_model_batch_sizes": [8, 16, 32],
    "minimum_smoke_writer_model_batch_size": 8,
}


def _projection_matches(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    return {name: value.get(name) for name in expected} == expected


def _profile_state_matches(config: Mapping[str, Any]) -> bool:
    profile = config.get("profile_run", {})
    return (
        profile.get("status")
        in {
            "awaiting_live_a40_macro49_profile",
            "sealed_from_live_a40_macro49_profile",
        }
        and _projection_matches(profile, _EXPECTED_PROFILE_STATIC)
        and profile.get("gates") == _EXPECTED_PROFILE_GATES
        and _throughput_baseline_matches(config)
        and _profile_artifact_matches(config)
    )


def _formal_state_matches(config: Mapping[str, Any]) -> bool:
    formal = config.get("formal_run", {})
    return (
        formal.get("status")
        in {
            "blocked_until_live_profile_passes_and_is_sealed",
            "ready_after_live_profile_seal",
            "formal_result_sealed",
        }
        and _projection_matches(formal, _EXPECTED_FORMAL_STATIC)
        and _formal_artifact_matches(config)
    )


def _evaluation_state_matches(config: Mapping[str, Any]) -> bool:
    evaluation = config.get("evaluation", {})
    return (
        evaluation.get("formal_status")
        in {
            "blocked_until_new_residual_deployment_graph_live_profile",
            "sealed_from_live_residual_deployment_profile",
        }
        and _projection_matches(evaluation, _EXPECTED_EVALUATION_STATIC)
        and _evaluation_artifact_matches(config)
    )


def _state_machine_matches(config: Mapping[str, Any]) -> bool:
    profile = config.get("profile_run", {})
    formal = config.get("formal_run", {})
    evaluation = config.get("evaluation", {})
    state = (
        config.get("status"),
        profile.get("status"),
        formal.get("status"),
        evaluation.get("formal_status"),
    )
    return (
        state in _COHERENT_STATES
        and _profile_state_matches(config)
        and _formal_state_matches(config)
        and _evaluation_state_matches(config)
    )


_EXPECTED_METHOD = {
    "name": "frozen_v6_counterfactual_null_condition_kernel_program_residual",
    "writer_input": (
        "exact task language plus exactly one action-hidden teacher video"
    ),
    "dynamic_value": "one_raw_teacher_video_only",
    "language_only_lora_path": False,
    "deployment_expert_bank_read": False,
    "deployment_output": "one complete 38-target rank16 public LoRA",
}

_EXPECTED_AUTHORITIES = {
    "task_expert_config": {"path": "configs/pi05_video_expert_manifold_v1.json"},
    "target_data_manifest": {
        "path": "configs/pi05_target_data_v1/manifest.json"
    },
    "evaluation_config": {"path": "configs/pi05_target_evaluation_v1.json"},
    "lora_contract": {"path": "configs/pi05_lora_v1.json"},
    "source_base_config": {"path": "configs/pi05_source_base_v1.json"},
}

_EXPECTED_INITIALIZATION = {
    "kind": "strict_load_and_freeze_historical_v6_fast_macro400",
    "checkpoint": (
        "runs/outputs/pi05_as_writer_v6_decay400_taskcomplete_dev_r4_b20_"
        "seed7_s2400_4efa737_20260729/checkpoints/step_00000400"
    ),
    "writer_state_tensor_count": 600,
    "writer_parameter_count": 10_775_296,
    "residual_memory": "elementwise_zero_on_fresh_or_memory_only_exact_resume",
    "optimizer": "not_instantiated",
    "scheduler": "not_instantiated",
    "scaler": "not_instantiated",
}

_EXPECTED_INFORMATION_WALL = {
    "writer_video_split_roles": ["train", "validation", "test"],
    "writer_forbidden_inputs": [
        "action",
        "proprio",
        "state",
        "reward",
        "terminal",
        "task_id",
        "filename",
        "object_pose",
        "hidden_normalization",
        "policy_outcome",
    ],
    "source_actions_enter_only": "correct_condition_functional_loss",
    "negative_action_forwards": 0,
    "validation_actions_read": 0,
    "test_actions_read": 0,
}


def _base_config_matches(config: Mapping[str, Any]) -> bool:
    return (
        config.get("schema_version") == V6_PRIOR_CONFIG_SCHEMA
        and config.get("status") in {state[0] for state in _COHERENT_STATES}
        and config.get("method") == _EXPECTED_METHOD
        and config.get("authorities") == _EXPECTED_AUTHORITIES
        and config.get("initialization") == _EXPECTED_INITIALIZATION
        and config.get("information_wall") == _EXPECTED_INFORMATION_WALL
        and _writer_matches(config.get("writer", {}))
        and _feature_and_update_match(config)
        and _data_and_runtime_match(config)
        and _state_machine_matches(config)
    )


def load_v6_prior_config(path: Path) -> dict[str, Any]:
    """Load only the active residual family; retired Tangent configs fail closed."""

    path = path.resolve()
    if path != V6_PRIOR_CANONICAL_CONFIG.resolve() or not path.is_file():
        raise ExpertManifoldError("non-canonical residual Writer config")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExpertManifoldError("invalid residual Writer config JSON") from error
    if not _base_config_matches(config):
        raise ExpertManifoldError("residual Writer config violates its sealed contract")
    for name in config["authorities"]:
        authority_path(config, name)
    base = (REPO_ROOT / _EXPECTED_INITIALIZATION["checkpoint"]).resolve()
    if not base.is_dir() or not (base / "writer.safetensors").is_file():
        raise ExpertManifoldError("historical v6 initialization asset is missing")
    return config

def runtime_for_mode(
    config: Mapping[str, Any],
    mode: str,
) -> tuple[int, tuple[int, ...], int]:
    """Return total, checkpoints, and schedule start for one active mode."""

    if mode == "mechanism-profile":
        profile = config["profile_run"]
        if profile["status"] != "awaiting_live_a40_macro49_profile":
            raise ExpertManifoldError("mechanism profile is not in its launch state")
        return 1, (), int(profile["schedule_macro"])
    if mode == "formal":
        profile = config["profile_run"]
        formal = config["formal_run"]
        if (
            profile["status"] != "sealed_from_live_a40_macro49_profile"
            or not isinstance(profile["artifact_evidence"], Mapping)
            or formal["status"] != "ready_after_live_profile_seal"
        ):
            raise ExpertManifoldError("formal residual training is blocked by profile state")
        return (
            int(formal["total_macros"]),
            tuple(int(value) for value in formal["checkpoint_macros"]),
            0,
        )
    raise ExpertManifoldError("unsupported residual Writer mode")
