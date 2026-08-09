import copy
import json
from pathlib import Path

import pytest

from ember.expert_manifold.contract import (
    ExpertManifoldError,
    load_task_expert_config,
)
from ember.expert_manifold.inference import inspect_v6_prior_writer_asset
from ember.expert_manifold.v6_prior_contract import (
    REPO_ROOT,
    assemble_v6_prior_evaluation_smoke_evidence,
    load_v6_prior_config,
)
from ember.pi05_source_checkpoint import read_json


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/pi05_v6_prior_policy_effective_writer_v1.json"


def _smoke_evidence() -> dict:
    panel_entry_ids = [f"entry-{index}" for index in range(32)]
    panel_sampled_frames = [105] * 32
    return {
        "commit": "clean-pushed",
        "root": "runs/outputs/smoke",
        "device": "NVIDIA A40",
        "checkpoint_kind": "historical_v6_macro400_load_only",
        "video_condition": "correct",
        "video_sampling": "without_replacement",
        "validation_task_count": 8,
        "state_count": 1,
        "scientific_rows": 8,
        "generated_entries": 8,
        "cache_entries": 8,
        "writer_state_tensor_count": 600,
        "writer_model_batch_size": 32,
        "profiled_writer_model_batch_sizes": [8, 16, 32],
        "writer_generation_measurements": [
            {
                "batch_size": batch_size,
                "generated_entries": 32 * 2,
                "max_observed_forward_batch_size": batch_size,
                "forward_batch_sizes_per_repeat": [batch_size] * (32 // batch_size),
                "wall_seconds": 32.0 / batch_size * 4.0,
                "loras_per_second": float(32 * 2) / (32.0 / batch_size * 4.0),
                "repeat_wall_seconds": [16.0 / batch_size * 4.0] * 2,
                "peak_allocated_bytes": batch_size * 100,
                "peak_reserved_bytes": batch_size * 120,
                "device_total_bytes": 10_000,
                "memory_headroom_bytes": 10_000 - batch_size * 120,
                "required_memory_headroom_bytes": 1_000,
                "comparison_panel_shared_across_candidates": True,
                "panel_entry_count": 32,
                "panel_total_sampled_frames": sum(panel_sampled_frames),
                "longest_video_included": True,
                "max_sampled_video_frames": 105,
                "sampled_frame_counts": list(panel_sampled_frames),
                "entry_ids": list(panel_entry_ids),
                "stable": True,
            }
            for batch_size in (8, 16, 32)
        ],
        "throughput_comparison_panel": (
            "same_fixed_longest_first_request_panel_all_candidates"
        ),
        "writer_modules_released": True,
        "source_policy_reused_for_rollout": True,
        "source_policy_reloaded": False,
        "batch_shape_bf16_roundoff_accepted": True,
        "redundant_writer_forwards": 0,
        "writer_lora_storage": "template_native_mixed_bfloat16_float32",
        "writer_lora_tensor_bytes_per_entry": 2_641_920,
        "writer_lora_bfloat16_tensor_count": 72,
        "writer_lora_float32_tensor_count": 4,
        "generator_workers": 1,
        "max_peak_allocated_bytes": 3200,
        "max_peak_reserved_bytes": 3840,
        "max_post_release_allocated_bytes": 100,
        "max_post_release_reserved_bytes": 120,
        "throughput_selection_rule": (
            "highest_measured_fixed_panel_loras_per_second_with_stable_"
            "longest_video_batch"
        ),
        "retry_count": 0,
        "failure_count": 0,
        "teacher_action_reads": 0,
        "teacher_state_reads": 0,
        "reward_reads": 0,
        "terminal_reads": 0,
        "oom_count": 0,
        "nonfinite_count": 0,
        "success_interpretation": "execution_smoke_only_not_performance_evidence",
    }


def test_v6_prior_evaluation_is_sealed_from_live_smoke_artifacts() -> None:
    evaluation = load_v6_prior_config(CONFIG)["evaluation"]
    assert evaluation["throughput_policy"] == (
        "highest_measured_throughput_with_device_memory_headroom"
    )
    assert evaluation["minimum_smoke_writer_model_batch_size"] == 8
    assert evaluation["formal_status"] == "sealed"
    evidence = evaluation["online_smoke_evidence"]
    assert evidence["writer_model_batch_size"] == 8
    assert evidence["profiled_writer_model_batch_sizes"] == [8, 16, 32]
    assert evidence["scientific_rows"] == 8
    assert evidence["generated_entries"] == 8
    assert evidence["retry_count"] == evidence["failure_count"] == 0


def test_formal_seal_accepts_only_complete_live_smoke_evidence(
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    config["evaluation"] = {
        "throughput_policy": "highest_measured_throughput_with_device_memory_headroom",
        "minimum_smoke_writer_model_batch_size": 8,
        "formal_status": "sealed",
        "online_smoke_evidence": _smoke_evidence(),
    }
    config["gradient_profile"]["status"] = (
        "ready_after_cpu_and_single_a40_throughput_smoke"
    )
    path = tmp_path / "sealed.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert load_v6_prior_config(path)["evaluation"]["formal_status"] == "sealed"

    del config["evaluation"]["online_smoke_evidence"]["writer_modules_released"]
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="scientific boundary changed"):
        load_v6_prior_config(path)


def test_throughput_seal_selects_fastest_stable_candidate(
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    smoke = _smoke_evidence()
    smoke["writer_model_batch_size"] = 16
    largest = smoke["writer_generation_measurements"][-1]
    largest["required_memory_headroom_bytes"] = 7_000
    largest["stable"] = False
    config["evaluation"] = {
        "throughput_policy": "highest_measured_throughput_with_device_memory_headroom",
        "minimum_smoke_writer_model_batch_size": 8,
        "formal_status": "sealed",
        "online_smoke_evidence": smoke,
    }
    config["gradient_profile"]["status"] = (
        "ready_after_cpu_and_single_a40_throughput_smoke"
    )
    path = tmp_path / "stable-selection.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert load_v6_prior_config(path)["evaluation"]["formal_status"] == "sealed"

    largest["stable"] = True
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="scientific boundary changed"):
        load_v6_prior_config(path)


def test_throughput_seal_breaks_equal_throughput_ties_toward_larger_batch(
    tmp_path: Path,
) -> None:
    config = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    smoke = _smoke_evidence()
    largest = smoke["writer_generation_measurements"][-1]
    largest["wall_seconds"] = 8.0
    largest["repeat_wall_seconds"] = [4.0, 4.0]
    largest["loras_per_second"] = 8.0
    config["evaluation"] = {
        "throughput_policy": "highest_measured_throughput_with_device_memory_headroom",
        "minimum_smoke_writer_model_batch_size": 8,
        "formal_status": "sealed",
        "online_smoke_evidence": smoke,
    }
    config["gradient_profile"]["status"] = (
        "ready_after_cpu_and_single_a40_throughput_smoke"
    )
    path = tmp_path / "stable-tie.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert load_v6_prior_config(path)["evaluation"]["formal_status"] == "sealed"

    smoke["writer_model_batch_size"] = 16
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="scientific boundary changed"):
        load_v6_prior_config(path)


def test_old_expert_asset_config_cannot_enter_canonical_runtime() -> None:
    old = REPO_ROOT / "configs/pi05_video_expert_manifold_v1.json"
    with pytest.raises(ExpertManifoldError, match="scientific boundary changed"):
        load_v6_prior_config(old)


def test_v6_task_expert_authority_ignores_retired_writer_seals(
    tmp_path: Path,
) -> None:
    old = REPO_ROOT / "configs/pi05_video_expert_manifold_v1.json"
    config = json.loads(old.read_text(encoding="utf-8"))
    config["topological_writer"] = {"retired": True}
    config["meta_training"] = {"retired": True}
    path = tmp_path / "task_experts.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert load_task_expert_config(path)["task_experts"]["task_count"] == 24

    config["task_experts"]["formal_run"]["selected_stop_step"] = 1000
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="task-expert scientific boundary"):
        load_task_expert_config(path)


def test_live_smoke_evidence_is_assembled_from_profile_and_vertical_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ember.writer.evaluation_cache as evaluation_cache

    profile_root = tmp_path / "profile"
    vertical_root = tmp_path / "vertical"
    profile_root.mkdir()
    vertical_root.mkdir()
    git = {
        "branch": "codex/test",
        "commit": "clean-pushed",
        "origin_main": "main",
        "upstream": "origin/codex/test",
        "upstream_commit": "clean-pushed",
        "dirty_paths": [],
    }
    adapter = {
        "kind": "expert_manifold_writer",
        "video_condition": "correct",
        "video_schedule": {"sampling_mode": "without_replacement"},
        "writer_asset": {
            "kind": "historical_v6_macro400_load_only",
            "writer_state": {"state_tensor_count": 600},
        },
    }
    profile_tasks = [
        {"suite": f"suite_{index}", "task_id": index, "init_state_ids": [0, 1, 2, 3]}
        for index in range(8)
    ]
    vertical_tasks = [
        {"suite": f"suite_{index}", "task_id": index, "init_state_ids": [0]}
        for index in range(8)
    ]
    profile_reference = "profile-contract"
    vertical_reference = "vertical-contract"
    profile_contract = {
        "contract_reference": profile_reference,
        "git": git,
        "adapter": adapter,
        "mode": "smoke",
        "role": "validation",
        "tasks": profile_tasks,
        "parallel": {
            "physical_gpu_count": 1,
            "physical_gpu_ids": [0],
            "replicas_per_gpu": 1,
            "writer_generators_per_gpu": 1,
            "writer_generation_batch_size": 32,
        },
    }
    smoke = _smoke_evidence()
    profile_entry_ids = [
        f"{task['suite']}_task_{int(task['task_id']):02d}_state_{state_id:03d}"
        for task in profile_tasks
        for state_id in task["init_state_ids"]
    ]
    for measurement in smoke["writer_generation_measurements"]:
        measurement["entry_ids"] = profile_entry_ids
    profile = {
        "schema_version": "ember_pi05_writer_generation_profile_v1",
        "contract_reference": profile_reference,
        "git": git,
        "root": str(profile_root),
        "device": "NVIDIA A40",
        "preflight": {
            "compute_applications": [],
            "device_names": ["NVIDIA A40"],
            "physical_gpu_ids": [0],
        },
        "profiled_writer_model_batch_sizes": smoke[
            "profiled_writer_model_batch_sizes"
        ],
        "selected_writer_model_batch_size": smoke["writer_model_batch_size"],
        "selection_rule": (
            "highest_measured_fixed_panel_loras_per_second_with_stable_"
            "longest_video_batch"
        ),
        "throughput_comparison_panel": (
            "same_fixed_longest_first_request_panel_all_candidates"
        ),
        "warmup_runs_per_batch": 1,
        "measured_runs_per_batch": 2,
        "longest_sampled_video_frames": 105,
        "writer_generation_measurements": smoke[
            "writer_generation_measurements"
        ],
        "writer_modules_released": True,
        "source_policy_reused": True,
        "oom_count": 0,
        "nonfinite_count": 0,
    }
    storage = {
        "tensor_count": 76,
        "tensor_bytes": 2_641_920,
        "dtype_tensor_counts": {"BF16": 72, "F32": 4},
        "dtype_by_name": {
            f"tensor_{index}": "BF16" if index < 72 else "F32"
            for index in range(76)
        },
    }
    vertical_descriptor = {"lora_storage_per_entry": storage}
    vertical_contract = {
        "contract_reference": vertical_reference,
        "git": git,
        "adapter": adapter,
        "mode": "smoke",
        "role": "validation",
        "tasks": vertical_tasks,
        "parallel": {
            "physical_gpu_ids": [0],
            "writer_generation_batch_size": 32,
        },
        "writer_lora_cache": vertical_descriptor,
    }
    worker_id = "0-r0"
    results = {
        "contract_reference": vertical_reference,
        "overall": {"episodes": 8},
        "rows": [{} for _ in range(8)],
        "workers": [
            {
                "worker_id": worker_id,
                "gpu_name": "NVIDIA A40",
                "source_policy_reloaded": False,
            }
        ],
        "launcher_attempts": {"attempts": [{"event": "completed"}]},
        "launcher": {
            "return_codes": {worker_id: 0},
            "preflight": {
                "compute_applications": [],
                "device_names": ["NVIDIA A40"],
                "physical_gpu_ids": [0],
            },
        },
        "writer_generation": {
            "generator_workers": 1,
            "assigned_entries": 8,
            "generated_entries": 8,
            "reused_entries": 0,
            "max_observed_forward_batch_size": 8,
            "generation_batch_size": [32],
            "redundant_writer_forwards": 0,
            "batch_shape_bf16_roundoff_accepted": True,
            "all_source_policy_processes_reused_for_rollout": True,
            "all_writer_modules_released": True,
            "all_source_policies_not_reloaded": True,
            "gpu_names": ["NVIDIA A40"],
            "max_peak_allocated_bytes": 3_200,
            "max_peak_reserved_bytes": 3_840,
            "max_post_release_allocated_bytes": 100,
            "max_post_release_reserved_bytes": 120,
        },
    }
    (profile_root / "run_contract.json").write_text(
        json.dumps(profile_contract), encoding="utf-8"
    )
    (profile_root / "writer_generation_profile.json").write_text(
        json.dumps(profile), encoding="utf-8"
    )
    (vertical_root / "run_contract.json").write_text(
        json.dumps(vertical_contract), encoding="utf-8"
    )
    (vertical_root / "results.json").write_text(
        json.dumps(results), encoding="utf-8"
    )
    manifest = {
        "entry_ids": [f"entry-{index}" for index in range(8)],
        "descriptor": vertical_descriptor,
    }
    monkeypatch.setattr(
        evaluation_cache,
        "validate_writer_cache_manifest",
        lambda *_args, **_kwargs: manifest,
    )
    evidence = assemble_v6_prior_evaluation_smoke_evidence(
        profile_root=profile_root,
        vertical_root=vertical_root,
    )
    assert evidence["writer_model_batch_size"] == 32
    assert evidence["generated_entries"] == 8

    profile["writer_generation_measurements"][0]["entry_ids"][0] = "outside-panel"
    (profile_root / "writer_generation_profile.json").write_text(
        json.dumps(profile), encoding="utf-8"
    )
    with pytest.raises(ExpertManifoldError, match="evidence is incomplete"):
        assemble_v6_prior_evaluation_smoke_evidence(
            profile_root=profile_root,
            vertical_root=vertical_root,
        )


def test_historical_v6_warm_start_is_a_real_load_only_evaluation_asset() -> None:
    config = load_v6_prior_config(CONFIG)
    checkpoint = (REPO_ROOT / config["initialization"]["checkpoint"]).resolve()
    historical_source = read_json(checkpoint.parent.parent / "run_contract.json")[
        "source"
    ]

    asset = inspect_v6_prior_writer_asset(
        config,
        checkpoint,
        historical_source,
        require_formal=False,
    )

    assert asset["kind"] == "historical_v6_macro400_load_only"
    assert asset["source_macro"] == 400
    assert asset["method_macro"] == 0
    assert asset["writer_state"]["state_tensor_count"] == 600
    assert asset["writer_state"]["state_value_count"] == 12_064_064
    storage = asset["writer_state"]["template_lora_storage"]
    assert storage["tensor_count"] == 76
    assert storage["parameter_count"] == 1_287_168
    assert storage["tensor_bytes"] == 2_641_920
    assert storage["dtype_tensor_counts"] == {"BF16": 72, "F32": 4}
    assert storage["dtype_parameter_counts"] == {
        "BF16": 1_253_376,
        "F32": 33_792,
    }
    assert len(storage["dtype_by_name"]) == 76
    assert storage["dtype_by_name"]["model.action_in_proj.lora_A.default.weight"] == "F32"
