from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ember.expert_manifold.v6_prior_deployment_seal as seal_module
from ember.expert_manifold.v6_prior_contract import (
    V6_PRIOR_CANONICAL_CONFIG,
)


def _config() -> dict:
    return json.loads(V6_PRIOR_CANONICAL_CONFIG.read_text(encoding="utf-8"))


def _storage() -> dict:
    return {
        "tensor_count": 76,
        "tensor_bytes": 2_641_920,
        "dtype_tensor_counts": {"BF16": 72, "F32": 4},
    }


def _adapter(config: dict) -> dict:
    return {
        "schema_version": ("ember_pi05_v6_condition_program_residual_eval_adapter_v8"),
        "kind": "expert_manifold_writer",
        "arm": "expert_manifold_v6_condition_residual_correct",
        "video_condition": "correct",
        "video_schedule": {"sampling_mode": "without_replacement"},
        "config": {"schema": config["schema_version"]},
        "writer_asset": {
            "architecture": config["writer"]["architecture"],
            "program_residual_value_count": 20_971_520,
            "deployment_trainable_parameter_count": 0,
            "generated_lora_tensor_count": 76,
            "method_macro": 0,
            "residual_state": {
                "kind": "fresh_elementwise_zero",
                "bytes": 0,
                "tensor_count": 0,
            },
            "writer_state": {"template_lora_storage": _storage()},
        },
        "evaluation_authority": {
            "throughput_policy": config["evaluation"]["throughput_policy"]
        },
        "information_wall": {
            "video_is_only_dynamic_value": True,
            "language_only_lora_path": False,
            "deployment_expert_bank_read": False,
            "teacher_action_reads": 0,
            "teacher_state_reads": 0,
            "reward_reads": 0,
            "terminal_reads": 0,
        },
    }


def _git() -> dict:
    commit = "a" * 40
    return {
        "branch": "codex/bci-continuation",
        "commit": commit,
        "upstream": "origin/codex/bci-continuation",
        "upstream_commit": commit,
        "authority_ref": "origin/codex/bci-continuation",
        "authority_contains_commit": True,
        "dirty_paths": [],
    }


def _tasks(states: list[int]) -> list[dict]:
    return [
        {"suite": f"suite_{index}", "task_id": index, "init_state_ids": states}
        for index in range(8)
    ]


def _requests(run: dict) -> tuple[SimpleNamespace, ...]:
    result = []
    for task in run["tasks"]:
        for state in task["init_state_ids"]:
            suite = task["suite"]
            task_id = task["task_id"]
            result.append(
                SimpleNamespace(
                    suite=suite,
                    task_id=task_id,
                    init_state_id=state,
                    entry_id=f"{suite}_task_{task_id:02d}_state_{state:03d}",
                )
            )
    return tuple(result)


def test_vertical_smoke_is_required_and_bound_to_selected_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    adapter = _adapter(config)
    git = _git()
    profile_run = {"git": git, "adapter": adapter, "tasks": _tasks([0, 1, 2, 3])}
    vertical_run = {
        "schema_version": "ember_pi05_target_eval_launch_v2",
        "mode": "smoke",
        "role": "validation",
        "content_hash_policy": "disabled_by_owner",
        "contract_reference": "vertical-contract",
        "git": git,
        "adapter": adapter,
        "tasks": _tasks([0]),
        "parallel": {
            "physical_gpu_count": 1,
            "physical_gpu_ids": [2],
            "replicas_per_gpu": 1,
            "writer_generators_per_gpu": 1,
            "writer_generation_batch_size": 16,
        },
    }
    requests = _requests(vertical_run)
    descriptor = {"lora_storage_per_entry": _storage()}
    vertical_run["writer_lora_cache"] = descriptor
    rows = [
        {
            "suite": request.suite,
            "task_id": request.task_id,
            "init_state_id": request.init_state_id,
            "success": False,
            "steps": 1,
            "writer": {"sealed": True},
        }
        for request in requests
    ]
    results = {
        "schema_version": "ember_pi05_target_eval_results_v2",
        "contract_reference": "vertical-contract",
        "adapter": adapter,
        "overall": {"episodes": 8},
        "rows": rows,
        "workers": [
            {
                "worker_id": "2-r0",
                "gpu_name": "NVIDIA A40",
                "source_policy_reloaded": False,
            }
        ],
        "launcher_attempts": {"attempts": [{"event": "completed"}]},
        "launcher": {
            "return_codes": {"2-r0": 0},
            "preflight": {
                "physical_gpu_ids": [2],
                "compute_applications": [],
                "device_names": ["NVIDIA A40"],
            },
        },
        "writer_generation": {
            "generator_workers": 1,
            "assigned_entries": 8,
            "generated_entries": 8,
            "reused_entries": 0,
            "generated_batches": 1,
            "generation_batch_size": [16],
            "max_observed_forward_batch_size": 8,
            "redundant_writer_forwards": 0,
            "batch_shape_bf16_roundoff_accepted": True,
            "all_source_policy_processes_reused_for_rollout": True,
            "all_writer_modules_released": True,
            "all_source_policies_not_reloaded": True,
            "gpu_names": ["NVIDIA A40"],
        },
    }
    manifest = {
        "descriptor": descriptor,
        "entry_ids": [request.entry_id for request in requests],
    }
    monkeypatch.setattr(seal_module, "writer_cache_requests", _requests)
    monkeypatch.setattr(
        seal_module, "validate_writer_episode", lambda *_args, **_kwargs: True
    )
    assert seal_module._vertical_matches(
        config, profile_run, vertical_run, results, manifest, 16
    )
    results["writer_generation"]["generated_entries"] = 7
    assert not seal_module._vertical_matches(
        config, profile_run, vertical_run, results, manifest, 16
    )


def test_assembler_reads_both_roots_and_records_raw_artifact_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_root = tmp_path / "profile"
    vertical_root = tmp_path / "vertical"
    cache_root = vertical_root / "cache"
    profile_root.mkdir()
    vertical_root.mkdir()
    cache_root.mkdir()
    profile = {"selected_writer_model_batch_size": 8}
    for root, name, value in (
        (profile_root, "run_contract.json", {"git": {"commit": "b" * 40}}),
        (profile_root, "writer_generation_profile.json", profile),
        (vertical_root, "run_contract.json", {}),
        (vertical_root, "results.json", {}),
    ):
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    manifest_path = cache_root / "cache_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(seal_module, "_profile_matches", lambda *_args: True)
    monkeypatch.setattr(seal_module, "_vertical_matches", lambda *_args: True)
    monkeypatch.setattr(
        seal_module, "validate_writer_cache_manifest", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        seal_module, "writer_cache_manifest_path", lambda _run: manifest_path
    )
    evidence = seal_module.assemble_v6_prior_evaluation_smoke_evidence(
        config=_config(),
        profile_root=profile_root,
        vertical_root=vertical_root,
        repo_root=tmp_path,
    )
    assert evidence["profile"]["path"] == "profile/writer_generation_profile.json"
    assert evidence["vertical"]["path"] == "vertical/results.json"
    assert evidence["cache_manifest"]["path"] == "vertical/cache/cache_manifest.json"
    assert evidence["run_commit"] == "b" * 40
    (vertical_root / "results.json").unlink()
    with pytest.raises(OSError):
        seal_module.assemble_v6_prior_evaluation_smoke_evidence(
            config=_config(),
            profile_root=profile_root,
            vertical_root=vertical_root,
            repo_root=tmp_path,
        )
