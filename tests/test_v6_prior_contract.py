from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior_contract import (
    _gradient_profile_evidence_matches,
    _resume_profile_evidence_matches,
    assemble_v6_prior_gradient_profile_evidence,
    load_v6_prior_config,
    runtime_for_mode,
)
from ember.expert_manifold.v6_prior_runtime import RuntimeSegment, _run_contract
from ember.pi05_source_checkpoint import DistributedContext


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pi05_v6_prior_policy_effective_writer_v1.json"


def _git_evidence(commit: str) -> dict:
    return {
        "branch": "codex/frozen",
        "commit": commit,
        "origin_main": "main-commit",
        "upstream": "origin/codex/frozen",
        "upstream_commit": commit,
        "dirty_paths": [],
    }


def _topology_evidence() -> list[dict]:
    rows = []
    for rank in range(6):
        host_index, local_rank = divmod(rank, 3)
        host = f"gpu0{host_index + 1}"
        numa = min(local_rank, 1)
        rows.append(
            {
                "rank": rank,
                "local_rank": local_rank,
                "host": host,
                "cuda_visible_devices": "0,1,2",
                "device_name": "NVIDIA A40",
                "physical_gpu": local_rank,
                "device": f"cuda:{local_rank}",
                "numa_node": numa,
                "cpu_affinity": [0, 1] if numa == 0 else [2, 3],
            }
        )
    return rows


def _gradient_evidence() -> dict:
    norms = {
        "positive": {
            "compiler": 2.0,
            "factor_heads": 4.0,
            "global": math.sqrt(20.0),
        },
        "expert": {
            "compiler": 8.0,
            "factor_heads": 1.0,
            "global": math.sqrt(65.0),
        },
        "ranking": {
            "compiler": 1.0,
            "factor_heads": 8.0,
            "global": math.sqrt(65.0),
        },
    }
    return {
        "schema_version": ("ember_pi05_v6_prior_gradient_profile_artifact_evidence_v1"),
        "root": "/retained/gradient",
        "git": _git_evidence("gradient-commit"),
        "config_schema": "ember_pi05_v6_prior_policy_effective_writer_v1",
        "config_bytes": 1234,
        "world_size": 6,
        "tasks_per_rank": 4,
        "rank_topology": _topology_evidence(),
        "schedule_start_macro": 49,
        "schedule_stop_macro": 50,
        "completed_diagnostic_macros": 1,
        "task_count": 24,
        "action_queries_per_task": 20,
        "total_action_queries": 480,
        "unique_action_queries": 480,
        "counterfactual_counts": {
            "reversed": 8,
            "shuffled": 8,
            "wrong": 8,
        },
        "longest_correct_sampled_frames": 105,
        "unweighted_gradient_norms": norms,
        "maximum_auxiliary_fraction": 0.25,
        "recommended_weights": {"expert": 0.0625, "ranking": 0.125},
        "applied_gradient_fractions": {
            "expert": {"compiler": 0.25, "factor_heads": 0.015625},
            "ranking": {"compiler": 0.0625, "factor_heads": 0.25},
        },
        "seal_rule": (
            "each_auxiliary_at_most_one_quarter_positive_gradient_in_both_"
            "compiler_and_factor_heads"
        ),
        "initialization": {
            "mode": "historical_v6_macro400_load_only",
            "optimizer": "fresh",
            "scheduler": "fresh",
            "rng": "fresh_seed",
        },
        "expert_bank": {"step": 2000, "task_count": 24},
        "ownership": {
            "frozen_parameter_count": 7_060_992,
            "trainable_parameter_count": 3_714_304,
            "trainable_tensor_count": 41,
            "source_policy_trainable_parameter_count": 0,
        },
        "method_verified": True,
        "information_wall_verified": True,
        "invocation_count": 1,
        "step_seconds": 120.0,
        "max_cuda_allocated_bytes": 10_000,
        "max_cuda_reserved_bytes": 12_000,
        "oom_count": 0,
        "nonfinite_count": 0,
        "content_hash_policy": "disabled_by_owner",
    }


def _checkpoint_comparison(macro: int) -> dict:
    return {
        "macro": macro,
        "cursor_semantic_equal": True,
        "checkpoint_contract_semantic_equal": True,
        "rng_rank_count": 6,
        "rng_semantic_equal": True,
        "scheduler_semantic_equal": True,
        "amp_semantic_equal": True,
        "optimizer": {
            "param_groups_equal": True,
            "scientific_atol": 0.0002,
            "scientific_rtol": 0.002,
            "max_abs_tolerance": 0.0000075,
            "global_relative_l2_tolerance": 0.00001,
            "tensor_count": 82,
            "max_abs": 0.0,
            "global_relative_l2": 0.0,
            "worst_tensor": None,
            "moment_fields": {
                name: {
                    "tensor_count": 41,
                    "max_abs": 0.0,
                    "global_relative_l2": 0.0,
                    "worst_tensor": None,
                }
                for name in ("exp_avg", "exp_avg_sq")
            },
        },
        "writer": {
            "tensor_schema_equal": True,
            "frozen_exact": True,
            "frozen_tensor_count": 559,
            "trainable_tensor_count": 41,
            "scientific_atol": 0.0002,
            "scientific_rtol": 0.002,
            "max_abs_tolerance": 0.0000075,
            "global_relative_l2_tolerance": 0.00001,
            "tensor_count": 600,
            "max_abs": 0.0,
            "global_relative_l2": 0.0,
            "worst_tensor": None,
        },
    }


def _resume_evidence(gradient: dict) -> dict:
    comparisons = [_checkpoint_comparison(1), _checkpoint_comparison(3)]
    return {
        "schema_version": ("ember_pi05_v6_prior_resume_profile_artifact_evidence_v1"),
        "gradient_root": gradient["root"],
        "resumed_root": "/retained/resumed",
        "contiguous_root": "/retained/contiguous",
        "gradient_commit": gradient["git"]["commit"],
        "profile_git": _git_evidence("profile-commit"),
        "config_schema": "ember_pi05_v6_prior_policy_effective_writer_v1",
        "config_bytes": 2345,
        "auxiliary_weights": dict(gradient["recommended_weights"]),
        "world_size": 6,
        "tasks_per_rank": 4,
        "rank_topology": _topology_evidence(),
        "invocation_counts": {"resumed": 2, "contiguous": 1},
        "metrics_rows": {"resumed": 3, "contiguous": 3},
        "checkpoint_macros": [1, 3],
        "scientific_tolerances": {
            "scientific_atol": 0.0002,
            "scientific_rtol": 0.002,
            "writer_max_abs": 0.0000075,
            "writer_relative_l2": 0.00001,
        },
        "run_contracts_equal": True,
        "scientific_metrics_equivalent": True,
        "checkpoint_semantics_equivalent": True,
        "checkpoint_comparisons": comparisons,
        "metric_max_abs_difference": 0.0,
        "metric_max_relative_difference": 0.0,
        "writer_max_abs_difference": 0.0,
        "writer_relative_l2_difference": 0.0,
        "step_seconds": {"resumed": 30.0, "contiguous": 30.0},
        "macros_per_second": {"resumed": 0.1, "contiguous": 0.1},
        "max_cuda_allocated_bytes": 10_000,
        "max_cuda_reserved_bytes": 12_000,
        "oom_count": 0,
        "nonfinite_count": 0,
        "content_hash_policy": "disabled_by_owner",
    }


def _task_record(ordinal: int, *, task_visit: int) -> dict:
    kind = ("reversed", "shuffled", "wrong")[ordinal % 3]
    return {
        "task_ordinal": ordinal,
        "global_task_id": ordinal,
        "suite": ("libero_spatial", "libero_object")[ordinal % 2],
        "task_id": ordinal,
        "task_visit": task_visit,
        "teacher_demo": (ordinal + task_visit) % 50,
        "counterfactual_kind": kind,
        "counterfactual_global_task_id": 100 + ordinal if kind == "wrong" else None,
        "counterfactual_demo": ordinal % 50 if kind == "wrong" else None,
        "functional_loss": 0.5,
        "expert_loss": 0.25,
        "expert_direction": 0.1,
        "expert_log_norm": 0.2,
        "ranking_loss": 0.3,
        "ranking_margin": 0.05,
        "correct_expert_cosine": 0.6,
        "counterfactual_expert_cosine": 0.4,
        "correct_effective_norm": 2.0,
        "counterfactual_effective_norm": 1.5,
        "expert_effective_norm": 2.5,
        "correct_raw_frames": 521 if ordinal == 0 else 101,
        "correct_sampled_frames": 105 if ordinal == 0 else 21,
        "counterfactual_raw_frames": 101,
        "counterfactual_sampled_frames": 21,
    }


def _synthetic_run_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
    monkeypatch.setenv("NCCL_ALGO", "Ring")
    monkeypatch.setenv("NCCL_PROTO", "Simple")
    tasks = tuple(
        SimpleNamespace(
            ordinal=ordinal,
            global_task_id=ordinal,
            suite=("libero_spatial", "libero_object")[ordinal % 2],
            task_id=ordinal,
            language=f"task {ordinal}",
            authority=SimpleNamespace(
                path=tmp_path / f"task-{ordinal}.hdf5",
                expected_bytes=1000 + ordinal,
            ),
        )
        for ordinal in range(24)
    )
    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_runtime.git_state",
        lambda _root: _git_evidence("gradient-commit"),
    )
    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_runtime._rank_topology",
        lambda _context: _topology_evidence(),
    )
    monkeypatch.setattr(
        "ember.expert_manifold.v6_prior_runtime.torch.cuda.get_device_name",
        lambda _device: "NVIDIA A40",
    )
    consumed = {
        "query": {
            "start_step": 49,
            "stop_step": 50,
            "global_examples": 480,
            "unique_query_rows": 480,
            "min_examples_per_task": 20,
            "max_examples_per_task": 20,
            "identity_evidence": "cursor_counts_and_dataset_row_coverage",
        },
        "teacher_video_seed": config["data"]["teacher_video_seed"],
        "videos_per_task_visit": 1,
        "min_video_visits_per_task": 1,
        "max_video_visits_per_task": 1,
        "min_unique_videos_per_task": 1,
        "max_unique_videos_per_task": 1,
    }
    return _run_contract(
        args=SimpleNamespace(
            mode="gradient-profile",
            config=CONFIG,
            expert_bank_root=tmp_path / "experts",
            data_root=tmp_path / "data",
            num_workers=2,
        ),
        config=config,
        context=DistributedContext(0, 0, 6, torch.device("cuda:0")),
        segment=RuntimeSegment(1, (), 0, 1, 49, 50),
        source={"checkpoint": "source"},
        tokenizer={"path": "tokenizer"},
        tasks=tasks,
        sampler=object(),
        video_schedule=SimpleNamespace(
            consumed_identity_summary=lambda *_args: consumed
        ),
        expert={
            "training_commit": "81101fe",
            "tasks": [{"task_ordinal": ordinal} for ordinal in range(24)],
        },
        warm_start=SimpleNamespace(
            checkpoint=tmp_path / "warm-start",
            state_tensor_count=600,
            state_value_count=12_064_064,
        ),
        ownership=SimpleNamespace(
            frozen_parameter_count=7_060_992,
            trainable_parameter_count=3_714_304,
            frozen_tensor_count=483,
            trainable_tensor_count=41,
        ),
        trainable_names=tuple(f"compiler.parameter_{index}" for index in range(41)),
    )


def test_v6_prior_config_unlocks_only_gradient_profile_after_online_smoke() -> None:
    config = load_v6_prior_config(CONFIG)
    assert runtime_for_mode(config, "gradient-profile") == (1, ())
    with pytest.raises(ExpertManifoldError, match="profile runtime is not sealed"):
        runtime_for_mode(config, "profile")
    with pytest.raises(ExpertManifoldError, match="formal runtime is not sealed"):
        runtime_for_mode(config, "formal")

    profiled = deepcopy(config)
    gradient = _gradient_evidence()
    profiled["gradient_profile"]["status"] = "sealed_from_live_train24_gradient_profile"
    profiled["gradient_profile"]["artifact_evidence"] = gradient
    profiled["objective"]["auxiliary_weights"].update(
        {
            "status": "sealed_from_live_train24_gradient_profile",
            **gradient["recommended_weights"],
        }
    )
    profiled["profile_run"]["status"] = "ready_after_live_gradient_profile"
    assert runtime_for_mode(profiled, "profile") == (3, (1, 3))
    with pytest.raises(ExpertManifoldError, match="formal runtime is not sealed"):
        runtime_for_mode(profiled, "formal")

    resumed = deepcopy(profiled)
    resumed["profile_run"]["status"] = "sealed_from_live_a40_resume_profile_evidence"
    resumed["profile_run"]["artifact_evidence"] = _resume_evidence(gradient)
    resumed["formal_run"]["status"] = "sealed_from_live_a40_resume_profile_evidence"
    assert runtime_for_mode(resumed, "formal") == (50, (10, 25, 50))


def test_v6_prior_profile_seals_fail_closed_on_status_or_weight_only() -> None:
    config = load_v6_prior_config(CONFIG)
    status_only = deepcopy(config)
    status_only["gradient_profile"][
        "status"
    ] = "sealed_from_live_train24_gradient_profile"
    status_only["objective"]["auxiliary_weights"].update(
        {
            "status": "sealed_from_live_train24_gradient_profile",
            "expert": 0.1,
            "ranking": 0.1,
        }
    )
    status_only["profile_run"]["status"] = "ready_after_live_gradient_profile"
    with pytest.raises(ExpertManifoldError, match="profile runtime is not sealed"):
        runtime_for_mode(status_only, "profile")

    gradient = _gradient_evidence()
    assert _gradient_profile_evidence_matches(gradient)
    wrong_weight = deepcopy(gradient)
    wrong_weight["recommended_weights"]["expert"] = 0.1
    assert not _gradient_profile_evidence_matches(wrong_weight)

    resume = _resume_evidence(gradient)
    assert _resume_profile_evidence_matches(resume)
    wrong_topology = deepcopy(resume)
    wrong_topology["rank_topology"][1]["physical_gpu"] = 0
    assert not _resume_profile_evidence_matches(wrong_topology)


def test_v6_prior_config_requires_artifact_lineage_for_each_unlock(
    tmp_path: Path,
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    gradient = _gradient_evidence()
    config["gradient_profile"].update(
        {
            "status": "sealed_from_live_train24_gradient_profile",
            "artifact_evidence": gradient,
        }
    )
    config["objective"]["auxiliary_weights"].update(
        {
            "status": "sealed_from_live_train24_gradient_profile",
            **gradient["recommended_weights"],
        }
    )
    config["profile_run"]["status"] = "ready_after_live_gradient_profile"
    path = tmp_path / "gradient-sealed.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert load_v6_prior_config(path)["profile_run"]["status"] == (
        "ready_after_live_gradient_profile"
    )

    config["objective"]["auxiliary_weights"]["expert"] = 0.1
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="scientific boundary"):
        load_v6_prior_config(path)

    config["objective"]["auxiliary_weights"]["expert"] = gradient[
        "recommended_weights"
    ]["expert"]
    config["profile_run"].update(
        {
            "status": "sealed_from_live_a40_resume_profile_evidence",
            "artifact_evidence": _resume_evidence(gradient),
        }
    )
    config["formal_run"]["status"] = "sealed_from_live_a40_resume_profile_evidence"
    path.write_text(json.dumps(config), encoding="utf-8")
    assert load_v6_prior_config(path)["formal_run"]["status"] == (
        "sealed_from_live_a40_resume_profile_evidence"
    )

    config["profile_run"]["artifact_evidence"]["gradient_commit"] = "wrong"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="scientific boundary"):
        load_v6_prior_config(path)


def test_gradient_seal_is_assembled_from_complete_retained_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "gradient-root"
    root.mkdir()
    contract = _synthetic_run_contract(monkeypatch, tmp_path)
    records = [_task_record(ordinal, task_visit=49) for ordinal in range(24)]
    gradient = _gradient_evidence()
    profile = {
        "schema_version": "ember_pi05_v6_prior_gradient_profile_seal_v1",
        "schedule_macro": 49,
        "task_count": 24,
        "action_queries_per_task": 20,
        "total_action_queries": 480,
        "unique_action_queries": 480,
        "counterfactual_counts": {
            "reversed": 8,
            "shuffled": 8,
            "wrong": 8,
        },
        "unweighted_gradient_norms": gradient["unweighted_gradient_norms"],
        "maximum_auxiliary_fraction": 0.25,
        "recommended_weights": gradient["recommended_weights"],
        "seal_rule": gradient["seal_rule"],
        "task_records": records,
        "step_seconds": 120.0,
        "max_cuda_allocated_bytes": 10_000,
        "max_cuda_reserved_bytes": 12_000,
        "oom_count": 0,
        "nonfinite_count": 0,
        "content_hash_policy": "disabled_by_owner",
    }
    completion = {
        "schema_version": "ember_pi05_v6_prior_writer_completion_v1",
        "mode": "gradient-profile",
        "completed_diagnostic_macros": 1,
        "schedule_start_macro": 49,
        "schedule_stop_macro": 50,
        "gradient_profile_complete": True,
        "oom_count": 0,
        "nonfinite_count": 0,
        "content_hash_policy": "disabled_by_owner",
    }
    (root / "run_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (root / "gradient_profile.json").write_text(json.dumps(profile), encoding="utf-8")
    (root / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
    (root / "invocations.jsonl").write_text(
        json.dumps(
            {
                "argv": ["train_v6_prior_writer.py", "--mode", "gradient-profile"],
                "started_unix": 1.0,
                "resume": None,
                "requested_stop_after_macro": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assembled = assemble_v6_prior_gradient_profile_evidence(root)
    assert assembled["recommended_weights"] == {
        "expert": 0.0625,
        "ranking": 0.125,
    }
    assert _gradient_profile_evidence_matches(assembled)

    profile["counterfactual_counts"]["wrong"] = 7
    (root / "gradient_profile.json").write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="evidence is incomplete"):
        assemble_v6_prior_gradient_profile_evidence(root)


def test_v6_prior_config_rejects_language_bypass_and_unprofiled_weights(
    tmp_path: Path,
) -> None:
    baseline = json.loads(CONFIG.read_text(encoding="utf-8"))
    bypass = deepcopy(baseline)
    bypass["method"]["language_only_lora_path"] = True
    bypass_path = tmp_path / "bypass.json"
    bypass_path.write_text(json.dumps(bypass), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="scientific boundary"):
        load_v6_prior_config(bypass_path)

    weights = deepcopy(baseline)
    weights["objective"]["auxiliary_weights"]["expert"] = 0.1
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(json.dumps(weights), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="scientific boundary"):
        load_v6_prior_config(weights_path)

    serial = deepcopy(baseline)
    serial["evaluation"]["minimum_smoke_writer_model_batch_size"] = 1
    batched_path = tmp_path / "serial.json"
    batched_path.write_text(json.dumps(serial), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="scientific boundary"):
        load_v6_prior_config(batched_path)
