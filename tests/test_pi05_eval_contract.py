from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import (
    build_run_contract,
    inspect_installed_target_tasks,
    inspect_source_checkpoint,
    load_evaluation_authorities,
    load_run_contract,
    policy_noise_seed,
    resolve_role_task_keys,
    _validate_recipe,
)
from ember.libero_evaluation import sha256_file
from ember.pi05_source_checkpoint import canonical_hash
from ember.writer.inference import WRITER_ADAPTER_SCHEMA, task_video_mapping


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pi05_target_evaluation_v1.json"


def test_evaluation_authorities_and_roles_are_sealed() -> None:
    authorities = load_evaluation_authorities(CONFIG, ROOT)
    assert {
        role: len(
            resolve_role_task_keys(
                authorities.protocol,
                role,
                authorities.seen_panel if role == "seen_panel" else None,
            )
        )
        for role in (
            "all_targets",
            "development_train",
            "seen_panel",
            "validation",
            "test",
            "final_source",
        )
    } == {
        "all_targets": 40,
        "development_train": 24,
        "seen_panel": 8,
        "validation": 8,
        "test": 8,
        "final_source": 32,
    }
    assert resolve_role_task_keys(
        authorities.protocol, "seen_panel", authorities.seen_panel
    ) == (
        ("libero_spatial", 0),
        ("libero_spatial", 2),
        ("libero_object", 5),
        ("libero_object", 2),
        ("libero_goal", 1),
        ("libero_goal", 8),
        ("libero_10", 9),
        ("libero_10", 7),
    )
    assert authorities.normalization["authority"]["validation_or_test_numeric_reads"] == 0


def test_installed_target_contract_seals_bddl_and_fixed_states(tmp_path: Path) -> None:
    authorities = load_evaluation_authorities(CONFIG, ROOT)
    tasks, paths = inspect_installed_target_tasks(
        authorities,
        role="test",
        state_count=3,
        libero_config_dir=tmp_path / "libero_config",
    )
    assert len(tasks) == 8
    assert sum(len(task.init_state_ids) for task in tasks) == 24
    assert all(task.installed_init_state_count >= 50 for task in tasks)
    assert all(len(task.bddl_sha256) == len(task.init_states_sha256) == 64 for task in tasks)
    assert Path(paths["init_states"]).is_dir()
    protocol_by_key = {
        (row["suite"], int(row["task_id"])): row
        for row in authorities.protocol["test_tasks"]
    }
    for task in tasks:
        sealed = protocol_by_key[(task.suite, task.task_id)]
        assert task.init_states_file == sealed["init_states_file"]
        assert task.init_states_sha256 == sealed["init_states_sha256"]


def test_installed_target_contract_rejects_changed_sealed_test_hash(
    tmp_path: Path,
) -> None:
    authorities = load_evaluation_authorities(CONFIG, ROOT)
    protocol = copy.deepcopy(authorities.protocol)
    protocol["test_tasks"][0]["init_states_sha256"] = "0" * 64
    changed = replace(authorities, protocol=protocol)
    with pytest.raises(Pi05EvaluationError, match="test fixed-state hash differs"):
        inspect_installed_target_tasks(
            changed,
            role="test",
            state_count=1,
            libero_config_dir=tmp_path / "libero_config",
        )


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    (
        ("policy", "action_dim", 8, "policy recipe"),
        ("policy", "precision", "float32", "policy recipe"),
        ("environment", "terminate_on_success", False, "environment recipe"),
        ("rng", "inference_seed", 8, "RNG recipe"),
    ),
)
def test_evaluation_recipe_rejects_scientific_constant_drift(
    section: str,
    key: str,
    value: object,
    message: str,
) -> None:
    authorities = load_evaluation_authorities(CONFIG, ROOT)
    config = copy.deepcopy(authorities.config)
    config[section][key] = value
    with pytest.raises(Pi05EvaluationError, match=message):
        _validate_recipe(config, authorities.protocol)


def test_policy_noise_seed_is_per_rollout_and_replan() -> None:
    keys = [
        ("libero_spatial", task_id, state_id, replan)
        for task_id in range(2)
        for state_id in range(3)
        for replan in range(4)
    ]
    forward = {
        key: policy_noise_seed(7, key[0], key[1], key[2], key[3]) for key in keys
    }
    reverse = {
        key: policy_noise_seed(7, key[0], key[1], key[2], key[3])
        for key in reversed(keys)
    }
    assert forward == reverse
    assert len(set(forward.values())) == len(keys)
    assert forward[("libero_spatial", 0, 0, 0)] == 6161069403093503947


def test_run_contract_hash_detects_tampering(tmp_path: Path) -> None:
    authorities = load_evaluation_authorities(CONFIG, ROOT)
    tasks, paths = inspect_installed_target_tasks(
        authorities,
        role="test",
        state_count=1,
        libero_config_dir=tmp_path / "libero_config",
    )
    model = {
        "source_authority_hashes": {
            name: authorities.hashes[name]
            for name in ("normalization", "overlap_audit", "source_manifest")
        }
    }
    contract = build_run_contract(
        authorities=authorities,
        tasks=tasks,
        libero_paths=paths,
        model=model,
        tokenizer={"sha256": "a" * 64},
        output_dir=tmp_path,
        role="test",
        mode="smoke",
        replicas_per_gpu=1,
        command=("evaluate_pi05.py", "prepare"),
    )
    path = tmp_path / "run_contract.json"
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert load_run_contract(path)["contract_sha256"] == contract["contract_sha256"]
    subset = build_run_contract(
        authorities=authorities,
        tasks=tasks,
        libero_paths=paths,
        model=model,
        tokenizer={"sha256": "a" * 64},
        output_dir=tmp_path / "subset",
        role="test",
        mode="smoke",
        replicas_per_gpu=5,
        physical_gpu_ids=(2, 3),
        command=("evaluate_pi05.py", "prepare"),
    )
    assert subset["parallel"]["physical_gpu_ids"] == [2, 3]
    assert subset["parallel"]["physical_gpu_count"] == 2
    assert subset["parallel"]["worker_count"] == 10
    assert subset["parallel"]["omp_threads_per_worker"]["5"] == 1
    assert subset["parallel"]["omp_threads_per_worker"]["6"] == 1
    contract["tasks"][0]["init_state_ids"] = [49]
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(Pi05EvaluationError, match="hash changed"):
        load_run_contract(path)

    task_keys = tuple((task.suite, task.task_id) for task in tasks)
    task_roles = {key: task.split_role for key, task in zip(task_keys, tasks)}
    correct_mapping = list(task_video_mapping(task_keys, task_roles, "correct"))
    wrong_mapping = list(
        task_video_mapping(task_keys, task_roles, "cross_suite_wrong")
    )
    shared_writer = {
        "schema_version": WRITER_ADAPTER_SCHEMA,
        "kind": "as_writer",
        "writer_method": "as_writer",
        "execution_backend": "two_stage_cached_per_sample_lora_batched_replan",
        "config": {
            "path": str(ROOT / "configs/pi05_as_writer_action_forecast_v2.json"),
            "sha256": "b" * 64,
        },
        "training_run": {"run_contract_sha256": "c" * 64},
        "checkpoint": {
            "cursor": 12,
            "manifest_file_sha256": "d" * 64,
            "writer_state_sha256": "f" * 64,
        },
        "video_data": {"target_data_manifest_file_sha256": "e" * 64},
        "lora_contract_sha256": (
            "da14fb2cdfc6ca575f97ba5d70fd2d0a70efb0a243b5028b6fd728d19b097d87"
        ),
        "video_schedule": {"seed": 7, "demo_count": 50},
        "pairing_sha256": "1" * 64,
    }
    correct = build_run_contract(
        authorities=authorities,
        tasks=tasks,
        libero_paths=paths,
        model=model,
        tokenizer={"sha256": "a" * 64},
        output_dir=tmp_path / "correct",
        role="test",
        mode="smoke",
        replicas_per_gpu=1,
        command=("evaluate_pi05.py", "prepare"),
        adapter={
                **shared_writer,
                "arm": "as_writer_correct_video",
                "video_condition": "correct",
                "task_video_mapping_sha256": canonical_hash(correct_mapping),
                "task_video_mapping": correct_mapping,
        },
    )
    wrong = build_run_contract(
        authorities=authorities,
        tasks=tasks,
        libero_paths=paths,
        model=model,
        tokenizer={"sha256": "a" * 64},
        output_dir=tmp_path / "wrong",
        role="test",
        mode="smoke",
        replicas_per_gpu=1,
        command=("evaluate_pi05.py", "prepare"),
        adapter={
                **shared_writer,
                "arm": "as_writer_cross_suite_wrong_video",
                "video_condition": "cross_suite_wrong",
                "task_video_mapping_sha256": canonical_hash(wrong_mapping),
                "task_video_mapping": wrong_mapping,
        },
    )
    assert correct["paired_control_sha256"] == wrong["paired_control_sha256"]
    assert correct["contract_sha256"] != wrong["contract_sha256"]


def test_source_checkpoint_inspection_requires_generic_base_and_raw_policy_contract(
    tmp_path: Path,
) -> None:
    authorities = load_evaluation_authorities(CONFIG, ROOT)
    source_run = tmp_path / "source_run"
    checkpoint = source_run / "checkpoints" / "step_00000001"
    model = checkpoint / "policy"
    model.mkdir(parents=True)
    model_config = {
        "type": "pi05",
        "dtype": "bfloat16",
        "chunk_size": 50,
        "n_action_steps": 10,
        "num_inference_steps": 10,
        "tokenizer_max_length": 200,
        "max_action_dim": 32,
        "max_state_dim": 32,
        "image_resolution": [224, 224],
        "input_features": {
            "observation.images.base_0_rgb": {
                "type": "VISUAL",
                "shape": [3, 224, 224],
            },
            "observation.images.left_wrist_0_rgb": {
                "type": "VISUAL",
                "shape": [3, 224, 224],
            },
            "observation.images.right_wrist_0_rgb": {
                "type": "VISUAL",
                "shape": [3, 224, 224],
            },
            "observation.state": {"type": "STATE", "shape": [8]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [7]}},
    }
    (model / "config.json").write_text(
        json.dumps(model_config, sort_keys=True) + "\n", encoding="utf-8"
    )
    (model / "model.safetensors").write_bytes(b"fake-safe-tensors")
    source_contract = {
        "schema_version": "ember_pi05_source_launch_v1",
        "mode": "smoke",
        "config_sha256": authorities.hashes["source_base_config"],
        "models": authorities.source_base_config["models"],
        "features": authorities.source_base_config["features"],
        "optimization": authorities.source_base_config["optimization"],
        "task_ids": authorities.source_base_config["data"]["active_task_ids"],
        "git": {"commit": "a" * 40},
        "authorities": authorities.source_base_config["authorities"],
    }
    (source_run / "run_contract.json").write_text(
        json.dumps(source_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contract_sha = canonical_hash(source_contract)
    trainer = {
        "schema_version": "ember_pi05_source_trainer_state_v1",
        "contract_sha256": contract_sha,
        "optimizer_step": 1,
        "micro_step": 1,
        "ema_enabled": True,
    }
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(trainer, sort_keys=True) + "\n", encoding="utf-8"
    )
    files = [
        {
            "path": str(path.relative_to(checkpoint)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted((model / "config.json", model / "model.safetensors", checkpoint / "trainer_state.json"))
    ]
    manifest = {
        "schema_version": "ember_pi05_source_checkpoint_v1",
        "contract_sha256": contract_sha,
        "optimizer_step": 1,
        "micro_step": 1,
        "files": files,
        "aggregate_sha256": canonical_hash(files),
    }
    (checkpoint / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inspected = inspect_source_checkpoint(
        authorities,
        source_run,
        checkpoint,
        evaluation_mode="smoke",
    )
    assert inspected["optimizer_step"] == 1
    assert inspected["model_path"] == str(model)
    assert inspected["frozen_policy_subdir"] == "policy"
    assert inspected["source_training_commit"] == "a" * 40
