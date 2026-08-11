from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from ember.pi05_assets import (
    Pi05EvaluationError,
    configure_libero_runtime_assets,
    prepare_libero_config,
)
from ember.pi05_eval_contract import (
    build_run_contract,
    git_state_is_clean_pushed_or_frozen_authority,
    inspect_installed_target_tasks,
    inspect_source_checkpoint,
    load_evaluation_authorities,
    load_run_contract,
    policy_noise_seed,
    resolve_role_task_keys,
    _validate_recipe,
)
from ember.expert_manifold.inference import (
    EXPERT_MANIFOLD_ADAPTER_SCHEMA,
    EXPERT_MANIFOLD_WRITER_KIND,
)
from ember.expert_manifold.v6_prior_contract import (
    V6_PRIOR_CANONICAL_CONFIG,
    V6_PRIOR_CONFIG_SCHEMA,
)
from ember.pi05_lora import pi05_target_names
from ember.expert_manifold.video_schedule import (
    task_video_mapping,
    video_schedule_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def test_clean_detached_frozen_authority_checkout_is_launchable() -> None:
    commit = "a" * 40
    detached = {
        "branch": "",
        "commit": commit,
        "upstream": None,
        "upstream_commit": None,
        "authority_ref": "origin/codex/bci-continuation",
        "authority_contains_commit": True,
        "dirty_paths": [],
    }
    assert git_state_is_clean_pushed_or_frozen_authority(detached)
    detached["authority_contains_commit"] = False
    assert not git_state_is_clean_pushed_or_frozen_authority(detached)


def _pi05_template_dtype_by_name() -> dict[str, str]:
    return {
        f"{target}.lora_{factor}.default.weight": (
            "F32"
            if target in {"model.action_in_proj", "model.action_out_proj"}
            else "BF16"
        )
        for target in pi05_target_names()
        for factor in ("A", "B")
    }


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
    assert (
        authorities.normalization["authority"]["validation_or_test_numeric_reads"] == 0
    )


def test_libero_config_accepts_a_host_local_assets_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets_root = tmp_path / "libero-assets"
    assets_root.mkdir()
    monkeypatch.setenv("EMBER_LIBERO_ASSETS_ROOT", str(assets_root))
    paths = prepare_libero_config(tmp_path / "libero-config")
    assert paths["assets"] == str(assets_root.resolve())


def test_libero_runtime_uses_the_contract_asset_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets_root = tmp_path / "libero-assets"
    assets_root.mkdir()
    package = ModuleType("libero")
    runtime = ModuleType("libero.libero")
    runtime._assets_path_cache = None
    package.libero = runtime
    monkeypatch.setitem(sys.modules, "libero", package)
    monkeypatch.setitem(sys.modules, "libero.libero", runtime)

    configure_libero_runtime_assets(assets_root)

    assert runtime._assets_path_cache == str(assets_root.resolve())


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
    assert all(task.bddl_bytes > 0 and task.init_states_bytes > 0 for task in tasks)
    assert Path(paths["init_states"]).is_dir()
    protocol_by_key = {
        (row["suite"], int(row["task_id"])): row
        for row in authorities.protocol["test_tasks"]
    }
    for task in tasks:
        sealed = protocol_by_key[(task.suite, task.task_id)]
        assert task.init_states_file == sealed["init_states_file"]


def test_installed_target_contract_rejects_changed_sealed_test_file(
    tmp_path: Path,
) -> None:
    authorities = load_evaluation_authorities(CONFIG, ROOT)
    protocol = copy.deepcopy(authorities.protocol)
    protocol["test_tasks"][0]["init_states_file"] = "wrong.pruned_init"
    changed = replace(authorities, protocol=protocol)
    with pytest.raises(Pi05EvaluationError, match="test fixed-state file differs"):
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


def test_run_contract_uses_explicit_reference_and_owned_root(tmp_path: Path) -> None:
    authorities = load_evaluation_authorities(CONFIG, ROOT)
    tasks, paths = inspect_installed_target_tasks(
        authorities,
        role="test",
        state_count=1,
        libero_config_dir=tmp_path / "libero_config",
    )
    model = {"optimizer_step": 1000}
    contract = build_run_contract(
        authorities=authorities,
        tasks=tasks,
        libero_paths=paths,
        model=model,
        tokenizer={"path": "/tokenizer.model"},
        output_dir=tmp_path,
        role="test",
        mode="smoke",
        replicas_per_gpu=1,
        physical_gpu_ids=tuple(range(6)),
        command=("evaluate_pi05.py", "prepare"),
    )
    path = tmp_path / "run_contract.json"
    path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert (
        load_run_contract(path)["contract_reference"] == contract["contract_reference"]
    )
    subset = build_run_contract(
        authorities=authorities,
        tasks=tasks,
        libero_paths=paths,
        model=model,
        tokenizer={"path": "/tokenizer.model"},
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
    all_configured = build_run_contract(
        authorities=authorities,
        tasks=tasks,
        libero_paths=paths,
        model=model,
        tokenizer={"path": "/tokenizer.model"},
        output_dir=tmp_path / "all-configured",
        role="test",
        mode="smoke",
        replicas_per_gpu=1,
        physical_gpu_ids=tuple(range(8)),
        command=("evaluate_pi05.py", "prepare"),
    )
    assert all_configured["parallel"]["physical_gpu_ids"] == list(range(8))
    assert all_configured["parallel"]["physical_gpu_count"] == 8
    with pytest.raises(Pi05EvaluationError, match="configured node topology"):
        build_run_contract(
            authorities=authorities,
            tasks=tasks,
            libero_paths=paths,
            model=model,
            tokenizer={"path": "/tokenizer.model"},
            output_dir=tmp_path / "too-many",
            role="test",
            mode="smoke",
            replicas_per_gpu=1,
            physical_gpu_ids=tuple(range(9)),
            command=("evaluate_pi05.py", "prepare"),
        )
    contract["output_dir"] = str(tmp_path / "elsewhere")
    path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(Pi05EvaluationError, match="contract changed"):
        load_run_contract(path)


def _writer_contract_inputs(tmp_path: Path) -> tuple:
    authorities = load_evaluation_authorities(CONFIG, ROOT)
    tasks, paths = inspect_installed_target_tasks(
        authorities,
        role="test",
        state_count=1,
        libero_config_dir=tmp_path / "libero_config",
    )
    model = {"optimizer_step": 1000}
    task_keys = tuple((task.suite, task.task_id) for task in tasks)
    task_roles = {key: task.split_role for key, task in zip(task_keys, tasks)}
    correct_mapping = list(task_video_mapping(task_keys, task_roles, "correct"))
    schedule, pairing = video_schedule_contract(
        seed=7,
        demo_count=50,
        sampling_mode="without_replacement",
    )
    shared_writer = {
        "schema_version": EXPERT_MANIFOLD_ADAPTER_SCHEMA,
        "kind": EXPERT_MANIFOLD_WRITER_KIND,
        "execution_backend": (
            "online_frozen_v6_condition_program_residual_then_episode_lora_cache"
        ),
        "config": {
            "path": str(V6_PRIOR_CANONICAL_CONFIG),
            "schema": V6_PRIOR_CONFIG_SCHEMA,
        },
        "writer_asset": {
            "reference": "v6-prior:historical-macro400",
            "kind": "historical_v6_macro400_load_only",
            "method_macro": 0,
            "writer_parameter_count": 10_775_296,
            "program_residual_value_count": 20_971_520,
            "generated_lora_tensor_count": 76,
            "checkpoint": "/writer/checkpoints/step_00000400",
            "writer_state": {
                "template_lora_storage": {
                    "tensor_count": 76,
                    "parameter_count": 1_287_168,
                    "tensor_bytes": 2_641_920,
                    "dtype_tensor_counts": {"BF16": 72, "F32": 4},
                    "dtype_parameter_counts": {
                        "BF16": 1_253_376,
                        "F32": 33_792,
                    },
                    "dtype_by_name": _pi05_template_dtype_by_name(),
                }
            },
        },
        "evaluation_authority": {
            "formal_status": "blocked_until_new_residual_deployment_graph_live_profile",
            "throughput_policy": "highest_measured_batch_throughput_with_device_memory_headroom",
            "minimum_smoke_writer_model_batch_size": 8,
            "online_smoke_evidence": None,
        },
        "video_data": {"root": "/videos"},
        "lora_contract": {
            "reference": "configs/pi05_lora_v1.json:76tensors:1287168parameters"
        },
        "video_schedule": schedule,
        "pairing_reference": pairing,
    }
    return (
        authorities,
        tasks,
        paths,
        model,
        shared_writer,
        correct_mapping,
    )


def _build_writer_contract(
    *,
    inputs: tuple,
    output_dir: Path,
    arm: str,
    condition: str,
    mapping: list,
    writer_generation_batch_size: int = 8,
) -> dict:
    authorities, tasks, paths, model, shared_writer, _ = inputs
    return build_run_contract(
        authorities=authorities,
        tasks=tasks,
        libero_paths=paths,
        model=model,
        tokenizer={"path": "/tokenizer.model"},
        output_dir=output_dir,
        role="test",
        mode="smoke",
        replicas_per_gpu=1,
        physical_gpu_ids=tuple(range(6)),
        command=("evaluate_pi05.py", "prepare"),
        writer_generation_batch_size=writer_generation_batch_size,
        adapter={
            **shared_writer,
            "arm": arm,
            "video_condition": condition,
            "task_video_mapping_reference": "next-suite-v1",
            "task_video_mapping": mapping,
        },
    )


def test_expert_manifold_writer_pairing_is_sealed(tmp_path: Path) -> None:
    inputs = _writer_contract_inputs(tmp_path)
    _, tasks, _, _, _, correct_mapping = inputs
    task_keys = tuple((task.suite, task.task_id) for task in tasks)
    task_roles = {key: task.split_role for key, task in zip(task_keys, tasks)}
    wrong_mapping = list(task_video_mapping(task_keys, task_roles, "cross_suite_wrong"))
    correct = _build_writer_contract(
        inputs=inputs,
        output_dir=tmp_path / "correct",
        arm="expert_manifold_v6_condition_residual_correct",
        condition="correct",
        mapping=correct_mapping,
    )
    wrong = _build_writer_contract(
        inputs=inputs,
        output_dir=tmp_path / "wrong",
        arm="expert_manifold_v6_condition_residual_cross_suite_wrong",
        condition="cross_suite_wrong",
        mapping=wrong_mapping,
    )
    assert correct["paired_control"] == wrong["paired_control"]
    assert correct["contract_reference"] != wrong["contract_reference"]
    assert "writer_lora_execution" not in correct


def test_v6_prior_writer_requires_throughput_oriented_generation_batch(
    tmp_path: Path,
) -> None:
    inputs = _writer_contract_inputs(tmp_path)
    _, _, _, _, _, correct_mapping = inputs
    with pytest.raises(Pi05EvaluationError, match="throughput authority"):
        _build_writer_contract(
            inputs=inputs,
            output_dir=tmp_path / "batched",
            arm="expert_manifold_v6_condition_residual_correct",
            condition="correct",
            mapping=correct_mapping,
            writer_generation_batch_size=1,
        )
    batched = _build_writer_contract(
        inputs=inputs,
        output_dir=tmp_path / "batched",
        arm="expert_manifold_v6_condition_residual_correct",
        condition="correct",
        mapping=correct_mapping,
        writer_generation_batch_size=16,
    )
    assert batched["parallel"]["writer_generation_batch_size"] == 16


def test_reward_credit_seal_locks_the_measured_writer_batch8(
    tmp_path: Path,
) -> None:
    inputs = list(_writer_contract_inputs(tmp_path))
    shared_writer = inputs[4]
    shared_writer["evaluation_authority"] = {
        "formal_status": "sealed_from_unchanged_v6_residual_deployment_graph",
        "throughput_policy": (
            "highest_measured_batch_throughput_with_device_memory_headroom"
        ),
        "minimum_smoke_writer_model_batch_size": 8,
        "online_smoke_evidence": {"writer_model_batch_size": 8},
    }
    correct_mapping = inputs[5]
    exact = _build_writer_contract(
        inputs=tuple(inputs),
        output_dir=tmp_path / "batch8",
        arm="expert_manifold_v6_condition_residual_correct",
        condition="correct",
        mapping=correct_mapping,
        writer_generation_batch_size=8,
    )
    assert exact["parallel"]["writer_generation_batch_size"] == 8
    for batch_size in (16, 32):
        with pytest.raises(Pi05EvaluationError, match="selected Writer batch"):
            _build_writer_contract(
                inputs=tuple(inputs),
                output_dir=tmp_path / f"batch{batch_size}",
                arm="expert_manifold_v6_condition_residual_correct",
                condition="correct",
                mapping=correct_mapping,
                writer_generation_batch_size=batch_size,
            )


def test_pick_gc_seal_locks_the_measured_writer_batch32(
    tmp_path: Path,
) -> None:
    inputs = list(_writer_contract_inputs(tmp_path))
    shared_writer = inputs[4]
    shared_writer["evaluation_authority"] = {
        "formal_status": "sealed_from_live_pick_gc_deployment_profile",
        "throughput_policy": (
            "highest_measured_batch_throughput_with_device_memory_headroom"
        ),
        "minimum_smoke_writer_model_batch_size": 8,
        "online_smoke_evidence": {"writer_model_batch_size": 32},
    }
    correct_mapping = inputs[5]
    exact = _build_writer_contract(
        inputs=tuple(inputs),
        output_dir=tmp_path / "batch32",
        arm="expert_manifold_v6_condition_residual_correct",
        condition="correct",
        mapping=correct_mapping,
        writer_generation_batch_size=32,
    )
    assert exact["parallel"]["writer_generation_batch_size"] == 32
    for batch_size in (8, 16):
        with pytest.raises(Pi05EvaluationError, match="selected Writer batch"):
            _build_writer_contract(
                inputs=tuple(inputs),
                output_dir=tmp_path / f"batch{batch_size}",
                arm="expert_manifold_v6_condition_residual_correct",
                condition="correct",
                mapping=correct_mapping,
                writer_generation_batch_size=batch_size,
            )


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
    trainer = {
        "schema_version": "ember_pi05_source_trainer_state_v1",
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
        }
        for path in sorted(
            (
                model / "config.json",
                model / "model.safetensors",
                checkpoint / "trainer_state.json",
            )
        )
    ]
    manifest = {
        "schema_version": "ember_pi05_source_checkpoint_v1",
        "optimizer_step": 1,
        "micro_step": 1,
        "files": files,
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
