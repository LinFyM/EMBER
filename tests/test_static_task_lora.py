from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from ember.ecp.g1_evaluation import (
    _shared_scientific_contract,
    _single_training_authority,
    _task_record,
)
from ember.eval_adapters import validate_episode_adapter_fields
from ember.lora import identity_lora_state
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_lora import load_pi05_lora_contract
from ember.static_task_lora import (
    FrozenStaticTaskLoRAAdapter,
    PreparedStaticTaskLoRA,
    STATIC_TASK_LORA_EPISODE_SCHEMA,
    STATIC_TASK_LORA_KIND,
    STATIC_TASK_LORA_MANIFEST_SCHEMA,
    inspect_static_task_lora_bank,
    _evaluation_role_valid,
    validation_task_keys,
)


ROOT = Path(__file__).resolve().parents[1]


def test_static_validation_role_requires_fixed_eight_tasks_and_explicit_scope() -> None:
    keys = validation_task_keys()
    bank = {"evaluation_role": "validation", "arm": "ecp_policy_response_writer_full_correct_k1"}
    assert _evaluation_role_valid(bank, "validation", keys)
    assert not _evaluation_role_valid(bank, "test", keys)
    assert not _evaluation_role_valid(bank, "development_train", keys)
    assert not _evaluation_role_valid(bank, "validation", keys[:-1])
    assert not _evaluation_role_valid(bank, "validation", (*keys[:-1], ("libero_10", 3)))
    assert not _evaluation_role_valid({"arm": bank["arm"]}, "validation", keys)
    assert _evaluation_role_valid({**bank, "arm": "frozen_stable_carrier"}, "validation", keys)
    assert not _evaluation_role_valid({**bank, "arm": "unregistered_adapter"}, "validation", keys)


def test_static_adapter_batches_same_task_without_reordering_episode_noise() -> None:
    calls = []
    installed = []

    def predict(batch, *, noise, num_steps):
        calls.append((batch, noise, num_steps))
        return noise + batch["observation.state"][:, None, :]

    adapter = object.__new__(FrozenStaticTaskLoRAAdapter)
    adapter.policy = SimpleNamespace(predict_action_chunk=predict)
    adapter.install = installed.append
    prepared = [PreparedStaticTaskLoRA(("libero_goal", 0), {"init_state_id": i}) for i in range(8)]
    noise = torch.arange(8 * 50 * 7).reshape(8, 50, 7).float()
    batch = {"observation.state": torch.arange(8)[:, None].expand(8, 7).float()}
    actions = adapter.predict_action_chunk(prepared, batch, noise=noise, num_steps=10)
    assert len(calls) == 1 and calls[0][1] is noise and calls[0][2] == 10
    assert installed == [prepared[0]]
    torch.testing.assert_close(actions, noise + batch["observation.state"][:, None, :])
    mixed = [*prepared[:-1], PreparedStaticTaskLoRA(("libero_spatial", 2), {})]
    for invalid in (mixed, prepared[:-1], []):
        with pytest.raises(Pi05EvaluationError, match="share one task"):
            adapter.predict_action_chunk(invalid, batch, noise=noise, num_steps=10)
    assert len(calls) == 1


def test_static_task_lora_manifest_and_episode_evidence_are_exact(
    tmp_path: Path,
) -> None:
    lora_path = ROOT / "configs/pi05_lora_v1.json"
    lora = load_pi05_lora_contract(lora_path)
    run_root = tmp_path / "run"
    checkpoint = run_root / "checkpoints/step_00000005"
    checkpoint.mkdir(parents=True)
    adapter_path = checkpoint / "adapter.safetensors"
    save_file(
        {
            name: value.to(torch.bfloat16)
            for name, value in identity_lora_state(lora).items()
        },
        str(adapter_path),
    )
    checkpoint_manifest = {
        "schema_version": "ember_ecp_native_factor_g1_checkpoint_v1",
        "step": 5,
        "task_ordinal": 90,
        "global_task_id": 0,
        "rank_partition": {"carrier": [0, 12], "task": [12, 16]},
        "single_complete_rank16": True,
        "content_hash_policy": "disabled_by_owner",
        "files": {"adapter.safetensors": adapter_path.stat().st_size},
    }
    manifest_path = checkpoint / "manifest.json"
    manifest_path.write_text(json.dumps(checkpoint_manifest), encoding="utf-8")
    (run_root / "run_contract.json").write_text(
        json.dumps(
            {
                "schema_version": "ember_ecp_native_factor_g1_task_run_v1",
                "mode": "formal",
                "content_hash_policy": "disabled_by_owner",
                "device": "cuda:0",
                "runtime": {
                    "world_size": 1,
                    "torch_device": "cuda:0",
                    "cuda_visible_devices": "GPU-test",
                    "device_name": "NVIDIA A40",
                },
                "video": {"K": 1, "cross_video_weight": "identity_k1"},
                "video_contract": {"videos_per_task": 1},
                "functional_query": {"demo_indices": [1, 2]},
                "authorities": {"source_checkpoint": "source"},
                "native_factor": {"residual_rank": 4},
                "optimization": {"initial_segment_steps": 5},
                "information_wall": {"action_meta_installed": False},
                "pure_native_stage0": {
                    "action_meta_module_count": 0,
                    "action_meta_parameter_count": 0,
                    "policy_trainable_parameter_count": 0,
                    "stage0_trainable_parameter_count": 0,
                },
                "repository": {
                    "commit": "a" * 40,
                    "dirty_paths": [],
                    "branch": "",
                    "upstream": None,
                },
                "task": {
                    "suite": "libero_spatial",
                    "task_id": 0,
                    "ordinal": 90,
                    "global_task_id": 0,
                    "language": "exact language",
                },
            }
        ),
        encoding="utf-8",
    )
    (run_root / "segment_completion.json").write_text(
        json.dumps({"status": "segment_complete", "completed_steps": 5}),
        encoding="utf-8",
    )
    row = {
        "suite": "libero_spatial",
        "task_id": 0,
        "ordinal": 90,
        "global_task_id": 0,
        "language": "exact language",
        "step": 5,
        "run_root": str(run_root),
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_bytes": manifest_path.stat().st_size,
        "adapter_path": str(adapter_path),
        "adapter_bytes": adapter_path.stat().st_size,
        "single_complete_rank16": True,
    }
    source = {
        "source_run": str(tmp_path / "source"),
        "checkpoint": str(tmp_path / "source/checkpoints/step_00001000"),
        "model_path": str(tmp_path / "source/checkpoints/step_00001000/policy"),
    }
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(
        json.dumps(
            {
                "schema_version": STATIC_TASK_LORA_MANIFEST_SCHEMA,
                "status": "sealed",
                "arm": "ecp_native_factor_g1_free_code",
                "source": source,
                "lora_contract": {
                    "path": str(lora_path),
                    "bytes": lora_path.stat().st_size,
                },
                "rank_partition": {"carrier": [0, 12], "task": [12, 16]},
                "single_complete_rank16": True,
                "training_commit": "a" * 40,
                "shared_run_contract": {
                    "schema_version": "ember_ecp_native_factor_g1_task_run_v1",
                    "mode": "formal",
                },
                "tasks": [row],
                "information_wall": {
                    "action_meta_installed": False,
                    "second_adapter_deployed": False,
                    "teacher_video_runtime_reads": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    adapter = inspect_static_task_lora_bank(
        manifest_path=bank_path,
        source=source,
        task_keys=(("libero_spatial", 0),),
        evaluation_role="development_train",
        require_formal=True,
    )
    evidence = {
        "schema_version": STATIC_TASK_LORA_EPISODE_SCHEMA,
        **row,
        "init_state_id": 7,
    }
    assert adapter["kind"] == STATIC_TASK_LORA_KIND
    published_row, commit, shared = _task_record(run_root=run_root, step=5, lora=lora)
    assert published_row == row
    assert commit == "a" * 40
    assert shared["video_contract"] == {"videos_per_task": 1}
    assert validate_episode_adapter_fields(
        adapter,
        {"static_task_lora": evidence},
        suite="libero_spatial",
        task_id=0,
        init_state_id=7,
    )
    assert not validate_episode_adapter_fields(
        adapter,
        {"task_expert": evidence},
        suite="libero_spatial",
        task_id=0,
        init_state_id=7,
    )


def test_g1_evaluation_bank_rejects_mixed_commit_or_mechanism() -> None:
    base = {
        "schema_version": "ember_ecp_native_factor_g1_task_run_v1",
        "mode": "formal",
        "content_hash_policy": "disabled_by_owner",
        "authorities": {"source": "same"},
        "video_contract": {"videos_per_task": 1},
        "functional_query": {"demo_indices": [1, 2]},
        "native_factor": {"residual_rank": 4},
        "optimization": {"selection_lr": 0.02},
        "information_wall": {"action_meta_installed": False},
    }
    changed = dict(base)
    changed["native_factor"] = {"residual_rank": 8}
    shared = _shared_scientific_contract(base)
    changed_shared = _shared_scientific_contract(changed)
    row: dict[str, object] = {}
    with pytest.raises(ValueError, match="training commits"):
        _single_training_authority(((row, "a" * 40, shared), (row, "b" * 40, shared)))
    with pytest.raises(ValueError, match="scientific run contracts"):
        _single_training_authority(
            ((row, "a" * 40, shared), (row, "a" * 40, changed_shared))
        )


def test_g3_static_bank_accepts_only_matching_materialized_condition(
    tmp_path: Path,
) -> None:
    lora_path = ROOT / "configs/pi05_lora_v1.json"
    lora = load_pi05_lora_contract(lora_path)
    checkpoint = tmp_path / "adapters/task_00"
    checkpoint.mkdir(parents=True)
    adapter_path = checkpoint / "adapter.safetensors"
    save_file(identity_lora_state(lora), str(adapter_path))
    checkpoint_manifest = {
        "schema_version": "ember_ecp_shared_compiler_g3_materialized_adapter_v1",
        "condition": "correct_full",
        "compiler_macro": 5,
        "authority_id": 90,
        "global_task_id": 0,
        "suite": "libero_spatial",
        "task_id": 0,
        "single_complete_rank16": True,
    }
    checkpoint_manifest_path = checkpoint / "manifest.json"
    checkpoint_manifest_path.write_text(
        json.dumps(checkpoint_manifest), encoding="utf-8"
    )
    row = {
        "suite": "libero_spatial",
        "task_id": 0,
        "natural_program_authority_id": 90,
        "global_task_id": 0,
        "language": "exact language",
        "condition": "correct_full",
        "compiler_macro": 5,
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_bytes": checkpoint_manifest_path.stat().st_size,
        "adapter_path": str(adapter_path),
        "adapter_bytes": adapter_path.stat().st_size,
        "single_complete_rank16": True,
    }
    source = {
        "source_run": str(tmp_path / "source"),
        "checkpoint": str(tmp_path / "source/checkpoints/step_00001000"),
        "model_path": str(tmp_path / "source/checkpoints/step_00001000/policy"),
    }
    bank = {
        "schema_version": STATIC_TASK_LORA_MANIFEST_SCHEMA,
        "status": "sealed",
        "arm": "ecp_shared_compiler_g3_correct_full",
        "source": source,
        "lora_contract": {
            "path": str(lora_path),
            "bytes": lora_path.stat().st_size,
        },
        "rank_partition": {"carrier": [0, 12], "task": [12, 16]},
        "single_complete_rank16": True,
        "training_commit": "a" * 40,
        "materialization_commit": "b" * 40,
        "shared_run_contract": {
            "schema_version": "ember_ecp_shared_compiler_g3_run_v2",
            "stage": "g3_shared_compiler",
            "mode": "formal",
        },
        "compiler_checkpoint": {"path": "macro_00000005", "macro": 5},
        "condition": {
            "name": "correct_full",
            "view": "full",
            "video_demos": [5, 6, 7, 8],
            "K": 4,
        },
        "tasks": [row],
        "information_wall": {
            "action_meta_installed": False,
            "second_adapter_deployed": False,
            "teacher_video_runtime_reads": 0,
        },
    }
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    adapter = inspect_static_task_lora_bank(
        manifest_path=bank_path,
        source=source,
        task_keys=(("libero_spatial", 0),),
        evaluation_role="development_train",
        require_formal=True,
    )
    assert adapter["condition"]["name"] == "correct_full"
    checkpoint_manifest["condition"] = "first_final"
    checkpoint_manifest_path.write_text(
        json.dumps(checkpoint_manifest), encoding="utf-8"
    )
    row["checkpoint_manifest_bytes"] = checkpoint_manifest_path.stat().st_size
    bank["tasks"] = [row]
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    with pytest.raises(Pi05EvaluationError, match="checkpoint authority"):
        inspect_static_task_lora_bank(
            manifest_path=bank_path,
            source=source,
            task_keys=(("libero_spatial", 0),),
            evaluation_role="development_train",
            require_formal=True,
        )


@pytest.mark.parametrize("video_condition", [
    {"video_demos": [5]}, {"video_demos_by_global_task": {"0": [5]}},
])
def test_policy_response_writer_bank_requires_matching_k1_checkpoint(
    tmp_path: Path, video_condition: dict,
) -> None:
    lora_path = ROOT / "configs/pi05_lora_v1.json"
    lora = load_pi05_lora_contract(lora_path)
    checkpoint = tmp_path / "adapters/libero_spatial_task_00"
    checkpoint.mkdir(parents=True)
    adapter_path = checkpoint / "adapter.safetensors"
    save_file(identity_lora_state(lora), str(adapter_path))
    checkpoint_manifest = {
        "schema_version": "ember_ecp_policy_response_writer_materialized_adapter_v1",
        "condition": "correct_k1",
        "representation": "full",
        "writer_macro": 610,
        "writer_checkpoint": "macro_00000610",
        "video_demos": [5],
        "authority_id": 71,
        "global_task_id": 0,
        "suite": "libero_spatial",
        "task_id": 0,
        "single_complete_rank16": True,
    }
    checkpoint_manifest_path = checkpoint / "manifest.json"
    checkpoint_manifest_path.write_text(
        json.dumps(checkpoint_manifest), encoding="utf-8"
    )
    row = {
        "suite": "libero_spatial",
        "task_id": 0,
        "natural_program_authority_id": 71,
        "global_task_id": 0,
        "language": "exact language",
        "condition": "correct_k1",
        "representation": "full",
        "writer_macro": 610,
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_bytes": checkpoint_manifest_path.stat().st_size,
        "adapter_path": str(adapter_path),
        "adapter_bytes": adapter_path.stat().st_size,
        "single_complete_rank16": True,
    }
    source = {
        "source_run": str(tmp_path / "source"),
        "checkpoint": str(tmp_path / "source/checkpoints/step_00001000"),
        "model_path": str(tmp_path / "source/checkpoints/step_00001000/policy"),
    }
    bank = {
        "schema_version": STATIC_TASK_LORA_MANIFEST_SCHEMA,
        "status": "sealed",
        "arm": "ecp_policy_response_writer_full_correct_k1",
        "source": source,
        "lora_contract": {
            "path": str(lora_path),
            "bytes": lora_path.stat().st_size,
        },
        "rank_partition": {"carrier": [0, 12], "task": [12, 16]},
        "single_complete_rank16": True,
        "training_commit": "a" * 40,
        "materialization_commit": "b" * 40,
        "shared_run_contract": {
            "schema_version": "ember_policy_response_writer_shared_run_v1",
            "stage": "policy_response_writer_shared_positive_only",
            "mode": "formal",
            "representation": "full",
        },
        "writer_checkpoint": {"path": "macro_00000610", "macro": 610},
        "condition": {
            "name": "correct_k1",
            "representation": "full",
            **video_condition,
            "K": 1,
            "outcome_dependence": False,
            "gradient_use": False,
        },
        "tasks": [row],
        "information_wall": {
            "action_meta_installed": False,
            "second_adapter_deployed": False,
            "teacher_video_runtime_reads": 0,
            "writer_invocations_per_task_condition": 1,
            "validation_action_or_reward_reads": 0,
            "test_action_or_reward_reads": 0,
            "shuffled_or_reversed_use": False,
            "wrong_video_use": False,
        },
    }
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    adapter = inspect_static_task_lora_bank(
        manifest_path=bank_path,
        source=source,
        task_keys=(("libero_spatial", 0),),
        evaluation_role="development_train",
        require_formal=True,
    )
    assert adapter["writer_checkpoint"]["macro"] == 610

    bank["writer_checkpoint"]["macro"] = 1210
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    with pytest.raises(Pi05EvaluationError, match="manifest changed"):
        inspect_static_task_lora_bank(
            manifest_path=bank_path,
            source=source,
            task_keys=(("libero_spatial", 0),),
            evaluation_role="development_train",
            require_formal=True,
        )
    bank["writer_checkpoint"]["macro"] = 610

    checkpoint_manifest["video_demos"] = [6]
    checkpoint_manifest_path.write_text(
        json.dumps(checkpoint_manifest), encoding="utf-8"
    )
    row["checkpoint_manifest_bytes"] = checkpoint_manifest_path.stat().st_size
    bank["tasks"] = [row]
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    with pytest.raises(Pi05EvaluationError, match="checkpoint authority"):
        inspect_static_task_lora_bank(
            manifest_path=bank_path,
            source=source,
            task_keys=(("libero_spatial", 0),),
            evaluation_role="development_train",
            require_formal=True,
        )


def test_g3_language_bank_requires_zero_video_baseline_contract(
    tmp_path: Path,
) -> None:
    lora_path = ROOT / "configs/pi05_lora_v1.json"
    lora = load_pi05_lora_contract(lora_path)
    checkpoint = tmp_path / "adapters/task_00"
    checkpoint.mkdir(parents=True)
    adapter_path = checkpoint / "adapter.safetensors"
    save_file(identity_lora_state(lora), str(adapter_path))
    checkpoint_manifest = {
        "schema_version": "ember_ecp_g3_language_only_adapter_v1",
        "condition": "learned_language_only",
        "authority_id": 90,
        "global_task_id": 0,
        "suite": "libero_spatial",
        "task_id": 0,
        "single_complete_rank16": True,
    }
    checkpoint_manifest_path = checkpoint / "manifest.json"
    checkpoint_manifest_path.write_text(
        json.dumps(checkpoint_manifest), encoding="utf-8"
    )
    row = {
        "suite": "libero_spatial",
        "task_id": 0,
        "natural_program_authority_id": 90,
        "global_task_id": 0,
        "language": "exact language",
        "condition": "learned_language_only",
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_bytes": checkpoint_manifest_path.stat().st_size,
        "adapter_path": str(adapter_path),
        "adapter_bytes": adapter_path.stat().st_size,
        "single_complete_rank16": True,
    }
    source = {
        "source_run": str(tmp_path / "source"),
        "checkpoint": str(tmp_path / "source/checkpoints/step_00001000"),
        "model_path": str(tmp_path / "source/checkpoints/step_00001000/policy"),
    }
    bank = {
        "schema_version": STATIC_TASK_LORA_MANIFEST_SCHEMA,
        "status": "sealed",
        "arm": "ecp_shared_compiler_g3_learned_language_only",
        "source": source,
        "lora_contract": {
            "path": str(lora_path),
            "bytes": lora_path.stat().st_size,
        },
        "rank_partition": {"carrier": [0, 12], "task": [12, 16]},
        "single_complete_rank16": True,
        "training_commit": "a" * 40,
        "materialization_commit": "b" * 40,
        "shared_run_contract": {
            "schema_version": "ember_ecp_g3_language_only_baseline_v1",
            "stage": "g3_learned_language_only",
            "mode": "formal",
        },
        "condition": {
            "name": "learned_language_only",
            "view": None,
            "video_demos": [],
            "K": 0,
        },
        "tasks": [row],
        "information_wall": {
            "action_meta_installed": False,
            "second_adapter_deployed": False,
            "teacher_video_runtime_reads": 0,
        },
    }
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    adapter = inspect_static_task_lora_bank(
        manifest_path=bank_path,
        source=source,
        task_keys=(("libero_spatial", 0),),
        evaluation_role="development_train",
        require_formal=True,
    )
    assert adapter["condition"]["video_demos"] == []
    bank["condition"]["K"] = 1
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    with pytest.raises(Pi05EvaluationError, match="manifest changed"):
        inspect_static_task_lora_bank(
            manifest_path=bank_path,
            source=source,
            task_keys=(("libero_spatial", 0),),
            evaluation_role="development_train",
            require_formal=True,
        )
