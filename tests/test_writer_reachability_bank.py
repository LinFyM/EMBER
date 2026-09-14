"""Bounded diagnostic adapters cannot escape their train/support/state scope."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from ember.pi05_assets import Pi05EvaluationError
from ember.static_task_lora import FrozenStaticTaskLoRAAdapter, _evaluation_role_valid
from ember.writer import materialization, reachability_bank as bank


ROOT = Path(__file__).resolve().parents[1]


def saved(path, value):
    path.write_text(json.dumps(value))
    return bank._record(path)


@pytest.fixture
def diagnostic(tmp_path, monkeypatch):
    tasks = [row for row in json.loads((ROOT / "configs/pi05_target_data_v1/manifest.json").read_text())["tasks"]
             if row["split_role"] == "train"]
    checkpoint = {"path": str(tmp_path / "writer"), "macro": 900}
    monkeypatch.setattr(materialization, "inspect_writer_checkpoint", lambda _: ({}, checkpoint))
    git = {"commit": "a" * 40, "branch": "", "upstream": None, "dirty_paths": [],
           "authority_ref": "origin/main", "authority_contains_commit": True}
    run = {"schema": bank.TASK_SCHEMA, "source": {}, "writer_checkpoint": checkpoint,
           "smoke": False, "teacher_demo": 16, "support_pool": list(range(17, 42)),
           "heldout_pool": list(range(42, 46)), "support_positions": 64, "support_fm_realizations": 2,
           "heldout_positions": 128, "heldout_fm_realizations": 1, "diagnostic_git": git,
           "tasks": [row["global_task_id"] for row in tasks],
           "optimizer": {"lr": 1., "max_iter": 32, "max_eval": 48, "history_size": 10,
                         "tolerance_grad": 1e-7, "tolerance_change": 1e-9, "line_search_fn": "strong_wolfe"}}
    manifest = {"arm": bank.PREFIX + "free_q", "evaluation_role": "development_train",
        "diagnostic_only": True, "checkpoint_selection": False, "future_initialization_allowed": False,
        "init_state_ids": bank.STATES, "source": {}, "writer_checkpoint": checkpoint,
        "shared_run_contract": run, "training_commit": git["commit"], "asset_root": str(ROOT),
        "diagnostic_run": saved(tmp_path / "run.json", run),
        "tasks": [{**row, "allowed_init_state_ids": bank.STATES} for row in tasks],
        "information_wall": {"validation_action_or_reward_reads": 0, "test_action_or_reward_reads": 0,
            "heldout_gradients": False, "source_trainable": 0, "writer_meta_trainable": 0,
            "diagnostic_fit_only": True}}
    return manifest, tmp_path


def test_bank_requires_full_train24_and_sealed_support_fit(diagnostic):
    manifest, tmp_path = diagnostic
    assert bank.bank_provenance_valid(manifest)
    for name, value in (("evaluation_role", "validation"), ("checkpoint_selection", True),
                        ("future_initialization_allowed", True), ("arm", bank.PREFIX + "unknown"),
                        ("tasks", manifest["tasks"][:-1])):
        assert not bank.bank_provenance_valid({**manifest, name: value})
    for name, value in (("heldout_gradients", True), ("validation_action_or_reward_reads", 1),
                        ("test_action_or_reward_reads", 1), ("writer_meta_trainable", 1)):
        changed = deepcopy(manifest)
        changed["information_wall"][name] = value
        assert not bank.bank_provenance_valid(changed)
    changed = deepcopy(manifest)
    changed["shared_run_contract"]["smoke"] = True
    changed["diagnostic_run"] = saved(tmp_path / "smoke.json", changed["shared_run_contract"])
    assert not bank.bank_provenance_valid(changed)
    assert not _evaluation_role_valid(manifest, "validation", ())
    assert not _evaluation_role_valid(manifest, "test", ())


def panel(support):
    pool, count, realizations = (list(range(17, 42)), 64, 2) if support else (list(range(42, 46)), 128, 1)
    return {"pool": pool, "count": count, "action_demos": [pool[0]] * count,
        "action_frames": [0] * count, "action_start_indices": [1] * count,
        "action_chunk_size": 50, "action_start_offset": 1,
        "flow_realizations_per_position": realizations, "prediction_count": count * realizations,
        "gradients": support, "full10_noise_independent_of_fm": True}


def test_task_cannot_substitute_teacher_actions_held_labels_or_another_adapter(diagnostic):
    manifest, base = diagnostic
    original = base / "free_q.safetensors"
    original.write_bytes(b"factor-shape checking belongs to the existing static adapter")
    deployed = base / "adapter.safetensors"
    deployed.hardlink_to(original)
    row = {**manifest["tasks"][0], "adapter_path": str(deployed)}
    task = {**{key: row[key] for key in ("global_task_id", "suite", "task_id", "language")},
        "schema": bank.TASK_SCHEMA, "smoke": False, "complete": True, "all_arms_usable": True,
        "offline_diagnostic_only": True, "future_initialization_allowed": False,
        "checkpoint_selection": False, "source_trainable": 0, "writer_meta_trainable": 0,
        "heldout_gradients": False, "source": {}, "writer_checkpoint": manifest["writer_checkpoint"],
        "diagnostic_git": manifest["shared_run_contract"]["diagnostic_git"], "teacher_demo_indices": [16],
        "teacher_videos": [{"demo_index": 16, "raw_frame_count": 12, "frame_indices": [0, 5, 10, 11],
                            "sampled_frame_count": 4, "camera_view": "dual"}],
        "support": panel(True), "heldout": panel(False),
        "arms": {"free_q": {"arm": "free_q", "adapter": bank._record(original),
            "single_complete_rank16": True, "rank": 16, "alpha": 16, "target_count": 38,
            "factor_count": 76, "source_and_writer_frozen": True, "support_only_gradients": True,
            "only_q_optimized": True, "only_lora_factors_optimized": False}}}
    record = {"schema_version": bank.ADAPTER_SCHEMA, "arm": manifest["arm"],
        "global_task_id": row["global_task_id"], "original_adapter": bank._record(original),
        "single_complete_rank16": True}

    def accepted(value):
        row["diagnostic_task_manifest"] = saved(base / "task.json", value)
        record["diagnostic_task_manifest"] = row["diagnostic_task_manifest"]
        return bank.checkpoint_matches(manifest["arm"], record, row, (row["suite"], row["task_id"]), manifest)

    assert accepted(task)
    for field, key, value in (("support", "action_demos", [16] * 64),
                              ("heldout", "gradients", True),
                              ("heldout", "action_demos", [17] * 128)):
        changed = deepcopy(task)
        changed[field][key] = value
        assert not accepted(changed)
    assert not accepted({**task, "all_arms_usable": False})
    deployed.unlink()
    deployed.write_bytes(original.read_bytes())
    assert not accepted(task)  # Same byte count alone does not identify the fitted artifact.


def test_runtime_restricts_diagnostic_to_registered_initial_states():
    adapter = object.__new__(FrozenStaticTaskLoRAAdapter)
    adapter.records = {("libero_spatial", 0): {"allowed_init_state_ids": bank.STATES}}
    assert adapter.prepare_episode(suite="libero_spatial", task_id=0, init_state_id=32)
    with pytest.raises(Pi05EvaluationError, match="registered static-LoRA diagnostic"):
        adapter.prepare_episode(suite="libero_spatial", task_id=0, init_state_id=0)
