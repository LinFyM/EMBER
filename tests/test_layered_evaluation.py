from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from ember.eval_adapters import (episode_adapter_fields, inspect_static_task_lora_adapter,
                                 validate_episode_adapter_fields)
from ember.lora import LORA_B_SUFFIX, LoRATarget, identity_lora_state
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.recovery import _reinspect_adapter
from ember.pi05_lora import load_pi05_lora_contract
from ember.writer import evaluation
from ember.writer.evaluation import (EVALUATION_SCHEMA, FrozenLayeredWriterAdapter, episode_evidence,
                                     inspect_layered_writer_bank, validate_task_scope)
from ember.writer.materialization import (BANK_KIND, BANK_SCHEMA, RUN_SCHEMA, STAGE, adapter_metadata,
    condition_id, file_record, inspect_joint_checkpoint, method_metadata, paired_video_sets,
    planned_episodes, selection_contract)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = {key: f"/test/source/{key}" for key in ("source_run", "checkpoint", "model_path")}
GIT = {"branch": "", "commit": "a" * 40, "upstream": None, "dirty_paths": [],
       "authority_ref": "origin/main", "authority_contains_commit": True}


def _selection(**overrides):
    values = dict(role="development_train", task_ids=(0,), cardinality=1, arm="correct",
                  mode="per_init_ordinal", seed=7, init_state_ids=(0, 1), video_pool=tuple(range(8)))
    return selection_contract(**(values | overrides))


def _task_rows(ids, selection):
    target = json.loads((ROOT / "configs/pi05_target_data_v1/manifest.json").read_text())
    rows = []
    for task in target["tasks"]:
        if task["global_task_id"] not in ids:
            continue
        path = ROOT / "data/datasets" / target["dataset"]["revision"] / task["hdf5"]["relative_path"]
        rows.append({key: task[key] for key in ("global_task_id", "suite", "task_id", "language", "split_role")} |
                    {"teacher_source": {"path": str(path.resolve()), "bytes": task["hdf5"]["bytes"]},
                     "episodes": planned_episodes(selection, task["global_task_id"])})
    return rows, target


@pytest.fixture
def bank(tmp_path):
    checkpoint = tmp_path / "run/checkpoints/macro_00000016"
    checkpoint.mkdir(parents=True)
    run = {"schema_version": RUN_SCHEMA, "stage": STAGE, "mode": "formal", "git": GIT,
           "source": SOURCE, "config": {"observer": {"probe_seed": 1729}}, "model_config": {"horizon": 50}}
    (checkpoint.parent.parent / "run_contract.json").write_text(json.dumps(run))
    save_file({"probe": torch.zeros(50, 32)}, str(checkpoint / "ecp.safetensors"))
    for name in ("trainer_state.pt", "rank_00_state.pt"):
        (checkpoint / name).write_bytes(b"fixture - never deserialized by materialization")
    files = {path.name: {"bytes": path.stat().st_size} for path in checkpoint.iterdir()}
    (checkpoint / "checkpoint_manifest.json").write_text(json.dumps({"schema_version": "ember_ecp_checkpoint_v1",
        "stage": STAGE, "run_contract_schema": RUN_SCHEMA, "next_macro": 16, "world_size": 1, "files": files}))
    _, authority = inspect_joint_checkpoint(checkpoint)
    selection = _selection()
    rows, target = _task_rows([0], selection)
    native = next(row for row in target["tasks"] if row["global_task_id"] == 0)
    output = tmp_path / "bank"
    output.mkdir()
    lora_path = ROOT / "configs/pi05_lora_v1.json"
    state = identity_lora_state(load_pi05_lora_contract(lora_path))
    conditions = {}
    for episode in rows[0]["episodes"]:
        demos = episode["teacher_demo_indices"]
        key = condition_id(0, demos)
        if key in conditions:
            continue
        path = output / f"{key}.safetensors"
        save_file(state, str(path), metadata=adapter_metadata(key, authority))
        raw = native["demonstrations"]["episode_lengths"][demos[0]]
        indices = list(range(0, raw, 5))
        if indices[-1] != raw - 1:
            indices.append(raw - 1)
        conditions[key] = {key_: rows[0][key_] for key_ in ("suite", "task_id", "global_task_id", "language")} | {
            "condition_id": key, "teacher_demo_indices": demos, "teacher_videos": [{"demo_index": demos[0],
                "raw_frame_count": raw, "sampled_frame_count": len(indices), "frame_indices": indices}],
            "adapter": file_record(path), "writer_invocations": 1, "single_complete_rank16": True}
    manifest = {"schema_version": BANK_SCHEMA, "kind": BANK_KIND, "status": "sealed", "evaluation_role": "development_train",
        "arm": "correct", "selection": selection, "source": SOURCE, "asset_root": str(ROOT), "writer_checkpoint": authority,
        "materialization_git": GIT, "lora_contract": file_record(lora_path), "method": method_metadata(run),
        "tasks": rows, "conditions": list(conditions.values()), "single_complete_rank16": True,
        "information_wall": {"teacher_action_state_reward_terminal_reads": 0, "validation_test_gradients": False,
            "execution_adapters": 1, "action_meta_installed": False, "teacher_video_runtime_reads": 0,
            "writer_invocations_per_unique_condition": 1, "total_writer_invocations": len(conditions),
            "outcome_dependent_video_selection": False, "shuffled_reversed_wrong_no_video": False}}
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def _inspect(path):
    return inspect_layered_writer_bank(manifest_path=path, source=SOURCE, task_keys=(("libero_spatial", 0),),
        evaluation_role="development_train", require_formal=True, task_init_state_ids={("libero_spatial", 0): (0, 1)})


@pytest.mark.parametrize("k", [1, 2, 4])
def test_paired_sampling_is_deterministic_disjoint_and_outcome_independent(k):
    correct = _selection(cardinality=k, video_pool=tuple(range(16)))
    other = correct | {"arm": "same_task_other"}
    for ordinal in (0, 1, 49):
        left, right = paired_video_sets(correct, 12, ordinal)
        assert len(left) == len(set(left)) == len(right) == len(set(right)) == k
        assert not set(left) & set(right)
        assert paired_video_sets(other, 12, ordinal) == (left, right)
    assert correct["outcome_dependence"] is correct["gradient_use"] is False


def test_fixed_sets_share_one_condition_and_full_k_pool_can_support_correct_only():
    fixed = _selection(mode="fixed_per_task", cardinality=4, video_pool=(0, 1, 2, 3), fixed_videos={"0": [3, 1, 2, 0]})
    episodes = planned_episodes(fixed, 0)
    assert len({row["condition_id"] for row in episodes}) == 1
    assert all(row["video_ordinal"] == 0 and row["teacher_demo_indices"] == [0, 1, 2, 3] for row in episodes)
    with pytest.raises(ValueError, match="additional disjoint"):
        planned_episodes(fixed | {"arm": "same_task_other"}, 0)


def test_fixed_split_and_validation8_scope_are_enforced():
    train, _ = _task_rows([0], _selection())
    validate_task_scope(train, "development_train", ROOT)
    with pytest.raises(ValueError, match="excludes Test"):
        validate_task_scope(train, "test", ROOT)
    ids = [1, 3, 11, 13, 23, 26, 31, 32]
    rows, _ = _task_rows(ids, _selection(role="validation", task_ids=ids))
    validate_task_scope(rows, "validation", ROOT)
    with pytest.raises(ValueError, match="omits validation8"):
        validate_task_scope(rows[:-1], "validation", ROOT)
    with pytest.raises(ValueError, match="fixed sets"):
        _selection(role="validation", mode="fixed_per_task")


def test_inspection_dispatch_recovery_and_exact_episode_evidence(bank):
    path, _ = bank
    adapter = _inspect(path)
    task = SimpleNamespace(suite="libero_spatial", task_id=0, init_state_ids=(0, 1))
    dispatched = inspect_static_task_lora_adapter(manifest_path=path, source=SOURCE, tasks=(task,),
        evaluation_role="development_train", require_formal=True)
    assert dispatched == adapter and adapter["schema_version"] == EVALUATION_SCHEMA
    recovered = _reinspect_adapter(adapter, contract={"role": "development_train", "mode": "formal",
        "tasks": [{"suite": "libero_spatial", "task_id": 0, "init_state_ids": [0, 1]}]}, model=SOURCE)
    assert recovered == adapter
    task_row = adapter["tasks"][0]
    evidence = episode_evidence(adapter, task_row, task_row["episodes"][0])
    fields = episode_adapter_fields({"adapter": adapter}, object(), SimpleNamespace(evidence=evidence))
    assert validate_episode_adapter_fields(adapter, fields, suite="libero_spatial", task_id=0, init_state_id=0)
    assert not validate_episode_adapter_fields(adapter, fields, suite="libero_spatial", task_id=0, init_state_id=1)
    bad = copy.deepcopy(fields)
    bad["layered_writer_lora"]["teacher_demo_indices"] = [49]
    assert not validate_episode_adapter_fields(adapter, bad, suite="libero_spatial", task_id=0, init_state_id=0)
    assert not validate_episode_adapter_fields(None, fields, suite="libero_spatial", task_id=0, init_state_id=0)


@pytest.mark.parametrize("change", ["ordinal", "frames", "source", "checkpoint", "meta"])
def test_modified_pairing_or_provenance_is_rejected(bank, change):
    path, manifest = bank
    if change == "ordinal":
        manifest["tasks"][0]["episodes"][0]["video_ordinal"] = 9
    elif change == "frames":
        manifest["conditions"][0]["teacher_videos"][0]["frame_indices"] = [0]
    elif change == "source":
        manifest["source"] = SOURCE | {"checkpoint": "/wrong/source"}
    elif change == "checkpoint":
        manifest["writer_checkpoint"]["macro"] = 32
    else:
        manifest["information_wall"]["action_meta_installed"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(Pi05EvaluationError):
        _inspect(path)


def test_missing_init_states_are_rejected_before_workers_start(bank):
    path, _ = bank
    with pytest.raises(Pi05EvaluationError, match="fixed init states"):
        inspect_layered_writer_bank(manifest_path=path, source=SOURCE, task_keys=(("libero_spatial", 0),),
            evaluation_role="development_train", require_formal=True,
            task_init_state_ids={("libero_spatial", 0): tuple(range(50))})


def test_wrong_condition_file_is_rejected_even_with_valid_lora_shapes(bank):
    path, manifest = bank
    first, second = manifest["conditions"]
    first_path, second_path = Path(first["adapter"]["path"]), Path(second["adapter"]["path"])
    first_path.write_bytes(second_path.read_bytes())
    with pytest.raises(Pi05EvaluationError, match="identity"):
        _inspect(path)


def test_batched_execution_applies_independent_row_adapters_and_restores_identity(monkeypatch):
    contract = replace(load_pi05_lora_contract(ROOT / "configs/pi05_lora_v1.json"),
                       targets=(LoRATarget("linear", 3, 4),), rank=2, alpha=2)

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 4, bias=False)

        def predict_action_chunk(self, batch, *, noise, num_steps):
            assert num_steps == 10
            return self.linear(batch["input"]) + noise

    policy = Policy().eval()
    base_weight = policy.linear.weight.detach().clone()
    states = [identity_lora_state(contract) for _ in range(2)]
    for index, state in enumerate(states):
        state["linear" + LORA_B_SUFFIX].fill_(0.1 * (index + 1))
    adapter = {"kind": BANK_KIND, "schema_version": EVALUATION_SCHEMA, "source": SOURCE,
        "single_complete_rank16": True, "lora_contract": {"path": "fixture"},
        "writer_checkpoint": {}, "conditions": [{"condition_id": str(i), "adapter": {"path": str(i)}} for i in range(2)],
        "tasks": [{"suite": "libero_spatial", "task_id": 0}]}
    monkeypatch.setattr(evaluation, "load_pi05_lora_contract", lambda _path: contract)
    monkeypatch.setattr(evaluation, "_inspect_adapter_file", lambda *_args: None)
    monkeypatch.setattr(evaluation, "load_file", lambda path, **_kwargs: states[int(path)])
    runtime = FrozenLayeredWriterAdapter(policy=policy, source=SOURCE, evaluation_adapter=adapter,
        task_keys=(("libero_spatial", 0),), device=torch.device("cpu"), require_formal=True)
    prepared = [evaluation.PreparedLayeredLoRA(str(i), {}) for i in range(2)]
    x, noise = torch.randn(2, 3), torch.randn(2, 4)
    expected = torch.stack([torch.nn.functional.linear(x[i], base_weight) +
        torch.nn.functional.linear(torch.nn.functional.linear(x[i], states[i]["linear.lora_A.default.weight"]),
                                   states[i]["linear.lora_B.default.weight"]) + noise[i] for i in range(2)])
    runtime.install(prepared[0])
    observed = runtime.predict_action_chunk(prepared, {"input": x}, noise=noise, num_steps=10)
    torch.testing.assert_close(observed, expected)
    assert not policy.linear.lora_B["default"].weight.count_nonzero()
    with pytest.raises(Pi05EvaluationError, match="batch"):
        runtime.predict_action_chunk(prepared[:1], {"input": x}, noise=noise, num_steps=10)
    runtime.close()
