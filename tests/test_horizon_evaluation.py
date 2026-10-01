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
from ember.writer import evaluation, materialization
from ember.writer.evaluation import (EVALUATION_SCHEMA, FrozenHorizonWriterAdapter, episode_evidence,
                                     inspect_horizon_writer_bank, validate_task_scope)
from ember.writer.materialization import (BANK_KIND, BANK_SCHEMA, CONFIG_SCHEMA, RUN_SCHEMA, STAGE, TRAINING_SCHEMA, UPDATE_VERSION, adapter_metadata,
    condition_id, file_record, inspect_writer_checkpoint, method_metadata, paired_video_sets,
    planned_episodes, selection_contract)
from ember.writer.runtime import MODEL_DEFAULTS
from ember.writer.learning_data import EVENT_SCHEMA
from ember.writer.materialization import observer_mode_contract


ROOT = Path(__file__).resolve().parents[1]
SOURCE = {key: f"/test/source/{key}" for key in ("source_run", "checkpoint", "model_path")}
GIT = {"branch": "", "commit": "a" * 40, "upstream": None, "dirty_paths": [],
       "authority_ref": "origin/main", "authority_contains_commit": True}


def _selection(**overrides):
    values = dict(role="development_train", task_ids=(0,), cardinality=1, arm="correct",
                  mode="per_init_ordinal", seed=7, init_state_ids=(0, 1), video_pool=tuple(range(50)))
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
def bank(tmp_path, request):
    checkpoint = tmp_path / "run/checkpoints/macro_00000016"
    checkpoint.mkdir(parents=True)
    run = {"schema_version": RUN_SCHEMA, "stage": STAGE, "mode": "formal", "git": GIT,
           "source": copy.deepcopy(SOURCE), "config": {"update_version": UPDATE_VERSION, "data": {"version": "fixture_supervised_data_v1"}, "observer": {"probe_seed": 1729, "camera_view": "dual"}, "execution_precision": "native_bf16_writer_fm_fp32_lora"}, "model_config": {"horizon": 50}}
    run["model_config"] = dict(MODEL_DEFAULTS)
    run["config"]["data"] = {"version": EVENT_SCHEMA,
                            "action_start_offset": 1, "query_alignment": "post_action_observation_future_control_v1",
                            "maximum_updates": 1200}
    run["config"]["schema_version"] = CONFIG_SCHEMA
    run["config"]["optimization"] = {"loss": "main_fm_plus_video_teaching"}
    run["config"]["model"] = dict(run["model_config"])
    run["config"]["observer"].update(observer_mode_contract(run["model_config"]))
    run["config"]["observer"]["frame_chunk"] = 4
    (checkpoint.parent.parent / "run_contract.json").write_text(json.dumps(run))
    save_file({"probe": torch.zeros(50, 32)}, str(checkpoint / "ecp.safetensors"))
    torch.save({"schema_version": "ember_ecp_checkpoint_v1", "stage": STAGE, "next_macro": 16,
        "optimizer": {"state": {0: {"exp_avg": torch.ones(4)}}},
        "training_state": {"schema_version": TRAINING_SCHEMA, "updates": 16,
            "update_version": UPDATE_VERSION, "data_version": run["config"]["data"]["version"]}},
        checkpoint / "trainer_state.pt")
    (checkpoint / "rank_00_state.pt").write_bytes(b"fixture - RNG is never deserialized by materialization")
    files = {path.name: {"bytes": path.stat().st_size} for path in checkpoint.iterdir()}
    (checkpoint / "checkpoint_manifest.json").write_text(json.dumps({"schema_version": "ember_ecp_checkpoint_v1",
        "stage": STAGE, "run_contract_schema": RUN_SCHEMA, "next_macro": 16, "world_size": 1, "files": files}))
    _, authority = inspect_writer_checkpoint(checkpoint)
    selection = _selection(**getattr(request, "param", {}))
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
        "arm": "correct", "selection": selection, "source": copy.deepcopy(SOURCE), "asset_root": str(ROOT), "writer_checkpoint": authority,
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
    return inspect_horizon_writer_bank(manifest_path=path, source=SOURCE, task_keys=(("libero_spatial", 0),),
        evaluation_role="development_train", require_formal=True, task_init_state_ids={("libero_spatial", 0): (0, 1)})


@pytest.mark.parametrize("k", [1, 2, 4])
def test_paired_sampling_is_deterministic_disjoint_and_outcome_independent(k):
    correct = _selection(cardinality=k, video_pool=tuple(range(50)))
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
    with pytest.raises(ValueError, match="fixed split"):
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
    bad["horizon_writer_lora"]["teacher_demo_indices"] = [49]
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
        inspect_horizon_writer_bank(manifest_path=path, source=SOURCE, task_keys=(("libero_spatial", 0),),
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
    runtime = FrozenHorizonWriterAdapter(policy=policy, source=SOURCE, evaluation_adapter=adapter,
        task_keys=(("libero_spatial", 0),), device=torch.device("cpu"), require_formal=True)
    prepared = [evaluation.PreparedHorizonLoRA(str(i), {}) for i in range(2)]
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


def test_method_metadata_binds_repeated_full_reads_and_single_full_lora():
    model = dict(MODEL_DEFAULTS)
    run = {"model_config": model, "config": {"update_version": UPDATE_VERSION,
        "observer": observer_mode_contract(model), "execution_precision": "native_bf16_writer_fm_fp32_lora"}}
    method = method_metadata(run)
    assert method["native_response_shape"] == [50, 1024]
    assert method["generated_tensor_count"] == 76 and method["execution_rank"] == 16
    assert method["deployment_frozen_source_vjp"] is False
    assert method["source_parameter_training"] is False
    assert method["deployment_teacher_labels_loss_optimizer"] is False
    assert method["training_objective"] == "main_fm_plus_video_teaching" and method["update_version"] == UPDATE_VERSION
    assert method['native_image_tokens'] == 256
    assert method['visual_token_source'] == 'actual_final_256_image_patches_and_exact_task_span_tokens'
    assert method['camera'] == 'agentview_rotated_180'
    assert method['native_read'] == 'repeated_content_position_attention_over_all_50_raw_H_values'
    assert method['video_order'] == 'causal_RoPE_with_real_frame_positions_and_ordered_adjacent_roles'
    for arm in ('cross_suite_wrong', 'shuffled', 'reversed'):
        control = method_metadata(run, arm)
        assert control['control_transform'] == 'real_agentview_camera_RGB_before_complete_Writer_forward'
    assert method_metadata(run, 'no_video')['control_transform'] == 'identity_zero_delta_without_RGB_reads'


@pytest.mark.parametrize("field,value", [("schema_version", "ember_language_axial_writer_run_v1"),
    ("stage", "language_axial_writer_fresh"),
    ("update_version", "full_ab_pure_fm_joint_text_vl_action_meta_v1"),
    ("schema_version", "ember_horizon_relation_writer_joint_run_v1"),
    ("schema_version", "ember_process_pullback_writer_run_v1"),
    ("schema_version", "ember_native_correction_writer_run_v1"),
    ("stage", "horizon_relation_writer_fresh_fm_rl_joint"), ("mode", "profile"),
    ("stage", "native_correction_writer_fresh"),
    ("schema_version", "ember_semantic_path_writer_run_v1"),
    ("schema_version", "ember_local_field_writer_run_v1"),
    ("stage", "semantic_path_writer_fresh"),
    ("stage", "local_field_writer_fresh"),
    ("update_version", "local_field_main_fm_joint_credit_v1"),
    ("update_version", "semantic_path_main_fm_joint_credit_v1"),
    ("update_version", "native_correction_main_fm_joint_credit_v1"),
    ("update_version", "joint_fm_rl_same_version_v1"), ("execution_precision", "outer_bf16")])
def test_old_joint_or_profile_checkpoint_cannot_be_materialized_as_supervised(bank, field, value):
    _, manifest = bank
    checkpoint = Path(manifest["writer_checkpoint"]["path"])
    run_path = checkpoint.parent.parent / "run_contract.json"
    run = json.loads(run_path.read_text())
    (run["config"] if field in {"execution_precision", "update_version"} else run)[field] = value
    run_path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="formal supervised"):
        inspect_writer_checkpoint(checkpoint)


@pytest.mark.parametrize("field", ["schema", "architecture", "action_horizon", "camera_view", "horizon_read"])
def test_shape_compatible_old_writer_requires_its_frozen_runtime(bank, field):
    _, manifest = bank
    checkpoint = Path(manifest["writer_checkpoint"]["path"])
    run_path = checkpoint.parent.parent / "run_contract.json"
    run = json.loads(run_path.read_text())
    del run["model_config"][field]
    del run["config"]["model"][field]
    run_path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="architecture"):
        inspect_writer_checkpoint(checkpoint)


def test_checkpoint_rejects_shape_compatible_run_model_disagreement(bank):
    _, manifest = bank
    checkpoint = Path(manifest["writer_checkpoint"]["path"])
    run_path = checkpoint.parent.parent / "run_contract.json"
    run = json.loads(run_path.read_text())
    run["config"]["model"]["max_frames_per_encoder_call"] *= 2
    run_path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="disagree"):
        inspect_writer_checkpoint(checkpoint)


def test_checkpoint_accepts_registered_video_teaching_architecture(bank):
    _, manifest = bank
    observed, _ = inspect_writer_checkpoint(Path(manifest['writer_checkpoint']['path']))
    method = method_metadata(observed)
    assert method['model_config']['camera_view'] == 'agentview'
    assert method['model_config']['horizon_read'] == 'repeated_full'
    assert method['model_config']['procedure_blocks'] == 2
    assert method['model_config']['semantic_core_blocks'] == 2


@pytest.mark.parametrize('damage', ['observer_camera', 'observer_read', 'observer_patches',
                                    'observer_order', 'model_horizon', 'run_model'])
def test_checkpoint_rejects_native_model_and_observer_disagreement(bank, damage):
    _, manifest = bank
    checkpoint = Path(manifest['writer_checkpoint']['path'])
    path = checkpoint.parent.parent / 'run_contract.json'
    run = json.loads(path.read_text())
    if damage.startswith('observer_'):
        field = {'observer_camera': 'camera_view', 'observer_read': 'horizon_read',
                 'observer_patches': 'native_inputs', 'observer_order': 'video_order'}[damage]
        run['config']['observer'][field] = 'obsolete_read'
    elif damage == 'model_horizon':
        run['model_config']['action_horizon'] = run['config']['model']['action_horizon'] = 25
    else:
        run['config']['model']['procedure_blocks'] = 3
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match='architecture|formal supervised|disagree'):
        inspect_writer_checkpoint(checkpoint)


def test_checkpoint_inspection_keeps_optimizer_tensors_on_meta(bank, monkeypatch):
    _, manifest = bank
    checkpoint = Path(manifest["writer_checkpoint"]["path"])
    load = torch.load
    seen = []

    def metadata_load(path, **kwargs):
        assert kwargs == {"map_location": "meta", "mmap": True, "weights_only": True}
        result = load(path, **kwargs)
        assert result["optimizer"]["state"][0]["exp_avg"].is_meta
        seen.append(Path(path).name)
        return result

    monkeypatch.setattr(torch, "load", metadata_load)
    inspect_writer_checkpoint(checkpoint)
    assert seen == ["trainer_state.pt"]


@pytest.mark.parametrize("field,value", [("action_start_offset", 0), ("action_start_offset", True),
    ("query_alignment", "same_index"), ("version", "train24_cross_episode_k1_pretrained_video_v1")])
def test_checkpoint_refuses_old_execution_label_contract(bank, field, value):
    _, manifest = bank
    checkpoint = Path(manifest["writer_checkpoint"]["path"])
    run_path = checkpoint.parent.parent / "run_contract.json"
    run = json.loads(run_path.read_text())
    run["config"]["data"][field] = value
    run_path.write_text(json.dumps(run))
    with pytest.raises(ValueError):
        inspect_writer_checkpoint(checkpoint)


@pytest.mark.parametrize("field,value", [("schema_version", "ember_horizon_joint_training_state_v1"),
    ("updates", 15), ("update_version", "old_joint_version"), ("data_version", "different_data")])
def test_supervised_checkpoint_rejects_training_state_mismatch(bank, field, value):
    _, manifest = bank
    checkpoint = Path(manifest["writer_checkpoint"]["path"])
    trainer_path = checkpoint / "trainer_state.pt"
    trainer = torch.load(trainer_path, weights_only=True)
    trainer["training_state"][field] = value
    torch.save(trainer, trainer_path)
    manifest_path = checkpoint / "checkpoint_manifest.json"
    checkpoint_manifest = json.loads(manifest_path.read_text())
    checkpoint_manifest["files"]["trainer_state.pt"]["bytes"] = trainer_path.stat().st_size
    manifest_path.write_text(json.dumps(checkpoint_manifest))
    with pytest.raises(ValueError, match="training state or optimizer-update cursor"):
        inspect_writer_checkpoint(checkpoint)


@pytest.mark.parametrize("bank", [{"init_state_ids": tuple(range(32, 36))}], indirect=True)
def test_train_diagnostic_bank_requires_exact_evaluator_state_subset(bank):
    path, _ = bank
    arguments = dict(manifest_path=path, source=SOURCE, task_keys=(("libero_spatial", 0),),
                     evaluation_role="development_train", require_formal=True)
    adapter = inspect_horizon_writer_bank(**arguments,
        task_init_state_ids={("libero_spatial", 0): tuple(range(32, 36))})
    assert [row["init_state_id"] for row in adapter["tasks"][0]["episodes"]] == list(range(32, 36))
    for states in (tuple(range(4)), tuple(range(32, 37)), tuple(range(50))):
        with pytest.raises(Pi05EvaluationError, match="same exact fixed init states"):
            inspect_horizon_writer_bank(**arguments, task_init_state_ids={("libero_spatial", 0): states})


@pytest.mark.parametrize("overrides", [
    {"role": "validation"}, {"state_count": 10}, {"init_state_ids": [0, 1, 2, 3, 4]},
    {"init_state_ids": [32, 33, 34, 35, 35]}, {"init_state_ids": [36, 35, 34, 33, 32]},
])
def test_explicit_train_diagnostic_request_rejects_wrong_scope(overrides):
    values = {"role": "development_train", "init_state_ids": list(range(32, 36)), "state_count": 4}
    with pytest.raises(ValueError, match="development_train states32..35 and count4"):
        materialization.request_init_state_ids(**(values | overrides))


def test_validation_selection_rejects_offsets_even_for_direct_api():
    with pytest.raises(ValueError, match="canonical zero-based prefix"):
        _selection(role="validation", init_state_ids=(32, 33, 34, 35, 36))


def _diagnostic_args(**overrides):
    return SimpleNamespace(**({"role": "development_train", "mode": "screen", "state_count": 5,
        "init_state_ids": tuple(range(32, 37)), "occupancy_capture_selection": None} | overrides))


@pytest.mark.parametrize("count", [4, 5])
def test_evaluator_diagnostic_scope_state_subset_and_queue_are_exact(tmp_path, count):
    from ember.pi05_eval.preparation import _select_init_states, _task_subset_tasks, shards_from_contract
    from ember.pi05_eval_contract import TargetTaskContract

    task = TargetTaskContract("libero_spatial", 0, "train", "task", "folder", "task.bddl", 1,
                              "states", 1, 50, 220, tuple(range(5)))
    states = tuple(range(32, 32 + count))
    selected = _select_init_states(_diagnostic_args(state_count=count, init_state_ids=states), (task,))
    assert selected[0].init_state_ids == states
    assert task.init_state_ids == tuple(range(5))
    shards = shards_from_contract({"tasks": [{"suite": task.suite, "task_id": 0,
        "horizon": 220, "init_state_ids": selected[0].init_state_ids}],
        "parallel": {"envs_per_replica": 8, "shard_target_cost": 4160,
                     "physical_gpu_count": 1, "replicas_per_gpu": 1}})
    assert sorted(state for shard in shards for state in shard.init_state_ids) == list(states)
    path = tmp_path / "subset.json"
    manifest = {"schema_version": "ember_pi05_task_subset_selection_v1", "role": "development_train",
        "mode": "screen", "state_count": count, "init_state_ids": list(states),
        "task_ordinals": [0], "global_task_ids": [0],
        "tasks": [{"global_task_id": 0, "suite": "libero_spatial", "task_id": 0}],
        "outcome_dependence": False, "validation_use": False, "test_use": False}
    path.write_text(json.dumps(manifest))
    args = _diagnostic_args(task_subset_selection=path, state_count=count, init_state_ids=states)
    subset, record = _task_subset_tasks(args, selected, adapter_kind="static_task_lora")
    assert subset == selected and record["init_state_ids"] == list(states)
    for ids in (None, list(range(5)), list(states[:-1])):
        path.write_text(json.dumps(manifest | {"init_state_ids": ids}))
        with pytest.raises(Pi05EvaluationError, match="subset selection changed"):
            _task_subset_tasks(args, selected, adapter_kind="static_task_lora")


@pytest.mark.parametrize("change", [{"role": "validation"}, {"role": "test"}, {"mode": "formal"},
    {"state_count": 50}, {"init_state_ids": tuple(range(5))}, {"occupancy_capture_selection": Path("x")}])
def test_evaluator_explicit_offsets_reject_other_roles_counts_and_modes(change):
    from ember.pi05_eval.preparation import _explicit_diagnostic_states

    with pytest.raises(Pi05EvaluationError, match="development_train screen"):
        _explicit_diagnostic_states(_diagnostic_args(**change))


def test_evaluator_cli_accepts_diagnostic_ids_and_retains_count_argument():
    import argparse
    from scripts.evaluate_pi05 import _add_prepare_arguments

    parser = argparse.ArgumentParser()
    _add_prepare_arguments(parser)
    args = parser.parse_args(["--source-run", "/source", "--checkpoint", "/checkpoint",
        "--tokenizer-path", "/tokenizer", "--output-dir", "/output", "--role", "development_train",
        "--mode", "screen", "--replicas-per-gpu", "1", "--state-count", "5",
        "--init-state-ids", "32,33,34,35,36"])
    assert args.init_state_ids == tuple(range(32, 37)) and args.state_count == 5
