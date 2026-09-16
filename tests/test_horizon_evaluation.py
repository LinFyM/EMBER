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
from ember.pi05_source_checkpoint import Pi05SourceTrainingError
from ember.pi05_eval.recovery import _reinspect_adapter
from ember.pi05_lora import load_pi05_lora_contract
from ember.writer import evaluation, materialization
from ember.writer.evaluation import (EVALUATION_SCHEMA, FrozenHorizonWriterAdapter, episode_evidence,
                                     inspect_horizon_writer_bank, validate_task_scope)
from ember.writer.materialization import (BANK_KIND, BANK_SCHEMA, RUN_SCHEMA, STAGE, TRAINING_SCHEMA, UPDATE_VERSION, adapter_metadata,
    condition_id, file_record, inspect_writer_checkpoint, method_metadata, paired_video_sets,
    planned_episodes, selection_contract)
from ember.writer.runtime import MODEL_DEFAULTS
from ember.writer.training import observer_mode_contract, extension_record_path


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
    run["config"]["data"] = {"version": "v52_full_video_cross_episode_events_v1",
                            "action_start_offset": 1, "query_alignment": "post_action_observation_future_control_v1",
                            "maximum_updates": 1200}
    run["config"]["schema_version"] = "ember_language_axial_writer_config_v1"
    run["config"]["optimization"] = {"loss": "main_fm"}
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


@pytest.fixture
def resident_materialization(tmp_path, monkeypatch):
    from ember.writer import learning_data, runtime

    class State(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.writer = torch.nn.Linear(1, 1, bias=False)
            self.meta = torch.nn.Linear(1, 1, bias=False)
            self.vl_meta = torch.nn.Linear(1, 1, bias=False)
            self.register_buffer("probe", torch.randn(50, 32, generator=torch.Generator().manual_seed(1729)))
            self.loads = 0

        def load_state_dict(self, state, strict=True):
            assert strict is True
            self.loads += 1
            return super().load_state_dict(state, strict=strict)

    state = State()
    instance = SimpleNamespace(state=state, policy=torch.nn.Linear(1, 1), source=SOURCE,
        lora=load_pi05_lora_contract(ROOT / "configs/pi05_lora_v1.json"), observer=SimpleNamespace(probe=state.probe))
    source = tmp_path / "teacher.hdf5"
    source.write_bytes(b"CPU orchestration fixture")
    task = SimpleNamespace(suite="libero_spatial", suite_task_id=0,
                           authority=SimpleNamespace(task_id=0, path=source, language="exact task language"),
                           episode_lengths=(100,) * 50)
    runs, requests, builds = {}, [], []
    for step, value, arm in ((16, 1., "correct"), (48, 2., "same_task_other")):
        checkpoint = tmp_path / f"run/checkpoints/macro_{step:08d}"
        checkpoint.mkdir(parents=True)
        tensors = {name: tensor.detach().clone() for name, tensor in state.state_dict().items()}
        tensors["writer.weight"].fill_(value)
        tensors["meta.weight"].fill_(value * 10)
        tensors["vl_meta.weight"].fill_(value * 100)
        save_file(tensors, str(checkpoint / "ecp.safetensors"))
        runs[checkpoint] = {"source": copy.deepcopy(SOURCE), "model_config": dict(MODEL_DEFAULTS),
            "config": {"update_version": UPDATE_VERSION, "execution_precision": "native_bf16_writer_fm_fp32_lora",
                "model": dict(MODEL_DEFAULTS), "observer": {"probe_seed": 1729, "meta_rank": 4, "frame_chunk": 4,
                                                            **observer_mode_contract(MODEL_DEFAULTS)}}}
        requests.append({"checkpoint": str(checkpoint), "output": str(tmp_path / f"output_{step}"),
            "role": "development_train", "task_ids": [0], "k": 1, "arm": arm,
            "selection_mode": "fixed_per_task", "video_pool": [0, 1, 2, 3], "state_count": 10, "seed": 7})
    monkeypatch.setattr(materialization, "git_state", lambda _root: GIT)
    monkeypatch.setattr(materialization, "inspect_writer_checkpoint", lambda path: (runs[path], {"path": str(path)}))
    monkeypatch.setattr(learning_data, "load_learning_tasks", lambda *_args, **_kwargs: {0: task})
    monkeypatch.setattr(evaluation, "validate_task_scope", lambda *_args: None)
    monkeypatch.setattr(materialization, "RawTeacherVideoStore", lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None))

    def build(asset_root, config, device):
        builds.append((asset_root, config, device))
        instance.observer.frame_chunk = config["observer"]["frame_chunk"]
        return instance

    def compile_condition(current, _store, _task, demos, _output, _checkpoint):
        assert current is instance and current.observer.probe is current.state.probe
        path = _output / f"{condition_id(0, demos)}.safetensors"
        save_file({"value": state.writer.weight.detach()}, str(path))
        return {"condition_id": condition_id(0, demos), "global_task_id": 0, "teacher_demo_indices": list(demos),
                "adapter": file_record(path), "teacher_videos": [{"sampled_frame_count": 1}],
                "writer_value": float(state.writer.weight), "meta_value": float(state.meta.weight),
                "vl_meta_value": float(state.vl_meta.weight)}

    monkeypatch.setattr(runtime, "build_runtime", build)
    monkeypatch.setattr(materialization, "_compile_condition", compile_condition)
    return requests, runs, builds, state


@pytest.mark.parametrize("native_frame_chunk", [None, 16])
def test_resident_batch_loads_once_and_reloads_entire_checkpoint_per_manifest(resident_materialization, tmp_path,
                                                                           native_frame_chunk):
    requests, _, builds, state = resident_materialization
    paths = materialization.materialize_requests(asset_root=ROOT, requests=requests, device=torch.device("cpu"),
                                                 native_frame_chunk=native_frame_chunk)
    assert len(builds) == 1 and builds[0][1]["model"] == MODEL_DEFAULTS
    assert state.loads == 2
    for index, path in enumerate(paths):
        manifest = json.loads(path.read_text())
        assert manifest["schema_version"] == BANK_SCHEMA
        assert manifest["writer_checkpoint"]["path"] == requests[index]["checkpoint"]
        assert manifest["arm"] == requests[index]["arm"]
        assert manifest["conditions"][0]["writer_value"] == index + 1
        assert manifest["conditions"][0]["meta_value"] == (index + 1) * 10
        assert manifest["conditions"][0]["vl_meta_value"] == (index + 1) * 100
        assert manifest["method"]["training_objective"] == "main_fm"
        assert manifest["materialization_execution"]["native_frame_chunk"] == (native_frame_chunk or 4)
        assert manifest["method"]["observer"]["frame_chunk"] == 4
        assert manifest["information_wall"]["total_writer_invocations"] == 1
        assert len(manifest["tasks"][0]["episodes"]) == 10
    materialization.materialize(asset_root=ROOT, checkpoint=Path(requests[0]["checkpoint"]),
        output=tmp_path / "single", selection=_selection(mode="fixed_per_task"), device=torch.device("cpu"))
    assert len(builds) == 2 and state.loads == 3 and float(state.meta.weight) == 10


@pytest.mark.parametrize("cardinality", [2, 4])
def test_pullback_materialization_rejects_untrained_cardinality_before_runtime_build(resident_materialization,
                                                                                  cardinality):
    requests, _, builds, _ = resident_materialization
    requests[0]["k"] = cardinality
    with pytest.raises(ValueError, match="trained K=1"):
        materialization.materialize_requests(asset_root=ROOT, requests=requests, device=torch.device("cpu"))
    assert not builds


@pytest.mark.parametrize("field", ["source", "model_config", "observer", "camera_view"])
def test_resident_batch_rejects_cross_contract_reuse_before_loading(resident_materialization, field):
    requests, runs, builds, _ = resident_materialization
    changed = runs[Path(requests[1]["checkpoint"])]
    if field == "camera_view":
        changed["config"]["observer"]["camera_view"] = "agentview"
    elif field == "observer":
        changed["config"][field]["probe_seed"] += 1
    else:
        changed[field]["different_contract"] = True
    with pytest.raises(ValueError, match="identical source, model, and observer"):
        materialization.materialize_requests(asset_root=ROOT, requests=requests, device=torch.device("cpu"))
    assert not builds and not any(Path(request["output"]).exists() for request in requests)


def test_resident_batch_rejects_output_collision_and_per_request_asset_roots(resident_materialization):
    requests, _, builds, _ = resident_materialization
    with pytest.raises(ValueError, match="distinct new directories"):
        materialization.materialize_requests(asset_root=ROOT, requests=[requests[0], requests[0]], device=torch.device("cpu"))
    with pytest.raises(ValueError, match="asset root and device"):
        materialization.materialize_requests(asset_root=ROOT, requests=[requests[0] | {"asset_root": "/different"}], device=torch.device("cpu"))
    assert not builds


def test_checkpoint_change_preserves_episode_mapping(resident_materialization):
    requests, _, _, _ = resident_materialization
    requests[1]["arm"] = "correct"
    paths = materialization.materialize_requests(asset_root=ROOT, requests=requests, device=torch.device("cpu"))
    left, right = (json.loads(path.read_text()) for path in paths)
    assert left["tasks"] == right["tasks"]
    assert left["writer_checkpoint"] != right["writer_checkpoint"]
    assert left["conditions"][0]["writer_value"] != right["conditions"][0]["writer_value"]


@pytest.mark.parametrize("damage", ["checkpoint", "source", "method", "video", "frames"])
def test_condition_reuse_rejects_identity_and_generation_mismatch(bank, damage):
    path, manifest = bank
    run, checkpoint = inspect_writer_checkpoint(Path(manifest["writer_checkpoint"]["path"]))
    if damage == "checkpoint":
        manifest["writer_checkpoint"]["macro"] += 1
    elif damage == "source":
        manifest["source"]["checkpoint"] = "/different/source"
    elif damage == "method":
        manifest["method"]["frame_stride"] = 10
    elif damage == "video":
        manifest["conditions"][0]["teacher_demo_indices"] = [49]
    else:
        manifest["conditions"][0]["teacher_videos"][0]["frame_indices"][0] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        materialization._reusable_conditions(path, asset_root=ROOT, run=run,
            checkpoint=checkpoint, selection=_selection())


def test_materialization_reuses_valid_loras_and_compiles_only_missing_video(bank, tmp_path, monkeypatch):
    from ember.writer import learning_data, runtime

    path, old = bank
    checkpoint = Path(old["writer_checkpoint"]["path"])
    probe = torch.randn(50, 32, generator=torch.Generator().manual_seed(1729))
    save_file({"probe": probe}, str(checkpoint / "ecp.safetensors"))
    state = torch.nn.Module()
    state.register_buffer("probe", probe.clone())
    lora = load_pi05_lora_contract(ROOT / "configs/pi05_lora_v1.json")
    instance = SimpleNamespace(state=state, policy=torch.nn.Linear(1, 1), lora=lora, source=SOURCE,
                               observer=SimpleNamespace(frame_chunk=4))
    row = old["tasks"][0]
    target = json.loads((ROOT / "configs/pi05_target_data_v1/manifest.json").read_text())
    native = next(task for task in target["tasks"] if task["global_task_id"] == 0)
    task = SimpleNamespace(suite=row["suite"], suite_task_id=row["task_id"],
        authority=SimpleNamespace(task_id=0, language=row["language"], path=Path(row["teacher_source"]["path"])),
        episode_lengths=tuple(native["demonstrations"]["episode_lengths"]))
    # This orchestration test uses the sealed fixture provenance, without needing the real dataset.
    monkeypatch.setattr(materialization, "file_record", lambda path: dict(row["teacher_source"])
        if Path(path) == task.authority.path else file_record(path))
    monkeypatch.setattr(materialization, "git_state", lambda _: GIT)
    monkeypatch.setattr(runtime, "build_runtime", lambda *args: instance)
    monkeypatch.setattr(learning_data, "load_learning_tasks", lambda *args, **kwargs: {0: task})
    monkeypatch.setattr(materialization, "RawTeacherVideoStore", lambda *args, **kwargs: SimpleNamespace(close=lambda: None))
    compiled = []

    def compile_condition(_runtime, _store, _task, demos, output, authority):
        compiled.append(tuple(demos))
        key = condition_id(0, demos)
        destination = output / f"{key}.safetensors"
        save_file(identity_lora_state(lora), str(destination), metadata=adapter_metadata(key, authority))
        raw = native["demonstrations"]["episode_lengths"][demos[0]]
        indices = list(range(0, raw, 5))
        if indices[-1] != raw - 1:
            indices.append(raw - 1)
        return {key_: row[key_] for key_ in ("global_task_id", "suite", "task_id", "language")} | {
            "condition_id": key, "teacher_demo_indices": list(demos),
            "teacher_videos": [{"demo_index": demos[0], "raw_frame_count": raw,
                                "sampled_frame_count": len(indices), "frame_indices": indices}],
            "adapter": file_record(destination), "writer_invocations": 1, "single_complete_rank16": True}

    monkeypatch.setattr(materialization, "_compile_condition", compile_condition)
    # An obsolete selection declaration does not invalidate individually proved
    # task/video LoRAs; it must never become the new episode mapping.
    old["selection"].pop("schedule")
    path.write_text(json.dumps(old))
    selected = _selection(init_state_ids=(0, 1, 2))
    output = materialization.materialize(asset_root=ROOT, checkpoint=checkpoint,
        output=tmp_path / "repaired", selection=selected, device=torch.device("cpu"), reuse_manifest=path)
    repaired = json.loads(output.read_text())
    assert len(compiled) == 1
    assert repaired["compilation"]["reused_conditions"] == 2
    assert repaired["compilation"]["new_conditions"] == 1
    assert repaired["tasks"][0]["episodes"] == planned_episodes(selected, 0)
    for condition in old["conditions"]:
        reused = output.parent / Path(condition["adapter"]["path"]).name
        assert reused.stat().st_ino == Path(condition["adapter"]["path"]).stat().st_ino
    verified = inspect_horizon_writer_bank(manifest_path=output, source=SOURCE,
        task_keys=(("libero_spatial", 0),), evaluation_role="development_train", require_formal=True,
        task_init_state_ids={("libero_spatial", 0): (0, 1, 2)})
    assert len(verified["conditions"]) == 3


def test_batch_cli_reads_list_and_rejects_mixed_single_request_flags(tmp_path, monkeypatch, capsys):
    requests = [{"checkpoint": "/checkpoint", "output": "/output", "role": "development_train", "task_ids": [0], "k": 1}]
    path = tmp_path / "requests.json"
    path.write_text(json.dumps(requests))
    calls = []
    monkeypatch.setattr(materialization, "materialize_requests", lambda **kwargs: calls.append(kwargs) or [Path("/output/manifest.json")])
    monkeypatch.setattr(torch, "set_num_threads", lambda _threads: None)
    argv = ["materialize_writer.py", "--requests-json", str(path), "--asset-root", str(ROOT),
            "--device", "cpu", "--native-frame-chunk", "16"]
    monkeypatch.setattr("sys.argv", argv)
    materialization.main()
    assert calls == [{"asset_root": ROOT, "requests": requests, "device": torch.device("cpu"),
                      "native_frame_chunk": 16, "cpu_threads": 4}]
    assert "/output/manifest.json" in capsys.readouterr().out
    for option in (("--arm", "same_task_other"), ("--seed", "7")):
        monkeypatch.setattr("sys.argv", [*argv, *option])
        with pytest.raises(SystemExit) as error:
            materialization.main()
        assert error.value.code == 2 and len(calls) == 1


@pytest.mark.parametrize('camera_view,horizon_read,patches', [('agentview', 'fixed_mean', 256), ('dual', 'learned', 512)])
def test_method_metadata_binds_complete_video_reads_and_single_full_lora(camera_view, horizon_read, patches):
    model = MODEL_DEFAULTS | {'camera_view': camera_view, 'horizon_read': horizon_read}
    run = {"model_config": model, "config": {"update_version": UPDATE_VERSION,
        "observer": observer_mode_contract(model), "execution_precision": "native_bf16_writer_fm_fp32_lora"}}
    method = method_metadata(run)
    assert method["native_response_shape"] == [50, 1024]
    assert method["generated_tensor_count"] == 76 and method["execution_rank"] == 16
    assert method["deployment_frozen_source_vjp"] is False
    assert method["source_parameter_training"] is False
    assert method["deployment_teacher_labels_loss_optimizer"] is False
    assert method["training_objective"] == "main_fm" and method["update_version"] == UPDATE_VERSION
    assert method['native_image_tokens'] == patches
    assert method['visual_token_source'] == f'actual_final_{patches}_image_patches_and_exact_task_span_tokens'
    assert method['camera'] == ('agentview_and_eye_in_hand_rotated_180' if camera_view == 'dual' else 'agentview_rotated_180')
    assert method['native_read'] == ('all_50_horizon_positions_to_learned_content_position_read_uniform_init'
                                   if horizon_read == 'learned' else 'all_50_horizon_positions_to_fixed_mean_zero_query_and_bias')
    for arm in ('cross_suite_wrong', 'shuffled', 'reversed'):
        control = method_metadata(run, arm)
        assert control['control_transform'] == f'real_{camera_view}_camera_RGB_before_complete_Writer_forward'
    assert method_metadata(run, 'no_video')['control_transform'] == 'identity_zero_delta_without_RGB_reads'


@pytest.mark.parametrize("field,value", [("schema_version", "ember_horizon_relation_writer_joint_run_v1"),
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


@pytest.mark.parametrize('camera_view,horizon_read', [('agentview', 'fixed_mean'), ('dual', 'learned')])
def test_checkpoint_accepts_each_registered_mode_and_keeps_its_method_identity(bank, camera_view, horizon_read):
    _, manifest = bank
    checkpoint = Path(manifest['writer_checkpoint']['path'])
    path = checkpoint.parent.parent / 'run_contract.json'
    run = json.loads(path.read_text())
    run['model_config'].update(camera_view=camera_view, horizon_read=horizon_read)
    run['config']['model'] = dict(run['model_config'])
    run['config']['observer'].update(observer_mode_contract(run['model_config']))
    path.write_text(json.dumps(run))
    observed, _ = inspect_writer_checkpoint(checkpoint)
    method = method_metadata(observed)
    assert method['model_config']['camera_view'] == camera_view
    assert method['model_config']['horizon_read'] == horizon_read


@pytest.mark.parametrize('damage', ['observer_camera', 'observer_read', 'observer_patches', 'model_pair', 'run_model'])
def test_checkpoint_rejects_camera_and_horizon_mode_disagreement(bank, damage):
    _, manifest = bank
    checkpoint = Path(manifest['writer_checkpoint']['path'])
    path = checkpoint.parent.parent / 'run_contract.json'
    run = json.loads(path.read_text())
    single = run['model_config'] | {'camera_view': 'agentview', 'horizon_read': 'fixed_mean'}
    if damage.startswith('observer_'):
        field = {'observer_camera': 'camera_view', 'observer_read': 'horizon_read', 'observer_patches': 'native_inputs'}[damage]
        run['config']['observer'][field] = observer_mode_contract(single)[field]
    elif damage == 'model_pair':
        run['model_config']['horizon_read'] = run['config']['model']['horizon_read'] = 'fixed_mean'
    else:
        run['config']['model'] = single
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


@pytest.mark.parametrize("budget,macro,legacy", [(1800, 1500, True), (2100, 2100, False)])
def test_extended_checkpoint_uses_its_registered_runtime_without_rewriting_history(bank, budget, macro, legacy):
    _, manifest = bank
    old = Path(manifest["writer_checkpoint"]["path"])
    checkpoint = old.with_name(f"macro_{macro:08d}")
    old.rename(checkpoint)
    trainer_path = checkpoint / "trainer_state.pt"
    trainer = torch.load(trainer_path, weights_only=True)
    trainer["next_macro"] = trainer["training_state"]["updates"] = macro
    trainer["sampler_state"] = {"event_contract": {"maximum_updates": budget}}
    torch.save(trainer, trainer_path)
    path = checkpoint / "checkpoint_manifest.json"
    saved = json.loads(path.read_text())
    saved["next_macro"] = macro
    saved["files"]["trainer_state.pt"]["bytes"] = trainer_path.stat().st_size
    path.write_text(json.dumps(saved))
    parent_path = checkpoint.parent.parent / "run_contract.json"
    original = json.loads(parent_path.read_text()) | {"topology": {"world_size": 1}}
    parent_path.write_text(json.dumps(original))
    parent_bytes = parent_path.read_bytes()
    with pytest.raises(Pi05SourceTrainingError, match="invalid JSON authority"):
        inspect_writer_checkpoint(checkpoint)
    extension_path = (parent_path.with_name("run_contract_extended.json") if legacy else
                      extension_record_path(parent_path.parent, "run_contract.json", budget))
    extension_path.parent.mkdir(parents=True, exist_ok=True)
    extended = copy.deepcopy(original)
    extended["git"]["commit"] = "b" * 40
    extended["training"] = {"maximum_updates": budget}
    extension_path.write_text(json.dumps(extended))
    later_path = extension_record_path(parent_path.parent, "run_contract.json", 2400)
    later_path.parent.mkdir(parents=True, exist_ok=True)
    later = copy.deepcopy(extended)
    later["git"]["commit"] = "c" * 40
    later["training"]["maximum_updates"] = 2400
    later_path.write_text(json.dumps(later))
    observed, authority = inspect_writer_checkpoint(checkpoint)
    assert observed == extended and authority["training_commit"] == "b" * 40
    assert authority["run_contract"]["path"] == str(extension_path)
    assert parent_path.read_bytes() == parent_bytes
    extended["training"]["maximum_updates"] = 1200
    extension_path.write_text(json.dumps(extended))
    with pytest.raises(ValueError, match="exceeds the registered budget"):
        inspect_writer_checkpoint(checkpoint)
    extended["training"]["maximum_updates"] = budget
    extended["source"]["model_path"] = "/different/source"
    extension_path.write_text(json.dumps(extended))
    with pytest.raises(ValueError, match="exact-resume contract differs: source"):
        inspect_writer_checkpoint(checkpoint)


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


def test_explicit_train_diagnostic_request_and_count_compatibility(resident_materialization):
    requests, _, _, _ = resident_materialization
    request = requests[0] | {"init_state_ids": list(range(32, 36)), "state_count": 4}
    paths = materialization.materialize_requests(asset_root=ROOT, requests=[request], device=torch.device("cpu"))
    selected = json.loads(paths[0].read_text())["selection"]
    assert selected["init_state_ids"] == list(range(32, 36))
    assert materialization.request_init_state_ids(role="validation") == tuple(range(50))
    assert materialization.request_init_state_ids(role="validation", state_count=10) == tuple(range(10))


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


def test_single_cli_preserves_explicit_init_state_ids(monkeypatch):
    calls = []
    monkeypatch.setattr(materialization, "materialize", lambda **kwargs: calls.append(kwargs) or Path("/manifest.json"))
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr("sys.argv", ["materialize_writer.py", "--checkpoint", "/checkpoint",
        "--output", "/output", "--role", "development_train", "--task-ids", "0", "--k", "1",
        "--device", "cpu", "--state-count", "4", "--init-state-ids", "32,33,34,35"])
    materialization.main()
    assert calls[0]["selection"]["init_state_ids"] == list(range(32, 36))
    assert calls[0]["selection"]["seed"] == 20260907


def test_compile_uses_complete_dual_video_and_exact_language_once(tmp_path, monkeypatch):
    import numpy as np
    lora = replace(load_pi05_lora_contract(ROOT / "configs/pi05_lora_v1.json"),
                   targets=(LoRATarget("linear", 3, 4),), rank=2, alpha=2)
    calls = []
    def prepare(frames, indices, language):
        assert frames[0].shape == (2, 2, 3, 4, 4)
        assert indices[0].tolist() == [0, 5] and language == "exact task"
        calls.append("prepare")
        return object()
    def compile(condition):
        assert not torch.is_grad_enabled()
        calls.append("complete_writer")
        return identity_lora_state(lora)
    runtime = SimpleNamespace(prepare=prepare, compile=compile, lora=lora)
    task = SimpleNamespace(authority=SimpleNamespace(task_id=0, language="exact task"),
        episode_lengths=(6,), suite="libero_spatial", suite_task_id=0)
    video = SimpleNamespace(frames=np.zeros((2, 2, 3, 4, 4), dtype=np.uint8),
        frame_indices=np.array([0, 5]), raw_frame_count=6)
    record = materialization._compile_condition(runtime, SimpleNamespace(load=lambda *args: video),
        task, (0,), tmp_path, {"path": "/checkpoint", "macro": 16})
    assert calls == ["prepare", "complete_writer"] and record["writer_invocations"] == 1


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
