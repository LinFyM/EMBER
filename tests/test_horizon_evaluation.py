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
from ember.writer.materialization import (BANK_KIND, BANK_SCHEMA, RUN_SCHEMA, STAGE, TRAINING_SCHEMA, UPDATE_VERSION, adapter_metadata,
    condition_id, file_record, inspect_writer_checkpoint, method_metadata, paired_video_sets,
    planned_episodes, selection_contract)


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
           "source": SOURCE, "config": {"update_version": UPDATE_VERSION, "data": {"version": "fixture_supervised_data_v1"}, "observer": {"probe_seed": 1729}, "execution_precision": "native_mixed_without_outer_autocast"}, "model_config": {"horizon": 50}}
    run["model_config"]["compiler_language_mode"] = "first_query_only_v1"
    run["model_config"]["process_language_source"] = "frame_contextual_task_tokens_v1"
    run["config"]["model"] = dict(run["model_config"])
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
                           authority=SimpleNamespace(path=source, language="exact task language"))
    runs, requests, builds = {}, [], []
    for step, value, arm in ((16, 1., "correct"), (48, 2., "same_task_other")):
        checkpoint = tmp_path / f"run/checkpoints/macro_{step:08d}"
        checkpoint.mkdir(parents=True)
        tensors = {name: tensor.detach().clone() for name, tensor in state.state_dict().items()}
        tensors["writer.weight"].fill_(value)
        tensors["meta.weight"].fill_(value * 10)
        save_file(tensors, str(checkpoint / "ecp.safetensors"))
        runs[checkpoint] = {"source": copy.deepcopy(SOURCE), "model_config": {"width": 12},
            "config": {"update_version": UPDATE_VERSION, "execution_precision": "native_mixed_without_outer_autocast", "model": {"width": 999}, "observer": {"probe_seed": 1729, "meta_rank": 4, "frame_chunk": 4}}}
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
        return instance

    def compile_condition(current, _store, _task, demos, _output, _checkpoint):
        assert current is instance and current.observer.probe is current.state.probe
        return {"condition_id": condition_id(0, demos), "teacher_videos": [{"sampled_frame_count": 1}],
                "writer_value": float(state.writer.weight), "meta_value": float(state.meta.weight)}

    monkeypatch.setattr(runtime, "build_runtime", build)
    monkeypatch.setattr(materialization, "_compile_condition", compile_condition)
    return requests, runs, builds, state


def test_resident_batch_loads_once_and_reloads_entire_checkpoint_per_manifest(resident_materialization, tmp_path):
    requests, _, builds, state = resident_materialization
    paths = materialization.materialize_requests(asset_root=ROOT, requests=requests, device=torch.device("cpu"))
    assert len(builds) == 1 and builds[0][1]["model"] == {"width": 12}
    assert state.loads == 2
    for index, path in enumerate(paths):
        manifest = json.loads(path.read_text())
        assert manifest["schema_version"] == BANK_SCHEMA
        assert manifest["writer_checkpoint"]["path"] == requests[index]["checkpoint"]
        assert manifest["arm"] == requests[index]["arm"]
        assert manifest["conditions"][0]["writer_value"] == index + 1
        assert manifest["conditions"][0]["meta_value"] == (index + 1) * 10
        assert manifest["information_wall"]["total_writer_invocations"] == 1
        assert len(manifest["tasks"][0]["episodes"]) == 10
    materialization.materialize(asset_root=ROOT, checkpoint=Path(requests[0]["checkpoint"]),
        output=tmp_path / "single", selection=_selection(mode="fixed_per_task"), device=torch.device("cpu"))
    assert len(builds) == 2 and state.loads == 3 and float(state.meta.weight) == 10


@pytest.mark.parametrize("field", ["source", "model_config", "observer"])
def test_resident_batch_rejects_cross_contract_reuse_before_loading(resident_materialization, field):
    requests, runs, builds, _ = resident_materialization
    changed = runs[Path(requests[1]["checkpoint"])]
    if field == "observer":
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
    instance = SimpleNamespace(state=state, policy=torch.nn.Linear(1, 1), lora=lora, source=SOURCE)
    row = old["tasks"][0]
    target = json.loads((ROOT / "configs/pi05_target_data_v1/manifest.json").read_text())
    native = next(task for task in target["tasks"] if task["global_task_id"] == 0)
    task = SimpleNamespace(suite=row["suite"], suite_task_id=row["task_id"],
        authority=SimpleNamespace(task_id=0, language=row["language"], path=Path(row["teacher_source"]["path"])))
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
    argv = ["materialize_horizon_writer.py", "--requests-json", str(path), "--asset-root", str(ROOT), "--device", "cpu"]
    monkeypatch.setattr("sys.argv", argv)
    materialization.main()
    assert calls == [{"asset_root": ROOT, "requests": requests, "device": torch.device("cpu")}]
    assert "/output/manifest.json" in capsys.readouterr().out
    for option in (("--arm", "same_task_other"), ("--seed", "7")):
        monkeypatch.setattr("sys.argv", [*argv, *option])
        with pytest.raises(SystemExit) as error:
            materialization.main()
        assert error.value.code == 2 and len(calls) == 1


def test_method_metadata_describes_final_native_and_visual_tokens():
    method = method_metadata({"model_config": {}, "config": {"update_version": UPDATE_VERSION, "observer": {}, "execution_precision": "native_mixed_without_outer_autocast"}})
    assert method["native_response_shape"] == [50, 1024]
    assert method["native_response_source"] == "action_out_proj_input_after_final_normalization"
    assert method["visual_token_source"] == "actual_final_prefix_image_tokens"
    assert method["frame_attention"] == "four_past_plus_self_causal"
    assert method["macro_cursor"] == "optimizer_updates"
    assert method["training_stage"] == STAGE
    assert method["training_objective"] == "supervised_fm"
    assert method["update_version"] == UPDATE_VERSION


@pytest.mark.parametrize("field,value", [("schema_version", "ember_horizon_relation_writer_joint_run_v1"),
    ("stage", "horizon_relation_writer_fresh_fm_rl_joint"), ("mode", "profile"),
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


@pytest.mark.parametrize("field", ["compiler_language_mode", "process_language_source"])
def test_shape_compatible_old_writer_requires_its_frozen_runtime(bank, field):
    _, manifest = bank
    checkpoint = Path(manifest["writer_checkpoint"]["path"])
    run_path = checkpoint.parent.parent / "run_contract.json"
    run = json.loads(run_path.read_text())
    del run["model_config"][field]
    del run["config"]["model"][field]
    run_path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="architecture identity"):
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
    monkeypatch.setattr("sys.argv", ["materialize_horizon_writer.py", "--checkpoint", "/checkpoint",
        "--output", "/output", "--role", "development_train", "--task-ids", "0", "--k", "1",
        "--device", "cpu", "--state-count", "4", "--init-state-ids", "32,33,34,35"])
    materialization.main()
    assert calls[0]["selection"]["init_state_ids"] == list(range(32, 36))
    assert calls[0]["selection"]["seed"] == 20260907


def test_compile_uses_observer_arguments_including_actual_visual_tokens(tmp_path, monkeypatch):
    import numpy as np

    lora = replace(load_pi05_lora_contract(ROOT / "configs/pi05_lora_v1.json"),
                   targets=(LoRATarget("linear", 3, 4),), rank=2, alpha=2)
    arguments = tuple(object() for _ in range(6))
    response = object()
    calls = []

    def writer(*values):
        calls.append(values)
        return identity_lora_state(lora)

    observer = SimpleNamespace(device=torch.device("cpu"), prepare=lambda *args: "condition",
        responses=lambda condition: response, writer_arguments=lambda condition: arguments)
    runtime = SimpleNamespace(observer=observer, state=SimpleNamespace(writer=writer), lora=lora)
    task = SimpleNamespace(authority=SimpleNamespace(task_id=0, language="exact task"),
        episode_lengths=(6,), suite="libero_spatial", suite_task_id=0)
    video = SimpleNamespace(frames=np.zeros((2, 3, 4, 4), dtype=np.uint8),
        frame_indices=np.array([0, 5]), raw_frame_count=6)
    record = materialization._compile_condition(runtime, SimpleNamespace(load=lambda *args: video),
        task, (0,), tmp_path, {"path": "/checkpoint", "macro": 16})
    assert calls == [(response, *arguments)]
    assert record["writer_invocations"] == 1


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
