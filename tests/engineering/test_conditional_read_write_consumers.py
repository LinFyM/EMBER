"""§13 canonical full-only source, final-A banks and actual capture/A28 entrypoints.

Synthetic endpoint fixtures do not claim that a trained450 or GPU readout exists.
"""
from collections import OrderedDict
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, identity_lora_state
from ember.operator_writer import bank, joint_readout as readout, joint_training as study
from ember.operator_writer import materialization, run, specification as specs
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.preparation import _registered_trajectory_capture
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record


def consumer():
    spec = importlib.util.spec_from_file_location("conditional_A28_consumer", readout.REPO / "scripts/operator_joint_readouts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_full_only_identity_events_and_fresh_scope(tmp_path, monkeypatch):
    spec = specs.specification(specs.CONDITIONAL_SPEC_PATH)
    original = specs.specification(specs.SELF_READ_SPEC_PATH)
    assert spec["events"] == original["events"]
    assert spec["optimization"] == original["optimization"]
    assert spec["joint"]["loss_variant"] == "full"
    assert spec["joint"]["public_loss_coefficient"] == 0
    assert spec["execution"]["world_sizes"] == list(range(1, 7))
    assert spec["operator"] == study.CONDITIONAL_OPERATOR
    assert not {"context_value", "self_conditioned_native", "mode_T", "mode_U"} & spec["operator"].keys()
    monkeypatch.setattr(study, "CONDITIONAL_ROOT", tmp_path)
    args = SimpleNamespace(mode=study.CONDITIONAL_MODE, attempt="fresh", resume=None,
                           pilot_arm=None, microbatch=28, frame_chunk=8, stop_after_macro=None)
    study.validate_request(spec, args)
    contract = {"loss_variant": "full", "joint": spec["joint"]}
    study.validate_attempt(spec, args, contract, tmp_path / study.CONDITIONAL_MODE / "train/attempts/fresh")
    with pytest.raises(ValueError, match="loss identity"):
        study.validate_attempt(spec, args, {**contract, "loss_variant": study.LOSS},
                               tmp_path / study.CONDITIONAL_MODE / "train/attempts/fresh")
    metrics = [{"update": i, "mode": study.CONDITIONAL_MODE, "loss_variant": "full",
                "queries": 112, "jobs": [{"loss_variant": "full"}] * 4} for i in range(1, 451)]
    assert study.valid_metrics(metrics, mode=study.CONDITIONAL_MODE)
    metrics[0]["jobs"] = [{"loss_variant": "full", "public_query_reuse": True}] * 4
    assert not study.valid_metrics(metrics, mode=study.CONDITIONAL_MODE)


@pytest.mark.parametrize("mode", readout.CONDITIONAL_MODES)
def test_actual_prepare_routes_full400_and_full144(tmp_path, monkeypatch, mode):
    calibrated = mode in readout.control_calibration.MODES
    if calibrated:
        monkeypatch.setattr(readout.control_calibration, "ROOT", tmp_path)
    else:
        monkeypatch.setattr(readout, "CONDITIONAL_ROOT", tmp_path)
    runtime_mode = readout.control_calibration.MODE if calibrated else study.CONDITIONAL_MODE
    checkpoint = tmp_path / runtime_mode / "train/attempts/fresh/checkpoints/macro_00000450"
    capture_path = readout.capture_path(mode)
    capture = read_json(capture_path)
    seen = readout._seen_geometry(mode)
    states = readout.scope.STATES if seen else tuple(range(50))
    tasks = [SimpleNamespace(suite=r["suite"], task_id=r["task_id"], init_state_ids=states)
             for r in capture["full_conditions"]]
    path = readout.bank_path(mode, checkpoint)
    path.parent.mkdir(parents=True)
    write_json_atomic(path, {"kind": bank.KIND, "mode": mode, "joint_public_study": True,
                            "checkpoint": str(checkpoint)})
    args = SimpleNamespace(role=readout.scope.ROLE if seen else "validation", mode="formal",
                           static_task_lora_manifest=path, trajectory_capture_selection=capture_path)
    output = readout.evaluation_path(mode, checkpoint)
    prepared, stage = _registered_trajectory_capture(args, tasks, output, None, readout.REPO)
    assert capture["study_id"] == (readout.control_calibration.TASK if calibrated else study.CONDITIONAL_TASK)
    assert len(prepared["full_conditions"]) == (36 if seen else 8)
    assert all(r["init_state_id"] == (32 if seen else 0) for r in prepared["full_conditions"])
    assert prepared["passive_trace"] and not stage["full_conditions_only"]
    with pytest.raises(Pi05EvaluationError, match="scope changed"):
        _registered_trajectory_capture(args, tasks, output.with_name("public144"), None, readout.REPO)
    with pytest.raises(ValueError, match="owned complete450"):
        readout.bank_path(mode, checkpoint.with_name("macro_00000900"))
    with pytest.raises(ValueError, match="owned complete450"):
        readout.bank_path(mode, readout.OLD_CHECKPOINT)


def test_full_bank_saves_and_deploys_conditional_A_not_public_A(tmp_path, monkeypatch):
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(readout.REPO / "configs/pi05_lora_v1.json"), rank=128)
    common = identity_lora_state(lora)
    final = {key: value + .25 for key, value in common.items()}
    shapes = {key: tuple(value.shape) for key, value in final.items()}
    condition = {"condition_id": "scope_fixture", "global_task_id": 3, "teacher_demo": 1}
    calls = []
    def compile(*args, **kwargs):
        calls.append(1)
        assert not torch.is_grad_enabled()
        return final, None
    runtime = SimpleNamespace(device=torch.device("cpu"), compile=compile)
    data = SimpleNamespace(condition=lambda *a: ((), 11, 3),
                           videos=SimpleNamespace(frame_counts=lambda *a: (11, 3)))
    row, stats = materialization.write_condition(runtime, data, tmp_path, condition, shapes,
                                                 mode=study.CONDITIONAL_MODE, frame_chunk=8)
    saved = load_file(row["factors"]["path"])
    assert len(saved) == 76 and not stats["reused"]
    adapter = SimpleNamespace(bank={"mode": study.CONDITIONAL_MODE,
                                   "condition_factors": "complete_A0_plus_S_B0_plus_M"},
                              common=common, conditions={condition["condition_id"]: row},
                              states=OrderedDict(), lora=lora)
    deployed = bank.FrozenOperatorAdapter._state(adapter, condition["condition_id"])
    key = next(key for key in common if key.endswith(LORA_A_SUFFIX))
    torch.testing.assert_close(deployed[key], final[key])
    assert not torch.equal(deployed[key], common[key])
    _, cached = materialization.write_condition(runtime, data, tmp_path, condition, shapes,
                                                mode=study.CONDITIONAL_MODE, frame_chunk=8)
    assert cached["reused"] and len(calls) == 1


def test_training_and_reading_identity_remain_separate_full450_fixture(tmp_path, monkeypatch):
    import subprocess
    spec = specs.specification(specs.CONDITIONAL_SPEC_PATH)
    monkeypatch.setattr(study, "CONDITIONAL_ROOT", tmp_path)
    output = tmp_path / study.CONDITIONAL_MODE / "train/attempts/fresh"
    checkpoint = output / "checkpoints/macro_00000450"
    checkpoint.mkdir(parents=True)
    training_spec = tmp_path / "old_frozen/configs/operator_read_write_v1/conditional_read_write_fresh_spec.json"
    training_spec.parent.mkdir(parents=True)
    write_json_atomic(training_spec, spec)
    sampler = {"schema_version": spec["events"]["schema_version"], "seed": 20260928,
               "tasks": spec["events"]["task_ids"], "demo_pool": [0, 49], "query_offset": 1,
               "queries_per_task": 28, "teacher_rounds": [[20260928, 1, "task"], [20260928, 1, "task", 1]],
               "teacher_visits_per_round": 50, "teacher_demo_pool": list(range(50))}
    git = {"commit": "training-fixture", "branch": "", "dirty_paths": [], "pushed_ref": "origin/main"}
    contract = {"schema_version": run.SCHEMA, "stage": run.STAGE, "mode": study.CONDITIONAL_MODE,
                "joint": spec["joint"], "loss_variant": "full", "operator": spec["operator"],
                "optimizer": spec["optimization"], "events": spec["events"], "source_trainable": 0,
                "sampler": sampler, "spec": str(training_spec), "git": git}
    write_json_atomic(output / "run_contract.json", contract)
    trainer = {"stage": run.STAGE, "next_macro": 450, "metrics_rows": 450,
               "scheduler": {"last_epoch": 450}, "optimizer": {"param_groups": [{}]}, "scaler": None,
               "training_state": {"updates": 450, "mode": study.CONDITIONAL_MODE, "loss_variant": "full"},
               "sampler_state": {**sampler, "next_step": 450}}
    torch.save(trainer, checkpoint / "trainer_state.pt")
    save_file({"fixture": torch.ones(1)}, str(checkpoint / "ecp.safetensors"))
    torch.save({"fixture": True}, checkpoint / "rank_00_state.pt")
    write_json_atomic(checkpoint / "checkpoint_manifest.json", {
        "stage": run.STAGE, "run_contract_schema": run.SCHEMA, "next_macro": 450, "world_size": 1,
        "files": {name: {"bytes": (checkpoint / name).stat().st_size}
                  for name in ("ecp.safetensors", "trainer_state.pt", "rank_00_state.pt")}})
    rows = [{"update": i, "mode": study.CONDITIONAL_MODE, "loss_variant": "full", "queries": 112,
             "jobs": [{"loss_variant": "full"}] * 4} for i in range(1, 451)]
    (output / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json_atomic(output / "completion.json", {"updates": 450, "checkpoint": str(checkpoint)})
    monkeypatch.setattr(run, "frozen_git", lambda: {"commit": "new-reader-fixture"})
    monkeypatch.setattr(study, "git_state", lambda p: git)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="origin/main\n"))
    result = study.inspect_source(spec, checkpoint)
    assert result["git"]["commit"] == "training-fixture"
    assert result["spec"] == str(training_spec)  # actual absolute string, not a file_record
    contract["loss_variant"] = study.LOSS
    write_json_atomic(output / "run_contract.json", contract)
    with pytest.raises(ValueError, match="source loss/ECP"):
        study.inspect_source(spec, checkpoint)


@pytest.mark.parametrize('macro,arm', [(450, None), (900, None), (630, 'C12'), (630, 'D71')])
def test_A28_twelve_predictions_eight_passive_compiles(tmp_path, monkeypatch, macro, arm):
    module = consumer()
    monkeypatch.setattr(readout, "CONDITIONAL_ROOT", tmp_path)
    if macro == 900:
        monkeypatch.setattr(study, 'CONDITIONAL_CONTINUATION_ROOT', tmp_path)
    checkpoint = tmp_path / study.CONDITIONAL_MODE / f"train/attempts/fixture/checkpoints/macro_{macro:08d}"
    if arm is not None:
        from ember.operator_writer import support_diversity
        monkeypatch.setattr(support_diversity, 'ROOT', tmp_path)
        if arm == 'C12':
            monkeypatch.setattr(support_diversity, 'C12_CHECKPOINT', checkpoint)
    reading_spec = (specs.SUPPORT_DIVERSITY_SPEC_PATH if arm is not None else
                    specs.CONDITIONAL_CONTINUATION_SPEC_PATH if macro == 900 else specs.CONDITIONAL_SPEC_PATH)
    reader = {"commit": "fixture-reader", "branch": "", "dirty_paths": [], "pushed_ref": "origin/main"}
    monkeypatch.setattr(readout, "source_record", lambda *a, **k: ({}, {"git": {"commit": "fixture-trainer"},
                                                              "spec": "/data1/fixture-training-spec"}, reading_spec))
    monkeypatch.setattr(module, "frozen_git", lambda: reader)
    monkeypatch.setattr(module, "load_file", lambda *a, **k: {})
    from ember.writer import materialization_workers
    monkeypatch.setattr(materialization_workers, "_configure_device", lambda *a: None)
    common = {name + suffix: torch.ones((128, 4) if suffix == LORA_A_SUFFIX else (3, 128))
              for name in module.SELF_READ_SITES.values() for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX)}
    calls = []
    def compile(condition, *, frame_chunk, capture_mechanism):
        assert capture_mechanism
        calls.append(condition)
        h = torch.full((3, 50, 1024), 2.)
        x = {name: torch.ones(3, 50, 4) for name in module.SELF_READ_SITES.values()}
        targets = {name: {"A0": common[name + LORA_A_SUFFIX], "S": torch.full((128, 4), .5),
                          "B0": common[name + LORA_B_SUFFIX], "M": torch.full((3, 128), .25)}
                   for name in x}
        state = {name + suffix: item["A0"] + item["S"] if suffix == LORA_A_SUFFIX else item["B0"] + item["M"]
                 for name, item in targets.items() for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX)}
        return state, {"h": h, "x": x, "passes": [{"x": x, "h": h, "state": state}],
                       "mechanism": {"c": h + 1, "d": h + 2, "targets": targets}}
    writer = SimpleNamespace(public_state=lambda: common, load_state_dict=lambda *a, **k: None)
    writer.requires_grad_ = lambda *a: writer
    writer.eval = lambda: writer
    runtime = SimpleNamespace(writer=writer, policy=SimpleNamespace(eval=lambda: None), compile=compile,
                              processor=SimpleNamespace(training_batch=lambda x: x))
    def build(*a, **k):
        assert a[3] == study.CONDITIONAL_MODE
        return runtime
    monkeypatch.setattr(module, "build_runtime", build)
    class Data:
        def __init__(self, *a, **k): pass
        def batch(self, event): return {}
        def condition(self, runtime, task, teacher): return ((task, [0, 5, 11], teacher), None, None)
        def close(self): pass
    monkeypatch.setattr(module, "FormalData", Data)
    monkeypatch.setattr(module, "fm_prediction", lambda runtime, state, batch, flow, micro: flow["FM_target"][..., :7])
    module.a28(SimpleNamespace(model=study.CONDITIONAL_MODE, checkpoint=checkpoint, asset_root=readout.REPO,
                              device="cpu", cpu_threads=1, microbatch=28, native_frame_chunk=8, arm=arm))
    output = (tmp_path / arm if arm is not None else tmp_path) / "analysis/A28" / study.CONDITIONAL_MODE
    complete = read_json(output / "completion.json")
    assert (complete["rows"], complete["full"], complete["public"], complete["native_records"]) == (12, 8, 4, 8)
    assert "intermediate" not in complete and len(calls) == 8
    records = json.loads((output / "native_records.json").read_text())
    saved = torch.load(records[0]["raw"]["path"], map_location="cpu", weights_only=True)
    assert set(saved["targets"]) == {"Q8", "V8", "action_out"}
    assert saved["native_passes"] == 1 and saved["frame_indices"].tolist() == [0, 5, 11]
    assert all(saved[key].shape == (3, 50, 1024) for key in ("H", "c", "d"))
    assert all({"X", "A0", "S", "B0", "M"} <= fields.keys() for fields in saved["targets"].values())
    assert saved["training_spec"] == "/data1/fixture-training-spec"
    assert saved['reading_spec']['path'] == str(reading_spec)
    if arm is not None:
        assert complete['support_diversity_arm'] == saved['support_diversity_arm'] == arm
        assert all(row['support_diversity_arm'] == arm for row in json.loads((output / 'rows.json').read_text()))


def test_actual_original_scene_teacher_geometry_and_seen_runtime_route(monkeypatch):
    from ember.pi05_eval.scene import inspect_registered_scenes
    spec = specs.specification(specs.CONDITIONAL_SPEC_PATH)
    asset = Path('/data1/user/ymdai/projects/EMBER')
    rows, conditions, scenes, states = readout._geometry(readout.CONDITIONAL_SEEN_MODE, spec, asset)
    assert tuple(row['global_task_id'] for row in rows) == tuple(spec['events']['task_ids'])
    assert len(conditions) == 144 and states == (32, 33, 34, 35)
    assert all(len({ep['teacher_demo_indices'][0] for ep in row['episodes']}) == 4 for row in rows)
    inspect_registered_scenes(scenes, rows, states=states, schema='ember_operator_seen_task_scenes_v1')
    validation, videos, scenes400, states400 = readout._geometry(study.CONDITIONAL_MODE, spec, asset)
    assert len(videos) == 400 and states400 == tuple(range(50))
    assert all(len({ep['teacher_demo_indices'][0] for ep in row['episodes']}) == 50 for row in validation)
    inspect_registered_scenes(scenes400, validation, states=states400)
    calls = []
    monkeypatch.setattr(materialization, '_configure_device', lambda *a: None)
    monkeypatch.setattr(run, 'build_runtime', lambda *a: calls.append(a) or SimpleNamespace())
    materialization.OperatorCompiler(asset, {'spec': spec, 'mode': readout.CONDITIONAL_SEEN_MODE},
                                     torch.device('cpu'), 1)
    assert calls[0][3] == study.CONDITIONAL_MODE
