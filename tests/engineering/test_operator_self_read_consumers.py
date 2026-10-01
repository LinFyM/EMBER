"""§40 consumer scope and passive A28 artifacts; no future450 acceptance claim."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, LoRATarget, identity_lora_state
from ember.operator_writer import bank, joint_readout as readout, joint_training, run
from ember.operator_writer.model import OperatorReadWrite
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.preparation import _registered_trajectory_capture
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from safetensors import safe_open
from safetensors.torch import save_file


def consumer():
    spec = importlib.util.spec_from_file_location("self_read_A28_consumer", readout.REPO / "scripts/operator_joint_readouts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", readout.SELF_READ_MODES)
def test_self_read_source_rejects_actual_single_pass450(mode):
    checkpoint = (joint_training.CONTEXT_ROOT / "context/train/attempts/fresh/checkpoints/macro_00000450")
    assert checkpoint.is_dir()
    with pytest.raises(ValueError):
        readout.source_record(mode, checkpoint)
    with pytest.raises(ValueError, match="registered main/conditional endpoint"):
        readout.bank_path(mode, checkpoint.with_name("macro_00000900"))


@pytest.mark.parametrize("mode", readout.SELF_READ_MODES)
def test_actual_prepare_dispatch400_144_scope_fixture(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(joint_training, "SELF_READ_ROOT", tmp_path)
    public = mode in readout.PUBLIC_MODES
    capture_path = readout.capture_path(mode)
    registered = read_json(capture_path)
    states = readout.scope.STATES if public else tuple(range(50))
    tasks = [SimpleNamespace(suite=row["suite"], task_id=row["task_id"], init_state_ids=states)
             for row in registered["full_conditions"]]
    checkpoint = tmp_path / "self_read/train/attempts/fresh/checkpoints/macro_00000450"
    path = readout.bank_path(mode, checkpoint)
    path.parent.mkdir(parents=True)
    manifest = {"kind": bank.KIND, "mode": mode, "joint_public_study": True, "checkpoint": str(checkpoint),
                "native_reading": readout._self_read_evidence(mode)}
    write_json_atomic(path, manifest)
    args = SimpleNamespace(role=readout.scope.ROLE if public else "validation", mode="formal",
        state_count=4 if public else 50, init_state_ids=states if public else None,
        static_task_lora_manifest=path, trajectory_capture_selection=capture_path)
    output = readout.evaluation_path(mode, checkpoint)
    capture, stage = _registered_trajectory_capture(args, tasks, output, None, readout.REPO)
    assert registered["study_id"] == joint_training.SELF_READ_TASK
    assert len(capture["full_conditions"]) == (36 if public else 8)
    assert all(row["init_state_id"] == (32 if public else 0) for row in capture["full_conditions"])
    assert capture["passive_trace"] and not stage["full_conditions_only"]
    assert readout._self_read_evidence(mode)["output"] == ("beta" if public else "beta+M1")
    with pytest.raises(Pi05EvaluationError, match="scope changed"):
        _registered_trajectory_capture(args, tasks, output.with_name("intermediate400"), None, readout.REPO)
    if public:
        common = object()
        adapter = SimpleNamespace(bank={"mode": mode}, common=common)
        assert bank.FrozenOperatorAdapter._state(adapter, "paired_metadata") is common
        task = {"global_task_id": 0}
        episode = {"init_state_id": 32, "condition_id": "task_00_demos_37", "teacher_demo_indices": [37], "video_ordinal": 32}
        evidence_bank = {**manifest, "shared": {}, "scene_manifest": {}}
        assert bank.episode_evidence(evidence_bank, task, episode)["teacher_video_values_read"] == 0


def test_self_read_public_complete76_export_without_native_scope_fixture(tmp_path, monkeypatch):
    """Fresh public-only ECP fixture; deliberately not a valid trained450 source."""
    spec = run.specification(run.SELF_READ_SPEC_PATH)
    asset = Path("/data1/user/ymdai/projects/EMBER")
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(asset / spec["source"]["lora_contract"]), rank=128)
    state = identity_lora_state(lora)
    checkpoint = tmp_path / "self_read/train/attempts/fresh/checkpoints/macro_00000450"
    checkpoint.mkdir(parents=True)
    save_file({row["ecp_name"]: state[row["lora_name"]] for row in readout.factor_map(lora)},
              str(checkpoint / "ecp.safetensors"))
    write_json_atomic(checkpoint / "checkpoint_manifest.json", {"fixture": True})
    write_json_atomic(checkpoint.parent.parent / "run_contract.json", {"fixture": True})
    prior = read_json(readout.PUBLIC_FACTOR_MANIFEST)
    training = {"git": {"commit": "fixture"}, "source": prior["source"], "lora": lora.to_dict()}
    monkeypatch.setattr(joint_training, "SELF_READ_ROOT", tmp_path)
    monkeypatch.setattr(readout, "source_record", lambda *a: (spec, training, run.SELF_READ_SPEC_PATH))
    monkeypatch.setattr(run, "frozen_git", lambda: {"commit": "fixture-reader", "branch": "",
                        "dirty_paths": [], "pushed_ref": "origin/main"})
    from ember.operator_writer import materialization
    monkeypatch.setattr(materialization, "compile_conditions", lambda *a, **k: pytest.fail("public must not compile native"))
    path = readout.materialize("self_read_public", checkpoint, asset, device="cpu")
    manifest = read_json(path)
    assert len(manifest["conditions"]) == 144
    assert manifest["information_wall"]["teacher_video_values_read"] == 0
    with safe_open(manifest["shared"]["path"], framework="pt", device="cpu") as handle:
        assert len(handle.keys()) == 76 and handle.metadata()["mode"] == "self_read_public"
    keys = tuple((task["suite"], task["task_id"]) for task in manifest["tasks"])
    inspected = readout.inspect(manifest, path, training["source"], keys, readout.scope.ROLE, True,
                               {key: readout.scope.STATES for key in keys})
    assert inspected["native_reading"] == readout._self_read_evidence("self_read_public")
    invalid = {**manifest, "native_reading": {**manifest["native_reading"], "training_native_passes": 1}}
    with pytest.raises(ValueError, match="scope changed"):
        readout.inspect(invalid, path, training["source"], keys, readout.scope.ROLE, True,
                        {key: readout.scope.STATES for key in keys})


def test_passive_evidence_from_one_actual_core_compile(tmp_path, monkeypatch):
    module = consumer()
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(readout.REPO / "configs/pi05_lora_v1.json"), rank=128)
    lora = replace(lora, targets=tuple(LoRATarget(target.name, 4, 3) for target in lora.targets))
    writer = OperatorReadWrite(lora, identity_lora_state(lora), "self_read")
    with torch.no_grad():
        for name in module.SELF_READ_SITES.values():
            writer.writes[writer.names.index(name)].o.weight.normal_(std=.04)
            writer.public_state()[name + LORA_B_SUFFIX].fill_(.25)
    reads = []
    hidden = torch.randn(3, 50, 1024, generator=torch.Generator().manual_seed(13))
    def native(policy, state, probe, condition, names, **kwargs):
        reads.append(state)
        offset = state[writer.names[0] + LORA_B_SUFFIX].square().sum().tanh()
        return {name: torch.full((3, 50, 4), 2.) + offset for name in names}, hidden + len(reads)
    monkeypatch.setattr(run, "read_native_video", native)
    monkeypatch.setattr(run.Runtime, "restore_identity", lambda self: None)
    runtime = run.Runtime(None, writer, None, None, lora, {}, torch.device("cpu"), {})
    condition = (None, [0, 5, 11], None, None)
    with torch.no_grad():
        state, recorded = runtime.compile(condition, retain_native=True)
    common = writer.public_state()
    ref = module._save_native_evidence(tmp_path, 0, 40, condition[1], recorded, common,
                                       {"checkpoint": "fixture-only", "training_git": "fixture", "reading_git": {}})
    saved = torch.load(ref["path"], map_location="cpu", weights_only=True)
    assert len(reads) == 2 and state is recorded["passes"][1]["state"]
    assert saved["schema_version"] == "ember_self_read_native_evidence_v1"
    assert saved["frame_indices"].tolist() == condition[1]
    assert not torch.equal(saved["H0"], saved["H1"])
    for label, name in module.SELF_READ_SITES.items():
        for i in range(2):
            expected = torch.nn.functional.linear(recorded["passes"][i]["x"][name], common[name + LORA_A_SUFFIX])
            torch.testing.assert_close(saved["targets"][label][f"AX{i}"], expected)
            torch.testing.assert_close(saved["targets"][label][f"M{i}"],
                recorded["passes"][i]["state"][name + LORA_B_SUFFIX] - common[name + LORA_B_SUFFIX])
    assert len(reads) == 2  # Saving evidence never invokes another native pass.


def test_A28_twenty_records_one_compile_per_teacher_with_scope_fixture(tmp_path, monkeypatch):
    module = consumer()
    checkpoint = tmp_path / "self_read/train/attempts/fresh/checkpoints/macro_00000450"
    reading = {"commit": "fixture", "branch": "", "dirty_paths": [], "pushed_ref": "origin/main"}
    monkeypatch.setattr(readout, "source_record", lambda *a: ({}, {"git": reading, "spec": {}}, run.SELF_READ_SPEC_PATH))
    monkeypatch.setattr(joint_training, "SELF_READ_ROOT", tmp_path)
    monkeypatch.setattr(module, "frozen_git", lambda: reading)
    monkeypatch.setattr(module, "load_file", lambda *a, **k: {})
    from ember.writer import materialization_workers
    monkeypatch.setattr(materialization_workers, "_configure_device", lambda *a: None)
    compiled = []
    common = {name + suffix: torch.zeros((128, 4) if suffix == LORA_A_SUFFIX else (3, 128))
              for name in module.SELF_READ_SITES.values() for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX)}
    def compile(condition, *, frame_chunk, retain_native):
        assert retain_native
        compiled.append(condition)
        passes = [{"x": {name: torch.full((3, 50, 4), float(i + 1)) for name in module.SELF_READ_SITES.values()},
                   "h": torch.full((3, 50, 1024), float(i)),
                   "state": {key: value + i + 1 if key.endswith(LORA_B_SUFFIX) else value for key, value in common.items()}}
                  for i in range(2)]
        return passes[1]["state"], {"passes": passes, "x": passes[1]["x"], "h": passes[1]["h"]}
    writer = SimpleNamespace(public_state=lambda: common, load_state_dict=lambda *a, **k: None)
    writer.requires_grad_ = lambda *a: writer
    writer.eval = lambda: writer
    runtime = SimpleNamespace(writer=writer, policy=SimpleNamespace(eval=lambda: None), compile=compile,
                              processor=SimpleNamespace(training_batch=lambda x: x))
    def build(*a, **k):
        assert a[3] == "self_read"
        return runtime
    monkeypatch.setattr(module, "build_runtime", build)
    class Data:
        def __init__(self, *a, **k): pass
        def batch(self, event): return {}
        def condition(self, runtime, task, teacher): return ((task, [0, 5, 11], teacher), None, None)
        def close(self): pass
    monkeypatch.setattr(module, "FormalData", Data)
    monkeypatch.setattr(module, "fm_prediction", lambda runtime, state, batch, flow, micro: flow["FM_target"][..., :7])
    module.a28(SimpleNamespace(model="self_read", checkpoint=checkpoint, asset_root=readout.REPO,
        device="cpu", cpu_threads=6, microbatch=28, native_frame_chunk=8))
    output = tmp_path / "analysis/A28/self_read"
    complete = read_json(output / "completion.json")
    assert (complete["rows"], complete["full"], complete["intermediate"], complete["public"], complete["native_records"]) == (20, 8, 8, 4, 8)
    assert len(compiled) == 8
    rows = json.loads((output / "rows.json").read_text())
    assert all("native_ref" in row for row in rows if row["kind"] != "public")
    assert len({row["raw"]["path"] for row in rows}) == 20
