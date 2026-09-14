"""Real spawned CPU workers verify bank ownership, dynamic dispatch and failures."""

import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch

from ember.writer import materialization, materialization_workers as workers
from ember.writer.materialization import condition_id, file_record
from test_horizon_evaluation import ROOT, SOURCE, resident_materialization


class _State(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.writer, self.meta, self.vl_meta = [torch.nn.Linear(1, 1, bias=False) for _ in range(3)]
        self.register_buffer("probe", torch.randn(50, 32, generator=torch.Generator().manual_seed(1729)))
        self.loads = 0

    def load_state_dict(self, state, strict=True):
        self.loads += 1
        return super().load_state_dict(state, strict=strict)


def _cpu_initialize_worker(queue, ready, asset_root, config, cpu_threads):
    """Keep the actual worker/runtime lifecycle; replace only GPU/model/compile cost."""
    from ember.writer import runtime
    from ember.pi05_lora import load_pi05_lora_contract

    native_load = workers.load_file
    workers.load_file = lambda path, **_: native_load(path, device="cpu")
    workers._configure_device = lambda *_: []
    materialization.RawTeacherVideoStore = lambda *_args, **_kwargs: SimpleNamespace(close=lambda: None)

    def build(_root, current, _device):
        root = Path(current["fixture"]["root"])
        (root / f"build_{os.getpid()}").touch()
        return SimpleNamespace(state=_State(), policy=torch.nn.Linear(1, 1).requires_grad_(False),
            source=SOURCE, lora=load_pi05_lora_contract(ROOT / "configs/pi05_lora_v1.json"),
            fixture=current["fixture"], counts={})

    runtime.build_runtime = build
    materialization._compile_condition = _cpu_compile_condition
    workers._initialize_worker(queue, ready, asset_root, config, cpu_threads)


def _cpu_compile_condition(runtime, _store, task, demos, output, _checkpoint):
    mode = runtime.fixture["mode"]
    if demos == [0] and mode == "exception":
        raise RuntimeError("injected condition failure")
    if demos == [0] and mode == "exit":
        os._exit(17)
    assert not (output / "manifest.json").exists()
    assert all(not value.requires_grad for value in runtime.state.parameters())
    release = Path(runtime.fixture["root"]) / f"release_{output.name}"
    started = time.monotonic_ns()
    if demos == [0]:
        # One worker stays busy until its peer finishes two separate conditions:
        # a static task/GPU split or a serial executor cannot satisfy this oracle.
        deadline = time.monotonic() + 15
        while not release.exists():
            if time.monotonic() > deadline:
                raise RuntimeError("condition dispatch did not make concurrent progress")
            time.sleep(.01)
    else:
        runtime.counts[output] = runtime.counts.get(output, 0) + 1
        if runtime.counts[output] == 2:
            release.touch()
    identifier = condition_id(task.authority.task_id, demos)
    path = output / f"{identifier}.safetensors"
    # Exclusive writes make accidental duplicate compilation an actual failure.
    with path.open("xb") as handle:
        handle.write(b"CPU condition scheduling oracle")
    return {"condition_id": "unassigned_condition" if mode == "wrong_id" and demos == [0] else identifier,
        "global_task_id": task.authority.task_id, "teacher_demo_indices": list(demos),
        "adapter": file_record(path), "teacher_videos": [{"sampled_frame_count": 1}],
        "pid": os.getpid(), "started": started, "finished": time.monotonic_ns(),
        "loads": runtime.state.loads, "writer_value": float(runtime.state.writer.weight),
        "meta_value": float(runtime.state.meta.weight), "vl_meta_value": float(runtime.state.vl_meta.weight)}


@pytest.fixture
def concurrent_requests(resident_materialization, tmp_path, monkeypatch):
    from ember.writer.learning_data import load_learning_tasks

    requests, runs, _, _ = resident_materialization
    load_learning_tasks(ROOT, [0])[0].episode_lengths = (1000,) + (100,) * 49
    for run in runs.values():
        run["config"]["fixture"] = {"root": str(tmp_path), "mode": "success"}
    selected = {"selection_mode": "per_init_ordinal", "state_count": 4, "init_state_ids": [32, 33, 34, 35]}
    batch = [requests[0] | selected,
             requests[0] | selected | {"output": str(tmp_path / "same_checkpoint"), "arm": "same_task_other"},
             requests[1] | selected | {"arm": "correct"}]
    monkeypatch.setattr(workers, "_initialize_worker", _cpu_initialize_worker)
    return batch, runs


def test_spawn_workers_cover_bank_once_and_reuse_then_reload_checkpoint(concurrent_requests, tmp_path, monkeypatch):
    requests, _ = concurrent_requests
    children = {child.pid for child in mp.active_children()}
    parent_pid = os.getpid()
    original_write = materialization.write_json_atomic

    def parent_seal(path, value):
        assert os.getpid() == parent_pid and value["status"] == "sealed"
        assert len(list(path.parent.glob("*.safetensors"))) == 4
        original_write(path, value)

    monkeypatch.setattr(materialization, "write_json_atomic", parent_seal)
    paths = materialization.materialize_requests(asset_root=ROOT, requests=requests,
        devices=(torch.device("cuda:0"), torch.device("cuda:1")), cpu_threads=1)
    manifests = [json.loads(path.read_text()) for path in paths]
    assert len(list(tmp_path.glob("build_*"))) == 2
    assert manifests[0]["tasks"] == manifests[2]["tasks"]
    for index, manifest in enumerate(manifests):
        records = manifest["conditions"]
        assert len(records) == len({row["condition_id"] for row in records}) == 4
        assert sorted(row["teacher_demo_indices"][0] for row in records) == [0, 1, 2, 3]
        assert len({row["pid"] for row in records}) == 2
        value = 1 if index < 2 else 2
        assert all((row["loads"], row["writer_value"], row["meta_value"], row["vl_meta_value"])
                   == (value, value, value * 10, value * 100) for row in records)
        long = next(row for row in records if row["teacher_demo_indices"] == [0])
        assert sum(row["pid"] != long["pid"] and row["finished"] < long["finished"] for row in records) >= 2
        assert manifest["materialization_execution"]["workers"] == 2
    assert {child.pid for child in mp.active_children()} == children


@pytest.mark.parametrize("mode", ["exception", "exit", "wrong_id"])
def test_worker_failure_cannot_seal_or_leave_live_workers(concurrent_requests, mode):
    requests, runs = concurrent_requests
    for run in runs.values():
        run["config"]["fixture"]["mode"] = mode
    children = {child.pid for child in mp.active_children()}
    with pytest.raises((RuntimeError, ValueError), match="injected condition|terminated abruptly|mismatched"):
        materialization.materialize_requests(asset_root=ROOT, requests=requests[:1],
            devices=(torch.device("cuda:0"), torch.device("cuda:1")), cpu_threads=1)
    assert not (Path(requests[0]["output"]) / "manifest.json").exists()
    assert {child.pid for child in mp.active_children()} == children


def test_cli_routes_device_list_and_keeps_single_device_exclusive(tmp_path, monkeypatch):
    request = tmp_path / "requests.json"
    request.write_text("[]")
    calls = []
    monkeypatch.setattr(materialization, "materialize_requests", lambda **kwargs: calls.append(kwargs) or [])
    args = ["materialize_writer.py", "--requests-json", str(request),
            "--devices", "cuda:0,cuda:1,cuda:2,cuda:3", "--cpu-threads", "2"]
    monkeypatch.setattr("sys.argv", args)
    materialization.main()
    assert calls[0]["devices"] == tuple(torch.device(f"cuda:{index}") for index in range(4))
    assert calls[0]["cpu_threads"] == 2 and "device" not in calls[0]
    monkeypatch.setattr("sys.argv", [*args, "--device", "cpu"])
    with pytest.raises(SystemExit):
        materialization.main()
    with pytest.raises(ValueError, match="distinct"):
        workers.execution_devices(devices=("cuda", "cuda:0"))
