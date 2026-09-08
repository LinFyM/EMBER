"""Supervised sampling, shared updates and complete exact-resume boundaries."""
from copy import deepcopy
from pathlib import Path
import json
import time
from types import SimpleNamespace

import pytest
import torch

from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.pi05_source_checkpoint import DistributedContext
from ember.pi05_source_contract import append_jsonl, reconcile_metrics
from ember.writer import learning_data
from ember.writer.learning_data import WriterTrainingData, load_learning_tasks
from ember.writer.training import _update, _config, _optimization, _training_state, _run_segment


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_validation_cannot_enter_gradient_loader():
    with pytest.raises(ValueError, match="fixed development split"):
        load_learning_tasks(ROOT, [1])
    with pytest.raises(ValueError, match="excludes Test"):
        load_learning_tasks(ROOT, [6], role="test")


@pytest.fixture
def config():
    return _config(ROOT / "configs/pi05_horizon_writer_v1.json")


@pytest.fixture
def sampler(monkeypatch, config):
    tasks = {task: SimpleNamespace(suite=f"suite{task // 6}", authority=object()) for task in range(24)}
    monkeypatch.setattr(learning_data, "load_learning_tasks", lambda *_: tasks)
    monkeypatch.setattr(learning_data, "RawTeacherVideoStore", lambda *a, **kw:
                        SimpleNamespace(frame_counts=lambda task, demo: (20 + demo, 5 + demo // 5)))
    monkeypatch.setattr(learning_data, "FunctionalQueryDataset", lambda *a, **kw:
                        SimpleNamespace(task_episode_rows={}))
    return WriterTrainingData(ROOT, config["data"])


def test_actual_sampler_covers_suites_dynamic_k_and_restores_all_streams(sampler):
    draws = [sampler.next_iteration() for _ in range(24)]
    assert all(len({sampler.tasks[d["task"]].suite for d in iteration}) == 4 for iteration in draws)
    assert {len(d["video_demos"]) for iteration in draws for d in iteration} == {1, 2, 4}
    for iteration in draws:
        for draw in iteration:
            assert len(set(draw["video_demos"])) == len(draw["video_demos"])
            assert "episodes" not in draw
    saved = deepcopy(sampler.sampler_state())
    assert sum(saved["task_occurrences"].values()) == 24 * 4
    assert set(saved["streams"]) == {"task", "K", "video", "query"}
    future = [sampler.next_iteration() for _ in range(3)]
    sampler.restore_sampler(saved)
    assert [sampler.next_iteration() for _ in range(3)] == future
    corrupt = deepcopy(saved)
    corrupt["next_step"] += 1
    with pytest.raises(ValueError, match="cursor"):
        sampler.restore_sampler(corrupt)


def test_config_is_complete_and_rejects_silent_graph_or_supervision_reduction(tmp_path, config):
    import json
    assert config["model"]["horizon"] == 50 and config["model"]["blocks"] == 4
    assert config["model"]["factor_width"] == 256 and config["data"]["queries_per_task"] == 64
    assert "total_steps" not in config["data"]
    for section, key, value in (("model", "blocks", 3), ("model", "horizon", 25), ("data", "queries_per_task", 16)):
        changed = deepcopy(config)
        changed[section][key] = value
        path = tmp_path / "config.json"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="scientific contract"):
            _config(path)


def test_held_action_diagnostic_is_fixed_and_does_not_consume_training_sampler(monkeypatch, sampler):
    class HeldQueries:
        def __init__(self, authorities, demo_indices, action_chunk_size):
            assert tuple(demo_indices) == (42, 43, 44, 45) and action_chunk_size == 50
            self.task_episode_rows = {0: {demo: (demo * 2, demo * 2 + 1) for demo in demo_indices}}

        def __getitem__(self, row):
            return {"demo_index": row // 2, "frame_index": row % 2, "action": torch.zeros(50, 7)}

    monkeypatch.setattr(learning_data, "FunctionalQueryDataset", HeldQueries)
    before = deepcopy(sampler.sampler_state())
    _, trace = sampler.diagnostic_batch(0, seed=20260908, count=128)
    _, repeat = sampler.diagnostic_batch(0, seed=20260908, count=128)
    assert trace == repeat and set(trace["action_demos"]) == {42, 43, 44, 45}
    assert sampler.sampler_state() == before


class _ToySupervisedEngine:
    def __init__(self, state):
        self.state = state
        self.draws = []
        self.versions = []

    def backward(self, draw):
        self.draws.append(draw)
        self.versions.append(tuple(p.detach().clone() for p in self.state.parameters()))
        # One globally weighted condition. This is an update-cadence oracle,
        # not a proxy for the native FM/RL control objective.
        loss = .25 * sum(p.square().sum() for p in self.state.parameters())
        loss.backward()
        return {"flow_loss": float(loss.detach()), "queries": 64}

def test_supervised_update_uses_all_tasks_once_without_rollout_or_trust(sampler, config):
    state = torch.nn.Module()
    state.writer, state.meta = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    runtime = SimpleNamespace(state=state)
    engine = _ToySupervisedEngine(state)
    optimizer, scheduler = _optimization(state, config)
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    rows, norms = _update(engine, runtime, sampler, context, config, optimizer, scheduler, 1)
    assert len(rows) == 4 and sampler.sampler_state()["next_step"] == 1
    assert scheduler.last_epoch == 1
    assert all(int(value["step"]) == 1 for value in optimizer.state.values())
    assert norms["writer_grad_norm"] > 0 and norms["meta_grad_norm"] > 0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(config["optimization"]["lr"] * 2 / 8)
    # All four condition gradients see the same parameter version before Adam.
    for version in engine.versions[1:]:
        for a, b in zip(engine.versions[0], version, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_supervised_warmup_becomes_constant_instead_of_old_cosine(config):
    state = torch.nn.Linear(1, 1)
    optimizer, scheduler = _optimization(state, config)
    rates = []
    for _ in range(12):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad()
        state(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        scheduler.step()
    assert rates[:8] == pytest.approx([config["optimization"]["lr"] * k / 8 for k in range(1, 9)])
    assert rates[8:] == pytest.approx([config["optimization"]["lr"]] * 4)


def test_checkpoint_restores_next_update_and_sampler(tmp_path, monkeypatch, config):
    monkeypatch.setattr("ember.ecp.checkpoint.capture_rng", lambda _: torch.get_rng_state())
    monkeypatch.setattr("ember.ecp.checkpoint.restore_rng", lambda state, _: torch.set_rng_state(state))
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    model = torch.nn.Linear(3, 2)
    optimizer, scheduler = _optimization(model, config)

    def update():
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.randn(4, 3)).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss.detach()

    update()
    sampler = {"next_step": 1, "task_occurrences": {7: 1}}
    training = _training_state(config, 1)
    checkpoint = save_ecp_checkpoint(
        output_dir=tmp_path, macro=1, stage="supervised_test", context=context,
        model=model, optimizer=optimizer, scheduler=scheduler,
        run_contract_schema="supervised_test_v1", metrics_rows=4, sampler_state=sampler, training_state=training,
    )
    expected_loss = update()
    expected_weight = model.weight.detach().clone()
    args = dict(checkpoint=checkpoint, stage="supervised_test", context=context,
                model=model, optimizer=optimizer, scheduler=scheduler, run_contract_schema="supervised_test_v1")
    with pytest.raises(ValueError, match="cursor changed"):
        load_ecp_checkpoint(**args, expected_sampler_state={"next_step": 3})
    restored = {}
    assert load_ecp_checkpoint(**args, restored_state=restored) == (1, 4)
    assert restored == {"sampler_state": sampler, "training_state": training}
    assert scheduler.last_epoch == 1
    torch.testing.assert_close(update(), expected_loss)
    torch.testing.assert_close(model.weight, expected_weight)


def test_resume_retains_distinct_orphaned_exposure_and_step_evidence(tmp_path):
    for name in ("exposures", "metrics"):
        path = tmp_path / f"{name}.jsonl"
        for step in (1, 2):
            append_jsonl(path, {"step": step, "kind": name})
        assert reconcile_metrics(path, 1, 1, cursor_key="step", packet_label=name) == 1
    packets = list((tmp_path / "failure_packets").glob("*.jsonl"))
    assert len(packets) == 2
    assert all('"step": 2' in p.read_text() for p in packets)


@pytest.mark.parametrize("stop", [1, 2])
def test_segment_saves_complete_supervised_boundary(tmp_path, monkeypatch, sampler, config, stop):
    monkeypatch.setattr("ember.ecp.checkpoint.capture_rng", lambda _: torch.get_rng_state())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_: 0)
    state = torch.nn.Module()
    state.writer, state.meta = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    runtime = SimpleNamespace(state=state)
    engine = _ToySupervisedEngine(state)
    optimizer, scheduler = _optimization(state, config)
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    args = SimpleNamespace(output=tmp_path, stop_after_step=stop, mode="profile")
    _run_segment(args, context, config, runtime, sampler, engine, optimizer, scheduler,
                 (0, 0), stop, time.perf_counter())
    completion = json.loads((tmp_path / "completion.json").read_text())
    assert completion["optimizer_updates"] == stop and not completion["scientific_qualification"]
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == stop
    assert len((tmp_path / "exposures.jsonl").read_text().splitlines()) == stop * 4
    checkpoint, = (tmp_path / "checkpoints").glob("macro_*")
    trainer = torch.load(checkpoint / "trainer_state.pt", weights_only=False)
    assert trainer["training_state"] == _training_state(config, stop)
    assert trainer["sampler_state"]["next_step"] == stop
    assert trainer["scheduler"]["last_epoch"] == stop
