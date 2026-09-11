"""Supervised sampling, shared updates and complete exact-resume boundaries."""
from copy import deepcopy
from pathlib import Path
import json
import random
import time
from types import SimpleNamespace

import pytest
import torch

from ember.ecp.checkpoint import load_ecp_checkpoint, save_ecp_checkpoint
from ember.pi05_source_checkpoint import DistributedContext
from ember.pi05_source_contract import append_jsonl, reconcile_metrics
from ember.writer import learning_data
from ember.writer.learning_data import WriterTrainingData, load_learning_tasks
from ember.writer.training import _update, _config, _optimization, _training_state, _run_segment, _execute_step, _segment_limit, _checkpoint_nodes, _publish_contract
from ember.writer.replay import sum_writer_gradients


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_validation_cannot_enter_gradient_loader():
    with pytest.raises(ValueError, match="fixed development split"):
        load_learning_tasks(ROOT, [1])
    with pytest.raises(ValueError, match="excludes Test"):
        load_learning_tasks(ROOT, [6], role="test")


@pytest.fixture
def config(tmp_path):
    # Hold a complete K1 recipe and a short regular evidence schedule;
    # actual segment nodes are separately registered by each launch.
    value = json.loads((ROOT / "configs/pi05_video_change_reference.json").read_text())
    value["data"]["cardinalities"] = [1]
    value["data"]["conditions_per_task"] = 1
    value["data"]["version"] = "train24_supervised_suite_rng_cross_episode_k1_v2"
    value["optimization"].pop("fresh_joint_writer_and_meta", None)
    value["optimization"]["joint_train_all_writer_modules"] = True
    value["evidence"]["checkpoint_updates"] = [50, 100]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    return _config(path)


@pytest.fixture
def sampler(monkeypatch, config):
    tasks = {task: SimpleNamespace(suite=f"suite{task // 6}", authority=object()) for task in range(24)}
    monkeypatch.setattr(learning_data, "load_learning_tasks", lambda *_: tasks)
    monkeypatch.setattr(learning_data, "RawTeacherVideoStore", lambda *a, **kw:
                        SimpleNamespace(frame_counts=lambda task, demo: (20 + demo, 5 + demo // 5)))
    monkeypatch.setattr(learning_data, "FunctionalQueryDataset", lambda *a, **kw:
                        SimpleNamespace(task_episode_rows={}))
    return WriterTrainingData(ROOT, config["data"])


def test_actual_sampler_covers_suites_with_only_k1_and_restores_all_streams(sampler):
    draws = [sampler.next_iteration() for _ in range(24)]
    assert all(len({sampler.tasks[d["task"]].suite for d in iteration}) == 4 for iteration in draws)
    assert {len(d["video_demos"]) for iteration in draws for d in iteration} == {1}
    for iteration in draws:
        for draw in iteration:
            assert len(set(draw["video_demos"])) == len(draw["video_demos"])
            assert "episodes" not in draw
    saved = deepcopy(sampler.sampler_state())
    assert sum(saved["task_occurrences"].values()) == 24 * 4
    assert set(saved["streams"]) == {"task", "video", "query"}
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
    for section, key, value in (("model", "blocks", 3), ("model", "horizon", 25), ("data", "queries_per_task", 16),
                                ("data", "cardinalities", [1, 2, 4]), ("data", "tasks_per_update", 3),
                                ("data", "conditions_per_task", None), ("data", "conditions_per_task", True),
                                ("data", "conditions_per_task", 3)):
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
        loss = draw["query_count"] / 256 * sum(p.square().sum() for p in self.state.parameters())
        loss.backward()
        return {"flow_loss": float(loss.detach()) * 256 / draw["query_count"], "queries": draw["query_count"]}

@pytest.mark.parametrize("conditions", [1, 2])
def test_supervised_update_uses_all_tasks_once_without_rollout_or_trust(sampler, config, conditions):
    config["data"]["conditions_per_task"] = sampler.conditions_per_task = conditions
    state = torch.nn.Module()
    state.writer, state.meta = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    runtime = SimpleNamespace(state=state)
    engine = _ToySupervisedEngine(state)
    optimizer, scheduler = _optimization(state, config)
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    rows, norms = _update(engine, runtime, sampler, context, config, optimizer, scheduler, 1)
    assert len(rows) == 4 * conditions and sampler.sampler_state()["next_step"] == 1
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
@pytest.mark.parametrize("conditions", [1, 2])
def test_segment_saves_complete_supervised_boundary(tmp_path, monkeypatch, sampler, config, stop, conditions):
    config["data"]["conditions_per_task"] = sampler.conditions_per_task = conditions
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
    exposures = [json.loads(line) for line in (tmp_path / "exposures.jsonl").read_text().splitlines()]
    assert len(exposures) == stop * 4 * conditions
    metrics = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[-1])
    assert metrics["supervised_queries"] == sum(row["queries"] for row in exposures) == stop * 256
    assert metrics["condition_exposures"] == metrics["exposures"] == stop * 4 * conditions
    assert metrics["task_exposures"] == stop * 4
    latest = [row for row in exposures if row["step"] == stop]
    assert metrics["mean_flow_loss"] == pytest.approx(sum(row["flow_loss"] for row in latest) / len(latest))
    checkpoint, = (tmp_path / "checkpoints").glob("macro_*")
    trainer = torch.load(checkpoint / "trainer_state.pt", weights_only=False)
    assert trainer["training_state"] == _training_state(config, stop)
    assert trainer["sampler_state"]["next_step"] == stop
    assert trainer["scheduler"]["last_epoch"] == stop


def test_execution_chunking_is_configurable_but_formal_nodes_remain_regular(tmp_path, config):
    changed = deepcopy(config)
    changed["model"]["activation_checkpoint"] = False
    changed["model"]["edge_chunk"] = 24
    path = tmp_path / "config.json"
    path.write_text(json.dumps(changed))
    assert _config(path)["model"]["activation_checkpoint"] is False
    for nodes in ([24, 64], [100, 50], [50, 50]):
        changed["evidence"]["checkpoint_updates"] = nodes
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="multiples of 50"):
            _config(path)


@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_four_task_gradient_is_independent_of_uneven_rank_assignment(monkeypatch, config, world_size):
    """Real placement and SUM helper, against an explicit logical-batch oracle."""
    state = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        state.weight.copy_(torch.tensor([[.2, -.3]]))
    features = torch.tensor([[1., 0.], [0., 2.], [3., -1.], [-1., 4.]])
    targets = torch.tensor([[1.], [-1.], [2.], [0.]])
    loss = (state(features) - targets).square().mean()
    loss.backward()
    logical_gradient = state.weight.grad.clone()
    draws = tuple({"task": task, "occurrence": 0, "video_demos": (task,), "query_seed": 17 + task,
                   "frames": [19, 11, 7, 4][task], "job_id": task, "condition_index": 0,
                   "query_offset": 0, "query_count": 64} for task in range(4))
    data = SimpleNamespace(tasks={task: SimpleNamespace(suite=f"suite{task}") for task in range(4)})
    class Engine:
        def __init__(self, model):
            self.model = model
        def backward(self, draw):
            task = draw["task"]
            value = .25 * (self.model(features[task:task + 1]) - targets[task:task + 1]).square().sum()
            value.backward()
            return {"queries": 64, "flow_loss": float(value.detach())}
    models, local_rows = [], []
    for rank in range(world_size):
        model = deepcopy(state)
        model.zero_grad(set_to_none=True)
        context = DistributedContext(rank, rank, world_size, torch.device("cpu"))
        local_rows.append(_execute_step(Engine(model), data, context, config, draws, 1))
        models.append(model)
    assert sorted(row["task"] for rows in local_rows for row in rows) == [0, 1, 2, 3]
    assert sum(row["queries"] for rows in local_rows for row in rows) == 256
    if world_size == 3:
        assert sorted(map(len, local_rows)) == [1, 1, 2]
    combined = sum(model.weight.grad for model in models)
    def reduce(gradient, op):
        assert op == torch.distributed.ReduceOp.SUM
        gradient.copy_(combined)
    monkeypatch.setattr("ember.writer.replay.dist.all_reduce", reduce)
    for model in models:
        sum_writer_gradients(tuple(model.parameters()), world_size=world_size)
        torch.testing.assert_close(model.weight.grad, logical_gradient, rtol=1e-5, atol=1e-6)


def test_rank_count_cannot_expand_batch_or_create_idle_replicas(config):
    with pytest.raises(ValueError, match="useful ranks"):
        _execute_step(None, None, SimpleNamespace(world_size=5), config, (), 1)


def test_new_segment_nodes_do_not_mutate_or_invalidate_learning_contract(tmp_path, config):
    original = {"schema_version": "run", "stage": "supervised", "mode": "formal", "config": config,
                "model_config": config["model"], "topology": {"world_size": 4}, "source": {"policy": "frozen"},
                "training": {"checkpoint_updates": [50, 100]}}
    path = tmp_path / "run_contract.json"
    _publish_contract(path, original, resume=False)
    next_args = SimpleNamespace(mode="formal", stop_after_step=200, checkpoint_updates="150,200")
    assert _segment_limit(next_args, config) == 200
    assert _checkpoint_nodes(next_args, config) == (150, 200)
    resumed = deepcopy(original)
    resumed["training"]["checkpoint_updates"] = [150, 200]
    _publish_contract(path, resumed, resume=True)
    assert json.loads(path.read_text()) == original
    next_args.checkpoint_updates = "150,250"
    with pytest.raises(ValueError, match="stop at the last"):
        _segment_limit(next_args, config)
    resumed["topology"]["world_size"] = 3
    with pytest.raises(ValueError, match="topology"):
        _publish_contract(path, resumed, resume=True)


def test_physical_microbatches_leave_the_shared_recipe_unchanged(config):
    from ember.writer.training import _execution_config
    before = deepcopy(config)
    args = SimpleNamespace(policy_microbatches="8,4,8,8")
    for rank, expected in enumerate((8, 4, 8, 8)):
        local, plan = _execution_config(args, config, SimpleNamespace(world_size=4, rank=rank))
        assert local["runtime"]["policy_microbatch"] == expected
        assert local["data"]["tasks_per_update"] * local["data"]["queries_per_task"] == 256
        assert local["optimization"] == before["optimization"]
        assert plan == [8, 4, 8, 8]
    assert config == before
    with pytest.raises(ValueError, match="per rank"):
        _execution_config(args, config, SimpleNamespace(world_size=3, rank=0))


def test_dual_view_is_an_explicit_config_change_not_an_exact_resume(tmp_path, config):
    changed = deepcopy(config)
    changed["observer"]["camera_view"] = "dual"
    cfg_path = tmp_path / "dual.json"
    cfg_path.write_text(json.dumps(changed))
    assert _config(cfg_path)["observer"]["camera_view"] == "dual"
    original = {"schema_version": "run", "stage": "supervised", "mode": "formal", "config": config,
                "model_config": config["model"], "topology": {"world_size": 4}, "source": {"policy": "frozen"}}
    path = tmp_path / "run_contract.json"
    _publish_contract(path, original, resume=False)
    with pytest.raises(ValueError, match="exact-resume contract differs: config"):
        _publish_contract(path, {**original, "config": changed}, resume=True)
    changed["observer"]["camera_view"] = "unknown"
    cfg_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="camera_view"):
        _config(cfg_path)


def test_two_conditions_preserve_task_query_streams_and_resume(sampler, config):
    original = deepcopy(sampler.sampler_state())
    one = [sampler.next_iteration() for _ in range(24)]
    one_state = deepcopy(sampler.sampler_state())
    sampler.restore_sampler(original)
    sampler.conditions_per_task = 2
    two = [sampler.next_iteration() for _ in range(24)]
    two_state = deepcopy(sampler.sampler_state())
    for singles, pairs in zip(one, two, strict=True):
        assert [draw["job_id"] for draw in pairs] == list(range(8))
        for single, first, second in zip(singles, pairs[::2], pairs[1::2], strict=True):
            for key in ("task", "occurrence", "query_seed"):
                assert single[key] == first[key] == second[key]
            assert first["video_demos"] != second["video_demos"]
            assert len(first["video_demos"]) == len(second["video_demos"]) == 1
            assert (first["query_offset"], second["query_offset"]) == (0, 32)
            assert first["query_count"] == second["query_count"] == 32
    for name in ("task", "query"):
        assert one_state["streams"][name] == two_state["streams"][name]
    assert one_state["task_occurrences"] == two_state["task_occurrences"]
    expected = [sampler.next_iteration() for _ in range(3)]
    sampler.restore_sampler(two_state)
    assert [sampler.next_iteration() for _ in range(3)] == expected


def test_action_conditions_partition_original_full_selection_and_seed(sampler):
    from ember.writer.functional import task_logical_batch_policy_rng_seed

    class Queries:
        task_episode_rows = {0: {demo: tuple(demo * 100 + frame for frame in range(demo))
                                for demo in range(16, 42)}}
        loaded = []
        def __getitem__(self, index):
            self.loaded.append(index)
            return {"demo_index": index // 100, "frame_index": index % 100,
                    "action": torch.tensor([index], dtype=torch.float32)}
    sampler.queries = Queries()
    sampler.query_rows = sampler.queries.task_episode_rows
    rng = random.Random(123)
    indices = [rng.choice(sampler.query_rows[0][rng.choice(sampler.action_pool)]) for _ in range(64)]
    expected_seed = task_logical_batch_policy_rng_seed(
        optimization_seed=sampler.seed, task_id=0, task_visit=9,
        demo_indices=[index // 100 for index in indices], frame_indices=[index % 100 for index in indices],
    )
    before = deepcopy(sampler.sampler_state())
    outputs = [sampler.action_batch(0, 9, (demo,), query_seed=123, query_offset=offset, query_count=32)
               for demo, offset in ((3, 0), (4, 32))]
    assert sampler.queries.loaded == indices  # Select full64 twice, read only each condition's actual32.
    assert torch.cat([batch["action"] for batch, _ in outputs]).flatten().tolist() == indices
    assert all(trace["policy_rng_seed"] == expected_seed for _, trace in outputs)
    assert all(trace["policy_random_batch_size"] == 64 for _, trace in outputs)
    assert sampler.sampler_state() == before
    with pytest.raises(ValueError, match="overlap"):
        sampler.action_batch(0, 0, (16,), query_seed=123)
    with pytest.raises(ValueError, match="exceeds"):
        sampler.action_batch(0, 0, (0,), query_seed=123, query_offset=40, query_count=32)


@pytest.mark.parametrize("world_size", range(1, 7))
def test_eight_condition_jobs_preserve_global_gradient_and_use_all_ranks(monkeypatch, sampler, config, world_size):
    config["data"]["conditions_per_task"] = sampler.conditions_per_task = 2
    draws = sampler.next_iteration()
    state = torch.nn.Linear(2, 1, bias=False)
    features = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 8
    targets = torch.linspace(-1, 1, 8).reshape(8, 1)
    (state(features) - targets).square().mean().backward()
    expected = state.weight.grad.clone()
    class Engine:
        def __init__(self, model):
            self.model = model
        def backward(self, draw):
            job = draw["job_id"]
            loss = (self.model(features[job:job + 1]) - targets[job:job + 1]).square().mean()
            (loss / 8).backward()
            return {"queries": 32, "flow_loss": float(loss.detach())}
    models, rows = [], []
    for rank in range(world_size):
        model = deepcopy(state)
        model.zero_grad(set_to_none=True)
        context = DistributedContext(rank, rank, world_size, torch.device("cpu"))
        local = _execute_step(Engine(model), sampler, context, config, draws, 1)
        assert local  # No idle replica; repeated task IDs cannot overwrite jobs.
        rows.extend(local)
        models.append(model)
    assert sorted(row["job_id"] for row in rows) == list(range(8))
    assert sum(row["queries"] for row in rows) == 256
    assert sum(row["condition_weight"] for row in rows) == 1
    combined = sum(model.weight.grad for model in models)
    def reduce(gradient, op):
        assert op == torch.distributed.ReduceOp.SUM
        gradient.copy_(combined)
    monkeypatch.setattr("ember.writer.replay.dist.all_reduce", reduce)
    for model in models:
        sum_writer_gradients(tuple(model.parameters()), world_size=world_size)
        torch.testing.assert_close(model.weight.grad, expected)
    from ember.writer.training import _logical_batch
    contract = _logical_batch(config)
    assert contract["conditions"] == 8 and contract["queries_per_condition"] == 32
    assert contract["queries_per_update"] == 256 and contract["K"] == 1


def test_two_conditions_are_explicit_and_cannot_resume_one_condition(tmp_path, config):
    changed = deepcopy(config)
    changed["data"]["conditions_per_task"] = 2
    cfg_path = tmp_path / "two.json"
    cfg_path.write_text(json.dumps(changed))
    assert _config(cfg_path)["data"]["conditions_per_task"] == 2
    contract = {"schema_version": "run", "stage": "supervised", "mode": "formal", "config": config,
                "model_config": config["model"], "topology": {"world_size": 4}, "source": {"policy": "frozen"}}
    path = tmp_path / "run_contract.json"
    _publish_contract(path, contract, resume=False)
    with pytest.raises(ValueError, match="exact-resume contract differs: config"):
        _publish_contract(path, {**contract, "config": changed}, resume=True)


@pytest.mark.parametrize("key,value", [("job_id", 0), ("condition_index", 0),
                                      ("query_offset", 0), ("query_seed", -1),
                                      ("video_demos", (99, 100))])
def test_duplicate_or_misaligned_conditions_fail_before_backward(sampler, config, key, value):
    config["data"]["conditions_per_task"] = sampler.conditions_per_task = 2
    draws = list(sampler.next_iteration())
    draws[1] = {**draws[1], key: value}
    with pytest.raises(ValueError):
        _execute_step(None, sampler, SimpleNamespace(world_size=6, rank=0), config, draws, 1)
    draws = list(sampler.next_iteration())
    draws[1] = {**draws[1], "video_demos": draws[0]["video_demos"]}
    with pytest.raises(ValueError, match="distinct K1 videos"):
        _execute_step(None, sampler, SimpleNamespace(world_size=6, rank=0), config, draws, 1)
