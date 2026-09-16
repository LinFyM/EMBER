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
from ember.writer.training import _update, _config, _optimization, _training_state, _run_segment, _execute_step, _segment_limit, _checkpoint_nodes, _publish_contract, _run_contract, observer_mode_contract, _data_config, _publish_event_plan, extension_record_path
from ember.writer.replay import sum_writer_gradients


ROOT = Path(__file__).resolve().parents[1]


def test_registered_formal_recipe_reaches_git_guard_before_device_initialization(monkeypatch):
    from ember.writer import training

    monkeypatch.setattr(training, "git_state", lambda _: {"branch": "main"})
    args = SimpleNamespace(mode="formal", config=ROOT / "configs/pi05_writer.json")
    configured = training._config(args.config)
    configured["status"] = "registered_source_aligned_v52_learning"
    configured["evidence"]["profile_registration"]["status"] = "complete"
    monkeypatch.setattr(training, "_config", lambda _: configured)
    with pytest.raises(ValueError, match="clean pushed detached worktree"):
        training.run(args)


def test_formal_launch_rejects_unregistered_recipe(tmp_path):
    from ember.writer import training

    value = json.loads((ROOT / "configs/pi05_writer.json").read_text())
    value["status"] = "unregistered"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="post-profile checkpoint and exposure registration"):
        training.run(SimpleNamespace(mode="formal", config=path))


def test_fixed_validation_cannot_enter_gradient_loader():
    with pytest.raises(ValueError, match="fixed target split"):
        load_learning_tasks(ROOT, [1])
    with pytest.raises(ValueError, match="fixed target split"):
        load_learning_tasks(ROOT, [6])


@pytest.fixture
def config(tmp_path):
    # Hold a complete K1 recipe and a short test-only evidence schedule;
    # actual segment nodes are separately registered by each launch.
    value = json.loads((ROOT / "configs/pi05_writer.json").read_text())
    value["data"]["cardinalities"] = [1]
    value["data"]["conditions_per_task"] = 1
    value["optimization"].pop("fresh_joint_writer_and_meta", None)
    value["optimization"]["joint_train_all_writer_modules"] = True
    value["evidence"]["checkpoint_updates"] = [50, 100]
    value["evidence"]["supervised_validation"]["optimizer_updates"] = [0, 50, 100]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    return _config(path)


@pytest.fixture
def sampler(monkeypatch, config):
    tasks = {task: SimpleNamespace(suite=f"suite{task // 6}", authority=object(), episode_lengths=(4,) * 50)
             for task in range(24)}
    monkeypatch.setattr(learning_data, "load_learning_tasks", lambda *_: tasks)
    monkeypatch.setattr(learning_data, "RawTeacherVideoStore", lambda *a, **kw:
                        SimpleNamespace(frame_counts=lambda task, demo: (4, 2), close=lambda: None))
    monkeypatch.setattr(learning_data, "FunctionalQueryDataset", lambda *a, **kw:
                        SimpleNamespace(task_episode_rows={task: {demo: (0, 1, 2) for demo in range(46)}
                                                            for task in range(24)}, close=lambda: None))
    return WriterTrainingData(ROOT, config["data"])


def test_config_is_complete_and_rejects_silent_graph_or_supervision_reduction(tmp_path, config):
    assert config["model"]["action_horizon"] == 50 and config["model"]["semantic_core_blocks"] == 2
    assert config["model"]["procedure_blocks"] == 2 and config["data"]["queries_per_task"] == 21
    for section, key, value in (("model", "procedure_blocks", 3), ("model", "action_horizon", 25),
                               ("data", "queries_per_task", 16), ("observer", "vl_meta_rank", 0),
                               ("data", "cardinalities", [1, 2, 4]), ("data", "tasks_per_update", 3),
                               ("data", "conditions_per_task", None), ("data", "conditions_per_task", 2)):
        changed = deepcopy(config)
        changed[section][key] = value
        path = tmp_path / "config.json"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="contract|architecture"):
            _config(path)


@pytest.mark.parametrize('camera_view,horizon_read', [('agentview', 'fixed_mean'), ('dual', 'learned')])
def test_registered_modes_keep_baseline_events_training_and_execution_pairing(tmp_path, config, camera_view, horizon_read):
    changed = deepcopy(config)
    changed['model'].update(camera_view=camera_view, horizon_read=horizon_read)
    changed['observer'].update(observer_mode_contract(changed['model']))
    path = tmp_path / 'mode.json'
    path.write_text(json.dumps(changed))
    configured = _config(path)
    assert configured['data'] == config['data'] and configured['optimization'] == config['optimization']
    assert configured['data']['grouping'] == 'baseline' and 'event_groups' not in configured['data']
    assert configured['evidence']['qualification'] == config['evidence']['qualification']
    assert configured['design'] == 'docs/source_alignment_v52_plan.md'
    assert configured['source'] == {
        'evaluation_config': 'configs/pi05_source_aligned_evaluation.json',
        'checkpoint': 'runs/outputs/pi05_source_aligned_seed7_1k_20260915/checkpoints/step_00001000'}


@pytest.mark.parametrize('camera_view,horizon_read', [('agentview', 'fixed_mean'), ('dual', 'learned')])
def test_run_contract_records_the_actual_camera_and_full_horizon_read(config, monkeypatch, tmp_path, camera_view, horizon_read):
    config['model'].update(camera_view=camera_view, horizon_read=horizon_read)
    config['observer'].update(observer_mode_contract(config['model']))
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: SimpleNamespace(uuid='cpu-fixture'))
    modules = {name: torch.nn.Linear(1, 1) for name in ('writer', 'meta', 'vl_meta', 'text_meta')}
    runtime = SimpleNamespace(state=SimpleNamespace(**modules), policy=torch.nn.Linear(1, 1).requires_grad_(False), source={})
    context = SimpleNamespace(rank=0, local_rank=0, world_size=1, numa_node=None, cpu_affinity=None)
    run = _run_contract(SimpleNamespace(output=tmp_path, mode='formal'), context, config, runtime, {})
    assert run['information_wall']['native_read'] == (
        f'same-version final {camera_view} Z and full50 H to {horizon_read}; joint three-Meta checkpoint replay')
    assert run['model_config']['camera_view'] == camera_view and run['model_config']['horizon_read'] == horizon_read


@pytest.mark.parametrize('field', ['camera_view', 'native_inputs', 'horizon_read'])
def test_observer_mode_mismatch_is_rejected(tmp_path, config, field):
    single = config['model'] | {'camera_view': 'agentview', 'horizon_read': 'fixed_mean'}
    config['observer'][field] = observer_mode_contract(single)[field]
    path = tmp_path / 'mismatched.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='scientific contract'):
        _config(path)


def test_registered_a_and_b_cannot_exact_resume_each_other(tmp_path, config):
    original = {'schema_version': 'run', 'stage': 'supervised', 'mode': 'formal', 'config': config,
                'model_config': config['model'], 'topology': {'world_size': 4}, 'source': config['source']}
    path = tmp_path / 'run_contract.json'
    _publish_contract(path, original, resume=False)
    changed = deepcopy(original)
    changed['config']['model'].update(camera_view='agentview', horizon_read='fixed_mean')
    changed['config']['observer'].update(observer_mode_contract(changed['config']['model']))
    changed['model_config'] = dict(changed['config']['model'])
    with pytest.raises(ValueError, match='exact-resume contract differs: config'):
        _publish_contract(path, changed, resume=True)


@pytest.mark.parametrize("offset", [0, True, None])
def test_writer_rejects_previous_or_implicit_action_alignment(tmp_path, config, offset):
    changed = deepcopy(config)
    changed["data"]["action_start_offset"] = offset
    path = tmp_path / "config.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="scientific contract"):
        _config(path)
    with pytest.raises(ValueError, match="future-control labels"):
        WriterTrainingData(ROOT, changed["data"])


class _ToySupervisedEngine:
    def __init__(self, state):
        self.state = state
        self.draws = []
        self.versions = []

    def backward(self, draw):
        self.draws.append(draw)
        self.versions.append(tuple(p.detach().clone() for p in self.state.parameters()))
        # One globally weighted condition. This is an update-cadence oracle,
        # not a proxy for the native main FM control objective.
        main = sum(p.square().sum() for p in self.state.parameters())
        loss = draw["query_count"] / 84 * main
        loss.backward()
        return {"flow_loss": float(main.detach()), "queries": draw["query_count"]}

def test_supervised_update_uses_all_tasks_once_without_rollout_or_trust(sampler, config):
    conditions = 1
    state = torch.nn.Module()
    state.writer, state.meta = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    state.vl_meta = torch.nn.Linear(1, 1)
    state.text_meta = torch.nn.Linear(1, 1)
    runtime = SimpleNamespace(state=state)
    engine = _ToySupervisedEngine(state)
    optimizer, scheduler = _optimization(state, config)
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    rows, norms = _update(engine, runtime, sampler, context, config, optimizer, scheduler, 1)
    assert len(rows) == 4 * conditions and sampler.sampler_state()["next_step"] == 1
    assert scheduler.last_epoch == 1
    assert all(int(value["step"]) == 1 for value in optimizer.state.values())
    assert norms["writer_grad_norm"] > 0 and norms["meta_grad_norm"] > 0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(config["optimization"]["lr"] * 2 / 101)
    # All four condition gradients see the same parameter version before Adam.
    for version in engine.versions[1:]:
        for a, b in zip(engine.versions[0], version, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_original_cosine_clock_is_not_compressed_to_the_bounded_run(config):
    import math
    optimizer, scheduler = _optimization(torch.nn.Linear(1, 1), config)
    peak = config["optimization"]["lr"]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(peak / 101)
    rates = {}
    for step in range(1, 1201):
        optimizer.step()
        scheduler.step()
        if step in (99, 100, 1200):
            rates[step] = optimizer.param_groups[0]["lr"]
    assert rates[99] == pytest.approx(peak * (1 - .01 * (1 - 1 / 101)))
    for step in (100, 1200):
        expected = 1e-5 + (peak - 1e-5) * .5 * (1 + math.cos(math.pi * step / 12000))
        assert rates[step] == pytest.approx(expected)
    assert rates[1200] > .00029


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
    conditions = 1
    monkeypatch.setattr("ember.ecp.checkpoint.capture_rng", lambda _: torch.get_rng_state())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_: 0)
    state = torch.nn.Module()
    state.writer, state.meta = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    state.vl_meta = torch.nn.Linear(1, 1)
    state.text_meta = torch.nn.Linear(1, 1)
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
    assert metrics["supervised_queries"] == sum(row["queries"] for row in exposures) == stop * 84
    assert metrics["condition_exposures"] == metrics["exposures"] == stop * 4 * conditions
    assert metrics["task_exposures"] == stop * 4
    latest = [row for row in exposures if row["step"] == stop]
    assert metrics["mean_flow_loss"] == pytest.approx(sum(row["flow_loss"] for row in latest) / len(latest))
    assert "mean_local_field_loss" not in metrics and "mean_joint_loss" not in metrics
    checkpoint, = (tmp_path / "checkpoints").glob("macro_*")
    trainer = torch.load(checkpoint / "trainer_state.pt", weights_only=False)
    assert trainer["training_state"] == _training_state(config, stop)
    assert trainer["sampler_state"]["next_step"] == stop
    assert trainer["scheduler"]["last_epoch"] == stop


def test_execution_chunking_and_profile_selected_checkpoint_nodes_are_configurable(tmp_path, config):
    changed = deepcopy(config)
    changed["model"]["activation_checkpointing"] = False
    changed["model"]["max_frames_per_encoder_call"] = 24
    path = tmp_path / "config.json"
    path.write_text(json.dumps(changed))
    assert _config(path)["model"]["activation_checkpointing"] is False
    changed["evidence"]["checkpoint_updates"] = [24, 64, 137]
    path.write_text(json.dumps(changed))
    assert _config(path)["evidence"]["checkpoint_updates"] == [24, 64, 137]
    for nodes in ([0, 50], [100, 50], [50, 50], [True, 50], [24.5, 64]):
        changed["evidence"]["checkpoint_updates"] = nodes
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="increasing positive integers"):
            _config(path)


def test_smoke_requires_explicit_stop_before_profile_node_registration():
    config = _config(ROOT / "configs/pi05_writer.json")
    config["evidence"]["checkpoint_updates"] = []
    args = SimpleNamespace(mode="smoke", stop_after_step=2, checkpoint_updates=None)
    assert _checkpoint_nodes(args, config) == () and _segment_limit(args, config) == 2
    args.stop_after_step = None
    with pytest.raises(ValueError, match="explicit positive --stop-after-step"):
        _segment_limit(args, config)
    args.mode = "formal"
    with pytest.raises(ValueError, match="registered increasing positive integers"):
        _checkpoint_nodes(args, config)


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
                   "query_offset": 0, "query_count": 21} for task in range(4))
    data = SimpleNamespace(tasks={task: SimpleNamespace(suite=f"suite{task}") for task in range(4)})
    class Engine:
        def __init__(self, model):
            self.model = model
        def backward(self, draw):
            task = draw["task"]
            value = .25 * (self.model(features[task:task + 1]) - targets[task:task + 1]).square().sum()
            value.backward()
            return {"queries": 21, "flow_loss": float(value.detach())}
    models, local_rows = [], []
    for rank in range(world_size):
        model = deepcopy(state)
        model.zero_grad(set_to_none=True)
        context = DistributedContext(rank, rank, world_size, torch.device("cpu"))
        local_rows.append(_execute_step(Engine(model), data, context, config, draws, 1))
        models.append(model)
    assert sorted(row["task"] for rows in local_rows for row in rows) == [0, 1, 2, 3]
    assert sum(row["queries"] for rows in local_rows for row in rows) == 84
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


def test_budget_extension_is_explicit_bounded_and_leaves_the_scientific_config_unchanged(tmp_path, config):
    before = deepcopy(config)
    args = SimpleNamespace(mode="formal", resume=tmp_path / "checkpoints/macro_00001200", extend_to_update=1800,
                           stop_after_step=1500, checkpoint_updates="1300,1400,1500")
    assert _segment_limit(args, config) == 1500
    assert _data_config(args, config) == config["data"] | {"maximum_updates": 1800}
    assert _data_config(SimpleNamespace(**(vars(args) | {"extend_to_update": 2100})), config)["maximum_updates"] == 2100
    assert config == before
    for change in ({"extend_to_update": None}, {"extend_to_update": 12006}, {"extend_to_update": 1501},
                   {"resume": None}, {"resume": tmp_path / "checkpoints/macro_00000900"}, {"mode": "profile"}):
        with pytest.raises(ValueError, match="budget|extension"):
            _segment_limit(SimpleNamespace(**(vars(args) | change)), config)


def test_extended_event_registration_preserves_original_and_rejects_rewritten_queries(tmp_path, sampler):
    args = SimpleNamespace(output=tmp_path, resume=None, extend_to_update=None)
    original = sampler.event_plan()
    _publish_event_plan(args, original)
    original_bytes = (tmp_path / "training_events.json").read_bytes()
    extended = WriterTrainingData(ROOT, sampler.config | {"maximum_updates": 1800})
    try:
        args.resume, args.extend_to_update = tmp_path / "macro_00001200", 1800
        args.resume.mkdir()
        torch.save({"sampler_state": sampler.sampler_state()}, args.resume / "trainer_state.pt")
        _publish_event_plan(args, extended.event_plan())
        assert (tmp_path / "training_events.json").read_bytes() == original_bytes
        assert json.loads(extension_record_path(tmp_path, "training_events.json", 1800).read_text()) == extended.event_plan()
        changed = extended.event_plan()
        changed["events"][0]["action_frames"][0] += 1
        with pytest.raises(ValueError, match="original registered training events"):
            _publish_event_plan(args, changed)
        changed = extended.event_plan()
        changed["events"][-1]["policy_rng_seed"] += 1
        with pytest.raises(ValueError, match="exact-resume training events"):
            _publish_event_plan(args, changed)
    finally:
        extended.close()


def test_repeated_extension_preserves_legacy_and_every_previously_registered_query(tmp_path, sampler):
    (tmp_path / "training_events.json").write_text(json.dumps(sampler.event_plan()))
    previous = WriterTrainingData(ROOT, sampler.config | {"maximum_updates": 1800})
    extended = WriterTrainingData(ROOT, sampler.config | {"maximum_updates": 2100})
    following = WriterTrainingData(ROOT, sampler.config | {"maximum_updates": 2400})
    legacy = tmp_path / "training_events_extended.json"
    legacy.write_text(json.dumps(previous.event_plan()))
    legacy_bytes = legacy.read_bytes()
    resume = tmp_path / "checkpoints/macro_00001800"
    resume.mkdir(parents=True)
    torch.save({"sampler_state": previous.sampler_state()}, resume / "trainer_state.pt")
    args = SimpleNamespace(output=tmp_path, resume=resume, extend_to_update=2100)
    try:
        changed = extended.event_plan()
        changed["events"][1300 * 4]["action_frames"][0] += 1
        with pytest.raises(ValueError, match="previous registered training events"):
            _publish_event_plan(args, changed)
        _publish_event_plan(args, extended.event_plan())
        registered = extension_record_path(tmp_path, "training_events.json", 2100)
        registered_bytes = registered.read_bytes()
        torch.save({"sampler_state": extended.sampler_state()}, resume / "trainer_state.pt")
        args.extend_to_update = 2400
        changed = following.event_plan()
        changed["events"][2000 * 4]["policy_rng_seed"] += 1
        with pytest.raises(ValueError, match="previous registered training events"):
            _publish_event_plan(args, changed)
        _publish_event_plan(args, following.event_plan())
        assert legacy.read_bytes() == legacy_bytes and registered.read_bytes() == registered_bytes
        assert json.loads(extension_record_path(tmp_path, "training_events.json", 2400).read_text()) == following.event_plan()
    finally:
        previous.close()
        extended.close()
        following.close()


def test_mid_segment_resume_finishes_original_registered_boundary(tmp_path, monkeypatch, config):
    from ember.writer import training

    attempted, saved, diagnosed = [], [], []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)

    def update(*args):
        attempted.append(args[-1])
        return [], {}

    monkeypatch.setattr(training, "_update", update)
    monkeypatch.setattr(training, "_record_iteration", lambda *args: args[6] + 4)
    monkeypatch.setattr(training, "_validate_actions", lambda *args: diagnosed.append(args[-1]))
    monkeypatch.setattr(training, "save_ecp_checkpoint", lambda **kwargs: saved.append(kwargs["macro"]))
    args = SimpleNamespace(output=tmp_path, mode="formal", checkpoint_updates="50,100",
                           resume=tmp_path / "checkpoints" / "macro_00000050")
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    sampler = SimpleNamespace(sampler_state=lambda: {"next_step": 100})
    _run_segment(args, context, config, SimpleNamespace(state=None), sampler, None, None, None,
                 (50, 200), 100, time.perf_counter())
    assert attempted == list(range(51, 101))
    assert saved == diagnosed == [100]
    completion = json.loads((tmp_path / "completion.json").read_text())
    assert completion["optimizer_updates"] == 100 and completion["condition_exposures"] == 400


def test_physical_microbatches_leave_the_shared_recipe_unchanged(config):
    from ember.writer.training import _execution_config
    before = deepcopy(config)
    args = SimpleNamespace(policy_microbatches="8,4,8,8")
    for rank, expected in enumerate((8, 4, 8, 8)):
        local, plan = _execution_config(args, config, SimpleNamespace(world_size=4, rank=rank))
        assert local["runtime"]["policy_microbatch"] == expected
        assert local["data"]["tasks_per_update"] * local["data"]["queries_per_task"] == 84
        assert local["optimization"] == before["optimization"]
        assert plan == [8, 4, 8, 8]
    assert config == before
    with pytest.raises(ValueError, match="per rank"):
        _execution_config(args, config, SimpleNamespace(world_size=3, rank=0))


def test_unregistered_camera_binding_is_rejected_and_cannot_exact_resume(tmp_path, config):
    changed = deepcopy(config)
    changed["observer"]["camera_view"] = "agentview"
    cfg_path = tmp_path / "dual.json"
    cfg_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="scientific contract"):
        _config(cfg_path)
    original = {"schema_version": "run", "stage": "supervised", "mode": "formal", "config": config,
                "model_config": config["model"], "topology": {"world_size": 4}, "source": {"policy": "frozen"}}
    path = tmp_path / "run_contract.json"
    _publish_contract(path, original, resume=False)
    with pytest.raises(ValueError, match="exact-resume contract differs: config"):
        _publish_contract(path, {**original, "config": changed}, resume=True)
    changed["observer"]["camera_view"] = "unknown"
    cfg_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="scientific contract"):
        _config(cfg_path)


@pytest.mark.parametrize("field", ["video_prior", "spatial_supervision", "correction_supervision",
                                  "native_output_calibration", "local_field_supervision"])
def test_retired_supervision_and_prior_configs_are_rejected(tmp_path, config, field):
    changed = deepcopy(config)
    changed[field] = {}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="scientific contract"):
        _config(path)


@pytest.mark.parametrize("key,value", [("job_id", 0), ("condition_index", 1),
                                      ("query_offset", 1), ("query_count", 20),
                                      ("video_demos", (99, 100))])
def test_duplicate_or_misaligned_conditions_fail_before_backward(sampler, config, key, value):
    draws = list(sampler.next_iteration())
    draws[1] = {**draws[1], key: value}
    with pytest.raises(ValueError, match="four distinct"):
        _execute_step(None, sampler, SimpleNamespace(world_size=4, rank=0), config, draws, 1)
