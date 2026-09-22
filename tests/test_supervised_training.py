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
from ember.writer.training import _update, _config, _optimization, _training_state, _run_segment, _execute_step, _segment_limit, _checkpoint_nodes, _publish_contract, _require_topology_resume, _run_contract, observer_mode_contract, _publish_event_plan
from ember.writer.replay import sum_writer_gradients
from ember.writer.task_execution import condition_rank_groups, merge_condition_rows


ROOT = Path(__file__).resolve().parents[1]


def _test_recipe():
    value = json.loads((ROOT / 'configs/pi05_writer.json').read_text())
    from ember.writer.learning_data import EVENT_SCHEMA
    from ember.writer.training import TASK_MIXING_DECLARATION, UPDATE_VERSION
    value['experiment'] = dict(TASK_MIXING_DECLARATION)
    value['update_version'] = UPDATE_VERSION
    value['data'].update(tasks_per_update=12, queries_per_task=7, teaching_query_counts=[3, 2, 2] * 4,
                         version=EVENT_SCHEMA, event_schema_version=EVENT_SCHEMA, teaching_episode='cross_episode')
    value['data'].pop('teaching_queries_per_task')
    return value


def test_registered_formal_recipe_reaches_git_guard_before_device_initialization(monkeypatch):
    from ember.writer import training

    monkeypatch.setattr(training, "git_state", lambda _: {"branch": "main"})
    args = SimpleNamespace(mode="formal", config=ROOT / "configs/libero_24_8_8_coverage_v1/writer_task_diversity.json")
    configured = training._config(args.config)
    configured["status"] = "registered_video_teaching_learning"
    configured["evidence"]["profile_registration"]["status"] = "complete"
    monkeypatch.setattr(training, "_config", lambda _: configured)
    with pytest.raises(ValueError, match="clean pushed detached worktree"):
        training.run(args)


def test_formal_launch_rejects_unregistered_recipe(tmp_path):
    from ember.writer import training

    value = _test_recipe()
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
    value = _test_recipe()
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
    tasks = {task: SimpleNamespace(suite=f"suite{task // 6}", authority=object(), episode_lengths=(11,) * 50)
             for task in range(24)}
    monkeypatch.setattr(learning_data, "load_learning_tasks", lambda *_: tasks)
    monkeypatch.setattr(learning_data, "RawTeacherVideoStore", lambda *a, **kw:
                        SimpleNamespace(frame_counts=lambda task, demo: (4, 2), close=lambda: None))
    monkeypatch.setattr(learning_data, "FunctionalQueryDataset", lambda *a, **kw:
                        SimpleNamespace(task_episode_rows={task: {demo: tuple(range(10)) for demo in range(46)}
                                                            for task in range(24)}, close=lambda: None))
    return WriterTrainingData(ROOT, config["data"])


def test_config_is_complete_and_rejects_silent_graph_or_supervision_reduction(tmp_path, config):
    assert config["model"]["action_horizon"] == 50 and config["model"]["procedure_blocks"] == 2
    assert config["model"]["semantic_core_blocks"] == 2 and config["data"]["queries_per_task"] == 7
    for section, key, value in (("model", "procedure_blocks", 3), ("model", "action_horizon", 25),
                               ("data", "queries_per_task", 16), ("observer", "vl_meta_rank", 0),
                               ("data", "cardinalities", [1, 2, 4]), ("data", "tasks_per_update", 3),
                               ("model", "camera_view", "eye_in_hand"),
                               ("data", "conditions_per_task", None), ("data", "conditions_per_task", 2)):
        changed = deepcopy(config)
        changed[section][key] = value
        path = tmp_path / "config.json"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="contract|architecture"):
            _config(path)


def test_dual_camera_contract_binds_both_views_without_changing_learning_recipe(tmp_path, config):
    value = deepcopy(config)
    value['model']['camera_view'] = 'dual'
    path = tmp_path / 'dual.json'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='scientific contract'):
        _config(path)
    value['observer'].update(observer_mode_contract(value['model']))
    path.write_text(json.dumps(value))
    observed = _config(path)
    assert observed['observer']['native_inputs'].startswith('full512_patch_content')
    assert observed['data'] == config['data'] and observed['optimization'] == config['optimization']


def test_teaching_recipe_keeps_A_source_and_adds_bounded_joint_supervision(config):
    assert config['data']['grouping'] == 'baseline' and 'event_groups' not in config['data']
    assert config['data']['maximum_updates'] == 1500
    assert config['design'] == 'docs/video_teaching_writer_design.md'
    assert config['source'] == {
        'evaluation_config': 'configs/pi05_source_aligned_evaluation.json',
        'checkpoint': 'runs/outputs/pi05_source_aligned_seed7_1k_20260915/checkpoints/step_00001000'}


def test_run_contract_records_full_reads_and_both_label_routes(config, monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: SimpleNamespace(uuid='cpu-fixture'))
    modules = {name: torch.nn.Linear(1, 1) for name in ('writer', 'meta', 'vl_meta', 'text_meta')}
    runtime = SimpleNamespace(state=SimpleNamespace(**modules), policy=torch.nn.Linear(1, 1).requires_grad_(False), source={})
    context = SimpleNamespace(rank=0, local_rank=0, world_size=1, numa_node=None, cpu_affinity=None)
    run = _run_contract(SimpleNamespace(output=tmp_path, mode='formal'), context, config, runtime, {})
    assert run['information_wall']['native_read'] == (
        'repeated full50 H and ordered adjacent E content; joint three-Meta replay')
    assert run['model_config']['horizon_read'] == 'repeated_full'
    assert run['information_wall']['reading_meta_in_execution'] is False


@pytest.mark.parametrize('field', ['camera_view', 'native_inputs', 'horizon_read', 'video_order'])
def test_observer_mode_mismatch_is_rejected(tmp_path, config, field):
    config['observer'][field] = 'obsolete_read'
    path = tmp_path / 'mismatched.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='scientific contract'):
        _config(path)



def test_native_architecture_changes_cannot_exact_resume(tmp_path, config):
    original = {'schema_version': 'run', 'stage': 'supervised', 'mode': 'formal', 'config': config,
                'model_config': config['model'], 'topology': {'world_size': 4}, 'source': config['source']}
    path = tmp_path / 'run_contract.json'
    _publish_contract(path, original, resume=False)
    changed = deepcopy(original)
    changed['config']['model']['procedure_blocks'] = 3
    changed['model_config'] = dict(changed['config']['model'])
    with pytest.raises(ValueError, match='exact-resume contract differs: config'):
        _publish_contract(path, changed, resume=True)


def test_dynamic_topology_resume_is_explicit_and_keeps_the_logical_update(tmp_path):
    logical_batch = {"tasks": 4, "conditions": 4, "queries_per_update": 84}
    original = {
        "schema_version": "run", "stage": "supervised", "mode": "formal", "config": {},
        "model_config": {}, "topology": {"world_size": 4}, "source": {},
        "training": {"logical_batch": logical_batch},
    }
    path = tmp_path / "run_contract.json"
    _publish_contract(path, original, resume=False)
    changed = deepcopy(original)
    changed["topology"] = {"world_size": 2}
    with pytest.raises(ValueError, match="topology"):
        _publish_contract(path, changed, resume=True)
    _publish_contract(path, changed, resume=True, allow_topology_change=True)

    changed["training"]["logical_batch"] = {"tasks": 3}
    with pytest.raises(ValueError, match="logical Writer update"):
        _publish_contract(path, changed, resume=True, allow_topology_change=True)

    dynamic = {"training_control": {"kind": "validation_early_stopping"}}
    resume = SimpleNamespace(allow_topology_change=True, resume=tmp_path / "macro", extend_from=None,
                             phase_from=None)
    assert _require_topology_resume(resume, dynamic)
    with pytest.raises(ValueError, match="ordinary dynamic"):
        _require_topology_resume(SimpleNamespace(allow_topology_change=True, resume=None,
                                                 extend_from=None, phase_from=None), dynamic)


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
        loss = (1 / 12 + 1 / 36) * main
        loss.backward()
        return {"flow_loss": float(main.detach()), "queries": draw["query_count"],
                "teaching_queries": draw["teaching_count"], "teaching_query_offset": draw["teaching_offset"],
                "teaching_loss": float(main.detach()), "teaching_weight": 1 / 36}

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
    assert len(rows) == 12 * conditions and sampler.sampler_state()["next_step"] == 1
    assert scheduler.last_epoch == 1
    assert all(int(value["step"]) == 1 for value in optimizer.state.values())
    assert norms["writer_grad_norm"] > 0 and norms["meta_grad_norm"] > 0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(config["optimization"]["lr"] * 2 / 101)
    # All four condition gradients see the same parameter version before Adam.
    for version in engine.versions[1:]:
        for a, b in zip(engine.versions[0], version, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_original_first900_clock_and_registered_cosine_tail(config):
    import math
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
    optimizer, scheduler = _optimization(torch.nn.Linear(1, 1), config)
    reference = torch.optim.AdamW(torch.nn.Linear(1, 1).parameters(), lr=3e-4)
    original = CosineDecayWithWarmupSchedulerConfig(num_warmup_steps=100, num_decay_steps=12000,
        peak_lr=3e-4, decay_lr=1e-5).build(reference, 12000)
    rates = {}
    for step in range(2101):
        if step <= 900:
            assert optimizer.param_groups[0]['lr'] == pytest.approx(reference.param_groups[0]['lr'])
        if step in (900, 1200, 1500):
            rates[step] = optimizer.param_groups[0]['lr']
        if step > 1500:
            assert optimizer.param_groups[0]['lr'] == pytest.approx(rates[1500])
        optimizer.step(); scheduler.step()
        reference.step(); original.step()
    assert rates[1200] == pytest.approx(rates[900] * .55)
    assert rates[1500] == pytest.approx(rates[900] * .1)


def test_continuation_requires_full_parent_and_rejects_scientific_or_topology_change(tmp_path, config):
    from ember.writer.continuation import CONTINUATION, prepare_continuation, require_continuation_start

    child = deepcopy(config)
    child['continuation'] = dict(CONTINUATION)
    child['data']['maximum_updates'] = 2100
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(child))
    assert _config(path)['data']['maximum_updates'] == 2100
    args = SimpleNamespace(resume=None, extend_from=None, output=tmp_path/'child')
    with pytest.raises(ValueError, match='cannot start fresh'):
        require_continuation_start(args, child)
    parent = tmp_path/'parent'
    parent.mkdir()
    args.extend_from = parent/'checkpoints/macro_00001500'
    require_continuation_start(args, child)
    original = {'schema_version':'run', 'stage':'stage', 'mode':'formal', 'config':config,
                'model_config':config['model'], 'source':config['source'], 'topology':{'world_size':2},
                'execution':{'policy_microbatches':[16,16]}, 'git':{'commit':'sealed-parent'}}
    (parent/'run_contract.json').write_text(json.dumps(original))
    candidate = {**original, 'config':child}
    prepare_continuation(args, candidate)
    assert candidate['continuation']['parent_training_commit'] == 'sealed-parent'
    for section, key, value in [('config','optimization',dict(child['optimization'], lr=1e-3)),
                                ('topology','world_size',3)]:
        changed = deepcopy(candidate)
        changed[section][key] = value
        with pytest.raises(ValueError, match='budget and evidence|topology'):
            prepare_continuation(args, changed)
    args.resume = args.extend_from
    with pytest.raises(ValueError, match='cannot also exact-resume'):
        require_continuation_start(args, child)


def test_low_lr_phase_requires_n1800_and_only_registered_schedule_change(tmp_path):
    from ember.writer.continuation import (
        LOW_LR_REPAIR, prepare_phase_continuation, require_continuation_start,
    )

    parent_config = json.loads((ROOT / "configs/libero_24_8_8_coverage_v1/writer.json").read_text())
    child_config = json.loads((ROOT / "configs/libero_24_8_8_coverage_v1/writer_low_lr_repair.json").read_text())
    parent_root = tmp_path / "parent"
    checkpoint = parent_root / "checkpoints/macro_00001800"
    checkpoint.mkdir(parents=True)
    common = {
        "schema_version": "run", "stage": "stage", "mode": "formal",
        "model_config": parent_config["model"], "source": parent_config["source"],
        "topology": {"world_size": 4}, "execution": {"policy_microbatches": [16] * 4},
        "git": {"commit": "sealed-n1800"},
    }
    (parent_root / "run_contract.json").write_text(json.dumps({**common, "config": parent_config}))
    args = SimpleNamespace(resume=None, extend_from=None, phase_from=None, output=tmp_path / "child")
    with pytest.raises(ValueError, match="cannot start fresh"):
        require_continuation_start(args, child_config)
    args.phase_from = checkpoint
    require_continuation_start(args, child_config)
    candidate = {**common, "config": child_config}
    prepare_phase_continuation(args, candidate)
    assert candidate["phase_continuation"]["fixed_lr"] == LOW_LR_REPAIR["fixed_lr"]
    assert candidate["phase_continuation"]["parent_training_commit"] == "sealed-n1800"
    changed = deepcopy(candidate)
    changed["config"]["optimization"]["teaching_weight"] = 1.0
    with pytest.raises(ValueError, match="may change only"):
        prepare_phase_continuation(args, changed)


def test_history_inheritance_requires_training_rows_but_allows_no_diagnostics(tmp_path):
    from ember.writer.continuation import inherit_history

    parent = tmp_path / "parent"
    checkpoint = parent / "checkpoints/macro_00001800"
    checkpoint.mkdir(parents=True)
    (parent / "metrics.jsonl").write_text('{"step":1800}\n')
    (parent / "exposures.jsonl").write_text('{"step":1800}\n')
    output = tmp_path / "child"
    output.mkdir()
    inherit_history(checkpoint, output)
    assert (output / "metrics.jsonl").read_text() == '{"step":1800}\n'
    assert (output / "exposures.jsonl").read_text() == '{"step":1800}\n'
    assert not (output / "diagnostics.jsonl").exists()


def test_continuation_restores_real_optimizer_rng_scheduler_and_next_update(tmp_path, monkeypatch, sampler, config):
    from ember.writer.continuation import CONTINUATION
    from ember.writer.training import _restore, STAGE, RUN_SCHEMA

    monkeypatch.setattr('ember.ecp.checkpoint.capture_rng', lambda _: torch.get_rng_state())
    monkeypatch.setattr('ember.ecp.checkpoint.restore_rng', lambda state, _: torch.set_rng_state(state))
    context = DistributedContext(0, 0, 1, torch.device('cpu'))
    model = torch.nn.Linear(3, 2)
    optimizer, scheduler = _optimization(model, config)
    def update(state, opt, clock):
        opt.zero_grad(set_to_none=True)
        loss = state(torch.randn(4,3)).square().mean()
        loss.backward(); opt.step(); clock.step()
        return loss.detach()
    for _ in range(1500):
        sampler.next_iteration()
        update(model, optimizer, scheduler)
    parent, output = tmp_path/'parent', tmp_path/'continued'
    output.mkdir()
    checkpoint = save_ecp_checkpoint(output_dir=parent, macro=1500, stage=STAGE, context=context,
        model=model, optimizer=optimizer, scheduler=scheduler, run_contract_schema=RUN_SCHEMA,
        metrics_rows=18000, sampler_state=sampler.sampler_state(), training_state=_training_state(config,1500))
    for filename, steps in [('metrics.jsonl',range(1,1501)),
                            ('exposures.jsonl',[s for s in range(1,1501) for _ in range(12)]),
                            ('diagnostics.jsonl',[s for s in config['evidence']['supervised_validation']['optimizer_updates'] for _ in range(24)])]:
        (parent/filename).write_text(''.join(json.dumps({'step':s})+'\n' for s in steps))
    expected_loss = update(model, optimizer, scheduler)
    expected_weight = model.weight.detach().clone()
    child_config = deepcopy(config)
    child_config['continuation'] = dict(CONTINUATION)
    child_config['data']['maximum_updates'] = 2100
    child_data = WriterTrainingData(ROOT, child_config['data'])
    child_model = torch.nn.Linear(3,2)
    child_opt, child_clock = _optimization(child_model, child_config)
    args = SimpleNamespace(resume=None, extend_from=checkpoint, output=output, mode='formal')
    assert _restore(args,context,SimpleNamespace(state=child_model),child_data,child_opt,child_clock,child_config)==(1500,18000)
    assert child_clock.last_epoch == child_data.next_step == 1500
    assert all(int(v['step'])==1500 for v in child_opt.state.values())
    assert all(r['occurrence']==750 for r in child_data.next_iteration())
    torch.testing.assert_close(update(child_model,child_opt,child_clock),expected_loss)
    torch.testing.assert_close(child_model.weight,expected_weight)
    assert len((parent/'metrics.jsonl').read_text().splitlines())==1500
    assert len((output/'exposures.jsonl').read_text().splitlines())==18000
    child_data.close()


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
        run_contract_schema="supervised_test_v1", metrics_rows=12, sampler_state=sampler, training_state=training,
    )
    expected_loss = update()
    expected_weight = model.weight.detach().clone()
    args = dict(checkpoint=checkpoint, stage="supervised_test", context=context,
                model=model, optimizer=optimizer, scheduler=scheduler, run_contract_schema="supervised_test_v1")
    with pytest.raises(ValueError, match="cursor changed"):
        load_ecp_checkpoint(**args, expected_sampler_state={"next_step": 3})
    restored = {}
    assert load_ecp_checkpoint(**args, restored_state=restored) == (1, 12)
    assert restored == {"sampler_state": sampler, "training_state": training}
    assert scheduler.last_epoch == 1
    torch.testing.assert_close(update(), expected_loss)
    torch.testing.assert_close(model.weight, expected_weight)


def test_checkpoint_topology_transition_restores_full_trainer_state(tmp_path, monkeypatch, config):
    restored_rng = []
    monkeypatch.setattr("ember.ecp.checkpoint.capture_rng", lambda _: torch.get_rng_state())
    monkeypatch.setattr("ember.ecp.checkpoint.restore_rng", lambda state, _: restored_rng.append(state))
    old_context = DistributedContext(0, 0, 1, torch.device("cpu"))
    model = torch.nn.Linear(3, 2)
    optimizer, scheduler = _optimization(model, config)
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(4, 3)).square().mean().backward()
    optimizer.step()
    scheduler.step()
    checkpoint = save_ecp_checkpoint(
        output_dir=tmp_path, macro=1, stage="supervised_test", context=old_context,
        model=model, optimizer=optimizer, scheduler=scheduler,
        run_contract_schema="supervised_test_v1", metrics_rows=4,
        sampler_state={"next_step": 1}, training_state={"cursor": 1},
    )
    new_context = DistributedContext(1, 1, 2, torch.device("cpu"))
    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer, restored_scheduler = _optimization(restored_model, config)
    args = dict(
        checkpoint=checkpoint, stage="supervised_test", context=new_context,
        model=restored_model, optimizer=restored_optimizer, scheduler=restored_scheduler,
        run_contract_schema="supervised_test_v1",
    )
    with pytest.raises(ValueError, match="authority"):
        load_ecp_checkpoint(**args)
    restored = {}
    assert load_ecp_checkpoint(**args, restored_state=restored, allow_world_size_change=True) == (1, 4)
    torch.testing.assert_close(restored_model.weight, model.weight)
    assert restored_scheduler.last_epoch == scheduler.last_epoch
    assert restored == {
        "sampler_state": {"next_step": 1}, "training_state": {"cursor": 1},
        "topology_resume": {
            "checkpoint_world_size": 1, "current_world_size": 2,
            "checkpoint_rng_ranks": [0], "fresh_seeded_ranks": [1],
        },
    }
    assert not restored_rng


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
    assert len(exposures) == stop * 12 * conditions
    metrics = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[-1])
    assert metrics["supervised_queries"] == sum(row["queries"] for row in exposures) == stop * 84
    assert metrics["condition_exposures"] == metrics["exposures"] == stop * 12 * conditions
    assert metrics["task_exposures"] == stop * 12
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
    config = _test_recipe()
    config["evidence"]["checkpoint_updates"] = []
    args = SimpleNamespace(mode="smoke", stop_after_step=2, checkpoint_updates=None)
    assert _checkpoint_nodes(args, config) == () and _segment_limit(args, config) == 2
    args.stop_after_step = None
    with pytest.raises(ValueError, match="explicit positive --stop-after-step"):
        _segment_limit(args, config)
    args.mode = "formal"
    with pytest.raises(ValueError, match="registered increasing positive integers"):
        _checkpoint_nodes(args, config)


@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("full_queries", [False, True])
def test_twelve_task_gradient_matches_condition_mean_oracle(monkeypatch, config, world_size, full_queries):
    """Actual placement/SUM against separate means, including auxiliary counts 2/3."""
    if full_queries:
        config['data'].update(queries_per_task=21, teaching_query_counts=[7] * 12)
    main_count = config['data']['queries_per_task']
    aux_counts = config['data']['teaching_query_counts']
    state = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        state.weight.copy_(torch.tensor([[.2, -.3]]))
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(12, 21, 2, generator=generator)
    targets = torch.randn(12, 21, 1, generator=generator)
    teaching_features = torch.randn(12, 7, 2, generator=generator)
    teaching_targets = torch.randn(12, 7, 1, generator=generator)
    main_loss = (state(features[:, :main_count]) - targets[:, :main_count]).square().mean()
    teaching_loss = sum((state(teaching_features[t, :n]) - teaching_targets[t, :n]).square().mean()
                        for t, n in enumerate(aux_counts)) / 12
    (main_loss + teaching_loss / 3).backward()
    logical_gradient = state.weight.grad.clone()
    draws = tuple({'task': task, 'occurrence': 0, 'video_demos': (task,), 'query_seed': 17 + task,
                   'frames': [19, 11, 7, 4][task % 4], 'job_id': task, 'condition_index': 0,
                   'query_offset': 0, 'query_count': main_count, 'teaching_offset': 0,
                   'teaching_count': aux_counts[task]} for task in range(12))
    data = SimpleNamespace(tasks={task: SimpleNamespace(suite=f'suite{task}') for task in range(12)})
    class Engine:
        def __init__(self, model):
            self.model = model
        def backward(self, draw):
            task, count, taught_count = draw['task'], draw['query_count'], draw['teaching_count']
            value = (self.model(features[task, :count]) - targets[task, :count]).square().mean()
            taught = (self.model(teaching_features[task, :taught_count]) - teaching_targets[task, :taught_count]).square().mean()
            (value / 12 + taught / 36).backward()
            return {'queries': count, 'flow_loss': float(value.detach()),
                    'teaching_queries': taught_count, 'teaching_query_offset': 0,
                    'teaching_loss': float(taught.detach()), 'teaching_weight': 1 / 36,
                    'action_frames': list(range(count)), 'teaching_action_frames': list(range(taught_count))}
    models, local_rows = [], []
    for rank in range(world_size):
        model = deepcopy(state)
        model.zero_grad(set_to_none=True)
        context = DistributedContext(rank, rank, world_size, torch.device('cpu'))
        local_rows.append(_execute_step(Engine(model), data, context, config, draws, 1))
        models.append(model)
    logical_rows = merge_condition_rows([row for rows in local_rows for row in rows],
                                       main_queries=main_count, teaching_queries=aux_counts)
    assert [row['task'] for row in logical_rows] == list(range(12))
    assert sum(row['condition_weight'] for row in logical_rows) == pytest.approx(1)
    assert sum(row['teaching_weight'] for row in logical_rows) == pytest.approx(1 / 3)
    assert sum(row['queries'] for row in logical_rows) == 12 * main_count
    assert sum(row['teaching_queries'] for row in logical_rows) == sum(aux_counts)
    assert sum(row['flow_loss'] / 12 for row in logical_rows) == pytest.approx(float(main_loss.detach()))
    assert sum(row['teaching_loss'] / 36 for row in logical_rows) == pytest.approx(float(teaching_loss.detach()) / 3)
    assert all(local_rows)
    if world_size in (4, 6):
        assert {len(rows) for rows in local_rows} == {12 // world_size}
    combined = sum(model.weight.grad for model in models)
    def reduce(gradient, op):
        assert op == torch.distributed.ReduceOp.SUM
        gradient.copy_(combined)
    monkeypatch.setattr('ember.writer.replay.dist.all_reduce', reduce)
    for model in models:
        sum_writer_gradients(tuple(model.parameters()), world_size=world_size)
        torch.testing.assert_close(model.weight.grad, logical_gradient, rtol=1e-5, atol=1e-6)


def test_rank_count_cannot_expand_batch_or_create_idle_replicas(config):
    with pytest.raises(ValueError, match="useful ranks"):
        _execute_step(None, None, SimpleNamespace(world_size=7), config, (), 1)


def test_complete_conditions_cannot_duplicate_drop_or_change_weights():
    counts = [3, 2, 2] * 4
    rows = [{'job_id': task, 'task': task, 'query_offset': 0, 'queries': 7,
             'condition_weight': 1 / 12, 'task_weight': 1 / 12,
             'teaching_query_offset': 0, 'teaching_queries': counts[task], 'teaching_weight': 1 / 36}
            for task in range(12)]
    kwargs = dict(main_queries=7, teaching_queries=counts)
    assert len(merge_condition_rows(rows, **kwargs)) == 12
    with pytest.raises(ValueError, match='twelve distinct'):
        merge_condition_rows(rows + [rows[0]], **kwargs)
    with pytest.raises(ValueError, match='twelve distinct'):
        merge_condition_rows(rows[:-1], **kwargs)
    rows[0]['teaching_weight'] = 3 / 84
    with pytest.raises(ValueError, match='weighting'):
        merge_condition_rows(rows, **kwargs)


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


def test_registered_event_plan_is_immutable_and_budget_cannot_grow(tmp_path, sampler, config):
    args = SimpleNamespace(output=tmp_path, resume=None)
    original = sampler.event_plan()
    _publish_event_plan(args, original)
    with pytest.raises(ValueError, match='existing event plan'):
        _publish_event_plan(args, original)
    args.resume = tmp_path / 'checkpoints/macro_00000100'
    _publish_event_plan(args, original)
    changed = deepcopy(original)
    changed['events'][0]['teaching']['action_frames'][0] += 5
    with pytest.raises(ValueError, match='exact-resume training events'):
        _publish_event_plan(args, changed)
    with pytest.raises(ValueError, match='1500-update event budget'):
        _segment_limit(SimpleNamespace(mode='formal', stop_after_step=1800, checkpoint_updates='1800'), config)


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
    changed["observer"]["camera_view"] = "dual"
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
                                      ("teaching_offset", 1), ("teaching_count", 6),
                                      ("video_demos", (99, 100))])
def test_duplicate_or_misaligned_conditions_fail_before_backward(sampler, config, key, value):
    draws = list(sampler.next_iteration())
    draws[1] = {**draws[1], key: value}
    with pytest.raises(ValueError, match="twelve distinct"):
        _execute_step(None, sampler, SimpleNamespace(world_size=4, rank=0), config, draws, 1)
