"""Actual-event balance, complete recovery and globally weighted SUM oracles."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file

from ember.experience_compiler import training
from ember.experience_compiler.contract import TASKS36
from ember.experience_compiler.data import Event, QueryData
from ember.experience_compiler.interaction import Chain
from ember.experience_compiler.learning import finish_update, fresh_optimizer
from ember.experience_compiler.sampling import EventSampler
from ember.pi05_source_checkpoint import DistributedContext, read_json, write_json_atomic


def query_data():
    data = QueryData.__new__(QueryData)
    data.tasks = {task: SimpleNamespace(episode_lengths=tuple(65 + demo % 7 for demo in range(50)))
                  for task in TASKS36}
    data.rows = {task: {demo: range(length - 1) for demo, length in enumerate(item.episode_lengths)}
                 for task, item in data.tasks.items()}
    return data


def pool(*, phase=1, recursive_tasks=TASKS36[:16]):
    conditions = []
    for task in TASKS36:
        for ordinal in range(2):
            events = [dict(endpoint=10, incoming='MT', behavior_version='MT')]
            if task in recursive_tasks:
                events.extend(dict(endpoint=20 + point, incoming=f'incoming_{point + 1:03d}.safetensors',
                                   behavior_version=f'phi{phase}_version{point}') for point in range(ordinal + 1))
            conditions.append(dict(condition_id=f'phase{phase}_task{task}_demo{ordinal}', task_id=task,
                teacher_demo=ordinal, pool=f'pool_phase{phase}', record_path=f'/actual/phase{phase}/task{task}/{ordinal}',
                complete=True, actual_incoming_parameters=True, events=events))
    return dict(complete=True, conditions=conditions)


def sampler(manifest=None, *, phase=1):
    data = query_data()
    lengths = {task: {demo: len(rows) for demo, rows in data.rows[task].items()} for task in TASKS36}
    return EventSampler(pool(phase=phase) if manifest is None else manifest, lengths,
        task_ids=TASKS36, seed=20261010, phase=phase, query_coordinates=data.query_coordinates)


def test_every_nine_updates_covers_fixed36_and180_has_task_local160_nonmt():
    stream, events = sampler(), []
    for cycle in range(20):
        cycle_rows = [event for _ in range(9) for event in stream.next_batch()]
        assert Counter(event.task_id for event in cycle_rows) == Counter({task: 1 for task in TASKS36})
        assert all(event.update == cycle * 9 + i // 4 + 1 for i, event in enumerate(cycle_rows))
        events.extend(cycle_rows)
    assert Counter(event.task_id for event in events) == Counter({task: 20 for task in TASKS36})
    for task in TASKS36:
        layers = Counter('MT' if event.incoming == 'MT' else 'nonMT' for event in events if event.task_id == task)
        assert layers == ({'MT': 10, 'nonMT': 10} if task in TASKS36[:16] else {'MT': 20})
    assert stream.coverage()['nonMT_presentations'] == 160
    assert len(events) == 720 and all(not event.masked_experience for event in events)
    with pytest.raises(StopIteration):
        stream.next_batch()


def test_uniform_condition_then_endpoint_prevents_long_chain_weighting():
    stream = sampler(pool(recursive_tasks=()))
    # One condition has three legal MT endpoints, the other has only one.
    target = stream.conditions[0]
    manifest = pool(recursive_tasks=())
    target = next(row for row in manifest['conditions'] if row['condition_id'] == target['condition_id'])
    target['events'].extend([dict(endpoint=11, incoming='MT', behavior_version='MT'),
                             dict(endpoint=12, incoming='MT', behavior_version='MT')])
    stream = sampler(manifest)
    events = [e for _ in range(180) for e in stream.next_batch() if e.task_id == target['task_id']]
    assert sorted(Counter(e.condition_id for e in events).values()) == [10, 10]
    endpoints = Counter(e.endpoint for e in events if e.condition_id == target['condition_id'])
    assert set(endpoints) == {10, 11, 12} and max(endpoints.values()) - min(endpoints.values()) <= 1


def test_full_bag_and_query_stream_resume_is_independent_of_new_physical_sharding():
    original = sampler()
    data = query_data()
    for _ in range(37):
        original.next_batch()
    resumed = sampler()
    resumed.load_state_dict(original.state_dict())
    for _ in range(143):
        expected, actual = original.next_batch(), resumed.next_batch()
        assert expected == actual
        assert tuple(e for rank in range(4) for e in actual[rank::4]) == expected
        assert {e.position: e for rank in range(2) for e in actual[rank::2]} == {e.position: e for e in expected}
        for event in actual:
            assert len(event.queries28) == 28 and len({demo for demo, _ in event.queries28}) == 7
            assert event.teacher_demo not in {demo for demo, _ in event.queries28}
            for index, (demo, frame) in enumerate(event.queries28):
                length = len(data.rows[event.task_id][demo])
                interval = index % 4
                assert length * interval // 4 <= frame < length * (interval + 1) // 4
    assert resumed.state_dict() == original.state_dict()


def test_new_pool_only_and_missing_phase2_layers_retain_task_weight():
    manifest = pool(phase=2, recursive_tasks=TASKS36[::3])
    # A task without any MT layer is valid: its actual available layer owns20.
    for condition in manifest['conditions']:
        if condition['task_id'] == TASKS36[0]:
            condition['events'] = [e for e in condition['events'] if e['incoming'] != 'MT']
    stream = sampler(manifest, phase=2)
    events = [e for _ in range(180) for e in stream.next_batch()]
    assert {e.pool for e in events} == {'pool_phase2'}
    assert all(e.update >= 181 and 'phase2' in e.record_path for e in events)
    assert Counter(e.task_id for e in events) == Counter({task: 20 for task in TASKS36})
    assert stream.coverage()['per_task'][TASKS36[0]] == dict(MT=0, nonMT=20)


def test_actual_mt_coordinate_layer_does_not_replace_the_recorded_incoming_file():
    manifest = pool(phase=2, recursive_tasks=TASKS36[:1])
    for condition in manifest['conditions']:
        for event in condition['events']:
            event['incoming_is_MT'] = True
    stream = sampler(manifest, phase=2)
    events = [e for _ in range(180) for e in stream.next_batch()]
    assert stream.coverage()['nonMT_presentations'] == 0
    assert any(e.incoming.startswith('incoming_') for e in events)
    assert all(counts == dict(MT=20, nonMT=0) for counts in stream.coverage()['per_task'].values())


def test_resume_rejects_changed_actual_condition_plan():
    original = sampler()
    original.next_batch()
    altered = pool()
    altered['conditions'][0]['record_path'] = '/another/experience'
    with pytest.raises(ValueError, match='plan changed'):
        sampler(altered).load_state_dict(original.state_dict())


def test_actual_bootstrap_metadata_has_registered160_of720_coverage():
    root = Path('/data1/user/ymdai/ember_runs/parameter_conditioned_compiler_20261009/pools')
    if not root.exists():
        pytest.skip('canonical historical bootstrap is not installed')
    rows = [c for name in ('pool0', 'refresh180') for c in read_json(root / name / 'manifest.json')['conditions']]
    stream = sampler(dict(complete=True, conditions=rows))
    for _ in range(180):
        stream.next_batch()
    coverage = stream.coverage()
    assert (coverage['actual_conditions'], coverage['actual_endpoints'], coverage['nonMT_presentations']) == (216, 362, 160)
    assert len(coverage['nonMT_tasks']) == 16


def test_prepare_consumes_actual_incoming_full_history_and_success_noise(tmp_path):
    records = []
    observations = {f'o{i}': dict(images=torch.zeros(2, 3, 2, 2, dtype=torch.uint8),
                                proprio=torch.arange(8).float()) for i in range(3)}
    for index in range(2):
        records.append(dict(pre=f'o{index}', post=f'o{index + 1}', episode=index, step=0,
            hidden=torch.full((50, 1024), float(index)), proprio=torch.arange(8).float(),
            actions=torch.zeros(5, 7), executed=torch.ones(5, dtype=torch.bool),
            feedback=torch.zeros(4), noise_seed=71 + index))
    chain = Chain(records=records, observations=observations, endpoints=[1, 2],
        episodes=[dict(success=False, policy_noise_seeds=[71]), dict(success=True, policy_noise_seeds=[72])],
        behavior_versions=['MT', 'actual_phi'])
    torch.save(chain.to_record(), tmp_path / 'experience.pt')
    save_file({'A': torch.tensor([2.]), 'B': torch.tensor([3.])}, str(tmp_path / 'incoming_001.safetensors'))
    write_json_atomic(tmp_path / 'record.json', dict(task_id=TASKS36[0], teacher_demo=3, language='exact language',
        events=[dict(endpoint=1, incoming='MT', behavior_version='MT'),
                dict(endpoint=2, incoming='incoming_001.safetensors', behavior_version='actual_phi')]))
    runtime = SimpleNamespace(mt={'A': torch.ones(1, requires_grad=True), 'B': torch.ones(1, requires_grad=True)},
        device=torch.device('cpu'), processor=lambda value: {'input': value['observation.state'][None]},
        observation_features=lambda values: {key: torch.ones(2, 4) for key in values},
        teacher=lambda task, demo: {}, last_teacher_cost={})
    data = SimpleNamespace(query_batch=lambda event, processor: {'marker': event.queries28})
    event = Event(1, 0, TASKS36[0], 3, tuple((4, 0) for _ in range(28)), False, 11,
        'real', 'refresh', 2, 'incoming_001.safetensors', str(tmp_path), 'actual_phi')
    item = training.prepare_events(runtime, data, [event])[0]
    assert item['incoming']['A'].item() == 2 and not item['incoming']['A'].requires_grad
    assert len(item['experience']['hidden']) == 2 and item['support']['indices'].tolist() == [0, 1]
    assert item['support']['noise_seeds'] == [71, 72]
    assert item['keep_support']['indices'].tolist() == [1] and item['keep_support']['noise_seeds'] == [72]
    assert all(not value.requires_grad for value in item['experience'].values())
    with pytest.raises(ValueError, match='authority changed'):
        training.prepare_events(runtime, data, [replace(event, incoming='MT')])


def runtime_model():
    runtime = SimpleNamespace(compiler=torch.nn.Linear(1, 1, bias=False), device=torch.device('cpu'),
        source=dict(optimizer_step=1000, frozen_policy_subdir='source', source_training_commit='frozen'),
        lora=SimpleNamespace(to_dict=lambda: dict(rank=128, targets=38)))
    runtime.compiler.weight.data.fill_(.2)
    runtime.load_checkpoint = lambda path: runtime.compiler.load_state_dict(load_file(str(Path(path) / 'ecp.safetensors')))
    return runtime


def advance(runtime, optimizer, scheduler, stream, count):
    for _ in range(count):
        stream.next_batch()
        optimizer.zero_grad(set_to_none=True)
        runtime.compiler(torch.tensor([[.3]])).square().sum().backward()
        finish_update(runtime, optimizer, scheduler)


def test_complete_checkpoint_restores_adam_scheduler_rng_sampler_and_next_updates(tmp_path):
    context = DistributedContext(0, 0, 1, torch.device('cpu'))
    contract, topology = dict(schema_version='formal_v1', seed=20261010), training._topology(context)
    original, original_stream = runtime_model(), sampler()
    optimizer, scheduler = fresh_optimizer(original.compiler, stage='supervised')
    advance(original, optimizer, scheduler, original_stream, 90)
    path = training.save_checkpoint(original, optimizer, scheduler, original_stream, context,
        root=tmp_path, contract=contract, topology=topology, code_git={'commit': 'trained'}, previous_phases=[])
    assert {p.name for p in path.iterdir()} == {'ecp.safetensors', 'state.pt', 'manifest.json'}
    state = torch.load(path / 'state.pt', weights_only=False)
    assert 'model' not in state and state['scaler'] is None and state['macro_update'] == 90
    expected_rng = (random.random(), np.random.random(), torch.rand(3))
    restored, restored_stream = runtime_model(), sampler()
    new_optimizer, new_scheduler = fresh_optimizer(restored.compiler, stage='supervised')
    previous, migration = training.restore_checkpoint(restored, new_optimizer, new_scheduler, restored_stream,
        context, path, contract=contract, topology=topology)
    assert previous == [] and migration['restored_rank_rng'] == [0]
    assert random.random() == expected_rng[0] and np.random.random() == expected_rng[1]
    torch.testing.assert_close(torch.rand(3), expected_rng[2])
    assert new_scheduler.last_epoch == 90 and restored_stream.state_dict() == original_stream.state_dict()
    advance(original, optimizer, scheduler, original_stream, 3)
    advance(restored, new_optimizer, new_scheduler, restored_stream, 3)
    torch.testing.assert_close(restored.compiler.weight, original.compiler.weight, rtol=0, atol=0)
    assert restored_stream.state_dict() == original_stream.state_dict()


def test180_phase_transition_and_world_migration_keep_optimizer_and_logical_queries(tmp_path):
    old_context = DistributedContext(0, 0, 1, torch.device('cpu'))
    original, old_stream = runtime_model(), sampler()
    optimizer, scheduler = fresh_optimizer(original.compiler, stage='supervised')
    advance(original, optimizer, scheduler, old_stream, 180)
    contract = dict(schema_version='formal_v1', seed=20261010)
    path = training.save_checkpoint(original, optimizer, scheduler, old_stream, old_context,
        root=tmp_path, contract=contract, topology=training._topology(old_context), code_git={'commit': '180'}, previous_phases=[])
    new_context = DistributedContext(1, 1, 2, torch.device('cpu'))
    restored, stream = runtime_model(), sampler(phase=2)
    opt, schedule = fresh_optimizer(restored.compiler, stage='supervised')
    previous, migration = training.restore_checkpoint(restored, opt, schedule, stream, new_context, path,
        contract=contract, topology={'world_size': 2, 'ranks': [0, 1]})
    assert previous == [old_stream.state_dict()] and schedule.last_epoch == 180
    assert migration['fresh_rank_rng'] == {1: 20261011} and migration['physical_topology_changed']
    expected = sampler(phase=2)
    assert stream.next_batch() == expected.next_batch() and stream.cursor == 1
    assert random.random() == random.Random(20261011).random()
    assert opt.state and opt.param_groups[0]['lr'] == 3e-5


def _distributed_sum_worker(rank, rendezvous, destination):
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    try:
        runtime = runtime_model()
        optimizer, scheduler = fresh_optimizer(runtime.compiler, stage='supervised')
        x = torch.tensor([.1, .2, .3, .4])[rank::2]
        # Independent global4-event quadratic objective; no local mean.
        (.25 * (runtime.compiler.weight.reshape(()) * x).square().sum()).backward()
        result = finish_update(runtime, optimizer, scheduler, world_size=2)
        torch.save(dict(result=result, gradient=runtime.compiler.weight.grad,
                        moments=optimizer.state[runtime.compiler.weight]), Path(destination) / f'rank{rank}.pt')
    finally:
        dist.destroy_process_group()


def test_global_quarter_weights_are_summed_without_world_division(tmp_path):
    torch.multiprocessing.start_processes(_distributed_sum_worker,
        args=(str(tmp_path / 'gloo'), str(tmp_path)), nprocs=2, join=True, start_method='spawn')
    dense = runtime_model()
    objective = (dense.compiler.weight.reshape(()) * torch.tensor([.1, .2, .3, .4])).square().mean()
    expected = torch.autograd.grad(objective, dense.compiler.weight)[0]
    for rank in range(2):
        result = torch.load(tmp_path / f'rank{rank}.pt', weights_only=False)
        torch.testing.assert_close(result['gradient'], expected)
        assert result['result']['combined_gradient_norm'] == pytest.approx(float(expected.norm()))
        torch.testing.assert_close(result['moments']['exp_avg'], .1 * expected)
        torch.testing.assert_close(result['moments']['exp_avg_sq'], .001 * expected.square())


def test_profile_reduces_real_gradient_without_optimizer_step_or_rng_consumption(tmp_path, monkeypatch):
    context = DistributedContext(0, 0, 1, torch.device('cpu'))
    runtime = runtime_model()
    before = runtime.compiler.weight.detach().clone()
    rng = training._rank_rng(context)
    def backward(runtime, data, events, args, context):
        assert len(events) == 4
        runtime.compiler(torch.tensor([[.3]])).square().sum().backward()
        return dict(rank=0)
    monkeypatch.setattr(training, '_batch_backward', backward)
    result = training._profile(runtime, None, sampler(),
        SimpleNamespace(resume_checkpoint=None), context, tmp_path)
    assert result['optimizer_updates'] == 0 and len(result['logical_events']) == 4
    assert result['ranks'][0]['combined_gradient_norm'] > 0
    torch.testing.assert_close(runtime.compiler.weight, before, rtol=0, atol=0)
    assert runtime.compiler.weight.grad is None
    torch.testing.assert_close(training._rank_rng(context)['torch_cpu'], rng['torch_cpu'])
