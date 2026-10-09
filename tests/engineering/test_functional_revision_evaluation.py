"""Registered identities, actual paired sets and CPU execution/recovery consumer."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.experience_compiler import evaluation as ev, interaction
from ember.experience_compiler.contract import SCHEMA
from ember.experience_compiler.interaction import Chain
from ember.experience_compiler.storage import RecordWriter
from ember.pi05_eval_contract import policy_noise_seed
from ember.pi05_eval_queue import claim_next, fail_job, queue_summary
from ember.pi05_source_checkpoint import read_json, write_json_atomic


def prepared(tmp_path, kind='train180'):
    checkpoint = tmp_path / 'training/checkpoints' / f'step_{ev.KINDS[kind]:08d}'
    write_json_atomic(checkpoint / 'manifest.json', dict(schema_version=SCHEMA, macro_update=ev.KINDS[kind]))
    result = ev.prepare_evaluation(tmp_path, checkpoint, kind)
    return read_json(Path(result['contract'])), Path(result['output']), checkpoint


@pytest.mark.parametrize('kind', ['train180', 'train360'])
def test_train_panel_inherits_old_explicit_conditions_and_both_scene_rng_contracts(tmp_path, kind):
    contract, output, checkpoint = prepared(tmp_path, kind)
    original = read_json(ev.PARENT / 'evaluation/train180/evaluation_contract.json')
    for key in ['conditions', 'environment_contract', 'adaptation_environment_contract', 'policy_seed_root']:
        assert contract[key] == original[key]
    assert contract['conditions'][0]['teacher_demo'] == 43
    assert contract['conditions'][0]['seed'] == 6924217510202537008
    assert contract['arms'] == ['end'] and contract['MT_reuse'].endswith('train180/results.json')
    assert contract['no_evaluation_gradients'] and contract['final_actor_only'] and not contract['Test_opened']
    assert not contract['adaptation_environment_contract'].get('operator_read_write_scene')
    assert queue_summary(output / 'queue.sqlite3')['status_counts'] == {'pending': 16}
    assert ev.prepare_evaluation(tmp_path, checkpoint, kind)['output'] == str(output)
    with pytest.raises(ValueError, match='only train180/train360'):
        ev.prepare_evaluation(tmp_path, checkpoint, 'formal180')


def test_formal400_preserves_old_seed_and_full50_teacher_state_schedule(tmp_path):
    contract, output, _ = prepared(tmp_path, 'formal360')
    original = read_json(ev.PARENT / 'evaluation/formal180/evaluation_contract.json')
    assert contract['conditions'] == original['conditions']
    assert contract['conditions'][0]['seed'] == 2013708424421913872
    for task in {c['task_id'] for c in contract['conditions']}:
        rows = [c for c in contract['conditions'] if c['task_id'] == task]
        assert {c['teacher_demo'] for c in rows} == set(range(50))
        assert {c['init_state_id'] for c in rows} == set(range(50))
    claim = claim_next(output / 'queue.sqlite3', worker_id='CPU')
    assert claim.shard.suite == 'libero_10' and claim.shard.preferred_gpu is None
    assert claim.shard.estimated_cost == 1024 + 520
    bad = deepcopy(contract)
    bad['conditions'][0]['teacher_demo'] = bad['conditions'][1]['teacher_demo']
    with pytest.raises(ValueError, match='video_schedule'):
        ev._validate_panel(bad)


def test_original_formal_raw_rows_reproduce_real_retained_gained_lost_sets():
    contract = read_json(ev.PARENT / 'evaluation/formal180/evaluation_contract.json')
    old180 = ev._reference_rows(ev.PARENT / 'evaluation/formal180/results.json')
    mt = ev._reference_rows(contract['references']['MT'])
    comparison = ev.compare_rows(mt, old180)
    assert (comparison['reference_successes'], comparison['candidate_successes']) == (153, 127)
    assert (comparison['retained'], comparison['gained'], comparison['lost']) == (106, 21, 47)
    assert comparison['churn_count'] == 68 and comparison['success_set_jaccard'] == 106 / 174
    assert sum(t['lost'] for t in comparison['per_task']) == 47
    old360 = ev._reference_rows(ev.FULL360 / 'evaluation/formal360/results.json')
    comparison = ev.compare_rows(mt, old360)
    assert (comparison['candidate_successes'], comparison['retained'], comparison['gained'], comparison['lost']) == (130, 104, 26, 49)
    for name, score, retained, gained, lost in [('T2340', 161, 98, 29, 63), ('experience135', 135, 97, 30, 38)]:
        reference = ev._reference_rows(contract['references'][name])
        comparison = ev.compare_rows(reference, old180)
        assert (comparison['reference_successes'], comparison['retained'], comparison['gained'], comparison['lost']) == (score, retained, gained, lost)
    mismatched = deepcopy(old180)
    mismatched[0]['scene_reference']['path'] += '.changed'
    with pytest.raises(ValueError, match='actual scene'):
        ev.compare_rows(mt, mismatched)


def test_train_pairing_preserves_two_video_conditions_and_actual_success_keys():
    mt = ev._reference_rows(ev.PARENT / 'evaluation/train180/results.json', 'MT')
    assert len(mt) == 48 and sum(r['success'] for r in mt) == 27
    assert len({(r['suite'], r['task_id'], r['init_state_id']) for r in mt}) == 24
    candidate = deepcopy(mt)
    lost = next(r for r in candidate if r['success'])
    gained = next(r for r in candidate if not r['success'])
    lost['success'], gained['success'] = False, True
    result = ev.compare_rows(mt, candidate)
    assert (result['rows'], result['retained'], result['gained'], result['lost']) == (48, 26, 1, 1)
    assert result['success_set_jaccard'] == 26 / 28 and result['churn_fraction'] == 2 / 48
    assert result['lost_keys'][0][2] == lost['condition_id']
    candidate[0]['policy_noise_seeds'][0] += 1
    with pytest.raises(ValueError, match='RNG'):
        ev.compare_rows(mt, candidate)


def chain(condition):
    metrics = dict(actual_J=1, initial_reads=1, rereads=0, read_frames=5, environment_steps=15,
        resets=1, failure_environment_steps=0, tail_environment_steps=0, edit_seconds=.1,
        wall_seconds=.2, practice_success=True, stop_reason='own_success')
    incoming = {'factor': torch.ones(2)}
    outgoing = {'factor': torch.full((2,), condition['teacher_demo'] + 2.)}
    return Chain(states=[incoming, outgoing], records=[dict(episode=0)], endpoints=[1],
        episodes=[dict(init_state_id=1 if 0 in condition['final_state_ids'] else 0, success=True)],
        metrics=metrics, behavior_versions=['MT'])


def final_row(request):
    task, state = request['task'], request['state_id']
    root, scene = request['noise_root'], request['environment_contract']['operator_read_write_scene']['root']
    return dict(suite=task['suite'], task_id=task['task_id'], language=task['language'],
        split_role=task['split_role'], init_state_id=state, env_seed=root, policy_seed_root=root,
        policy_noise_seeds=[policy_noise_seed(root, task['suite'], task['task_id'], state, 0)],
        scene_reference=dict(path=str(Path(scene) / f"{task['suite']}_task_{task['task_id']:02d}_state_{state:03d}.npz"),
            bytes=1, restoration='CPU_fixture'), success=bool(state % 2), environment_steps=15, steps=5)


class Runtime:
    def __init__(self):
        self.io, self.loaded = RecordWriter(), []
        self.compiler = SimpleNamespace(eval=lambda: None)

    def load_checkpoint(self, checkpoint):
        self.loaded.append(checkpoint)


class Runner:
    instances = []

    def __init__(self, runtime, contract, gpu, *, slot_batch):
        self.contract, self.adaptations, self.finals = contract, [], []
        self.total_environment_steps, self.components = 0, dict(physical_batch_histogram={})
        self.instances.append(self)

    def run(self, provider):
        while (request := provider()) is not None:
            self.total_environment_steps += 15
            if request['kind'] == 'adapt':
                self.adaptations.append(request)
                assert request['step_budget'] == 1024 and request['behavior_version'].endswith('00000180')
                yield dict(request=request, chain=chain(request['condition']))
            else:
                self.finals.append(request)
                assert 'teacher' not in request and 'experience' not in request and request['kind'] == 'final'
                yield dict(request=request, row=final_row(request))

    def close(self):
        pass

    def preserve_partial(self, destination):
        write_json_atomic(Path(destination) / 'partial.json', dict(complete=False))


def test_frozen_cpu_consumer_saves_actual_endpoints_and_reuses_completed_conditions(tmp_path, monkeypatch):
    contract, output, checkpoint = prepared(tmp_path)
    args = SimpleNamespace(run_root=tmp_path, evaluation='train180', checkpoint=checkpoint,
        physical_gpu=0, slot_batch=16, claim_batch=2, worker_id='CPU', code_git='CPU_fixture')
    monkeypatch.setattr(interaction, 'Runner', Runner)
    runtime = Runtime()
    try:
        receipt = ev.run_evaluation(runtime, args)
        runtime.io.flush()
        consumer = Runner.instances[-1]
        assert receipt['complete'] and len(consumer.adaptations) == 16 and len(consumer.finals) == 48
        assert runtime.loaded == [checkpoint]
        records = list((output / 'conditions').glob('*/*/record.json'))
        assert len(records) == 16
        for path in records:
            record = read_json(path)
            assert record['complete'] and record['events'][0]['incoming'] == 'MT'
            assert set(record['practice_state_ids']).isdisjoint(record['final_state_ids'])
            assert (path.parent / 'end.safetensors').is_file() and (path.parent / 'experience.pt').is_file()
        receipt = ev.run_evaluation(runtime, args)
        consumer = Runner.instances[-1]
        assert receipt['complete'] and not consumer.adaptations and not consumer.finals
        assert len(list((output / 'conditions').glob('*/*/record.json'))) == 16
        # Pair the synthetic CPU outcomes only against equally explicit CPU scenes.
        raw = ev._reference_rows(contract['references']['MT'], 'MT')
        for row in raw:
            row['scene_reference'].update(bytes=1, restoration='CPU_fixture')
        monkeypatch.setattr(ev, '_reference_rows', lambda path, arm='end': raw)
        result = ev.aggregate_evaluation(tmp_path, 'train180')
        assert result['aggregated_complete'] and result['arms']['end']['row_count'] == 48
        assert result['cost']['practice_successes'] == 16 and result['cost']['condition_cost_totals']['actual_J'] == 16
        assert read_json(output / 'complete.json')['rows'] == 48
        assert ev.aggregate_evaluation(tmp_path, 'train180') == result
    finally:
        runtime.io.close()


def test_recovery_reuses_saved_adaptation_and_only_missing_final_rows(tmp_path):
    contract, _, _ = prepared(tmp_path)
    condition = contract['conditions'][0]
    output = tmp_path / 'consumer_recovery'
    ev.initialize_condition_queue(output / 'queue.sqlite3', [condition], contract['environment_contract'],
                                  contract_reference='CPU_fixture')
    runtime = Runtime()
    try:
        cases = ev._Cases(runtime, contract, output, 'failed', 1, 'CPU_fixture')
        request = cases.next_request()
        cases.adapted(dict(request=request, chain=chain(condition)))
        runtime.io.flush()
        first = cases.next_request()
        cases.final(dict(request=first, row=final_row(first)))
        case = cases.cases[condition['condition_id']]
        original_row = (case['destination'] / 'finals/final_end_32.json').read_bytes()
        fail_job(output / 'queue.sqlite3', job_id=condition['condition_id'], worker_id='failed',
                 claim_token=case['claim'].claim_token, error='CPU_fixture_failure')
        ev.initialize_condition_queue(output / 'queue.sqlite3', [condition], contract['environment_contract'],
                                      contract_reference='CPU_fixture', retry_failed=True)
        resumed = ev._Cases(runtime, contract, output, 'resumed', 1, 'CPU_fixture')
        requests = [resumed.next_request(), resumed.next_request()]
        assert [r['state_id'] for r in requests] == [33, 34]
        assert all(r['kind'] == 'final' for r in requests)
        for request in requests:
            resumed.final(dict(request=request, row=final_row(request)))
        resumed.publish_ready(flush=True)
        assert queue_summary(output / 'queue.sqlite3')['status_counts'] == {'complete': 1}
        assert len(list((output / 'conditions').glob('*/*/record.json'))) == 1
        assert (case['destination'] / 'finals/final_end_32.json').read_bytes() == original_row
        assert runtime.loaded == []  # Recovery reads end factors, never a new shared update.
    finally:
        runtime.io.close()


def test_worker_rejects_other_checkpoint_node_before_loading_model(tmp_path):
    _, _, checkpoint = prepared(tmp_path, 'train360')
    write_json_atomic(checkpoint / 'manifest.json', dict(schema_version=SCHEMA, macro_update=180))
    runtime = Runtime()
    try:
        with pytest.raises(ValueError, match='frozen shared checkpoint'):
            ev.run_evaluation(runtime, SimpleNamespace(run_root=tmp_path, evaluation='train360', checkpoint=checkpoint))
        assert not runtime.loaded
    finally:
        runtime.io.close()


def test_worker_failure_keeps_raw_negative_row_cost_and_partial_then_resumes(tmp_path, monkeypatch):
    _, output, checkpoint = prepared(tmp_path)
    args = SimpleNamespace(run_root=tmp_path, evaluation='train180', checkpoint=checkpoint,
        physical_gpu=0, slot_batch=16, claim_batch=2, worker_id='CPU', code_git='CPU_fixture')
    class Interrupted(Runner):
        def run(self, provider):
            for result in super().run(provider):
                yield result
                if 'row' in result:
                    raise RuntimeError('CPU_fixture_interruption')
    monkeypatch.setattr(interaction, 'Runner', Interrupted)
    runtime = Runtime()
    try:
        with pytest.raises(RuntimeError, match='interruption'):
            ev.run_evaluation(runtime, args)
        runtime.io.flush()
        rows = list((output / 'conditions').glob('*/*/finals/*.json'))
        assert len(rows) == 1 and read_json(rows[0])['success'] is False
        negative_bytes = rows[0].read_bytes()
        assert list((output / 'failures').glob('*/partial.json'))
        receipt = read_json(next((output / 'workers').glob('*.json')))
        assert not receipt['complete'] and receipt['environment_steps'] == 30
        assert queue_summary(output / 'queue.sqlite3')['status_counts'] == {'failed': 1, 'pending': 15}
        ev.prepare_evaluation(tmp_path, checkpoint, 'train180', retry_failed=True)
        monkeypatch.setattr(interaction, 'Runner', Runner)
        ev.run_evaluation(runtime, args)
        assert len(Runner.instances[-1].adaptations) == 15 and len(Runner.instances[-1].finals) == 47
        assert rows[0].read_bytes() == negative_bytes
        assert queue_summary(output / 'queue.sqlite3')['status_counts'] == {'complete': 16}
    finally:
        runtime.io.close()
