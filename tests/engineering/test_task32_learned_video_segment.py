"""Frozen selection and real consumer metadata checks, without policy/environment."""
import importlib.util
from collections import OrderedDict
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from ember.lora import LoRATarget
from ember.operator_writer.bank import FrozenOperatorAdapter, validate_episode
from ember.pi05_eval.trajectory_capture import capture_level
from ember.pi05_source_checkpoint import read_json
from ember.writer.materialization import file_record


@pytest.fixture(autouse=True)
def configured_physical_id(monkeypatch):
    # Metadata only; these checks do not initialize CUDA or the environment.
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')


@pytest.fixture(scope='module')
def owner():
    path = Path(__file__).resolve().parents[2] / 'scripts/task32_learned_video_segment.py'
    spec = importlib.util.spec_from_file_location('task32_segments', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_bank_consumer_never_adds_public_B0(tmp_path):
    name = 'model.action_in_proj'
    a, b = [f'{name}.lora_{f}.default.weight' for f in ('A', 'B')]
    path = tmp_path / 'complete.safetensors'
    save_file({a: torch.ones(1, 2), b: torch.full((2, 1), 7.)}, str(path))
    adapter = FrozenOperatorAdapter.__new__(FrozenOperatorAdapter)
    adapter.states = OrderedDict()
    adapter.conditions = {'selected': {'factors': file_record(path)}}
    adapter.bank = {'mode': 'learned_video_segment',
                    'condition_factors': 'complete_A0_plus_S_B0_plus_M'}
    adapter.common = {a: torch.full((1, 2), 3.), b: torch.full((2, 1), 11.)}
    adapter.lora = SimpleNamespace(rank=1, targets=(LoRATarget(name, 2, 2),))
    state = adapter._state('selected')
    assert torch.equal(state[b], torch.full((2, 1), 7.))


def test_actual_case_metadata_pairing_and_four_full(owner):
    banks, contracts, _ = owner.load_originals()
    bank = dict(banks[(17, 'parent')], mode='learned_video_segment')
    bank.pop('learning_limit_panel')
    bank['tasks'] = [next(t for t in bank['tasks'] if t['global_task_id'] == 32)]
    contract, task, cases = owner.build_cases(bank, contracts)
    assert len(cases) == len(set(c['evidence']['case_id'] for c in cases)) == 16
    assert len(set(task['init_state_ids'])) == 4
    assert contract['parallel']['envs_per_replica'] == 16
    full = []
    roots = []
    for case in cases:
        e, per_case = case['evidence'], case['contract']
        assert all(per_case[k] == contract[k] for k in ('policy', 'environment', 'rng', 'operator_read_write_scene'))
        assert validate_episode(per_case['adapter'], case['prepared_adapter'].evidence,
                                suite='libero_10', task_id=2, init_state_id=e['init_state_id'])
        roots.append(per_case['diagnostic_occupancy_capture']['trajectory_root'])
        level = capture_level(per_case['diagnostic_occupancy_capture'], task, e['init_state_id'])
        if level == 'full':
            full.append((e['teacher'], e['arm'], e['init_state_id']))
    assert len(set(roots)) == 16
    assert set(full) == {(43, 'E', 2), (43, 'L', 2), (43, 'E', 3), (43, 'L', 3)}


def test_canonical_rollout_keeps_repeated_init_and_case_contract(owner, monkeypatch, tmp_path):
    import ember.pi05_evaluation as consumer

    banks, contracts, _ = owner.load_originals()
    bank = dict(banks[(17, 'parent')], mode='learned_video_segment')
    contract, task, cases = owner.build_cases(bank, contracts)
    observed = []
    for case in cases:
        case['contract']['output_dir'] = str(tmp_path / case['evidence']['case_id'])

    def start(**kwargs):
        assert kwargs['task_adapter'] is None
        observed.append((kwargs['init_state_id'], kwargs['contract']['output_dir'], kwargs['capture_level']))
        return dict(init_state_id=kwargs['init_state_id'], steps=0, action_plan=deque(), obs={})

    def plan(slots, **kwargs):
        for slot in slots:
            if slot is not None:
                slot['action_plan'].append([0.] * 7)

    def finish(**kwargs):
        slot = kwargs['slot']
        assert kwargs['contract']['output_dir'].endswith(slot['frozen_video_segment_case']['case_id'])
        assert slot['episode_adapter'].key == slot['frozen_video_segment_case']['condition_id']
        return dict(init_state_id=slot['init_state_id'], success=True, steps=slot['steps'])

    monkeypatch.setattr(consumer, 'start_fixed_episode', start)
    monkeypatch.setattr(consumer, '_plan_action_chunks', plan)
    monkeypatch.setattr(consumer, 'record_passive_step', lambda *args: None)
    monkeypatch.setattr(consumer, 'finish_episode_row', finish)
    env = SimpleNamespace(step=lambda action: ({}, 0., True, {}))
    rows = consumer.rollout_shard(envs=[env]*16, init_states=None, task=task,
        state_ids=owner.STATE_IDS, contract=contract, policy=SimpleNamespace(reset=lambda: None),
        preprocess=None, postprocess=None, task_adapter=object(), episode_contexts=cases)
    assert len(rows) == len(observed) == 16
    assert [state for state, _, _ in observed] == list(owner.STATE_IDS)
    assert sum(level == 'full' for _, _, level in observed) == 4
    assert len({r['frozen_video_segment_case']['case_id'] for r in rows}) == 16


def test_delta_arrival_partition_preserves_full_TargetWrite(owner):
    import copy
    from ember.operator_writer.model import TargetWrite
    torch.manual_seed(31)
    parent = TargetWrite(12, 5)
    torch.nn.init.normal_(parent.o.weight, std=.04)
    learned = copy.deepcopy(parent)
    with torch.no_grad():
        learned.c.weight.add_(torch.randn_like(learned.c.weight)*.03)
        learned.o.weight.add_(torch.randn_like(learned.o.weight)*.02)
        x, h, a = torch.randn(5,50,12), torch.randn(5,50,1024), torch.randn(128,12)
        indices = torch.tensor([0,5,10,15,19])
        (mp,ms,e,l),keys=owner.segment_delta(parent,learned,a,x,h,indices,5)
        assert keys.shape == (5,128,50)
        torch.testing.assert_close(mp,parent(a,x,h))
        torch.testing.assert_close(ms,learned(a,x,h))
        torch.testing.assert_close(e+l,ms-mp,rtol=1e-4,atol=1e-6)
        (short_p,short_s,*_),_=owner.segment_delta(parent,learned,a,x[:2],h[:2],indices[:2],5)
        expected=short_s-short_p
        for k in keys[1:-1]:
            expected=expected-(expected@k)@k.T/50
        torch.testing.assert_close(e,expected,rtol=1e-4,atol=1e-6)
        # Complement includes the unsampled tail arrival19; zero cut excludes arrival5.
        (_,_,zero,all_l),_=owner.segment_delta(parent,learned,a,x,h,indices,0)
        assert torch.count_nonzero(zero)==0
        torch.testing.assert_close(all_l,ms-mp,rtol=1e-4,atol=1e-6)
