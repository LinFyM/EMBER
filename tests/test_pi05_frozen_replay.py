"""Frozen trajectory capture cannot change the evaluated policy or become selection."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.preparation import _explicit_diagnostic_states, _frozen_replay_capture


def _fixture(tmp_path):
    reference = tmp_path / 'reference'
    reference.mkdir()
    normalization = tmp_path / 'normalization.json'
    normalization.write_text('{"stats": {"action": [1, 2]}}')
    contract = {
        'schema_version': 'ember_pi05_target_eval_launch_v2',
        'content_hash_policy': 'disabled_by_owner',
        'contract_reference': 'ember_pi05_target_eval_launch_v2:reference',
        'output_dir': str(reference), 'role': 'validation', 'mode': 'formal', 'arm': 'correct',
        'model': {'checkpoint': '/source'}, 'environment': {'render_resolution': 256},
        'policy': {'num_inference_steps': 10}, 'rng': {'inference_seed': 7},
        'normalization': {'path': str(normalization)}, 'tokenizer': {'path': '/tokenizer', 'bytes': 100},
        'adapter': {'manifest': {'path': '/correct200/manifest.json', 'bytes': 123}},
        'tasks': [{'suite': 'libero_spatial', 'task_id': 1, 'init_state_ids': [0, 12, 25, 37]}],
    }
    (reference / 'run_contract.json').write_text(json.dumps(contract))
    (reference / 'launcher_completion.json').write_text(json.dumps({'return_codes': {'6-r0': 0}}))
    (reference / 'results.json').write_text(json.dumps({
        'contract_reference': contract['contract_reference'],
        'rows': [{'suite': 'libero_spatial', 'task_id': 1, 'init_state_id': state} for state in [0, 12, 25, 37]],
    }))
    registration = {
        'schema_version': 'ember_pi05_frozen_replay_registration_v1', 'role': 'validation',
        'reference_output': str(reference), 'init_state_ids': [0, 12, 25, 37],
        'training_gradient_use': False, 'checkpoint_selection_use': False,
        'test_use': False, 'outcome_dependent_selection': False,
    }
    path = tmp_path / 'registration.json'
    path.write_text(json.dumps(registration))
    args = SimpleNamespace(frozen_replay_registration=path, role='validation', mode='screen',
        state_count=4, init_state_ids=(0, 12, 25, 37), static_task_lora_manifest='/correct200/manifest.json')
    return args, contract, registration


def test_capture_reuses_complete_reference_with_explicit_nonselecting_scope(tmp_path):
    args, contract, _ = _fixture(tmp_path)
    assert _explicit_diagnostic_states(args) == (0, 12, 25, 37)
    capture, stage = _frozen_replay_capture(args, contract, tmp_path / 'replay')
    assert capture['training_gradient_use'] is False
    assert capture['checkpoint_selection_use'] is False
    assert capture['validation_use'] is True
    assert stage['predicate_source'] == 'installed_LIBERO_BDDL_goal_conjunction'
    assert capture['trajectory_root'] == str(tmp_path / 'replay/trajectories')


@pytest.mark.parametrize('change', ['model', 'adapter', 'noise', 'cases', 'reference_failure'])
def test_replay_rejects_changed_policy_pairing_or_failed_reference(tmp_path, change):
    args, original, _ = _fixture(tmp_path)
    contract = deepcopy(original)
    if change == 'model':
        contract['model']['checkpoint'] = '/different_source'
    elif change == 'adapter':
        contract['adapter']['manifest']['path'] = '/different_writer/manifest.json'
    elif change == 'noise':
        contract['rng']['inference_seed'] = 8
    elif change == 'cases':
        contract['tasks'][0]['init_state_ids'] = [49]
    else:
        (tmp_path / 'reference/launcher_completion.json').write_text(json.dumps({'return_codes': {'6-r0': 1}}))
    with pytest.raises(Pi05EvaluationError, match='completed reference'):
        _frozen_replay_capture(args, contract, tmp_path / 'replay')


@pytest.mark.parametrize('change', ['test', 'formal', 'selection', 'gradient', 'exploration'])
def test_capture_cannot_expand_into_test_training_or_checkpoint_selection(tmp_path, change):
    args, _, registration = _fixture(tmp_path)
    if change in {'test', 'formal'}:
        setattr(args, 'role' if change == 'test' else 'mode', change)
    elif change == 'exploration':
        args.exploration_sigma = True
    else:
        registration['checkpoint_selection_use' if change == 'selection' else 'training_gradient_use'] = True
        args.frozen_replay_registration.write_text(json.dumps(registration))
    with pytest.raises(Pi05EvaluationError, match='registered non-selecting'):
        _explicit_diagnostic_states(args)


def test_launcher_reinspection_validates_original_bank_without_expanding_replay_cases(tmp_path, monkeypatch):
    from ember.pi05_eval import recovery

    args, contract, _ = _fixture(tmp_path)
    contract['adapter']['kind'] = 'horizon_writer_lora_bank'
    reference_path = tmp_path / 'reference/run_contract.json'
    reference = json.loads(reference_path.read_text())
    reference['adapter'] = contract['adapter']
    reference['tasks'][0]['init_state_ids'] = list(range(50))
    reference_path.write_text(json.dumps(reference))
    contract['mode'] = 'screen'
    contract['output_dir'] = str(tmp_path / 'replay')
    capture, stage = _frozen_replay_capture(args, contract, tmp_path / 'replay')
    contract['diagnostic_occupancy_capture'], contract['diagnostic_stage_predicates'] = capture, stage
    observed = []
    def inspect(**kwargs):
        observed.extend(kwargs['tasks'][0].init_state_ids)
        return contract['adapter']
    monkeypatch.setattr(recovery, 'inspect_static_task_lora_adapter', inspect)
    assert recovery._reinspect_adapter(contract['adapter'], contract=contract, model=contract['model']) == contract['adapter']
    assert observed == list(range(50))
    assert contract['tasks'][0]['init_state_ids'] == [0, 12, 25, 37]
