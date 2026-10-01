"""Actual797 source identity plus a complete-boundary fixture; no GPU launch."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from ember.operator_writer import joint_training as study, run, specification as specs
from ember.pi05_source_checkpoint import read_json, write_json_atomic

ROOT = Path('/data1/user/ymdai/ember_runs/conditional_read_write_fresh_20261001')
ORIGIN_RUN = ROOT / study.CONDITIONAL_MODE / 'train/attempts/fresh/run_contract.json'


def actual_parent():
    value = read_json(ORIGIN_RUN)
    assert value['git']['commit'] == study.CONDITIONAL_ORIGIN_GIT
    assert value['spec'] == str(study.CONDITIONAL_ORIGIN_SPEC)
    return value


def migrated(parent, checkpoint, spec_path):
    return {**deepcopy(parent), 'git': {'commit': 'new-engineering-fixture', 'branch': '',
            'dirty_paths': [], 'pushed_ref': 'origin/main'}, 'spec': str(spec_path),
            'frame_chunk': 32, 'topology': {**parent['topology'], 'world_size': 5},
            'parent_checkpoint': str(checkpoint)}


def test_actual797_source_scope_and_recorded_new_engineering_identity(tmp_path):
    parent = actual_parent()
    spec = specs.specification(specs.CONDITIONAL_SPEC_PATH)
    checkpoint = tmp_path / 'fresh/checkpoints/macro_00000090'
    checkpoint.parent.parent.mkdir(parents=True)
    write_json_atomic(checkpoint.parent.parent / 'run_contract.json', parent)
    current = migrated(parent, checkpoint, specs.CONDITIONAL_SPEC_PATH)
    study.register_conditional_resume(spec, SimpleNamespace(resume=checkpoint), current)
    record = current['source_resume']
    assert record['origin_training_git'] == record['parent_training_git']['commit'] == study.CONDITIONAL_ORIGIN_GIT
    assert record['parent_training_spec'] == str(study.CONDITIONAL_ORIGIN_SPEC)
    assert record['current_training_git']['commit'] == 'new-engineering-fixture'
    assert record['current_training_spec'] == str(specs.CONDITIONAL_SPEC_PATH)
    assert run.resume_contract_compatible(parent, current, allow_topology_change=True)
    assert not run.resume_contract_compatible(parent, current, allow_topology_change=False)
    bad = deepcopy(current); bad['optimizer']['lr'] *= 2
    assert not run.resume_contract_compatible(parent, bad, allow_topology_change=True)
    bad = deepcopy(current); bad['source_resume']['parent_training_git']['commit'] = 'retagged'
    assert not run.resume_contract_compatible(parent, bad, allow_topology_change=True)
    bad_spec = tmp_path / 'changed_spec.json'
    changed = deepcopy(spec); changed['operator']['frame_stride'] = 10
    write_json_atomic(bad_spec, changed)
    bad = deepcopy(current); bad['spec'] = str(bad_spec)
    assert not run.resume_contract_compatible(parent, bad, allow_topology_change=True)
    foreign = deepcopy(parent); foreign['mode'] = study.SELF_READ_MODE
    assert not run.resume_contract_compatible(foreign, current, allow_topology_change=True)
    assert parent == actual_parent()  # source registration never mutates the parent original


@pytest.mark.parametrize('frame_chunk', [4, 8, 16, 32])
def test_current_conditional_packing_and_no_scope_expansion(frame_chunk):
    spec = specs.specification(specs.CONDITIONAL_SPEC_PATH)
    args = SimpleNamespace(mode=study.CONDITIONAL_MODE, attempt='resume90_world5', resume=Path('/data1/boundary'),
                           pilot_arm=None, microbatch=28, frame_chunk=frame_chunk, stop_after_macro=None)
    study.validate_request(spec, args)
    parent = actual_parent()
    assert run.packing_compatible(parent, {**parent, 'frame_chunk': frame_chunk})
    historical = specs.specification(specs.SELF_READ_SPEC_PATH)
    args.mode = study.SELF_READ_MODE
    if frame_chunk > 8:
        with pytest.raises(ValueError):
            study.validate_request(historical, args)
        old = {**parent, 'mode': study.SELF_READ_MODE}
        assert not run.packing_compatible(old, {**old, 'frame_chunk': frame_chunk})
    else:
        study.validate_request(historical, args)


def endpoint(checkpoint, contract, rows):
    checkpoint.mkdir(parents=True)
    macro = int(checkpoint.name.removeprefix('macro_'))
    torch.save({'stage': run.STAGE, 'next_macro': macro, 'metrics_rows': macro,
                'scheduler': {'last_epoch': macro}, 'optimizer': {'param_groups': [{}]}, 'scaler': None,
                'training_state': {'updates': macro, 'mode': study.CONDITIONAL_MODE, 'loss_variant': 'full'},
                'sampler_state': {**contract['sampler'], 'next_step': macro}}, checkpoint / 'trainer_state.pt')
    save_file({'fixture': torch.ones(1)}, str(checkpoint / 'ecp.safetensors'))
    world = contract['topology']['world_size']
    names = ['ecp.safetensors', 'trainer_state.pt']
    for rank in range(world):
        name = f'rank_{rank:02d}_state.pt'; names.append(name)
        torch.save({'fixture': True}, checkpoint / name)
    write_json_atomic(checkpoint / 'checkpoint_manifest.json', {
        'stage': run.STAGE, 'run_contract_schema': run.SCHEMA, 'next_macro': macro, 'world_size': world,
        'files': {name: {'bytes': (checkpoint / name).stat().st_size} for name in names}})
    write_json_atomic(checkpoint.parent.parent / 'run_contract.json', contract)
    (checkpoint.parent.parent / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_actual_parent90_to_world5_complete450_source_readout_fixture(tmp_path, monkeypatch):
    """Real source/header contracts; synthetic states only test checkpoint/readout plumbing."""
    parent = actual_parent()
    spec = specs.specification(specs.CONDITIONAL_SPEC_PATH)
    monkeypatch.setattr(study, 'CONDITIONAL_ROOT', tmp_path)
    attempts = tmp_path / study.CONDITIONAL_MODE / 'train/attempts'
    parent_cp = attempts / 'fresh/checkpoints/macro_00000090'
    final_cp = attempts / 'resume90_world5/checkpoints/macro_00000450'
    rows = [{'update': i, 'mode': study.CONDITIONAL_MODE, 'queries': 112, 'loss_variant': 'full',
             'jobs': [{'loss_variant': 'full'}] * 4} for i in range(1, 451)]
    endpoint(parent_cp, parent, rows[:90])
    current = migrated(parent, parent_cp, specs.CONDITIONAL_SPEC_PATH)
    args = SimpleNamespace(mode=study.CONDITIONAL_MODE, attempt='resume90_world5', resume=parent_cp,
                           pilot_arm=None, microbatch=28, frame_chunk=32, stop_after_macro=None)
    study.register_conditional_resume(spec, args, current)
    study.validate_attempt(spec, args, current, final_cp.parent.parent)
    endpoint(final_cp, current, rows)
    assert run.complete_checkpoint(parent_cp) and run.complete_checkpoint(final_cp)
    write_json_atomic(final_cp.parent.parent / 'completion.json', {'updates': 450, 'checkpoint': str(final_cp)})
    monkeypatch.setattr(run, 'frozen_git', lambda: {'commit': 'independent-reading-fixture'})
    original = study._inspect_frozen_source
    def source_check(contract, candidate_spec):
        if contract['git']['commit'] == 'new-engineering-fixture':
            assert read_json(Path(contract['spec'])) == candidate_spec
        else:
            original(contract, candidate_spec)
    monkeypatch.setattr(study, '_inspect_frozen_source', source_check)
    result = study.inspect_source(spec, final_cp)
    assert result['git']['commit'] == 'new-engineering-fixture'
    assert result['source_resume']['parent_training_git']['commit'] == study.CONDITIONAL_ORIGIN_GIT
    copied = (final_cp.parent.parent / 'metrics.jsonl').read_text().splitlines()
    copied[0] = copied[0].replace('"update": 1', '"update": 0')
    (final_cp.parent.parent / 'metrics.jsonl').write_text('\n'.join(copied) + '\n')
    with pytest.raises(ValueError, match='copied event history'):
        study.inspect_source(spec, final_cp)
