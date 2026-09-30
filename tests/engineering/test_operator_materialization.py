"""Actual operator compiler/cache and shared dynamic worker dispatch contracts."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from ember.operator_writer import materialization as current
from ember.operator_writer.bank import BANK_SCHEMA
from ember.pi05_source_checkpoint import read_json, write_json_atomic


def test_partial_reader_transition_requires_identical_training_and_explicit_previous_git(tmp_path):
    old = {'training_git': 'training', 'checkpoint': '450', 'spec': {'path': 'training_spec'},
           'source': {'checkpoint': 'source'}, 'materialization_git': {'commit': 'old'}}
    new = old | {'materialization_git': {'commit': 'new'}}
    write_json_atomic(tmp_path / 'materialization_contract.json', old)
    with pytest.raises(ValueError):
        current.register_partial(tmp_path, new, None)
    with pytest.raises(ValueError):
        current.register_partial(tmp_path, new | {'training_git': 'changed'}, 'old')
    current.register_partial(tmp_path, new, 'old')
    assert read_json(tmp_path / 'materialization_resume.json')['previous_contract'] == old
    assert read_json(tmp_path / 'materialization_contract.json') == new


def test_actual_operator_worker_skips_cached_video_and_uses_larger_native_batch(tmp_path, monkeypatch):
    from ember.operator_writer import run
    checkpoint = tmp_path / 'checkpoint'; checkpoint.mkdir()
    save_file({'weight': torch.tensor(2.)}, str(checkpoint / 'ecp.safetensors'))
    shapes = {f'target_{i}.lora_B.default.weight': (1, 1) for i in range(38)}
    conditions = [{'condition_id': f'condition_{i}', 'global_task_id': 3, 'teacher_demo': i} for i in range(3)]
    save_file({name: torch.ones(shape) for name, shape in shapes.items()},
              str(tmp_path / 'condition_0.safetensors'), metadata={
                  'schema_version': BANK_SCHEMA, 'condition_id': 'condition_0', 'mode': 'T_change_clock'})
    builds, videos, chunks = [], [], []

    class Writer(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(0.))

    writer = Writer()
    def compile_video(condition, *, frame_chunk):
        assert not torch.is_grad_enabled() and not writer.weight.requires_grad
        chunks.append(frame_chunk)
        return {name: torch.ones(shape) * writer.weight for name, shape in shapes.items()}, None

    def build(*args):
        builds.append(args)
        return SimpleNamespace(writer=writer, policy=torch.nn.Linear(1, 1), source={'checkpoint': 'source'},
                               device=torch.device('cpu'), compile=compile_video)

    class Data:
        def __init__(self, *_args, **kwargs):
            assert kwargs['query_labels'] is False and kwargs['role'] == 'validation'
            self.videos = SimpleNamespace(frame_counts=lambda *_: (100, 20))
        def condition(self, _runtime, task, demo):
            videos.append((task, demo)); return (), 100, 20
        def close(self):
            pass

    monkeypatch.setattr(run, 'build_runtime', build)
    monkeypatch.setattr(current, 'FormalData', Data)
    current.compile_conditions(tmp_path, {}, 'T_change_clock', checkpoint, {'checkpoint': 'source'},
                               tmp_path, conditions, shapes, devices=(torch.device('cpu'),),
                               frame_chunk=32, task_ids=(3,), role='validation', cpu_threads=1)
    assert len(builds) == 1 and sorted(videos) == [(3, 1), (3, 2)] and chunks == [32, 32]
    assert all(len(load_file(row['factors']['path'])) == 38 for row in conditions)
    stats = read_json(tmp_path / 'materialization_execution.json')['conditions']
    assert sum(row['reused'] for row in stats) == 1 and len(stats) == 3
    assert not list(tmp_path.glob('*.tmp'))
