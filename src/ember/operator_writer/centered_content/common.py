"""Exact source references, finite study lifecycle and compact artifact ownership."""
from pathlib import Path
import gc
import json
import os
import random
import socket
import numpy as np
import torch
from safetensors.torch import load_file
import ember.pi05_evaluation
from ember.pi05_assets import prepare_libero_config
from ember.pi05_eval_contract import git_state, git_state_is_clean_pushed_or_frozen_authority
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record
from ember.writer.topology import bind_current_process_to_cuda_numa
from ..run import build_runtime

ROOT = Path('/data1/user/ymdai/ember_runs/native_role_centered_content_20261004')
OLD = ROOT.parent / 'native_role_binding_compilation_20261004'
SELF = ROOT.parent / 'native_role_self_call_20261004'
ASSET = Path('/data1/user/ymdai/projects/EMBER')
TASKS = (12, 13, 14, 15, 17, 19, 43, 96)
HELD_TEACHERS = (47, 40, 33, 28, 46, 32, 1, 24, 43)
ARMS = ('raw_key', 'centered_rgb')
PARENT = ROOT.parent / 'conditional_read_write_continuation900_20261002/conditional_read_write/train/attempts/continuation/checkpoints/macro_00000900/ecp.safetensors'
LOCATOR = OLD / 'R/checkpoint64/binding.safetensors'
TRAIN_REFERENCE = ROOT.parent / 'operator_seen_task_diagnosis_20260929/attempts/scene_canonical144/T/evaluation/correct144/run_contract.json'
HELD_REFERENCE = ROOT.parent / 'conditional_read_write_continuation900_20261002/conditional_read_write/evaluation/900/correct400/run_contract.json'


def initialize():
    identity = git_state(Path(__file__).resolve().parents[4])
    if identity['branch'] or not git_state_is_clean_pushed_or_frozen_authority(identity):
        raise ValueError('actual consumers require clean pushed detached code')
    assets = ASSET / 'data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6'
    if os.environ.get('EMBER_LIBERO_ASSETS_ROOT') != str(assets):
        raise ValueError('explicit canonical assets routing required before imports')
    prepare_libero_config(ROOT / 'cache/libero_config')
    torch.cuda.set_device(0); torch.set_num_threads(8)
    affinity = bind_current_process_to_cuda_numa(0)
    torch.manual_seed(7); torch.cuda.manual_seed(7); np.random.seed(7); random.seed(7)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    spec = read_json(OLD / 'launch/parent_spec.json')
    runtime = build_runtime(ASSET, spec, torch.device('cuda:0'), 'conditional_read_write', evaluation=True)
    runtime.writer.load_state_dict(load_file(str(PARENT), device='cuda:0'), strict=True)
    runtime.policy.eval(); runtime.writer.train()
    return runtime, spec, identity, affinity


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy']); random.setstate(state['python'])


def compact(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone(memory_format=torch.contiguous_format)
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(compact(v) for v in value)
    return value


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        logical = value.numel() * value.element_size()
        if value.untyped_storage().nbytes() != logical:
            raise ValueError('logical artifact tensor owns unrelated batch storage')
        return logical
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def save_compact(value, path):
    if path.exists():
        raise ValueError(f'valid artifact already exists; no overwrite/recompute: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = compact(value); logical = tensor_bytes(packed)
    torch.save(packed, path); actual = path.stat().st_size
    if actual > logical * 1.05 + 1024 * 1024:
        raise ValueError('serialization inflation: stop further writes, retain actual logical arrays')
    return dict(path=str(path), bytes=actual, logical_tensor_bytes=logical)


def load_fixed(task, teacher, arm, device):
    old = OLD / 'native' / f'task{task:03d}_teacher{teacher:02d}.pt'
    fixed = torch.load(old, map_location='cpu', weights_only=True, mmap=True)
    # K has no training consumer after fixed p generation; never transfer it.
    fixed.pop('K')
    content = torch.load(ROOT / 'content' / f'task{task:03d}_teacher{teacher:02d}.pt',
                         map_location='cpu', weights_only=True, mmap=True)
    fixed['p'] = content[arm]
    def move(value):
        return {k: move(v) for k, v in value.items()} if isinstance(value, dict) else value.to(device)
    return move(fixed)


def load_labels():
    manifest = read_json(OLD / 'labels/manifest.json')
    if manifest['completed'] != 'complete' or manifest['held_privileged_read']:
        raise ValueError('complete retained train-only labels required')
    return {(r['task'], r['demo']): dict(np.load(r['path'])) for r in manifest['records']}


def topology(affinity):
    return dict(host=socket.gethostname(), world_size=1,
                physical_device=os.environ['CUDA_VISIBLE_DEVICES'], affinity=affinity)
