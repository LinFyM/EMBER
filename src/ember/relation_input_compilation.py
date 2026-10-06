"""Matched P/Q input selection on the original, immutable G450 graph.

Temporary owner for relation_input_compilation_20261007. No feedback model,
live geometry, new labels, or deployment reader is introduced here.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
from safetensors import safe_open

import ember
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.operator_writer.data import TASKS
from ember.operator_writer.native import read_native_video
from ember.writer.runtime import autocast

ROOT = Path('/data1/user/ymdai/ember_runs/relation_input_compilation_20261007')
OLD = Path('/data1/user/ymdai/ember_runs/relation_grounded_writer_20261006')
PARENT = OLD / 'train/attempts/fresh/checkpoints/macro_00000450'
LEGACY = OLD / 'frozen_PEFTfix'
FIELDS = ('p', 'R', 'semantic', 'presence', 'joint_type', 'joint_q', 'hand_q')
# Reuse precisely the saved scientific modules, without restoring their trainer.
if str(LEGACY / 'src/ember') not in ember.__path__:
    ember.__path__.append(str(LEGACY / 'src/ember'))
from ember.relation_writer.runtime import RelationWriter


class InputWriter(RelationWriter):
    """Same parameters and registration order; explicit frozen input choice."""
    def forward(self, native_inputs, h, frame_indices, *, capture_mechanism=False,
                physical_fields=None, **unused):
        if set(native_inputs) != set(self.names) or unused:
            raise ValueError('P/Q condition contains an undeclared field')
        if self.arm == 'P':
            if physical_fields is not None:
                raise ValueError('predicted-input P cannot receive teacher labels')
            with torch.no_grad():
                prediction = self.phi(h, frame_indices)
        elif self.arm == 'Q' and physical_fields is not None:
            if set(physical_fields) != set(FIELDS):
                raise ValueError('GT input must contain only the seven original fields')
            prediction = physical_fields
        else:
            raise ValueError('Q needs an existing train-only teacher cache')
        return self.from_fields(native_inputs, h, frame_indices, prediction,
                                capture_mechanism=capture_mechanism)

    def from_fields(self, native_inputs, h, frame_indices, prediction, *, capture_mechanism=False):
        """Original Omega/Read/ConditionalTarget expression and checkpoint policy."""
        from torch.utils.checkpoint import checkpoint
        w = self.omega(prediction, frame_indices)
        c, d = self.read(h, w, prediction)
        common, result, edits = self.public_state(), {}, {}
        for name, unit in zip(self.names, self.conditional_targets, strict=True):
            a0, b0 = common[name + LORA_A_SUFFIX], common[name + LORA_B_SUFFIX]
            inputs = (a0, b0, native_inputs[name], h, c, d)
            a, b, s, m = (checkpoint(unit, *inputs, use_reentrant=False, preserve_rng_state=False)
                          if torch.is_grad_enabled() else unit(*inputs))
            result[name + LORA_A_SUFFIX], result[name + LORA_B_SUFFIX] = a, b
            if capture_mechanism:
                edits[name] = {'S': s, 'M': m}
        self.last_prediction = prediction
        if capture_mechanism:
            self.last_mechanism = {'prediction': prediction, 'c': c, 'd': d, 'targets': edits}
        return result


def load_G_weights(checkpoint, device='cpu'):
    with safe_open(str(Path(checkpoint) / 'ecp.safetensors'), framework='pt', device=str(device)) as reader:
        keys = list(reader.keys())
        if Path(checkpoint).resolve() == PARENT:
            if {key.split('.', 1)[0] for key in keys} != {'G', 'F'}:
                raise ValueError('original parent is not the registered joint ECP')
            return {key[2:]: reader.get_tensor(key) for key in keys if key.startswith('G.')}
        if not Path(checkpoint).resolve().is_relative_to(ROOT / 'train'):
            raise ValueError('unregistered P/Q checkpoint')
        return {key: reader.get_tensor(key) for key in keys}


def build_runtime(assets, spec, device, arm, checkpoint=PARENT, *, freeze=True):
    from ember.relation_writer.runtime import build_runtime as original_runtime
    if arm not in ('P', 'Q') or tuple(spec['events']['task_ids']) != TASKS:
        raise ValueError('P/Q requires the original train36 scope')
    runtime = original_runtime(assets, spec, device)
    # Change only the input consumer method; every original module/parameter
    # instance and its registration order survive for correct Adam restoration.
    runtime.writer.__class__ = InputWriter
    runtime.writer.arm = arm
    runtime.writer.load_state_dict(load_G_weights(checkpoint, device=runtime.device), strict=True)
    if freeze:
        freeze_upstream(runtime)
    return runtime


def freeze_upstream(runtime):
    runtime.policy.requires_grad_(False).eval()
    runtime.writer.common.requires_grad_(False).eval()
    runtime.writer.phi.requires_grad_(False).eval()
    return tuple(p for p in runtime.writer.parameters() if p.requires_grad)


def make_labels(data, assets, spec):
    from ember.relation_writer.labels import LabelStore
    from ember.pi05_source_checkpoint import read_json
    class ExistingLabels(LabelStore):
        def _episode(self, task, demo, requested):
            path = self.cache_root / f'task{task}_demo{demo}.npz'
            if not path.is_file():
                raise ValueError(f'no existing physical cache: {path}')
            return super()._episode(task, demo, requested)
        def queries(self, *args, **kwargs):
            raise ValueError('this diagnostic cannot read query physical labels')
    if Path(spec['diagnostic']['labels_root']).resolve() != OLD / 'labels':
        raise ValueError('GT labels must reference the original immutable cache')
    labels = ExistingLabels(data, Path(assets), cache_root=OLD / 'labels')
    labels.set_semantics(read_json(OLD / 'labels/semantic_vectors.json')['vectors'])
    return labels


def cached_fields(labels, task, demo, indices, device):
    gt = labels.teacher(task, demo, np.asarray(indices, np.int64))
    if not np.all(gt['valid']) or not np.array_equal(gt['frame_indices'], indices):
        raise ValueError(f'used GT frames have internal missing fields: task{task}/demo{demo}')
    values = {name: torch.as_tensor(gt[name], device=device) for name in FIELDS}
    from ember.relation_writer.representation import check_physical
    if check_physical(values) != (len(indices),) or any(not torch.isfinite(v).all() for v in values.values()):
        raise ValueError('existing GT schema/finite contract failed')
    return values


def prepare_condition(runtime, data, labels, task, demo, frame_chunk):
    if task not in TASKS or data.role != 'train':
        raise ValueError('teacher input is restricted to original train36')
    condition, raw, sampled = data.condition(runtime, task, demo)
    original = condition[1].detach().cpu().tolist()
    if sampled < 2 or original[-1] != raw - 1:
        raise ValueError('original stride5 plus final frame contract changed')
    condition = (condition[0][:-1], condition[1][:-1], *condition[2:])
    used = condition[1].detach().cpu().tolist()
    physical = (cached_fields(labels, task, demo, used, runtime.device)
                if runtime.writer.arm == 'Q' else None)
    if runtime.writer.arm == 'P' and labels is not None:
        raise ValueError('P input owner must not open the GT store')
    runtime.restore_identity()
    with torch.no_grad(), autocast(runtime.device):
        x, h = read_native_video(runtime.policy, runtime.writer.public_state(), runtime.writer.probe,
                                 condition[:4], runtime.writer.names, frame_chunk=frame_chunk)[:2]
        fields = runtime.writer.phi(h, condition[1]) if physical is None else physical
    record = {'task': task, 'teacher_demo': demo, 'arm': runtime.writer.arm,
              'field_source': 'frozen_Phi' if physical is None else 'existing_train_teacher_GT',
              'raw_frames': raw, 'original_sampled_frames': sampled, 'sampled_frames': len(used),
              'original_indices': original, 'used_indices': used, 'native_reads': 1,
              'GT_query_reads': 0, 'GT_live_reads': 0}
    return x, h, condition[1], fields, record


def tensor_summary(value):
    value = value.detach().float()
    return {'rms': float(value.square().mean().sqrt()), 'norm': float(value.norm()),
            'max_abs': float(value.abs().max()), 'finite': bool(torch.isfinite(value).all())}


def mechanism_summary(writer):
    mechanism = writer.last_mechanism
    result = {'c': tensor_summary(mechanism['c']), 'd': tensor_summary(mechanism['d']),
              'targets': {name: {key: tensor_summary(value) for key, value in edits.items()}
                          for name, edits in mechanism['targets'].items()}}
    del writer.last_mechanism
    return result


def compile_condition(runtime, data, labels, task, demo, frame_chunk, capture=True):
    started = time.perf_counter()
    x, h, indices, fields, record = prepare_condition(runtime, data, labels, task, demo, frame_chunk)
    with torch.no_grad(), autocast(runtime.device):
        state = runtime.writer.from_fields(x, h, indices, fields, capture_mechanism=capture)
    validate_lora_state(state, runtime.lora)
    if capture:
        record['mechanism'] = mechanism_summary(runtime.writer)
    record['seconds'] = time.perf_counter() - started
    return state, record


def one_condition(runtime, data, labels, event, *, microbatch, frame_chunk):
    from ember.writer.function_credit import paired_functional_credit
    started = time.perf_counter()
    x, h, indices, fields, record = prepare_condition(
        runtime, data, labels, event['task'], event['teacher_demo'], frame_chunk)
    with torch.no_grad(), autocast(runtime.device):
        state = runtime.writer.from_fields(x, h, indices, fields, capture_mechanism=True)
    record['mechanism'] = mechanism_summary(runtime.writer)
    batch = runtime.processor.training_batch(data.batch(event))
    with autocast(runtime.device):
        credit = paired_functional_credit(runtime.policy, state, runtime.lora, batch,
            seed=event['flow_seed'], device=runtime.device, random_batch=28, offset=0,
            microbatch=microbatch, condition_weight=.25)
        replay = runtime.writer.from_fields(x, h, indices, fields)
        cotangent = credit['lora_cotangent']
        if set(replay) != set(cotangent) or any(not torch.isfinite(v).all() for v in cotangent.values()):
            raise ValueError('main FM lost a full finite38-target LoRA cotangent')
        torch.autograd.backward(tuple(replay.values()), tuple(cotangent[n].to(replay[n]) for n in replay))
    torch.cuda.synchronize(runtime.device)
    return {**record, 'absolute_update': event['update'], 'visit': event['visit'],
            'queries': len(event['queries']), 'query_demos': [q['demo'] for q in event['queries']],
            'query_frames': [q['frame'] for q in event['queries']], 'flow_seed': event['flow_seed'],
            'query_offset': event['query_offset'], 'condition_weight': .25,
            'flow_loss': credit['flow_loss'], 'loss': 'only_original_main_FM_full50x7',
            'FM_cotangent_norm': float(torch.stack([v.float().norm() for v in cotangent.values()]).norm()),
            'seconds': time.perf_counter() - started}
