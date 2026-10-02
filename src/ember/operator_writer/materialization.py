"""Resident operator compilers on the shared dynamic materialization queue."""
from __future__ import annotations

import os
from pathlib import Path
import time

import torch
from safetensors.torch import load_file, save_file

from ember.lora import LORA_B_SUFFIX
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record
from ember.writer.materialization_workers import MaterializationWorkers, _configure_device

from .data import FormalData


def register_partial(output, contract, previous_git):
    """A reader transition may change code identity, never the training source."""
    registration = output / 'materialization_contract.json'
    if registration.exists():
        old = read_json(registration)
        if old == contract:
            return
        if (previous_git is None or old.get('materialization_git', {}).get('commit') != previous_git
                or {k: v for k, v in old.items() if k != 'materialization_git'}
                != {k: v for k, v in contract.items() if k != 'materialization_git'}):
            raise ValueError('partial bank belongs to a different formal source')
        write_json_atomic(output / 'materialization_resume.json', {
            'previous_contract': old, 'current_contract': contract,
            'reused_condition_files': sorted(p.name for p in output.glob('*.safetensors')
                                           if p.name != 'shared.safetensors')})
    write_json_atomic(registration, contract)


def write_condition(runtime, data, output, condition, shapes, *, mode, frame_chunk):
    from .bank import BANK_SCHEMA, _factor_header
    from .joint_readout import CONDITIONAL_MODES
    from .prefix_change import passive_condition

    condition = dict(condition)
    path = output / f"{condition['condition_id']}.safetensors"
    metadata = {'schema_version': BANK_SCHEMA, 'condition_id': condition['condition_id'], 'mode': mode}
    started = time.monotonic()
    reused = path.exists()
    if reused:
        raw, sampled = data.videos.frame_counts(condition['global_task_id'], condition['teacher_demo'])
        _factor_header(path, shapes, metadata=metadata)
    else:
        pixels, raw, sampled = data.condition(runtime, condition['global_task_id'], condition['teacher_demo'])
        while True:
            try:
                with torch.no_grad():
                    passive = passive_condition(condition, mode)
                    state, native = runtime.compile(pixels, frame_chunk=frame_chunk,
                        **({"capture_prefix_stats": True} if passive else {}))
                break
            except torch.cuda.OutOfMemoryError:
                if frame_chunk <= 8:
                    raise
                frame_chunk = max(8, frame_chunk // 2)
                torch.cuda.empty_cache()
        factors = {name: value.detach().float().cpu().contiguous()
                   for name, value in state.items()
                   if mode in CONDITIONAL_MODES or name.endswith(LORA_B_SUFFIX)}
        if set(factors) != set(shapes):
            raise ValueError("video compiler did not produce its registered complete factors")
        temporary = path.with_suffix('.safetensors.tmp')
        save_file(factors, str(temporary), metadata=metadata)
        temporary.replace(path)
        if passive:
            torch.save({"condition": condition, "frame_indices": torch.as_tensor(pixels[1]).cpu(),
                        "statistics": native["prefix_statistics"],
                        "native_reads": 1, "additional_policy_forwards": 0},
                       output / f"{condition['condition_id']}_prefix_statistics.pt")
    condition.update(factors=file_record(path), raw_frames=raw, sampled_frames=sampled)
    return condition, {'condition_id': condition['condition_id'], 'reused': reused,
                       'device': str(runtime.device), 'pid': os.getpid(),
                       'frame_chunk': frame_chunk, 'seconds': time.monotonic() - started,
                       'peak_reserved_bytes': (torch.cuda.max_memory_reserved(runtime.device)
                                               if runtime.device.type == 'cuda' else 0)}


class OperatorCompiler:
    """One complete source/Writer and video-only reader per physical device."""
    def __init__(self, asset_root, config, device, cpu_threads):
        from .run import PILOT_ARMS, build_runtime
        from .joint_readout import CONDITIONAL_SEEN_MODE
        from .joint_training import CONDITIONAL_MODE
        from .prefix_change import MODE as PREFIX_MODE, SEEN_MODE as PREFIX_SEEN_MODE

        _configure_device(device, cpu_threads)
        runtime_mode = ('T' if config['mode'] in PILOT_ARMS else
                        'context' if config['mode'] == 'context_seen' else
                        PREFIX_MODE if config['mode'] == PREFIX_SEEN_MODE else
                        CONDITIONAL_MODE if config['mode'] == CONDITIONAL_SEEN_MODE else config['mode'])
        self.runtime = build_runtime(asset_root, config['spec'], device, runtime_mode)
        self.asset_root, self.config, self.data, self.request = asset_root, config, None, None

    def prepare(self, request):
        if self.request == request:
            return
        self.close()
        checkpoint, source, _output, _shapes, _frame_chunk = request
        if self.runtime.source != source:
            raise ValueError('materialization source differs from the formal training run')
        self.runtime.writer.load_state_dict(load_file(str(Path(checkpoint) / 'ecp.safetensors'),
                                                      device=str(self.runtime.device)), strict=True)
        self.runtime.writer.requires_grad_(False).eval()
        self.runtime.policy.eval()
        self.data = FormalData(self.asset_root, self.config['spec'], query_labels=False,
                               task_ids=self.config['task_ids'], role=self.config['role'])
        self.request = request

    def compile(self, job):
        _checkpoint, _source, output, shapes, frame_chunk = self.request
        return write_condition(self.runtime, self.data, Path(output), job, shapes,
                               mode=self.config['mode'], frame_chunk=frame_chunk)

    def close(self):
        if self.data is not None:
            self.data.close()
            self.data = None
        self.request = None


def compile_conditions(asset_root, spec, mode, checkpoint, source, output, conditions, shapes,
                       *, devices, frame_chunk, task_ids, role, cpu_threads):
    # Header-check existing factors on CPU; dispatch only unfinished videos, longest first.
    data = FormalData(asset_root, spec, query_labels=False, task_ids=task_ids, role=role)
    completed, statistics, pending = {}, [], []
    try:
        for condition in conditions:
            if (output / f"{condition['condition_id']}.safetensors").exists():
                runtime = type('HeaderReader', (), {'device': torch.device('cpu')})()
                value, stats = write_condition(runtime, data, output, condition, shapes,
                                                mode=mode, frame_chunk=frame_chunk)
                completed[value['condition_id']] = value
                statistics.append(stats)
            else:
                pending.append(condition)
        pending.sort(key=lambda row: data.videos.frame_counts(row['global_task_id'], row['teacher_demo'])[1],
                     reverse=True)
    finally:
        data.close()
    request = (str(checkpoint), source, str(output), shapes, frame_chunk)
    config = {'spec': spec, 'mode': mode, 'task_ids': task_ids, 'role': role}
    if pending:
        with MaterializationWorkers(asset_root=asset_root, config=config, devices=devices,
                                    cpu_threads=cpu_threads, compiler_factory=OperatorCompiler) as workers:
            for _job, (value, stats) in workers.compile(request, pending):
                if value['condition_id'] in completed:
                    raise ValueError('materialization queue returned a duplicate condition')
                completed[value['condition_id']] = value
                statistics.append(stats)
    conditions[:] = [completed[row['condition_id']] for row in conditions]
    write_json_atomic(output / 'materialization_execution.json', {
        'devices': [str(device) for device in devices], 'native_frame_chunk': frame_chunk,
        'cpu_threads_per_worker': cpu_threads, 'conditions': statistics})
