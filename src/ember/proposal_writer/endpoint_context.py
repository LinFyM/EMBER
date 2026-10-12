"""Immutable legal context for the single endpoint gradient owner.

The bank indexes actual parameter/H prefixes. Only the fixed Generator frontend
is cached; the endpoint and Actor continue to run with their current weights.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import time

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.runtime import autocast

from .contract import TASKS
from .runtime import finish_phases


class FrozenEndpointContexts:
    """One task-sharded producer, shared immutable files and bounded device reuse."""
    def __init__(self, runtime, context, root, history_cache, panel):
        self.runtime, self.context, self.root = runtime, context, Path(root)
        self.history_cache, self.panel = history_cache, panel
        contract = read_json(self.root/'run_contract.json')
        if contract.get('G_gradient_owner') != 'endpoint':
            raise ValueError('the active shared G update owner must be endpoint only')
        self.frontend = str(Path(contract['frozen_frontend_checkpoint']).resolve(strict=True))
        self.common = dict(frontend_checkpoint=self.frontend,
            initial=str((self.root/'initial.safetensors').resolve(strict=True)),
            bank_scale=str(runtime.bank_scale_path.resolve(strict=True)),
            merged_base=str((self.root/'base/merged_targets.safetensors').resolve(strict=True)),
            source_model=str(Path(runtime.source['model_path']).resolve(strict=True)))
        self.previous = Path(contract['frozen_context_cache_root']) if contract.get('frozen_context_cache_root') else None
        self.items, self.bytes, self.limit = OrderedDict(), 0, 512*1024**2
        self.parents, self.teachers, self.files = {}, {}, {}
        self.hits, self.loads, self.load_seconds = 0, 0, 0.

    def identity(self, task, parent, history):
        history = dict(history) if isinstance(history, dict) else dict(path=str(history))
        history['path'] = str(Path(history['path']).resolve(strict=True))
        return dict(**self.common, task=task, parent=parent['reference'],
            parent_weights=str(Path(parent['weights']).resolve(strict=True)), history=history)

    @staticmethod
    def validate(value, identity, demo):
        memory = value['memory']
        if (value['identity'] != identity or value['demo'] != demo
                or memory.ndim != 2 or memory.shape[1] != 256
                or value['H_records'] <= 0 or value['H_episodes'] <= 0
                or not value['producers'] or not torch.isfinite(memory).all()):
            raise ValueError('complete frozen context input/prefix/producers changed')

    def old_MT(self, identity, demo):
        """Reuse the original144 only when every available input identity agrees."""
        if self.previous is None or identity['parent'] != 'MT300': return None
        previous = self.previous
        if (str((previous/'initial.safetensors').resolve()) != identity['initial']
                or str((previous/'bank_scale.json').resolve()) != identity['bank_scale']
                or str((previous/'base/merged_targets.safetensors').resolve()) != identity['merged_base']
                or str(Path(read_json(previous/'run_contract.json')['source_checkpoint']).resolve()) != self.frontend
                or identity['parent_weights'] != identity['initial']): return None
        path = previous/'contexts'/f"task_{identity['task']:04d}_demo_{demo:02d}.pt"
        if not path.exists(): return None
        old = torch.load(path, map_location='cpu', weights_only=False)
        history = identity['history']
        if (old['task'] != identity['task'] or old['demo'] != demo
                or str(Path(old['history']).resolve()) != history['path']
                or str(Path(old['source_checkpoint']).resolve()) != self.frontend
                or old['producers'] != ['MT300']
                or history.get('records', old['H_records']) != old['H_records']
                or history.get('episodes', 1) != 1): return None
        return dict(identity=identity, demo=demo, memory=old['memory'], frames=old['frames'],
            H_records=old['H_records'], H_episodes=1, producers=['MT300'],
            source='original144_exact_frozen_MT_context', source_file=str(path.resolve()))

    def _new_contexts(self, task, pending):
        runtime = self.runtime
        # All parents/H for this task use the same four real teaching functions.
        # Encode each teaching once here, then discard these large native tensors.
        demos = {demo for _,demo,_ in pending}
        features = {demo:runtime.teaching(task,demo) for demo in self.panel[task]['policy_videos'] if demo in demos}
        rows_by_history, histories, records = {}, {}, []
        for identity, demo, path in pending:
            key = tuple(sorted(identity['history'].items()))
            if key not in histories:
                history = self.history_cache.get(identity['history'])
                requested = identity['history']
                if (not history.records or not history.episodes
                        or len(history.records) != requested.get('records',len(history.records))
                        or len(history.episodes) != requested.get('episodes',len(history.episodes))
                        or any(row['parameter_ref'] not in history.states for row in history.records)
                        or any(row['episode'] >= len(history.episodes) for row in history.records)):
                    raise ValueError('endpoint context requires the exact completed real H prefix')
                histories[key] = history
                rows_by_history[key] = history.model_rows(runtime)
                self.history_cache.remember_features(identity['history'],history)
            history = histories[key]; started = time.monotonic()
            memory = runtime.generator.context(features[demo],rows_by_history[key],history.states)
            value = dict(identity=identity, demo=demo, memory=memory.detach().cpu().contiguous(),
                frames=len(features[demo]['z']), H_records=len(history.records), H_episodes=len(history.episodes),
                producers=list(history.states), source='actual_frozen_G_frontend',
                context_seconds=time.monotonic()-started)
            self._write(path,value); records.append({k:v for k,v in value.items() if k!='memory'})
        runtime.feature_cache.clear(); runtime.feature_cache_size = 0
        runtime.raw_cache.clear(); runtime.raw_cache_size = 0
        return records

    def _write(self, path, value):
        self.validate(value,value['identity'],value['demo'])
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary = path.with_suffix(f'.rank_{self.context.rank:02d}.tmp')
        torch.save(value,temporary); temporary.replace(path)

    def prepare(self, bank):
        """Encode missing complete bank entries once before a dense F block."""
        started = time.monotonic(); records = []; runtime = self.runtime
        components_before = dict(runtime.components); decode_before = self.history_cache.decode_seconds
        frozen_before = runtime.psi_frozen
        runtime.psi_frozen = True; runtime.generator.eval().requires_grad_(False)
        try:
            with torch.no_grad(),autocast(runtime.device):
                for task_index, task in enumerate(TASKS):
                    pending = []
                    for p, parent in enumerate(bank[task]['parents']):
                        for h, history in enumerate(parent['histories']):
                            identity = self.identity(task,parent,history)
                            for demo in self.panel[task]['policy_videos']:
                                key = task,p,h,demo
                                path = self.root/'endpoint_contexts'/f'task_{task:04d}'/f'parent_{p:05d}_H_{h:05d}_demo_{demo:02d}.pt'
                                previous = self.files.get(key)
                                self.files[key] = path,identity
                                if previous == (path,identity): continue
                                if task_index % self.context.world_size != self.context.rank: continue
                                if path.exists():
                                    self.validate(torch.load(path,map_location='cpu',weights_only=False),identity,demo)
                                    continue
                                value = self.old_MT(identity,demo)
                                if value is None: pending.append((identity,demo,path))
                                else:
                                    self._write(path,value); records.append({k:v for k,v in value.items() if k!='memory'})
                    if pending: records.extend(self._new_contexts(task,pending))
        finally:
            runtime.psi_frozen = frozen_before
        if self.context.world_size > 1: dist.barrier(device_ids=[self.context.local_rank])
        finish_phases(runtime)
        write_json_atomic(self.root/'endpoint_contexts'/f"bank_{runtime.supervision_version:02d}_rank_{self.context.rank:02d}.json",
            dict(complete=True, frontend_checkpoint=self.frontend, bank_version=runtime.supervision_version,
                seconds=time.monotonic()-started, newly_cached=records, actual_entries=len(self.files),
                history_decode_seconds=self.history_cache.decode_seconds-decode_before,
                components={k:v-components_before.get(k,0.) for k,v in runtime.components.items()},
                no_Actor_representation_cache=True, no_labels_in_context=True))

    def get(self, task, parent_index, history_index, demo, parent, teacher):
        key = task,parent_index,history_index,demo
        if key not in self.files: raise ValueError('the current real bank must be prepared before F updates')
        if key in self.items:
            self.hits += 1; self.items.move_to_end(key); value = self.items[key]
        else:
            started = time.monotonic(); path,identity = self.files[key]
            value = torch.load(path,map_location='cpu',weights_only=False)
            self.validate(value,identity,demo)
            value = dict(value,memory=value['memory'].to(self.runtime.device))
            size = value['memory'].nbytes
            while self.items and self.bytes+size > self.limit:
                _,old = self.items.popitem(last=False); self.bytes -= old['memory'].nbytes
            self.items[key] = value; self.bytes += size
            self.loads += 1; self.load_seconds += time.monotonic()-started
        weights = str(Path(parent['weights']).resolve())
        if weights not in self.parents:
            layout = self.runtime.generator.layout
            packed = layout.pack(load_file(weights)).to(self.runtime.device)
            self.parents[weights] = (packed-layout.center)/layout.scale
        if task not in self.teachers:
            self.teachers[task] = load_file(teacher['weights'],device=str(self.runtime.device))
        return value,self.parents[weights],self.teachers[task]
