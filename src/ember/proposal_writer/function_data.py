"""Fixed non-held conditions and label-side native control queries for G."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.practice import processed
from .learning import HistoryCache
from .teacher import QueryStream

ROOT_SCHEMA = 'ember_full_g_native_function_identification_v1'
CHECKPOINT_SCHEMA = 'ember_full_g_native_function_checkpoint_v1'


def uniform_positions(length, count):
    return np.floor(np.linspace(0, length-1, min(count, length))).astype(int).tolist() if length else []


class FunctionInputs:
    """Label side is separate from the original parent/H teaching condition."""
    def __init__(self, runtime, root):
        self.runtime, self.root = runtime, Path(root)
        self.manifest = read_json(self.root/'input_manifest.json')
        self.conditions = {(c['task'], c['task_visit']): c for c in self.manifest['conditions']}
        self.targets = {t['target_id']: t for t in self.manifest['targets']}
        self.histories = HistoryCache(max_entries=3)
        self.weights = {}

    def state(self, path):
        if path not in self.weights:
            if len(self.weights) >= 4:
                self.weights.pop(next(iter(self.weights)))
            self.weights[path] = load_file(path, device=str(self.runtime.device))
        return self.weights[path]

    def success_segments(self, target):
        if target['success_H_segments']:
            return target['success_H_segments']
        if not target['replay_required']:
            return []
        receipt = read_json(self.root/'target_replays'/target['target_id']/'complete.json')
        return [dict(path=receipt['history'], episode=0, record_start=0, records=receipt['records'])] \
            if receipt['replay_success'] else []

    def history_queries(self, segments):
        """At most eight; equal episode mean, uniform distinct replans within it."""
        if not segments:
            return []
        if len(segments) > 8:
            raise ValueError('registered existing success episodes exceed this query budget')
        queries = []
        for episode_index, segment in enumerate(segments):
            count = 8//len(segments)+int(episode_index < 8 % len(segments))
            positions = uniform_positions(segment['records'], count)
            history = self.histories.get(segment['path'])
            for index in positions:
                position = segment['record_start']+index
                record = history.records[position]
                queries.append(dict(raw=history.observations[record['pre']], noise_seed=record['noise_seed'],
                    weight=1/len(segments)/len(positions), source=dict(path=segment['path'],
                    episode=segment['episode'], record=position, observation=record['pre'])))
        return queries

    def queries(self, condition):
        task, visit, demo = (condition[k] for k in ('task', 'task_visit', 'teacher_demo'))
        stream = QueryStream(self.runtime, task, demo, visit, logical_queries=28, query_demo_indices=range(46))
        try:
            normal = []
            for query in condition['normal_queries']:
                value = stream.data.observation(stream.rows[query['demo']][query['frame']])
                raw = dict(images=torch.from_numpy(np.stack((value['observation.images.camera1'],
                    value['observation.images.camera2']))), proprio=torch.from_numpy(value['observation.state']))
                normal.append(dict(raw=raw, noise_seed=query['noise_seed'], weight=1/8, source=query))
        finally:
            stream.close()
        target = self.targets[condition['T_target_id']]
        segments = self.success_segments(target)
        history = self.histories.get(condition['H'])
        keeping = []
        for episode_index, row in enumerate(history.episodes):
            ref = row['actual_parameter_ref']
            producer = (str(self.root/'initial.safetensors') if ref == 'MT300'
                        else condition['parent_weights'] if ref == condition['parent'] else None)
            if row['success'] and producer and Path(producer).resolve() == Path(target['weights']).resolve():
                positions = [i for i, r in enumerate(history.records) if r['episode'] == episode_index]
                keeping.append(dict(path=condition['H']['path'], episode=episode_index,
                    record_start=positions[0], records=len(positions)))
        return [('normal', normal), ('T_success', self.history_queries(segments)),
                ('producer_keep', self.history_queries(keeping))]

    def batch(self, queries):
        from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch
        language = self.runtime.tasks[self.current_task].authority.language
        items = [processed(self.runtime, q['raw'], language) for q in queries]
        resolution = self.runtime.policy.config.image_resolution
        for item in items:
            for key, value in item.items():
                if key.startswith('observation.images.') and value.shape[-2:] != resolution:
                    # The same official resize precedes concatenation of 128px
                    # offline and 256px real-practice observations. PI05 then
                    # applies its unchanged range conversion and image masks.
                    item[key] = resize_with_pad_torch(value.permute(0, 2, 3, 1), *resolution).permute(0, 3, 1, 2)
        batch = {k: torch.cat([v[k] for v in items]) for k in items[0]}
        noise = torch.stack([torch.randn(50, 32, generator=torch.Generator().manual_seed(q['noise_seed']))
                             for q in queries]).to(self.runtime.device)
        return batch, noise

    def compile(self, condition):
        """One canonical parent/H/teaching compiler for learning and readback."""
        runtime = self.runtime
        parent = self.state(condition['parent_weights'])
        history = self.histories.get(condition['H'])
        rows = history.model_rows(runtime)
        self.histories.remember_features(condition['H'], history)
        states = history.states
        features = runtime.teaching(condition['task'], condition['teacher_demo'])
        from ember.writer.runtime import autocast
        with autocast(runtime.device):
            return runtime.generator.compile(parent, features, rows, states)
