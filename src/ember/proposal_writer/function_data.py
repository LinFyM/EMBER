"""Fixed non-held conditions and label-side native control queries for G."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import copy

import numpy as np
import torch
from safetensors.torch import load_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.practice import processed
from .contract import TASKS, OPTIMIZER, seed
from .full_writer import link, learning_tasks
from .learning import HistoryCache
from .path import save_history
from .teacher import QueryStream

ROOT_SCHEMA = 'ember_full_g_native_function_identification_v1'
CHECKPOINT_SCHEMA = 'ember_full_g_native_function_checkpoint_v1'


def prepare(args):
    """Freeze audited identities and queries; source assets remain read-only."""
    root = Path(args.root)
    audit = read_json(args.input_audit)
    sources = audit['sources']
    source = Path(sources['full_writer_root'])
    selector = Path(sources['complete_pair_seen36_root']).parent.parent
    old = read_json(selector/'run_contract.json')
    authority = copy.deepcopy(read_json(Path(old['run_scoped_task_authority'])))
    authority.update(schema_version='ember_full_g_function_task_authorities_v1',
                     run_root=str(root), training_tasks=list(TASKS), seen36=list(TASKS), novel36=[])
    authority['tasks'] = [q for q in authority['tasks'] if q['authority']['task_id'] in TASKS]
    for row in authority['tasks']:
        row['gradient_roles'] = ['G']
    root.mkdir(parents=True, exist_ok=True)
    for name in ('initial.safetensors', 'task_lora_contract.json', 'bank_scale.json',
                 'coordinates.json', 'base'):
        link(root/name, source/name)
    environment = copy.deepcopy(old['environment'])
    environment['tasks'] = [q for q in environment['tasks'] if q['global_task_id'] in TASKS]
    contract = dict(schema_version=ROOT_SCHEMA, checkpoint_schema=CHECKPOINT_SCHEMA,
        design='docs/designs/problem_first_compiler_candidate_20261013.md#12',
        source=old['source'], mt_checkpoint=old['mt_checkpoint'], environment=environment,
        training_tasks=list(TASKS), G_seen_tasks=list(TASKS), G_novel_tasks=[],
        run_scoped_task_authority=str(root/'task_authorities.json'),
        source_checkpoint=str(source/'checkpoints/RL/update_00000009'),
        G_objective='native10_complete_G_first5x7_MSE', optimizer=OPTIMIZER,
        arms={'F_only': ['endpoint'], 'full_G': ['meta', 'reader', 'static', 'endpoint']},
        updates=72, checkpoint_updates=[18, 36, 54, 72], logical_condition_batch=4,
        profile_B4_per_arm=2, replay_registered_targets=27, replay_maximum=72,
        query_rollouts=1952, complete_rollout_maximum=2024, physical_starts_maximum=2040,
        failed_partial_episode_maximum=16, environment_steps_maximum=1081200,
        limits=dict(added_peak_GiB=64),
        new_held_practice=False, Test=False, no_intermediate_model_selection=True)
    write_json_atomic(root/'task_authorities.json', authority)
    write_json_atomic(root/'run_contract.json', contract)
    write_json_atomic(root/'inputs/asset_audit.json', audit)
    (root/'inputs/scientific_contract.md').write_text(Path(args.scientific_contract).read_text())
    tasks = learning_tasks(args.asset_root, root, TASKS, 'train')
    runtime = SimpleNamespace(tasks=tasks)
    conditions = []
    for task_row in audit['fixed_288_panel']['rows']:
        task = task_row['task']
        for context_index, context in enumerate(task_row['contexts']):
            for video_index, demo in enumerate(task_row['teacher_videos']):
                visit = context_index*4+video_index
                stream = QueryStream(runtime, task, demo, visit, logical_queries=28,
                                     query_demo_indices=range(46))
                try:
                    queries = [dict(stream.query(i), noise_seed=seed(212, task, visit, i)) for i in range(8)]
                finally:
                    stream.close()
                conditions.append(dict(context, task=task, teacher_demo=demo, task_visit=visit,
                    condition_id=f'task_{task:04d}_visit_{visit}', normal_queries=queries,
                    sampled_teacher_frames=task_row['teacher_frames'][str(demo)]))
    targets = audit['T_targets']
    if (len(conditions) != 288 or sorted({c['task'] for c in conditions}) != sorted(TASKS)
            or len([t for t in targets if t['replay_required']]) != 27):
        raise ValueError('fixed all36/288 condition and 27 registered replay identities required')
    write_json_atomic(root/'input_manifest.json', dict(complete=True, conditions=conditions,
        targets=targets, category_weights='mean per category, equal available categories',
        source_episode_reuse_is_not_independent_evidence=True,
        query_labels_enter_condition=False, normal_queries=2304))
    original_panel = read_json(selector/'panel.json')
    write_json_atomic(root/'panel.json', dict(original_panel,
        tasks=[q for q in original_panel['tasks'] if q['task_id'] in TASKS]))
    return contract


def replay_targets(args, runtime, runner):
    """Only one registered first-success attempt per missing actual T."""
    manifest = read_json(args.root/'input_manifest.json')
    tasks = {q['global_task_id']: q for q in runner.contract['tasks']}
    requests = []
    for target in manifest['targets']:
        if not target['replay_required']:
            continue
        directory = args.root/'target_replays'/target['target_id']
        if (directory/'complete.json').exists():
            continue
        rng = target['first_success_RNG']
        if rng['env_seed'] != rng['policy_seed_root']:
            raise ValueError('registered target replay needs identical native scene/policy root')
        requests.append(dict(episode_id=target['target_id'], task=tasks[target['task']],
            state_id=target['first_success_state'], noise_root=rng['policy_seed_root'],
            state=load_file(target['weights'], device=str(runtime.device)),
            parameter_ref=target['query_reference'], capture_history=True, capture_phi=False))
    by_id = {t['target_id']: t for t in manifest['targets']}
    for output in runner.run(sorted(requests, key=lambda r: r['task']['horizon'], reverse=True)):
        target = by_id[output['request']['episode_id']]
        directory = args.root/'target_replays'/target['target_id']
        directory.mkdir(parents=True, exist_ok=True)
        history = output['history']
        original = read_json(Path(target['query_receipt']))
        row = next(r for r in original['rows'] if r['reference'] == target['query_reference']
                   and r['init_state_id'] == target['first_success_state'])
        trace = history.episodes[0]['policy_noise_seeds']
        old_trace = row['policy_noise_seeds']
        if trace[:min(len(trace), len(old_trace))] != old_trace[:min(len(trace), len(old_trace))]:
            raise ValueError('registered target replay lost original full noise prefix')
        left, right = row['initial_proprio'], output['row']['initial_proprio']
        if len(left) != len(right) or max(abs(a-b) for a, b in zip(left, right)) > 1e-5:
            raise ValueError('registered target replay changed the actual initial observation')
        save_history(directory/'history.pt.gz', history)
        write_json_atomic(directory/'complete.json', dict(complete=True, target_id=target['target_id'],
            source=target, original_success=bool(row['success']), replay_success=bool(output['row']['success']),
            outcome_changed=bool(row['success']) != bool(output['row']['success']),
            history=str(directory/'history.pt.gz'), row=output['row'],
            records=len(history.records), environment_steps=output['row']['environment_steps']))
    write_json_atomic(args.root/'complete/target_replays.json', dict(complete=True,
        targets=[read_json(args.root/'target_replays'/t['target_id']/'complete.json')
                 for t in manifest['targets'] if t['replay_required']],
        no_replacement_or_retry_for_negative_replay=True, runner=runner.components))


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

    def compile(self, condition, *, frozen_memory=False, source=None):
        """One canonical parent/H/teaching compiler for learning and readback."""
        runtime = self.runtime
        parent = self.state(condition['parent_weights'])
        memory = None
        cache = self.root/'frozen_F_contexts'/f"{condition['condition_id']}.pt"
        if frozen_memory and cache.exists():
            saved = torch.load(cache, map_location='cpu', weights_only=False)
            if saved['source'] != source:
                raise ValueError('F-only memory came from a different frozen frontend')
            memory = saved['memory'].to(runtime.device)
            features, rows, states = None, None, None
        else:
            history = self.histories.get(condition['H'])
            rows = history.model_rows(runtime)
            self.histories.remember_features(condition['H'], history)
            states = history.states
            features = runtime.teaching(condition['task'], condition['teacher_demo'])
            if frozen_memory:
                from ember.writer.runtime import autocast
                with torch.no_grad(), autocast(runtime.device):
                    memory = runtime.generator.context(features, rows, states).detach()
                cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(source=source, memory=memory.cpu()), cache)
        from ember.writer.runtime import autocast
        with autocast(runtime.device):
            return runtime.generator.compile(parent, features, rows, states, memory=memory)
