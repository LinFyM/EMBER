"""Finite frozen first-G diagnostic; reuses canonical generation and policy execution.

This consumer owns provenance and readout only. Retire its entrypoint after the
registered diagnostic is consumed; frozen code and raw results retain history.
"""
from dataclasses import asdict
from pathlib import Path
import copy
import json
import time

import torch
from safetensors.torch import load_file, save_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_eval_queue import (EvaluationTask, EvaluationShard,
    initialize_queue, claim_next, complete_job, fail_job, publish_json_exclusive, queue_summary)
from .analysis import paired, billing
from .path import begin_condition, generate_candidates, load_history, history_prefix

ARM = 'first_G_from_same_MT_bootstrap'


def destination(root, row):
    return Path(root)/'conditions'/f"task_{row['task']:04d}_state_{row['init_state_id']:03d}"


def prepare_queues(root, *, workers, shard_size=32, recover=False):
    """Ready saved parameters never wait for the 52 missing endpoint forwards."""
    root = Path(root); panel = read_json(root/'first_g_panel.json')
    if panel['arm'] != ARM or panel['counts'] != dict(existing_first_G=348, needs_generation=52,
            reuse_final_result=69, new_query_episodes=331, conditions=400):
        raise ValueError('first-G provenance census differs from the registered diagnostic')
    queues = {}
    for kind, needs_generation in (('saved', False), ('missing', True)):
        rows = [r for r in panel['rows'] if not r['reuse_final_result'] and r['needs_generation'] == needs_generation]
        tasks = [EvaluationTask(t['suite'], t['task_id'], t['horizon'],
            tuple(r['init_state_id'] for r in rows if r['task'] == t['global_task_id']))
            for t in panel['environment']['tasks'] if any(r['task'] == t['global_task_id'] for r in rows)]
        base = root/'queues'/kind; plan_file = base/'plan.json'
        if plan_file.exists():
            plan = read_json(plan_file)
            if plan['checkpoint'] != panel['checkpoint']:
                raise ValueError('frozen diagnostic checkpoint changed')
            shards = [EvaluationShard(**{**s, 'init_state_ids': tuple(s['init_state_ids'])}) for s in plan['shards']]
        else:
            parts = [(s, s.init_state_ids[first:first+shard_size]) for s in tasks
                     for first in range(0, len(s.init_state_ids), shard_size)]
            horizon = max(s.horizon for s in tasks)
            # The existing queue uses a non-null hint to prioritize Long. Claims
            # omit physical_gpu, so this marker never binds a task to a card.
            shards = [EvaluationShard(f'{kind}_{ordinal:03d}', ordinal, s.suite, s.task_id,
                s.horizon, ids, s.horizon*len(ids), 0 if s.horizon == horizon else None)
                for ordinal, (s, ids) in enumerate(parts)]
            plan = dict(checkpoint=panel['checkpoint'], shard_size=shard_size,
                initial_physical_workers=workers, dynamic_claim_without_physical_GPU=True,
                shards=[asdict(s) for s in shards])
            write_json_atomic(plan_file, plan)
        initialize_queue(base/'jobs.sqlite', shards, contract_reference=str(root/'first_g_panel.json'),
            recover_claims=recover, retry_failed=recover)
        queues[kind] = str(base/'jobs.sqlite')
    reuse_final_rows(root, panel)
    return queues


def reuse_final_rows(root, panel):
    """Only actual selected G000 finals share the same fixed policy and RNG."""
    for row in panel['rows']:
        if not row['reuse_final_result']: continue
        original = read_json(Path(row['original_final_query']))['selected']
        if original['reference'] != 'G_000' or row['needs_generation']:
            raise ValueError('only the original selected first-G final row is reusable')
        result = {**original, 'arm': ARM, 'first_G_weights': row['first_G_path'],
            'fixed_parameter_result_reused_from': row['original_final_query'],
            'actual_new_environment_steps': 0, 'original_selected': row['original_selected']}
        file = destination(root, row)/'query.json'
        if file.exists() and read_json(file) != result:
            raise ValueError('reused first-G result changed')
        write_json_atomic(file, result)


@torch.no_grad()
def materialize_missing(args, runtime, row):
    """The actual own MT episode is the entire legal history of a pre-G STOP."""
    out = destination(args.root, row); weights = out/'G_000.safetensors'
    if weights.exists():
        provenance = read_json(out/'generation.json')
        if provenance['checkpoint'] != str(args.checkpoint.resolve()):
            raise ValueError('saved diagnostic endpoint belongs to another kernel')
        return weights
    old_path = read_json(Path(row['path_source']))
    history = load_history(row['history_source'])
    if (old_path['attempts'] or old_path['completed_rounds'] != 0 or len(history.episodes) != 1
            or not history.records or any(r['parameter_ref'] != 'MT300' for r in history.records)):
        raise ValueError('missing first G requires its own complete one-episode MT prefix')
    episode = history.episodes[0]
    if (episode.get('model_failure') or episode.get('budget_truncated')
            or episode['init_state_id'] == row['init_state_id']):
        raise ValueError('bootstrap is incomplete or includes the final initial state')
    prefix = history_prefix(history, len(history.records), 1)
    task = next(t for t in runtime.environment_tasks if t['global_task_id'] == row['task'])
    path, _ = begin_condition(runtime, task, row['teacher_demo'], old_path['identity'],
        old_path['path_ordinal'], old_path['noise_root'], 1, None, False, None, False, False)
    path.history = prefix
    path.environment_steps = int(episode['environment_steps'])
    features = runtime.teaching(row['task'], row['teacher_demo'])
    refs = generate_candidates(runtime, path, 'MT300', features)
    if refs != ['G_000'] or path.attempts[0]['parent'] != 'MT300':
        raise ValueError('diagnostic must generate exactly one first MT-parent proposal')
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out/'generation.json', dict(checkpoint=str(args.checkpoint.resolve()),
        history_source=row['history_source'], path_source=row['path_source'],
        actual_prefix_records=len(prefix.records), actual_prefix_episodes=1, MT_episode=episode,
        teacher_demo=row['teacher_demo'], parent='MT300', attempt=path.attempts[0],
        kernel_identity=runtime.fixed_kernel_identity, new_practice_episodes=0, Actor_called=False,
        no_gradients=True, later_H_or_query_used=False))
    save_file(path.states['G_000'], str(weights))
    return weights


def conditions(args, runtime, runner, panel, rows):
    requests = []; by_identity = {}; results = []
    for row in rows:
        file = destination(args.root, row)/'query.json'
        if file.exists():
            results.append(read_json(file)); continue
        weights = (materialize_missing(args, runtime, row) if row['needs_generation']
                   else Path(row['first_G_path']))
        task = next(t for t in panel['environment']['tasks'] if t['global_task_id'] == row['task'])
        identity = f"first_G_task_{row['task']:04d}_state_{row['init_state_id']:03d}"
        by_identity[identity] = (row, weights)
        requests.append(dict(task=task, state=load_file(str(weights), device=str(runtime.device)),
            parameter_ref='G_000', state_id=row['init_state_id'], noise_root=row['noise_root'],
            episode_id=identity, remaining=task['horizon']+10, snapshots=False, uses_teaching=False,
            capture_history=False, capture_hidden=False, registered_scene=row['registered_scene']))
    for output in runner.run(requests):
        row, weights = by_identity[output['request']['episode_id']]
        result = {**output['row'], 'global_task_id': row['task'], 'teacher_demo': row['teacher_demo'],
            'reference_teacher_demo': row['teacher_demo'], 'video_ordinal': row['video_ordinal'],
            'reference': 'G_000', 'arm': ARM, 'checkpoint': str(args.checkpoint.resolve()),
            'first_G_weights': str(weights), 'original_selected': row['original_selected'],
            'query_excluded_from_H': True, 'fixed_unique_LoRA': True,
            'actual_new_environment_steps': output['row']['environment_steps']}
        if output['history'].records:
            result['first_native_normalized_actions'] = output['history'].records[0]['normalized_actions'].tolist()
        paired([row['MT_row']], [result])
        # Publish each completed row immediately; a local worker failure keeps it.
        write_json_atomic(destination(args.root, row)/'query.json', result)
        results.append(result)
    return results


def worker(args, runtime, runner):
    from .pipeline import load_version
    from .learning import memory_peak
    state = load_version(runtime, args.checkpoint)
    runtime.actor.eval().requires_grad_(False)
    panel = read_json(args.root/'first_g_panel.json')
    if ([state['stage'], state['next_update']] != ['RL', 9]
            or state['G_sampler']['next_update'] != 38349
            or str(args.checkpoint.resolve()) != panel['checkpoint']):
        raise ValueError('first-G diagnostic requires the complete frozen RL9/F38349')
    runtime.environment_tasks = panel['environment']['tasks']
    tasks = {(t['suite'], t['task_id']): t['global_task_id'] for t in runtime.environment_tasks}
    completed = 0
    while True:
        claimed = next(((kind, claim) for kind in ('saved', 'missing')
            if (claim := claim_next(args.root/'queues'/kind/'jobs.sqlite', worker_id=args.worker_id)) is not None), None)
        if claimed is None: break
        kind, claim = claimed; base = args.root/'queues'/kind; queue = base/'jobs.sqlite'
        try:
            tick = time.monotonic(); before = copy.deepcopy(runner.components)
            runtime_before = dict(runtime.components); before_steps = runner.environment_steps
            rows = [r for r in panel['rows'] if r['task'] == tasks[claim.task_key]
                and r['init_state_id'] in claim.shard.init_state_ids]
            results = []
            for first in range(0, len(rows), args.condition_batch):
                results.extend(conditions(args, runtime, runner, panel, rows[first:first+args.condition_batch]))
            relative = Path('shards')/claim.shard.job_id/f'attempt_{claim.attempt:03d}.json'
            size = publish_json_exclusive(base/relative, dict(rows=results, checkpoint=panel['checkpoint']))
            complete_job(queue, job_id=claim.shard.job_id, worker_id=args.worker_id, claim_token=claim.claim_token,
                rows_path=str(relative), rows_bytes=size, row_count=len(results), successes=sum(bool(r['success']) for r in results))
            record = dict(queue=kind, job_id=claim.shard.job_id, task=tasks[claim.task_key],
                horizon=claim.shard.horizon, conditions=len(results), slots=args.slots,
                condition_batch=args.condition_batch, seconds=time.monotonic()-tick,
                actual_new_environment_steps=runner.environment_steps-before_steps,
                physical_batch_histogram={k:v-before['physical_batch_histogram'].get(k,0)
                    for k,v in runner.components['physical_batch_histogram'].items()}, peak=memory_peak(runtime.device),
                runner={k:v-before[k] for k,v in runner.components.items() if k != 'physical_batch_histogram'},
                runtime={k:v-runtime_before.get(k,0.) for k,v in runtime.components.items()})
            file = args.root/'metrics'/f'first_G_physical_{args.worker_id}.jsonl'; file.parent.mkdir(parents=True, exist_ok=True)
            with file.open('a') as stream: stream.write(json.dumps(record)+'\n')
            completed += 1
            if args.max_groups is not None and completed >= args.max_groups: break
        except BaseException as error:
            fail_job(queue, job_id=claim.shard.job_id, worker_id=args.worker_id, claim_token=claim.claim_token, error=repr(error))
            raise
    return {kind: queue_summary(args.root/'queues'/kind/'jobs.sqlite') for kind in ('saved', 'missing')}


def summarize(root):
    root = Path(root); panel = read_json(root/'first_g_panel.json')
    first = [read_json(destination(root, r)/'query.json') for r in panel['rows']]
    full = [read_json(Path(r['original_final_query']))['selected'] for r in panel['rows']]
    mt = [r['MT_row'] for r in panel['rows']]
    comparisons = dict(first_G_vs_MT=paired(mt, first), first_G_vs_full_Writer=paired(full, first))
    result = dict(complete=True, arm=ARM, checkpoint=panel['checkpoint'], raw_rows=first,
        MT_raw_rows=mt, full_Writer_raw_rows=full, comparisons=comparisons, per_task={}, per_suite={}, subsets={},
        expected_counts=panel['counts'], actual_generations=len(list((root/'conditions').glob('*/generation.json'))),
        reused_final_rows=sum('fixed_parameter_result_reused_from' in r for r in first),
        new_final_rows=sum('fixed_parameter_result_reused_from' not in r for r in first),
        billing=billing(root), posthoc_validation_diagnostic=True, replaces_formal_result=False, Test_enabled=False,
        inherited_MT_bootstrap_episodes=400, inherited_MT_bootstrap_environment_steps=
            sum(r['MT_bootstrap_environment_steps'] for r in panel['rows']))
    if (len(first), result['actual_generations'], result['reused_final_rows'], result['new_final_rows']) != (400,52,69,331):
        raise ValueError('actual diagnostic costs/coverage differ from its fixed scope')
    def compare(indices):
        return dict(first_G_vs_MT=paired([mt[i] for i in indices], [first[i] for i in indices]),
                    first_G_vs_full_Writer=paired([full[i] for i in indices], [first[i] for i in indices]))
    for field, out in (('global_task_id', 'per_task'), ('suite', 'per_suite')):
        for value in sorted({r[field] for r in first}):
            result[out][str(value)] = compare([i for i,r in enumerate(first) if r[field] == value])
    groups = dict(original_STOP52=lambda r:r['needs_generation'], original_final_MT111=lambda r:r['original_selected']=='MT300',
        original_final_first_G69=lambda r:r['original_selected']=='G_000',
        original_final_later_G220=lambda r:r['original_selected'] not in ('MT300','G_000'),
        reused69=lambda r:r['reuse_final_result'], new331=lambda r:not r['reuse_final_result'])
    for name, predicate in groups.items():
        result['subsets'][name] = {**compare([i for i,r in enumerate(panel['rows']) if predicate(r)]),
                                  'original_policy_selected_subset_not_randomized': True}
    result['breadth'] = sum(r['first_G_vs_MT']['candidate_success']>0 for r in result['per_task'].values())
    write_json_atomic(root/'results.json', result)
    return result
