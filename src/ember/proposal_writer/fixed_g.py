"""One frozen-G endpoint per legal MT prefix, followed by new paired queries."""
from dataclasses import asdict
from pathlib import Path
import copy
import fcntl
import json
import time

import torch
from safetensors.torch import load_file, save_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_eval_queue import (EvaluationTask, EvaluationShard, initialize_queue,
    claim_next, complete_job, fail_job, publish_json_exclusive, queue_summary)
from .analysis import paired, billing
from .path import begin_condition, generate_candidates, load_history, history_prefix, cpu_state
from .endpoint_context import FrozenEndpointContexts
from ember.writer.runtime import autocast


def read_panel(root):
    panel = read_json(Path(root)/'fixed_g_panel.json'); rows = panel['rows']
    if not Path(panel['checkpoint']).is_absolute():
        raise ValueError('fixed-G checkpoint must have one explicit absolute identity')
    expected = {'seen': (576, 36), 'held': (400, 400)}
    arm = panel['arm']; cursor = panel['F_cursor']
    if (arm not in expected or cursor not in (38349, 38781)
            or (arm == 'held' and cursor != 38781)):
        raise ValueError('unregistered fixed-G arm or endpoint cursor')
    count, endpoints = expected[arm]
    if panel['counts'] != dict(conditions=count, endpoints=endpoints):
        raise ValueError('fixed-G panel cardinality differs from its contract')
    tasks = {t['global_task_id']: t for t in panel['environment']['tasks']}
    if (len(tasks) != len(panel['environment']['tasks'])
            or len({(t['suite'], t['task_id']) for t in tasks.values()}) != len(tasks)):
        raise ValueError('fixed-G environment global/local task identities must be unique')
    keys = {(r['task'], r['init_state_id']) for r in rows}
    if len(rows) != count or len(keys) != count or {r['task'] for r in rows} != set(tasks):
        raise ValueError('fixed-G rows must uniquely cover every registered task')
    signatures = endpoint_input_identities(rows,tasks)
    check_cardinality(arm,rows,tasks,signatures,endpoints)
    return panel


def check_cardinality(arm,rows,tasks,signatures,endpoints):
    """Require complete seen query16 or held single-video-per-state coverage."""
    if len(signatures) != endpoints:
        raise ValueError('fixed-G unique endpoint cardinality differs')
    per_task = [sum(r['task'] == t for r in rows) for t in tasks]
    if len(tasks) != (36 if arm == 'seen' else 8) or set(per_task) != {16 if arm == 'seen' else 50}:
        raise ValueError('fixed-G task/query coverage differs')
    if arm == 'seen' and any(len({r['endpoint_key'] for r in rows if r['task'] == t}) != 1 for t in tasks):
        raise ValueError('seen query16 requires one shared endpoint per task')
    if arm == 'held' and any(len({r['teacher_demo'] for r in rows if r['task'] == t}) != 50 for t in tasks):
        raise ValueError('held K1 requires fifty distinct teaching videos per task')


def endpoint_input_identities(rows,tasks):
    """Bind each complete MT prefix to one legal teaching and paired RNG."""
    signatures = {}
    for row in rows:
        endpoint = row['endpoint_key']
        if not isinstance(endpoint, str) or not endpoint or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in endpoint):
            raise ValueError('endpoint key must be a stable safe path component')
        signature = {k: row[k] for k in ('task', 'teacher_demo', 'path_source', 'history_source',
            'prefix_records', 'prefix_episodes')}
        signature['memory_cache'] = row.get('memory_cache')
        if signature['prefix_episodes'] != 1 or signature['prefix_records'] < 1:
            raise ValueError('fixed-G input requires one complete nonempty MT episode')
        if endpoint in signatures and signatures[endpoint] != signature:
            raise ValueError('shared endpoint has different teaching or actual MT prefix')
        signatures[endpoint] = signature
        task = tasks[row['task']]; mt = row['MT_row']; noise = row['noise_root']
        required = dict(global_task_id=row['task'], init_state_id=row['init_state_id'],
            env_seed=noise, policy_seed_root=noise, **{k: task[k] for k in ('suite', 'task_id', 'language')})
        if any(mt.get(k) != v for k, v in required.items()) or 'registered_scene' not in row:
            raise ValueError('MT reference and fixed-G query identity differ')
    return signatures


def destination(root, row):
    return Path(root)/'conditions'/f"task_{row['task']:04d}_state_{row['init_state_id']:03d}"


def prepare_queues(root, *, workers, shard_size=32, recover=False):
    root = Path(root); panel = read_panel(root); rows = panel['rows']
    tasks = [EvaluationTask(t['suite'], t['task_id'], t['horizon'],
        tuple(r['init_state_id'] for r in rows if r['task'] == t['global_task_id']))
        for t in panel['environment']['tasks']]
    parts = [(t, t.init_state_ids[start:start+shard_size]) for t in tasks
        for start in range(0, len(t.init_state_ids), shard_size)]
    horizon = max(t.horizon for t in tasks)
    shards = [EvaluationShard(f'fixed_G_{i:03d}', i, t.suite, t.task_id, t.horizon,
        ids, t.horizon*len(ids), 0 if t.horizon == horizon else None) for i, (t, ids) in enumerate(parts)]
    base = root/'queues'/'fixed_g'; file = base/'plan.json'
    identity_keys = ('task', 'init_state_id', 'teacher_demo', 'video_ordinal', 'path_source',
        'history_source', 'prefix_records', 'prefix_episodes', 'endpoint_key', 'noise_root', 'registered_scene')
    plan = dict(checkpoint=panel['checkpoint'], F_cursor=panel['F_cursor'], arm=panel['arm'],
        rows=[{k: r[k] for k in identity_keys} for r in rows], shard_size=shard_size,
        shards=[asdict(s) for s in shards])
    if file.exists() and read_json(file) != json.loads(json.dumps(plan)):
        raise ValueError('fixed-G frozen panel or queue layout changed')
    write_json_atomic(file, plan)
    initialize_queue(base/'jobs.sqlite', shards, contract_reference=str(root/'fixed_g_panel.json'),
        recover_claims=recover, retry_failed=recover)
    return dict(fixed_g=str(base/'jobs.sqlite'))


def cached_endpoint(runtime,row):
    """The identical frozen MT condition can reuse its complete lawful m."""
    value=torch.load(row['memory_cache'],map_location='cpu',weights_only=False)
    identity=dict(frontend_checkpoint=str(Path(runtime.frozen_frontend_checkpoint).resolve()),
        initial=str((runtime.root/'initial.safetensors').resolve()),
        bank_scale=str(runtime.bank_scale_path.resolve()),
        merged_base=str((runtime.root/'base/merged_targets.safetensors').resolve()),
        source_model=str(Path(runtime.source['model_path']).resolve()),task=row['task'],parent='MT300',
        parent_weights=str((runtime.root/'initial.safetensors').resolve()),
        history=dict(path=str(Path(row['history_source']).resolve())))
    FrozenEndpointContexts.validate(value,identity,row['teacher_demo'])
    if value['H_records']!=row['prefix_records'] or value['H_episodes']!=1 or value['producers']!=['MT300']:
        raise ValueError('cached fixed-G memory differs from the complete actual MT prefix')
    first=row['MT_episode_evidence']
    if (first['actual_parameter_ref']!='MT300' or first['replans']!=row['prefix_records']
            or first.get('model_failure') or first.get('budget_truncated')):
        raise ValueError('cached input must retain its complete MT producer evidence')
    with autocast(runtime.device):
        state=runtime.generator.generate(runtime.mt,None,None,None,memory=value['memory'].to(runtime.device))
    return state,first


@torch.no_grad()
def materialize(args, runtime, panel, row):
    out = args.root/'endpoints'/row['endpoint_key']; out.mkdir(parents=True, exist_ok=True)
    weights = out/'G_000.safetensors'
    identity = {k: row[k] for k in ('endpoint_key', 'task', 'teacher_demo', 'path_source',
        'history_source', 'prefix_records', 'prefix_episodes')}
    identity.update(checkpoint=panel['checkpoint'], F_cursor=panel['F_cursor'])
    identity['memory_cache']=row.get('memory_cache')
    with (out/'generation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if weights.exists():
            if read_json(out/'generation.json')['input_identity'] != identity:
                raise ValueError('saved fixed-G endpoint belongs to a different condition or kernel')
            return weights
        if row.get('memory_cache'):
            if panel['arm']!='seen':raise ValueError('only the registered seen MT memories are referenced')
            state,first=cached_endpoint(runtime,row)
            if any(r['init_state_id']==first['init_state_id'] for r in panel['rows']
                   if r['endpoint_key']==row['endpoint_key']):raise ValueError('cached MT practice overlaps final query')
            temporary=weights.with_suffix('.tmp');save_file(cpu_state(state),str(temporary))
            write_json_atomic(out/'generation.json',dict(input_identity=identity,
                kernel_identity=runtime.fixed_kernel_identity,MT_episode=first,Actor_called=False,
                exact_frozen_memory_reused=True,new_practice_episodes=0,no_gradients=True,later_H_or_query_used=False))
            temporary.replace(weights)
            return weights
        history = load_history(row['history_source']); first = history.episodes[0]
        n = row['prefix_records']; records = history.records[:n]
        if (first.get('model_failure') or first.get('budget_truncated')
                or first['actual_parameter_ref'] != 'MT300' or n != first['replans']
                or len(records) != n or any(r['episode'] != 0 or r['parameter_ref'] != 'MT300' for r in records)):
            raise ValueError('registered input is not the complete first MT-produced episode')
        if any(r['init_state_id'] == first['init_state_id'] for r in panel['rows']
                if r['endpoint_key'] == row['endpoint_key']):
            raise ValueError('fixed-G query initial state leaked into its practice history')
        prefix = history_prefix(history, n, 1)
        old = read_json(Path(row['path_source']))
        task = next(t for t in runtime.environment_tasks if t['global_task_id'] == row['task'])
        path, _ = begin_condition(runtime, task, row['teacher_demo'], row['endpoint_key'],
            old.get('path_ordinal', 0), row['noise_root'], 1, None, False, None, False, False)
        path.history = prefix; path.environment_steps = first['environment_steps']
        features = runtime.teaching(row['task'], row['teacher_demo'])
        if generate_candidates(runtime, path, 'MT300', features) != ['G_000']:
            raise ValueError('fixed-G generation must produce one complete deterministic endpoint')
        provenance = dict(input_identity=identity, kernel_identity=runtime.fixed_kernel_identity,
            MT_episode=first, attempt=path.attempts[0], Actor_called=False, no_gradients=True,
            new_practice_episodes=0, later_H_or_query_used=False)
        temporary = weights.with_suffix('.tmp')
        save_file(cpu_state(path.states['G_000']), str(temporary))
        write_json_atomic(out/'generation.json', provenance)
        temporary.replace(weights)
    return weights


def conditions(args, runtime, runner, panel, rows):
    requests = []; owners = {}; results = []; states = {}
    for row in rows:
        file = destination(args.root, row)/'query.json'
        if file.exists():
            saved = read_json(file)
            if (saved['checkpoint'] != panel['checkpoint'] or saved['F_cursor'] != panel['F_cursor']
                    or saved['endpoint_key'] != row['endpoint_key'] or saved['arm'] != panel['arm']
                    or saved['teacher_demo'] != row['teacher_demo']):
                raise ValueError('completed query belongs to another fixed-G kernel')
            paired([row['MT_row']], [saved]); results.append(saved); continue
        weights = materialize(args, runtime, panel, row)
        if row['endpoint_key'] not in states:
            states[row['endpoint_key']] = load_file(str(weights), device=str(runtime.device))
        task = next(t for t in panel['environment']['tasks'] if t['global_task_id'] == row['task'])
        identity = f"fixed_G_task_{row['task']:04d}_state_{row['init_state_id']:03d}"
        owners[identity] = row, weights
        requests.append(dict(task=task, state=states[row['endpoint_key']], parameter_ref='G_000',
            state_id=row['init_state_id'], noise_root=row['noise_root'], episode_id=identity,
            remaining=task['horizon']+10, snapshots=False, uses_teaching=False,
            capture_history=False, capture_hidden=False, registered_scene=row['registered_scene']))
    for output in runner.run(requests):
        row, weights = owners[output['request']['episode_id']]
        result = {**output['row'], 'global_task_id': row['task'], 'teacher_demo': row['teacher_demo'],
            'video_ordinal': row['video_ordinal'], 'reference': 'G_000', 'arm': panel['arm'],
            'checkpoint': panel['checkpoint'], 'F_cursor': panel['F_cursor'], 'endpoint_key': row['endpoint_key'],
            'fixed_G_weights': str(weights), 'query_excluded_from_H': True, 'fixed_unique_LoRA': True,
            'actual_new_environment_steps': output['row']['environment_steps']}
        if output['history'].records:
            first = output['history'].records[0]
            result.update(first_native_normalized_actions=first['normalized_actions'].tolist(),
                first_noise_seed=first['noise_seed'])
        if result.get('model_failure') or result.get('budget_truncated'):
            write_json_atomic(destination(args.root, row)/'query_failure.json', result)
            raise ValueError('incomplete engineering query cannot count as a fixed-G final row')
        paired([row['MT_row']], [result])
        write_json_atomic(destination(args.root, row)/'query.json', result); results.append(result)
    return results


def worker(args, runtime, runner):
    from .pipeline import load_version
    from .learning import memory_peak
    panel = read_panel(args.root); state = load_version(runtime, args.checkpoint)
    if (state['G_sampler']['next_update'] != panel['F_cursor']
            or str(args.checkpoint.resolve()) != panel['checkpoint']):
        raise ValueError('loaded fixed-G checkpoint differs from the frozen panel')
    runtime.generator.eval().requires_grad_(False); runtime.actor.eval().requires_grad_(False)
    runtime.environment_tasks = panel['environment']['tasks']
    tasks = {(t['suite'], t['task_id']): t['global_task_id'] for t in runtime.environment_tasks}
    base = args.root/'queues'/'fixed_g'; queue = base/'jobs.sqlite'; completed = 0
    while (claim := claim_next(queue, worker_id=args.worker_id)) is not None:
        try:
            tick = time.monotonic(); before = copy.deepcopy(runner.components)
            runtime_before = dict(runtime.components); before_steps = runner.environment_steps
            rows = [r for r in panel['rows'] if r['task'] == tasks[claim.task_key]
                and r['init_state_id'] in claim.shard.init_state_ids]
            results = []
            for start in range(0, len(rows), args.condition_batch):
                results.extend(conditions(args, runtime, runner, panel, rows[start:start+args.condition_batch]))
            relative = Path('shards')/claim.shard.job_id/f'attempt_{claim.attempt:03d}.json'
            size = publish_json_exclusive(base/relative, dict(rows=results, checkpoint=panel['checkpoint'], F_cursor=panel['F_cursor']))
            complete_job(queue, job_id=claim.shard.job_id, worker_id=args.worker_id, claim_token=claim.claim_token,
                rows_path=str(relative), rows_bytes=size, row_count=len(results), successes=sum(bool(r['success']) for r in results))
            record = dict(job_id=claim.shard.job_id, task=tasks[claim.task_key], horizon=claim.shard.horizon,
                conditions=len(results), slots=args.slots, condition_batch=args.condition_batch,
                seconds=time.monotonic()-tick, actual_new_environment_steps=runner.environment_steps-before_steps,
                physical_batch_histogram={k:v-before['physical_batch_histogram'].get(k,0)
                    for k,v in runner.components['physical_batch_histogram'].items()}, peak=memory_peak(runtime.device),
                runner={k:v-before[k] for k,v in runner.components.items() if k != 'physical_batch_histogram'},
                runtime={k:v-runtime_before.get(k,0.) for k,v in runtime.components.items()})
            file = args.root/'metrics'/f'fixed_G_physical_{args.worker_id}.jsonl'; file.parent.mkdir(parents=True, exist_ok=True)
            with file.open('a') as stream: stream.write(json.dumps(record)+'\n')
            completed += 1
            if args.max_groups is not None and completed >= args.max_groups: break
        except BaseException as error:
            fail_job(queue, job_id=claim.shard.job_id, worker_id=args.worker_id, claim_token=claim.claim_token, error=repr(error))
            raise
    return dict(fixed_g=queue_summary(queue))


def complete_readout(root):
    root = Path(root); panel = read_panel(root)
    rows = [read_json(destination(root, r)/'query.json') for r in panel['rows']]
    for source, result in zip(panel['rows'], rows):
        if (result['checkpoint'] != panel['checkpoint'] or result['F_cursor'] != panel['F_cursor']
                or result['endpoint_key'] != source['endpoint_key'] or result['arm'] != panel['arm']
                or result['teacher_demo'] != source['teacher_demo']
                or result['actual_new_environment_steps'] <= 0):
            raise ValueError('fixed-G readout contains a different kernel')
    mt = [r['MT_row'] for r in panel['rows']]
    result = dict(complete=True, arm=panel['arm'], checkpoint=panel['checkpoint'], F_cursor=panel['F_cursor'],
        raw_rows=rows, MT_raw_rows=mt, fixed_G_vs_MT=paired(mt, rows), per_task={}, per_suite={},
        actual_generations=len(list((root/'endpoints').glob('*/generation.json'))),
        new_final_rows=len(rows), reused_final_rows=0, billing=billing(root), Test_enabled=False,
        new_practice_episodes=0, new_gradients=0, counts=panel['counts'])
    if result['actual_generations'] != panel['counts']['endpoints']:
        raise ValueError('fixed-G endpoint coverage is incomplete')
    for field, out in [('global_task_id', 'per_task'), ('suite', 'per_suite')]:
        for value in sorted({r[field] for r in rows}):
            indices = [i for i, row in enumerate(rows) if row[field] == value]
            result[out][str(value)] = paired([mt[i] for i in indices], [rows[i] for i in indices])
    result['breadth'] = sum(r['candidate_success'] > 0 for r in result['per_task'].values())
    write_json_atomic(root/'results.json', result)
    write_json_atomic(root/'complete.json', dict(complete=True, results=str(root/'results.json'), no_automatic_successor=True))
    return result
