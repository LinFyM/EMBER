"""Frozen learning metadata and actual phi180 refresh on the shared queue."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import traceback

import numpy as np

from ember.pi05_eval_queue import claim_next, complete_job, completed_jobs, fail_job, publish_json_exclusive, queue_summary
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from .contract import EVENT_SCHEMA, MT_PATH, RUN_ROOT, SEED, SOURCE, STAGE, TASKS36, condition_seed, learning_environment, training_tasks
from .interaction import Runner
from .storage import save_condition


ROOT = RUN_ROOT
PARENT = Path('/data1/user/ymdai/ember_runs/parameter_conditioned_compiler_20261009')
CHECKPOINT_SCHEMA = 'ember_functional_revision_learning_checkpoint_v1'


def _immutable(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'frozen formal metadata changed: {path}')
    else:
        write_json_atomic(path, value)


def bootstrap_metadata():
    conditions, origins = [], []
    for pool in ('pool0', 'refresh180'):
        path = PARENT / 'pools' / pool / 'manifest.json'
        manifest = read_json(path)
        if not manifest.get('complete'):
            raise ValueError('only complete original pools are legal bootstrap')
        origins.append(str(path))
        for record in manifest['conditions']:
            if not record.get('actual_incoming_parameters') or not record.get('complete'):
                raise ValueError('bootstrap event lacks actual parameter/experience provenance')
            conditions.append(dict(record, source_manifest=str(path)))
    counts, tasks, teachers = Counter(), set(), defaultdict(set)
    for condition in conditions:
        tasks.add(condition['task_id'])
        teachers[condition['task_id']].add(condition['teacher_demo'])
        for event in condition['events']:
            counts['MT' if event['incoming'] == 'MT' else 'nonMT'] += 1
    nonMT_tasks = {c['task_id'] for c in conditions if any(e['incoming'] != 'MT' for e in c['events'])}
    if (len(conditions), counts['MT'], counts['nonMT'], len(nonMT_tasks)) != (216, 314, 48, 16):
        raise ValueError('fixed original216 event coverage changed')
    if tasks != set(TASKS36) or any(len(values) != 6 for values in teachers.values()):
        raise ValueError('bootstrap must retain all36 tasks and six teachers per task')
    return dict(schema_version=EVENT_SCHEMA, complete=True, pool='bootstrap', conditions=conditions,
        original_manifests=origins, endpoint_counts=dict(counts), actual_nonMT_tasks=sorted(nonMT_tasks))


def prepare_learning(root):
    """Before first learning, freeze all72 unseen-teacher refresh candidates."""
    root = Path(root)
    bootstrap = bootstrap_metadata()
    tasks = training_tasks()
    original_panel = read_json(PARENT / 'evaluation/train180/evaluation_contract.json')
    excluded = defaultdict(set)
    for condition in bootstrap['conditions'] + original_panel['conditions']:
        excluded[condition['task_id']].add(condition['teacher_demo'])
    refresh = []
    for task_id in TASKS36:
        task = tasks[task_id]
        bag = np.random.default_rng(np.random.SeedSequence([SEED, 0xA180, task_id])).permutation(50)
        demos = [int(demo) for demo in bag if int(demo) not in excluded[task_id]][:2]
        for ordinal, demo in enumerate(demos):
            refresh.append(dict(condition_id=f'functional_refresh180_task{task_id:03d}_demo{demo:02d}',
                task_id=task_id, suite=task.suite, suite_task_id=task.suite_task_id,
                language=task.authority.language, teacher_demo=demo,
                seed=condition_seed(task_id, demo, ordinal, domain=0xA180),
                excluded_states=[32, 33, 34], pool='refresh180'))
    contract = dict(schema_version=CHECKPOINT_SCHEMA, stage=STAGE, seed=SEED,
        source=SOURCE, MT_reference=str(MT_PATH), task_ids=list(TASKS36),
        logical_batch=4, updates=360, phase_boundaries=[180, 360], checkpoints=[90, 180, 270, 360],
        optimizer=dict(name='AdamW', fresh=True, learning_rate=3e-5, betas=[.9, .999], eps=1e-8,
                       weight_decay=0., global_clip=1., scheduler='constant'),
        objective=dict(FM_queries_per_event=28, cross_episode=True, keep_coefficient=1., PG=False),
        compilation=dict(rank=128, targets=38, support=16, step_budget=1024,
                         loop_stop='own_success_or_environment_budget_after_editing_success_E'),
        fresh_shared_model=True, bootstrap_behavior_is_historical=True, formal_checkpoint=360,
        evaluation=dict(train=[180, 360], correct_paired400=[360], null=False, held_controls=False, Test=False),
        limits=dict(GPU_hours=20, scientific_elapsed_hours=8, additional_GiB=224, total_cache_GiB=64))
    _immutable(root / 'run_contract.json', contract)
    _immutable(root / 'pools/bootstrap/manifest.json', bootstrap)
    checkpoint = root / 'training/checkpoints/step_00000180'
    collection = dict(schema_version=EVENT_SCHEMA, pool='refresh180', version=180,
        behavior_checkpoint=str(checkpoint), conditions=refresh, step_budget=1024,
        environment_contract=learning_environment(),
        excluded_teachers={str(k): sorted(v) for k, v in excluded.items()},
        selection='seeded50bag_excluding_original_six_and_original_train48_before_first_update')
    _immutable(root / 'pools/refresh180/collection_contract.json', collection)
    return dict(bootstrap_conditions=216, refresh_conditions=len(refresh), phase1_nonMT_exposures=160)


def prepare_collection(root, *, recover_claims=False, retry_failed=False):
    from .evaluation import initialize_condition_queue
    path = Path(root) / 'pools/refresh180'
    contract = read_json(path / 'collection_contract.json')
    initialize_condition_queue(path / 'queue.sqlite3', contract['conditions'],
        contract['environment_contract'], contract_reference=str(path / 'collection_contract.json'),
        recover_claims=recover_claims, retry_failed=retry_failed)
    return contract


def collect(runtime, args):
    """Each finished real chain is saved before its queue claim is committed."""
    import time
    path = Path(args.run_root) / 'pools/refresh180'
    contract = prepare_collection(args.run_root)
    checkpoint = Path(contract['behavior_checkpoint'])
    manifest = read_json(checkpoint / 'manifest.json')
    if (int(manifest['macro_update']) != 180 or manifest.get('schema_version') != CHECKPOINT_SCHEMA
            or manifest.get('compiler_schema') != 'ember_functional_revision_compiler_v1'
            or manifest.get('complete') is not True):
        raise ValueError('refresh behavior must be the complete frozen180 model')
    runtime.load_checkpoint(checkpoint)
    runtime.compiler.eval()
    conditions = {c['condition_id']: c for c in contract['conditions']}
    tasks = {t['global_task_id']: t for t in contract['environment_contract']['tasks']}
    claims, resumed = {}, {}
    runner = Runner(runtime, contract['environment_contract'], args.physical_gpu, slot_batch=args.slot_batch)

    def provider():
        claim = claim_next(path / 'queue.sqlite3', worker_id=args.worker_id, physical_gpu=args.physical_gpu)
        if claim is None:
            raise StopIteration
        condition = conditions[claim.shard.job_id]
        destination = path / 'conditions' / condition['condition_id']
        if (destination / 'record.json').exists():
            record = read_json(destination / 'record.json')
            if (not record.get('complete') or record.get('behavior_checkpoint') != str(checkpoint)
                    or any(record.get(key) != value for key, value in condition.items())):
                raise ValueError('existing refresh chain has incompatible behavior identity')
            _finish_refresh(path, claim, args.worker_id, record)
            resumed[condition['condition_id']] = True
            return provider()
        claims[condition['condition_id']] = claim
        return dict(kind='adapt', condition=condition, task=tasks[condition['task_id']],
                    step_budget=1024, behavior_version='functional_phi180')

    try:
        for result in runner.run(provider):
            condition, chain = result['request']['condition'], result['chain']
            destination = path / 'conditions' / condition['condition_id']
            future = runtime.io.submit(save_condition, destination, condition, chain,
                byte_cost=sum(v.nbytes for raw in chain.observations.values() for v in raw.values()))
            record = future.result()
            record.update(behavior_checkpoint=str(checkpoint), collection_code_git=args.code_git)
            write_json_atomic(destination / 'record.json', record)
            _finish_refresh(path, claims.pop(condition['condition_id']), args.worker_id, record)
        runtime.io.flush()
        write_json_atomic(path / 'workers' / f'{args.worker_id}.json', dict(complete=True,
            code_git=args.code_git, components=runner.components, actual_environment_steps=runner.total_environment_steps,
            resumed_completed_conditions=sorted(resumed), unix=time.time()))
    except BaseException:
        runner.preserve_partial(path / 'partial' / args.worker_id)
        for claim in claims.values():
            fail_job(path / 'queue.sqlite3', job_id=claim.shard.job_id, worker_id=args.worker_id,
                     claim_token=claim.claim_token, error=traceback.format_exc())
        raise
    finally:
        runner.close()


def _finish_refresh(path, claim, worker, record):
    shard = path / 'shards' / f'{claim.shard.job_id}_attempt{claim.attempt}.json'
    if shard.exists():
        if read_json(shard) != dict(records=[record]):
            raise ValueError('saved refresh shard differs from actual condition')
        size = shard.stat().st_size
    else:
        size = publish_json_exclusive(shard, dict(records=[record]))
    complete_job(path / 'queue.sqlite3', job_id=claim.shard.job_id, worker_id=worker,
        claim_token=claim.claim_token, rows_path=str(shard.relative_to(path)), rows_bytes=size,
        row_count=1, successes=int(record['metrics']['practice_success']))


def aggregate_collection(root):
    path = Path(root) / 'pools/refresh180'
    contract, summary = read_json(path / 'collection_contract.json'), queue_summary(path / 'queue.sqlite3')
    rows = [record for job in completed_jobs(path / 'queue.sqlite3')
            for record in read_json(path / job['rows_path'])['records']]
    if len(rows) != 72 or {r['condition_id'] for r in rows} != {c['condition_id'] for c in contract['conditions']}:
        raise ValueError(f'refresh72 is incomplete: {summary}')
    rows.sort(key=lambda r: r['condition_id'])
    manifest = dict(schema_version=EVENT_SCHEMA, complete=True, pool='refresh180', version=180,
        conditions=rows, collection_contract=str(path / 'collection_contract.json'), queue_summary=summary,
        behavior_checkpoint=contract['behavior_checkpoint'])
    _immutable(path / 'manifest.json', manifest)
    return manifest
