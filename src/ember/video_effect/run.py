"""One formal owner surface for source patches, Phi, compilation and fixed final."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import time

import torch
from safetensors.torch import load_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic, git_state
from ember.pi05_source_setup import initialize_distributed
from ember.pi05_eval_queue import (EvaluationShard, initialize_queue, claim_next, complete_job,
    fail_job, publish_json_exclusive, queue_summary)
from ember.proposal_writer.runtime import Runtime
from ember.proposal_writer.contract import ASSET_ROOT, RUN_ROOT as RUNTIME_ASSETS, TASKS
from ember.proposal_writer.learning import memory_peak
from ember.proposal_writer.path import save_history
from ember.writer.practice import EpisodeRunner
from .learning import PROBES, TRAIN_TASKS, TRAIN_DEMOS, feature_file, source_patches, frozen_phi
from .compiler import Compilation, ActualEpisodes, case_request, FINAL_NOISE, ADAPT_NOISE


def make_queues(root, run):
    """Immutable scientific work, dynamic physical ownership and long-first cost."""
    rows = []
    lengths = {q['authority']['task_id']: q['episode_lengths'] for q in read_json(Path(run['task_metadata']))['tasks']}
    for task in run['environment']['tasks']:
        t = task['global_task_id']
        demos = TRAIN_DEMOS if t in TRAIN_TASKS else (0, 1)
        rows.append(EvaluationShard(f'features_t{t:04d}', len(rows), task['suite'], task['task_id'],
            task['horizon'], tuple(demos), sum(lengths[t][d] for d in demos)))
    initialize_queue(root / 'features.sqlite', rows, contract_reference=str(root / 'run_contract.json'))
    mixed, panel = [], {}
    for task in run['environment']['tasks']:
        t = task['global_task_id']
        if t not in PROBES: continue
        identity = f'bootstrap_t{t:04d}'
        mixed.append(EvaluationShard(identity, len(mixed), task['suite'], task['task_id'],
            task['horizon'], tuple(run['adaptation_states']), task['horizon'] * 12))
        panel[identity] = dict(stage='bootstrap', task=t)
        for arm in ('video0', 'video1', 'no_video'):
            identity = f'adapt_t{t:04d}_{arm}'
            mixed.append(EvaluationShard(identity, len(mixed), task['suite'], task['task_id'],
                task['horizon'], tuple(run['adaptation_states']), task['horizon'] * 192))
            panel[identity] = dict(stage='adapt', task=t, arms=[arm])
            identity = f'final_t{t:04d}_{arm}'
            mixed.append(EvaluationShard(identity, len(mixed), task['suite'], task['task_id'],
                task['horizon'], tuple(range(16)), task['horizon'] * 16))
            panel[identity] = dict(stage='final', task=t, arm=arm)
    initialize_queue(root / 'work.sqlite', mixed, contract_reference=str(root / 'run_contract.json'))
    with sqlite3.connect(root / 'work.sqlite', timeout=60) as database:
        database.execute('CREATE TABLE IF NOT EXISTS work_ready (job_id TEXT PRIMARY KEY)')
    write_json_atomic(root / 'queue_plan.json', dict(features=[asdict(r) for r in rows],
        work=[asdict(r) for r in mixed], panel=panel, persistent=True, final_rows=512))


def ready_ids(root):
    plan = read_json(Path(root) / 'queue_plan.json')['panel']
    with sqlite3.connect(Path(root) / 'work.sqlite', timeout=60) as database:
        ready = {row[0] for row in database.execute('SELECT job_id FROM work_ready')}
    finals = {identity for identity,row in plan.items() if row['stage']=='final'}
    return [identity for identity, row in plan.items()
            if row['stage']=='bootstrap' or (identity in ready and
                (row['stage']!='final' or finals.issubset(ready)))]


def condition_ready(root, task, arm, *, stage='final'):
    with sqlite3.connect(Path(root) / 'work.sqlite', timeout=60) as database:
        database.execute('INSERT OR IGNORE INTO work_ready(job_id) VALUES (?)', (f'{stage}_t{task:04d}_{arm}',))


def _publish(queue, claim, args, record, rows, successes=0):
    relative = Path('shards') / queue.stem / f'{claim.shard.job_id}_attempt_{claim.attempt:03d}.json'
    size = publish_json_exclusive(args.root / relative, record)
    complete_job(queue, job_id=claim.shard.job_id, worker_id=args.worker_id, claim_token=claim.claim_token,
        rows_path=str(relative), rows_bytes=size, row_count=rows, successes=successes)


def features(args, runtime, tasks):
    queue, chunk = args.root / 'features.sqlite', args.frame_chunk
    while (claim := claim_next(queue, worker_id=args.worker_id)) is not None:
        try:
            task = tasks[claim.task_key]; measurements = []
            for demo in claim.shard.init_state_ids:
                file = feature_file(args.root, task['global_task_id'], demo)
                if file.exists(): continue
                torch.cuda.reset_peak_memory_stats(runtime.device); started = time.monotonic()
                value = source_patches(runtime, task['global_task_id'], demo, chunk)
                file.parent.mkdir(parents=True, exist_ok=True); temporary = file.with_suffix('.tmp')
                torch.save(value, temporary); temporary.replace(file)
                measurements.append(dict(task=task['global_task_id'], demo=demo, frames=len(value['frame_indices']),
                    seconds=time.monotonic()-started, chunk=chunk, peak=memory_peak(runtime.device)))
                # The next registered video, rather than a discarded cache profile.
                free, _ = torch.cuda.mem_get_info(runtime.device)
                if chunk < 64 and free > 12*1024**3: chunk = 64
            _publish(queue, claim, args, dict(complete=True, measurements=measurements,
                input_demo_ids=list(claim.shard.init_state_ids), pure_spatial_features=True), len(claim.shard.init_state_ids))
        except BaseException as error:
            fail_job(queue, job_id=claim.shard.job_id, worker_id=args.worker_id, claim_token=claim.claim_token, error=repr(error))
            raise


def bootstrap(args, runtime, runner, task):
    """16 MT cases/task: four old cases plus 12 new, physically shared once."""
    root = args.root / 'bootstrap' / f'task_{task["global_task_id"]:04d}'
    states = read_json(args.root/'run_contract.json')['adaptation_states']
    missing = [i for i in states if not (root / f'state_{i:03d}/row.json').exists()]
    if not missing and (root / 'complete.json').exists():
        if not read_json(root / 'complete.json')['complete']:
            raise ValueError('registered bootstrap is incomplete')
        return
    requests = [case_request(task, runtime.mt, 'MT300', i,
        f'MT_bootstrap_t{task["global_task_id"]:04d}_state_{i:03d}', snapshots=True) for i in missing]
    for history, source in ActualEpisodes(args.root, runner).run(requests):
        init = history.episodes[0]['init_state_id']; dest = root / f'state_{init:03d}'
        dest.mkdir(parents=True, exist_ok=True)
        for name in ('history.pt.gz', 'row.json'):
            (dest / name).symlink_to(source / name)
    write_json_atomic(root / 'complete.json', dict(complete=True, unique_episodes=len(states),
        logical_condition_consumers=3, noise_root=ADAPT_NOISE, states=states,
        old_reused_episodes=4,new_unique_episodes=12,
        parameters='MT300merged_complete_r8_identity', prefix_source='this_batch_real_own_interaction'))


def reuse_fixed_MT_final(root, directory, arm, ref, weights, state_ids):
    """Every unchanged policy uses the one pre-registered official MT128 source."""
    if ref != 'MT300': return
    original = read_json(Path(read_json(root/'run_contract.json')['fixed_MT_reference']))
    task = int(directory.name.split('_')[1])
    rows = {r['init_state_id']:r for r in original['rows'] if r['global_task_id']==task and r['arm']=='MT'}
    if set(rows)!=set(range(16)): raise ValueError('official MT reference is incomplete')
    for i in state_ids:
        target = directory / f'state_{i:03d}.json'
        if target.exists(): continue
        row = dict(rows[i])
        if row['actual_parameter_ref'] != 'MT300' or row['settling_steps'] != 10:
            raise ValueError('official MT source violates fixed policy or settling')
        row.update(arm=arm,teacher_demo=None if arm in ('MT','no_video') else int(arm[-1]),
            reused_fixed_MT_reference=True, original_reference=str(read_json(root/'run_contract.json')['fixed_MT_reference']),
            new_compilation_LoRA=weights, new_compilation_parameter_ref=ref)
        write_json_atomic(target, row)


def final_cases(args, runtime, runner, task, arm, state_ids):
    directory = args.root / 'final' / f'task_{task["global_task_id"]:04d}_{arm}'
    directory.mkdir(parents=True, exist_ok=True)
    if arm == 'MT':
        state, ref, weights = runtime.mt, 'MT300', str(RUNTIME_ASSETS / 'initial.safetensors')
    else:
        source = args.root / 'conditions' / f'task_{task["global_task_id"]:04d}_{arm}'
        receipt = read_json(source / 'complete.json')
        if not receipt['complete']: raise ValueError('final policy is not frozen')
        weights = receipt['final_LoRA']; ref = receipt['final_parameter_ref']
        state = load_file(weights, device=str(runtime.device))
    reuse_fixed_MT_final(args.root, directory, arm, ref, weights, state_ids)
    requests = [dict(task=task, state=state, parameter_ref=ref, state_id=i,
        noise_root=FINAL_NOISE, episode_id=f'final_t{task["global_task_id"]:04d}_{arm}_state_{i:03d}',
        capture_history=False, capture_hidden=False, capture_phi=False, uses_teaching=False)
        for i in state_ids if not (directory / f'state_{i:03d}.json').exists()]
    for result in runner.run(requests) if requests else ():
        row = result['row']; init = result['request']['state_id']
        if row.get('model_failure') or row.get('budget_truncated'):
            raise RuntimeError('incomplete final case cannot replace a scientific result')
        first = result['history'].records[0]
        row.update(global_task_id=task['global_task_id'], arm=arm, LoRA=weights,
            first_noise_seed=first['noise_seed'], first_native_normalized_actions=first['normalized_actions'].tolist(),
            teacher_demo=None if arm in ('MT','no_video') else int(arm[-1]),
            exact_L_in_policy=True, fixed_complete_LoRA=True, compiler_runs_in_final=False,
            query_excluded_from_all_training_and_adaptation=True, internal_probe_not_held=True)
        write_json_atomic(directory / f'state_{init:03d}.json', row)
    return [read_json(directory / f'state_{i:03d}.json') for i in state_ids]


def work_event(kind, worker):
    print('EMBER_WORK_EVENT ' + json.dumps(dict(kind=kind, worker=worker)), flush=True)


def compile_conditions(args, runtime, runner, context, task, arms, phi):
    outputs=[]
    for arm in arms:
        base=args.root/'conditions'/f'task_{task["global_task_id"]:04d}_{arm}'
        if not (base/'complete.json').exists():
            compiler=Compilation(args.root,runtime,runner,context,task,arm,phi)
            compiler.run();del compiler
        outputs.append(str(base/'complete.json'))
        condition_ready(args.root,task['global_task_id'],arm)
        work_event('ready',args.worker_id)
    return outputs


def worker(args, runtime, runner, context, tasks):
    queue = args.root / 'work.sqlite'; plan = read_json(args.root / 'queue_plan.json')['panel']
    phi = None
    if read_json(args.root/'run_contract.json')['prediction_recompute_enabled']:
        phi = frozen_phi(args.root, runtime.device)
    while True:
        with sqlite3.connect(queue,timeout=60) as database:
            if database.execute("SELECT COUNT(*) FROM jobs WHERE status='failed'").fetchone()[0]: break
        eligible = ready_ids(args.root)
        claim = claim_next(queue, worker_id=args.worker_id, eligible_job_ids=eligible)
        if claim is None:
            work_event('idle', args.worker_id)
            # The sole owner wakes a persistent consumer on actual new ready work.
            # No interval polling, CPU/GPU forward, or stage self-notification.
            command = sys.stdin.readline().strip()
            if command == 'stop' or not command: break
            if command != 'wake': raise ValueError('unknown physical work wakeup')
            continue
        row = plan[claim.shard.job_id]; task = next(t for t in tasks.values() if t['global_task_id']==row['task'])
        tick = time.monotonic()
        try:
            if row['stage'] == 'bootstrap':
                bootstrap(args, runtime, runner, task)
                for arm in ('video0','video1','no_video'): condition_ready(args.root,row['task'],arm,stage='adapt')
                output,successes=[str(args.root/'bootstrap'/f'task_{row["task"]:04d}'/'complete.json')],0
            elif row['stage'] == 'adapt':
                arms = row.get('arms', ('video0','video1','no_video'))
                output, successes = compile_conditions(args,runtime,runner,context,task,arms,phi),0
            else:
                if runner.slots!=16: runner.close();runner.slots=16
                output = final_cases(args, runtime, runner, task, row['arm'], claim.shard.init_state_ids)
                successes = sum(r['success'] for r in output)
            adapt = row['stage'] == 'adapt'
            rows = len(claim.shard.init_state_ids) if row['stage']!='final' else len(output)
            _publish(queue, claim, args, dict(complete=True, work=row, output=output,
                row_count_basis='practice_states' if row['stage']!='final' else 'final_cases',
                compiled_conditions=len(output) if adapt else None,
                seconds=time.monotonic()-tick, peak=memory_peak(runtime.device), actual_slots=runner.slots), rows, successes)
            work_event('ready', args.worker_id)
        except BaseException as error:
            fail_job(queue,job_id=claim.shard.job_id,worker_id=args.worker_id,claim_token=claim.claim_token,error=repr(error))
            raise
    if args.command == 'worker': work_event('exiting', args.worker_id)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['features','train','worker'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--worker-id')
    p.add_argument('--resume',type=Path)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--frame-chunk',type=int,default=32)
    p.add_argument('--slots',type=int,default=4)
    p.add_argument('--physical-gpu',type=int,required=True)
    return p.parse_args()


def main():
    args = parse_args(); identity = git_state()
    if identity['dirty_paths'] or identity['branch']:
        raise ValueError('formal consumers require clean pushed detached code')
    subprocess.run(['git','merge-base','--is-ancestor',identity['commit'],'origin/main'],check=True)
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    torch.set_num_threads(8)
    write_json_atomic(args.root / 'code' / f'{args.command}_{args.worker_id or context.rank}_{identity["commit"]}.json',
        dict(**identity, world=context.world_size, rank=context.rank, frame_chunk=args.frame_chunk, slots=args.slots))
    if args.command == 'train':
        if not read_json(args.root/'run_contract.json').get('Phi_training_enabled',True):
            raise ValueError('this frozen-Phi scientific run forbids shared training')
        from .learning import train
        train(args, context); return
    run = read_json(args.root / 'run_contract.json')
    runtime = Runtime(context.device, asset_root=ASSET_ROOT, root=RUNTIME_ASSETS, task_ids=TASKS)
    runtime.policy.eval().requires_grad_(False)
    runtime.episode_event_log = os.environ.get('EMBER_EPISODE_EVENT_LOG')
    environment = run['environment']; tasks = {(t['suite'],t['task_id']):t for t in environment['tasks']}
    runner = None
    try:
        if args.command == 'features': features(args, runtime, tasks); return
        os.environ.update(EMBER_LIBERO_ASSETS_ROOT=environment['libero_paths']['assets'],MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
        from ember.pi05_assets import prepare_libero_config, configure_libero_runtime_assets
        prepare_libero_config(args.root/'libero_config');configure_libero_runtime_assets(Path(environment['libero_paths']['assets']))
        environment['own_physical_observations'] = args.command == 'worker'
        runner = EpisodeRunner(runtime,environment,args.physical_gpu,slots=args.slots)
        worker(args,runtime,runner,context,tasks)
    finally:
        if runner is not None: runner.close()
        runtime.close()


if __name__ == '__main__': main()
