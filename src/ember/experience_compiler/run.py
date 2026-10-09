"""Bounded functional-revision implementation checks; formal learning unregistered."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from .contract import ASSET_ROOT, RUN_ROOT, SCHEMA, STAGE, SEED, learning_environment


PARENT = Path('/data1/user/ymdai/ember_runs/parameter_edit_credit_diagnostic_20261009')
PANEL = ((0, 29), (13, 4), (20, 15), (32, 44))


def prepare(root):
    import numpy as np
    import torch
    original = read_json(PARENT / 'run_contract.json')
    by_key = {(c['task_id'], c['teacher_demo']): c for c in original['conditions']}
    conditions = []
    for position, key in enumerate(PANEL):
        source = by_key[key]
        record = torch.load(source['source_experience'], weights_only=False, map_location='cpu')
        practiced = sorted({e['init_state_id'] for e in record['episodes']})
        query_ids = [i for i in range(50) if i not in {*practiced, 32, 33, 34}][:2]
        roots = []
        for query in range(2):
            # Same environment root, independent policy/exploration domains.
            roots.append({name: int(np.random.SeedSequence([SEED, position, query, domain]).generate_state(1)[0])
                          for name, domain in [('environment', 1), ('outgoing_policy', 2),
                              ('incoming_policy', 3), ('outgoing_exploration', 4),
                              ('incoming_exploration', 5), ('outgoing_reservoir', 6),
                              ('incoming_reservoir', 7)]})
        conditions.append(dict(position=position, condition_id=source['condition_id'],
            task_id=key[0], teacher_demo=key[1], language=source['language'], suite=source['suite'],
            suite_task_id=source['suite_task_id'], incoming=source['weights']['I'],
            source_experience=source['source_experience'], source_record=source['source_record'],
            endpoint=source['last_event']['endpoint'], behavior_version=source['last_event']['behavior_version'],
            original_incoming=source['last_event']['incoming'], practiced_states=practiced,
            query_state_ids=query_ids, roots=roots))
    contract = dict(schema_version=SCHEMA, stage=STAGE, seed=SEED, asset_root=str(ASSET_ROOT),
        inherited_contract=str(PARENT / 'run_contract.json'), original_inputs_read_only=True,
        conditions=conditions, environment_contract=learning_environment(),
        support=dict(maximum_distinct_decisions=16, selection='floor_linspace_all_cumulative_decisions'),
        queries=dict(episodes_per_event=2, exploration_sigma=.1, reservoir=16,
                     baseline='same_initial_state_independent_policy_and_exploration_rng'),
        limits=dict(FM_updates=2, PG_updates=1, new_query_episodes=16, environment_steps=5440,
                    GPU_hours=2, scientific_elapsed_hours=3, new_artifact_GiB=40),
        no_new_adaptation=True, formal_training_registered=False, Test_opened=False)
    path = Path(root) / 'run_contract.json'
    if path.exists() and read_json(path) != contract:
        raise ValueError('bounded immutable scientific contract changed')
    write_json_atomic(path, contract)
    return dict(path=str(path), query_states=[c['query_state_ids'] for c in conditions])


def check_budget(root):
    path = Path(root) / 'costs.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    starts = {r['attempt']: r for r in rows if r['event'] == 'start'}
    stops = {r['attempt']: r for r in rows if r['event'] == 'stop'}
    seconds = sum((stops[k]['unix'] if k in stops else time.time()) - r['unix'] for k, r in starts.items())
    if seconds >= 2 * 3600 or (starts and time.time() - min(r['unix'] for r in starts.values()) >= 3 * 3600):
        raise RuntimeError('registered compute observation line reached; preserve partial evidence')
    return dict(GPU_hours=seconds / 3600)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check', 'profile'))
    parser.add_argument('--run-root', type=Path, default=RUN_ROOT)
    parser.add_argument('--physical-gpu', type=int)
    parser.add_argument('--support-microbatch', type=int, default=4)
    parser.add_argument('--fm-microbatch', type=int, default=28)
    parser.add_argument('--frame-chunk', type=int, default=8)
    parser.add_argument('--experience-chunk', type=int, default=16)
    parser.add_argument('--native-frame-chunk', type=int, default=8)
    parser.add_argument('--slot-batch', type=int, default=8)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        print(prepare(args.run_root), flush=True)
        return
    import torch
    from ember.writer.topology import bind_current_process_to_cuda_numa
    from .runtime import Runtime
    from .profile import verify_native, profile
    if args.physical_gpu is None or os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.physical_gpu):
        raise ValueError('launch must expose exactly the admitted physical GPU')
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('actual GPU consumers require clean pushed detached source')
    git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    torch.cuda.set_device(0)
    affinity = bind_current_process_to_cuda_numa(0)
    if affinity is None:
        raise ValueError('GPU-local NUMA affinity unavailable')
    torch.set_num_threads(4)
    check_budget(args.run_root)
    attempt = f'{args.command}_{time.time_ns()}'
    start = dict(event='start', attempt=attempt, unix=float(os.environ.get('EMBER_LAUNCH_UNIX', time.time())), hostname=socket.gethostname(),
        physical_gpu=args.physical_gpu, pid=os.getpid(), code_git=git, command=args.command)
    append_jsonl(args.run_root / 'costs.jsonl', start)
    runtime, failed = None, None
    try:
        contract = read_json(args.run_root / 'run_contract.json')
        runtime = Runtime(ASSET_ROOT, 'cuda:0', frame_chunk=args.frame_chunk,
            experience_chunk=args.experience_chunk, native_frame_chunk=args.native_frame_chunk,
            cache_root=args.run_root / 'frozen_features', support_microbatch=args.support_microbatch)
        runtime.profile_root = args.run_root
        runtime.profile_deadline = time.monotonic() + max(0, (2 - check_budget(args.run_root)['GPU_hours']) * 3600)
        runtime.profile_git = git
        if args.command == 'check':
            result = verify_native(runtime, contract)
        else:
            result = profile(runtime, contract, args)
        write_json_atomic(args.run_root / f'{args.command}_completion.json',
            dict(complete=True, code_git=git, result=result, attempt=attempt, unix=time.time()))
    except BaseException:
        failed = traceback.format_exc()
        write_json_atomic(args.run_root / f'failure_{attempt}.json', dict(error=failed, code_git=git))
        raise
    finally:
        try:
            if runtime is not None:
                write_json_atomic(args.run_root / f'reading_cost_{attempt}.json', dict(code_git=git,
                    attempted_neural_full_video_reads=runtime.neural_reads,
                    attempted_neural_read_frames=runtime.neural_read_frames,
                    native_teacher_encoded_frames=runtime.native_teacher_frames,
                    native_teacher_feature_seconds=runtime.feature_seconds,
                    native_encoded_own_observations=runtime.image_encoded_observations,
                    own_observation_cache_hits=runtime.image_cache_hits))
                runtime.close()
        except BaseException:
            failed = traceback.format_exc()
            raise
        finally:
            append_jsonl(args.run_root / 'costs.jsonl', dict(event='stop', attempt=attempt,
                unix=time.time(), failed=failed is not None, duration_seconds=time.time() - start['unix']))


if __name__ == '__main__':
    main()
