"""One CLI and cost/provenance ledger for the complete Compiler batch."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import uuid

from ember.pi05_source_checkpoint import read_json, write_json_atomic


CODE_ROOT = Path(__file__).resolve().parents[3]


def frozen_git():
    from ember.pi05_eval_contract import git_state
    state = git_state(CODE_ROOT)
    if state['branch'] or state['dirty_paths']:
        raise ValueError('canonical computation requires clean detached frozen code')
    refs = subprocess.run(['git', 'branch', '-r', '--contains', state['commit']], cwd=CODE_ROOT,
                          text=True, capture_output=True, check=True).stdout.splitlines()
    if 'origin/main' not in [r.strip() for r in refs]:
        raise ValueError('canonical computation commit must already be pushed on main')
    return dict(commit=state['commit'], branch='', dirty_paths=[], pushed_ref='origin/main')


def append_cost(root, record):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'costs.jsonl').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(record, sort_keys=True) + '\n')
        handle.flush()


@contextmanager
def cost_interval(root, phase, physical_gpus, *, rank=None):
    identity = f'{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex}'
    record = dict(identity=identity, phase=phase, node=socket.gethostname(), pid=os.getpid(),
                  rank=rank, physical_gpus=list(physical_gpus), start=time.time())
    append_cost(Path(root), dict(record, event='start'))
    try:
        yield
    finally:
        append_cost(Path(root), dict(record, event='end', end=time.time()))


def cost_summary(root, *, now=None, phases=None):
    path = Path(root) / 'costs.jsonl'
    if not path.exists():
        return dict(GPU_hours=0., compute_wall_hours=0., science_elapsed_hours=0.)
    records, now = {}, time.time() if now is None else now
    with path.open() as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        lines = handle.read().splitlines()
    for line in lines:
        row = json.loads(line)
        if phases is not None and row['phase'].split('_')[0] not in phases:
            continue
        records[row['identity']] = row
    if not records:
        return dict(GPU_hours=0., compute_wall_hours=0., science_elapsed_hours=0.)
    by_device = {}
    starts, ends = [], []
    for row in records.values():
        start, end = row['start'], row.get('end', now)
        starts.append(start)
        ends.append(end)
        for device in row['physical_gpus']:
            by_device.setdefault((row['node'], device), []).append((start, end))
    seconds = 0.
    for intervals in by_device.values():
        previous = None
        for start, end in sorted(intervals):
            if previous is None:
                previous = [start, end]
            elif start <= previous[1]:
                previous[1] = max(previous[1], end)
            else:
                seconds += previous[1] - previous[0]
                previous = [start, end]
        if previous is not None:
            seconds += previous[1] - previous[0]
    science = [r for r in records.values() if r['phase'].split('_')[0] in {'collect', 'train', 'worker'}]
    science_hours = 0. if not science else (max(ends) - min(r['start'] for r in science)) / 3600
    return dict(GPU_hours=seconds / 3600, compute_wall_hours=(max(ends) - min(starts)) / 3600,
                science_elapsed_hours=science_hours, device_count=len(by_device), intervals=len(records))


def check_budget(root, *, storage=False):
    summary = cost_summary(root)
    if summary['GPU_hours'] >= 40 or summary['science_elapsed_hours'] >= 16:
        raise RuntimeError(f'actual scientific/resource boundary: {summary}')
    if storage:
        usage = int(subprocess.run(['du', '-s', '-B1', str(root)], check=True, capture_output=True,
                                  text=True).stdout.split()[0])
        if usage > 224 * 1024**3:
            raise RuntimeError(f'actual new artifact peak exceeds 224GiB: {usage} bytes')
        summary['artifact_bytes'] = usage
    return summary


def register_launch(args, runtime, context, data, start, *, topology_resume=None):
    from .contract import SCHEMA, STAGE, MT_PATH, SOURCE, TASKS36
    contract = dict(schema_version=SCHEMA, stage=STAGE, git=frozen_git(), asset_root=str(args.asset_root),
        command=sys.argv, output=str(args.output), source=runtime.source, source_configuration=SOURCE,
        MT=dict(path=str(MT_PATH), bytes=Path(MT_PATH).stat().st_size,
                use='actual fixed initial complete LoRA, frozen RMS and teacher probe'),
        task_ids=list(TASKS36), updates=[180, 360], stop_update=args.stop_update,
        logical_events_per_update=4, FM_queries_per_event=28, FM_query_episode_intervals=[7, 4],
        pool_versions={k: v['version'] for k, v in data.pools.items()},
        physical_gpus=args.physical_gpus, world_size=context.world_size,
        physical={k: getattr(args, k) for k in ('microbatch', 'frame_chunk', 'native_frame_chunk',
                    'decoder_chunk', 'experience_chunk', 'cpu_threads')},
        optimization=dict(lr=3e-5, weight_decay=0., clip=1., scheduler='constant', initialization='fresh'),
        objective='mean FM_out + .2 mean relu(FM_out - stopgrad FM_in); globally normalized /4 events',
        derivative='actual incoming/raw E/H stopped; all shared parameter/read/experience/editor/decoder live',
        condition_optimizer=False, held_offline_gradients=False, Test_opened=False, start_update=start,
        resume=None if args.resume is None else str(args.resume), topology_resume=topology_resume)
    path = args.output / 'run_contract.json'
    if path.exists():
        if args.resume is None:
            raise ValueError('fresh training cannot overwrite an existing contract')
        write_json_atomic(args.output / f'resume_{start:08d}_{os.getpid()}.json', contract)
    else:
        write_json_atomic(path, contract)
    write_json_atomic(args.output / f'events_{start:08d}_{args.stop_update:08d}.json',
        dict(schema_version=SCHEMA, sampling=data.state_dict(), events=[
            e.as_dict() for update in range(start, args.stop_update) for e in data.events(update)]))


def parser():
    from .contract import ASSET_ROOT, RUN_ROOT
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=['profile', 'train', 'collect-prepare', 'collect', 'collect-aggregate', 'prepare', 'worker', 'aggregate', 'cost'])
    result.add_argument('--asset-root', type=Path, default=ASSET_ROOT)
    result.add_argument('--run-root', type=Path, default=RUN_ROOT)
    result.add_argument('--output', type=Path)
    result.add_argument('--checkpoint', type=Path)
    result.add_argument('--resume', type=Path)
    result.add_argument('--stage', choices=['train180', 'train360', 'formal180', 'formal360'], default='formal360')
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--physical-gpu', type=int, default=0)
    result.add_argument('--physical-gpus', type=lambda s: list(map(int, s.split(','))), default=[0])
    result.add_argument('--worker-id', default=None)
    result.add_argument('--microbatch', type=int, default=28)
    result.add_argument('--pool', choices=['pool0', 'refresh180'], default='pool0')
    result.add_argument('--stop-update', type=int, choices=[180, 360], default=180)
    result.add_argument('--slot-batch', type=int, default=8)
    result.add_argument('--frame-chunk', type=int, default=16)
    result.add_argument('--native-frame-chunk', type=int, default=16)
    result.add_argument('--decoder-chunk', type=int, default=16384)
    result.add_argument('--experience-chunk', type=int, default=16)
    result.add_argument('--cpu-threads', type=int, default=8)
    result.add_argument('--allow-topology-change', action='store_true')
    result.add_argument('--recover-claims', action='store_true')
    result.add_argument('--retry-failed', action='store_true')
    return result


def main():
    args = parser().parse_args()
    args.output = args.output or args.run_root / ('training' if args.command == 'train' else args.command)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args.physical_gpu)
    os.environ['LIBERO_CONFIG_PATH'] = str(args.output / f'libero_config_{os.getpid()}')
    if args.command in {'profile', 'train', 'worker', 'collect'}:
        from ember.pi05_assets import prepare_libero_config
        from .contract import learning_environment
        os.environ['EMBER_LIBERO_ASSETS_ROOT'] = learning_environment(asset_root=args.asset_root)['libero_paths']['assets']
        prepare_libero_config(Path(os.environ['LIBERO_CONFIG_PATH']))
        import torch
        torch.set_num_threads(args.cpu_threads)
        torch.backends.cuda.matmul.allow_tf32 = True
        args.output.mkdir(parents=True, exist_ok=True)
        # A targeted owned-process stack dump distinguishes CPU/IO waits from
        # CUDA stalls without polling or changing scientific computation.
        import faulthandler, signal
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
        # Detached background shells may inherit SIGINT ignored. Restore the
        # Python handler so a controlled stop preserves unfinished real facts.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        devices = args.physical_gpus if args.command == 'train' else [args.physical_gpu]
        with cost_interval(args.run_root, args.command + '_process', devices):
            check_budget(args.run_root)
            args.code_git = frozen_git()
            if args.command == 'train':
                # EGL uses the physical device, whereas CUDA uses local rank.
                os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args.physical_gpus[int(os.environ.get('LOCAL_RANK', '0'))])
            try:
                if args.command == 'train':
                    from .learning import train
                    train(args)
                elif args.command == 'profile':
                    from .profile import profile
                    profile(args)
                elif args.command == 'collect':
                    from .collection import worker
                    worker(args)
                else:
                    from .evaluation import worker
                    worker(args)
            except BaseException as error:
                write_json_atomic(args.output / f'failure_{os.getpid()}.json',
                    dict(command=sys.argv, error=repr(error), traceback=traceback.format_exc(), cost=cost_summary(args.run_root)))
                raise
    elif args.command == 'collect-prepare':
        from .collection import prepare
        print(prepare(args.output, args.pool, checkpoint=args.checkpoint, asset_root=args.asset_root,
                      code_git=frozen_git(), recover_claims=args.recover_claims, retry_failed=args.retry_failed))
    elif args.command == 'collect-aggregate':
        from .collection import aggregate
        result = aggregate(args.output)
        print(dict(pool=result['pool'], conditions=len(result['conditions']), complete=True))
    elif args.command == 'prepare':
        from .evaluation import prepare
        print(prepare(args.output, args.checkpoint, args.stage, asset_root=args.asset_root,
                      recover_claims=args.recover_claims, retry_failed=args.retry_failed))
    elif args.command == 'aggregate':
        from .evaluation import aggregate
        print(json.dumps(aggregate(args.output)))
    else:
        print(json.dumps(check_budget(args.run_root, storage=True)))


if __name__ == '__main__':
    main()
