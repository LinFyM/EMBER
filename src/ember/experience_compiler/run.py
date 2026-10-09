"""Single canonical functional-revision learning, refresh and readout entrypoint."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_contract import append_jsonl
from .collection import ROOT, prepare_learning
from .contract import ASSET_ROOT, SEED


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('prepare', 'train', 'collect', 'evaluate', 'aggregate'))
    result.add_argument('--run-root', type=Path, default=ROOT)
    result.add_argument('--physical-gpu', type=int)
    result.add_argument('--phase', type=int, choices=(1, 2), default=1)
    result.add_argument('--stop-update', type=int, choices=(90, 180, 270, 360), default=180)
    result.add_argument('--resume-checkpoint', type=Path)
    result.add_argument('--profile-only', action='store_true')
    result.add_argument('--checkpoint', type=Path)
    result.add_argument('--evaluation', choices=('train180', 'train360', 'formal360'))
    result.add_argument('--worker-id')
    result.add_argument('--attempt')
    result.add_argument('--support-microbatch', type=int, default=64)
    result.add_argument('--adjoint-microbatch', type=int, default=64)
    result.add_argument('--fm-microbatch', type=int, default=56)
    result.add_argument('--frame-chunk', type=int, default=8)
    result.add_argument('--experience-chunk', type=int, default=16)
    result.add_argument('--native-frame-chunk', type=int, default=8)
    result.add_argument('--slot-batch', type=int, default=16)
    result.add_argument('--recover-claims', action='store_true')
    result.add_argument('--retry-failed', action='store_true')
    return result


def prepare(args):
    from .collection import prepare_collection
    from .evaluation import prepare_evaluation
    root = args.run_root
    result = prepare_learning(root)
    prepare_collection(root, recover_claims=args.recover_claims, retry_failed=args.retry_failed)
    for kind, update in (('train180', 180), ('train360', 360), ('formal360', 360)):
        prepare_evaluation(root, root / f'training/checkpoints/step_{update:08d}', kind,
                           recover_claims=args.recover_claims, retry_failed=args.retry_failed)
    return result


def aggregate(args):
    if args.evaluation:
        from .evaluation import aggregate_evaluation
        return aggregate_evaluation(args.run_root, args.evaluation)
    from .collection import aggregate_collection
    return aggregate_collection(args.run_root)


def _environment(args, context):
    if args.command == 'train':
        visible = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
        args.physical_gpu = visible[context.local_rank]
    elif os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.physical_gpu):
        raise ValueError('independent worker must expose exactly its admitted physical GPU')
    if args.command in {'collect', 'evaluate'}:
        from ember.pi05_assets import prepare_libero_config
        contract = read_json(args.run_root / ('pools/refresh180/collection_contract.json'
            if args.command == 'collect' else f'evaluation/{args.evaluation}/evaluation_contract.json'))
        environment = contract['environment_contract']
        os.environ.update(EMBER_LIBERO_ASSETS_ROOT=environment['libero_paths']['assets'],
            MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID=str(args.physical_gpu))
        prepare_libero_config(args.run_root / 'libero_config')


def _GPU_command(args):
    import random
    import numpy as np
    import torch
    from ember.pi05_source_setup import initialize_distributed
    from .runtime import Runtime
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('formal source must be clean and pushed before detached launch')
    if subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip():
        raise ValueError('formal computation must use a frozen detached worktree')
    args.code_git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if args.command == 'train':
        visible = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
        args.physical_gpu = visible[int(os.environ.get('LOCAL_RANK', 0))]
    args.worker_id = args.worker_id or f'{socket.gethostname()}_gpu{args.physical_gpu}_pid{os.getpid()}'
    attempt = f'{args.command}_{args.worker_id}_{time.time_ns()}'
    start = dict(event='start', attempt=attempt, unix=float(os.environ.get('EMBER_LAUNCH_UNIX', time.time())),
        hostname=socket.gethostname(), physical_gpu=args.physical_gpu, pid=os.getpid(), code_git=args.code_git,
        command=args.command, phase=args.phase if args.command == 'train' else None,
        evaluation=args.evaluation, profile_only=args.profile_only,
        rank=int(os.environ.get('RANK', 0)), world_size=int(os.environ.get('WORLD_SIZE', 1)))
    append_jsonl(args.run_root / 'costs.jsonl', start)
    runtime, failed = None, None
    try:
        context = initialize_distributed(require_numa=True, defer_process_group=True)
        _environment(args, context)
        torch.set_num_threads(4)
        # Shared fresh phi is identical across physical topologies; rank RNG
        # streams are introduced/restored by training after model initialization.
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        runtime = Runtime(ASSET_ROOT, context.device, frame_chunk=args.frame_chunk,
            experience_chunk=args.experience_chunk, native_frame_chunk=args.native_frame_chunk,
            cache_root=args.run_root / 'frozen_features', feature_cache_bytes=32 * 1024**3,
            support_microbatch=args.support_microbatch)
        if args.command == 'train':
            from .training import run_training
            result = run_training(runtime, args, context)
        elif args.command == 'collect':
            from .collection import collect
            result = collect(runtime, args)
        else:
            from .evaluation import run_evaluation
            result = run_evaluation(runtime, args)
        write_json_atomic(args.run_root / 'attempts' / f'{attempt}.json',
            dict(complete=True, code_git=args.code_git, result=result, unix=time.time()))
    except BaseException:
        failed = traceback.format_exc()
        write_json_atomic(args.run_root / 'failures' / f'{attempt}.json',
            dict(error=failed, code_git=args.code_git, args={key: str(value) for key, value in vars(args).items()}))
        raise
    finally:
        try:
            if runtime is not None:
                write_json_atomic(args.run_root / 'reading_cost' / f'{attempt}.json', dict(code_git=args.code_git,
                    attempted_neural_full_video_reads=runtime.neural_reads,
                    attempted_neural_read_frames=runtime.neural_read_frames,
                    native_teacher_encoded_frames=runtime.native_teacher_frames,
                    native_teacher_feature_seconds=runtime.feature_seconds,
                    native_encoded_own_observations=runtime.image_encoded_observations,
                    own_observation_cache_hits=runtime.image_cache_hits,
                    record_wait_seconds=runtime.io.wait_seconds))
                runtime.close()
        except BaseException:
            failed = traceback.format_exc()
            raise
        finally:
            append_jsonl(args.run_root / 'costs.jsonl', dict(event='stop', attempt=attempt,
                unix=time.time(), failed=failed is not None, duration_seconds=time.time() - start['unix']))
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'prepare':
        print(prepare(args), flush=True)
    elif args.command == 'aggregate':
        print(aggregate(args), flush=True)
    else:
        if args.command == 'evaluate' and (args.checkpoint is None or args.evaluation is None):
            raise ValueError('readout must identify one registered checkpoint and explicit panel')
        _GPU_command(args)


if __name__ == '__main__':
    main()
