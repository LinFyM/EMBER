"""Thin entrypoint for persistent workers over the fixed90 transfer panel."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['batch', 'worker'])
    p.add_argument('--phase', required=True, choices=['pilot', 'remaining'])
    p.add_argument('--label', required=True)
    p.add_argument('--gpus')
    p.add_argument('--gpu-id', type=int)
    p.add_argument('--replicas', type=int, default=1)
    args = p.parse_args()
    from ember.pi05_eval.cross_task_transfer import ROOT, consumers, run_worker
    if (ROOT/'retired_entrypoints.json').exists() or (ROOT/'completion.json').exists():
        raise SystemExit('Finite transfer study is sealed')
    if args.command == 'worker':
        os.environ.update(MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                          MUJOCO_EGL_DEVICE_ID=str(args.gpu_id))
        run_worker(1 if args.phase=='pilot' else 3, args.label, args.gpu_id, args.phase)
        return
    gpus = [int(x) for x in args.gpus.split(',')]
    processes, logs = [], []
    try:
        for gpu in gpus:
            for replica in range(args.replicas):
                label = f'{args.label}_gpu{gpu}_r{replica}'
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                log = (ROOT/'launch'/f'{label}.log').open('xb')
                logs.append(log)
                command = [sys.executable, str(Path(__file__).resolve()), 'worker',
                           '--phase', args.phase, '--label', label, '--gpu-id', str(gpu)]
                processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
        language, _ = consumers()
        language.write(ROOT/'launch'/f'{args.label}_worker_processes.json',
            dict(gpus=gpus, replicas_per_GPU=args.replicas, pids=[process.pid for process in processes],
                 frozen_repo=str(Path(__file__).resolve().parents[1]), phase=args.phase))
        with ThreadPoolExecutor(max_workers=len(processes)) as pool:
            codes = list(pool.map(lambda process: process.wait(), processes))
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        for log in logs:
            log.close()
    if any(codes):
        raise SystemExit(f'Actual workers failed: {codes}')


if __name__ == '__main__':
    main()
