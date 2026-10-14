"""Canonical pure-image compiler cache, matched learning and endpoint readouts."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import torch

from ember.pi05_source_checkpoint import read_json, write_json_atomic, git_state
from ember.pi05_source_setup import initialize_distributed
from ember.writer.practice import EpisodeRunner
from .contract import ASSET_ROOT
from .runtime import Runtime


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['image-cache','image-train','image-readout'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--asset-root',type=Path,default=ASSET_ROOT)
    parser.add_argument('--compiler-arm',choices=['V','VL','both'],default='both')
    parser.add_argument('--stop',type=int)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--worker-id')
    parser.add_argument('--readout-role',choices=['seen','validation'],default='seen')
    parser.add_argument('--frame-chunk',type=int,default=32)
    parser.add_argument('--slots',type=int,default=16)
    parser.add_argument('--checkpoint-layers',action='store_true')
    parser.add_argument('--physical-gpu',type=int,required=True)
    return parser.parse_args()


def main():
    args=parse_args();identity=git_state()
    if identity['dirty_paths'] or identity['branch']:
        raise ValueError('formal image consumers require a clean pushed detached frozen worktree')
    subprocess.run(['git','merge-base','--is-ancestor',identity['commit'],'origin/main'],check=True)
    context=initialize_distributed(require_numa=True,defer_process_group=True)
    torch.set_num_threads(8)
    write_json_atomic(args.root/'code'/f'{args.command}_{args.compiler_arm}_{args.worker_id or context.rank}_{identity["commit"]}.json',
        dict(**identity,world=context.world_size,rank=context.rank,frame_chunk=args.frame_chunk,slots=args.slots,
            actual_physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[context.local_rank])))
    if args.command=='image-train':
        if args.compiler_arm=='both':raise ValueError('each independent optimizer has one registered arm')
        from .image_learning import learn
        learn(args,context);return
    if not args.worker_id:raise ValueError('persistent queue requires a worker identity')
    run=read_json(args.root/'run_contract.json')
    held=args.command=='image-readout' and args.readout_role=='validation'
    environment=run['validation_environment' if held else 'environment']
    runtime=Runtime(context.device,asset_root=args.asset_root,root=args.root,frame_chunk=args.frame_chunk,
        task_ids=tuple(t['global_task_id'] for t in environment['tasks']),task_role=environment['role'])
    runtime.policy.eval().requires_grad_(False)
    runtime.episode_event_log=os.environ.get('EMBER_EPISODE_EVENT_LOG')
    runner=None
    try:
        if args.command=='image-cache':
            from .image_data import feature_cache
            feature_cache(args,runtime);return
        os.environ.update(EMBER_LIBERO_ASSETS_ROOT=environment['libero_paths']['assets'],MUJOCO_GL='egl',
            PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(args.physical_gpu))
        from ember.pi05_assets import prepare_libero_config,configure_libero_runtime_assets
        prepare_libero_config(args.root/'libero_config');configure_libero_runtime_assets(Path(environment['libero_paths']['assets']))
        runner=EpisodeRunner(runtime,environment,args.physical_gpu,slots=args.slots)
        from .image_evaluation import readout_worker
        readout_worker(args,runtime,runner)
    finally:
        if runner is not None:runner.close()
        runtime.close()


if __name__=='__main__':main()
