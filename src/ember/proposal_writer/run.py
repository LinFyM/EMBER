"""Canonical fixed-G selector training, pair collection and strict evaluation."""
from __future__ import annotations

import argparse
import json
import os
import math
from pathlib import Path
import subprocess

import torch
from safetensors.torch import load_file, save_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic, git_state
from ember.pi05_source_setup import initialize_distributed
from ember.writer.practice import EpisodeRunner

from .contract import (RUN_ROOT, ASSET_ROOT, TASKS, SOURCE, MT_PATH, OPTIMIZER, SCHEMA,
                       SELECTOR_SCHEMA, CHECKPOINT_SCHEMA, G_OBJECTIVE, materialize_panel, seed, parameter_ref)
from .runtime import Runtime
from .path import save_history


def task_panel(root, task):
    return next(row for row in materialize_panel(root)['tasks'] if row['task_id'] == task)


def physical_value_for_rank(variable, default, context):
    """Consume an explicit physical allocation in CUDA-visible rank order."""
    chunks=os.environ.get(variable)
    values=[int(value) for value in chunks.split(',')] if chunks is not None else [int(default)]*context.world_size
    if len(values)!=context.world_size or any(value<1 for value in values):
        raise ValueError(f'{variable} must be positive and match the actual world size')
    return values[context.local_rank]


def frame_chunk_for_rank(default, context):
    return physical_value_for_rank('EMBER_META_FRAME_CHUNKS',default,context)


def create_runtime(args, context, environment):
    physical=read_json(args.root/'run_contract.json').get('physical_execution',{})
    return Runtime(context.device,asset_root=args.asset_root,root=args.root,
        frame_chunk=frame_chunk_for_rank(args.frame_chunk,context),
        response_microbatch=physical.get('response_microbatch',16),
        task_ids=tuple(t['global_task_id'] for t in environment['tasks']),task_role=environment['role'])


def save_state(path, state):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(path))



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','formal-prepare','formal','student-data','local','RL'])
    parser.add_argument('--root',type=Path,default=RUN_ROOT)
    parser.add_argument('--asset-root',type=Path,default=ASSET_ROOT)
    parser.add_argument('--task',type=int,default=0)
    parser.add_argument('--stop',type=int)
    parser.add_argument('--reuse-root',type=Path,help='immutable complete previous E/MT/bootstrap/scale root')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--condition-ordinal',type=int)
    parser.add_argument('--worker-id',help='persistent formal dynamic queue worker identity')
    parser.add_argument('--condition-batch',type=int,help='measured physical independent condition group')
    parser.add_argument('--max-groups',type=int,help='optional physical worker exit after complete groups')
    parser.add_argument('--video-arm',choices=['correct','other','wrong'],default='correct')
    parser.add_argument('--microbatch',type=int,default=28)
    parser.add_argument('--frame-chunk',type=int,default=16)
    parser.add_argument('--slots',type=int,default=8)
    parser.add_argument('--physical-gpu',type=int)
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--shard-count',type=int,default=1)
    args=parser.parse_args()
    if args.command=='prepare':
        from .full_writer import prepare
        result=prepare(args);print(json.dumps(dict(root=str(args.root),updates=result['updates'])));return
    if args.command=='formal-prepare':
        from .evaluation import prepare_formal
        result=prepare_formal(args);print(json.dumps(dict(formal_rows=len(result['rows']),held_offline_labels_read=False)));return
    identity=git_state()
    if identity['dirty_paths'] or identity['branch']:
        raise ValueError('formal consumers require a clean pushed detached frozen worktree')
    subprocess.run(['git','merge-base','--is-ancestor',identity['commit'],'origin/main'],check=True)
    write_json_atomic(args.root/'code'/f'{args.command}_{identity["commit"]}.json',identity)
    shared=args.command in ('local','RL')
    if shared:
        args.physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[int(os.environ.get('LOCAL_RANK','0'))])
        if args.command not in read_json(args.root/'run_contract.json')['updates']:
            raise ValueError('shared learning update window must be registered before formal launch')
    if args.physical_gpu is None:parser.error('GPU consumer requires its actual physical GPU')
    context=initialize_distributed(require_numa=True,defer_process_group=True)
    args.slots=physical_value_for_rank('EMBER_EPISODE_SLOTS',args.slots,context)
    torch.set_num_threads(8)
    environment=read_json(args.root/'formal_panel.json')['environment'] if args.command=='formal' else read_json(args.root/'run_contract.json')['environment']
    if args.command=='formal' and (not args.worker_id or not args.condition_batch or args.condition_batch<1):
        parser.error('formal needs its persistent queue worker identity and measured condition batch')
    resident_student=args.command=='student-data' and bool(args.worker_id)
    if resident_student and (not args.condition_batch or args.condition_batch<1):
        parser.error('resident student needs its physical condition batch')
    if resident_student and args.max_groups is not None and args.max_groups<1:
        parser.error('physical group limit must be positive')
    if not shared and args.command!='formal' and not resident_student and args.task not in [t['global_task_id'] for t in environment['tasks']]:raise ValueError('task outside registered consumer role')
    runtime=create_runtime(args,context,environment)
    consumer=args.worker_id or (str(args.stop) if shared else f'task{args.task}')
    write_json_atomic(args.root/'code'/f'{args.command}_{consumer}_{identity["commit"]}_rank{context.local_rank:02d}_execution.json',
        dict(frame_chunk=runtime.frame_chunk,local_rank=context.local_rank,world_size=context.world_size,
            response_microbatch=runtime.response_microbatch,physical_gpu=args.physical_gpu,episode_slots=args.slots,
            logical_condition_batch=4 if shared else None))
    runner=None
    try:
        if args.command=='local':
            from .pipeline import run
            run(args,runtime,None,environment,context);return
        os.environ.update(EMBER_LIBERO_ASSETS_ROOT=environment['libero_paths']['assets'],MUJOCO_GL='egl',
            PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(args.physical_gpu))
        from ember.pi05_assets import prepare_libero_config,configure_libero_runtime_assets
        prepare_libero_config(args.root/'libero_config')
        configure_libero_runtime_assets(Path(environment['libero_paths']['assets']))
        runner=EpisodeRunner(runtime,environment,args.physical_gpu,slots=args.slots)
        if args.command=='formal':
            from .evaluation import formal_worker
            formal_worker(args,runtime,runner)
        elif resident_student:
            from .evaluation import student_worker
            student_worker(args,runtime,runner,environment)
        else:
            from .pipeline import run
            run(args,runtime,runner,environment,context)
    finally:
        if runner is not None:runner.close()
        runtime.close()


if __name__=='__main__':main()
