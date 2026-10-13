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



def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','formal-prepare','formal','student-data','local','RL',
        'function-prepare','function-train','function-replay','function-readout'])
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
    parser.add_argument('--arm',choices=['F_only','full_G'])
    parser.add_argument('--input-audit',type=Path)
    parser.add_argument('--held-prefix-audit',type=Path)
    parser.add_argument('--scientific-contract',type=Path)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--profile-microbatch',type=int,default=24)
    parser.add_argument('--readout-role',choices=['seen','validation'],default='validation')
    return parser.parse_args()


def prepare_cpu(args):
    if args.command=='function-prepare':
        if not all((args.input_audit,args.held_prefix_audit,args.scientific_contract)):
            raise ValueError('function prepare requires frozen source, held-prefix audits and full scientific contract')
        from .function_data import prepare
        from .function_evaluation import prepare_panels
        result=prepare(args);prepare_panels(args)
        print(json.dumps(dict(root=str(args.root),conditions=288,source=result['source_checkpoint'])));return True
    if args.command=='prepare':
        from .full_writer import prepare
        result=prepare(args);print(json.dumps(dict(root=str(args.root),updates=result['updates'])));return True
    if args.command=='formal-prepare':
        from .evaluation import prepare_formal
        result=prepare_formal(args);print(json.dumps(dict(formal_rows=len(result['rows']),held_offline_labels_read=False)));return True
    return False


def consumer_environment(args,context):
    shared=args.command in ('local','RL','function-train')
    if shared:
        args.physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[context.local_rank])
        if args.command in ('local','RL') and args.command not in read_json(args.root/'run_contract.json')['updates']:
            raise ValueError('shared learning update window must be registered before formal launch')
    if args.command in ('function-train','function-readout') and args.arm is None:
        raise ValueError('native G consumer requires its registered parameter arm')
    if args.command=='function-readout':
        environment=read_json(args.root/'readout_panels'/f'{args.readout_role}.json')['environment']
        if not args.worker_id:raise ValueError('first-G readout requires a persistent queue worker identity')
    else:
        environment=read_json(args.root/'formal_panel.json')['environment'] if args.command=='formal' else read_json(args.root/'run_contract.json')['environment']
    if args.command=='formal' and (not args.worker_id or not args.condition_batch or args.condition_batch<1):
        raise ValueError('formal needs its persistent queue worker identity and measured condition batch')
    resident_student=args.command=='student-data' and bool(args.worker_id)
    if resident_student and (not args.condition_batch or args.condition_batch<1):
        raise ValueError('resident student needs its physical condition batch')
    if resident_student and args.max_groups is not None and args.max_groups<1:
        raise ValueError('physical group limit must be positive')
    if not shared and args.command not in ('formal','function-replay','function-readout') and not resident_student and args.task not in [t['global_task_id'] for t in environment['tasks']]:
        raise ValueError('task outside registered consumer role')
    return environment,shared,resident_student


def consume(args,runtime,runner,environment,context,resident_student):
    if args.command=='function-train':
        from .function_learning import learn
        learn(args,runtime,context)
    elif args.command=='function-replay':
        from .function_data import replay_targets
        replay_targets(args,runtime,runner)
    elif args.command=='function-readout':
        from .function_evaluation import readout
        readout(args,runtime,runner)
    elif args.command=='formal':
        from .evaluation import formal_worker
        formal_worker(args,runtime,runner)
    elif resident_student:
        from .evaluation import student_worker
        student_worker(args,runtime,runner,environment)
    else:
        from .pipeline import run
        run(args,runtime,runner,environment,context)


def main():
    args=parse_args()
    if prepare_cpu(args):return
    identity=git_state()
    if identity['dirty_paths'] or identity['branch']:
        raise ValueError('formal consumers require a clean pushed detached frozen worktree')
    subprocess.run(['git','merge-base','--is-ancestor',identity['commit'],'origin/main'],check=True)
    write_json_atomic(args.root/'code'/f'{args.command}_{identity["commit"]}.json',identity)
    if args.physical_gpu is None:raise ValueError('GPU consumer requires its actual physical GPU')
    context=initialize_distributed(require_numa=True,defer_process_group=True)
    args.slots=physical_value_for_rank('EMBER_EPISODE_SLOTS',args.slots,context)
    torch.set_num_threads(8)
    environment,shared,resident_student=consumer_environment(args,context)
    runtime=create_runtime(args,context,environment)
    runtime.episode_event_log=os.environ.get('EMBER_EPISODE_EVENT_LOG')
    consumer=args.worker_id or (f"{args.arm}_{'profile' if args.profile else 'train'}"
        if args.command=='function-train' else str(args.stop) if shared else f'task{args.task}')
    write_json_atomic(args.root/'code'/f'{args.command}_{consumer}_{identity["commit"]}_rank{context.local_rank:02d}_execution.json',
        dict(frame_chunk=runtime.frame_chunk,local_rank=context.local_rank,world_size=context.world_size,
            response_microbatch=runtime.response_microbatch,physical_gpu=args.physical_gpu,episode_slots=args.slots,
            logical_condition_batch=4 if shared else None))
    runner=None
    try:
        if args.command in ('local','function-train'):
            consume(args,runtime,None,environment,context,resident_student);return
        os.environ.update(EMBER_LIBERO_ASSETS_ROOT=environment['libero_paths']['assets'],MUJOCO_GL='egl',
            PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(args.physical_gpu))
        from ember.pi05_assets import prepare_libero_config,configure_libero_runtime_assets
        prepare_libero_config(args.root/'libero_config')
        configure_libero_runtime_assets(Path(environment['libero_paths']['assets']))
        runner=EpisodeRunner(runtime,environment,args.physical_gpu,slots=args.slots)
        consume(args,runtime,runner,environment,context,resident_student)
    finally:
        if runner is not None:runner.close()
        runtime.close()


if __name__=='__main__':main()
