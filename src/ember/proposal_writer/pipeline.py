"""Registered multistart G480, frozen-kernel decision training and fixed reports."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import initialize_deferred_process_group
from ember.writer.practice import History

from .contract import TASKS, OPTIMIZER, seed, SCHEMA
from .learning import (Baseline, train_cfm, train_local, on_policy_update, checkpoint,
                       assigned_tasks, resume_checkpoint)
from .path import (compile_condition, evaluate_fixed, history_prefix, save_history, load_history,
                   restore_compilation,retire_consumed_parameters)
from .teacher import functional_labels


def load_version(runtime, directory, *, fixed_psi=True):
    state = torch.load(Path(directory) / 'state.pt', map_location='cpu', weights_only=False)
    if state['schema_version'] != SCHEMA:
        raise ValueError('checkpoint belongs to a retired parameter family')
    runtime.generator.load_state_dict(state['psi']); runtime.actor.load_state_dict(state['theta'])
    runtime.version=state['psi_version']
    runtime.fixed_kernel_identity=dict(checkpoint=str(Path(directory).resolve()),psi_version=runtime.version,
        checkpoint_stage=state['stage'],checkpoint_update=state['next_update'])
    if fixed_psi:
        runtime.freeze_psi()
    return state


def panels(root):
    return {r['task_id']: r for r in read_json(Path(root) / 'panel.json')['tasks']}


def task_spec(contract, task):
    return next(r for r in contract['tasks'] if r['global_task_id'] == task)



def local_data(args, runtime, runner, contract):
    state = load_version(runtime, args.checkpoint)
    if state['stage'] != 'G' or state['next_update'] != 480:
        raise ValueError('local uses final fixed G480 kernel')
    task, panel = task_spec(contract, args.task), panels(args.root)[args.task]
    for video_index, demo in enumerate(panel['policy_videos']):
        for repeat in range(2):
            ordinal = video_index * 2 + repeat
            destination = Path(args.root) / 'local_data' / f'task_{args.task:04d}_condition_{ordinal:02d}'
            if (destination / 'local_labels.json').exists():
                continue
            if (destination / 'path.json').exists():
                path = restore_compilation(destination, runtime)
            else:
                path = compile_condition(runtime, runner, task, panel, demo,
                    identity=destination.name, path_ordinal=ordinal, noise_root=seed(6, args.task, ordinal), uniform=True)
                path.save(destination)
            prefixes, rows = [], []
            for i, boundary in enumerate(path.boundaries):
                P = boundary['practiced']
                raw = evaluate_fixed(runtime, runner, path, task, panel['local_query_states'],
                                     seed(60, args.task, ordinal, i), references=P)
                rows.extend([{**row, 'prefix_index': i} for row in raw])
                reward = {name: sum(int(r['success']) for r in raw if r['reference'] == name) / 4 for name in P}
                prefixes.append(dict(records=boundary['records'], episodes=boundary['episodes'], practiced=P,
                    budget=[(1024-boundary['environment_steps'])/1024,
                            (32-boundary['attempts'])/32, boundary['episodes']/100, len(P)/33], reward=reward))
            write_json_atomic(destination / 'local_labels.json', dict(task=args.task, demo=demo,
                condition=path.identity, prefixes=prefixes, raw_rows=rows, complete=True,
                query_excluded_from_H=True, uniform_behavior_not_on_policy=True))


def local_examples(root, runtime):
    examples = []
    for file in sorted((Path(root) / 'local_data').glob('*/local_labels.json')):
        record = read_json(file)
        examples.append(dict(task=record['task'], prefixes=record['prefixes'],
                             path=restore_compilation(file.parent, runtime)))
    if len(examples) != 16 or any(sum(x['task'] == task for x in examples) != 4 for task in TASKS):
        raise ValueError('local supervision requires all 16 registered condition paths')
    return examples


def rl(args, runtime, runner, contract, context):
    state = load_version(runtime, args.checkpoint)
    if state['stage'] != 'local' or state['next_update'] != 128:
        raise ValueError('RL starts from π-local128; no best-node selection')
    baseline = Baseline().to(runtime.device)
    baseline.load_state_dict(state['baseline'])
    optimizer = torch.optim.AdamW(runtime.actor.parameters(), **OPTIMIZER)
    optimizer.load_state_dict(state['optimizer'])
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), **OPTIMIZER)
    initialize_deferred_process_group(context, rendezvous_root=args.root / 'rendezvous')
    start = resume_checkpoint(args.resume, 'RL', runtime, optimizer, context, baseline=baseline,
                              baseline_optimizer=baseline_optimizer) if args.resume else 0
    panel_by_task = panels(args.root)
    for update in range(start, 16):
        paths, rewards, controls = [], [], []
        runtime.actor.eval(); baseline.eval()
        for task_id in assigned_tasks(context):
            task, panel = task_spec(contract, task_id), panel_by_task[task_id]
            demo = panel['policy_videos'][update % 2]
            identity = f'RL_{update+1:03d}_task_{task_id:04d}'
            destination = args.root / 'RL_paths' / identity
            # New paths only at the current fixed batch θ; no stale-path replay pool.
            path = compile_condition(runtime, runner, task, panel, demo, identity=identity,
                path_ordinal=update, noise_root=seed(7, task_id, update), baseline=baseline)
            query_state = panel['rl_query_states'][update % 4]
            rows = evaluate_fixed(runtime, runner, path, task, [query_state], seed(70, task_id, update))
            rewards.append(float(next(r['success'] for r in rows if r['arm'] == 'selected')))
            controls.append(float(next(r['success'] for r in rows if r['arm'] == 'MT')))
            path.save(destination)
            write_json_atomic(destination / 'query.json', dict(rows=rows, selected_locked_before_query=True,
                current_theta_update=update, PSI_checkpoint=480, condition_batch=update+1))
            paths.append(path)
        on_policy_update(runtime, context, args.root, update+1, paths, rewards, controls,
                         optimizer, baseline, baseline_optimizer)
        saved=checkpoint(args.root,'RL',update+1,runtime,optimizer,context,
            sampler=dict(next_update=update+1, task_order=list(TASKS), new_paths_per_task=1,
                         query_pool_cycle=4, condition_weight=.25), baseline=baseline,
            baseline_optimizer=baseline_optimizer)
        for path in paths:
            retire_consumed_parameters(args.root/'RL_paths'/path.identity,saved/'manifest.json',
                {k:v.cpu() for k,v in runtime.mt.items()},runtime.prox_scales)
        del paths


def report(args, runtime, runner, contract):
    state = load_version(runtime, args.checkpoint)
    version = 'local128' if state['stage'] == 'local' and state['next_update'] == 128 else 'RL16'
    if version == 'RL16' and (state['stage'] != 'RL' or state['next_update'] != 16):
        raise ValueError('report nodes are fixed π-local128 and π-RL16')
    task, panel = task_spec(contract, args.task), panels(args.root)[args.task]
    runtime.actor.eval()
    for ordinal, demo in enumerate(panel['report_videos']):
        arm = args.video_arm
        if arm != 'correct' and (version != 'RL16' or args.task not in (12, 32) or ordinal != 0):
            continue
        video_task = None
        if arm == 'other': demo = panel['other_video']
        if arm == 'wrong':
            video_task = 32 if args.task == 12 else 12
            demo = panels(args.root)[video_task]['report_videos'][0]
        destination = args.root / 'reports' / version / arm / f'task_{args.task:04d}_condition_{ordinal:02d}'
        if (destination / 'complete.json').exists():
            continue
        if (destination / 'path.json').exists():
            path = restore_compilation(destination, runtime)
        else:
            path = compile_condition(runtime, runner, task, panel, demo, identity=destination.name,
                path_ordinal=ordinal, noise_root=seed(8, args.task, ordinal), video_task=video_task)
            path.save(destination, retain_U=version == 'RL16' and arm == 'correct' and args.task in (12,32) and ordinal == 0)
        rows = evaluate_fixed(runtime, runner, path, task, panel['report_states'], seed(80, args.task, ordinal))
        write_json_atomic(destination / 'query.json', dict(rows=rows, finite_train_diagnostic=True,
            teacher_video_reused_six_times=True, single_fixed_task_LoRA=True, final_feedback_not_used_for_selection=True))
        if version == 'RL16' and arm == 'correct' and args.task in (12,32) and ordinal == 0:
            U = evaluate_fixed(runtime, runner, path, task, panel['report_states'], seed(80, args.task, ordinal),
                               references=path.valid)
            write_json_atomic(destination / 'U_diagnostic.json', dict(rows=U, complete_U=True,
                diagnosis_after_final_locked=True, diagnosis_does_not_change_selected=True))
        write_json_atomic(destination / 'complete.json',dict(complete=True,version=version,input_arm=arm,
            code_stage='fixed report',condition=path.identity,query_rows=len(rows),environment_steps=path.environment_steps))
        retire_consumed_parameters(destination,destination/'complete.json',{k:v.cpu() for k,v in runtime.mt.items()},runtime.prox_scales)


def run(args, runtime, runner, contract, context):
    if args.command == 'G':
        train_cfm(runtime, context, args.root, stop=args.stop, resume=args.resume)
    elif args.command == 'local-data': local_data(args, runtime, runner, contract)
    elif args.command == 'local':
        load_version(runtime, args.checkpoint)
        baseline = Baseline().to(runtime.device)
        train_local(runtime,context,args.root,local_examples(args.root,runtime),
            stop=128,resume=args.resume,baseline=baseline)
        if context.is_main:
            proof=args.root/'checkpoints/local/update_00000128/manifest.json'
            for file in sorted((args.root/'local_data').glob('*/path.json')):
                retire_consumed_parameters(file.parent,proof,{k:v.cpu() for k,v in runtime.mt.items()},runtime.prox_scales)
    elif args.command == 'RL': rl(args, runtime, runner, contract, context)
    elif args.command == 'report': report(args, runtime, runner, contract)
