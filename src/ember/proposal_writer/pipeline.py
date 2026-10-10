"""Registered stage consumers; one bounded refresh and fixed reporting panels."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_source_setup import initialize_deferred_process_group
from ember.writer.practice import History

from .contract import TASKS, OPTIMIZER, EXPERT_ROOT, seed
from .learning import (Baseline, train_cfm, train_local, on_policy_update, checkpoint,
                       assigned_tasks, resume_checkpoint)
from .path import (compile_condition, evaluate_fixed, history_prefix, save_history, load_history,
                   restore_compilation)
from .teacher import functional_labels


def load_version(runtime, directory, *, fixed_psi=True):
    state = torch.load(Path(directory) / 'state.pt', map_location='cpu', weights_only=False)
    runtime.generator.load_state_dict(state['psi']); runtime.actor.load_state_dict(state['theta'])
    runtime.version = state['psi_version']
    if fixed_psi:
        runtime.freeze_psi()
    return state


def panels(root):
    return {r['task_id']: r for r in read_json(Path(root) / 'panel.json')['tasks']}


def task_spec(contract, task):
    return next(r for r in contract['tasks'] if r['global_task_id'] == task)


def refresh(args, runtime, runner, contract):
    state = load_version(runtime, args.checkpoint)
    if state['stage'] != 'G' or state['next_update'] != 320:
        raise ValueError('refresh uses the fixed G320 version once')
    task, panel = task_spec(contract, args.task), panels(args.root)[args.task]
    destination = Path(args.root) / 'refresh' / f'task_{args.task:04d}'
    if (destination / 'path.json').exists():
        path = restore_compilation(destination, runtime)
    else:
        path = compile_condition(runtime, runner, task, panel, panel['policy_videos'][0],
            identity=f'refresh_{args.task:04d}', path_ordinal=0, noise_root=seed(9, args.task),
            uniform=True, snapshots=True)
        path.save(destination)
    eligible = [r for r in path.boundaries if r['actual_parameter_ref'].startswith('G_') and
                not r['row'].get('model_failure')]
    if not eligible:
        write_json_atomic(destination / 'refresh_event.json', dict(missing_label=True,
            reason='no actual tried valid G proposal; no MT-renaming or success-based replacement'))
        return
    first = eligible[0]; parent_ref = first['actual_parameter_ref']
    event_root = Path(args.root) / 'events' / f'task_{args.task:04d}_event_02'
    if (event_root / 'event.json').exists():
        return
    event_root.mkdir(parents=True, exist_ok=True)
    parent = path.states[parent_ref]
    save_file(parent, str(event_root / 'parent.safetensors'))
    history = history_prefix(path.history, first['records'], first['episodes'])
    save_history(event_root / 'history.pt.gz', history)
    event = dict(event_id=event_root.name, task_id=args.task, event_ordinal=2,
        teacher_demo=panel['policy_videos'][0], parent_ref=parent_ref, parent=str(event_root / 'parent.safetensors'),
        role='nonheld_shared_training', history_kind='first_actual_valid_G320_candidate_and_actual_H',
        selection_states=panel['selection_states'], audit_states=panel['audit_states'], rng_roots=panel['rng_roots'],
        history_path=str(event_root / 'history.pt.gz'), labels_path=str(event_root / 'functional_labels.pt'),
        candidate_selected_by_chronology=True, outcome_used_for_selection=False, generator_version=320)
    expert = load_file(str(EXPERT_ROOT / f'task_{args.task:04d}_u480/lora.safetensors'), device=str(runtime.device))
    labels = functional_labels(runtime, runner, event, history, task, expert)
    torch.save(labels, event_root / 'functional_labels.pt')
    support = runtime.responses({parent_ref: parent, 'MT300': runtime.mt}, history, task['language'])
    difference = sum(float((parent[k] - runtime.mt[k].cpu().float()).square().sum()) for k in parent) ** .5
    native_RMSE = float((support[parent_ref][:, :35] - support['MT300'][:, :35]).square().mean().sqrt())
    event.update(rec_labels=len(labels['rec']), keep_labels=len(labels['keep']),
        recovery_attempts=len(labels['recovery_validations']), parent_parameter_change_L2=difference,
        parent_native_first5_RMSE_relative_MT=native_RMSE, complete=True)
    write_json_atomic(event_root / 'event.json', event)
    write_json_atomic(destination / 'refresh_event.json', event)


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
            demo = panel['policy_videos'][int(np.random.default_rng(seed(43, task_id, update)).integers(2))]
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
        checkpoint(args.root, 'RL', update+1, runtime, optimizer, context,
            sampler=dict(next_update=update+1, task_order=list(TASKS), new_paths_per_task=1,
                         query_pool_cycle=4, condition_weight=.25), baseline=baseline,
            baseline_optimizer=baseline_optimizer)
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
            path.save(destination, retain_U=arm == 'correct' and args.task in (12,32) and ordinal == 0)
        rows = evaluate_fixed(runtime, runner, path, task, panel['report_states'], seed(80, args.task, ordinal))
        write_json_atomic(destination / 'query.json', dict(rows=rows, finite_train_diagnostic=True,
            teacher_video_reused_six_times=True, single_fixed_task_LoRA=True, final_feedback_not_used_for_selection=True))
        if arm == 'correct' and args.task in (12,32) and ordinal == 0:
            U = evaluate_fixed(runtime, runner, path, task, panel['report_states'], seed(80, args.task, ordinal),
                               references=path.valid)
            write_json_atomic(destination / 'U_diagnostic.json', dict(rows=U, complete_U=True,
                diagnosis_after_final_locked=True, diagnosis_does_not_change_selected=True))
        write_json_atomic(destination / 'complete.json', dict(complete=True, version=version, input_arm=arm,
            code_stage='fixed report', condition=path.identity, query_rows=len(rows), environment_steps=path.environment_steps))


def run(args, runtime, runner, contract, context):
    if args.command == 'G':
        train_cfm(runtime, context, args.root, stop=args.stop, resume=args.resume)
    elif args.command == 'refresh': refresh(args, runtime, runner, contract)
    elif args.command == 'local-data': local_data(args, runtime, runner, contract)
    elif args.command == 'local':
        load_version(runtime, args.checkpoint)
        baseline = Baseline().to(runtime.device)
        train_local(runtime, context, args.root, local_examples(args.root, runtime),
                    stop=128, resume=args.resume, baseline=baseline)
    elif args.command == 'RL': rl(args, runtime, runner, contract, context)
    elif args.command == 'report': report(args, runtime, runner, contract)
