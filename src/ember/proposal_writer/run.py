"""Thin canonical commands for the registered complete-parameter pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import subprocess
import time

import torch
from safetensors.torch import load_file, save_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic, git_state
from ember.pi05_source_setup import initialize_distributed
from ember.writer.practice import EpisodeRunner, History
from ember.writer.runtime import autocast
from ember.writer.model import DirectLoRAParameters
from ember.writer.function_credit import NativeFlowPrediction, mean_velocity_loss
from ember.lora import identity_lora_state, validate_lora_state, task_lora_state_dict

from .contract import (RUN_ROOT, ASSET_ROOT, TASKS, SOURCE, MT_PATH, EXPERT_ROOT, OPTIMIZER,
                       SCHEMA, materialize_panel, environment_contract, seed)
from .runtime import Runtime
from .teacher import QueryStream, fit_teacher, functional_labels, endpoint_distribution


def task_panel(root, task):
    panel = read_json(Path(root) / 'panel.json')
    return next(row for row in panel['tasks'] if row['task_id'] == task)


def save_state(path, state):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(path))


def prepare(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    panel = materialize_panel(root)
    contract = dict(schema_version=SCHEMA, stage='pilot', design='docs/designs/video_guided_proposal_writer_20261010.md#13',
        source=SOURCE, mt_checkpoint=str(MT_PATH), training_tasks=list(TASKS), panel=str(root / 'panel.json'),
        environment=environment_contract(), seed=7, meta_seeds=[202610101, 202610102], meta_rank=32,
        meta_alpha=32, task_rank=128, task_targets=38, task_factors=76, probe_seed=1729,
        limits=dict(GPU_hours=52, added_peak_GiB=64),
        elapsed_prediction_hours=[16,24], GPUh_prediction=[45,51],
        updates=dict(teacher=[160, 480], G=[160, 320, 480], actor_local=128, actor_RL=16),
        logical_batch=dict(teacher_queries=112, G_conditions=4, actor_conditions=4),
        optimizer=OPTIMIZER, compilation=dict(environment_steps=1024, proposals_per_decision=4,
            proposal_attempts=32, parameter_euler_steps=16, response_observations=16),
        exact_resume='full parameters/optimizer/sampler/rank RNG/schema; explicit topology migrations separately recorded',
        git=git_state(), code_status='initial implementation consumers pending',
        information_wall='no T model/features; teacher labels only training endpoints; query after final lock; no held use')
    path = root / 'run_contract.json'
    if path.exists():
        original = read_json(path)
        for key in ('source', 'training_tasks', 'panel', 'meta_rank', 'updates', 'logical_batch', 'compilation'):
            if original[key] != contract[key]:
                raise ValueError('registered pilot science changed')
    else:
        write_json_atomic(path, contract)
    print(json.dumps(dict(panel=str(root / 'panel.json'), contract=str(path), tasks=len(panel['tasks']))))


def _episode_request(runtime, task, state, reference, state_id, noise_root, identity, **fields):
    return dict(task=task, state=state, parameter_ref=reference, state_id=int(state_id), noise_root=int(noise_root),
                episode_id=identity, **fields)


def collect(args, runtime, runner, contract):
    panel = task_panel(args.root, args.task)
    task = next(t for t in contract['tasks'] if t['global_task_id'] == args.task)
    from .path import save_history, load_history
    root = Path(args.root) / 'events'
    expert = {k: v.float() for k, v in load_file(str(EXPERT_ROOT / f'task_{args.task:04d}_u480/lora.safetensors'),
                                              device=str(runtime.device)).items()}
    validate_lora_state(expert, runtime.lora)
    for ordinal in (0, 1):
        destination = root / f'task_{args.task:04d}_event_{ordinal:02d}'
        if (destination / 'event.json').exists():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        event = dict(event_id=destination.name, task_id=args.task, event_ordinal=ordinal,
            teacher_demo=panel['teacher_videos'][ordinal], parent_ref='MT300', parent=str(MT_PATH),
            role='nonheld_shared_training', history_kind='empty' if ordinal == 0 else 'one_actual_MT_episode',
            selection_states=panel['selection_states'], audit_states=panel['audit_states'],
            rng_roots=panel['rng_roots'])
        history_path = destination/'history.pt.gz'
        captured = destination/'history_capture.json'
        history = (load_history(history_path) if captured.exists() else
                   History(states={'MT300': {k: v.detach().cpu() for k, v in runtime.mt.items()}}))
        if ordinal == 1 and not captured.exists():
            request = _episode_request(runtime, task, runtime.mt, 'MT300', panel['practice_states'][0],
                seed(3, args.task, ordinal), event['event_id'], remaining=task['horizon'] + 10,
                snapshots=True, uses_teaching=False)
            output = next(runner.run([request])); history = output['history']
        if not captured.exists():
            save_history(history_path,history)
            write_json_atomic(captured, dict(event_id=event['event_id'], complete=True,
                code=git_state(), episodes=history.episodes, records=len(history.records),
                actual_practice=True, labels_complete=False))
        labels = functional_labels(runtime, runner, event, history, task, expert)
        torch.save(labels, destination / 'functional_labels.pt')
        event.update(history_path=str(destination/'history.pt.gz'), labels_path=str(destination / 'functional_labels.pt'),
            rec_labels=len(labels['rec']), keep_labels=len(labels['keep']), recovery_attempts=len(labels['recovery_validations']),
            capture_components=deep_components(runtime, runner), complete=True)
        write_json_atomic(destination / 'event.json', event)
    print(json.dumps(dict(task=args.task, collected=True, components=deep_components(runtime, runner))))


def deep_components(runtime, runner):
    return dict(native=dict(runner.components), environment_steps=runner.environment_steps,
                reader=dict(runtime.components))


def fit(args, runtime):
    destination = Path(args.root) / 'events' / f'task_{args.task:04d}_event_{args.event:02d}'
    event = read_json(destination / 'event.json')
    labels = torch.load(event['labels_path'], map_location='cpu', weights_only=False)
    parent = runtime.mt if event['parent_ref'] == 'MT300' else load_file(event['parent'], device=str(runtime.device))
    completed = destination / 'teacher' / 'checkpoints' / f'update_{args.stop:08d}' / 'manifest.json'
    if completed.exists() and read_json(completed)['complete']:
        print(json.dumps(dict(event_id=event['event_id'], trained_to=args.stop, existing_complete_checkpoint=True)))
        return
    if args.resume is None:
        available = destination / 'teacher' / 'checkpoints' / 'update_00000160'
        if (available / 'manifest.json').exists() and args.stop == 480:
            args.resume = available
    fit_teacher(runtime, event, parent, labels, destination / 'teacher', microbatch=args.microbatch,
                stop=args.stop,resume=args.resume,function_microbatch=args.function_microbatch)
    print(json.dumps(dict(event_id=event['event_id'], trained_to=args.stop)))


def audit(args, runtime, runner, contract):
    panel = task_panel(args.root, args.task)
    task = next(t for t in contract['tasks'] if t['global_task_id'] == args.task)
    destination = Path(args.root) / 'events' / f'task_{args.task:04d}_event_{args.event:02d}'
    event = read_json(destination / 'event.json')
    states = {'parent': runtime.mt if event['parent_ref'] == 'MT300' else
              load_file(event['parent'], device=str(runtime.device))}
    for update in (160, 480):
        p = destination / 'teacher' / 'checkpoints' / f'update_{update:08d}' / 'lora.safetensors'
        if not p.exists():
            raise ValueError('predefined teacher endpoint absent; cannot change selection nodes')
        states[str(update)] = load_file(str(p), device=str(runtime.device))
    root = destination / 'readout'; root.mkdir(exist_ok=True)
    requests = []
    for pool, values, domain in [('selection', panel['selection_states'], 4), ('audit', panel['audit_states'], 5)]:
        for name, state in states.items():
            for state_id in values:
                identity = f'{pool}_{name}_{state_id:03d}'
                if (root / f'{identity}.json').exists():
                    continue
                request = _episode_request(runtime, task, state, name, state_id,
                    seed(domain, args.task, args.event), f'{event["event_id"]}_{identity}', snapshots=False, uses_teaching=False)
                request.update(pool=pool, output_name=identity)
                requests.append(request)
    requests.sort(key=lambda r: -r['task']['horizon'])
    for output in runner.run(requests):
        request = output['request']
        row = dict(**output['row'], pool=request['pool'], endpoint=request['parameter_ref'], event_id=event['event_id'])
        if output['history'].records and not output['row'].get('model_failure'):
            first = output['history'].records[0]
            row.update(first_native_normalized_actions=first['normalized_actions'].tolist(),
                       first_noise_seed=first['noise_seed'], first_action_mask=first['executed'].tolist())
        write_json_atomic(root / f'{request["output_name"]}.json', row)
    rows = [read_json(root / f'{pool}_{name}_{state_id:03d}.json') for pool, values in
            [('selection', panel['selection_states']), ('audit', panel['audit_states'])]
            for name in states for state_id in values]
    selection = {name: [r for r in rows if r['pool'] == 'selection' and r['endpoint'] == name] for name in states}
    distribution = endpoint_distribution({u: selection[str(u)] for u in (160, 480)}, selection['parent'])
    write_json_atomic(destination / 'teacher_distribution.json', distribution)
    write_json_atomic(root / 'aggregate.json', dict(rows=rows, q_T=distribution, components=deep_components(runtime, runner),
        audit_is_selection=False, complete=True))
    print(json.dumps(dict(event_id=event['event_id'], q_T=distribution, rows=len(rows))))






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'profile', 'collect', 'teacher', 'audit',
                                          'G', 'refresh', 'local-data', 'local', 'RL', 'report'])
    parser.add_argument('--root', type=Path, default=RUN_ROOT)
    parser.add_argument('--asset-root', type=Path, default=ASSET_ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--profile-history', type=Path)
    parser.add_argument('--task', type=int, choices=TASKS, default=12)
    parser.add_argument('--event', type=int, default=0)
    parser.add_argument('--stop', type=int, choices=[160, 320, 480], default=480)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--video-arm', choices=['correct', 'other', 'wrong'], default='correct')
    parser.add_argument('--microbatch', type=int, default=56)
    parser.add_argument('--function-microbatch',type=int,choices=[1,2,4],default=4)
    parser.add_argument('--profile-components',choices=['full','execution'],default='full')
    parser.add_argument('--frame-chunk', type=int, default=4)
    parser.add_argument('--frame-chunks', type=lambda s: [int(v) for v in s.split(',')], default=[2, 4, 8])
    parser.add_argument('--slots', type=int, default=8)
    parser.add_argument('--physical-gpu', type=int, required=False)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args); return
    shared = args.command in {'G','local','RL'}
    if shared:
        args.physical_gpu = int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[int(os.environ.get('LOCAL_RANK', '0'))])
    if args.physical_gpu is None:
        parser.error('GPU consumers require explicit physical GPU identity')
    if args.command != 'profile':
        identity = git_state()
        if identity['dirty_paths'] or identity['branch']:
            raise ValueError('formal consumers require a clean pushed detached frozen worktree')
        subprocess.run(['git', 'merge-base', '--is-ancestor', identity['commit'], 'origin/main'], check=True)
        write_json_atomic(args.root / 'code' / f'{args.command}_{identity["commit"]}.json', identity)
    context = initialize_distributed(require_numa=True, defer_process_group=True)
    if context.world_size != 1 and not shared:
        raise ValueError('these independent teacher/profile consumers use one worker per GPU')
    torch.set_num_threads(8)
    runtime = Runtime(context.device, asset_root=args.asset_root, frame_chunk=args.frame_chunk)
    contract = environment_contract(args.asset_root)
    for task in contract['tasks']:
        if runtime.tasks[task['global_task_id']].authority.language != task['language']:
            raise ValueError('task/video/action/environment exact language identity differs')
    from ember.pi05_assets import prepare_libero_config
    os.environ.update(EMBER_LIBERO_ASSETS_ROOT=contract['libero_paths']['assets'],
                      MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID=str(args.physical_gpu))
    prepare_libero_config(args.root / 'libero_config')
    runner = EpisodeRunner(runtime, contract, args.physical_gpu, slots=args.slots)
    try:
        if args.command == 'collect': collect(args, runtime, runner, contract)
        elif args.command == 'teacher': fit(args, runtime)
        elif args.command == 'audit': audit(args, runtime, runner, contract)
        elif args.command == 'profile':
            from .profile import profile,execution_profile
            consumer=profile if args.profile_components=='full' else execution_profile
            consumer(args,runtime,runner,contract)
        else:
            from .pipeline import run
            run(args, runtime, runner, contract, context)
    finally:
        runner.close(); runtime.close()


if __name__ == '__main__':
    main()
