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
from ember.lora import identity_lora_state, validate_lora_state, task_lora_state_dict, LORA_A_SUFFIX, LORA_B_SUFFIX
from ember.pi05_lora import load_pi05_lora_contract, derive_pi05_lora_rank

from .contract import (RUN_ROOT, ASSET_ROOT, TASKS, SOURCE, MT_PATH, EXPERT_ROOT, OPTIMIZER,
                       SCHEMA, PRIOR_ROOT, CARRY_IN_GPUH, EVENT_KINDS, materialize_panel, environment_contract, seed)
from .runtime import Runtime
from .teacher import QueryStream, fit_teacher, functional_labels, endpoint_distribution


def task_panel(root, task):
    panel = read_json(Path(root) / 'panel.json')
    return next(row for row in panel['tasks'] if row['task_id'] == task)


def save_state(path, state):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(path))


def coordinates(root, asset_root):
    """Actual deterministic Λ0 and fixed proximal units, before any teacher update."""
    if (root/'coordinates.json').exists():
        return
    original = load_pi05_lora_contract(asset_root/SOURCE['lora_contract'])
    current = derive_pi05_lora_rank(original, rank=8)
    initial = identity_lora_state(current)
    save_state(root/'initial.safetensors', initial)
    mt = load_file(str(MT_PATH))
    scales, dense_rms = {}, {}
    for target in current.targets:
        a_name, b_name = target.name + LORA_A_SUFFIX, target.name + LORA_B_SUFFIX
        a = float(initial[a_name].square().mean().sqrt())
        dense = mt[b_name].float() @ mt[a_name].float() * (original.alpha/original.rank)
        dense_rms[target.name] = float(dense.square().mean().sqrt())
        scales[a_name], scales[b_name] = a, dense_rms[target.name]/(8**.5*a)
    write_json_atomic(root/'task_lora_contract.json', current.to_dict())
    write_json_atomic(root/'coordinates.json', dict(schema_version=SCHEMA, task_rank=8,
        valid_coordinates=current.parameter_count, initial=str(root/'initial.safetensors'), initial_seed=7,
        base_source=SOURCE, MT=str(MT_PATH), base_rule='FP32(W_source+alpha/rank*B_MT@A_MT) then native dtype',
        S_prox=scales, actual_MT_dense_RMS=dense_rms, S_G_not_yet_formed=True,
        center='shared actual nonzero Kaiming A0/B0, no gauge restart'))


def prepare(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    panel = materialize_panel(root)
    original_panel = read_json(PRIOR_ROOT/'panel.json')
    if any(new[k] != old[k] for new, old in zip(panel['tasks'], original_panel['tasks'], strict=True)
           for k in ('task_id', 'video_order', 'init_order')):
        raise ValueError('rank8 successor changed original preregistered panel')
    coordinates(root, args.asset_root)
    seed_events(root,panel)
    contract = dict(schema_version=SCHEMA, stage='pilot', design='docs/designs/video_guided_proposal_writer_20261010.md#13',
        source=SOURCE, mt_checkpoint=str(MT_PATH), training_tasks=list(TASKS), panel=str(root / 'panel.json'),
        environment=environment_contract(), seed=7, meta_seeds=[202610101, 202610102], meta_rank=32,
        meta_alpha=32, task_rank=8, task_targets=38, task_factors=76, probe_seed=1729,
        limits=dict(GPU_hours=96, carry_in_GPU_hours=CARRY_IN_GPUH, added_peak_GiB=64,
            storage_scope='old and new ROOT retained growth plus new peak'),
        prior_root=str(PRIOR_ROOT), prior_cost_identity='18.60291210386488 timed plus .0375 conservative untimed',
        event_kinds=list(EVENT_KINDS), task_initial=str(root/'initial.safetensors'),
        coordinates=str(root/'coordinates.json'), bank_scale=str(root/'bank_scale.json'),
        elapsed_prediction_hours=[14,24], engineering_prediction_hours=[4,8], new_GPUh_prediction=[54,75],
        updates=dict(teacher=[160, 480], G=[240, 480], actor_local=128, actor_RL=16),
        logical_batch=dict(teacher_queries=112, G_conditions=4, actor_conditions=4),
        rng_schedule=dict(root='SeedSequence([20261010,domain,task,...])',
            domains=dict(practice=3,selection=4,audit=5,local=6,RL=7,report=8,query_episode=10,
                query_frame=11,recovery=12,rec_sample=13,keep_sample=14,decisions=30,proposals=31,practice_episode=32,
                CFM_event_and_time=40,CFM_noise=41,local_update=42,local_query=60,RL_query=70,report_query=80),
            initial_H160='practice[0],seed(3,task,1)',initial_H480='practice[1],seed(3,task,2)',
            RL_video='policy_videos[update%2]',RL_query='rl_query_states[update%4]',
            source_noise_probe=1729,task_initial_seed=7,meta_initial_seeds=[202610101,202610102]),
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


def event_record(root,panel,task_id,ordinal):
    destination=root/'events'/f'task_{task_id:04d}_event_{ordinal:02d}'
    seed_root = root/'events'/f'task_{task_id:04d}_event_00'/'teacher/checkpoints'
    node = 160 if ordinal in (1,3) else 480
    producer = f'seed{node}_task_{task_id:04d}'
    parent_ref = producer if ordinal in (1,2) else 'MT300'
    parent_path = seed_root/f'update_{node:08d}'/'lora.safetensors' if ordinal in (1,2) else root/'initial.safetensors'
    event = dict(event_id=destination.name, task_id=task_id, event_ordinal=ordinal, event_kind=EVENT_KINDS[ordinal],
        teacher_demo=panel['teacher_videos'][int(ordinal in (1,2))], parent_ref=parent_ref, parent=str(parent_path),
        role='nonheld_shared_training', coordinates=str(root/'coordinates.json'),
        base_identity=dict(source=SOURCE,MT=str(MT_PATH),task_rank=8), history_kind=('empty' if ordinal==0 else f'independent_actual_seed{node}_episode'),
        history_producer=None if ordinal==0 else producer, selection_states=panel['selection_states'],
        audit_states=panel['audit_states'], rng_roots=panel['rng_roots'], schema_version=SCHEMA)
    return event,node


def seed_events(root,panel):
    """Empty-H events need no policy load or GPU allocation."""
    from .path import save_history
    initial=load_file(str(root/'initial.safetensors'))
    for row in panel['tasks']:
        event,_=event_record(root,row,row['task_id'],0)
        destination=root/'events'/event['event_id'];destination.mkdir(parents=True,exist_ok=True)
        if (destination/'event.json').exists():continue
        history_path=destination/'history.pt.gz';labels_path=destination/'functional_labels.pt'
        save_history(history_path,History(states={'MT300':initial}))
        torch.save(dict(rec=[],keep=[],recovery_validations=[]),labels_path)
        write_json_atomic(destination/'history_capture.json',dict(complete=True,code=git_state(),
            episodes=[],records=0,actual_practice=False,actual_producer=None,labels_complete=True))
        event.update(history_path=str(history_path),labels_path=str(labels_path),rec_labels=0,keep_labels=0,
            recovery_attempts=0,capture_components={},complete=True)
        write_json_atomic(destination/'event.json',event)


def collect(args, runtime, runner, contract):
    from .path import save_history, load_history
    panel = task_panel(args.root, args.task)
    task = next(t for t in contract['tasks'] if t['global_task_id'] == args.task)
    ordinal = args.event
    if ordinal not in range(4):
        raise ValueError('only registered seed/mid/late/return events')
    destination = args.root/'events'/f'task_{args.task:04d}_event_{ordinal:02d}'
    if (destination/'event.json').exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    event,node=event_record(args.root,panel,args.task,ordinal)
    seed_root=args.root/'events'/f'task_{args.task:04d}_event_00'/'teacher/checkpoints'
    producer=event['history_producer']
    captured = destination/'history_capture.json'
    history_path = (args.root/'events'/f'task_{args.task:04d}_event_01'/'history.pt.gz'
                    if ordinal==3 else destination/'history.pt.gz')
    if captured.exists() or ordinal==3:
        history = load_history(history_path)
    else:
        history = History(states={'MT300': {k:v.detach().cpu() for k,v in runtime.mt.items()}})
        if ordinal in (1,2):
            producer_state = load_file(str(seed_root/f'update_{node:08d}'/'lora.safetensors'), device=str(runtime.device))
            request = _episode_request(runtime, task, producer_state, producer, panel['practice_states'][ordinal-1],
                seed(3,args.task,ordinal), destination.name, remaining=task['horizon']+10,
                snapshots=True, uses_teaching=False)
            output = next(runner.run([request])); history = output['history']
            history.states['MT300'] = {k:v.detach().cpu() for k,v in runtime.mt.items()}
        save_history(history_path,history)
    if not captured.exists():
        write_json_atomic(captured, dict(event_id=event['event_id'], complete=True, code=git_state(),
            episodes=history.episodes, records=len(history.records), actual_producer=event['history_producer'],
            labels_complete=False, independent_history_not_H_mix=True, shared_history_source=str(history_path)))
    expert = None
    if history.records:
        expert = load_file(str(EXPERT_ROOT/f'task_{args.task:04d}_u480/lora.safetensors'), device=str(runtime.device))
        validate_lora_state(expert,runtime.expert_lora)
    labels = functional_labels(runtime, runner, event, history, task, expert)
    torch.save(labels,destination/'functional_labels.pt')
    event.update(history_path=str(history_path), labels_path=str(destination/'functional_labels.pt'),
        rec_labels=len(labels['rec']), keep_labels=len(labels['keep']), recovery_attempts=len(labels['recovery_validations']),
        capture_components=deep_components(runtime,runner), complete=True)
    write_json_atomic(destination/'event.json',event)
    print(json.dumps(dict(task=args.task,event_ordinal=ordinal,collected=True,components=deep_components(runtime,runner))))


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
                                          'G', 'local-data', 'local', 'RL', 'report'])
    parser.add_argument('--root', type=Path, default=RUN_ROOT)
    parser.add_argument('--asset-root', type=Path, default=ASSET_ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--profile-consumer',choices=['full','throughput'],default='full')
    parser.add_argument('--task', type=int, choices=TASKS, default=12)
    parser.add_argument('--event', type=int, default=0)
    parser.add_argument('--stop', type=int, choices=[160, 240, 480], default=480)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--video-arm', choices=['correct', 'other', 'wrong'], default='correct')
    parser.add_argument('--microbatch', type=int, default=28)
    parser.add_argument('--function-microbatch',type=int,choices=[1,2,4],default=4)
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
    runtime = Runtime(context.device, asset_root=args.asset_root, root=args.root, frame_chunk=args.frame_chunk)
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
            from .profile import profile
            profile(args,runtime,runner,contract)
        else:
            from .pipeline import run
            run(args, runtime, runner, contract, context)
    finally:
        runner.close(); runtime.close()


if __name__ == '__main__':
    main()
