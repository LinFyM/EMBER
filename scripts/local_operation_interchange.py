#!/usr/bin/env python3
"""One-shot §20 content transplant using the canonical frozen image compiler."""
from pathlib import Path
import argparse
import os
import subprocess
import time

import torch
from safetensors.torch import load_file, save_file

from ember.lora import validate_lora_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic, git_state
from ember.pi05_source_setup import initialize_distributed
from ember.pi05_eval_queue import claim_next, fail_job
from ember.proposal_writer.contract import ASSET_ROOT
from ember.proposal_writer.runtime import Runtime
from ember.proposal_writer.image_compiler import pure_features
from ember.proposal_writer.image_history import model_history
from ember.proposal_writer.image_learning import load_compiler
from ember.proposal_writer.image_data import publish_record
from ember.proposal_writer.learning import memory_peak
from ember.proposal_writer.analysis import paired, billing, interaction_ledger
from ember.writer.practice import EpisodeRunner
from ember.writer.runtime import autocast

ARMS = ('C', 'O', 'P', 'N')
WINDOWS = ('correct_window', 'same_task_other_window', 'cross_up_window', 'cross_down_window')
STATES = [25, 4, 47, 7, 13, 35, 28, 19, 12, 5, 14, 21, 48, 37, 36, 16]


def feature_file(root, task, demo):
    return root/'image_features'/f'task_{task:04d}_demo_{demo:02d}.pt'


def lora_file(root, demo, arm):
    return root/'loras'/f'task_0043_demo_{demo:02d}_{arm}.safetensors'


def case_id(demo, arm, state):
    return f'demo_{demo:02d}_{arm}_state_{state:03d}'


def validate_contract(root):
    run = read_json(root/'run_contract.json')
    if (run['stage'] != 'local_operation_interchange' or run['training_updates'] != 0
            or run['compiler_arm'] != 'V' or run['receiver_task'] != 43
            or run['feature_tasks'] != [43, 96] or run['feature_demos'] != list(range(4))
            or run['init_state_ids'] != STATES or run['noise_root'] != 1235739335
            or [run[k] for k in ('query_rollouts', 'physical_starts_maximum',
                'failed_partial_episode_maximum', 'environment_steps_maximum', 'inflight_maximum')]
                != [256, 272, 16, 111520, 16]
            or run['history']['H']['records'] != 36 or run['history']['practice_state_id'] != 2):
        raise ValueError('only the frozen §20 panel, H and physical budget are authorized')
    mapping = read_json(root/'inputs/cross_scene_operation_audit.json')['frozen_intervention_mapping']
    expected = [([70,75,80],[75,80,85],[50,55,60],[75,80,85]),
        ([75,80,85],[70,75,80],[55,60,65],[75,80,85]),
        ([70,75,80],[70,75,80],[55,60,65],[75,80,85]),
        ([70,75,80],[70,75,80],[60,65,70],[85,90,95])]
    for demo, entry in enumerate(mapping):
        actual = [(entry[k]['task'], entry[k]['demo'], entry[k]['frames']) for k in WINDOWS]
        wanted = [(43,demo,expected[demo][0]),(43,(demo+1)%4,expected[demo][1]),
            (96,demo,expected[demo][2]),(96,demo,expected[demo][3])]
        if entry['receiver_demo'] != demo or actual != wanted:
            raise ValueError('frozen video/window identities changed')
    if len(mapping) != 4: raise ValueError('four fixed receiver teaching videos required')
    return run, mapping


def positions(indices, frames):
    values = []
    for frame in frames:
        found = torch.nonzero(indices == frame).flatten()
        if len(found) != 1 or frame % 5: raise ValueError('window requires exact raw stride5 indices')
        values.append(int(found[0]))
    return values


def transplant(receiver, donor, target_frames, donor_frames):
    target = positions(receiver['indices'], target_frames)
    source = positions(donor['indices'], donor_frames)
    if len(target) != 3 or len(source) != 3: raise ValueError('exactly three complete dual-camera frames')
    z = receiver['z_img'].clone()
    z[target] = donor['z_img'][source]
    keep = torch.ones(len(z), dtype=torch.bool); keep[target] = False
    if not torch.equal(z[keep], receiver['z_img'][keep]) or not torch.equal(z[target], donor['z_img'][source]):
        raise ValueError('replacement changed non-window content')
    return z, target, source


@torch.no_grad()
def compile_panel(args, runtime):
    root = args.root; run, mapping = validate_contract(root)
    measurements = []; chunk = args.frame_chunk
    for task in (43, 96):
        for demo in range(4):
            path = feature_file(root, task, demo)
            if path.exists(): continue
            torch.cuda.reset_peak_memory_stats(runtime.device)
            started = time.monotonic()
            value = pure_features(runtime, task, demo, frame_chunk=chunk)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(value, path)
            peak = memory_peak(runtime.device)
            record = dict(task=task, demo=demo, frames=len(value['indices']), frame_chunk=chunk,
                seconds=time.monotonic()-started, peak=peak, actual_registered_feature=True)
            measurements.append(record)
            # Remaining real videos use a larger chunk when the first real read leaves room.
            free, _ = torch.cuda.mem_get_info(runtime.device)
            if chunk < 64 and free > 16*1024**3: chunk = 64
            write_json_atomic(root/'feature_resources.json', dict(rows=measurements,
                next_frame_chunk=chunk, no_discarded_profile=True, source_native_queries=0))
    parent = Path(run['source_parent'])
    model, lora, saved = load_compiler(parent, 'V', Path(run['source_checkpoint']), runtime.device)
    del saved
    model.eval().requires_grad_(False)
    history = torch.load(run['history_features'], map_location='cpu', weights_only=False)
    if history['source']['H'] != run['history']['H']: raise ValueError('real receiver MT H changed')
    h = model_history(history, runtime.device)
    pool = {(task,demo):torch.load(feature_file(root,task,demo),map_location='cpu',weights_only=False)
            for task in (43,96) for demo in range(4)}
    rows = []
    for entry in mapping:
        demo = entry['receiver_demo']; receiver = pool[43,demo]
        for arm, window in zip(ARMS, WINDOWS):
            output = lora_file(root,demo,arm)
            if output.exists():
                rows.append(read_json(output.with_suffix('.json'))); continue
            donor = entry[window]
            z, target, source = transplant(receiver, pool[donor['task'],donor['demo']],
                entry['correct_window']['frames'], donor['frames'])
            started = time.monotonic()
            with autocast(runtime.device):
                state = model(z.to(runtime.device), receiver['indices'].to(runtime.device), h)
            validate_lora_state(state, lora)
            if len(state) != 76 or any(not torch.isfinite(v).all() for v in state.values()):
                raise ValueError('one finite complete38 rank8 LoRA required')
            output.parent.mkdir(exist_ok=True)
            save_file({k:v.detach().cpu().contiguous() for k,v in state.items()}, str(output))
            row = dict(demo=demo,arm=arm,receiver_task=43,donor_task=donor['task'],donor_demo=donor['demo'],
                target_raw_frames=entry['correct_window']['frames'],donor_raw_frames=donor['frames'],
                target_positions=target,donor_positions=source,frames=len(receiver['indices']),
                indices=receiver['indices'].tolist(),checkpoint=run['source_checkpoint'],training_commit=run['training_source_commit'],
                full_Reader_H_F_forward=True,H=run['history']['H'],H_language_hidden=False,Compiler_direct_L=False,
                outside_window_unchanged=True,replacement_before_Reader=True,complete_LoRA=str(output),
                seconds=time.monotonic()-started,feature_source=receiver['source'])
            write_json_atomic(output.with_suffix('.json'),row);rows.append(row)
    if len(rows)!=16: raise ValueError('all16 conditions are required before final readouts')
    write_json_atomic(root/'compilation_complete.json',dict(complete=True,conditions=rows,
        compiler_forwards=16,feature_videos=8,training_updates=0,peak=memory_peak(runtime.device)))


def passive_capture(root):
    return dict(mode='compact',full_conditions=[],trajectory_root=str(root/'traces'),
        passive_trace=dict(trace_root=str(root/'traces')))


@torch.no_grad()
def evaluate(args, runtime):
    root=args.root;run,_=validate_contract(root)
    if not read_json(root/'compilation_complete.json')['complete']: raise ValueError('all16 compiled conditions required')
    environment=run['environment'];environment['diagnostic_occupancy_capture']=passive_capture(root)
    environment['diagnostic_stage_predicates']=dict(full_conditions_only=False)
    from ember.pi05_assets import prepare_libero_config, configure_libero_runtime_assets
    from ember.pi05_eval.trajectory_capture import validate_passive_trace_row
    os.environ.update(EMBER_LIBERO_ASSETS_ROOT=environment['libero_paths']['assets'],MUJOCO_GL='egl',
        PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID=str(args.physical_gpu))
    prepare_libero_config(root/'libero_config');configure_libero_runtime_assets(Path(environment['libero_paths']['assets']))
    task=environment['tasks'][0]
    if task['global_task_id']!=43 or task['horizon']!=400:raise ValueError('actual task43 source horizon400 required')
    factors={(d,a):load_file(str(lora_file(root,d,a)),device=str(runtime.device)) for d in range(4) for a in ARMS}
    runner=EpisodeRunner(runtime,environment,args.physical_gpu,slots=args.slots)
    queue=root/'queues/final.sqlite'
    try:
        while not (root/'stop_after_current_shard.json').exists() and (claim:=claim_next(queue,worker_id=args.worker_id)) is not None:
            started=time.monotonic();outputs={};requests=[]
            try:
                for init in claim.shard.init_state_ids:
                    for demo in range(4):
                        for arm in ARMS:
                            identity=case_id(demo,arm,init);output=root/'rows'/f'{identity}.json'
                            outputs[identity]=(output,demo,arm)
                            if output.exists():continue
                            requests.append(dict(task=task,state=factors[demo,arm],parameter_ref=identity,state_id=init,
                                noise_root=run['noise_root'],episode_id=identity,remaining=410,
                                capture_history=False,capture_phi=False,capture_hidden=False,uses_teaching=False,
                                capture_root=str(root/'traces'/identity)))
                for actual in runner.run(requests):
                    output,demo,arm=outputs[actual['request']['episode_id']];row=actual['row']
                    first=actual['history'].records[0]
                    row.update(episode_id=actual['request']['episode_id'],global_task_id=43,compiler_arm=arm,
                        teacher_demo=demo,complete_LoRA=str(lora_file(root,demo,arm)),checkpoint=run['source_checkpoint'],
                        first_native_normalized_actions=first['normalized_actions'].tolist(),first_noise_seed=first['noise_seed'],
                        compiler_exits_before_final_episode=True,teaching_in_final_episode=False,
                        new_adaptation_episodes=0,H=run['history']['H'])
                    write_json_atomic(output,row)
                    validate_passive_trace_row(row,environment,task)
                    bodies=row['continuous_control_trace']['body_registry']
                    butter=[b for b in bodies if b['body_name']=='butter_2_main']
                    if len(butter)!=1 or butter[0]['name']!='butter_2' or butter[0]['kind']!='object':
                        raise ValueError('actual receiver butter body binding changed')
                rows=[read_json(v[0]) for v in outputs.values()]
                resource=dict(worker=args.worker_id,slots=args.slots,rows=len(rows),seconds=time.monotonic()-started,
                    peak=memory_peak(runtime.device),components=dict(runner.components),phi_or_hidden_collected=False)
                receipt=root/'shard_receipts'/f'{claim.shard.job_id}_{claim.attempt:03d}.json'
                write_json_atomic(receipt,dict(complete=True,rows=rows,resources=resource))
                publish_record(queue,claim,args.worker_id,receipt,len(rows),successes=sum(r['success'] for r in rows),rows_per_state=16)
                if not (root/'first_actual_readout_resources.json').exists():
                    write_json_atomic(root/'first_actual_readout_resources.json',resource)
            except BaseException as error:
                write_json_atomic(root/'stop_after_current_shard.json',dict(error=repr(error),no_new_claims=True))
                fail_job(queue,job_id=claim.shard.job_id,worker_id=args.worker_id,claim_token=claim.claim_token,error=repr(error))
                raise
    finally:runner.close()


def summarize(root):
    run,_=validate_contract(root)
    rows=[read_json(p) for p in sorted((root/'rows').glob('*.json'))]
    expected={(d,a,s) for d in range(4) for a in ARMS for s in STATES}
    if {(r['teacher_demo'],r['compiler_arm'],r['init_state_id']) for r in rows}!=expected or len(rows)!=256:
        raise ValueError('all256 unique fixed scientific outcomes required')
    mt=read_json(root/'inputs/MT_reference.json')['rows'];parts={};comparisons={}
    for demo in range(4):
        part={a:[r for r in rows if r['teacher_demo']==demo and r['compiler_arm']==a] for a in ARMS}
        parts[str(demo)]={a:sum(r['success'] for r in part[a]) for a in ARMS}
        comparisons[str(demo)]={f'{x}_to_{y}':paired(part[x],part[y]) for x,y in [('C','O'),('O','P'),('N','P')]}
        comparisons[str(demo)]['MT_to_C']=paired(mt,part['C'])
    totals={a:sum(r['success'] for r in rows if r['compiler_arm']==a) for a in ARMS}
    aggregate_pairs={}
    for name in ('C_to_O','O_to_P','N_to_P','MT_to_C'):
        values=[comparisons[str(d)][name] for d in range(4)]
        record={k:sum(v[k] for v in values) for k in ('rows','parent_success','candidate_success',
            'net','retained','gained','lost','churn','success_set_overlap')}
        union=record['parent_success']+record['candidate_success']-record['retained']
        record['success_Jaccard']=record['retained']/union if union else 1.
        record['gained_cases']=[[d,s[1]] for d,v in enumerate(values) for s in v['gained_states']]
        record['lost_cases']=[[d,s[1]] for d,v in enumerate(values) for s in v['lost_states']]
        record['independent_MT_episodes']=16 if name=='MT_to_C' else None
        aggregate_pairs[name]=record
    write_json_atomic(root/'results.json',dict(complete=True,rows=rows,successes=totals,per_video=parts,
        per_video_paired=comparisons,aggregate_pairs=aggregate_pairs,MT_unique_successes=7,MT_unique_episodes=16,
        dependent_video_state_grid=True,not_strict_held400=True,training=0,held=False,Test=False,
        no_automatic_successor=True,interaction=interaction_ledger(root),billing=billing(root)))
    write_json_atomic(root/'complete.json',dict(complete=True,rows=256,scope='local_operation_interchange',Test=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['compile','evaluate','summarize'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--physical-gpu',type=int,default=0)
    parser.add_argument('--worker-id',default='compiler')
    parser.add_argument('--frame-chunk',type=int,default=32)
    parser.add_argument('--slots',type=int,default=3)
    args=parser.parse_args();identity=git_state()
    if identity['dirty_paths'] or identity['branch']:raise ValueError('formal consumers require clean detached source')
    subprocess.run(['git','merge-base','--is-ancestor',identity['commit'],'origin/main'],check=True)
    run,_=validate_contract(args.root)
    if args.command=='summarize':summarize(args.root);return
    context=initialize_distributed(require_numa=True,defer_process_group=True)
    if context.world_size!=1:raise ValueError('independent consumers do not use NCCL')
    torch.set_num_threads(8)
    write_json_atomic(args.root/'code'/f'{args.command}_{args.worker_id}.json',dict(**identity,
        physical_gpu=args.physical_gpu,slots=args.slots,frame_chunk=args.frame_chunk))
    runtime=Runtime(context.device,asset_root=ASSET_ROOT,root=args.root,task_ids=(43,96),task_role='train')
    runtime.policy.eval().requires_grad_(False);runtime.episode_event_log=os.environ.get('EMBER_EPISODE_EVENT_LOG')
    try:
        if args.command=='compile':compile_panel(args,runtime)
        else:evaluate(args,runtime)
    finally:runtime.close()


if __name__=='__main__':main()
