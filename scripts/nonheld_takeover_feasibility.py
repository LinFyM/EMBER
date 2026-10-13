"""Temporary zero-update label-side diagnostic; retire after the fixed panel closes."""
from __future__ import annotations
import argparse, json, math, os, subprocess, time
from pathlib import Path
import torch
from safetensors.torch import load_file
from ember.lora import validate_lora_state
from ember.pi05_lora import load_pi05_lora_contract,derive_pi05_lora_rank
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.pi05_source_setup import initialize_distributed
from ember.pi05_eval_contract import policy_noise_seed
from ember.proposal_writer.contract import ASSET_ROOT,SOURCE
from ember.proposal_writer.runtime import Runtime
from ember.proposal_writer.path import save_history,load_history
from ember.proposal_writer.learning import memory_peak
from ember.writer.practice import EpisodeRunner

TASKS=(4,17,25,34)
TEACHERS=Path('/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261011/full36_common_teacher')

def inputs(args):
    panel=read_json(args.panel);run=read_json(args.source_root/'run_contract.json')
    if (tuple(panel['tasks'])!=TASKS or panel['max_original_episodes']!=8
        or panel['max_paired_continuations']!=24 or panel['max_total_episodes']!=32
        or panel['max_environment_steps_including_settling']!=10640):raise ValueError('frozen8/12/32 panel required')
    protocol=read_json(ASSET_ROOT/SOURCE['data_protocol']);suites=protocol['split']['suites']
    trained=set()
    for index,suite in enumerate(('libero_spatial','libero_object','libero_goal','libero_10')):
        part=suites[suite];trained.update(index*10+int(x) for x in part['train'])
    if set(TASKS)-trained:raise ValueError('panel crosses train24 information wall')
    lora=derive_pi05_lora_rank(load_pi05_lora_contract(ASSET_ROOT/SOURCE['lora_contract']),rank=8)
    environment=run['environment'];task_by={x['global_task_id']:x for x in environment['tasks']};weights={};cases=[];teachers={}
    for condition in panel['conditions']:
        task=condition['global_task_id'];spec=task_by[task]
        if spec['language']!=condition['language']:raise ValueError('exact task language changed')
        teacher_file=TEACHERS/'teachers'/f'task_{task:04d}'/'E.json';teacher=read_json(teacher_file)
        if not teacher['complete'] or not teacher['shared_target_for_all_parent_H']:raise ValueError('original common E required')
        teachers[str(task)]={k:teacher[k] for k in ('reference','weights','selected_update')};teachers[str(task)]['receipt']=str(teacher_file)
        weights[task]={name:load_file(str(path)) for name,path in [('P',condition['incoming_weights']),('E',teacher['weights'])]}
        for state in weights[task].values():validate_lora_state(state,lora)
        generation=read_json(Path(condition['source_generation_receipt']));identity=generation['input_identity']
        if identity['F_cursor']!=38349 or identity['teacher_demo']!=condition['teacher_demo']:raise ValueError('saved first G condition changed')
        credit=read_json(Path(condition['source_query_receipt']))
        for selected in condition['source_rows']:
            row=selected['source_row'];state=selected['init_state_id'];matches=[x for x in credit['rows'] if x.get('reference')=='G_000' and x['init_state_id']==state]
            if (len(matches)!=1 or matches[0]!=row or row['split_role']!='train'
                or row['success']!=selected['recorded_success'] or row['env_seed']!=row['policy_seed_root']
                or Path(condition['incoming_weights']).resolve()!=Path(row['fixed_G_weights']).resolve()):raise ValueError('original P row/weights/RNG identity changed')
            cases.append(dict(identity=f'task_{task:04d}_state_{state:03d}',task=task,state=state,
                role=selected['selected_as'],expected_success=selected['recorded_success'],spec=spec,
                noise_root=row['policy_seed_root'],original=row,P_weights=condition['incoming_weights'],
                E=teachers[str(task)],teacher_demo=condition['teacher_demo']))
    if len(cases)!=8:raise ValueError('exact eight original cases required')
    return panel,environment,weights,cases,teachers

def request(case,state,ref,*,identity,snapshot=None,capture=True):
    horizon=case['spec']['horizon'];remaining=horizon+10 if snapshot is None else horizon-snapshot['episode_control_steps']
    return dict(task=case['spec'],state=state,parameter_ref=ref,state_id=case['state'],noise_root=case['noise_root'],
        episode_id=identity,remaining=remaining,snapshots=capture,
        capture_history=capture,capture_phi=False,uses_teaching=False,
        **({'snapshot':snapshot} if snapshot is not None else {}))

def row(output,case,arm):
    result=dict(output['row'],global_task_id=case['task'],diagnostic_case=case['identity'],arm=arm,
        label_side_only=True,not_Writer_condition=True,parameters_frozen=True)
    record=output['history'].records[0] if output['history'].records else None
    result['first_native_normalized_actions']=record['normalized_actions'].tolist() if record else None
    result['first_noise_seed']=record['noise_seed'] if record else None
    result['first_executed_actions_padded']=record['actions'].tolist() if record else None
    return result

def profile(runner,runtime,start,before):
    torch.cuda.synchronize(runtime.device)
    return dict(seconds=time.monotonic()-start,environment_steps=runner.environment_steps-before['steps'],
        plans=runner.components['plans']-before['plans'],peak=memory_peak(runtime.device),
        cumulative_runner=runner.components)

def run(args):
    panel,environment,weights,cases,teachers=inputs(args)
    if args.check_only:
        print(json.dumps(dict(complete=True,case_count=8,slot_count=12,all_train24=True,all76_factor_shapes_valid=True,
            shared_E=teachers,zero_GPU=True,zero_episode=True)));return
    args.root.mkdir(parents=True,exist_ok=True)
    if (args.root/'complete.json').exists():raise ValueError('completed fixed diagnostic cannot restart')
    code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip() or subprocess.check_output(['git','branch','--show-current'],text=True).strip():raise ValueError('clean detached source required')
    subprocess.run(['git','merge-base','--is-ancestor',code,'origin/main'],check=True)
    write_json_atomic(args.root/'panel.json',panel)
    write_json_atomic(args.root/'run_contract.json',dict(schema_version='ember_nonheld_takeover_feasibility_v1',
        code_commit=code,source_root=str(args.source_root),source_panel=str(args.panel),environment=environment,
        cases=[{k:v for k,v in c.items() if k!='spec'} for c in cases],common_E=teachers,
        max_original_episodes=8,max_paired_continuations=24,max_episodes=32,max_environment_steps=10640,
        snapshot_schema='ember_actual_practice_snapshot_v4',snapshot_index_rule='frozen fractions of actual replay replan count; original case role retained if outcome changes',
        registered_snapshot_denominator=12,updates=0,new_G=0,held=False,Test=False,
        extra_policy_forward_for_action_comparison=False,added_peak_GiB=4,slots=args.slots))
    context=initialize_distributed(require_numa=True,defer_process_group=True)
    runtime=Runtime(context.device,root=args.source_root,frame_chunk=16,task_ids=TASKS)
    runtime.policy.eval().requires_grad_(False)
    from ember.pi05_assets import prepare_libero_config,configure_libero_runtime_assets
    os.environ.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',EMBER_LIBERO_ASSETS_ROOT=environment['libero_paths']['assets'])
    prepare_libero_config(args.root/'libero_config');configure_libero_runtime_assets(Path(environment['libero_paths']['assets']))
    runner=EpisodeRunner(runtime,environment,args.physical_gpu,slots=args.slots)
    actual_rows=[];slots=[];requests=[];owners={};replays={};phases={}
    write_json_atomic(args.root/'runtime_started.json',dict(code_commit=code,physical_gpu=args.physical_gpu,
        slots=args.slots,source_policy_and_MT_loaded=True,Writer_initialized=False,updates=0,new_G=0))
    try:
        for case in cases:
            folder=args.root/'replays'/case['identity'];folder.mkdir(parents=True,exist_ok=True)
            if (folder/'row.json').exists() and (folder/'history.pt.gz').exists():
                replays[case['identity']]=(read_json(folder/'row.json'),load_history(folder/'history.pt.gz'));continue
            r=request(case,weights[case['task']]['P'],'G_000',identity=case['identity']+'_replay');requests.append(r);owners[r['episode_id']]=case
        start=time.monotonic();before=dict(steps=runner.environment_steps,plans=runner.components['plans'])
        for output in runner.run(sorted(requests,key=lambda r:r['task']['horizon'],reverse=True)):
            case=owners[output['request']['episode_id']];folder=args.root/'replays'/case['identity'];raw=row(output,case,'P_replay')
            if raw.get('model_failure'):raise ValueError('actual P model failure')
            raw['replay_matches_expected_success']=raw['success']==case['expected_success']
            old=case['original'];raw['source_initial_proprio_max_abs_difference']=max(abs(a-b) for a,b in zip(raw['initial_proprio'],old['initial_proprio']))
            raw['native_noise_common_prefix_matches_source']=raw['policy_noise_seeds'][:min(len(raw['policy_noise_seeds']),len(old['policy_noise_seeds']))]==old['policy_noise_seeds'][:min(len(raw['policy_noise_seeds']),len(old['policy_noise_seeds']))]
            save_history(folder/'history.pt.gz',output['history']);write_json_atomic(folder/'row.json',raw);replays[case['identity']]=(raw,output['history'])
        phases['replays']=profile(runner,runtime,start,before);runner.release_renderers();torch.cuda.empty_cache()
        for case in cases:
            replay,history=replays[case['identity']];n=len(history.records);fractions=(.5,) if case['role']=='recorded_success' else (.25,.75);seen=set()
            for ordinal,fraction in enumerate(fractions):
                index=math.floor((n-1)*fraction);identity=case['identity']+f'_slot_{ordinal}';folder=args.root/'slots'/identity;folder.mkdir(parents=True,exist_ok=True)
                item=dict(identity=identity,task=case['task'],case=case['identity'],original_role=case['role'],
                    expected_success=case['expected_success'],replay_success=replay['success'],replay_matches_expected=replay['replay_matches_expected_success'],
                    original_replans=case['original']['replans'],actual_replay_replans=n,fraction=fraction,pre_action_replan_index=index,
                    history_source=str(args.root/'replays'/case['identity']/'history.pt.gz'),valid=True)
                if n<1 or index<0 or index>=n:item.update(valid=False,missing_reason='no actual pre-action snapshot')
                elif index in seen:item.update(valid=False,missing_reason='duplicate registered snapshot position')
                elif replay['source_initial_proprio_max_abs_difference']>1e-5 or not replay['native_noise_common_prefix_matches_source']:item.update(valid=False,missing_reason='source scene or native noise identity mismatch')
                snapshot=history.records[index].get('snapshot') if item['valid'] else None
                if item['valid'] and (snapshot is None or snapshot.get('schema_version')!='ember_actual_practice_snapshot_v4'):item.update(valid=False,missing_reason='snapshot v4 missing')
                seen.add(index)
                if item['valid']:
                    torch.save(snapshot,folder/'snapshot.pt');item.update(snapshot=str(folder/'snapshot.pt'),pre_action_control_steps=snapshot['episode_control_steps'],
                        remaining_horizon=case['spec']['horizon']-snapshot['episode_control_steps'],noise_root=case['noise_root'])
                slots.append(item);write_json_atomic(folder/'slot.json',item)
        if len(slots)!=12:raise ValueError('all twelve registered slots required')
        write_json_atomic(args.root/'slots.json',slots)
        requests=[];owners={};case_by={c['identity']:c for c in cases}
        for slot in slots:
            if not slot['valid']:continue
            case=case_by[slot['case']];snapshot=torch.load(slot['snapshot'],map_location='cpu',weights_only=False)
            for arm in ('P','E'):
                file=args.root/'slots'/slot['identity']/(arm+'.json')
                if file.exists():continue
                ref='G_000' if arm=='P' else case['E']['reference']
                r=request(case,weights[case['task']][arm],ref,identity=slot['identity']+'_'+arm,snapshot=snapshot,capture=False)
                requests.append(r);owners[r['episode_id']]=(case,slot,arm,file)
        start=time.monotonic();before=dict(steps=runner.environment_steps,plans=runner.components['plans'])
        for output in runner.run(sorted(requests,key=lambda r:r['remaining'],reverse=True)):
            case,slot,arm,file=owners[output['request']['episode_id']];raw=row(output,case,arm)
            expected=policy_noise_seed(case['noise_root'],case['spec']['suite'],case['spec']['task_id'],case['state'],slot['pre_action_control_steps']//5)
            expected_trace=[policy_noise_seed(case['noise_root'],case['spec']['suite'],case['spec']['task_id'],case['state'],
                slot['pre_action_control_steps']//5+i) for i in range(raw['replans'])]
            if (raw['settling_steps']!=0 or raw['restored_control_steps']!=slot['pre_action_control_steps']
                or raw['first_noise_seed']!=expected or raw['policy_noise_seeds']!=expected_trace):raise ValueError('snapshot continuation/noise identity changed')
            raw.update(snapshot_source=slot['snapshot'],snapshot_slot=slot['identity'],remaining_official_horizon=slot['remaining_horizon'])
            write_json_atomic(file,raw)
        phases['continuations']=profile(runner,runtime,start,before)
        counts=dict(P_fail_E_success=0,P_success_E_fail=0,both_success=0,both_fail=0,missing=0);paired=[]
        for slot in slots:
            item=dict(slot)
            if not slot['valid']:counts['missing']+=1;paired.append(item);continue
            folder=args.root/'slots'/slot['identity'];p,e=read_json(folder/'P.json'),read_json(folder/'E.json')
            if max(abs(a-b) for a,b in zip(p['initial_proprio'],e['initial_proprio']))>1e-5:raise ValueError('P/E did not restore the same actual snapshot')
            common=min(len(p['policy_noise_seeds']),len(e['policy_noise_seeds']))
            if not common or p['policy_noise_seeds'][:common]!=e['policy_noise_seeds'][:common]:raise ValueError('P/E native noise streams not paired')
            key='both_success' if p['success'] and e['success'] else 'P_success_E_fail' if p['success'] else 'P_fail_E_success' if e['success'] else 'both_fail';counts[key]+=1
            difference=torch.tensor(p['first_native_normalized_actions'])-torch.tensor(e['first_native_normalized_actions'])
            item.update(P=p,E=e,outcome=key,first_chunk_normalized_action_RMS=float(difference.square().mean().sqrt()),
                first_chunk_normalized_action_max_abs=float(difference.abs().max()),action_difference_is_not_causal_correctness=True);paired.append(item)
        actual_rows=[value[0] for value in replays.values()]+[item[arm] for item in paired if item['valid'] for arm in ('P','E')]
        steps=sum(r['environment_steps'] for r in actual_rows)
        if len(actual_rows)>32 or steps>10640:raise ValueError('registered interaction bound exceeded')
        if runtime._generator is not None or runtime._actor is not None:raise ValueError('diagnostic initialized Writer')
        result=dict(complete=True,registered_slot_denominator=12,counts=counts,slots=paired,replays=[x[0] for x in replays.values()],
            original_replay_episodes=8,paired_continuations=len(actual_rows)-8,total_episodes=len(actual_rows),actual_environment_steps=steps,
            updates=0,new_G=0,held=False,Test=False,all_labels_outside_original_H_and_bank=True,phases=phases,
            no_first_chunk_error_or_new_LoRA_causal_claim=True,code_commit=code,peak=memory_peak(runtime.device))
        write_json_atomic(args.root/'results.json',result);write_json_atomic(args.root/'complete.json',dict(complete=True,results=str(args.root/'results.json'),code_commit=code,total_episodes=len(actual_rows),actual_environment_steps=steps,no_automatic_successor=True))
        print(json.dumps(dict(complete=True,counts=counts,episodes=len(actual_rows),environment_steps=steps)))
    finally:runner.close();runtime.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True);parser.add_argument('--physical-gpu',type=int);parser.add_argument('--slots',type=int,default=8);parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args();run(args)
