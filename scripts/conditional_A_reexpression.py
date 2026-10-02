#!/usr/bin/env python3
"""Eight-condition frozen reexpression and its two registered execution consumers."""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file

from ember.operator_writer import reexpression as owner
from ember.operator_writer.data import FormalData
from ember.operator_writer.run import build_runtime,frozen_git
from ember.operator_writer.functional_readout import fixed_flow,fm_prediction,generation_prediction,masked_risk,LocalEffects
from ember.operator_writer.bank import BANK_SCHEMA
from ember.lora import LORA_A_SUFFIX,LORA_B_SUFFIX,validate_lora_state
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.writer.materialization import file_record
from ember.writer.materialization_workers import _configure_device

ASSET=Path('/data1/user/ymdai/projects/EMBER')


def functional_readback():
    records,unfavorable,comparisons=[],[],[]
    for task in owner.TASKS:
        panel=read_json(owner.joint_readout.FIXED_PANEL/'fixed_panels.json')[str(task)]
        flow,_=fixed_flow(task,panel)
        for teacher in owner.TEACHERS[task]:
            key=f'task{task:03d}_teacher{teacher:02d}'
            record=read_json(owner.ROOT/'functional'/f'{key}.json')
            if record['queries']!=panel['A'] or record['flow_seed']!=panel['A_flow_seed']:
                raise ValueError('completed functional rows lost original A28 identity')
            raw={r['arm']:torch.load(r['raw']['path'],map_location='cpu',weights_only=False) for r in record['rows']}
            for value in raw.values():
                if value['prediction_FM'].shape!=(28,50,7) or value['action_generated_normalized'].shape!=(28,50,7):
                    raise ValueError('completed function/ten-step query counts changed')
            a,b=record['rows'];valid=raw['Original']['valid_mask']
            signs=(raw['Original']['action_generated_normalized'].sign()!=raw['Reexpressed']['action_generated_normalized'].sign())
            for i,query in enumerate(panel['A']):
                row=dict(task=task,teacher=teacher,query=query,query_ordinal=i,
                    FM_full50_delta=b['FM']['per_query_full50'][i]-a['FM']['per_query_full50'][i],
                    FM_first5_delta=b['FM']['per_query_first5'][i]-a['FM']['per_query_first5'][i],
                    generated_full50_delta=b['generation']['per_query_full50'][i]-a['generation']['per_query_full50'][i],
                    generated_first5_delta=b['generation']['per_query_first5'][i]-a['generation']['per_query_first5'][i],
                    generated_valid_future_delta=b['generation']['per_query_valid_future'][i]-a['generation']['per_query_valid_future'][i],
                    generated_sign_changes_first5=int(signs[i,:5].sum()),
                    generated_motion6_sign_changes_valid=int((signs[i,:,:6]*valid[i,:,None]).sum()),
                    generated_gripper_sign_changes_valid=int((signs[i,:,6]*valid[i]).sum()))
                comparisons.append(row)
                if any(row[k]>0 for k in ('FM_full50_delta','FM_first5_delta','generated_valid_future_delta','generated_first5_delta')):
                    unfavorable.append(row)
            records.append(record)
    fits=[dict(task=task,teacher=teacher,**fit) for task in owner.TASKS for teacher in owner.TEACHERS[task]
          for fit in read_json(owner.ROOT/'compilation'/f'task{task:03d}_teacher{teacher:02d}.json')['fits']]
    if len(fits)!=304 or sum(len(r['rows']) for r in records)!=16:
        raise ValueError('functional readback did not cover8x38 and two arms')
    aggregates={}
    for arm in owner.ARMS:
        rows=[row for record in records for row in record['rows'] if row['arm']==arm]
        aggregates[arm]={kind:{k:sum(row[kind][k] for row in rows)/8 for k in
            ('full50','first5','full50_motion6','full50_gripper1','valid_future','valid_first5')}
            for kind in ('FM','generation')}
    return dict(condition_records=records,teacher_fit=fits,aggregates=aggregates,
        query_comparisons=comparisons,all_unfavorable_queries=unfavorable,
        FM_records=448,generation_records=448,unique_condition_queries=224,unique_task_queries=112)


def paired_closed_loop():
    indexed={};paths=[];full=0
    for arm in owner.ARMS:
        rows=[]
        for slot in (0,1):
            path=owner.ROOT/arm/'evaluation'/f'teacher{slot}'/'results.json';paths.append(file_record(path))
            rows.extend(read_json(path)['rows'])
        if len(rows)!=32:raise ValueError('finite arm did not complete32 rows')
        indexed[arm]={}
        for row in rows:
            e=row['operator_read_write_lora'];key=(e['global_task_id'],e['teacher_demo'],row['init_state_id'])
            if key in indexed[arm] or key[0] not in owner.TASKS or key[1] not in owner.TEACHERS[key[0]] or key[2] not in owner.STATES:
                raise ValueError('closed-loop duplicated/unregistered condition-state')
            indexed[arm][key]=row
            for field in ('continuous_control_trace','occupancy_trajectory','stage_predicates'):
                if not row.get(field):raise ValueError('row missing action/continuous/goal/compact evidence')
            trajectory=row['occupancy_trajectory'];trace=row['continuous_control_trace']['trace']
            if (trajectory['capture_level']!=('full' if key[2]==0 else 'compact')
                or trace['steps']!=row['steps'] or trace['samples']!=row['steps']+1):
                raise ValueError('full16/all-step passive recording changed')
            for item in (trajectory,trace):
                if not Path(item['path']).is_file() or Path(item['path']).stat().st_size!=item['bytes']:
                    raise ValueError('trajectory/continuous raw artifact absent or changed size')
            full+=key[2]==0
    if set(indexed['Original'])!=set(indexed['Reexpressed']) or full!=16:
        raise ValueError('paired scene panel/full capture identities changed')
    records=[]
    for key,a in sorted(indexed['Original'].items()):
        b=indexed['Reexpressed'][key]
        n=min(len(a['policy_noise_seeds']),len(b['policy_noise_seeds']))
        if (a['scene_reference']!=b['scene_reference'] or a['env_seed']!=b['env_seed']
            or a['policy_seed_root']!=b['policy_seed_root'] or a['policy_noise_seeds'][:n]!=b['policy_noise_seeds'][:n]):
            raise ValueError('finite scene/env/policy RNG pairing violated')
        records.append(dict(task=key[0],teacher=key[1],state=key[2],Original=a,Reexpressed=b))
    return records,paths


def success_summary(records):
    sets={arm:{(r['task'],r['teacher'],r['state']) for r in records if r[arm]['success']} for arm in owner.ARMS}
    original,edited=sets['Original'],sets['Reexpressed']
    return dict(rows=len(records),Original=len(original),Reexpressed=len(edited),
        retained=len(original&edited),gained=len(edited-original),lost=len(original-edited),
        churn=len(original^edited),success_set_jaccard=len(original&edited)/len(original|edited) if original|edited else 1,
        success_sets={arm:sorted(v) for arm,v in sets.items()},gained_keys=sorted(edited-original),lost_keys=sorted(original-edited))


def readback():
    import numpy as np
    records,paths=paired_closed_loop();functional=functional_readback()
    summary=success_summary(records)
    by_task={str(t):success_summary([r for r in records if r['task']==t]) for t in owner.TASKS}
    by_teacher={f'{t}/{teacher}':success_summary([r for r in records if r['task']==t and r['teacher']==teacher])
        for t in owner.TASKS for teacher in owner.TEACHERS[t]}
    delta=np.asarray([[sum(int(r['Reexpressed']['success'])-int(r['Original']['success'])
        for r in records if r['task']==t and r['state']==s) for s in owner.STATES] for t in owner.TASKS])
    indices=np.random.default_rng(20261002).integers(0,4,(20000,4,4))
    sampled=delta[np.arange(4)[None,:,None],indices].sum((1,2))
    processes=read_json(owner.ROOT/'launch/batch_processes.json')
    output=dict(study=owner.ROOT.name,scope='fixed train4 x teacher2 x state4 x arm2; no qualification or selection',
        reading_git=frozen_git(),consumer_contract=file_record(owner.ROOT/'launch/consumer_contract.json'),
        functional=functional,closed_loop=dict(aggregate=summary,per_task=by_task,per_teacher=by_teacher,
            paired_rows=records,raw_paths=paths,all_losses=[r for r in records if r['Original']['success'] and not r['Reexpressed']['success']],
            all_failed_cases=[dict(arm=arm,row=r) for r in records for arm in owner.ARMS if not r[arm]['success']],
            bootstrap=dict(unit='within fixed task, resample four physical states; both teachers kept together',
                repetitions=20000,delta_successes_CI95=np.quantile(sampled,[.025,.975]).tolist())),
        resources=processes,training_updates=0)
    write_json_atomic(owner.ROOT/'readback.json',output)
    write_json_atomic(owner.ROOT/'completion.json',dict(status='complete',conditions=8,paired_LoRA_files=16,
        full_targets=38,FM_queries=448,generation_queries=448,closed_loop_rows=64,physical_initial_states=16,
        full_capture_rows=16,training_updates=0,GPU_hours=processes['GPU_hours'],
        process_exit_codes=[r['exit_code'] for r in processes['processes']],readback=file_record(owner.ROOT/'readback.json')))
    print('complete',summary,flush=True)


def compilation(runtime,data,task,teacher,frame_chunk):
    key=f'task{task:03d}_teacher{teacher:02d}'
    record=owner.ROOT/'compilation'/f'{key}.json'
    paths={arm:owner.ROOT/arm/'bank'/f'{key}.safetensors' for arm in owner.ARMS}
    sm_path=record.with_name(key+'_original_SM.safetensors')
    if record.exists():
        info=read_json(record)
        if info['checkpoint']!=str(owner.CHECKPOINT) or info['formula']!=owner.FORMULA:
            raise ValueError('partial compilation belongs to a different source/intervention')
        states={arm:load_file(str(path)) for arm,path in paths.items()}
        sm=load_file(str(sm_path))
        for state in states.values():validate_lora_state(state,runtime.lora)
        return states,{name:sm[name+'.S'] for name in runtime.writer.names},info
    condition,raw,sampled=data.condition(runtime,task,teacher)
    with torch.no_grad():
        original,native=runtime.compile(condition,frame_chunk=frame_chunk,capture_mechanism=True)
    fields=native['mechanism']['targets']
    original={k:v.detach().float().cpu().contiguous() for k,v in original.items()}
    edited,coefficients,fits=owner.reexpress_compilation(original,native,runtime.writer.names)
    sm={name+'.'+k:fields[name][k].detach().float().cpu().contiguous()
        for name in runtime.writer.names for k in ('S','M')}
    for arm,state in zip(owner.ARMS,(original,edited),strict=True):
        validate_lora_state(state,runtime.lora)
        paths[arm].parent.mkdir(parents=True,exist_ok=True)
        save_file(state,str(paths[arm]),metadata=dict(schema_version=BANK_SCHEMA,mode=arm,condition_id=key))
    record.parent.mkdir(parents=True,exist_ok=True)
    save_file(sm,str(sm_path))
    coef_path=record.with_name(key+'_C.safetensors');save_file(coefficients,str(coef_path))
    info=dict(task=task,teacher=teacher,condition_id=key,checkpoint=str(owner.CHECKPOINT),
        formula=owner.FORMULA,native_reads=1,frame_indices=condition[1].cpu().tolist(),
        raw_frames=raw,sampled_frames=sampled,probe_seed=1729,tau=1,frame_stride=5,
        original_SM=file_record(sm_path),coefficients=file_record(coef_path),
        factors={arm:file_record(path) for arm,path in paths.items()},fits=fits)
    write_json_atomic(record,info)
    runtime.writer.last_mechanism={}
    return dict(Original=original,Reexpressed=edited),{name:sm[name+'.S'] for name in runtime.writer.names},info


def read_condition(runtime,states,s_edits,batch,flow,valid,task,teacher,microbatch,reference):
    key=f'task{task:03d}_teacher{teacher:02d}'
    path=owner.ROOT/'functional'/f'{key}.json'
    if path.exists():return read_json(path)
    effects=LocalEffects(runtime.policy,states['Original'],states['Reexpressed'],s_edits,runtime.device,microbatch)
    rows=[]
    for arm in owner.ARMS:
        with effects.capture('FM') if arm=='Original' else torch.no_grad():
            velocity=fm_prediction(runtime,{k:v.to(runtime.device) for k,v in states[arm].items()},batch,flow,microbatch)
        with effects.capture('generation') if arm=='Original' else torch.no_grad():
            generated=generation_prediction(runtime,{k:v.to(runtime.device) for k,v in states[arm].items()},batch,flow,microbatch)
        raw=owner.ROOT/'functional'/f'{key}_{arm}.pt'
        raw.parent.mkdir(parents=True,exist_ok=True)
        physical=runtime.processor.unnormalize_action(generated.to(runtime.device)).cpu()
        torch.save(dict(prediction_FM=velocity,action_generated_normalized=generated,
            action_generated_physical=physical,executed_first5=physical[:,:5],valid_mask=valid,
            queries=flow['queries'],flow_seed=flow['flow_seed'],target_ref=file_record(reference),
            noise_field='noise',time_field='time',generation_steps=10,query_offset=1),raw)
        rows.append(dict(arm=arm,task=task,teacher=teacher,FM=masked_risk(velocity,flow['FM_target'][...,:7],valid),
            generation=masked_risk(generated,flow['action'],valid),raw=file_record(raw)))
    record=dict(task=task,teacher=teacher,queries=flow['queries'],flow_seed=flow['flow_seed'],
        target_ref=file_record(reference),rows=rows,local_effects=effects.summary())
    write_json_atomic(path,record)
    return record


def produce(args):
    spec,training,spec_path=owner.source_record()
    git=frozen_git();owner.register_inputs()
    panels=read_json(owner.joint_readout.FIXED_PANEL/'fixed_panels.json')
    write_json_atomic(owner.ROOT/'launch/consumer_contract.json',dict(checkpoint=str(owner.CHECKPOINT),
        source=training['source'],training_git=training['git']['commit'],training_spec=training['spec'],
        reading_git=git,reading_spec=file_record(spec_path),fixed_panels=file_record(owner.joint_readout.FIXED_PANEL/'fixed_panels.json'),
        formula=owner.FORMULA,updates=0,tasks=list(owner.TASKS),teachers=owner.TEACHERS,
        FM_queries=448,generation_queries=448,environment_rows=64,physical_initial_states=16,
        native_reads=8,full_sites=38,microbatch=args.microbatch,frame_chunk=args.frame_chunk))
    started=time.monotonic();_configure_device(torch.device('cuda:0'),args.cpu_threads)
    runtime=build_runtime(ASSET,spec,torch.device('cuda:0'),owner.joint_readout.CONDITIONAL_MODE)
    runtime.writer.load_state_dict(load_file(str(owner.CHECKPOINT/'ecp.safetensors'),device='cuda:0'),strict=True)
    runtime.writer.requires_grad_(False).eval();runtime.policy.eval()
    common={k:v.detach().float().cpu().contiguous() for k,v in runtime.writer.public_state().items()}
    save_file(common,str(owner.ROOT/'public.safetensors'))
    teacher_data=FormalData(ASSET,spec,query_labels=False,task_ids=owner.TASKS)
    try:
        for task in owner.TASKS:
            for teacher in owner.TEACHERS[task]:
                compilation(runtime,teacher_data,task,teacher,args.frame_chunk)
                print('compiled',task,teacher,flush=True)
    finally:teacher_data.close()
    # Labelled execution inputs become accessible only after all analytic C are sealed.
    flows={task:fixed_flow(task,panels[str(task)]) for task in owner.TASKS}
    query_data=FormalData(ASSET,spec,task_ids=owner.TASKS)
    records=[]
    try:
        for task in owner.TASKS:
            flow,reference=flows[task]
            raw=query_data.batch(dict(task=task,teacher_demo=owner.TEACHERS[task][0],queries=panels[str(task)]['A']))
            valid=~raw['action_is_pad'];batch=runtime.processor.training_batch(raw)
            for teacher in owner.TEACHERS[task]:
                key=f'task{task:03d}_teacher{teacher:02d}'
                states={arm:load_file(str(owner.ROOT/arm/'bank'/f'{key}.safetensors')) for arm in owner.ARMS}
                sm=load_file(str(owner.ROOT/'compilation'/f'{key}_original_SM.safetensors'))
                records.append(read_condition(runtime,states,{n:sm[n+'.S'] for n in runtime.writer.names},
                    batch,flow,valid,task,teacher,args.microbatch,reference))
                print('read',task,teacher,flush=True)
    finally:query_data.close()
    for arm in owner.ARMS:
        for slot in (0,1):owner.register_bank(arm,slot,runtime.lora,training,spec_path,git)
    write_json_atomic(owner.ROOT/'producer_completion.json',dict(status='complete',conditions=len(records),
        full_sites=38,native_reads=8,FM_queries=448,generation_queries=448,updates=0,
        seconds=time.monotonic()-started,peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',nargs='?',choices=('produce','readback'),default='produce')
    parser.add_argument('--microbatch',type=int,choices=(7,14,28),default=14)
    parser.add_argument('--frame-chunk',type=int,default=8)
    parser.add_argument('--cpu-threads',type=int,default=6)
    args=parser.parse_args()
    if args.frame_chunk<1 or args.cpu_threads<1:parser.error('packing must be positive')
    if args.phase=='readback':readback()
    else:produce(args)


if __name__=='__main__':main()
