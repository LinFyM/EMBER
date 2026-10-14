#!/usr/bin/env python3
"""One-shot frozen §19 input dependence; canonical Compiler and FP32 Gram only."""
from pathlib import Path
from types import SimpleNamespace
from itertools import product
from collections import Counter
import argparse, json, math, os, statistics, subprocess, time

METRICS=('d_E_target','d_E_partner','d_X_target','d_X_partner','d_MT','q')
DEMOS=(46,47,48,49)


def read(path):return json.loads(Path(path).read_text())
def finite(value):
    value=float(value)
    if not math.isfinite(value):raise ValueError('nonfinite operator distance')
    return value
def distance(objective,X,Y):return finite(objective(X,Y)[0])
def preference(dt,dp,den):return None if den==0 else finite((dt-dp)/den)


def physical_keys(arm,a,b,demo):
    return [(arm,v,demo,h,l) for v,h,l in product((a,b),(a,b),(a,b) if arm=='VL' else (None,))]
def logical_key(arm,t,p,demo,mask):
    return (arm,(t,p)[mask[0]],demo,(t,p)[mask[1]],(t,p)[mask[2]] if arm=='VL' else None)
def physical_id(key):
    arm,v,d,h,l=key;return f'{arm}_v{v:04d}_d{d:02d}_H{h:04d}_L{l if l is not None else "none"}'


def validate(run,authority):
    tasks=run['training_tasks'];mapping={e['task']:e for e in run['mapping']}
    if (len(tasks)!=36 or tasks!=sorted(set(tasks)) or set(mapping)!=set(tasks)
            or len(run['mapping'])!=36 or run['demo_ordinals']!=list(DEMOS) or run['arms']!=['V','VL']
            or run['source_checkpoint_update']!=25920 or run['physical_forward_count']!=864 or run['logical_row_count']!=1728):
        raise ValueError('fixed36/144/864/1728 contract required')
    official={e['global_task_id']:e for e in authority['tasks']}
    for i,t in enumerate(tasks):
        row=mapping[t];p=row['paired_task']
        if (p!=tasks[(i+18)%36] or mapping[p]['paired_task']!=t or p==t
                or row['suite']!=official[t]['suite'] or row['paired_suite']!=official[p]['suite']
                or row['suite']==row['paired_suite'] or row['paired_role']!=mapping[p]['role']):
            raise ValueError('original explicit involution and authority suite required')
    return mapping,official,[(t,mapping[t]['paired_task']) for t in tasks if t<mapping[t]['paired_task']]


def select_inputs(inputs,key):
    """L has its own frozen source, never the language attached to the video."""
    arm,v,d,h,l=key
    video=inputs._resident(('analysis_video',v,d),{k:inputs.host[v,d][k] for k in ('z_img','indices')})
    history=inputs._resident(('MT_H1',h),inputs.histories[h])
    language=None if arm=='V' else inputs._resident(('analysis_language',l,d),{'language':inputs.host[l,d]['language']})['language']
    return video['z_img'],video['indices'],history,language


def sources(parent,key,official,histories):
    from ember.proposal_writer.image_data import feature_file
    from ember.proposal_writer.image_history import history_file
    arm,v,d,h,l=key
    return dict(video=dict(task=v,demo=d,cache=str(feature_file(parent,v,d)),field=['z_img','indices'],
        raw_RGB=official[v]['hdf5']['path']),H=dict(task=h,cache=str(history_file(parent,h)),
        fields=['images','prepost','facts','record_indices'],raw_prefix=histories[h]['H']),
        L=None if arm=='V' else dict(task=l,cache=str(feature_file(parent,l,d)),field='language',
            exact_language=official[l]['authority']['language']))


def statistics_for(rows,metrics=METRICS):
    tasks=sorted({r['target'] for r in rows});result=dict(rows=len(rows),tasks=len(tasks),
        demos=sorted({r['demo'] for r in rows}),per_task_row_count={str(t):sum(r['target']==t for r in rows) for t in tasks},metrics={})
    for name in metrics:
        task_means=[]
        for task in tasks:
            values=[r[name] for r in rows if r['target']==task and r[name] is not None]
            if values:task_means.append(statistics.mean(values))
        defined=sum(r[name] is not None for r in rows)
        result['metrics'][name]=dict(mean=statistics.mean(task_means) if task_means else None,
            defined_rows=defined,undefined_rows=len(rows)-defined,defined_tasks=len(task_means),
            weighting='task equal; four demos equal within each registered mask')
    return result


def details(rows,metrics=METRICS):
    groups={'per_task':('target',str),'per_demo':('demo',str),'per_suite':('stratum',str)}
    output=dict(all=statistics_for(rows,metrics))
    for name,(field,label) in groups.items():
        output[name]={label(k):statistics_for([r for r in rows if r[field]==k],metrics) for k in sorted({r[field] for r in rows})}
    output['E_sources']={field:{label:statistics_for([r for r in rows if r.get(field)==flag],metrics) for label,flag in [('zero',True),('nonzero',False)]}
        for field in ('E_target_zero','E_partner_zero','video_E_zero','H_E_zero','L_E_zero') if any(r.get(field) is not None for r in rows)}
    output['zero_E_pair_denominator']=sum(r.get('E_pair_denominator')==0 for r in rows) if any('E_pair_denominator' in r for r in rows) else None
    return output


def aggregate(rows,other):
    output={}
    for arm in ('V','VL'):
        part=[r for r in rows if r['arm']==arm];others=[r for r in other if r['arm']==arm]
        output[arm]=dict(**details(part),per_mask={m:details([r for r in part if r['mask']==m]) for m in sorted({r['mask'] for r in part})},
            same_task_other=details(others,('d_X_other',)))
    return dict(arms=output,logical_rows=len(rows),same_task_other_rows=len(other),
        q_is_teacher_direction_only=True,zero_denominator_q=None,no_thresholds_or_checkpoint_selection=True)


def run(root,device='cuda:0'):
    import torch
    from safetensors.torch import load_file
    from ember.proposal_writer.image_learning import load_compiler
    from ember.proposal_writer.image_data import ImageInputs
    from ember.proposal_writer.learning import OperatorObjective,memory_peak
    from ember.writer.runtime import autocast
    from ember.pi05_source_checkpoint import write_json_atomic as write
    root=Path(root);contract=read(root/'run_contract.json')
    parent=Path(contract['parent'])
    mapping,official,pairs=validate(contract,read(parent/'task_authorities.json'))
    if (root/'completion.json').exists() or any((root/f).exists() for f in ('physical_rows.jsonl','rows.jsonl','same_task_other.jsonl')):
        raise ValueError('one-shot readout cannot repeat saved physical forwards')
    device=torch.device(device)
    if device.type!='cuda':raise ValueError('formal consumer uses registered actual CUDA0; CPU checks use only small helpers')
    torch.cuda.set_device(device);started=time.monotonic();models={}
    code=os.environ.get('EMBER_SOURCE_COMMIT') or contract.get('source_commit') or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    checkpoints={a:Path(contract['checkpoints'][a]) for a in ('V','VL')}
    for arm,checkpoint in checkpoints.items():
        model,lora,state=load_compiler(parent,arm,checkpoint,device)
        if state['next_update']!=25920:raise ValueError('only original complete25920')
        models[arm]=model.eval().requires_grad_(False);del state
    objective=OperatorObjective(SimpleNamespace(lora=lora),parent);inputs=ImageInputs(parent,device)
    mt=load_file(str(parent/'initial.safetensors'),device=str(device));scale=read(parent/'operator_scale.json')
    teacher_files={r['task']:r['weights'] for r in scale['sources']}
    histories={e['task']:e for e in read(parent/'history_manifest.json')['training']}
    old_nodes={a:{(r['task'],r['demo']):r for r in read(parent/'arms'/a/'nodes/update_00025920.json')['rows']} for a in ('V','VL')}
    setup=time.monotonic()-started;rows=[];others=[];compat=[];physical_counts=Counter();physical_ids=set()
    with (root/'physical_rows.jsonl').open('x') as physical, (root/'rows.jsonl').open('x') as logical, (root/'same_task_other.jsonl').open('x') as changed, torch.no_grad():
        emit=lambda stream,row:stream.write(json.dumps(row,allow_nan=False)+'\n')
        for a,b in pairs:
            teachers={t:load_file(teacher_files[t],device=str(device)) for t in (a,b)}
            den=distance(objective,teachers[a],teachers[b])
            energy={t:finite(objective(mt,teachers[t])[1]) for t in (a,b)};correct={}
            for demo in DEMOS:
                for arm in ('V','VL'):
                    outputs={};raw={}
                    for key in physical_keys(arm,a,b,demo):
                        identity=physical_id(key)
                        if identity in physical_ids:raise ValueError('duplicate physical Compiler input')
                        tick=time.monotonic();first=not physical_ids
                        if first:torch.cuda.reset_peak_memory_stats(device)
                        args=select_inputs(inputs,key)
                        with autocast(device):state=models[arm](*args)
                        if set(state)!=set(mt) or any(state[k].shape!=mt[k].shape for k in state):raise ValueError('complete38 target output shape')
                        outputs[key]=state
                        distances={str(t):distance(objective,state,teachers[t]) for t in (a,b)}
                        base=dict(physical_id=identity,key=list(key),arm=arm,pair=[a,b],demo=demo,
                            input_sources=sources(parent,key,official,histories),checkpoint=str(checkpoints[arm]),
                            d_E_by_task=distances,d_MT=distance(objective,state,mt),E_pair_denominator=den,
                            label_paths={str(t):teacher_files[t] for t in (a,b)},MT_path=str(parent/'initial.safetensors'))
                        raw[key]=base;emit(physical,base);physical_ids.add(identity);physical_counts[arm]+=1
                        if first:
                            torch.cuda.synchronize(device);write(root/'first_consumer.json',dict(**base,
                                elapsed_seconds=time.monotonic()-tick,setup_seconds=setup,peak=memory_peak(device),
                                no_grad=not torch.is_grad_enabled(),eval=True,source_commit=code,counts_toward_864=True))
                    for t,p in ((a,b),(b,a)):
                        full=(arm,t,demo,t,t if arm=='VL' else None);partner=(arm,p,demo,p,p if arm=='VL' else None)
                        old=old_nodes[arm][t,demo]['BA'];new=raw[full]['d_E_by_task'][str(t)]
                        compat.append(dict(arm=arm,task=t,demo=demo,physical_id=physical_id(full),original_BA=old,new_d_E=new,difference=new-old))
                        correct[arm,t,demo]={k:v.detach().cpu() for k,v in outputs[full].items()}
                        for mask in product((0,1),repeat=3 if arm=='VL' else 2):
                            key=logical_key(arm,t,p,demo,mask);base=raw[key];state=outputs[key];v,h,l=key[1],key[3],key[4]
                            dt,dp=base['d_E_by_task'][str(t)],base['d_E_by_task'][str(p)]
                            row=dict(arm=arm,target=t,paired_task=p,demo=demo,mask=''.join(map(str,mask)),
                                switches=dict(video=bool(mask[0]),H=bool(mask[1]),L=bool(mask[2]) if arm=='VL' else None),
                                input_sources=base['input_sources'],physical_id=base['physical_id'],
                                correct_target_physical_id=physical_id(full),correct_partner_physical_id=physical_id(partner),
                                d_E_target=dt,d_E_partner=dp,d_X_target=distance(objective,state,outputs[full]),
                                d_X_partner=distance(objective,state,outputs[partner]),d_MT=base['d_MT'],q=preference(dt,dp,den),
                                E_pair_denominator=den,q_defined=den!=0,E_target_zero=energy[t]==0,E_partner_zero=energy[p]==0,
                                video_E_zero=energy[v]==0,H_E_zero=energy[h]==0,L_E_zero=None if l is None else energy[l]==0,
                                suite=mapping[t]['suite'],role=mapping[t]['role'],
                                stratum='support' if mapping[t]['role']=='source_support' else mapping[t]['suite'])
                            rows.append(row);emit(logical,row)
                    del outputs,state,args,raw
            for arm in ('V','VL'):
                for task in (a,b):
                    for i,demo in enumerate(DEMOS):
                        next_demo=DEMOS[(i+1)%4]
                        row=dict(arm=arm,target=task,demo=demo,other_demo=next_demo,
                            correct_physical_id=physical_id((arm,task,demo,task,task if arm=='VL' else None)),
                            other_physical_id=physical_id((arm,task,next_demo,task,task if arm=='VL' else None)),
                            d_X_other=distance(objective,correct[arm,task,demo],correct[arm,task,next_demo]),extra_forward=0,
                            E_target_zero=energy[task]==0,suite=mapping[task]['suite'],role=mapping[task]['role'],
                            stratum='support' if mapping[task]['role']=='source_support' else mapping[task]['suite'])
                        others.append(row);emit(changed,row)
            del correct,teachers
    expected=dict(V=288,VL=576)
    if dict(physical_counts)!=expected or Counter(r['arm'] for r in rows)!=dict(V=576,VL=1152) or Counter(r['arm'] for r in others)!=dict(V=144,VL=144):
        raise ValueError('complete physical/logical/other denominators required')
    difference=[abs(r['difference']) for r in compat]
    write(root/'fixed_correct_compatibility.json',dict(rows=compat,count=len(compat),max_abs_difference=max(difference),
        mean_abs_difference=statistics.mean(difference),original_nodes={a:str(parent/'arms'/a/'nodes/update_00025920.json') for a in ('V','VL')},
        precision='canonical BF16 Compiler / FP32 Gram; raw differences only, no rerun or dtype/checkpoint selection'))
    summary=aggregate(rows,others);summary.update(physical_forward_count=len(physical_ids),physical_by_arm=dict(physical_counts))
    write(root/'aggregate.json',summary);torch.cuda.synchronize(device)
    write(root/'completion.json',dict(complete=True,source_commit=code,source_training_commit=contract['source_training_commit'],
        parent=str(parent),checkpoints={a:str(p) for a,p in checkpoints.items()},physical_forward_count=len(physical_ids),
        physical_by_arm=dict(physical_counts),logical_row_count=len(rows),logical_by_arm=dict(Counter(r['arm'] for r in rows)),
        same_task_other_by_arm=dict(Counter(r['arm'] for r in others)),elapsed_seconds=time.monotonic()-started,
        peak=memory_peak(device),frozen_input_GPU_bytes=inputs.bytes,frozen_input_GPU_limit=inputs.limit,
        no_grad_eval=True,source_forward=0,native_action_forward=0,environment_starts=0,environment_steps=0,
        training_updates=0,held=False,Test=False,persisted_generated_LoRAs=0,no_automatic_successor=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args();run(args.root,args.device)
