"""Frozen per-video LoRAs and two preregistered, paired checkpoint reports."""
from pathlib import Path
import gc
import time
import torch
from safetensors.torch import save_file

from ember.lora import validate_lora_state
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.reference.analysis import paired,pairing_valid,read_cases
from ember.reference.accounting import summary
from ember.proposal_writer.analysis import billing
from .model import load_basis
from .readout import inputs,panels,phase_root,generated_path,saved_case,UPDATES

HISTORY=Path('/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261014')
G165=Path('/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/full_writer_after24849_20261013/diagnostics/first_G_validation400_20261013/results.json')
MT153=Path('/data1/user/ymdai/ember_runs/operator_read_write_learning_20260928/stage1/MT/evaluation/correct400/results.json')
ARMS=('source','reference','prior144','operator1170','linear64','generated')
COMPARISONS=tuple((a,b) for i,a in enumerate(ARMS) for b in ARMS[i+1:])


def materialize_shard(args,runtime,rows):
    """Only RGB/indices enter Reader; labels and task metadata stay outside."""
    from .reader import ImageReader
    from .features import load_cached,extract,load_video
    run=inputs(args.root)[0];tick=time.monotonic()
    missing=[]
    for row in rows:
        destination=generated_path(args.root,args.update,row)
        if not destination.is_file():missing.append(row);continue
        metadata=read_json(destination.with_suffix('.json'))
        expected=dict(reader_update=args.update,basis_checkpoint=run['frozen_basis'],role=row['role'],
            task=row['task'],init_state_id=row['init_state_id'],teacher_demo=row['teacher_demo'],path=str(destination))
        if any(metadata.get(k)!=v for k,v in expected.items()):raise ValueError('saved condition LoRA identity changed')
    if not missing:return dict(generated=0,reused=len(rows),seconds=0.)
    checkpoint=args.root/'checkpoints'/f'update_{args.update:08d}'/'state.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if (saved.get('next_update',saved.get('update'))!=args.update or saved.get('phase')!='video_initial'
            or saved.get('schema')!=run['schema'] or saved.get('frozen_basis')!=run['frozen_basis']):
        raise ValueError('Reader checkpoint is not the registered endpoint')
    reader=ImageReader().to(runtime.device);reader.load_state_dict(saved['reader']);reader.eval().requires_grad_(False)
    units=saved['code_units'].to(runtime.device,dtype=torch.float32)
    if units.shape!=(64,) or not bool(torch.isfinite(units).all()) or bool((units<1e-6).any()):
        raise ValueError('invalid fixed bank-only code units')
    decoder=load_basis(Path(run['frozen_basis'])).to(runtime.device).requires_grad_(False)
    cache={(r['task'],r['demo']):r for r in read_json(args.root/'features/manifest.json')['entries'] if r['kind']=='original'}
    entries=[]
    try:
        with torch.no_grad():
            for row in missing:
                descriptor=cache.get((row['task'],row['teacher_demo']))
                if row['role']=='seen':
                    if descriptor is None:raise ValueError('seen video has no registered frozen feature')
                    feature=load_cached(Path(descriptor['cache_path']),runtime.device)
                else:
                    # Held RGB is read one video at a time, without offline labels or persistent prefix cache.
                    video=load_video(runtime,dict(kind='original',task=row['task'],demo=row['teacher_demo']))
                    feature=extract(runtime,video,getattr(args,'frame_chunk',8));del video
                if Path(feature['source']['model']).resolve()!=Path(runtime.source['model_path']).resolve():
                    raise ValueError('teaching feature belongs to a different frozen source')
                feature['z_img']=feature['z_img'].to(runtime.device)
                feature['indices']=feature['indices'].to(runtime.device)
                with torch.autocast(device_type=runtime.device.type,dtype=torch.bfloat16,enabled=runtime.device.type=='cuda'):
                    y=reader(feature['z_img'],feature['indices'])
                code=units*y.float();state=decoder.decode(code)
                validate_lora_state(state,runtime.lora)
                if not bool(torch.isfinite(code).all()) or any(not bool(torch.isfinite(v).all()) for v in state.values()):
                    raise ValueError('nonfinite generated LoRA is not executable')
                destination=generated_path(args.root,args.update,row);destination.parent.mkdir(parents=True,exist_ok=True)
                temporary=destination.with_suffix('.partial')
                save_file({k:v.detach().cpu().float().contiguous() for k,v in state.items()},str(temporary))
                metadata=dict(phase='video_initial',reader_update=args.update,basis_checkpoint=run['frozen_basis'],
                    checkpoint=str(checkpoint),role=row['role'],task=row['task'],init_state_id=row['init_state_id'],
                    teacher_demo=row['teacher_demo'],path=str(destination),rank=8,targets=38,code=code.cpu().tolist(),
                    held_offline_labels_read=False,teacher_input='pure_image_prefix_and_frame_indices')
                write_json_atomic(destination.with_suffix('.json'),metadata);temporary.rename(destination);entries.append(metadata)
                del feature,y,code,state
    finally:
        del reader,decoder,saved,units;gc.collect()
        if runtime.device.type=='cuda':torch.cuda.empty_cache()
    return dict(generated=len(entries),reused=len(rows)-len(entries),seconds=time.monotonic()-tick,
        held_cache_disk_bytes=0,Reader_and_decoder_released_before_execution=True)



def paired_rows(before,after):
    """Old MT lacks proprio; its retained scene and actual noise identify pairing."""
    a=sorted(before,key=lambda r:r['init_state_id']);b=sorted(after,key=lambda r:r['init_state_id'])
    if len(a)!=len(b):return False
    fields=('suite','task_id','init_state_id','env_seed','policy_seed_root')
    for x,y in zip(a,b,strict=True):
        if any(x.get(k)!=y.get(k) for k in fields):return False
        noises=[r.get('first_noise_seed',r.get('policy_noise_seeds',[None])[0]) for r in (x,y)]
        if noises[0] is None or noises[0]!=noises[1]:return False
        scenes=[r.get('scene_reference') for r in (x,y)]
        if any(scenes) and (not all(scenes) or any(scenes[0].get(k)!=scenes[1].get(k) for k in ('path','bytes'))):return False
        if all('initial_proprio' in r and 'first_noise_seed' in r for r in (x,y)):
            if not pairing_valid([x],[y]):return False
        elif not all(scenes):return False
    return True


def summarize_task(task,spec,arms,*,identity=False):
    anchor=next(iter(arms.values()))
    if not anchor or any(not values or not paired_rows(anchor,values) for values in arms.values()):
        raise ValueError(f'paired initial state/scene/native noise mismatch for task{task}')
    result=dict(task=task,suite=spec['suite'],group='target24' if task<40 else 'source_seen_support12',
        identity=identity,cases=len(anchor),success_sets={a:sorted(r['init_state_id'] for r in rows if r['success']) for a,rows in arms.items()})
    comparisons=[(a,b) for i,a in enumerate(arms) for b in list(arms)[i+1:]]
    result['comparisons']={a+'_to_'+b:paired(arms[a],arms[b]) for a,b in comparisons}
    for value in result['comparisons'].values():
        union=sorted(set(value['before_successes'])|set(value['after_successes']))
        value.update(success_set_union=union,jaccard=len(value['retained'])/len(union) if union else 1.)
    if 'reference' in arms:
        gains=set(result['success_sets']['reference'])-set(result['success_sets']['source'])
        result['reference_new_gains']=sorted(gains)
        result['reference_gains_preserved']={a:sorted(gains&set(values)) for a,values in result['success_sets'].items()}
    return result


def aggregate(rows):
    if not rows:return dict(tasks=[],cases=0,successes={},breadth={},comparisons={})
    arms=rows[0]['success_sets'];comparisons=rows[0]['comparisons']
    result=dict(tasks=[r['task'] for r in rows],cases=sum(r['cases'] for r in rows),
        successes={a:sum(len(r['success_sets'][a]) for r in rows) for a in arms},
        breadth={a:sum(bool(r['success_sets'][a]) for r in rows) for a in arms},comparisons={})
    for name in comparisons:
        values=[r['comparisons'][name] for r in rows]
        result['comparisons'][name]={k:sum(len(v[k]) for v in values) for k in ('retained','gained','lost')}
        result['comparisons'][name].update(churn=sum(v['churn'] for v in values),success_set_overlap=sum(v['success_set_overlap'] for v in values),
            before=sum(v['before'] for v in values),after=sum(v['after'] for v in values),
            breadth_improved=sum(v['after']>v['before'] for v in values),
            success_set_union=sum(len(v['success_set_union']) for v in values))
        union=result['comparisons'][name]['success_set_union']
        result['comparisons'][name]['jaccard']=sum(len(v['retained']) for v in values)/union if union else 1.
    if 'reference' in arms:
        result['reference_new_gains']=sum(len(r['reference_new_gains']) for r in rows)
        result['reference_gains_preserved']={a:sum(len(r['reference_gains_preserved'][a]) for r in rows) for a in arms}
    return result


def grouped(rows):
    return dict(total=aggregate(rows),tasks=rows,
        suites={s:aggregate([r for r in rows if r['suite']==s]) for s in sorted({r['suite'] for r in rows})},
        groups={g:aggregate([r for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})},
        identity=aggregate([r for r in rows if r['identity']]),nonidentity=aggregate([r for r in rows if not r['identity']]))


def seen_history(run,bank,tasks,reference_root):
    roots={a:Path(run[k]) for a,k in [('prior144','parent_root'),('operator1170','operator_root'),('linear64','linear_root')]}
    result={}
    for task,item in bank.items():
        base=reference_root/'selection'/f'task_{task:04d}'
        arms=dict(source=read_cases(base/'update_00000000',range(16)),
            reference=read_cases(base/f'update_{item["selected_update"]:08d}',range(16)))
        arms.update({a:read_cases(root/'readout'/f'task_{task:04d}',range(16)) for a,root in roots.items()})
        # These explicit historical artifacts are compared, never loaded as current model state.
        if any(not v or not pairing_valid(arms['source'],v) for v in arms.values()):
            raise ValueError('historical complete36 source/reference/144/1170/linear pairing changed')
        result[task]=arms
    return result


def validation_history():
    d72=HISTORY/'operator_native_joint_learning/results.json'
    d144=HISTORY/'operator_D_adjacent_stability/results.json'
    source={name:read_json(path) for name,path in [('MT153',MT153),('G165',G165),('D72',d72),('D144',d144)]}
    rows={name:(value['arms']['operator_only']['validation']['raw_rows'] if name.startswith('D') else value.get('raw_rows',value.get('rows'))) for name,value in source.items()}
    for name,values in rows.items():
        if len(values)!=400 or sum(r['success'] for r in values)!={'MT153':153,'G165':165,'D72':175,'D144':168}[name]:
            raise ValueError('fixed historical validation arm identity/count changed')
    return rows,dict(MT153=str(MT153),G165=str(G165),D72=str(d72),D144=str(d144))


def checkpoint_readout(root,update,history=None,validation=None):
    run,bank,tasks,reference_root=inputs(root);schedule=panels(root);seen=[];held=[];raw={}
    history=seen_history(run,bank,tasks,reference_root) if history is None else history
    validation=validation_history()[0] if validation is None else validation
    for role in ('seen','validation'):
        rows=[saved_case(root,update,dict(r,role=role),tasks[r['task']],run) for r in schedule[role]];raw[role]=rows
        for task in sorted({r['task'] for r in schedule[role]}):
            current=[r for r in rows if r['global_task_id']==task]
            if role=='seen':
                arms=dict(history[task],generated=current)
                seen.append(summarize_task(task,tasks[task],arms,identity=bank[task]['selected_update']==0))
            else:
                arms={a:[r for r in values if r.get('global_task_id',10*('libero_spatial','libero_object','libero_goal','libero_10').index(r['suite'])+int(r['task_id']))==task] for a,values in validation.items()}
                arms['generated']=current;held.append(summarize_task(task,tasks[task],arms))
    return dict(reader_update=update,seen=grouped(seen),validation=grouped(held),raw_rows=raw)


def adjacent(first,second,tasks):
    result={}
    for role in ('seen','validation'):
        rows=[]
        for task in sorted({r['global_task_id'] for r in first['raw_rows'][role]}):
            arms={name:[r for r in panel['raw_rows'][role] if r['global_task_id']==task]
                for name,panel in [('update900',first),('update1800',second)]}
            identity=role=='seen' and next(r['identity'] for r in first['seen']['tasks'] if r['task']==task)
            rows.append(summarize_task(task,tasks[task],arms,identity=identity))
        result[role]=grouped(rows)
    return result


def mt_seen_subset(root,raw,reference_root,tasks):
    path=reference_root/'inputs/MT128_reference.json';previous=read_json(path)['rows'];rows=[]
    for task in sorted({r['global_task_id'] for r in previous}):
        arms=dict(MT=[r for r in previous if r['global_task_id']==task],
            generated=[r for r in raw if r['global_task_id']==task])
        rows.append(summarize_task(task,tasks[task],arms))
    return dict(source=str(path),new_MT_episodes=0,scope='original8task/128 subset only',**grouped(rows))


def analyze(root):
    root=Path(root);run,bank,tasks,reference_root=inputs(root)
    history=seen_history(run,bank,tasks,reference_root);validation,provenance=validation_history()
    nodes={str(update):checkpoint_readout(root,update,history,validation) for update in UPDATES
        if (phase_root(root,update)/'readout_completion.json').is_file() and read_json(phase_root(root,update)/'readout_completion.json')['complete']}
    for value in nodes.values():value['MT128_subset']=mt_seen_subset(root,value['raw_rows']['seen'],reference_root,tasks)
    complete=set(nodes)==set(map(str,UPDATES));selected=None
    if complete:
        selected=max(UPDATES,key=lambda u:(nodes[str(u)]['validation']['total']['successes']['generated'],-u))
    result=dict(complete=complete,phase='video_initial',schema=run['schema'],basis_checkpoint=run['frozen_basis'],
        checkpoints=nodes,selected_update=selected,selection_rule='higher strict correct400; tie earlier; no seen/loss/union selection',
        adjacent=adjacent(nodes['900'],nodes['1800'],tasks) if complete else None,
        validation_references=provenance,legacy_MT_pairing='registered scene path/bytes plus task/state/env/policy/actual first noise; original MT has no proprio field',limits=run['limits'],interactions=summary(root),billing=billing(root),
        actual_interaction_adaptation=0,identity_reused_rows=0,Test=False,
        scope='H-empty initial compilation; no video causality or final complete-H qualification claim')
    folder=root/'analysis';folder.mkdir(exist_ok=True);write_json_atomic(folder/'results.json',result)
    if complete:write_json_atomic(root/'completion.json',result)
    lines=['# 固定控制基底上的真实视频初始编译','',f'完整两点：{complete}；按correct400冻结：{selected}。','',
        '|update|seen /576|validation /400|参考新增288保存|source损失|','|---|---|---|---|---|']
    for u,node in nodes.items():
        seen=node['seen']['total'];lines.append(f"|{u}|{seen['successes']['generated']}|{node['validation']['total']['successes']['generated']}|{seen['reference_gains_preserved']['generated']}|{seen['comparisons']['source_to_generated']['lost']}|")
    lines+=['','完整逐task/suite、target/support/identity、全部参照与相邻R/G/L/churn/overlap/breadth见results.json。',
        '所有Reader条件（含五identity任务）均真实执行；seen为训练域诊断，held反馈不产生梯度。',
        '两点单独高点不证明稳定，也不证明教学动态内容或完整反馈方法已生效。']
    (folder/'report.md').write_text('\n'.join(lines)+'\n');return result
