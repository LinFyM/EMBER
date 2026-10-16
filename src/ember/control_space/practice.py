"""Actual fixed-code episodes, lossless raw H and reusable observation features."""
from __future__ import annotations

from copy import deepcopy
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED,as_completed
from pathlib import Path
import gzip
import io
import fcntl
import os
import time

import numpy as np
import torch

from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.pi05_eval_contract import policy_noise_seed
from ember.reference.accounting import reserve,finish,connect,host_resources
from ember.writer.practice import History
from ember.writer.runtime import autocast

H_SCHEMA='ember_feedback_raw_H_v1'
PHI_SCHEMA='ember_feedback_unique_observation_phi_v1'
_RAW_CACHE=OrderedDict()
_RAW_CACHE_BYTES=0
_RAW_CACHE_HITS=0


def specifications(command):
    common={k:v for k,v in command.items() if k not in ('rows','states','command')}
    values=command.get('rows') or [dict(state_id=s) for s in command['states']]
    if not 1<=len(values)<=32:raise ValueError('a useful worker pack contains1..32 actual rows')
    result=[]
    for value in values:
        row=dict(common,**value);identity=Path(row['identity'])
        if identity.is_absolute() or '..' in identity.parts:raise ValueError('episode identity must stay under its root')
        if row['role'] not in ('practice','query'):raise ValueError('only registered practice/query may execute')
        row['task']=int(row['task']);row['state_id']=int(row['state_id']);row['noise_root']=int(row['noise_root'])
        if row.get('LoRA'):
            if row['role']!='query' or not Path(row['LoRA']).is_absolute():raise ValueError('saved complete controllers are query-only')
            row['y']=None
        else:
            y=torch.as_tensor(row['y'],dtype=torch.float32)
            if y.shape!=(64,) or not bool(y.isfinite().all()):raise ValueError('behavior y must be a finite actual64-vector')
            row['y']=y.tolist()
        result.append(row)
    if len({(r['identity'],r['state_id']) for r in result})!=len(result):raise ValueError('pack repeats an episode')
    return result


def paths(root,spec):
    folder=Path(root)/'episodes'/spec['identity']
    return folder/f'state_{spec["state_id"]:03d}.json',folder/f'state_{spec["state_id"]:03d}'


def case_id(spec):return f'{spec["role"]}_{spec["identity"]}_state_{spec["state_id"]:03d}'


def validate_row(row,spec,task):
    expected=dict(global_task_id=spec['task'],suite=task['suite'],task_id=task['task_id'],language=task['language'],
        init_state_id=spec['state_id'],env_seed=spec['noise_root'],policy_seed_root=spec['noise_root'],
        actual_parameter_ref=spec['parameter_ref'],role=spec['role'],identity=spec['identity'],
        settling_steps=10,restored_control_steps=0,source_frozen=True,MT_loaded=False,full_native_flow_steps=10,executed_chunk=5)
    expected.update({k:spec[k] for k in ('behavior_model','meta_update','condition_id','panel_ordinal','LoRA','arm','teacher_demo') if k in spec})
    if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('saved episode producer/panel/native contract changed')
    if spec.get('LoRA'):
        if row['behavior_y'] is not None:raise ValueError('saved controller must not invent a behavior code')
    elif not torch.allclose(torch.tensor(row['behavior_y']),torch.tensor(spec['y']),atol=1e-6,rtol=1e-6):
        raise ValueError('saved episode has a different actual sampled behavior code')
    steps=row.get('steps',0)
    if (type(row.get('success')) is not bool or not 0<steps<=task['horizon'] or row.get('model_failure')
            or row.get('budget_truncated') or row.get('failure_type') or (not row['success'] and steps!=task['horizon'])
            or row.get('environment_steps')!=steps+10):raise ValueError('engineering/incomplete row cannot complete a panel')
    noises=[policy_noise_seed(spec['noise_root'],task['suite'],task['task_id'],spec['state_id'],i) for i in range((steps+4)//5)]
    if row.get('policy_noise_seeds')!=noises or row.get('first_noise_seed')!=noises[0]:
        raise ValueError('actual episode noise schedule changed')


def validate_history(history,row,spec):
    if len(history.episodes)!=1 or not history.records:raise ValueError('actual episode H is incomplete')
    if sum(int(r['executed'].sum()) for r in history.records)!=row['steps']:raise ValueError('H actual action mask lost control steps')
    for record in history.records:
        if record['parameter_ref']!=spec['parameter_ref'] or record['hidden'].shape!=(2,50,1024):
            raise ValueError('H lost the actual producer or same-call native hidden')
        for key in (record['pre'],record['post']):
            raw=history.observations[key]
            if raw['images'].dtype!=torch.uint8 or raw['images'].shape!=(2,3,256,256) or raw['proprio'].shape!=(8,):
                raise ValueError('H lost actual full-resolution raw observations')


def save_history(destination,history,row,spec,*,runtime=None):
    """Archive raw RGB and same-call hidden, without dense phi or repeated factors."""
    validate_history(history,row,spec)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    state={spec['parameter_ref']:torch.tensor(spec['y'],dtype=torch.float32)}
    payload=dict(schema=H_SCHEMA,complete=True,row=row,cache_key=f'{spec["identity"]}_s{spec["state_id"]:03d}_{destination.name[:-6]}',
        history=dict(records=history.records,observations=history.observations,episodes=history.episodes,
            rgb=history.rgb,states=state,image_features={}))
    temporary=destination.with_suffix(destination.suffix+f'.partial_{os.getpid()}')
    with temporary.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=3,mtime=0) as stream:torch.save(payload,stream)
        raw.flush();os.fsync(raw.fileno())
    temporary.replace(destination)
    if runtime is not None and spec.get('cache_root'):
        observation_cache(runtime,history,payload['cache_key'],Path(spec['cache_root']),fill_missing=False)
    return destination.stat().st_size


def load_history(path):
    global _RAW_CACHE_BYTES,_RAW_CACHE_HITS
    key=str(Path(path).resolve())
    if key in _RAW_CACHE:
        history,size=_RAW_CACHE.pop(key);_RAW_CACHE[key]=(history,size);_RAW_CACHE_HITS+=1;return history
    # The torch zip reader seeks backwards; seeking through gzip repeatedly
    # decompresses the same episode. Inflate one bounded episode only once.
    with gzip.open(path,'rb') as stream:raw=stream.read()
    payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    del raw
    if payload.get('schema')!=H_SCHEMA or not payload.get('complete'):raise ValueError('incomplete raw H archive')
    history=History(**payload['history']);history._cache_key=payload['cache_key'];history._archive_row=payload['row']
    if history.image_features or any(torch.as_tensor(v).shape!=(64,) for v in history.states.values()):
        raise ValueError('raw H must retain codes, without dense image caches or repeated LoRA factors')
    primary=sum(r['images'].numel()+r['proprio'].numel()*r['proprio'].element_size() for r in history.observations.values())+sum(r['hidden'].numel()*r['hidden'].element_size() for r in history.records)
    size=int(primary*1.25)
    while _RAW_CACHE and _RAW_CACHE_BYTES+size>4*2**30:
        _,(_,old)=_RAW_CACHE.popitem(last=False);_RAW_CACHE_BYTES-=old
    if size<=4*2**30:_RAW_CACHE[key]=(history,size);_RAW_CACHE_BYTES+=size
    return history


def history_cache_info():return dict(entries=len(_RAW_CACHE),primary_reserved_bytes=_RAW_CACHE_BYTES,hits=_RAW_CACHE_HITS,cache_limit_GiB=4)


def condition_spent(root,task):
    from .contract import condition_id
    prefix='practice_'+condition_id(task)+'_practice_'
    with connect(root) as db:
        return db.execute('SELECT coalesce(sum(control+dummy),0) FROM attempts WHERE substr(case_id,1,?)=?',
            (len(prefix),prefix)).fetchone()[0]


def reserve_episode(root,spec,horizon,limits):
    if spec['role']!='practice':return reserve(root,case_id(spec),horizon,limits=limits)
    with (Path(root)/'practice_budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if condition_spent(root,spec['task'])+horizon+10>8192:
            raise RuntimeError('condition8192 budget cannot cover another complete episode; retain actual partial panel')
        return reserve(root,case_id(spec),horizon,limits=limits)


def finish_engineering(root,row,path):
    """Known failed execution keeps actual cost and remains eligible for one retry."""
    with connect(root) as db:
        status=db.execute('SELECT status FROM attempts WHERE identity=?',(row['attempt_id'],)).fetchone()
        if status is None:raise ValueError('failed raw result lost its physical reservation')
        if status[0]=='failed_engineering':return
        if status[0]!='reserved_unknown':raise ValueError('cannot relabel a valid completed interaction')
        db.execute("UPDATE attempts SET status='failed_engineering',control=?,dummy=?,finished=?,row_path=? WHERE identity=?",
            (row['environment_steps']-row['settling_steps'],row['settling_steps'],time.time(),str(path),row['attempt_id']))


def adopt(root,spec,task):
    """A committed raw result wins over an interrupted small pointer/ledger write."""
    destination,folder=paths(root,spec)
    if not destination.exists():
        for archive in sorted(folder.glob('attempt_*.pt.gz'),reverse=True):
            row=load_history(archive)._archive_row
            row['history_bytes']=archive.stat().st_size
            if not row.get('valid_panel_row',True):
                if row['identity']!=spec['identity'] or row['behavior_y']!=spec['y']:raise ValueError('failed raw sample changed')
                finish_engineering(root,row,archive);continue
            validate_row(row,spec,task)
            write_json_atomic(destination,row);break
    if not destination.exists():
        for receipt in sorted(folder.glob('attempt_*.json'),reverse=True):
            row=read_json(receipt)
            if not row.get('valid_panel_row',True):
                if row['identity']!=spec['identity'] or row['behavior_y']!=spec['y']:raise ValueError('failed raw sample changed')
                finish_engineering(root,row,receipt);continue
            validate_row(row,spec,task)
            if spec['role']=='practice' and not Path(row['history_path']).is_file():continue
            write_json_atomic(destination,row);break
    if not destination.exists():return None
    row=read_json(destination);validate_row(row,spec,task)
    if spec['role']=='practice' and not Path(row['history_path']).is_file():raise ValueError('valid practice row lost raw H')
    with connect(root) as db:
        receipt=db.execute('SELECT status FROM attempts WHERE identity=?',(row['attempt_id'],)).fetchone()
    if receipt is None:raise ValueError('saved episode has no original physical budget reservation')
    if receipt[0]=='reserved_unknown':finish(root,row['attempt_id'],row,destination)
    elif receipt[0]!='complete':raise ValueError('unrecognized saved interaction status')
    return row


def complete_episode(root,runtime,output,spec,task):
    destination,folder=paths(root,spec);attempt=output['request']['episode_id'];ordinal=attempt.rsplit('_attempt_',1)[1]
    archive=folder/f'attempt_{ordinal}.pt.gz';receipt=folder/f'attempt_{ordinal}.json'
    history=output['history'];row=dict(output['row']);first=history.records[0]
    row.update(global_task_id=spec['task'],identity=spec['identity'],role=spec['role'],behavior_y=spec['y'],
        attempt_id=attempt,producer_parameter_ref=spec['parameter_ref'],noise_root=spec['noise_root'],first_noise_seed=first['noise_seed'],
        first_native_normalized_actions=first['normalized_actions'].tolist(),source_frozen=True,MT_loaded=False,
        full_native_flow_steps=10,executed_chunk=5,history_path=str(archive) if spec['role']=='practice' else None)
    row.update({k:spec[k] for k in ('behavior_model','meta_update','condition_id','panel_ordinal','LoRA','arm','teacher_demo') if k in spec})
    error=None
    try:validate_row(row,spec,task)
    except ValueError as failure:error=str(failure)
    row.update(valid_panel_row=error is None,engineering_error=error)
    if spec['role']=='practice':
        tick=time.monotonic()
        row['history_primary_raw_bytes']=sum(r['images'].numel()+r['proprio'].numel()*r['proprio'].element_size() for r in history.observations.values())+sum(r['hidden'].numel()*r['hidden'].element_size() for r in history.records)
        row['history_bytes']=save_history(archive,history,row,spec,runtime=runtime)
        row['history_archive_seconds']=time.monotonic()-tick
    write_json_atomic(receipt,row)
    if error is None:write_json_atomic(destination,row)
    if error is None:finish(root,attempt,row,destination)
    else:finish_engineering(root,row,receipt)
    if error is not None:raise RuntimeError(f'actual failed episode retained at {receipt}: {error}')
    return row


def decoded_parameters(runtime,decoder,units,specs):
    from ember.lora import validate_lora_state
    from safetensors.torch import load_file
    states={}
    for spec in specs:
        ref=spec['parameter_ref']
        if ref not in states:
            with torch.no_grad():
                states[ref]=(load_file(spec['LoRA'],device=str(runtime.device)) if spec.get('LoRA') else
                    decoder.decode(torch.tensor(spec['y'],device=runtime.device)*units))
            validate_lora_state(states[ref],runtime.lora)
            if any(not bool(v.isfinite().all()) for v in states[ref].values()):raise ValueError('nonfinite decoded LoRA cannot execute')
        elif not spec.get('LoRA') and not torch.allclose(torch.tensor(spec['y']),torch.tensor(next(r['y'] for r in specs if r['parameter_ref']==ref))):
            raise ValueError('one parameter_ref cannot label two actual codes')
    return states


def panel(root,runtime,runner,command,decoder,units):
    root=Path(root);run=read_json(root/'run_contract.json');tasks={r['global_task_id']:r for r in run['environment']['tasks']}
    specs=specifications(command);rows={};missing=[]
    for spec in specs:
        if spec['task'] not in tasks:raise ValueError('practice/query task is outside current36')
        if spec.get('cache_root') and not Path(spec['cache_root']).resolve().is_relative_to((root/'cache/H').resolve()):
            raise ValueError('transient dense H cache must stay under this run cache/H')
        saved=adopt(root,spec,tasks[spec['task']])
        if saved is None:
            with connect(root) as db:
                unknown=db.execute("SELECT count(*) FROM attempts WHERE case_id=? AND status='reserved_unknown'",(case_id(spec),)).fetchone()[0]
            if unknown:raise RuntimeError('unknown actual interaction has no restorable raw H/row; preserve full charge and report boundary')
            missing.append(spec)
        else:rows[(spec['identity'],spec['state_id'])]=saved
    states=decoded_parameters(runtime,decoder,units,missing)
    def requests():
        for spec in missing:
            task=tasks[spec['task']];attempt=reserve_episode(root,spec,task['horizon'],run['limits'])
            yield dict(task=task,state=states[spec['parameter_ref']],parameter_ref=spec['parameter_ref'],
                state_id=spec['state_id'],noise_root=spec['noise_root'],episode_id=attempt,specification=spec,
                capture_history=spec['role']=='practice',capture_hidden=spec['role']=='practice',
                capture_phi=spec['role']=='practice',capture_rgb=False,uses_teaching=False)
    before=deepcopy(runner.components);tick=time.monotonic();peak=None
    archival_workers=int(command.get('archive_workers',2));pending={}
    def accept(future):
        spec=pending.pop(future);rows[(spec['identity'],spec['state_id'])]=future.result()
    with ThreadPoolExecutor(max_workers=archival_workers) as archive_executor:
        for output in runner.run(requests()) if missing else ():
            if peak is None:
                peak=host_resources();peak['environment_PIDs']=[p.pid for p in runner.environments.processes if p is not None]
            spec=output['request']['specification']
            pending[archive_executor.submit(complete_episode,root,runtime,output,spec,tasks[spec['task']])]=spec
            if len(pending)>=archival_workers*2:
                done,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:accept(future)
        for future in as_completed(list(pending)):accept(future)
    elapsed=time.monotonic()-tick;ordered=[rows[(s['identity'],s['state_id'])] for s in specs]
    components={k:v-before[k] for k,v in runner.components.items() if k!='physical_batch_histogram'}
    components['physical_batch_histogram']={k:v-before['physical_batch_histogram'].get(k,0) for k,v in runner.components['physical_batch_histogram'].items()}
    return dict(complete=True,rows=ordered,row_paths=[str(paths(root,s)[0]) for s in specs],
        history_paths=[r['history_path'] for r in ordered if r['history_path']],states=[s['state_id'] for s in specs],
        resources=dict(archive_workers=archival_workers,new_episodes=len(missing),reused_valid_rows=len(specs)-len(missing),seconds=elapsed,
            episodes_per_second=len(missing)/max(elapsed,1e-6),runner=components,
            peak_environment_resources=peak,after_panel_resources=host_resources()))


@torch.no_grad()
def observation_features(runtime,observations,*,chunk=16):
    """Real raw RGB through source vision/projector only, without a fake suffix."""
    if runtime.execution._active_state is not None:raise ValueError('observation vision requires closed execution adapter context')
    result={};items=list(observations.items())
    for first in range(0,len(items),chunk):
        part=items[first:first+chunk];images=torch.stack([raw['images'] for _,raw in part]).to(runtime.device).float()/255
        batch={'observation.images.base_0_rgb':images[:,0],'observation.images.left_wrist_0_rgb':images[:,1]}
        pixels,_=runtime.policy._preprocess_images(batch)
        with autocast(runtime.device):phi=torch.cat([runtime.policy.model.paligemma_with_expert.embed_image(v) for v in pixels[:2]],1)
        if phi.shape!=(len(part),512,2048) or not bool(phi.isfinite().all()):raise ValueError('actual observation vision shape/finite changed')
        result.update({key:value.detach().cpu().to(torch.bfloat16) for (key,_),value in zip(part,phi,strict=True)})
    return result


def allocate_dense(file,count):
    """All per-condition caches share one16GiB transient allocation budget."""
    parent=file.parent.parent;parent.mkdir(parents=True,exist_ok=True)
    with (parent/'.dense_budget.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        used=sum(p.stat().st_size for pattern in ('*.npy','*.partial_*') for p in parent.rglob(pattern))
        required=count*512*2048*2+256
        if used+required>16*2**30:raise RuntimeError('global temporary dense H16GiB limit; raw H preserved')
        temporary=file.with_suffix(f'.partial_{os.getpid()}')
        array=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.uint16,shape=(count,512,2048))
        array.flush();del array;temporary.replace(file)


def observation_cache(runtime,history,key,cache_root=None,*,fill_missing=True,image_chunk=16):
    """One BF16 slot per actual observation; event pre/post normally remain views."""
    keys=list(history.observations);source=str(runtime.source['model_path']);filled=set();meta=None
    if cache_root is None:
        tensor=torch.empty(len(keys),512,2048,dtype=torch.bfloat16)
    else:
        folder=Path(cache_root);folder.mkdir(parents=True,exist_ok=True);name=key.replace('/','__')
        file=folder/f'{name}.npy';meta=folder/f'{name}.json'
        if file.exists():
            old=read_json(meta)
            if old['schema']!=PHI_SCHEMA or old['keys']!=keys or old['source']!=source:raise ValueError('dense cache observation/source identity changed')
            filled=set(old['filled']);array=np.load(file,mmap_mode='r+')
        else:
            write_json_atomic(meta,dict(schema=PHI_SCHEMA,keys=keys,source=source,filled=[]))
            allocate_dense(file,len(keys));array=np.load(file,mmap_mode='r+')
        if array.dtype!=np.uint16 or array.shape!=(len(keys),512,2048):raise ValueError('unique observation cache lost BF16 shape')
        tensor=torch.from_numpy(array).view(torch.bfloat16)
    def write(values):
        if not values:return
        for obs,value in values.items():
            index=keys.index(obs)
            if value.shape!=(512,2048) or not bool(value.isfinite().all()):raise ValueError('raw observation phi lost complete real pixels')
            tensor[index].copy_(value.to(torch.bfloat16));filled.add(index)
        if meta is not None:
            array.flush();write_json_atomic(meta,dict(schema=PHI_SCHEMA,keys=keys,source=source,filled=sorted(filled)))
    write({k:v for k,v in history.image_features.items() if k in history.observations and keys.index(k) not in filled})
    if fill_missing:
        for first in range(0,len(keys),image_chunk):
            missing={keys[i]:history.observations[keys[i]] for i in range(first,min(first+image_chunk,len(keys))) if i not in filled}
            if missing:write(observation_features(runtime,missing,chunk=image_chunk))
        if len(filled)!=len(keys):raise ValueError('model events cannot consume an unfilled observation slot')
    return tensor,{key:i for i,key in enumerate(keys)}


def event_facts(processor,record,episode,quantiles=None):
    proprio=record['proprio'].detach().float();actions=record['actions'].detach().float();mask=record['executed'].bool()
    if proprio.shape!=(2,8) or actions.shape!=(5,7) or mask.shape!=(5,) or record['feedback'].shape!=(5,):
        raise ValueError('actual H facts lost proprio/actions/mask/feedback shapes')
    q=quantiles or {k:getattr(processor,'_'+k).detach().cpu() for k in ('state_q01','state_q99','action_q01','action_q99')}
    state=processor._quantile_transform(proprio,q['state_q01'],q['state_q99'],False)
    normalized=torch.zeros_like(actions)
    if mask.any():normalized[mask]=processor._quantile_transform(actions[mask],q['action_q01'],q['action_q99'],False)
    facts=torch.cat((state.flatten(),normalized.flatten(),mask.float(),record['feedback'].detach().float(),
        torch.tensor([episode,record['step']],dtype=torch.float32)/8192))
    if facts.shape!=(63,) or not bool(facts.isfinite().all()):raise ValueError('nonfinite or wrong63D actual facts')
    return facts


def iter_model_events(runtime,histories,*,cache_root=None,image_chunk=16):
    """Stopgrad real evidence; only observation features are reusable, never learned E/H."""
    if isinstance(histories,(str,Path,History)):histories=[histories]
    quantiles={k:getattr(runtime.processor,'_'+k).detach().cpu() for k in ('state_q01','state_q99','action_q01','action_q99')}
    offset=0
    for item in histories:
        history=load_history(item) if isinstance(item,(str,Path)) else item
        key=getattr(history,'_cache_key',None)
        if cache_root is not None and key is None:raise ValueError('disk cache requires an immutable raw archive identity')
        phi,indices=observation_cache(runtime,history,key,cache_root,fill_missing=True,image_chunk=image_chunk)
        for record in history.records:
            pre,post=indices[record['pre']],indices[record['post']]
            images=phi[pre:post+1] if post==pre+1 else torch.stack((phi[pre],phi[post]))
            hidden=record['hidden'].detach();y=torch.as_tensor(history.states[record['parameter_ref']],dtype=torch.float32).detach()
            if hidden.shape!=(2,50,1024) or y.shape!=(64,):raise ValueError('H lost same-call native hidden or actual behavior y')
            yield dict(images=images,hidden=hidden,facts=event_facts(runtime.processor,record,record['episode']+offset,quantiles),y=y,
                parameter_ref=record['parameter_ref'],episode=record['episode']+offset,step=record['step'])
        offset+=len(history.episodes)


def model_events(runtime,histories,*,cache_root=None,image_chunk=16):
    return list(iter_model_events(runtime,histories,cache_root=cache_root,image_chunk=image_chunk))
