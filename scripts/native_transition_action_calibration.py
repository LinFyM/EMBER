#!/usr/bin/env python3
"""Single bounded native/500gamma/readback owner; no Writer or environment path."""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import socket
import time
from pathlib import Path

import h5py
import numpy as np
import torch

# Initialize the existing facade before its internal source-loading owner.
import ember.pi05_evaluation

from ember.operator_writer.native import _NativeFrameCall
from ember.operator_writer.native_action_calibration import FIT, TRAIN, fresh_pair, optimization, paired_forward, sample_events, update
from ember.pi05_eval.worker_setup import load_policy
from ember.pi05_eval_contract import git_state, git_state_is_clean_pushed_or_frozen_authority, inspect_source_checkpoint, load_evaluation_authorities
from ember.pi05_processing import Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.data import RawTeacherVideoStore
from ember.writer.learning_data import load_learning_tasks
from ember.writer.runtime import autocast
from ember.writer.topology import bind_current_process_to_cuda_numa

ROOT=Path('/data1/user/ymdai/ember_runs/native_transition_action_calibration_20261003')
ASSET=Path('/data1/user/ymdai/projects/EMBER')
REPO=Path(__file__).resolve().parents[1]
SCHEMA='ember_native_action_calibration_v1'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)


def append(name,value):
    with (ROOT/name).open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')


def rng():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy']);random.setstate(state['python'])


def native_call(wrapper,frames,tokens,mask,probe):
    """Capture the actual projection of the same legal state-free native call."""
    captured=[]
    handle=wrapper.policy.model.action_out_proj.register_forward_hook(lambda _m,_a,v:captured.append(v))
    try:
        with torch.no_grad(),autocast(torch.device('cuda:0')):h,=wrapper(frames,tokens,mask)
    finally:handle.remove()
    if len(captured)!=1 or h.shape!=(len(frames),50,1024) or captured[0].shape!=(len(frames),50,32):
        raise ValueError('bare Source native H/velocity changed')
    mu=(probe[None]-captured[0].float())[:,:5,:7]
    if not torch.isfinite(h).all() or not torch.isfinite(mu).all():raise ValueError('nonfinite native')
    return h.detach().float().cpu(),mu.detach().cpu()


def read_frames(wrapper,pixels,tokens,mask,probe,profile):
    values=[];start=0
    for chunk in (64,64,128) if not profile else ():
        torch.cuda.reset_peak_memory_stats();beg=time.monotonic()
        try:
            h,mu=native_call(wrapper,pixels[start:start+chunk],tokens,mask,probe)
            torch.cuda.synchronize();seconds=time.monotonic()-beg
            profile.append(dict(frame_start=start,frames=chunk,frame_chunk=chunk,seconds=seconds,
                frame_per_second=chunk/seconds,reserved_GiB=torch.cuda.max_memory_reserved()/2**30,success=True))
            values.append((h,mu));start+=chunk
        except torch.cuda.OutOfMemoryError:
            append('launch/failures.jsonl',dict(stage='native frame128 profile',frame_start=start,
                frames=chunk,elapsed=time.monotonic()-beg,scientific_negative=False))
            torch.cuda.empty_cache();profile.append(dict(frame_start=start,frames=chunk,frame_chunk=chunk,success=False));break
    small=[r for r in profile if r['success'] and r['frame_chunk']==64]
    large=[r for r in profile if r['success'] and r['frame_chunk']==128]
    # Both rates use128 legal frames; the first64 includes cold startup.
    chunk=128 if large and large[0]['frame_per_second']>=128/sum(r['seconds'] for r in small[:2]) else 64
    for k in range(start,len(pixels),chunk):values.append(native_call(wrapper,pixels[k:k+chunk],tokens,mask,probe))
    return torch.cat([v[0] for v in values]),torch.cat([v[1] for v in values]),chunk


def save_video(record,h,mu,processor,identity):
    slots=record['interval_slots'];indices=record['frame_indices']
    with h5py.File(record['hdf5'],'r') as f:
        actions=f[f"data/demo_{record['demo']}/actions"]
        raw=np.stack([np.asarray(actions[indices[k]+1:indices[k]+6],dtype=np.float32) for k in slots])
    if raw.shape!=(len(slots),5,7):raise ValueError('offset1 complete5 labels changed')
    target=processor.normalize_action(torch.from_numpy(raw)).cpu()
    value=dict(H=h,mu=mu,target=target,mask=torch.ones((len(slots),5),dtype=torch.bool),
        frame_indices=torch.tensor(indices),interval_slots=torch.tensor(slots),record=record,
        source=identity,original_native_git=git_state(REPO),information_wall=dict(
            native_fields=['obs/agentview_rgb','obs/eye_in_hand_rgb','exact language','fixed probe1729/tau1'],
            labels_only='actions[p+1:p+6]; normalization used only here; no state/reward/terminal values'))
    path=ROOT/'native'/f"task{record['task']:03d}_demo{record['demo']:02d}.pt";save(path,value)
    return str(path)


def materialize(records):
    paths=[ROOT/'native'/f"task{r['task']:03d}_demo{r['demo']:02d}.pt" for r in records]
    if all(p.exists() for p in paths):return dict(loaded_source_this_consumer=0,reused_videos=176)
    config=read_json(ASSET/'configs/operator_read_write_v1/conditional_read_write_fresh_spec.json')['source']
    authority=load_evaluation_authorities(ASSET/config['evaluation_config'],ASSET)
    cp=ASSET/config['checkpoint'];identity=inspect_source_checkpoint(authority,cp.parent.parent,cp,evaluation_mode='formal')
    stats=read_json(ASSET/config['normalization'])['stats'];tokenpath=ASSET/config['tokenizer']
    policy,processor,_=load_policy(Path(identity['model_path']),stats,tokenpath,read_json(ASSET/'configs/pi05_target_evaluation_v1.json')['policy'])
    policy.requires_grad_(False)
    if any(p.requires_grad or 'lora_' in n for n,p in policy.named_parameters()):raise ValueError('native must be bare frozen Source1000')
    tokenizer=Pi05TeacherPrefixTokenizer(tokenpath,200,'cuda:0')
    probe=torch.randn((50,32),generator=torch.Generator().manual_seed(1729)).to('cuda:0')
    wrapper=_NativeFrameCall(policy,(),probe)
    tasks=load_learning_tasks(ASSET,TRAIN,protocol_path=config['data_protocol'])
    store=RawTeacherVideoStore(tuple(t.authority for t in tasks.values()),frame_stride=5,camera_view='dual')
    profile=[];started=time.monotonic();new=0;chosen=[]
    try:
        order=sorted(TRAIN,key=lambda t:-sum(r['frame_count'] for r in records if r['task']==t))
        for task in order:
            todo=[r for r,p in zip(records,paths,strict=True) if r['task']==task and not p.exists()]
            if not todo:continue
            videos=[store.load(task,r['demo']) for r in todo]
            if any(v.frame_indices.tolist()!=r['frame_indices'] for v,r in zip(videos,todo,strict=True)):raise ValueError('native frame metadata changed')
            pixels=torch.from_numpy(np.concatenate([v.frames for v in videos])).to('cuda:0')
            tokens,mask,_=tokenizer([todo[0]['language']]);h,mu,chunk=read_frames(wrapper,pixels,tokens,mask,probe,profile)
            begin=0
            for record in todo:
                end=begin+record['frame_count'];save_video(record,h[begin:end].clone(),mu[begin:end].clone(),processor,identity);begin=end;new+=1
            chosen.append(dict(task=task,frame_chunk=chunk,frames=len(h)));del pixels,h,mu,videos
    finally:store.close()
    result=dict(new_videos=new,loaded_source_this_consumer=1,seconds=time.monotonic()-started,
        frames=sum(r['frame_count'] for r in records),profile=profile,chosen=chosen,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        source=identity,probe_seed=1729,tau=1,source_trainable=0,source_LoRA=0,
        field_wall='same RGB/language Source call; actions only output labels; State prompt omitted')
    write_json_atomic(ROOT/'native/manifest.json',result)
    del wrapper,policy,processor,tokenizer,probe;gc.collect();torch.cuda.empty_cache()
    return result


def cached(records):
    values={};Hs=[];mus=[];ys=[];offset=0;lookup={}
    for record in records:
        path=ROOT/'native'/f"task{record['task']:03d}_demo{record['demo']:02d}.pt"
        raw=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if (raw['H'].shape!=(record['frame_count'],50,1024) or raw['mu'].shape!=(record['frame_count'],5,7)
                or raw['frame_indices'].tolist()!=record['frame_indices'] or raw['interval_slots'].tolist()!=record['interval_slots']):
            raise ValueError('cache shapes/indices do not match registered videos')
        values[(record['task'],record['demo'])]=dict(offset=offset,raw=raw,path=str(path))
        for i,slot in enumerate(record['interval_slots']):lookup[(record['task'],record['demo'],slot)]=(offset+slot,len(ys)+i)
        Hs.append(raw['H']);mus.append(raw['mu']);ys.extend(raw['target']);offset+=record['frame_count']
    return values,lookup,torch.cat(Hs).to('cuda:0'),torch.cat(mus).to('cuda:0'),torch.stack(ys).to('cuda:0')


def checkpoint(models,opts,scheds,cursor,mode,events):
    folder=ROOT/'checkpoints'/f'update{cursor:03d}'
    state=dict(schema=SCHEMA,models=models.state_dict(),optimizer=[o.state_dict() for o in opts],
        scheduler=[s.state_dict() for s in scheds],scaler=None,sampler=dict(cursor=cursor,seed=20261003,
        event_path=str(ROOT/'events.npy'),fit_tasks=list(FIT),queries_per_task=8),RNG=rng(),
        topology=dict(host=socket.gethostname(),world_size=1,cuda_visible=os.environ['CUDA_VISIBLE_DEVICES']),
        physical_mode=mode,updates_per_arm=500,contract=str(REPO/'docs/designs/native_transition_action_calibration_diagnostic.md'),git=git_state(REPO))
    save(folder/'recovery.pt',state)
    write_json_atomic(folder/'manifest.json',dict(complete=True,next_update=cursor,schema=SCHEMA,
        state=str(folder/'recovery.pt'),bytes=(folder/'recovery.pt').stat().st_size))


def train(H,mu,y,lookup,events,resume):
    models=fresh_pair('cuda:0');opts,scheds=optimization(models);start=0
    ids=torch.tensor([[lookup[tuple(e)] for e in row] for row in events],device='cuda:0')
    def batch(k):
        frame,label=ids[k].unbind(-1)
        return H[frame],H[frame+1],mu[frame],y[label]
    if resume:
        points=[k for k in (0,250,500) if (ROOT/'checkpoints'/f'update{k:03d}'/'manifest.json').exists()]
        raw=torch.load(ROOT/'checkpoints'/f'update{max(points):03d}'/'recovery.pt',map_location='cpu',weights_only=False)
        models.load_state_dict(raw['models']);start=raw['sampler']['cursor'];mode=raw['physical_mode']
        for o,s,osd,ssd in zip(opts,scheds,raw['optimizer'],raw['scheduler'],strict=True):o.load_state_dict(osd);s.load_state_dict(ssd)
        restore_rng(raw['RNG'])
    else:
        initial=rng();profiles=[]
        for mode in ('sequential','packed'):
            models=fresh_pair('cuda:0');opts,scheds=optimization(models);restore_rng(initial)
            torch.cuda.reset_peak_memory_stats();beg=time.monotonic()
            with autocast(torch.device('cuda:0')):record=update(models,opts,scheds,*batch(0),mode)
            torch.cuda.synchronize();profiles.append(dict(**record,seconds=time.monotonic()-beg,
                peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
        mode=min(profiles,key=lambda r:r['seconds'])['physical_mode']
        write_json_atomic(ROOT/'training_profile.json',dict(rows=profiles,updates_per_arm=2,
            reset_initialization_and_RNG=True,selected=mode,max_physical_queries_per_arm=160,
            reason='all20 tasks×8; paired packing compares320 real rows without new samples'))
        models=fresh_pair('cuda:0');opts,scheds=optimization(models);restore_rng(initial)
        checkpoint(models,opts,scheds,0,mode,events)
    beg=time.monotonic()
    for k in range(start,500):
        update_beg=time.monotonic()
        with autocast(torch.device('cuda:0')):record=update(models,opts,scheds,*batch(k),mode)
        torch.cuda.synchronize();append('training_metrics.jsonl',dict(update=k+1,seconds=time.monotonic()-update_beg,**record))
        if k+1 in (250,500):checkpoint(models,opts,scheds,k+1,mode,events)
    return models,dict(start_update=start,end_update=500,seconds=time.monotonic()-beg,mode=mode,
        source_removed_before_learning=True,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30)


def predict_all(models,H,mu,values,records):
    models.eval();beg=time.monotonic();written=0
    with torch.no_grad():
        for record in records:
            out=ROOT/'predictions'/f"task{record['task']:03d}_demo{record['demo']:02d}.pt"
            if out.exists():continue
            value=values[(record['task'],record['demo'])];raw=value['raw']
            frames=raw['interval_slots'].to('cuda:0')+value['offset']
            with autocast(torch.device('cuda:0')):residual=paired_forward(models,H[frames],H[frames+1])[:,:,:5].float()
            preds=residual+mu[frames][None]
            if not torch.isfinite(preds).all():raise ValueError('nonfinite fixed500 prediction')
            save(out,dict(mu=mu[frames].cpu(),F=preds[0].cpu(),P=preds[1].cpu(),target=raw['target'],
                mask=raw['mask'],departure_frames=raw['frame_indices'][raw['interval_slots']],
                arrival_frames=raw['frame_indices'][raw['interval_slots']+1],action_offset=1,
                action_indices=raw['frame_indices'][raw['interval_slots']][:,None]+torch.arange(1,6)[None],
                record=record,native_path=value['path'],checkpoint=str(ROOT/'checkpoints/update500/manifest.json'),
                readback_git=git_state(REPO)));written+=1
    return dict(new_prediction_videos=written,total_videos=176,intervals=sum(r['legal_intervals'] for r in records),seconds=time.monotonic()-beg)


def main(resume):
    start=time.monotonic();identity=git_state(REPO)
    if identity['branch'] or identity['dirty_paths'] or not git_state_is_clean_pushed_or_frozen_authority(identity):
        raise ValueError('calculation requires clean pushed detached source')
    if (ROOT/'GPU_completion.json').exists():raise ValueError('already completed; no automatic repeat')
    bind_current_process_to_cuda_numa();torch.set_num_threads(6);torch.set_float32_matmul_precision('high')
    records=read_json(ROOT/'analysis/metadata.json')['records'];native=materialize(records)
    values,lookup,H,mu,y=cached(records);events=sample_events(records)
    if (ROOT/'events.npy').exists():
        if not np.array_equal(np.load(ROOT/'events.npy'),events):raise ValueError('shared logical event stream changed')
    else:np.save(ROOT/'events.npy',events)
    models,training=train(H,mu,y,lookup,events,resume);readback=predict_all(models,H,mu,values,records)
    write_json_atomic(ROOT/'GPU_completion.json',dict(complete=True,schema=SCHEMA,git=identity,
        native=native,training=training,readback=readback,seconds=time.monotonic()-start,
        no_Writer_LoRA_environment_or_held=True,no_automatic_successor=True))
    print(json.dumps(dict(event='entire_GPU_consumer_complete',updates_per_arm=500,videos=176,intervals=5882)),flush=True)


if __name__=='__main__':
    args=argparse.ArgumentParser();args.add_argument('--resume',action='store_true');main(args.parse_args().resume)
