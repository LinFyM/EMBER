"""Frozen teaching memory and globally bounded unique-observation H features."""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import fcntl
import json
import os
import time

import torch

from ember.pi05_source_checkpoint import read_json
from ember.writer.runtime import autocast
from .practice import load_history

FEATURE_SCHEMA='ember_source_image_prefix_features_v1'
LEGACY_SCHEMA='ember_image_only_gemma_compiler_v1'
CACHE_SCHEMA='ember_frozen_tensor_cache_v1'
PREPROCESS=dict(camera_order=['agentview','eye_in_hand'],rotation_degrees=180,
    resize='resize_with_pad_torch',model_resolution=224,frame_stride=5,terminal_included=True,
    tokens_per_frame=512,token_width=2048)


def _indices(indices,raw_count):
    value=torch.as_tensor(indices)
    if (value.ndim!=1 or not len(value) or value.dtype not in (torch.int32,torch.int64)
            or type(raw_count) is not int or raw_count<=0):raise ValueError('video needs full original frame indices')
    expected=list(range(0,raw_count,5))
    if expected[-1]!=raw_count-1:expected.append(raw_count-1)
    if value.tolist()!=expected:raise ValueError('video lost stride5/start/terminal')
    return value.to(dtype=torch.int64,device='cpu')


@lru_cache(maxsize=2)
def _read_cached(path):
    value=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    source=value['source'];schema=value.get('schema_version')
    required=dict(dual_rgb=True,stride=5,post_norm=True,Gemma_input='512 image tokens only',
        L_in_Gemma=False,action_suffix=False,probe=False,old_meta_loaded=False)
    if (not value.get('complete') or schema not in (FEATURE_SCHEMA,LEGACY_SCHEMA) or not source.get('model')
            or any(source.get(k)!=v for k,v in required.items())
            or (schema==FEATURE_SCHEMA and source.get('preprocess')!=PREPROCESS)):
        raise ValueError('teaching cache is not the frozen pure-source image prefix')
    images=value['z_img'];indices=_indices(value['indices'],source['raw_frames'])
    if images.dtype!=torch.bfloat16 or images.shape!=(len(indices),512,2048) or not bool(images.isfinite().all()):
        raise ValueError('teaching cache lost complete finite BF16 image tokens')
    return dict(z_img=images,indices=indices,source=dict(source))


def load_cached(path,device='cpu'):
    value=_read_cached(str(path))
    return dict(z_img=value['z_img'].to(device),indices=value['indices'].to(device),source=value['source'])


@torch.no_grad()
def observation_features(runtime,observations,*,chunk=16):
    """Actual rotated raw RGB through frozen vision/projector, with no L/suffix."""
    if type(chunk) is not int or chunk<=0:raise ValueError('image_chunk must be positive')
    if runtime.execution._active_state is not None:raise ValueError('vision requires a closed adapter context')
    result={};items=list(observations.items())
    for first in range(0,len(items),chunk):
        part=items[first:first+chunk];images=torch.stack([r['images'] for _,r in part]).to(runtime.device).float()/255
        batch={'observation.images.base_0_rgb':images[:,0],'observation.images.left_wrist_0_rgb':images[:,1]}
        pixels,_=runtime.policy._preprocess_images(batch)
        with autocast(runtime.device):phi=torch.cat([runtime.policy.model.paligemma_with_expert.embed_image(v) for v in pixels[:2]],1)
        if phi.shape!=(len(part),512,2048) or not bool(phi.isfinite().all()):raise ValueError('actual observation vision shape/finite changed')
        result.update({k:v.detach().cpu().to(torch.bfloat16) for (k,_),v in zip(part,phi,strict=True)})
    return result


def event_facts(processor,record,episode,quantiles=None):
    proprio=record['proprio'].detach().float();actions=record['actions'].detach().float();mask=record['executed'].bool()
    if proprio.shape!=(2,8) or actions.shape!=(5,7) or mask.shape!=(5,) or record['feedback'].shape!=(5,):
        raise ValueError('actual H lost state/actions/mask/feedback')
    q=quantiles or {k:getattr(processor,'_'+k).detach().cpu() for k in ('state_q01','state_q99','action_q01','action_q99')}
    state=processor._quantile_transform(proprio,q['state_q01'],q['state_q99'],False)
    normalized=torch.zeros_like(actions)
    if mask.any():normalized[mask]=processor._quantile_transform(actions[mask],q['action_q01'],q['action_q99'],False)
    value=torch.cat((state.flatten(),normalized.flatten(),mask.float(),record['feedback'].detach().float(),
        torch.tensor([episode,record['step']],dtype=torch.float32)/8192))
    if value.shape!=(63,) or not bool(value.isfinite().all()):raise ValueError('actual facts63 must be finite')
    return value


@contextmanager
def _file_lock(path):
    with path.open('a') as lock:
        while True:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(.01)
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


class _BoundedWriter:
    def __init__(self,raw,limit):self.raw,self.limit=raw,limit
    def write(self,data):
        if self.raw.tell()+len(data)>self.limit:raise BufferError('frozen cache header exceeded reserved space')
        return self.raw.write(data)
    def flush(self):self.raw.flush()


class _TensorCache:
    """Only this run's regenerable tensors; ready+reserved+partial share one cap."""
    def __init__(self,root,capacity):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.capacity=int(capacity);self.control=min(8*2**20,max(16384,self.capacity//32))
        self.manifest=self.root/'manifest.json'

    @contextmanager
    def locked(self):
        with _file_lock(self.root/'.budget.lock'):
            state=read_json(self.manifest) if self.manifest.exists() else dict(schema=CACHE_SCHEMA,capacity=self.capacity,next_id=0,entries={})
            if state['schema']!=CACHE_SCHEMA or state['capacity']!=self.capacity:raise ValueError('wrong frozen cache schema/capacity')
            yield state

    def save(self,state):
        encoded=json.dumps(state,indent=2,sort_keys=True)+'\n'
        if 2*len(encoded.encode())>self.control:raise RuntimeError('frozen cache manifest exceeded control allowance')
        temporary=self.root/'manifest.partial'
        with temporary.open('w') as stream:
            stream.write(encoded);stream.flush();os.fsync(stream.fileno())
        temporary.replace(self.manifest)

    def entry(self,key,metadata):
        with self.locked() as state:
            if key not in state['entries']:
                index=state['next_id'];state['next_id']+=1
                state['entries'][key]=dict(file=f'tensor_{index:08d}.pt',status='absent',bytes=0,metadata=metadata,touched=time.time())
                self.save(state)
            entry=state['entries'][key]
            if entry['metadata']!=metadata:raise ValueError('immutable archive/source feature identity changed')
            return entry['file']

    def reserve(self,state,key,needed):
        entry=state['entries'][key]
        used=sum(e['bytes'] for e in state['entries'].values())
        victims=sorted((k for k,e in state['entries'].items() if k!=key and e['status']=='ready'),key=lambda k:state['entries'][k]['touched'])
        for other in victims:
            if used+needed<=self.capacity-self.control:break
            old=state['entries'][other];(self.root/old['file']).unlink(missing_ok=True)
            used-=old['bytes'];old.update(status='absent',bytes=0)
        available=used+needed<=self.capacity-self.control
        entry.update(status='reserved' if available else 'absent',bytes=needed if available else 0,touched=time.time())
        self.save(state);return available

    def release(self,key,temporary):
        with self.locked() as state:
            temporary.unlink(missing_ok=True);(self.root/state['entries'][key]['file']).unlink(missing_ok=True)
            state['entries'][key].update(status='absent',bytes=0)
            self.save(state)

    def get(self,key,metadata,required,build):
        tick=time.monotonic();file=self.entry(key,metadata);destination=self.root/file;temporary=self.root/(file+'.partial')
        with _file_lock(self.root/(file+'.lock')):
            with self.locked() as state:
                entry=state['entries'][key]
                if entry['status'] in ('ready','reserved') and destination.exists():
                    # Keep eviction locked only until CPU torch.load closes its fd.
                    with destination.open('rb') as stream:value=torch.load(stream,map_location='cpu',weights_only=True,mmap=False)
                    entry.update(status='ready',bytes=destination.stat().st_size,touched=time.time());self.save(state)
                    return value,dict(hit=True,cached=True,bytes=entry['bytes'],seconds=time.monotonic()-tick)
                temporary.unlink(missing_ok=True);entry.update(status='absent',bytes=0);self.save(state)
            needed=int(required())+65536
            with self.locked() as state:cached=self.reserve(state,key,needed)
            try:
                value=build().detach().cpu().contiguous()
                if not cached:return value,dict(hit=False,cached=False,bytes=value.numel()*value.element_size(),seconds=time.monotonic()-tick)
                with temporary.open('wb') as raw:
                    torch.save(value,_BoundedWriter(raw,needed));raw.flush();os.fsync(raw.fileno())
                with self.locked() as state:
                    size=temporary.stat().st_size;temporary.replace(destination)
                    state['entries'][key].update(status='ready',bytes=size,touched=time.time());self.save(state)
                return value,dict(hit=False,cached=True,bytes=size,seconds=time.monotonic()-tick)
            except BufferError:
                self.release(key,temporary)
                return value,dict(hit=False,cached=False,bytes=value.numel()*value.element_size(),seconds=time.monotonic()-tick)
            except BaseException:
                self.release(key,temporary);raise


class FrozenFeatures:
    """One owner for immutable video M and unique raw-H observation features."""
    def __init__(self,root,runtime,reader_frozen,*,history_capacity=16*2**30,memory_capacity=2**30):
        if not 0<history_capacity<=16*2**30 or not 0<memory_capacity<=2**30:raise ValueError('frozen cache capacities exceed the registered bounds')
        self.root,self.runtime=Path(root),runtime
        self.reader=reader_frozen.eval().requires_grad_(False)
        run=read_json(self.root/'run_contract.json')
        self.reader_identity=run['initial_reader_checkpoint'];self.source=str(runtime.source['model_path'])
        self.history_cache=_TensorCache(self.root/'cache/H',history_capacity)
        self.memory_cache=_TensorCache(self.root/'cache/M',memory_capacity)
        self.measurements=[]

    def memory(self,entry):
        if entry.get('kind','original')!='original':raise ValueError('only the registered original teaching is a condition')
        key=f'task_{int(entry["task"]):04d}_demo_{int(entry["demo"]):03d}'
        metadata=dict(cache_path=str(entry['cache_path']),reader=self.reader_identity,source=self.source)
        loaded={}
        def teacher():
            if not loaded:loaded.update(load_cached(entry['cache_path']))
            return loaded
        def required():return len(teacher()['indices'])*8*256*4
        @torch.no_grad()
        def build():
            value=teacher();device=self.runtime.device
            with autocast(device):_,memory=self.reader.prepare_video(value['z_img'].to(device),value['indices'].to(device),checkpoint_layers=False)
            return memory
        value,measure=self.memory_cache.get(key,metadata,required,build)
        if value.ndim!=3 or value.shape[0]!=1 or value.shape[2]!=256 or not value.shape[1] or value.shape[1]%8:
            raise ValueError('frozen teaching memory lost full frame slots')
        self.measurements.append(dict(kind='teaching',key=key,frames=value.shape[1]//8,**measure))
        return value

    def _observations(self,path,history,image_chunk):
        keys=list(history.observations);metadata=dict(source=self.source,observations=len(keys))
        @torch.no_grad()
        def build():
            result=torch.empty(len(keys),512,2048,dtype=torch.bfloat16)
            for first in range(0,len(keys),image_chunk):
                part=keys[first:first+image_chunk]
                values=observation_features(self.runtime,{k:history.observations[k] for k in part},chunk=image_chunk)
                for offset,key in enumerate(part):result[first+offset].copy_(values[key])
            return result
        phi,measure=self.history_cache.get(str(Path(path).resolve()),metadata,lambda:len(keys)*512*2048*2,build)
        if phi.shape!=(len(keys),512,2048) or phi.dtype!=torch.bfloat16:raise ValueError('cached H observations lost complete BF16 patches')
        self.measurements.append(dict(kind='history',archive=str(path),frames=len(keys),events=len(history.records),**measure))
        return phi,{key:i for i,key in enumerate(keys)}

    def events(self,history_paths,image_chunk=16):
        if type(image_chunk) is not int or image_chunk<=0:raise ValueError('image_chunk must be positive')
        if isinstance(history_paths,(str,Path)):history_paths=[history_paths]
        quantiles={k:getattr(self.runtime.processor,'_'+k).detach().cpu() for k in ('state_q01','state_q99','action_q01','action_q99')}
        result=[];episode_offset=0
        for path in history_paths:
            history=load_history(path);phi,indices=self._observations(path,history,image_chunk)
            for record in history.records:
                pre,post=indices[record['pre']],indices[record['post']]
                images=phi[pre:post+1] if post==pre+1 else torch.stack((phi[pre],phi[post]))
                hidden=record['hidden'].detach();y=torch.as_tensor(history.states[record['parameter_ref']],dtype=torch.float32).detach()
                if hidden.shape!=(2,50,1024) or y.shape!=(64,):raise ValueError('actual H lost native hidden or behavior code')
                result.append(dict(images=images,hidden=hidden,facts=event_facts(self.runtime.processor,record,record['episode']+episode_offset,quantiles),
                    y=y,parameter_ref=record['parameter_ref'],episode=record['episode']+episode_offset,step=record['step']))
            episode_offset+=len(history.episodes)
        return result
