"""Frozen teaching memory and globally bounded unique-observation H features."""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import fcntl
import json
import os
import sqlite3
import time

import torch

from ember.pi05_source_checkpoint import read_json
from ember.writer.runtime import autocast
from .practice import load_history

FEATURE_SCHEMA='ember_source_image_prefix_features_v1'
LEGACY_SCHEMA='ember_image_only_gemma_compiler_v1'
CACHE_SCHEMA='ember_frozen_tensor_cache_v2'
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


def load_video(runtime,descriptor):
    """RGB-only store owns stride5, terminal and exactly one180-degree rotation."""
    if descriptor['kind']!='original':raise ValueError('current teaching is an original RGB-only video')
    return runtime.videos.load(int(descriptor['task']),int(descriptor['demo']))


@torch.no_grad()
def extract(runtime, video, frame_chunk=32):
    """Run all512 real image tokens through frozen Gemma, with no L or suffix."""
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, resize_with_pad_torch
    from ember.operator_writer.native import _teacher_attention_kernel
    frames = torch.from_numpy(video.frames)
    if (frames.dtype != torch.uint8 or frames.ndim != 5 or frames.shape[1:3] != (2, 3)
            or len(frames) != len(video.frame_indices) or not len(frames)
            or type(frame_chunk) is not int or frame_chunk <= 0):
        raise ValueError('source image prefix requires full ordered dual-camera uint8 RGB')
    indices = _indices(video.frame_indices, video.raw_frame_count)
    if getattr(runtime, '_generator', None) is not None or runtime.execution._active_state is not None:
        raise ValueError('pure source image extraction cannot overlap learned or execution adapters')
    core, bridge = runtime.policy.model, runtime.policy.model.paligemma_with_expert
    dtype = bridge.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
    runtime.policy.eval().requires_grad_(False)
    images, chunks = [], []
    started = time.monotonic()
    for first in range(0, len(frames), frame_chunk):
        selected = frames[first:first + frame_chunk].to(runtime.device)
        with autocast(torch.device(runtime.device)):
            pixels = selected.flatten(0, 1).float().div(255).permute(0, 2, 3, 1)
            pixels = (resize_with_pad_torch(pixels, 224, 224) * 2 - 1).permute(0, 3, 1, 2)
            raw = bridge.embed_image(pixels).reshape(len(selected), 512, 2048)
            padding = torch.ones(len(selected), 512, dtype=torch.bool, device=runtime.device)
            mask = core._prepare_attention_masks_4d(make_att_2d_masks(padding, torch.zeros_like(padding)))
            with _teacher_attention_kernel(mask):
                (z, suffix), _ = bridge(attention_mask=mask,
                    position_ids=torch.arange(512, device=runtime.device)[None].expand(len(selected), -1),
                    past_key_values=None, inputs_embeds=[raw.to(dtype), None], use_cache=False, adarms_cond=[None, None])
            if suffix is not None or z.shape != (len(selected), 512, 2048) or not bool(torch.isfinite(z).all()):
                raise ValueError('pure image-prefix post-norm output changed')
            images.append(z.detach().to(dtype=torch.bfloat16, device='cpu').contiguous())
        chunks.append(len(selected))
    return dict(z_img=torch.cat(images), indices=indices,
        source=dict(model=str(runtime.source['model_path']), dual_rgb=True, stride=5, post_norm=True,
            Gemma_input='512 image tokens only', L_in_Gemma=False, action_suffix=False, probe=False,
            old_meta_loaded=False, preprocess=dict(PREPROCESS), input_resolution=list(frames.shape[-2:]),
            raw_frames=video.raw_frame_count, frame_chunks=chunks, seconds=time.monotonic()-started))

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
    """SQLite owns identities, budget and LRU; immutable payloads identify themselves.

    Old JSON/tensor-only caches require an explicit forensic cleanup before use.
    No old payload is adopted and no second cache directory bypasses its quota.
    """
    def __init__(self,root,capacity):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.capacity=int(capacity);self.control=min(8*2**20,max(65536,self.capacity//32))
        self.database=self.root/'index.sqlite'
        if (self.root/'manifest.json').exists() or (self.root/'manifest.partial').exists() or next(self.root.glob('tensor_*.pt*'),None):
            raise RuntimeError('legacy frozen cache needs preserved forensic metadata and explicit parent cleanup before SQLite initialization; no tensor-only adoption')
        with self.transaction() as db:
            db.execute('CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY,value TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS tensors (id INTEGER PRIMARY KEY AUTOINCREMENT,key TEXT UNIQUE NOT NULL,metadata TEXT NOT NULL,status TEXT NOT NULL,bytes INTEGER NOT NULL,touched REAL NOT NULL)')
            for key,value in (('schema',CACHE_SCHEMA),('capacity',str(self.capacity))):
                db.execute('INSERT OR IGNORE INTO settings VALUES (?,?)',(key,value))
                if db.execute('SELECT value FROM settings WHERE name=?',(key,)).fetchone()[0]!=value:
                    raise ValueError('wrong frozen cache schema/capacity')
            if not db.execute('SELECT count(*) FROM tensors').fetchone()[0] and (next(self.root.glob('entry_*.pt'),None) or next(self.root.glob('entry_*.pt.partial'),None)):
                raise RuntimeError('unindexed frozen cache payloads need explicit parent forensic cleanup')

    @contextmanager
    def transaction(self):
        # Rollback journal, never WAL on NFS. A stable database inode plus SQLite
        # byte-range locks replaces independently cached JSON rename generations.
        db=sqlite3.connect(self.database,timeout=60,isolation_level=None)
        db.row_factory=sqlite3.Row
        db.execute('PRAGMA busy_timeout=60000');db.execute('PRAGMA temp_store=MEMORY')
        db.execute('PRAGMA synchronous=FULL')
        if db.execute('PRAGMA journal_mode').fetchone()[0]!='delete':
            db.close();raise ValueError('frozen cache requires SQLite DELETE rollback journal')
        try:
            db.execute('BEGIN IMMEDIATE');yield db
            size=db.execute('PRAGMA page_count').fetchone()[0]*db.execute('PRAGMA page_size').fetchone()[0]
            if 3*size>self.control:raise RuntimeError('frozen cache database/journal exceeded control allowance')
            db.execute('COMMIT')
        except BaseException:
            if db.in_transaction:db.execute('ROLLBACK')
            raise
        finally:db.close()

    def entry(self,key,metadata):
        encoded=json.dumps(metadata,sort_keys=True)
        with self.transaction() as db:
            db.execute("INSERT OR IGNORE INTO tensors(key,metadata,status,bytes,touched) VALUES (?,?,'absent',0,?)",(key,encoded,time.time()))
            entry=db.execute('SELECT * FROM tensors WHERE key=?',(key,)).fetchone()
            if entry['metadata']!=encoded:raise ValueError('immutable archive/source feature identity changed')
            return f'entry_{entry["id"]:010d}.pt'

    def reserve(self,db,key,needed):
        used=db.execute('SELECT coalesce(sum(bytes),0) FROM tensors').fetchone()[0]
        victims=db.execute("SELECT id,key,bytes FROM tensors WHERE key!=? AND status='ready' ORDER BY touched",(key,)).fetchall()
        for old in victims:
            if used+needed<=self.capacity-self.control:break
            (self.root/f'entry_{old["id"]:010d}.pt').unlink(missing_ok=True)
            db.execute("UPDATE tensors SET status='absent',bytes=0 WHERE key=?",(old['key'],));used-=old['bytes']
        available=used+needed<=self.capacity-self.control
        db.execute('UPDATE tensors SET status=?,bytes=?,touched=? WHERE key=?',
            ('reserved' if available else 'absent',needed if available else 0,time.time(),key))
        return available

    def release(self,key,destination,temporary):
        with self.transaction() as db:
            temporary.unlink(missing_ok=True);destination.unlink(missing_ok=True)
            db.execute("UPDATE tensors SET status='absent',bytes=0 WHERE key=?",(key,))

    @staticmethod
    def load(destination,key,metadata,*,mmap=False):
        if mmap:payload=torch.load(destination,map_location='cpu',weights_only=True,mmap=True)
        else:
            with destination.open('rb') as stream:payload=torch.load(stream,map_location='cpu',weights_only=True,mmap=False)
        if (not isinstance(payload,dict) or payload.get('schema')!=CACHE_SCHEMA or payload.get('key')!=key
                or payload.get('metadata')!=metadata or not isinstance(payload.get('value'),torch.Tensor)):
            raise ValueError('frozen cache payload schema/key/source metadata mismatch; preserve evidence')
        return payload['value']

    def evict(self,key,source):
        with self.transaction() as db:entry=db.execute('SELECT * FROM tensors WHERE key=?',(key,)).fetchone()
        if entry is None:return dict(key=key,indexed=False,deleted_bytes=0,files=[])
        file=f'entry_{entry["id"]:010d}.pt';destination=self.root/file;temporary=self.root/(file+'.partial')
        with _file_lock(self.root/(file+'.lock')):
            with self.transaction() as db:
                entry=db.execute('SELECT * FROM tensors WHERE key=?',(key,)).fetchone()
                if entry is None:return dict(key=key,indexed=False,deleted_bytes=0,files=[])
                if file!=f'entry_{entry["id"]:010d}.pt':raise RuntimeError('retirement requires all consumers closed')
                metadata=json.loads(entry['metadata'])
                if metadata.get('source')!=source:raise ValueError('retirement source differs from the frozen execution source')
                files=[]
                if destination.exists():
                    witness=self.load(destination,key,metadata,mmap=True);del witness
                    files.append(dict(path=str(destination),bytes=destination.stat().st_size,payload_identity_validated=True))
                if temporary.exists():
                    files.append(dict(path=str(temporary),bytes=temporary.stat().st_size,
                        payload_identity_validated=False,binding='own indexed key/source temporary; never consumed'))
                for artifact in files:Path(artifact['path']).unlink()
                db.execute("DELETE FROM tensors WHERE key=?",(key,))
                (self.root/(file+'.lock')).unlink(missing_ok=True)
                return dict(key=key,indexed=True,source=source,status_before=entry['status'],
                    charged_bytes_released=entry['bytes'],deleted_bytes=sum(x['bytes'] for x in files),files=files)

    def get(self,key,metadata,required,build):
        tick=time.monotonic();file=self.entry(key,metadata);destination=self.root/file;temporary=self.root/(file+'.partial')
        # Stable per-key flock only deduplicates builds and releases on death.
        # All index/budget decisions still belong to the one SQLite transaction.
        with _file_lock(self.root/(file+'.lock')):
            with self.transaction() as db:
                entry=db.execute('SELECT * FROM tensors WHERE key=?',(key,)).fetchone()
                if entry is None or file!=f'entry_{entry["id"]:010d}.pt':
                    raise RuntimeError('cache entry retired while requested; retirement requires all consumers closed')
                if entry['status'] in ('ready','reserved') and destination.exists():
                    value=self.load(destination,key,metadata);size=destination.stat().st_size
                    if size>entry['bytes']:raise RuntimeError('frozen cache payload exceeds its original reservation')
                    db.execute("UPDATE tensors SET status='ready',bytes=?,touched=? WHERE key=?",(size,time.time(),key))
                    return value,dict(hit=True,cached=True,bytes=size,seconds=time.monotonic()-tick)
                temporary.unlink(missing_ok=True)
                db.execute("UPDATE tensors SET status='absent',bytes=0 WHERE key=?",(key,))
            needed=int(required())+65536+len(key.encode())+len(json.dumps(metadata).encode())
            with self.transaction() as db:cached=self.reserve(db,key,needed)
            try:
                value=build().detach().cpu().contiguous()
                if not cached:return value,dict(hit=False,cached=False,bytes=value.numel()*value.element_size(),seconds=time.monotonic()-tick)
                payload=dict(schema=CACHE_SCHEMA,key=key,metadata=metadata,value=value)
                with temporary.open('wb') as raw:
                    torch.save(payload,_BoundedWriter(raw,needed));raw.flush();os.fsync(raw.fileno())
                with self.transaction() as db:
                    size=temporary.stat().st_size;temporary.replace(destination)
                    db.execute("UPDATE tensors SET status='ready',bytes=?,touched=? WHERE key=?",(size,time.time(),key))
                return value,dict(hit=False,cached=True,bytes=size,seconds=time.monotonic()-tick)
            except BufferError:
                self.release(key,destination,temporary)
                return value,dict(hit=False,cached=False,bytes=value.numel()*value.element_size(),seconds=time.monotonic()-tick)
            except BaseException:
                self.release(key,destination,temporary);raise


class FrozenFeatures:
    """One owner for immutable video M and unique raw-H observation features."""
    def __init__(self,root,runtime,reader_frozen,*,history_capacity=16*2**30,memory_capacity=2**30):
        if not 0<history_capacity<=16*2**30 or not 0<memory_capacity<=2**30:raise ValueError('frozen cache capacities exceed the registered bounds')
        self.root,self.runtime=Path(root),runtime
        self.reader=reader_frozen.eval().requires_grad_(False)
        run=read_json(self.root/'run_contract.json')
        self.reader_identity=run['initial_reader_checkpoint'];self.source=str(runtime.source['model_path']);self.phase=run.get('phase')
        self.history_cache=_TensorCache(self.root/'cache/H',history_capacity)
        self.memory_cache=_TensorCache(self.root/'cache/M',memory_capacity)
        self.measurements=[]

    def memory(self,entry,frame_chunk=32):
        if entry.get('kind','original')!='original':raise ValueError('only registered original teaching may condition revision')
        held=entry.get('held',False);task,demo=int(entry['task']),int(entry['demo']);raw_count=count=None
        if task in (3,6,11,16,23,26,31,39) and not held:raise ValueError('validation teaching requires explicit RGB-only held identity')
        key=f'task_{task:04d}_demo_{demo:03d}'
        if held:
            if task not in (3,6,11,16,23,26,31,39) or 'cache_path' in entry:raise ValueError('held teaching cannot consume a nonheld prefix cache')
            authority=self.runtime.videos.authorities[task]
            raw_count,count=self.runtime.videos.frame_counts(task,demo)
            metadata=dict(held=True,task=task,demo=demo,authority=str(authority.path),authority_bytes=authority.expected_bytes,
                reader=self.reader_identity,source=self.source,preprocess=PREPROCESS,raw_frames=raw_count)
        else:metadata=dict(cache_path=str(entry['cache_path']),reader=self.reader_identity,source=self.source)
        loaded={}
        def teacher():
            if not loaded:
                loaded.update(extract(self.runtime,load_video(self.runtime,entry),frame_chunk) if held else load_cached(entry['cache_path']))
            if loaded['source']['model']!=self.source:raise ValueError('teaching source differs from the frozen execution source')
            return loaded
        def required():return (count if held else len(teacher()['indices']))*8*256*4
        @torch.no_grad()
        def build():
            value=teacher();device=self.runtime.device
            with autocast(device):_,memory=self.reader.prepare_video(value['z_img'].to(device),value['indices'].to(device),checkpoint_layers=False)
            return memory
        try:value,measure=self.memory_cache.get(key,metadata,required,build)
        finally:loaded.clear()  # Full held prefix is RAM-only, never retained across calls.
        if value.ndim!=3 or value.shape[0]!=1 or value.shape[2]!=256 or not value.shape[1] or value.shape[1]%8:
            raise ValueError('frozen teaching memory lost full frame slots')
        frames=value.shape[1]//8
        self.measurements.append(dict(kind='teaching',key=key,identity=metadata,held=held,frames=frames,
            raw_frames=raw_count,frame_equivalent=frames,M_builds=int(not measure['hit']),M_hits=int(measure['hit']),
            prefix_frames_built=frames if held and not measure['hit'] else 0,new_prefix_files=0,**measure))
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

    def evict_history(self,archive_paths):
        """Parent calls after complete close; raw archives/facts are never deleted here."""
        if self.phase!='functional_feedback_validation180':raise ValueError('raw-H retirement is restricted to this validation batch')
        keys=[];allowed=(self.root/'episodes').resolve()
        for path in archive_paths:
            path=Path(path)
            if not path.is_absolute() or not path.resolve().is_relative_to(allowed):raise ValueError('retirement requires an own-ROOT absolute episode archive')
            keys.append(str(path.resolve()))
        rows=[self.history_cache.evict(key,self.source) for key in dict.fromkeys(keys)]
        return dict(complete=True,cache_root=str(self.history_cache.root),archives=rows,
            deleted_bytes=sum(r['deleted_bytes'] for r in rows),raw_deleted=False)

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
