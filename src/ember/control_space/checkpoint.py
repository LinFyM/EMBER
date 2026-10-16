"""Atomic fresh H-revision/Adam/cursor checkpoints and explicit topology migration."""
from pathlib import Path
import os,random,shutil
import numpy as np
import torch
import torch.distributed as dist
from ember.pi05_source_checkpoint import capture_rng,restore_rng,write_json_atomic,read_json,barrier
from .contract import SCHEMA,BASIS,INITIAL_READER
from ember.reference.contract import ASSETS,SOURCE,NORMALIZATION


def path(root,update):return Path(root)/'checkpoints'/f'update_{update:08d}'/'state.pt'


def save(root,reader,optimizer,scheduler,stream,context,units,physical,*,training_seconds):
    update=stream.next_update;destination=path(root,update).parent
    local=dict(rank=context.rank,local_rank=context.local_rank,RNG=capture_rng(context),device=str(context.device),
        NUMA_node=context.numa_node,CPU_affinity=context.cpu_affinity,hostname=os.uname().nodename)
    states=[None]*context.world_size
    if context.world_size>1:dist.all_gather_object(states,local)
    else:states=[local]
    if context.is_main:
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():raise ValueError('saved effective revision update cannot be silently replaced')
        temporary=destination.with_name(destination.name+f'.partial_{os.getpid()}');temporary.mkdir(exist_ok=False)
        state=dict(schema=SCHEMA,phase='functional_feedback',next_update=update,revision=reader.state_dict(),
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=None,stream=stream.state_dict(),
            rank_RNG=states,topology=dict(world_size=context.world_size,hostname=os.uname().nodename,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')),
            frozen_basis=str(BASIS),frozen_reader=str(INITIAL_READER),frozen_source=str(ASSETS/SOURCE),normalization=str(ASSETS/NORMALIZATION),
            code_units=units.detach().cpu(),physical=physical,training_seconds=training_seconds)
        torch.save(state,temporary/'state.pt')
        write_json_atomic(temporary/'manifest.json',dict(schema=SCHEMA,complete=True,next_update=update,
            revision=True,Adam=True,scheduler=True,scaler=None,full_stream=True,rank_RNG=context.world_size,frozen_basis=str(BASIS)))
        temporary.rename(destination);write_json_atomic(Path(root)/'checkpoints/latest.json',dict(update=update,path=str(destination/'state.pt')))
        retire(root)
    barrier(context)


def retire(root):
    folder=Path(root)/'checkpoints'
    rolling=sorted(p for p in folder.glob('update_*') if p.is_dir() and '.partial_' not in p.name and int(p.name.split('_')[-1]) not in (90,180))
    for old in rolling[:-2]:
        record=read_json(old/'manifest.json')
        if record['schema']!=SCHEMA or not record['complete']:raise ValueError('ambiguous checkpoint cannot be retired')
        with (folder/'retention.jsonl').open('a') as handle:handle.write(__import__('json').dumps(dict(retired=str(old),update=record['next_update']))+'\n')
        shutil.rmtree(old)


def restore(root,update,reader,optimizer,scheduler,context):
    state=torch.load(path(root,update),map_location='cpu',weights_only=False)
    if (state['schema']!=SCHEMA or state['phase']!='functional_feedback' or state['next_update']!=update
            or state['frozen_basis']!=str(BASIS) or state['frozen_reader']!=str(INITIAL_READER)
            or state['frozen_source']!=str(ASSETS/SOURCE) or state['normalization']!=str(ASSETS/NORMALIZATION)):
        raise ValueError('revision resume identity/phase/logical cursor changed')
    reader.load_state_dict(state['revision'],strict=True);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    if context.rank<len(state['rank_RNG']):restore_rng(state['rank_RNG'][context.rank]['RNG'],context);rng='saved_rank'
    else:
        seed=7+update*1009+context.rank;random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed(seed);rng=f'new_rank_seed{seed}'
    migration=dict(from_topology=state['topology'],to_world_size=context.world_size,rank=context.rank,RNG_source=rng,
        logical_update=update,bitwise_exact=False)
    return state,migration


def latest(root):
    """A renamed complete transaction wins if the small latest pointer was interrupted."""
    folder=Path(root)/'checkpoints';updates=[]
    for item in folder.glob('update_*'):
        if '.partial_' in item.name or not (item/'manifest.json').is_file():continue
        record=read_json(item/'manifest.json')
        if record['schema']!=SCHEMA or not record['complete'] or not (item/'state.pt').is_file():
            raise ValueError('incomplete named revision checkpoint')
        updates.append(record['next_update'])
    update=max(updates,default=0)
    if (folder/'latest.json').is_file() and read_json(folder/'latest.json')['update']>update:
        raise ValueError('latest revision checkpoint lost its complete state')
    return update
