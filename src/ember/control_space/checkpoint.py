"""Atomic complete E/D states at the four registered recovery boundaries."""
from pathlib import Path
import os,random
import numpy as np
import torch
import torch.distributed as dist
from ember.pi05_source_checkpoint import capture_rng,restore_rng,read_json,write_json_atomic
from .contract import SCHEMA,SAVES


def gather(value,context):
    if context.world_size==1:return [value]
    rows=[None]*context.world_size;dist.all_gather_object(rows,value);return rows


def save(root,update,model,optimizer,scheduler,context,physical):
    if update not in SAVES:raise ValueError('unregistered recovery boundary')
    rng=gather(capture_rng(context),context)
    if context.is_main:
        destination=Path(root)/'checkpoints'/f'update_{update:08d}'
        temporary=destination.with_name(destination.name+f'.partial_{os.getpid()}')
        temporary.mkdir(parents=True,exist_ok=False)
        state=dict(schema=SCHEMA,next_update=update,model=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),sampler=dict(seed=7,next_update=update,visits_per_task=update//9,
                queries_per_task=7*(update//9),task_domain=1010,point_domain=1011,noise_domain=1012),
            rank_rng=rng,topology=dict(world_size=context.world_size,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
            units=read_json(Path(root)/'units.json'),bank=read_json(Path(root)/'bank.json'),physical=physical)
        torch.save(state,temporary/'state.pt')
        write_json_atomic(temporary/'manifest.json',dict(schema=SCHEMA,update=update,complete=True,
            source_frozen=True,rank=8,targets=38,optimizer_rng_sampler=True,readout=update==144))
        temporary.rename(destination)
    if context.world_size>1:dist.barrier()


def restore(root,update,model,optimizer,scheduler,context):
    if update not in SAVES:raise ValueError('unregistered resume boundary')
    state=torch.load(Path(root)/'checkpoints'/f'update_{update:08d}'/'state.pt',map_location='cpu',weights_only=False)
    if (state['schema'],state['next_update'],state['bank'],state['units'])!=(SCHEMA,update,
            read_json(Path(root)/'bank.json'),read_json(Path(root)/'units.json')):
        raise ValueError('E/D bank/units/schema/cursor changed')
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    if context.rank<len(state['rank_rng']):restore_rng(state['rank_rng'][context.rank],context);rng_source='saved_rank'
    else:
        random.seed(7+context.rank);np.random.seed(7+context.rank);torch.manual_seed(7+context.rank);rng_source='new_rank_seed7_plus_rank'
    if context.is_main:
        write_json_atomic(Path(root)/f'resume_{update:08d}_world{context.world_size}.json',dict(
            previous_topology=state['topology'],world_size=context.world_size,
            original_rng_ranks=len(state['rank_rng']),new_rank_rule='seed7_plus_rank',bitwise_exact=False,rng_source=rng_source))
    return state['physical']
