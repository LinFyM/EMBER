"""Atomic phase migration and bounded rolling complete E/D recovery states."""
from pathlib import Path
import os,random,shutil
import numpy as np
import torch
import torch.distributed as dist
from ember.pi05_source_checkpoint import capture_rng,restore_rng,read_json,write_json_atomic
from .contract import SCHEMA,PARENT_ROOT,PARENT_UPDATE


def gather(value,context):
    if context.world_size==1:return [value]
    rows=[None]*context.world_size;dist.all_gather_object(rows,value);return rows


def latest(root):
    path=Path(root)/'checkpoint_latest.json'
    return read_json(path) if path.exists() else dict(update=PARENT_UPDATE,path=str(PARENT_ROOT/'checkpoints/update_00000144/state.pt'))


def save(root,update,model,optimizer,scheduler,context,physical,clock,stop,*,keep=False):
    rng=gather(capture_rng(context),context)
    if context.is_main:
        root=Path(root);destination=root/'checkpoints'/f'update_{update:08d}'
        temporary=destination.with_name(destination.name+f'.partial_{os.getpid()}')
        temporary.mkdir(parents=True,exist_ok=False)
        state=dict(schema=SCHEMA,phase='operator_from144',next_update=update,phase_update=update-144,
            parent_checkpoint=str(PARENT_ROOT/'checkpoints/update_00000144/state.pt'),
            model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            sampler=dict(seed=7,next_update=update,visits_per_task=update//9,task_domain=1010,point_domain=1011,
                         parent_queries_per_task=112,phase_queries_consumed=0),
            rank_rng=rng,topology=dict(world_size=context.world_size,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),
            units=read_json(root/'units.json'),operator_constants=read_json(root/'operator_constants.json'),
            bank=read_json(root/'bank.json'),physical=physical,clock=clock,stop=stop)
        torch.save(state,temporary/'state.pt')
        write_json_atomic(temporary/'manifest.json',dict(schema=SCHEMA,update=update,complete=True,
            optimizer_rng_sampler=True,retained=keep,clock=clock,stop=stop))
        temporary.rename(destination)
        previous=latest(root)
        write_json_atomic(root/'checkpoint_latest.json',dict(update=update,path=str(destination/'state.pt')))
        # New complete state is durable before removing an older replaceable point.
        preserved={update,previous['update']}
        for manifest in (root/'checkpoints').glob('update_*/manifest.json'):
            record=read_json(manifest)
            if record['update'] not in preserved and not record['retained']:shutil.rmtree(manifest.parent)
    if context.world_size>1:dist.barrier()


def restore(root,model,optimizer,scheduler,context):
    record=latest(root);state=torch.load(record['path'],map_location='cpu',weights_only=False)
    parent=state['next_update']==144 and state['schema']=='ember_source_control_space_r8_seed7_v1'
    if not parent and (state['schema']!=SCHEMA or state['phase']!='operator_from144'):
        raise ValueError('incompatible objective migration/recovery schema')
    if state['bank']!=read_json(root/'bank.json') or state['units']!=read_json(root/'units.json'):
        raise ValueError('frozen bank or 76 original S units changed')
    if not parent and state['operator_constants']!=read_json(root/'operator_constants.json'):
        raise ValueError('operator units changed during recovery')
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    if context.rank<len(state['rank_rng']):restore_rng(state['rank_rng'][context.rank],context)
    else:
        random.seed(7+context.rank);np.random.seed(7+context.rank);torch.manual_seed(7+context.rank)
    if context.is_main:
        write_json_atomic(Path(root)/f'resume_{record["update"]:08d}_world{context.world_size}.json',dict(
            previous_topology=state['topology'],world_size=context.world_size,original_rng_ranks=len(state['rank_rng']),
            rank_rng=state['rank_rng'].__len__(),new_rank_rule='seed7_plus_rank',objective_migration=parent,
            same_objective_exact=False,bitwise_exact=False,parent_query_cursor_preserved=112,query_consumption=0))
    return state
