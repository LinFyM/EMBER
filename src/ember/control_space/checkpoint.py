"""Complete new-generator Adam/stream/RNG recovery, with bounded rolling retention."""
from pathlib import Path
import os,random,shutil,json
import numpy as np
import torch
import torch.distributed as dist
from ember.pi05_source_checkpoint import capture_rng,restore_rng,write_json_atomic,read_json,barrier
from .contract import SCHEMA,PHASE,PERMANENT,INITIAL_READER,ASSETS,SOURCE,NORMALIZATION,SOURCE_INPUTS,OPERATOR_FILE,OBJECTIVE,checkpoint_path


def save(root,model,optimizer,scheduler,stream,context,physical,*,training_seconds):
    update=stream.next_update;destination=checkpoint_path(root,update).parent
    local=dict(rank=context.rank,RNG=capture_rng(context),device=str(context.device),NUMA_node=context.numa_node,
        CPU_affinity=context.cpu_affinity,hostname=os.uname().nodename)
    ranks=[None]*context.world_size
    if context.world_size>1:dist.all_gather_object(ranks,local)
    else:ranks=[local]
    if context.is_main:
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():raise ValueError('a valid new-generator checkpoint cannot be overwritten')
        temporary=destination.with_name(destination.name+f'.partial_{os.getpid()}');temporary.mkdir(exist_ok=False)
        state=dict(schema=SCHEMA,phase=PHASE,next_update=update,stage='pair' if update<=1000 else 'joint',objective=OBJECTIVE,generator=model.state_dict(),
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=None,stream=stream.state_dict(),rank_RNG=ranks,
            topology=dict(world_size=context.world_size,hostname=os.uname().nodename,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')),
            source_inputs=str(SOURCE_INPUTS),operator_constants=str(OPERATOR_FILE),
            frozen_reader=str(INITIAL_READER),frozen_source=str(ASSETS/SOURCE),normalization=str(ASSETS/NORMALIZATION),
            physical=physical,training_seconds=training_seconds)
        torch.save(state,temporary/'state.pt')
        write_json_atomic(temporary/'manifest.json',dict(schema=SCHEMA,complete=True,next_update=update,stage=state['stage'],
            full_generator=True,Adam=True,scheduler=True,scaler=None,full_stream=True,rank_RNG=context.world_size))
        temporary.rename(destination);write_json_atomic(destination.parent/'latest.json',dict(update=update,path=str(destination/'state.pt')))
        retire(root)
    barrier(context)


def retire(root):
    folder=Path(root)/'checkpoints'
    rolling=sorted(p for p in folder.glob('update_*') if p.is_dir() and '.partial_' not in p.name and int(p.name.split('_')[-1]) not in PERMANENT)
    for old in rolling[:-1]:
        record=read_json(old/'manifest.json')
        if record['schema']!=SCHEMA or not record['complete']:raise ValueError('ambiguous checkpoint cannot be retired')
        with (folder/'retention.jsonl').open('a') as handle:handle.write(json.dumps(dict(retired=str(old),update=record['next_update']))+'\n')
        shutil.rmtree(old)


def restore(root,update,model,optimizer,scheduler,context):
    state=torch.load(checkpoint_path(root,update),map_location='cpu',weights_only=False)
    if (state['schema']!=SCHEMA or state['phase']!=PHASE or state['next_update']!=update
            or state['source_inputs']!=str(SOURCE_INPUTS) or state.get('objective')!=OBJECTIVE or state.get('operator_constants')!=str(OPERATOR_FILE) or state['frozen_reader']!=str(INITIAL_READER)
            or state['frozen_source']!=str(ASSETS/SOURCE) or state['normalization']!=str(ASSETS/NORMALIZATION)):
        raise ValueError('new source-weight generator identity/phase/logical cursor changed')
    model.load_state_dict(state['generator'],strict=True);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    if context.rank<len(state['rank_RNG']):restore_rng(state['rank_RNG'][context.rank]['RNG'],context);origin='saved_rank'
    else:
        seed=7+update*1009+context.rank;random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed(seed);origin=f'new_rank_seed{seed}'
    return state,dict(from_topology=state['topology'],to_world_size=context.world_size,rank=context.rank,RNG_source=origin,
        logical_update=update,bitwise_exact=False)


def latest(root):
    updates=[]
    for item in (Path(root)/'checkpoints').glob('update_*'):
        if '.partial_' in item.name:continue
        record=read_json(item/'manifest.json')
        if record['schema']!=SCHEMA or not record['complete'] or not (item/'state.pt').is_file():raise ValueError('incomplete named generator checkpoint')
        updates.append(record['next_update'])
    return max(updates,default=0)
