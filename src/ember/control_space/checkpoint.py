"""Complete fresh native-writer recovery and one bounded atomic rolling point."""
from pathlib import Path
import os,random,shutil,json
import numpy as np
import torch
import torch.distributed as dist
from ember.pi05_source_checkpoint import capture_rng,restore_rng,write_json_atomic,read_json,barrier
from .contract import SCHEMA,PHASE,PERMANENT,INITIAL_READER,ASSETS,SOURCE,NORMALIZATION,OBJECTIVE,checkpoint_path,H_ROOT


def omega_identity(lora):
    return [dict(target=t.name,shape=[t.in_features,8],seed=2026101700+i,std=t.in_features**-.5)
            for i,t in enumerate(lora.targets)]


def save(root,model,optimizer,scheduler,stream,context,physical,*,update,training_seconds,disposable=False,frozen_response_cache=None):
    destination=checkpoint_path(root,update).parent
    local=dict(rank=context.rank,RNG=capture_rng(context),device=str(context.device),NUMA_node=context.numa_node,
               frozen_source_F0=frozen_response_cache or {},
               CPU_affinity=context.cpu_affinity,hostname=os.uname().nodename)
    ranks=[None]*context.world_size
    if context.world_size>1:dist.all_gather_object(ranks,local)
    else:ranks=[local]
    if context.is_main:
        from ember.reference.contract import ASSETS as shared_assets
        from ember.pi05_lora import load_pi05_lora_contract,derive_pi05_lora_rank
        lora=derive_pi05_lora_rank(load_pi05_lora_contract(shared_assets/'configs/pi05_lora_rank128_aligned.json'),rank=8)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():raise ValueError('a complete checkpoint cannot be overwritten')
        temporary=destination.with_name(destination.name+f'.partial_{os.getpid()}');temporary.mkdir(exist_ok=False)
        state=dict(schema=SCHEMA,phase=PHASE,next_update=update,objective=OBJECTIVE,disposable=disposable,
            model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=None,
            stream=stream.state_dict(),profile_cursor=update if disposable else None,
            profile_task_visits={t:update for t in (0,13,20,32)} if disposable else None,
            rank_RNG=ranks,topology=dict(world_size=context.world_size,hostname=os.uname().nodename,
                CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')),source=str(ASSETS/SOURCE),
            normalization=str(ASSETS/NORMALIZATION),frozen_reader=str(INITIAL_READER),source_H=str(H_ROOT),
            omega=omega_identity(lora),physical=physical,training_seconds=training_seconds)
        torch.save(state,temporary/'state.pt')
        write_json_atomic(temporary/'manifest.json',dict(schema=SCHEMA,phase=PHASE,complete=True,next_update=update,
            full_model=True,Adam=True,scheduler=True,scaler=None,full_stream=True,rank_RNG=context.world_size,disposable=disposable))
        temporary.rename(destination)
        write_json_atomic(destination.parent/'latest.json',dict(update=update,path=str(destination/'state.pt')))
        if not disposable:
            rolling=sorted(p for p in destination.parent.glob('update_*') if '.partial_' not in p.name
                           and int(p.name.split('_')[-1]) not in PERMANENT)
            for old in rolling[:-1]:
                receipt=read_json(old/'manifest.json')
                if receipt['schema']!=SCHEMA or not receipt['complete']:raise ValueError('ambiguous rolling checkpoint')
                with (destination.parent/'retention.jsonl').open('a') as h:h.write(json.dumps(dict(retired=str(old),update=receipt['next_update']))+'\n')
                shutil.rmtree(old)
    barrier(context)


def validate(state,update,*,disposable=False):
    if (state['schema']!=SCHEMA or state['phase']!=PHASE or state['next_update']!=update
            or state['objective']!=OBJECTIVE or state['disposable']!=disposable
            or state['source']!=str(ASSETS/SOURCE) or state['normalization']!=str(ASSETS/NORMALIZATION)
            or state['frozen_reader']!=str(INITIAL_READER) or state['source_H']!=str(H_ROOT)):
        raise ValueError('native first-write checkpoint/phase/source/H/optimizer scope changed')
    from ember.pi05_lora import load_pi05_lora_contract,derive_pi05_lora_rank
    lora=derive_pi05_lora_rank(load_pi05_lora_contract(ASSETS/'configs/pi05_lora_rank128_aligned.json'),rank=8)
    if state['omega']!=omega_identity(lora):raise ValueError('public Omega identity changed across recovery')


def restore(root,update,model,optimizer,scheduler,context,*,disposable=False):
    state=torch.load(checkpoint_path(root,update),map_location='cpu',weights_only=False)
    validate(state,update,disposable=disposable)
    model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    if context.rank<len(state['rank_RNG']):restore_rng(state['rank_RNG'][context.rank]['RNG'],context);origin='saved_rank'
    else:
        seed=7+update*1009+context.rank;random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed(seed)
        origin=f'new_rank_seed{seed}'
    return state,dict(from_topology=state['topology'],to_world_size=context.world_size,rank=context.rank,
        RNG_source=origin,logical_update=update,bitwise_exact=False)


def load_model(root,update,device):
    from .model import NativeOperatorContext
    state=torch.load(checkpoint_path(root,update),map_location='cpu',weights_only=False,mmap=True)
    validate(state,update)
    with torch.random.fork_rng(devices=[]):model=NativeOperatorContext()
    model.load_state_dict(state['model'],strict=True)
    del state
    return model.to(device)


def latest(root):
    points=[]
    for p in (Path(root)/'checkpoints').glob('update_*'):
        if '.partial_' in p.name:continue
        manifest=read_json(p/'manifest.json')
        if manifest['schema']!=SCHEMA or not manifest['complete'] or not (p/'state.pt').is_file():
            raise ValueError('incomplete named native-writer checkpoint')
        points.append(manifest['next_update'])
    return max(points,default=0)
