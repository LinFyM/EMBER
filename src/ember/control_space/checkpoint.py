"""Independent meta-update transactions, fresh Adam and complete behavior cursors."""
from pathlib import Path
import os,random,shutil,subprocess
import numpy as np
import torch
import torch.distributed as dist
from ember.pi05_source_checkpoint import read_json,write_json_atomic,barrier
from .contract import SCHEMA,BASIS,INITIAL_READER,checkpoint as path


def rng(context):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state(context.device) if context.device.type=='cuda' else None)


def save(root,model,optimizer,scheduler,stream,context,units,baselines,beta,physical,training_seconds):
    update=stream.next_update;destination=path(root,update).parent
    local=dict(rank=context.rank,RNG=rng(context),hostname=os.uname().nodename,device=str(context.device))
    ranks=[None]*context.world_size
    if context.world_size>1:dist.all_gather_object(ranks,local)
    else:ranks=[local]
    if context.is_main:
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():raise ValueError('an effective saved meta update must not be overwritten')
        temporary=destination.with_name(destination.name+f'.partial_{os.getpid()}');temporary.mkdir(exist_ok=False)
        state=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[3],text=True).strip(),schema=SCHEMA,phase='feedback_proposals',meta_update=update,model=model.state_dict(),
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=None,stream=stream.state_dict(),
            rank_RNG=ranks,topology=dict(world_size=context.world_size,hostname=os.uname().nodename,
                CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),device=str(context.device)),
            frozen_basis=str(BASIS),initial_reader_checkpoint=str(INITIAL_READER),code_units=units.detach().cpu(),
            baselines=baselines,beta=beta,physical=physical,training_seconds=training_seconds,
            behavior_paths=str(Path(root)/'conditions'),fresh_optimizer_from900=True)
        torch.save(state,temporary/'state.pt')
        write_json_atomic(temporary/'manifest.json',dict(schema=SCHEMA,complete=True,meta_update=update,
            Adam=True,scheduler=True,RNG_ranks=context.world_size,full_stream=True,beta=beta,full_behavior_paths=True))
        temporary.rename(destination);write_json_atomic(Path(root)/'checkpoints/latest.json',dict(meta_update=update,path=str(destination/'state.pt')))
        retire(root)
    barrier(context)


def retire(root):
    folder=Path(root)/'checkpoints';rolling=sorted(p for p in folder.glob('meta_update_*')
        if '.partial_' not in p.name and int(p.name.split('_')[-1]) not in (0,9))
    for old in rolling[:-2]:
        record=read_json(old/'manifest.json')
        if not record['complete'] or record['schema']!=SCHEMA:raise ValueError('ambiguous checkpoint cannot retire')
        with (folder/'retention.jsonl').open('a') as f:f.write(__import__('json').dumps(dict(retired=str(old),meta_update=record['meta_update']))+'\n')
        shutil.rmtree(old)


def load(root,update):
    state=torch.load(path(root,update),map_location='cpu',weights_only=False,mmap=True)
    if (state['schema'],state['phase'],state['meta_update'],state['frozen_basis'])!=(SCHEMA,'feedback_proposals',update,str(BASIS)):
        raise ValueError('independent feedback checkpoint identity/phase mismatch')
    return state


def restore_rng(state,context):
    if context.rank<len(state['rank_RNG']):
        saved=state['rank_RNG'][context.rank]['RNG'];random.setstate(saved['python']);np.random.set_state(saved['numpy']);torch.set_rng_state(saved['torch_cpu'])
        if saved['torch_cuda'] is not None:torch.cuda.set_rng_state(saved['torch_cuda'],context.device);source='saved_rank'
        else:torch.cuda.manual_seed(7);source='fresh_CUDA_seed7_from_CPU_initializer'
    else:
        seed=7+state['meta_update']*1009+context.rank;random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);source=f'new_rank_seed{seed}'
    return dict(from_topology=state['topology'],world_size=context.world_size,rank=context.rank,RNG_source=source,bitwise_exact=False)


def latest(root):
    values=[]
    for p in (Path(root)/'checkpoints').glob('meta_update_*'):
        if '.partial_' in p.name:continue
        m=read_json(p/'manifest.json')
        if m['schema']!=SCHEMA or not m['complete'] or not (p/'state.pt').is_file():raise ValueError('incomplete named meta checkpoint')
        values.append(m['meta_update'])
    return max(values,default=-1)
