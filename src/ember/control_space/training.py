"""Fresh joint FM and complete-path score updates, one saved meta transaction each."""
from pathlib import Path
import json,random,time,shutil
import numpy as np
import torch
import torch.distributed as dist
from ember.pi05_source_setup import initialize_distributed,initialize_deferred_process_group
from ember.pi05_source_checkpoint import DistributedContext,read_json,write_json_atomic
from ember.proposal_writer.learning import synchronize_gradients
from ember.reference.accounting import host_resources
from .contract import INITIAL_READER,BASIS,OPTIMIZER,condition_root,checkpoint
from .data import make_runtime,Stream
from .feedback import FeedbackReader
from .model import load_basis
from .learning import QueryBank,fm_gradients,pg_gradients,gradient_norm
from .checkpoint import save,load,restore_rng


def optimizer_for(model):
    optimizer=torch.optim.AdamW(model.parameters(),**OPTIMIZER)
    return optimizer,torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)


def initialize_fresh(root):
    if checkpoint(root,0).is_file():return
    torch.manual_seed(7);np.random.seed(7);random.seed(7)
    initial=torch.load(INITIAL_READER,map_location='cpu',mmap=True,weights_only=False)
    model=FeedbackReader();model.video.load_state_dict(initial['reader'],strict=True)
    optimizer,scheduler=optimizer_for(model);stream=Stream(root)
    baselines={int(k):v for k,v in read_json(Path(root)/'run_contract.json')['initial_baselines'].items()}
    save(root,model,optimizer,scheduler,stream,DistributedContext(0,0,1,torch.device('cpu')),initial['code_units'],
        baselines,None,dict(measurements=[]),0.)


def reduction(context,value,op=dist.ReduceOp.SUM):
    tensor=torch.as_tensor(value,device=context.device,dtype=torch.float64)
    if context.world_size>1:dist.all_reduce(tensor,op=op)
    return tensor.cpu().tolist()


def copy_gradients(model):return [None if p.grad is None else p.grad.detach().float().clone() for p in model.parameters()]
def norm(values):return float(torch.stack([v.square().sum() for v in values if v is not None]).sum().sqrt()) if any(v is not None for v in values) else 0.


def combined_gradients(runtime,model,decoder,units,queries,events,paths,context,physical,optimizer):
    ooms=[];micro=physical['microbatch'];pack=physical['task_pack'];chunk=physical['event_chunk']
    while True:
        optimizer.zero_grad(set_to_none=True);error=None
        try:
            fm=fm_gradients(runtime,model,decoder,units,queries,events,context,micro,pack)
            fm_local=copy_gradients(model);optimizer.zero_grad(set_to_none=True)
            pg=pg_gradients(runtime,model,paths,events,context,chunk)
        except torch.cuda.OutOfMemoryError as failure:error=repr(failure)
        if not reduction(context,int(error is not None),dist.ReduceOp.MAX):break
        optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache();ooms.append(dict(microbatch=micro,task_pack=pack,event_chunk=chunk,error=error))
        if chunk>4:chunk=max(4,chunk//2)
        elif micro>7:micro=max(7,micro//2)
        elif pack>1:pack=max(1,pack//2)
        else:raise RuntimeError('minimum useful physical FM/H workload OOM; preserve all actual paths')
    synchronize_gradients(model,context);pg_global=copy_gradients(model)
    for p,g in zip(model.parameters(),fm_local,strict=True):p.grad=g
    synchronize_gradients(model,context);fm_global=copy_gradients(model)
    return fm_global,pg_global,dict(**fm,**pg,OOM_attempts=ooms,microbatch=micro,task_pack=pack,event_chunk=chunk)


def apply_update(model,optimizer,scheduler,fm,pg,beta):
    fn,pn=norm(fm),norm(pg);calibrated=beta is None and pn>0
    if calibrated:beta=min(1.,.25*fn/pn) if fn>0 else 1.
    for p,f,g in zip(model.parameters(),fm,pg,strict=True):
        if f is None and (g is None or beta is None):p.grad=None;continue
        combined=torch.zeros_like(p) if f is None else f
        if g is not None and beta is not None:combined=combined+beta*g
        p.grad=combined
    head=gradient_norm(model.delta_head.parameters())
    body=gradient_norm(p for n,p in model.named_parameters() if not n.startswith(('video.','delta_head.')))
    combined=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
    applied=fn>0 or pn>0
    if applied:optimizer.step();scheduler.step()
    return dict(optimizer_step_applied=applied,FM_norm=fn,PG_norm=pn,beta=beta,beta_calibrated_now=calibrated,gradient_norm=combined,
        delta_head_norm=head,H_body_norm=body)


def train(args):
    if not 0<=args.start<9 or args.stop!=args.start+1:raise ValueError('one complete current logical update per train transaction')
    context=initialize_distributed(require_numa=True,defer_process_group=True);torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=True
    runtime=make_runtime(args.root,context.device);saved=load(args.root,args.start)
    model=FeedbackReader().to(context.device);model.load_state_dict(saved['model']);model.train()
    optimizer,scheduler=optimizer_for(model);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
    migration=restore_rng(saved,context);stream=Stream(args.root,saved['stream']);events=stream.next()
    recorded=read_json(args.root/'batches'/f'meta_{args.start:08d}.json')['events']
    events=[dict(e,horizon=r['horizon']) for e,r in zip(events,recorded,strict=True)]
    if json.loads(json.dumps(events))!=recorded:raise ValueError('behavior/FＭ/task stream changed during fixed-theta batch')
    paths={e['task']:read_json(condition_root(args.root,e['task'])/'path.json') for e in events}
    baselines={int(k):v for k,v in saved['baselines'].items()}
    for task,path in paths.items():
        if not path.get('query_complete') or path['meta_update']!=args.start:raise ValueError('one whole on-policy path and independent query required')
        path['baseline_old']=baselines[task]
    decoder=load_basis(BASIS).to(context.device).requires_grad_(False);units=saved['code_units'].to(context.device)
    initialize_deferred_process_group(context,rendezvous_root=args.root/'rendezvous');queries=QueryBank(runtime)
    # Useful updates change only physical packing, never logical112 or path weights.
    micro=(7,14,28,56,112)[min(args.start,4)];pack=min(4,1+args.start)
    chunk=(16,32,64,128,256,512)[min(args.start,5)];physical=dict(saved['physical'],microbatch=micro,task_pack=pack,event_chunk=chunk)
    tick=time.monotonic();torch.cuda.reset_peak_memory_stats()
    try:
        fm,pg,result=combined_gradients(runtime,model,decoder,units,queries,events,paths,context,physical,optimizer)
        gradients=apply_update(model,optimizer,scheduler,fm,pg,saved['beta']);beta=gradients['beta']
        torch.cuda.synchronize(context.device)
        seconds=reduction(context,time.monotonic()-tick,dist.ReduceOp.MAX)
        totals=reduction(context,[result['FM'],result['PG'],result['actual_proposals'],result['query_rows']])
        if totals[3]!=112 or int(totals[2])!=sum(len(p['proposals']) for p in paths.values()):
            raise ValueError('global FM weights or complete actual proposal score coverage changed')
        for task,path in paths.items():baselines[task]=.9*baselines[task]+.1*path['R']
        measure=dict(meta_update=args.stop,seconds=seconds,FM=totals[0],PG=totals[1],actual_proposals=int(totals[2]),
            FM_queries=112,**gradients,world_size=context.world_size,**{k:result[k] for k in ['microbatch','task_pack','event_chunk','OOM_attempts']},
            peak_allocated_MiB=reduction(context,torch.cuda.max_memory_allocated()/2**20,dist.ReduceOp.MAX),
            peak_reserved_MiB=reduction(context,torch.cuda.max_memory_reserved()/2**20,dist.ReduceOp.MAX))
        physical.update({k:measure[k] for k in ['microbatch','task_pack','event_chunk']});physical['measurements']=[*saved['physical']['measurements'],measure]
        save(args.root,model,optimizer,scheduler,stream,context,units,baselines,beta,physical,saved['training_seconds']+seconds)
        folder=args.root/'training';folder.mkdir(exist_ok=True)
        write_json_atomic(folder/f'meta_{args.stop:08d}_rank_{context.rank}.json',dict(result,measure=measure,migration=migration,
            resources=host_resources(),tasks=[e['task'] for e in events],returns={t:p['R'] for t,p in paths.items()}))
        if context.is_main:
            write_json_atomic(folder/f'meta_{args.stop:08d}.json',dict(complete=True,**measure))
            print('EMBER_USEFUL_OBSERVATION '+json.dumps(measure),flush=True)
    finally:
        queries.close();runtime.videos.close();runtime.execution.close()
        if dist.is_initialized():dist.destroy_process_group()
