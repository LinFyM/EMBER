"""One fresh 144-update trajectory; useful physical comparisons, complete resumes."""
from pathlib import Path
import json,random,time
import numpy as np
import torch
import torch.distributed as dist
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.pi05_source_setup import initialize_distributed,initialize_deferred_process_group
from ember.reference.training import Runtime
from ember.proposal_writer.learning import synchronize_gradients
from .contract import SCHEMA,OPTIMIZER,SAVES
from .data import QueryBank,logical_update
from .model import ControlSpace
from .learning import References,local_credit,response_metrics
from .checkpoint import save,restore,gather


def emit(**values):
    print('EMBER_WORK_EVENT '+json.dumps(values),flush=True)


def measure_credit(runtime,model,optimizer,references,queries,groups,context,active,microbatch):
    local=groups[len(groups)*context.rank//active:len(groups)*(context.rank+1)//active] if context.rank<active else []
    attempts=[];counter={};started=time.monotonic()
    while True:
        optimizer.zero_grad(set_to_none=True);failed=None;responses=[];addresses=[]
        try:
            loss,responses,addresses=local_credit(runtime,model,references,queries,local,microbatch,counter)
        except torch.cuda.OutOfMemoryError:
            failed='OOM';optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        except Exception as error:failed=repr(error)
        statuses=gather(failed,context)
        if any(e and e!='OOM' for e in statuses):raise RuntimeError(f'functional credit failed: {statuses}')
        if not any(statuses):break
        attempts.append(dict(microbatch=microbatch,rank_failures=statuses))
        optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        if microbatch==1:raise RuntimeError('native credit still OOM at one physical query')
        microbatch=max(1,microbatch//2)
    synchronize_gradients(model,context)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    gradients={name:float(sum(p.grad.float().square().sum() for p in module.parameters() if p.grad is not None).sqrt())
               for name,module in [('encoder',model.encoder),('decoder_head',model.decoder.output)]}
    optimizer.step();torch.cuda.synchronize(context.device)
    payload=dict(responses=responses,queries=addresses,counters=counter,OOM=attempts,
        microbatch=microbatch,rank=context.rank,seconds=time.monotonic()-started,
        peak_allocated_MiB=torch.cuda.max_memory_allocated(context.device)/2**20,
        peak_reserved_MiB=torch.cuda.max_memory_reserved(context.device)/2**20)
    measures=gather(payload,context)
    rows=[r for m in measures for r in m.pop('responses')]
    metrics=response_metrics(rows)
    return dict(**metrics,rank_measures=measures,gradient_norm=float(norm),gradients=gradients,
        active_ranks=active,world_size=context.world_size,seconds=max(m['seconds'] for m in measures),
        minimum_microbatch=min(m['microbatch'] for m in measures))


def physical_config(update,context,physical):
    if update<6:
        ranks,batch=((6,7),(4,14),(2,28))[update%3]
        return min(context.world_size,ranks),batch
    # All launched ranks do useful work until the registered complete boundary.
    return context.world_size,physical.get('microbatch',14)


def select_physical(physical,context):
    rows=physical['profiles'];configs={}
    for row in rows:configs.setdefault((row['active_ranks'],row['minimum_microbatch']),[]).append(row['seconds'])
    ranks,batch=min(configs,key=lambda c:float(np.median(configs[c])))
    physical.update(recommended_world_size=ranks,microbatch=batch,selection='median actual useful update wall; no loss/checkpoint selection',
        comparisons=[dict(active_ranks=r,microbatch=b,median_seconds=float(np.median(v))) for (r,b),v in configs.items()])


def train(args):
    context=initialize_distributed(require_numa=True,defer_process_group=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    runtime=Runtime(args.root,context.device)
    torch.manual_seed(7);np.random.seed(7);random.seed(7)
    model=ControlSpace(runtime.lora,runtime.identity,read_json(args.root/'units.json')).to(context.device)
    optimizer=torch.optim.AdamW(model.parameters(),**OPTIMIZER)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    initialize_deferred_process_group(context,rendezvous_root=args.root/'ipc')
    bank=read_json(args.root/'bank.json');physical=dict(profiles=[],microbatch=7,recommended_world_size=context.world_size)
    if args.start:physical=restore(args.root,args.start,model,optimizer,scheduler,context)
    metrics_path=args.root/'training_metrics.jsonl'
    if context.is_main and metrics_path.exists():
        last=json.loads(metrics_path.read_text().splitlines()[-1])
        if last['update']!=args.start:raise RuntimeError('unsaved optimizer work cannot be silently replayed')
    references=References(runtime,model.layout.names);queries=QueryBank(runtime)
    try:
        for update in range(args.start,args.stop):
            torch.cuda.reset_peak_memory_stats(context.device)
            groups=logical_update(bank,update);active,batch=physical_config(update,context,physical)
            measures=measure_credit(runtime,model,optimizer,references,queries,groups,context,active,batch)
            scheduler.step();measures.update(update=update+1,groups=groups,logical_queries=28,logical_responses=56)
            if update<6:physical['profiles'].append({k:measures[k] for k in ('active_ranks','minimum_microbatch','seconds','rank_measures')})
            if update==5:
                select_physical(physical,context)
                if context.is_main:
                    write_json_atomic(args.root/'physical_execution.json',physical)
                    emit(kind='throughput_selected',physical=physical,remaining_updates=138,
                        remaining_seconds_estimate=138*min(c['median_seconds'] for c in physical['comparisons']))
            if context.is_main:
                with metrics_path.open('a') as f:f.write(json.dumps(measures)+'\n')
                if update<6:emit(kind='useful_training_update',update=update+1,seconds=measures['seconds'],
                    MSE=measures['MSE'],lower_bound=measures['ignored_code_MSE_lower_bound'],gradients=measures['gradients'])
            if update+1 in SAVES:save(args.root,update+1,model,optimizer,scheduler,context,physical)
        if context.is_main:write_json_atomic(args.root/f'training_exit_{args.stop:08d}.json',dict(
            complete=True,start=args.start,stop=args.stop,physical=physical,world_size=context.world_size,schema=SCHEMA))
    finally:
        queries.close();runtime.execution.close()
        if context.world_size>1:dist.destroy_process_group()
