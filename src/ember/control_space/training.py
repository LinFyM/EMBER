"""One fresh180 trajectory, shared native credit, and a bounded disposable consumer."""
from pathlib import Path
import datetime,json,time,random
import numpy as np
import torch
import torch.distributed as dist
from ember.pi05_source_setup import initialize_distributed,initialize_deferred_process_group
from ember.pi05_source_checkpoint import write_json_atomic
from ember.proposal_writer.learning import synchronize_gradients
from ember.reference.training import Runtime
from ember.reference.accounting import host_resources
from .contract import OPTIMIZER,ROOT,FINAL
from .data import Stream
from .learning import Compiler,distributed_credit
from .checkpoint import save,restore


def append(root,name,value):
    path=Path(root)/'training'/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as h:h.write(json.dumps(value)+'\n')


def reduction(context,value,op=dist.ReduceOp.SUM):
    tensor=torch.as_tensor(value,device=context.device,dtype=torch.float64)
    if context.world_size>1:dist.all_reduce(tensor,op=op)
    return tensor.cpu().tolist()


def norm(parameters):
    values=[p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def initialize(args,context):
    from .model import NativeOperatorContext
    runtime=Runtime(args.root,context.device)
    random.seed(7);np.random.seed(7);torch.manual_seed(7)
    model=NativeOperatorContext().to(context.device)
    optimizer=torch.optim.AdamW(model.parameters(),**OPTIMIZER)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    stream=Stream(runtime);elapsed=0.;migration=dict(fresh=True,seed=7)
    if args.start:
        state,migration=restore(args.root,args.start,model,optimizer,scheduler,context,disposable=args.command=='profile')
        stream.load_state_dict(state['stream']);elapsed=state['training_seconds']
        if args.command=='profile' and state['profile_cursor']!=args.start:
            raise ValueError('disposable consumer cursor was not preserved')
    compiler=Compiler(runtime,ROOT,support_microbatch=args.support_microbatch)
    if args.start and context.rank<len(state['rank_RNG']):
        compiler.F0={int(t):v for t,v in state['rank_RNG'][context.rank].get('frozen_source_F0',{}).items()}
    return runtime,model,optimizer,scheduler,stream,compiler,elapsed,migration


def one_update(args,context,model,optimizer,scheduler,events,compiler,stream,*,consumer=False):
    optimizer.zero_grad(set_to_none=True);tick=time.monotonic();torch.cuda.reset_peak_memory_stats()
    value=distributed_credit(model,compiler,stream,events,context,microbatch=args.microbatch,consumer=consumer)
    synchronize_gradients(model,context)
    full=norm(model.parameters());head=norm(p for n,p in model.named_parameters() if n.startswith('pressure.'))
    body=norm(p for n,p in model.named_parameters() if not n.startswith('pressure.'))
    video=norm(p for n,p in model.named_parameters() if n.startswith('video_attention.'))
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step();scheduler.step();torch.cuda.synchronize(context.device)
    seconds=reduction(context,time.monotonic()-tick,dist.ReduceOp.MAX)
    peak=reduction(context,[torch.cuda.max_memory_allocated()/2**20,torch.cuda.max_memory_reserved()/2**20],dist.ReduceOp.MAX)
    total=reduction(context,[value['FM'],value['query_rows'],value['conditions']])
    if total[1:]!=[112.,4.]:raise ValueError('physical distribution changed4 task/112-query global mass')
    return dict(FM=total[0],queries=112,conditions=4,seconds=seconds,compilers_per_second=4/seconds,
        queries_per_second=112/seconds,world_size=context.world_size,support_microbatch=args.support_microbatch,
        FM_microbatch=args.microbatch,peak_allocated_MiB=peak[0],peak_reserved_MiB=peak[1],
        gradient_norm=full,pressure_gradient_norm=head,context_gradient_norm=body,video_attention_gradient_norm=video,
        local_query_rows=value['query_rows'],local_compilations=value['compilations'],
        local_actual_F_out_consumer=value['consumer'],physical_FM_batches=value['physical_FM_batches'],
        memory=compiler.memory.drain_measurements())


def train(args):
    disposable=args.command=='profile';limit=4 if disposable else FINAL
    if not 0<=args.start<args.stop<=limit:raise ValueError('unregistered learning/profile exposure')
    updates=args.root/'training/updates.jsonl'
    if updates.exists():
        with updates.open('rb') as h:h.seek(max(0,updates.stat().st_size-16384));last=json.loads(h.read().splitlines()[-1])
        if last['update']!=args.start:raise RuntimeError('unsaved effective updates cannot be silently repeated')
    context=initialize_distributed(require_numa=True,defer_process_group=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    runtime,model,optimizer,scheduler,stream,compiler,elapsed,migration=initialize(args,context)
    initialize_deferred_process_group(context,rendezvous_root=args.root/'rendezvous')
    append(args.root,f'topology_rank_{context.rank}.jsonl',migration);began=time.monotonic()
    try:
        for update in range(args.start,args.stop):
            if disposable:
                args.support_microbatch=(4,8,16,16)[update]
                args.microbatch=(7,14,28,28)[update]
                compiler.support_microbatch=args.support_microbatch
            events=([stream.event_for(task,update) for task in (0,13,20,32)] if disposable else stream.next(update))
            report=one_update(args,context,model,optimizer,scheduler,events,compiler,stream,
                              consumer=disposable and update==0)
            elapsed+=report['seconds'];report.update(update=update+1,logical_events=events,disposable=disposable)
            append(args.root,f'rank_{context.rank}.jsonl',report)
            if context.is_main:
                append(args.root,'updates.jsonl',{k:v for k,v in report.items() if k not in ('memory','local_compilations','local_actual_F_out_consumer')})
                if update==args.start:
                    print('EMBER_WORK_EVENT '+json.dumps(dict(kind='training_observation',**{k:v for k,v in report.items()
                        if k not in ('memory','logical_events','local_compilations','local_actual_F_out_consumer')})),flush=True)
            if (update+1)%9==0 or update+1 in (90,180) or update+1==args.stop:
                save(args.root,model,optimizer,scheduler,stream,context,
                    dict(support_microbatch=args.support_microbatch,FM_microbatch=args.microbatch),
                    update=update+1,training_seconds=elapsed,disposable=disposable,frozen_response_cache=compiler.F0)
        result=dict(complete=True,start=args.start,stop=args.stop,disposable=disposable,
            logical_FM_queries=(args.stop-args.start)*112,world_size=context.world_size,
            training_seconds=elapsed,job_wall_seconds=time.monotonic()-began,
            resources=host_resources(),end_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if context.is_main:write_json_atomic(args.root/'training'/f'segment_{args.start:08d}_{args.stop:08d}.json',result)
        return result
    finally:
        compiler.close();stream.close();runtime.execution.close()
        if dist.is_initialized():dist.destroy_process_group()
