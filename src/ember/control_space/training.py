"""Equal36 operator learning, then unchanged-Adam native full-factor FM/BA."""
from pathlib import Path
from types import SimpleNamespace
import datetime,json,time,random
import numpy as np
import torch
import torch.distributed as dist
from ember.pi05_source_setup import initialize_distributed,initialize_deferred_process_group
from ember.pi05_source_checkpoint import write_json_atomic,read_json
from ember.proposal_writer.learning import synchronize_gradients
from ember.reference.training import Runtime
from ember.reference.accounting import host_resources
from ember.writer.runtime import autocast
from .contract import OPTIMIZER,TASKS,OPERATOR_FILE
from .data import Stream
from .weights import load_source_inputs
from .model import SourceWeightCompiler
from .video import MemoryStore
from .learning import References,OperatorLoss,QueryBank,assignments,fm_credit
from .checkpoint import save,restore


def append(root,name,value):
    path=Path(root)/'training'/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:handle.write(json.dumps(value)+'\n')


def reduction(context,value,op=dist.ReduceOp.SUM):
    tensor=torch.as_tensor(value,device=context.device,dtype=torch.float64)
    if context.world_size>1:dist.all_reduce(tensor,op=op)
    return tensor.cpu().tolist()


def norm(parameters):
    values=[p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def initialize(args,context):
    torch.manual_seed(7);np.random.seed(7);random.seed(7)
    inputs=load_source_inputs(args.root);model=SourceWeightCompiler(inputs).to(context.device)
    identity={k:v.to(context.device) for k,v in inputs['identity'].items()};del inputs
    optimizer=torch.optim.AdamW(model.parameters(),**OPTIMIZER);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    stream=Stream();elapsed=0.;migration=dict(fresh=True,seed=7)
    if args.start:
        state,migration=restore(args.root,args.start,model,optimizer,scheduler,context)
        stream=Stream(state['stream']);elapsed=state['training_seconds']
    constants=read_json(OPERATOR_FILE)['denominators'];objective=OperatorLoss(model.lora,constants)
    references=References(SimpleNamespace(device=context.device,identity=identity),identity,capacity=36)
    paths={row['task']:row['path'] for row in read_json(args.root/'run_contract.json')['references']}
    memory=MemoryStore(args.root,context.device)
    return model,optimizer,scheduler,stream,elapsed,migration,objective,references,paths,memory


def gradients(model,events,context,physical,memory,objective,references,paths,runtime,queries):
    pack,row_chunk,micro,checkpointed=physical;result=dict(BA=0.,L_A=0.,L_B=0.,L_pair=0.,FM=0.,conditions=0,query_rows=0,LoRA_cotangent_norm=0.,physical_FM_batches=[])
    owned=[(i,None) for i in range(context.rank,len(events),context.world_size)] if len(events)==36 else assignments(context.world_size,context.rank)
    for first in range(0,len(owned),pack):
        group=owned[first:first+pack];selected=[events[i] for i,_ in group]
        memories=[memory.training(e['task'],e['teacher_demo']).to(context.device) for e in selected]
        with autocast(context.device):states=model(memories,row_chunk=row_chunk,checkpointed=checkpointed)
        teachers=[references.get(paths[e['task']]) for e in selected]
        if len(events)==36:
            errors_A,errors_B=objective.pair(states,teachers)
            loss_A=errors_A.mean(-1).sum()/36;loss_B=errors_B.mean(-1).sum()/36
            loss=(loss_A+loss_B)/2
            with torch.no_grad():errors,_=objective(states,teachers);BA=errors.mean(-1).sum()/36
            loss.backward()
            result['L_A']+=float(loss_A.detach());result['L_B']+=float(loss_B.detach())
            result['L_pair']+=float(loss.detach());result['BA']+=float(BA)
        else:
            samples=[queries.sample(e,indices)[0] for e,(_,indices) in zip(selected,group,strict=True)]
            credit=fm_credit(runtime,samples,states,teachers,objective,micro)
            for key in ('BA','FM','query_rows'):result[key]+=credit[key]
            result['LoRA_cotangent_norm']+=credit['LoRA_cotangent_norm'];result['physical_FM_batches']+=credit['physical_FM_batches']
        result['conditions']+=len(group)
    return result


def backward(model,events,context,physical,memory,objective,references,paths,runtime,queries,optimizer,*,retry=True):
    pack,row_chunk,micro,checkpointed=physical;failures=[]
    while True:
        optimizer.zero_grad(set_to_none=True);error=None;value=None
        try:value=gradients(model,events,context,(pack,row_chunk,micro,checkpointed),memory,objective,references,paths,runtime,queries)
        except torch.cuda.OutOfMemoryError as failure:error=repr(failure)
        if not reduction(context,int(error is not None),dist.ReduceOp.MAX):return value,(pack,row_chunk,micro,checkpointed),failures
        failures.append(dict(task_pack=pack,row_chunk=row_chunk,microbatch=micro,checkpointed=checkpointed,local_error=error))
        optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        if not retry:raise torch.cuda.OutOfMemoryError(json.dumps(failures))
        if not checkpointed:checkpointed=True
        elif pack>1:pack=max(1,pack//2)
        elif micro>7:micro=max(7,micro//2)
        elif row_chunk>128:row_chunk=max(128,row_chunk//2)
        else:raise RuntimeError('source-weight workload OOM at minimum physical pack1/micro7/row128')


def physical(args):return (args.task_pack,args.row_chunk,args.microbatch,not args.no_checkpoint)


def one_update(args,context,model,optimizer,scheduler,events,config,memory,objective,refs,paths,runtime,queries):
    tick=time.monotonic();torch.cuda.reset_peak_memory_stats();memory.drain_measurements()
    value,chosen,ooms=backward(model,events,context,config,memory,objective,refs,paths,runtime,queries,optimizer,retry=args.command!='profile')
    synchronize_gradients(model,context)
    before=norm(model.parameters());head_A=norm(p for n,p in model.named_parameters() if n.startswith('head_A.'))
    head_B=norm(p for n,p in model.named_parameters() if n.startswith('head_B.'))
    body=norm(p for n,p in model.named_parameters() if not n.startswith(('head_A.','head_B.')))
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();scheduler.step();torch.cuda.synchronize(context.device)
    seconds=reduction(context,time.monotonic()-tick,dist.ReduceOp.MAX)
    peak=reduction(context,[torch.cuda.max_memory_allocated()/2**20,torch.cuda.max_memory_reserved()/2**20],dist.ReduceOp.MAX)
    total=reduction(context,[value['BA'],value['FM'],value['query_rows'],value['conditions'],value['L_A'],value['L_B'],value['L_pair']])
    expected=0 if len(events)==36 else 112
    if total[2]!=expected or (len(events)==36 and total[3]!=36):raise ValueError('physical splitting changed global logical mass')
    report=dict(BA=total[0],L_A=total[4],L_B=total[5],L_pair=total[6],FM=total[1],queries=total[2],conditions=36 if expected==0 else 4,generator_condition_forwards=total[3],seconds=seconds,
        LoRAs_per_second=(36 if expected==0 else 4)/seconds,queries_per_second=expected/seconds,
        task_pack=chosen[0],row_chunk=chosen[1],microbatch=chosen[2],checkpointed=chosen[3],world_size=context.world_size,
        peak_allocated_MiB=peak[0],peak_reserved_MiB=peak[1],OOM_attempts=ooms,gradient_norm=before,
        head_A_gradient_norm=head_A,head_B_gradient_norm=head_B,body_gradient_norm=body,
        local_LoRA_cotangent_norm=value['LoRA_cotangent_norm'],physical_FM_batches=value['physical_FM_batches'],memory=memory.drain_measurements())
    return report,chosen


def train(args):
    if not 0<=args.start<args.stop<=1180 or (args.start<1000<args.stop):raise ValueError('train segments end at the saved1000 phase boundary')
    updates=args.root/'training/updates.jsonl'
    if updates.exists():
        with updates.open('rb') as handle:handle.seek(max(0,updates.stat().st_size-16384));last=json.loads(handle.read().splitlines()[-1])
        if last['update']!=args.start:raise RuntimeError('unsaved effective updates require explicit reconciliation, not silent replay')
    context=initialize_distributed(require_numa=True,defer_process_group=True);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    model,optimizer,scheduler,stream,elapsed,migration,objective,refs,paths,memory=initialize(args,context)
    runtime=Runtime(args.root,context.device) if args.start>=1000 else None;queries=QueryBank(runtime) if runtime else None
    initialize_deferred_process_group(context,rendezvous_root=args.root/'rendezvous')
    append(args.root,f'topology_rank_{context.rank}.jsonl',migration);config=physical(args);started=time.monotonic()
    try:
        for update in range(args.start,args.stop):
            events=stream.next();report,config=one_update(args,context,model,optimizer,scheduler,events,config,memory,objective,refs,paths,runtime,queries)
            elapsed+=report['seconds'];report.update(update=update+1,stage='pair' if update<1000 else 'joint',logical_events=events)
            append(args.root,f'rank_{context.rank}.jsonl',report)
            if context.is_main:
                append(args.root,'updates.jsonl',{k:v for k,v in report.items() if k!='memory'})
                if update==args.start:print('EMBER_WORK_EVENT '+json.dumps(dict(kind='training_observation',**{k:v for k,v in report.items() if k not in ('memory','logical_events')})),flush=True)
            if (update+1)%90==0 or update+1 in (1000,1090,1180):save(args.root,model,optimizer,scheduler,stream,context,dict(zip(('task_pack','row_chunk','microbatch','checkpointed'),config)),training_seconds=elapsed)
        result=dict(complete=True,start=args.start,stop=args.stop,world_size=context.world_size,training_seconds=elapsed,
            job_wall_seconds=time.monotonic()-started,resources=host_resources(),end_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if context.is_main:write_json_atomic(args.root/'training'/f'segment_{args.start:08d}_{args.stop:08d}.json',result)
        return result
    finally:
        memory.close()
        if queries:queries.close()
        if runtime:runtime.execution.close()
        if dist.is_initialized():dist.destroy_process_group()
