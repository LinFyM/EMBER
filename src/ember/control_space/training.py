"""Fresh image Reader learning with exact fixed-decoder FM credit and full resume."""
from pathlib import Path
import datetime,json,time,random
import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from ember.pi05_source_setup import initialize_distributed,initialize_deferred_process_group
from ember.pi05_source_checkpoint import write_json_atomic,read_json,barrier
from ember.proposal_writer.learning import synchronize_gradients
from ember.reference.accounting import host_resources
from .contract import BASIS,OPTIMIZER,SCHEMA
from .data import make_runtime,Stream
from .model import load_basis
from .reader import ImageReader
from .learning import QueryBank,update_gradients
from .checkpoint import save,restore


def label_codes(root):
    destination=Path(root)/'code_labels.pt'
    if destination.exists():return torch.load(destination,map_location='cpu',weights_only=False)
    decoder=load_basis(BASIS);saved=torch.load(BASIS,map_location='cpu',weights_only=False,mmap=True)
    raw=torch.tensor(saved['spectrum'][:64],dtype=torch.float32);units=raw.clamp_min(1e-6)
    bank=read_json(Path(root)/'bank.json')['tasks'];labels={}
    center=decoder.layout.unpack(decoder.layout.center)
    with torch.no_grad():
        for row in bank:
            state=center if row['selected_path'] is None else load_file(row['selected_path'])
            labels[row['task']]=decoder.encode(state)/units
    result=dict(schema=SCHEMA,frozen_basis=str(BASIS),code_units=units,floor_triggered=int((raw<1e-6).sum()),
        producer_y=labels,producer_paths={r['task']:r['selected_path'] for r in bank})
    temporary=destination.with_suffix('.partial');torch.save(result,temporary);temporary.replace(destination)
    return result


def initialize(args,context):
    runtime=make_runtime(args.root,context.device);torch.manual_seed(7);np.random.seed(7);random.seed(7)
    reader=ImageReader().to(context.device);decoder=load_basis(BASIS).to(context.device).requires_grad_(False)
    labels=label_codes(args.root);units=labels['code_units'].to(context.device)
    labels={int(k):v.to(context.device) for k,v in labels['producer_y'].items()}
    optimizer=torch.optim.AdamW(reader.parameters(),**OPTIMIZER);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    stream=Stream(args.root);elapsed=0.;migration=dict(fresh=True,seed=7)
    physical=dict(microbatch=7,task_pack=1,measurements=[])
    if args.start:
        state,migration=restore(args.root,args.start,reader,optimizer,scheduler,context)
        stream=Stream(args.root,state['stream']);elapsed=state['training_seconds'];physical=state['physical']
        if stream.next_update!=args.start or sum(stream.visits.values())!=4*args.start:raise ValueError('Reader logical stream lost its saved cursor')
    physical['profiling_new_topology']=not args.start or migration['from_topology']['world_size']!=context.world_size
    if physical['profiling_new_topology']:physical['measurements']=[]
    return runtime,reader,decoder,labels,units,optimizer,scheduler,stream,elapsed,physical,migration


def reduction(context,value,op=dist.ReduceOp.SUM):
    tensor=torch.as_tensor(value,device=context.device,dtype=torch.float64)
    if context.world_size>1:dist.all_reduce(tensor,op=op)
    return tensor.cpu().tolist()


def choice(index,physical):
    # First useful updates exercise7/14/28, then available multitask/query packs.
    candidates=((7,1),(14,1),(28,1),(28,2),(56,2),(56,4),(112,4))
    if index<len(candidates):return candidates[index]
    finite=[m for m in physical['measurements'] if not m['OOM_attempts']]
    if not finite:return physical['microbatch'],physical['task_pack']
    best=min(finite,key=lambda m:m['seconds'])
    return best['microbatch'],best['task_pack']


def gradients(runtime,reader,decoder,units,labels,queries,events,context,microbatch,pack,optimizer):
    failures=[]
    while True:
        optimizer.zero_grad(set_to_none=True);error=None;result=None
        try:result=update_gradients(runtime,reader,decoder,units,labels,queries,events,context,microbatch,pack)
        except torch.cuda.OutOfMemoryError as failure:error=repr(failure)
        failed=reduction(context,int(error is not None),dist.ReduceOp.MAX)
        if not failed:return result,microbatch,pack,failures
        failures.append(dict(microbatch=microbatch,task_pack=pack,local_error=error))
        optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        if microbatch>7:microbatch=max(7,microbatch//2)
        elif pack>1:pack=max(1,pack//2)
        else:raise RuntimeError('registered useful FM workload OOM at minimum physical7/one task')


def append(root,name,event):
    path=Path(root)/'training'/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:handle.write(json.dumps(event)+'\n')


def train(args):
    if not 0<=args.start<args.stop<=1800 or args.start%9 or args.stop%9:
        raise ValueError('training segment must preserve complete36-task cycles within1800')
    context=initialize_distributed(require_numa=True,defer_process_group=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    initialized=initialize(args,context)
    runtime,reader,decoder,labels,units,optimizer,scheduler,stream,elapsed,physical,migration=initialized
    initialize_deferred_process_group(context,rendezvous_root=args.root/'rendezvous')
    queries=QueryBank(runtime);measurements=[];started=time.monotonic()
    append(args.root,f'topology_rank_{context.rank}.jsonl',migration)
    try:
        for update in range(args.start,args.stop):
            tick=time.monotonic();events=stream.next();micro,pack=choice(update-args.start if physical['profiling_new_topology'] else 99,physical)
            torch.cuda.reset_peak_memory_stats()
            result,micro,pack,ooms=gradients(runtime,reader,decoder,units,labels,queries,events,context,micro,pack,optimizer)
            synchronize_gradients(reader,context)
            norm=float(torch.nn.utils.clip_grad_norm_(reader.parameters(),1.,error_if_nonfinite=True))
            optimizer.step();scheduler.step();torch.cuda.synchronize(context.device)
            seconds=float(reduction(context,time.monotonic()-tick,dist.ReduceOp.MAX));elapsed+=seconds
            peak=reduction(context,[torch.cuda.max_memory_allocated()/2**20,torch.cuda.max_memory_reserved()/2**20,max(result['physical_FM_batches'])],dist.ReduceOp.MAX)
            total=reduction(context,[result['FM'],result['weighted_auxiliary'],result['query_rows'],result['auxiliary_rows']])
            if total[2:]!=[112.,4.]:raise ValueError('physical sharding changed logical FM112/aux4')
            measure=dict(update=update+1,microbatch=micro,task_pack=pack,seconds=seconds,queries_per_second=112/seconds,
                peak_allocated_MiB=peak[0],peak_reserved_MiB=peak[1],actual_max_FM_batch=peak[2],OOM_attempts=ooms,world_size=context.world_size)
            physical.update(microbatch=micro,task_pack=pack)
            if physical['profiling_new_topology'] and update-args.start<7:physical['measurements'].append(measure)
            measurements.append(measure)
            event=dict(result,**measure,gradient_norm=norm,task_visits=[dict(task=e['task'],visit=e['visit'],
                teacher=e['original']['demo'],auxiliary=e['auxiliary']['key'],demos=e['demos'],frames=e['frames']) for e in events])
            append(args.root,f'rank_{context.rank}.jsonl',event)
            if context.is_main:
                append(args.root,'updates.jsonl',dict(measure,FM=total[0],weighted_auxiliary=total[1],gradient_norm=norm))
                if update<7:print('EMBER_WORK_EVENT '+json.dumps(dict(kind='training_observation',**measure,FM=total[0],weighted_auxiliary=total[1],
                    FM_Reader_gradient_norm=result['FM_Reader_gradient_norm'],head_gradient_norm=result['head_gradient_norm'],
                    body_gradient_norm=result['body_gradient_norm'])),flush=True)
            if (update+1)%90==0 or update+1==args.stop:
                save(args.root,reader,optimizer,scheduler,stream,context,units,physical,training_seconds=elapsed)
        result=dict(complete=True,start=args.start,stop=args.stop,world_size=context.world_size,measurements=measurements,
            cumulative_training_seconds=elapsed,job_wall_seconds=time.monotonic()-started,resources=host_resources(),
            end_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if context.is_main:
            write_json_atomic(args.root/'training'/f'segment_{args.start:08d}_{args.stop:08d}.json',result)
            print('EMBER_TRAIN_SEGMENT '+json.dumps({k:result[k] for k in ('start','stop','world_size','job_wall_seconds')}),flush=True)
        return result
    finally:
        queries.close();runtime.videos.close();runtime.execution.close()
        if dist.is_initialized():dist.destroy_process_group()
