"""One fresh H-revision trajectory with exact native FM and full recovery."""
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
from .contract import BASIS,INITIAL_READER,OPTIMIZER
from .data import Stream
from .model import load_basis
from .reader import ImageReader
from .feedback import RevisionReader
from .features import FrozenFeatures
from .learning import QueryBank,fm_gradients,gradient_norm
from .checkpoint import save,restore


def frozen_reader(device):
    saved=torch.load(INITIAL_READER,map_location='cpu',weights_only=False,mmap=True)
    if saved['phase']!='video_initial' or saved['next_update']!=900 or saved['frozen_basis']!=str(BASIS):
        raise ValueError('only the original frozen Reader900 and its saved units may condition revision')
    reader=ImageReader();reader.load_state_dict(saved['reader'],strict=True)
    units=saved['code_units'].float().to(device)
    if units.shape!=(64,) or not bool(units.isfinite().all()) or bool((units<=0).any()):raise ValueError('invalid original code units')
    return reader.to(device).requires_grad_(False).eval(),units


def initialize(args,context):
    runtime=Runtime(args.root,context.device);video,units=frozen_reader(context.device)
    torch.manual_seed(7);np.random.seed(7);random.seed(7)
    model=RevisionReader().to(context.device);decoder=load_basis(BASIS).to(context.device).requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),**OPTIMIZER);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    stream=Stream(args.root);elapsed=0.;migration=dict(fresh=True,seed=7)
    physical=dict(microbatch=7,task_pack=1,event_chunk=16,image_chunk=16,measurements=[])
    if args.start:
        state,migration=restore(args.root,args.start,model,optimizer,scheduler,context)
        stream=Stream(args.root,state['stream']);elapsed=state['training_seconds'];physical=state['physical']
        if stream.next_update!=args.start:raise ValueError('revision update stream lost its complete checkpoint')
    physical['profiling_new_topology']=not args.start or migration['from_topology']['world_size']!=context.world_size
    if physical['profiling_new_topology']:physical['measurements']=[]
    features=FrozenFeatures(args.root,runtime,video)
    return runtime,model,decoder,units,optimizer,scheduler,stream,elapsed,physical,migration,features


def reduction(context,value,op=dist.ReduceOp.SUM):
    tensor=torch.as_tensor(value,device=context.device,dtype=torch.float64)
    if context.world_size>1:dist.all_reduce(tensor,op=op)
    return tensor.cpu().tolist()


def choice(index,physical):
    candidates=((7,1,16,16),(14,1,64,32),(28,1,128,64),(28,2,256,64),(56,2,512,128),(112,4,512,128))
    if index<len(candidates):return candidates[index]
    finite=[m for m in physical['measurements'] if not m['OOM_attempts']]
    if not finite:return tuple(physical[k] for k in ('microbatch','task_pack','event_chunk','image_chunk'))
    best=min(finite,key=lambda m:m['seconds'])
    return tuple(best[k] for k in ('microbatch','task_pack','event_chunk','image_chunk'))


def gradients(runtime,model,decoder,units,queries,features,events,context,physical,optimizer):
    failures=[];micro,pack,event_chunk,image_chunk=physical
    while True:
        optimizer.zero_grad(set_to_none=True);error=None;result=None
        try:result=fm_gradients(runtime,model,decoder,units,queries,features,events,context,micro,pack,event_chunk,image_chunk)
        except torch.cuda.OutOfMemoryError as failure:error=repr(failure)
        failed=reduction(context,int(error is not None),dist.ReduceOp.MAX)
        if not failed:return result,(micro,pack,event_chunk,image_chunk),failures
        failures.append(dict(microbatch=micro,task_pack=pack,event_chunk=event_chunk,image_chunk=image_chunk,local_error=error))
        optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        if event_chunk>16:event_chunk=max(16,event_chunk//2)
        elif micro>7:micro=max(7,micro//2)
        elif pack>1:pack=max(1,pack//2)
        elif image_chunk>8:image_chunk=max(8,image_chunk//2)
        else:raise RuntimeError('registered complete-H FM workload OOM at minimum physical7/event16/image8')


def append(root,name,event):
    path=Path(root)/'training'/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:handle.write(json.dumps(event)+'\n')


def train(args):
    if not 0<=args.start<args.stop<=180 or args.start%9 or args.stop%9:
        raise ValueError('revision segments preserve complete36-task cycles inside180 updates')
    updates=args.root/'training/updates.jsonl'
    if updates.is_file():
        with updates.open('rb') as handle:
            handle.seek(max(0,updates.stat().st_size-16384));last=handle.read().splitlines()[-1]
        if json.loads(last)['update']!=args.start:raise RuntimeError('unsaved effective revision updates cannot be silently replayed')
    context=initialize_distributed(require_numa=True,defer_process_group=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    runtime,model,decoder,units,optimizer,scheduler,stream,elapsed,physical,migration,features=initialize(args,context)
    initialize_deferred_process_group(context,rendezvous_root=args.root/'rendezvous')
    queries=QueryBank(runtime);measurements=[];started=time.monotonic()
    append(args.root,f'topology_rank_{context.rank}.jsonl',migration)
    try:
        for update in range(args.start,args.stop):
            tick=time.monotonic();events=stream.next();requested=choice(update-args.start if physical['profiling_new_topology'] else 99,physical)
            torch.cuda.reset_peak_memory_stats();features.measurements=[]
            result,chosen,ooms=gradients(runtime,model,decoder,units,queries,features,events,context,requested,optimizer)
            synchronize_gradients(model,context)
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
            optimizer.step();scheduler.step();torch.cuda.synchronize(context.device)
            seconds=float(reduction(context,time.monotonic()-tick,dist.ReduceOp.MAX));elapsed+=seconds
            peak=reduction(context,[torch.cuda.max_memory_allocated()/2**20,torch.cuda.max_memory_reserved()/2**20,max(result['physical_FM_batches'])],dist.ReduceOp.MAX)
            total=reduction(context,[result['FM'],result['query_rows']])
            if total[1]!=112.:raise ValueError('physical sharding changed the equal-weight logical112 FM queries')
            measure=dict(update=update+1,microbatch=chosen[0],task_pack=chosen[1],event_chunk=chosen[2],image_chunk=chosen[3],seconds=seconds,
                queries_per_second=112/seconds,peak_allocated_MiB=peak[0],peak_reserved_MiB=peak[1],actual_max_FM_batch=peak[2],
                OOM_attempts=ooms,world_size=context.world_size)
            physical.update(dict(zip(('microbatch','task_pack','event_chunk','image_chunk'),chosen,strict=True)))
            if physical['profiling_new_topology'] and update-args.start<6:physical['measurements'].append(measure)
            measurements.append(measure)
            event=dict(result,**measure,gradient_norm=norm,feature_cache=features.measurements,
                task_visits=[dict(task=e['task'],visit=e['visit'],new_visit=e['new_visit'],teacher=e['original']['demo'],
                    prefix=e['prefix'],demos=e['demos'],frames=e['frames'],query_cursor_before=e['query_cursor_before'],query_cursor_after=e['query_cursor_after']) for e in events])
            append(args.root,f'rank_{context.rank}.jsonl',event)
            if context.is_main:
                append(args.root,'updates.jsonl',dict(measure,FM=total[0],gradient_norm=norm))
                if update-args.start<6:print('EMBER_WORK_EVENT '+json.dumps(dict(kind='training_observation',**measure,FM=total[0],
                    LoRA_cotangent_norm=result['LoRA_cotangent_norm'],head_gradient_norm=result['head_gradient_norm'],body_gradient_norm=result['body_gradient_norm'],
                    H_conditions=result['conditions'],H_forward_seconds=result['H_forward_seconds'],native_and_revision_backward_seconds=result['native_and_revision_backward_seconds'])),flush=True)
            if (update+1)%9==0:save(args.root,model,optimizer,scheduler,stream,context,units,physical,training_seconds=elapsed)
        result=dict(complete=True,start=args.start,stop=args.stop,world_size=context.world_size,measurements=measurements,
            cumulative_training_seconds=elapsed,job_wall_seconds=time.monotonic()-started,resources=host_resources(),
            end_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if context.is_main:
            write_json_atomic(args.root/'training'/f'segment_{args.start:08d}_{args.stop:08d}.json',result)
            print('EMBER_TRAIN_SEGMENT '+json.dumps({k:result[k] for k in ('start','stop','world_size','job_wall_seconds')}),flush=True)
        return result
    finally:
        queries.close();runtime.execution.close()
        if dist.is_initialized():dist.destroy_process_group()
