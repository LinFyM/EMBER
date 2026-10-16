"""Bounded disposable physical profiles, never a scientific checkpoint."""
import json,time
import torch
import torch.distributed as dist
from ember.pi05_source_setup import initialize_distributed,initialize_deferred_process_group
from ember.pi05_source_checkpoint import write_json_atomic
from ember.reference.training import Runtime
from .data import Stream
from .learning import QueryBank
from .training import initialize,one_update


def run(args):
    context=initialize_distributed(require_numa=True,defer_process_group=True);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    model,optimizer,scheduler,stream,elapsed,migration,objective,refs,paths,memory=initialize(args,context)
    initialize_deferred_process_group(context,rendezvous_root=args.root/'rendezvous');runtime=queries=None
    results=[];started=time.monotonic();largest=(36+context.world_size-1)//context.world_size
    configurations=[(1,512,7,True),(min(6,largest),4096,7,False),(largest,4096,7,False)]
    configurations=list(dict.fromkeys(configurations))
    try:
        for stage in ('BA','joint'):
            if stage=='joint':
                runtime=Runtime(args.root,context.device);queries=QueryBank(runtime)
                state=Stream().state_dict();state.update(next_update=1000,visits={t:1000 for t in state['visits']});stream=Stream(state)
                configurations=[(1,512,7,True),(4,4096,56,True)]
            for config in configurations:
                events=stream.next();entry=dict(stage=stage,requested=dict(zip(('task_pack','row_chunk','microbatch','checkpointed'),config)),
                    BA_equivalents=int(stage=='BA'),joint_equivalents=int(stage=='joint'),FM_queries=112 if stage=='joint' else 0)
                try:
                    report,chosen=one_update(args,context,model,optimizer,scheduler,events,config,memory,objective,refs,paths,runtime,queries)
                    entry.update(report,OOM=False)
                except torch.cuda.OutOfMemoryError as failure:
                    optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache();entry.update(OOM=True,error=repr(failure))
                results.append(entry)
                if context.is_main:
                    write_json_atomic(args.root/'profiles'/f'world_{context.world_size}.json',dict(complete=False,disposable=True,world_size=context.world_size,results=results))
                if context.is_main:print('EMBER_PHYSICAL_PROFILE '+json.dumps({k:v for k,v in entry.items() if k!='memory'}),flush=True)
        if not any(e['stage']=='joint' and not e['OOM'] for e in results):raise RuntimeError('no finite disposable native physical profile')
        result=dict(complete=True,disposable=True,formal_initialization_used=False,world_size=context.world_size,results=results,
            actual_BA_equivalents=sum(r['BA_equivalents'] for r in results),actual_joint_equivalents=sum(r['joint_equivalents'] for r in results),
            actual_FM_queries=sum(r['FM_queries'] for r in results),wall_seconds=time.monotonic()-started)
        if context.is_main:write_json_atomic(args.root/'profiles'/f'world_{context.world_size}.json',result)
        return result
    finally:
        memory.close()
        if queries:queries.close()
        if runtime:runtime.execution.close()
        if dist.is_initialized():dist.destroy_process_group()
