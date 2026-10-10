"""Disposable rank8 context, gradient and physical-throughput consumers."""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
import torch
from safetensors.torch import load_file, save_file
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.runtime import autocast
from ember.writer.model import DirectLoRAParameters
from ember.writer.function_credit import FlowSample, NativeFlowPrediction, mean_velocity_loss
from ember.writer.practice import processed
from .contract import PRIOR_ROOT, MT_PATH, OPTIMIZER, seed
from .teacher import QueryStream, tree_slice, function_credit, save_checkpoint, fit_teacher
from .path import load_history


def measured(runtime, consume):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(runtime.device)
    torch.cuda.synchronize(runtime.device); start=time.monotonic()
    try:
        result=consume()
        torch.cuda.synchronize(runtime.device)
        return dict(seconds=time.monotonic()-start, OOM=False,
            peak_GiB=torch.cuda.max_memory_allocated(runtime.device)/2**30,
            reserved_GiB=torch.cuda.max_memory_reserved(runtime.device)/2**30, result=result)
    except torch.cuda.OutOfMemoryError:
        return dict(seconds=time.monotonic()-start,OOM=True,
            peak_GiB=torch.cuda.max_memory_allocated(runtime.device)/2**30,
            reserved_GiB=torch.cuda.max_memory_reserved(runtime.device)/2**30)


def profile(args,runtime,runner,contract):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    panel=read_json(args.root/'panel.json')['tasks']
    _,task,demo=max((runtime.tasks[r['task_id']].episode_lengths[d],r['task_id'],d)
                   for r in panel for d in r['teacher_videos'])
    source_state={k:v.detach().cpu().clone() for k,v in runtime.generator.state_dict().items()}
    save_file({k:v.detach().cpu().contiguous() for k,v in runtime.execution.merged.items()},str(root/'merged_targets.safetensors'))
    write_json_atomic(root/'base_manifest.json',dict(source=runtime.source,MT=str(MT_PATH),
        merged_targets=38,merged_bytes=sum(v.nbytes for v in runtime.execution.merged.values()),
        execution='frozen merged target weights plus one38/r8 residual',read_and_experts='original source',
        coordinates=str(args.root/'coordinates.json')))
    condition=runtime.condition(task,demo)
    with autocast(runtime.device):raw=runtime.generator.meta.encode_raw(condition)
    readers=reader_consumers(args,runtime,root,condition,raw,source_state) if args.profile_consumer=='full' else []
    teacher=teacher_consumers(runtime,root,panel)
    functional,context_rmse=native_consumers(runtime,root)
    runtime.generator.load_state_dict(source_state);runtime.generator.zero_grad(set_to_none=True)
    write_json_atomic(root/'complete.json',dict(complete=True,disposable=True,formal_initial_never_updated=True,
        fresh_G_never_practiced=True,reader=readers,teacher=teacher,functional=functional))
    print(dict(complete=True,output=str(root),context_RMSE=context_rmse,reader=readers,teacher=teacher,functional=functional))


def reader_consumers(args,runtime,root,condition,raw,source_state):
    readers=[]
    for chunk in args.frame_chunks:
        runtime.generator.load_state_dict(source_state)
        runtime.generator.meta.frame_chunk=chunk
        def backward():
            runtime.generator.zero_grad(set_to_none=True)
            parent=runtime.generator.layout.pack(runtime.mt)
            xi=torch.randn_like(parent)*runtime.generator.layout.valid
            with autocast(runtime.device):
                features=runtime.generator.meta(condition,raw_prefix=raw)
                velocity=runtime.generator(xi,.5,parent,features,[],{'MT300':runtime.mt})
                loss=(velocity.float()[runtime.generator.layout.valid]+xi[runtime.generator.layout.valid]).square().mean()
            loss.backward()
            groups={group:sum(float(p.grad.float().square().sum()) for p,g in
                zip(runtime.generator.meta.values,runtime.generator.meta.groups,strict=True)
                if g==group and p.grad is not None)**.5 for group in ('gemma','action')}
            if any(v<=0 for v in groups.values()) or any(p.grad is not None for p in runtime.policy.parameters()):
                raise ValueError('rank8 CFM did not reach both meta groups with frozen source')
            return dict(loss=float(loss.detach()),meta_gradient_norm=groups,frames=len(condition[0]),
                        profile_coordinate_scale='unit scale, before legal bank; never formal')
        readers.append(dict(frame_chunk=chunk,**measured(runtime,backward)))
        write_json_atomic(root/'reader_profile.json',dict(physical_chunks=readers,disposable=True))
        if readers[-1]['OOM']:break
        if len(readers)==1:
            shared_resume_smoke(runtime,root)
    runtime.generator.load_state_dict(source_state);runtime.generator.zero_grad(set_to_none=True)
    return readers


def teacher_consumers(runtime,root,panel):
    stream=QueryStream(runtime,12,panel[0]['teacher_videos'][0],0)
    sample,_=stream.sample(runtime,0)
    owner=NativeFlowPrediction(runtime.policy)
    teacher=[]
    for microbatch in (28,56,112):
        model=DirectLoRAParameters(runtime.mt).to(runtime.device)
        def update():
            model.zero_grad(set_to_none=True)
            for first in range(0,112,microbatch):
                last=min(first+microbatch,112)
                part=FlowSample(tree_slice(sample.arguments,first,last),sample.target[first:last],sample.action_width)
                with autocast(runtime.device):
                    pred=torch.func.functional_call(owner,runtime.task_parameters(model()),(part,),strict=False)
                    loss=mean_velocity_loss(pred,part.target,part.action_width)*(last-first)/112
                loss.backward()
            b=sum(float(p.grad.square().sum()) for k,p in model().items() if '.lora_B.' in k)**.5
            if b<=0 or any(p.grad is not None for p in runtime.policy.parameters()):
                raise ValueError('actual MT-base/nonzeroA/B0 teacher gradient consumer failed')
            return dict(initial_B_gradient=b,valid_coordinates=runtime.lora.parameter_count)
        teacher.append(dict(microbatch=microbatch,**measured(runtime,update)))
        if microbatch==28 and not teacher[-1]['OOM']:
            optimizer=torch.optim.AdamW(model.parameters(),**OPTIMIZER)
            optimizer.step()
            event=dict(task_id=12,teacher_demo=panel[0]['teacher_videos'][0],event_ordinal=0,
                event_id='profile_rank8_resume',role='disposable_profile')
            saved=save_checkpoint(root/'teacher_resume',model,optimizer,stream,1,event)
            continued=fit_teacher(runtime,event,runtime.mt,dict(rec=[],keep=[]),root/'teacher_resume_continued',
                microbatch=28,stop=2,resume=saved)
            write_json_atomic(root/'teacher_resume_consumer.json',dict(complete=True,next_query=224,
                actual_optimizer_resume=True,valid_coordinates=runtime.lora.parameter_count,
                B_nonzero=any(bool(v.count_nonzero()) for k,v in continued.items() if '.lora_B.' in k),
                profile_only=True,checkpoint=str(saved)))
            del continued,optimizer
        del model
        write_json_atomic(root/'teacher_profile.json',dict(physical_batches=teacher,disposable=True))
        if teacher[-1]['OOM']:break
    stream.close();del sample,owner
    return teacher


def native_consumers(runtime,root):
    # Reuse historical OWN observations only for context/throughput, never a rank8 H/keep label.
    history=load_history(PRIOR_ROOT/'events/task_0012_event_01/history.pt.gz')
    observations=[history.observations[r['pre']] for r in history.records[:4]]
    batch_items=[processed(runtime,o,runtime.tasks[12].authority.language) for o in observations]
    batch={k:torch.cat([v[k] for v in batch_items]) for k in batch_items[0]}
    noise=torch.randn(4,50,32,generator=torch.Generator().manual_seed(1729)).to(runtime.device)
    with torch.no_grad():
        original_MT=load_file(str(MT_PATH),device=str(runtime.device))
        with runtime.execution.source_expert():
            old=runtime.native_actions([original_MT],batch,noise,
                batch_indices=torch.zeros(4,dtype=torch.long,device=runtime.device),checkpointed=False)
        current=runtime.native_actions([runtime.mt],batch,noise,
            batch_indices=torch.zeros(4,dtype=torch.long,device=runtime.device),checkpointed=False)
    context_rmse=float((old.float()-current.float()).square().mean().sqrt())
    if not torch.isfinite(current).all():raise ValueError('MT-merged initial native response nonfinite')
    labels=[dict(batch={k:v[i:i+1].cpu() for k,v in batch.items()},noise=noise[i:i+1].cpu(),
        target=current[i:i+1].float().cpu(),mask=torch.ones(1,5,7,dtype=torch.bool)) for i in range(4)]
    functional=[]
    for chunk in (1,2,4):
        model=DirectLoRAParameters(runtime.mt).to(runtime.device)
        with torch.no_grad():
            for name,p in model().items():
                if '.lora_B.' in name:p.add_(1e-3)
        result=measured(runtime,lambda:function_credit(runtime,model,labels,microbatch=chunk))
        result.update(microbatch=chunk);functional.append(result);del model
        if result['OOM']:break
    inference=native_inference(runtime,batch,noise)
    write_json_atomic(root/'native_context_profile.json',dict(context_RMSE_MT128_vs_merged_rank8_initial=context_rmse,
        numerical_equivalence_not_bitwise=True,functional=functional,physical_inference=inference,
        observation_source='historical actual own RGB/proprio; no formal H/keep labels created',
        task_export_factors=len(runtime.mt),task_values=runtime.lora.parameter_count,meta_not_in_task_export=True))
    return functional,context_rmse


def shared_resume_smoke(runtime,root):
    from .learning import checkpoint,resume_checkpoint
    optimizer=torch.optim.AdamW(runtime.generator.parameters(),**OPTIMIZER)
    optimizer.step();runtime.mark_psi_update()
    context=SimpleNamespace(world_size=1,is_main=True,rank=0,local_rank=0,device=runtime.device)
    saved=checkpoint(root/'resume_smoke','G',1,runtime,optimizer,context,
        sampler=dict(next_update=1,condition_batch=4,profile_only=True))
    next_update=resume_checkpoint(saved,'G',runtime,optimizer,context)
    if next_update!=1 or not optimizer.state:raise ValueError('real ψ/meta optimizer resume lost its cursor/state')
    write_json_atomic(root/'shared_resume_consumer.json',dict(complete=True,checkpoint=str(saved),
        next_update=next_update,optimizer_states=len(optimizer.state),schema='rank8_multistart_v2',profile_only=True))


def native_inference(runtime,batch,noise):
    results=[]
    for count in (4,8,16,32):
        repeats=count//4
        selected={k:v.repeat(repeats,*([1]*(v.ndim-1))) for k,v in batch.items()}
        selected_noise=noise.repeat(repeats,1,1)
        def consume():
            with torch.no_grad():
                actions=runtime.native_actions([runtime.mt],selected,selected_noise,
                    batch_indices=torch.zeros(count,dtype=torch.long,device=runtime.device),checkpointed=False)
            if not torch.isfinite(actions).all():raise ValueError('larger physical native ODE batch nonfinite')
            return dict(observations=count,ten_native_calls=True,duplicate_observations_only_for_throughput=True)
        result=measured(runtime,consume)
        if not result['OOM']:result['observations_per_second']=count/result['seconds']
        results.append(dict(physical_batch=count,**result))
        if result['OOM']:break
    return results
