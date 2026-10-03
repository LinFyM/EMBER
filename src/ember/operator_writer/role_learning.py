"""Fixed X/H/c/d, live A/B compilation and actual own-hidden coordinate credit.

Task-only orchestration of ConditionalTarget and canonical NativeFlowPrediction;
retired after the registered fixed64/96-row diagnostic.
"""
from __future__ import annotations
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.pi05_source_checkpoint import write_json_atomic
from ember.writer.function_credit import NativeFlowPrediction, flow_sample, _add
from ember.writer.materialization import file_record
from ember.writer.runtime import autocast
from .native import read_native_video
from .role_labels import ROOT, TEACHERS

GROUPS = ('a_x','a_z','a_context','a_dynamic','a_out',
          'b_key','b_delta','b_context','b_dynamic','b_out')
FINAL = 'model.action_out_proj'


class RoleFlowPrediction(NativeFlowPrediction):
    """Return the same actual velocity and post-AdaRMSNorm projection input."""
    def forward(self,sample,prepared=None):
        inputs=[]
        handle=self.policy.get_submodule(FINAL).register_forward_pre_hook(
            lambda module,args: inputs.append(args[0]))
        try:
            velocity=super().forward(sample,prepared)
        finally:
            handle.remove()
        if len(inputs)!=1 or inputs[0].shape[-2:]!=(50,1024):
            raise ValueError('actual action_out hidden capture changed')
        return velocity,inputs[0]


def freeze_heads(writer):
    writer.requires_grad_(False)
    parameters=[]
    for unit in writer.conditional_targets:
        for name in GROUPS:
            getattr(unit,name).requires_grad_(True)
            parameters.extend(getattr(unit,name).parameters())
    if len(parameters)!=380:
        raise ValueError('both arms must train all38 times ten A/B weight tensors')
    return tuple(parameters)


def raw_batch(data,task,queries):
    return default_collate([data.queries[data.rows[task][q['demo']][q['frame']]] for q in queries])


def query_geometry(labels,task,queries,device):
    values=[]
    for row in queries:
        item=labels['queries'][f'{task}:{row["demo"]}']
        i=int(np.searchsorted(item['frames'],row['frame']))
        if i>=len(item['frames']) or item['frames'][i]!=row['frame']:
            raise ValueError('exact query geometry missing')
        values.append(item['relative_m'][i])
    return torch.as_tensor(np.stack(values),device=device,dtype=torch.float32)


@torch.no_grad()
def native_cache(runtime,data,*,create,frame_chunk=128):
    records,cached=[],{}
    for task,teachers in TEACHERS.items():
        for teacher in teachers:
            path=ROOT/'native'/f'task{task:03d}_teacher{teacher:02d}.pt'
            if create:
                condition,raw,sampled=data.condition(runtime,task,teacher)
                runtime.restore_identity();started=time.monotonic()
                with autocast(runtime.device):
                    x,h=read_native_video(runtime.policy,runtime.writer.public_state(),runtime.writer.probe,
                        condition,runtime.writer.names,frame_chunk=frame_chunk,checkpoint_frames=False)
                    c,d=runtime.writer.interpreter(h,condition[1])
                fixed=dict(X=x,H=h,c=c,d=d,frame_indices=condition[1])
                torch.save({k:({n:v.cpu().contiguous() for n,v in value.items()} if isinstance(value,dict)
                    else value.cpu().contiguous()) for k,value in fixed.items()},path)
                record=dict(task=task,teacher=teacher,raw_frames=raw,sampled_frames=sampled,
                    frame_chunk=frame_chunk,actual_frame_upper_bound_exhausted=sampled<=frame_chunk,
                    seconds=time.monotonic()-started,public_native_reads=1,
                    fields_read=['obs/agentview_rgb','obs/eye_in_hand_rgb'])
                write_json_atomic(path.with_suffix('.json'),record)
                del x,h,c,d,fixed,condition
            fixed=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            if set(fixed)!= {'X','H','c','d','frame_indices'} or set(fixed['X'])!=set(runtime.writer.names):
                raise ValueError('only fixed full38 native X/H/c/d may be cached')
            def gpu(value):
                return {k:gpu(v) for k,v in value.items()} if isinstance(value,dict) else value.to(runtime.device)
            cached[(task,teacher)]=gpu(fixed)
            records.append(file_record(path))
    if create:
        write_json_atomic(ROOT/'native/manifest.json',dict(records=records,native_reads=8,same_cache_both_arms=True,
            fields=['X','H','c','d','frame_indices'],A_S_K_deltaZ_Value_M_cached=False))
    return cached


def target(writer,unit,name,fixed):
    public=writer.public_state()
    return unit(public[name+LORA_A_SUFFIX],public[name+LORA_B_SUFFIX],
                fixed['X'][name],fixed['H'],fixed['c'],fixed['d'])[:2]


@torch.no_grad()
def compile_fixed(runtime,fixed):
    state={}
    with autocast(runtime.device):
        for name,unit in zip(runtime.writer.names,runtime.writer.conditional_targets,strict=True):
            state[name+LORA_A_SUFFIX],state[name+LORA_B_SUFFIX]=target(runtime.writer,unit,name,fixed)
    validate_lora_state(state,runtime.lora)
    return state


def paired_credit(runtime,states,batch,r,seed,micro,geometry):
    owner=RoleFlowPrediction(runtime.policy);total=28
    credits=[dict(FM=0.,self_geometry=0.,lora_cotangent={}) for _ in states]
    for start in range(0,total,micro):
        stop=min(total,start+micro)
        sliced={k:v[start:stop] if isinstance(v,torch.Tensor) and v.ndim and len(v)==total else v
                for k,v in batch.items()}
        sample=flow_sample(runtime.policy,sliced,seed=seed,device=runtime.device,random_batch=28,offset=start)
        prepared=owner.prepare(sample)
        leaves={k:torch.stack([s[k] for s in states]).detach().requires_grad_(True) for k in states[0]}
        def forward(values):
            return torch.func.functional_call(owner,{'policy.'+k:v for k,v in values.items()},
                                               (sample,prepared),strict=False)
        velocity,h=torch.vmap(forward)(leaves)
        projected=torch.einsum('bnsh,bch->bnsc',h.float(),leaves[FINAL+LORA_A_SUFFIX][:,:3].float())
        losses=(velocity[...,:7].float()-sample.target[None,...,:7].float()).square().mean((1,2,3))
        coordinates=(projected-r[None,start:stop,None]).square().mean((1,2,3))
        gradients=torch.autograd.grad((losses+(coordinates*.5 if geometry else 0)).sum(),tuple(leaves.values()))
        weight=(stop-start)/28
        for index,credit in enumerate(credits):
            _add(credit['lora_cotangent'],{k:g[index] for k,g in zip(leaves,gradients,strict=True)},weight/8)
            credit['FM']+=float(losses[index].detach())*weight
            credit['self_geometry']+=float(coordinates[index].detach())*weight
    return credits


def step(runtime,cached,data,labels,entry,arm,optimizer,parameters,micro):
    started=time.monotonic();optimizer.zero_grad(set_to_none=True);rows=[]
    for task,teachers in TEACHERS.items():
        event=entry[str(task)]
        batch=runtime.processor.training_batch(raw_batch(data,task,event['queries']))
        r=query_geometry(labels,task,event['queries'],runtime.device)
        states=[compile_fixed(runtime,cached[(task,teacher)]) for teacher in teachers]
        runtime.restore_identity()
        with autocast(runtime.device):
            credits=paired_credit(runtime,states,batch,r,event['flow_seed'],micro,arm=='G')
        del states
        for teacher,credit in zip(teachers,credits,strict=True):
            fixed=cached[(task,teacher)]; teacher_loss=None
            with autocast(runtime.device):
                for name,unit in zip(runtime.writer.names,runtime.writer.conditional_targets,strict=True):
                    a,b=target(runtime.writer,unit,name,fixed)
                    outputs=[a,b];cotangents=[credit['lora_cotangent'][name+LORA_A_SUFFIX].to(a),
                                             credit['lora_cotangent'][name+LORA_B_SUFFIX].to(b)]
                    if name==FINAL:
                        label=labels['teachers'][f'{task}:{teacher}']
                        if not np.array_equal(label['frames'],fixed['frame_indices'][:-1].cpu().numpy()):
                            raise ValueError('teacher geometric origin indices differ from actual native')
                        rt=torch.as_tensor(label['relative_m'],device=runtime.device,dtype=torch.float32)
                        teacher_loss=(F.linear(fixed['X'][name][:-1].float(),a[:3].float()).float()
                                      -rt[:,None]).square().mean()
                        if arm=='G':outputs.append(teacher_loss);cotangents.append(torch.full_like(teacher_loss,.5/8))
                    torch.autograd.backward(outputs,cotangents)
            rows.append(dict(task=task,teacher=teacher,weight=1/8,flow_seed=event['flow_seed'],queries=event['queries'],
                FM=credit['FM'],self_geometry=credit['self_geometry'],teacher_geometry=float(teacher_loss.detach()),
                complete_factor_cotangent=76,teacher_geometry_applications=int(arm=='G')))
    norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    group_norms={g:float(torch.stack([getattr(u,g).weight.grad.float().norm()
        for u in runtime.writer.conditional_targets]).norm()) for g in GROUPS}
    if any(p.grad is not None for p in runtime.writer.parameters() if not p.requires_grad):
        raise ValueError('frozen public/native/interpreter acquired gradients')
    optimizer.step();torch.cuda.synchronize()
    return dict(rows=rows,seconds=time.monotonic()-started,microbatch=micro,physical_suffix_queries=2*micro,
        teacher_conditions_packed=2,frozen_prefix_shared=True,unclipped_gradient_norm=float(norm),gradient_groups=group_norms,
        allocated_peak_GiB=torch.cuda.max_memory_allocated()/2**30,reserved_peak_GiB=torch.cuda.max_memory_reserved()/2**30)
