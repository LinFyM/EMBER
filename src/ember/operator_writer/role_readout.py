"""Fixed B20 official FM/ten-step projection and teacher-origin readback."""
from __future__ import annotations
from contextlib import contextmanager
import torch
from safetensors.torch import save_file
from ember.lora import LORA_A_SUFFIX,copy_task_lora_state_
from ember.pi05_source_checkpoint import write_json_atomic
from ember.writer.function_credit import FlowSample,flow_sample
from ember.writer.materialization import file_record
from ember.writer.runtime import autocast
from .role_labels import ROOT,TEACHERS
from .role_learning import FINAL,RoleFlowPrediction,compile_fixed,raw_batch,query_geometry

PRIOR=ROOT.parent/'operator_chain_diagnosis_20260929/functional_credit_transport/group0'


@contextmanager
def capture_projection(policy,matrices):
    """Passive actual ten denoise calls; matrices are these conditions' complete A."""
    projections,times=[],[]
    denoise=policy.model.denoise_step
    def called(*args,**kwargs):
        hidden=[]
        handle=policy.get_submodule(FINAL).register_forward_pre_hook(lambda module,inputs:hidden.append(inputs[0]))
        try:result=denoise(*args,**kwargs)
        finally:handle.remove()
        if len(hidden)!=1 or hidden[0].shape[1:]!=(50,1024):
            raise ValueError('official actual action_out hidden changed')
        h=hidden[0].float()
        a=matrices.to(device=h.device,dtype=h.dtype)
        z=torch.einsum('bsh,bch->bsc',h,a)
        projections.append(z.detach().float().cpu())
        tau=args[3] if len(args)>3 else kwargs['timestep']
        times.append(tau.detach().float().cpu())
        return result
    policy.model.denoise_step=called
    try:yield projections,times
    finally:policy.model.denoise_step=denoise


@torch.no_grad()
def readout(runtime,data,labels,panels,cached,out):
    records=[]
    for task,teachers in TEACHERS.items():
        panel=panels[str(task)];queries=panel['B'];size=20
        refpath=PRIOR/f'task{task:03d}_B_query_noise_target.pt'
        ref=torch.load(refpath,map_location='cpu',weights_only=False,mmap=True)
        if ref['queries']!=queries or ref['flow_seed']!=panel['B_flow_seed']:
            raise ValueError('fixed B20 identities changed')
        raw=raw_batch(data,task,queries);batch=runtime.processor.training_batch(raw)
        r=query_geometry(labels,task,queries,runtime.device)
        for teacher in teachers:
            identifier=f'task{task:03d}_teacher{teacher:02d}'
            state=compile_fixed(runtime,cached[(task,teacher)])
            factor=out/'bank'/f'{identifier}.safetensors';factor.parent.mkdir(exist_ok=True)
            save_file({k:v.cpu().contiguous() for k,v in state.items()},str(factor))
            runtime.restore_identity();owner=RoleFlowPrediction(runtime.policy)
            with autocast(runtime.device):
                sample=flow_sample(runtime.policy,batch,seed=ref['flow_seed'],device=runtime.device,random_batch=20,offset=0)
                noise=ref['noise'].to(runtime.device)
                sample=FlowSample((*sample.arguments[:5],noise,sample.arguments[6]),noise-sample.arguments[4],7)
                velocity,h=torch.func.functional_call(owner,{'policy.'+k:v for k,v in state.items()},(sample,),strict=False)
                zfm=torch.nn.functional.linear(h,state[FINAL+LORA_A_SUFFIX][:3])
                copy_task_lora_state_(runtime.policy,state,runtime.lora)
                a=state[FINAL+LORA_A_SUFFIX][:3].expand(size,-1,-1)
                obs={k:v for k,v in batch.items() if k!='action'}
                with capture_projection(runtime.policy,a) as (projections,taus):
                    generated=runtime.policy.predict_action_chunk(obs,noise=noise,num_steps=10)
            if len(projections)!=10 or abs(float(taus[0][0])-1)>1e-6:
                raise ValueError('official B20 requires ten actual flow calls starting tau1')
            teacher_label=labels['teachers'][f'{task}:{teacher}']
            with autocast(runtime.device):
                zt=torch.nn.functional.linear(cached[(task,teacher)]['X'][FINAL][:-1],state[FINAL+LORA_A_SUFFIX][:3])
            row=dict(schema='ember_role_coordinate_readout_v1',arm=out.name,task=task,teacher=teacher,
                queries=queries,flow_seed=ref['flow_seed'],query_offset=1,target_ref=file_record(refpath),
                prediction_FM=velocity.float().cpu(),FM_target=sample.target.float().cpu(),
                action_generated_normalized=generated.float().cpu(),action_target=batch['action'].cpu(),
                valid_mask=~raw['action_is_pad'],noise=noise.cpu(),time=sample.arguments[-1].cpu(),
                geometry_target=r.cpu(),geometry_FM=zfm.float().cpu(),
                geometry_generated=torch.stack(projections,1),taus=torch.stack(taus,1),
                teacher_projection=zt.float().cpu(),teacher_target=torch.as_tensor(teacher_label['relative_m']),
                teacher_origin_frames=teacher_label['frames'],projection_after_last_AdaRMSNorm=True,
                official_num_steps=10,extra_forward=False)
            path=out/'functional'/f'{identifier}_B.pt';path.parent.mkdir(exist_ok=True)
            torch.save(row,path)
            records.append(dict(condition_id=identifier,global_task_id=task,teacher_demo=teacher,
                factors=file_record(factor),functional=str(path)))
    write_json_atomic(out/'functional/manifest.json',dict(records=records,query_panel='B20_only',
        condition_queries=160,teacher_origins=256,no_A28_matrix=True))
    runtime.restore_identity()
    return records
