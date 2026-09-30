"""§34: one fixed CPU projection and the existing official B20 consumer."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.operator_writer.learning_limit import (PROJECTED_ROOT as ROOT, ROOT as PRIOR,
    TASKS, TEACHERS, register_inputs, register_bank)
from operator_learning_limit_diagnosis import ASSET, FormalData, panels, prior_readout, source_setup


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def project():
    started = time.time()
    torch.set_num_threads(4)
    conditions = [(t,d) for t in TASKS for d in TEACHERS[t]]
    stems = [f'task{t:03d}_teacher{d:02d}' for t,d in conditions]
    banks = {arm: [load_file(str(PRIOR/arm/'bank'/f'{s}.safetensors')) for s in stems]
             for arm in ('parent','D')}
    z = [torch.load(PRIOR/'parent'/f'{s}_Z.pt', map_location='cpu', weights_only=False) for s in stems]
    names = list(z[0])
    if len(names)!=38 or any(set(v)!=set(names) for v in z):
        raise ValueError('complete parent38-target Z required')
    output = [{k:v.clone() for k,v in state.items()} for state in banks['parent']]
    shared, rows = {}, []
    for name in names:
        akey,bkey = name+LORA_A_SUFFIX,name+LORA_B_SUFFIX
        a = banks['parent'][0][akey].double()
        h = a@a.T
        y = [banks['D'][i][bkey].double()-banks['parent'][i][bkey].double() for i in range(8)]
        zi = [v[name].double() for v in z]
        if any(v.shape!=(256,128) for v in zi):
            raise ValueError('parent Z shape changed')
        gram = sum(v@h@v.T for v in zi)
        cross = sum(y[i]@h@zi[i].T for i in range(8))
        vals,vecs = torch.linalg.eigh(gram)
        keep = vals>vals[-1]*1e-12  # relative singular cutoff 1e-6, not eigenvalue cutoff
        basis = vecs[:,keep]
        delta = (cross@basis/vals[keep])@basis.T
        if not torch.isfinite(delta).all():
            raise ValueError('nonfinite projected shared delta O')
        shared[name] = delta.contiguous()
        per_condition = []
        for i,(task,teacher) in enumerate(conditions):
            correction = delta@zi[i]
            residual = y[i]-correction
            output[i][bkey] = (banks['parent'][i][bkey].double()+correction).float().contiguous()
            if not torch.isfinite(output[i][bkey]).all():
                raise ValueError('nonfinite FP32 projected B')
            per_condition.append({'task':task,'teacher':teacher,
                'target_BA_energy':float((y[i]@h*y[i]).sum()),
                'residual_BA_energy':float((residual@h*residual).sum()),
                'target_B_energy':float(y[i].square().sum()),
                'residual_B_energy':float(residual.square().sum())})
        rows.append({'target':name,'retained_rank':int(keep.sum()),
            'delta_O_frobenius_norm':float(delta.norm()),'conditions':per_condition})
    for stem,state in zip(stems,output,strict=True):
        if len(state)!=76:
            raise ValueError('complete38 A/B projected bank required')
        path=ROOT/'PZ/bank'/f'{stem}.safetensors'
        path.parent.mkdir(parents=True,exist_ok=True)
        save_file(state,str(path))
    projection = ROOT/'projection';projection.mkdir(parents=True,exist_ok=True)
    save_file(shared,str(projection/'delta_O.safetensors'))
    target = sum(c['target_BA_energy'] for row in rows for c in row['conditions'])
    residual = sum(c['residual_BA_energy'] for row in rows for c in row['conditions'])
    record = {'status':'complete','source_root':str(PRIOR),'singular_relative_cutoff':1e-6,
        'definition':'Y_i=B_D_i-B_parent_i; H=A_parent A_parent^T; shared FP64 Gram/eigh minimizes sum ||(Y_i-delta_O Z_parent_i) A_parent||_F^2; B_PZ=B_parent+delta_O Z_parent',
        'conditions':[{'task':t,'teacher':d,'parent':str(PRIOR/'parent/bank'/f'{s}.safetensors'),
            'D':str(PRIOR/'D/bank'/f'{s}.safetensors'),'Z':str(PRIOR/'parent'/f'{s}_Z.pt')}
            for (t,d),s in zip(conditions,stems,strict=True)],
        'target_BA_energy':target,'residual_BA_energy':residual,
        'explained_BA_fraction':1-residual/target,'rows':rows,'optimizer_updates':0,
        'native_model_forwards':0,'CPU_seconds':time.time()-started,
        'limitations':'unconstrained fixed-Z projection of observed train repairs, not learned O or a deployable new Writer'}
    save_json(projection/'projection.json',record)
    register_inputs('PZ')
    for slot in (0,1):
        register_bank('PZ',slot)
    print(json.dumps({'explained_BA_fraction':record['explained_BA_fraction'],
        'action_in_delta_O_norm':next(v['delta_O_frobenius_norm'] for v in rows if v['target']=='model.action_in_proj'),
        'CPU_seconds':record['CPU_seconds']}),flush=True)


def generate(task_ids, device, microbatch):
    started = time.time()
    physical = int(device.split(':')[1])
    allowed = set(os.sched_getaffinity(0))
    local = set(range(28))|set(range(56,84)) if physical<4 else set(range(28,56))|set(range(84,112))
    os.sched_setaffinity(0,allowed&local)
    torch.set_num_threads(4)
    torch.cuda.set_device(physical)
    spec,runtime = source_setup(device)
    data = FormalData(ASSET,spec,task_ids=task_ids)
    reader,panel = prior_readout(),panels()
    out = ROOT/'PZ/B20';out.mkdir(parents=True,exist_ok=True)
    try:
        for task in task_ids:
            reference_path = reader.PRIOR/f'group0/task{task:03d}_B_query_noise_target.pt'
            reference = torch.load(reference_path,map_location='cpu',weights_only=False)
            observation,noise,target,valid = reader.b_batch(data,runtime,task,panel[str(task)]['B'],reference,torch=torch)
            for teacher in TEACHERS[task]:
                stem = f'task{task:03d}_teacher{teacher:02d}'
                state = load_file(str(ROOT/'PZ/bank'/f'{stem}.safetensors'),device=device)
                validate_lora_state(state,runtime.lora)
                runtime.restore_identity()
                prediction = reader.generated(runtime,state,observation,noise,microbatch,torch)[:,:,:7]
                torch.save({'prediction':prediction,'risk':reader.risk(prediction,target,valid,torch),
                    'target_ref':str(reference_path),'queries':panel[str(task)]['B'],'valid':valid,
                    'D_ref':str(PRIOR/'D'/f'{stem}_endpoint.pt'),'microbatch':microbatch},
                    out/f'{stem}_PZ.pt')
                print('B20',task,teacher,'complete',flush=True)
        save_json(out/('completion_'+','.join(map(str,task_ids))+'.json'),
            {'status':'complete','tasks':task_ids,'conditions':2*len(task_ids),'microbatch':microbatch,
             'peak_allocated_GPU_GiB':torch.cuda.max_memory_allocated()/2**30,'wall_seconds':time.time()-started})
    finally:
        data.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',action='store_true')
    parser.add_argument('--tasks',default='0,12,20,32')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--microbatch',type=int,default=20)
    args=parser.parse_args()
    if args.project:
        project()
    else:
        tasks=tuple(map(int,args.tasks.split(',')))
        if not tasks or not set(tasks)<=set(TASKS):
            raise ValueError('unregistered projected B20 task')
        generate(tasks,args.device,args.microbatch)


if __name__=='__main__':
    main()
