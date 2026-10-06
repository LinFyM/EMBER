"""One frozen parent-B projection and the registered historical B20 consumer."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.operator_writer.learning_limit import (FIXED_B_ROOT as ROOT, ROOT as OLD,
    TASKS, TEACHERS, BASE_BANK, register_inputs, register_bank)
from ember.pi05_source_checkpoint import read_json, write_json_atomic

SEALED = Path('/data1/user/ymdai/projects/EMBER-projected-repair-formal')
READOUT = OLD.parent/'operator_chain_diagnosis_20260929/self_conditioned_readout/analysis_script_v2.py'


def leaf(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def project():
    started = time.time()
    torch.set_num_threads(4)
    records = []
    for task in TASKS:
        for teacher in TEACHERS[task]:
            stem = f'task{task:03d}_teacher{teacher:02d}'
            inputs = {arm: OLD/arm/'bank'/f'{stem}.safetensors' for arm in ('parent','D')}
            parent, direct = (load_file(str(inputs[arm])) for arm in ('parent','D'))
            if set(parent) != set(direct) or len(parent) != 76:
                raise ValueError('complete38 parent/D factor registry changed')
            output = dict(parent)
            for bkey in (k for k in parent if k.endswith(LORA_B_SUFFIX)):
                akey = bkey[:-len(LORA_B_SUFFIX)] + LORA_A_SUFFIX
                b, delta = parent[bkey].double(), direct[bkey].double()-parent[bkey].double()
                u, singular, _ = torch.linalg.svd(b, full_matrices=False)
                keep = singular > 1e-6*singular.max()
                basis = u[:,keep]
                correction = basis@(basis.T@delta)
                residual = delta-correction
                h = parent[akey].double()@parent[akey].double().T
                output[bkey] = (b+correction).float().contiguous()
                if output[bkey].shape != b.shape or not torch.isfinite(output[bkey]).all():
                    raise ValueError('nonfinite or reshaped PB factor')
                records.append({'task':task,'teacher':teacher,'site':bkey[:-len(LORA_B_SUFFIX)],
                    'rank':int(keep.sum()),'target_B_energy':float(delta.square().sum()),
                    'retained_B_energy':float(correction.square().sum()),
                    'removed_B_energy':float(residual.square().sum()),
                    'target_BA_energy':float((delta@h*delta).sum()),
                    'retained_BA_energy':float((correction@h*correction).sum()),
                    'removed_BA_energy':float((residual@h*residual).sum()),
                    'parent':str(inputs['parent']),'D':str(inputs['D'])})
            path = ROOT/'PB/bank'/f'{stem}.safetensors'
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                raise ValueError('refuse to overwrite projected bank')
            save_file(output,str(path))
    record = {'status':'complete','source_root':str(OLD),'projection_space':'complete_parent_B_column_space',
        'singular_relative_cutoff':1e-6,'formula':'A_PB=A_T; B_PB=B_T+U_keep U_keep.T (B_D-B_T)',
        'arithmetic':'CPU FP64 SVD/projection; complete ordinary FP32 banks',
        'rows':records,'optimizer_updates':0,'native_teacher_forwards':0,'seconds':time.time()-started}
    write_json_atomic(ROOT/'projection/projection.json',record)
    register_inputs('PB')
    for slot in (0,1):
        register_bank('PB',slot)
    print(json.dumps({'sites':len(records),'projection_seconds':record['seconds']}),flush=True)


def generate():
    from ember.operator_writer.data import FormalData
    from ember.pi05_eval_contract import load_evaluation_authorities
    from ember.pi05_source_setup import load_policy
    from ember.pi05_processing import Pi05LiberoProcessor
    from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
    from ember.writer.functional import prepare_frozen_writer_policy
    from ember.writer.topology import bind_current_process_to_cuda_numa

    started = time.time()
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    bind_current_process_to_cuda_numa(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = torch.device('cuda:0')
    base = read_json(BASE_BANK)
    asset, spec = Path(base['asset_root']), read_json(Path(base['spec']['path']))
    authorities = load_evaluation_authorities(asset/spec['source']['evaluation_config'],asset)
    policy = load_policy(Path(base['source']['model_path']),authorities.source_base_config,device)
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(asset/spec['source']['lora_contract']),rank=128)
    prepare_frozen_writer_policy(policy,lora)
    policy.model.gradient_checkpointing_disable()
    processor = Pi05LiberoProcessor(read_json(asset/spec['source']['normalization'])['stats'],
                                   asset/spec['source']['tokenizer'],200,str(device))
    runtime = SimpleNamespace(policy=policy,lora=lora,processor=processor,device=device)
    leaf('ember.writer.flow',SEALED/'src/ember/writer/flow.py')
    reader = leaf('fixed_b_registered_B20_leaf',READOUT)
    panel = read_json(reader.PRIOR/'group0/fixed_panels.json')
    data = FormalData(asset,spec,task_ids=TASKS)
    out = ROOT/'PB/B20'
    out.mkdir(parents=True,exist_ok=True)
    profiles = []
    microbatch = 20
    try:
        for task in TASKS:
            event = panel[str(task)]
            if tuple(event['teachers']) != TEACHERS[task] or len(event['B']) != 20:
                raise ValueError('registered teachers/B20 changed')
            reference_path = reader.PRIOR/f'group0/task{task:03d}_B_query_noise_target.pt'
            reference = torch.load(reference_path,map_location='cpu',weights_only=False)
            obs,noise,target,valid = reader.b_batch(data,runtime,task,event['B'],reference,torch=torch)
            for teacher in TEACHERS[task]:
                stem = f'task{task:03d}_teacher{teacher:02d}'
                state = load_file(str(ROOT/'PB/bank'/f'{stem}.safetensors'),device=str(device))
                validate_lora_state(state,lora)
                if not profiles:
                    # Repeats of the same registered events only; no new scientific rows.
                    for batch in (10,20,40):
                        copies = 2 if batch == 40 else 1
                        po = {k:(torch.cat([v]*copies) if isinstance(v,torch.Tensor) and v.ndim and len(v)==20 else v)
                              for k,v in obs.items()}
                        pn = torch.cat([noise]*copies)
                        torch.cuda.synchronize(); tick = time.time(); torch.cuda.reset_peak_memory_stats()
                        reader.generated(runtime,state,po,pn,batch,torch)
                        torch.cuda.synchronize()
                        profiles.append({'batch':batch,'samples':len(pn),'seconds':time.time()-tick,
                            'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30})
                    microbatch = max(profiles[:2],key=lambda row:row['samples']/row['seconds'])['batch']
                    write_json_atomic(out/'profile.json',{'rows':profiles,'selected_batch':microbatch,
                        'stop_reason':'20 distinct queries per fixed condition; batch40 uses repetitions only, no new events or conditions'})
                tick = time.time()
                prediction = reader.generated(runtime,state,obs,noise,microbatch,torch)[:,:,:7]
                torch.save({'prediction':prediction,'risk':reader.risk(prediction,target,valid,torch),
                    'target_ref':str(reference_path),'queries':event['B'],'valid':valid,
                    'D_ref':str(OLD/'D'/f'{stem}_endpoint.pt'),'microbatch':microbatch,
                    'seconds':time.time()-tick},out/f'{stem}_PB.pt')
                print('B20',task,teacher,'complete',flush=True)
    finally:
        data.close()
    write_json_atomic(out/'completion.json',{'status':'complete','conditions':8,'predictions':160,
        'seconds':time.time()-started,'B20_leaf':str(READOUT),'flow_leaf':str(SEALED/'src/ember/writer/flow.py'),
        'runtime_tree':str(Path(__file__).resolve().parents[1]),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project',action='store_true')
    args = parser.parse_args()
    project() if args.project else generate()
