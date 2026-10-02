#!/usr/bin/env python3
"""Unique bounded live/stop §102 batch; retired on delivery, never a trainer mode."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import socket
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from torch.utils.data import default_collate

from ember.pi05_evaluation import rollout_shard  # canonical facade initializes internal imports
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, copy_task_lora_state_, validate_lora_state
from ember.operator_writer.run import build_runtime, specification, CONTINUATION2340_SPEC_PATH
from ember.operator_writer.data import FormalData
from ember.operator_writer.native import read_native_video
from ember.operator_writer.bank import inspect_bank
from ember.operator_writer.state_credit_diagnostic import CreditReader, compile_content, coupled_credit, writer_vjp, residual_output
from ember.operator_writer.state_credit_evaluation import evaluate, condition_id, TEACHERS
from ember.pi05_assets import prepare_libero_config
from ember.pi05_eval_contract import git_state, git_state_is_clean_pushed_or_frozen_authority
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record
from ember.writer.runtime import autocast
from ember.writer.topology import bind_current_process_to_cuda_numa

ASSET = Path('/data1/user/ymdai/projects/EMBER')
ROOT = Path('/data1/user/ymdai/ember_runs/state_coupled_functional_credit_20261002')
OLD = ROOT.parent/'operator_learning_limit_diagnosis_20260930'
PRIOR = ROOT.parent/'operator_chain_diagnosis_20260929/functional_credit_transport/group0'
ECP = ROOT.parent/'operator_read_write_learning_20260928/continuation2340/T/train/attempts/continuation/checkpoints/macro_00002340/ecp.safetensors'
FEATURE32 = ROOT.parent/'task32_learned_video_segment_20261002/native'
TASKS = (0, 12, 20, 32)


def save_pt(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def rng_state():
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def optimizer(runtime, gamma):
    params = (*runtime.writer.writes.parameters(), *gamma.parameters())
    return torch.optim.AdamW(params, lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=1e-4), params


def raw_batch(data, task, queries):
    if set(TEACHERS[task]) & {q['demo'] for q in queries}:
        raise ValueError('teacher entered the action-query stream')
    return default_collate([data.queries[data.rows[task][q['demo']][q['frame']]] for q in queries])


def error(pairs):
    numerator = sum(float((a.float()-b.float()).square().sum()) for a,b in pairs)
    denominator = sum(float(b.float().square().sum()) for a,b in pairs)
    return dict(relative_L2=(numerator/max(denominator, 1e-30))**.5,
                max_abs=max(float((a.float()-b.float()).abs().max()) for a,b in pairs))


def native_features(runtime, data, create):
    records, cached = [], {}
    for task in TASKS:
        for teacher in TEACHERS[task]:
            path = ROOT/'native'/f'{condition_id(task,teacher)}_H_K.pt'
            if task == 32:
                path = FEATURE32/f'task032_teacher{teacher}_H_K.pt'
            elif create:
                condition, raw, sampled = data.condition(runtime, task, teacher)
                runtime.restore_identity()
                started = time.monotonic()
                with torch.no_grad(), autocast(runtime.device):
                    x,h = read_native_video(runtime.policy, runtime.writer.public_state(),
                        runtime.writer.probe, condition, runtime.writer.names,
                        frame_chunk=64, checkpoint_frames=False)
                keys = {n: F.normalize(F.linear(v.float(), runtime.writer.public_state()[n+LORA_A_SUFFIX].float()),
                    dim=-1, eps=1e-6).transpose(1,2).cpu() for n,v in x.items()}
                save_pt(path, dict(H=h.cpu(), K=keys, frame_indices=condition[1].cpu(),
                    parameter_sources=dict(parent=file_record(ECP)), provenance=dict(task=task,teacher=teacher,
                    hdf5=str(data.tasks[task].authority.path), fields_read=['obs/agentview_rgb','obs/eye_in_hand_rgb'],
                    raw_frames=raw, sampled_frames=sampled, frame_chunk=64,
                    seconds=time.monotonic()-started, public_native_reads=1, git=git_state(Path(__file__).resolve().parents[1]))))
                del x,h,keys
            if not path.exists():
                raise FileNotFoundError('native owner must finish six captures before second arm starts')
            record = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            if (set(record['K']) != set(runtime.writer.names) or record['H'].shape[1:] != (50,1024)
                    or any(k.shape != (len(record['H']),128,50) for k in record['K'].values())
                    or len(record['frame_indices']) != len(record['H'])):
                raise ValueError('public H/K cache incomplete')
            cached[(task,teacher)] = dict(H=record['H'].to(runtime.device),
                K={n:v.to(runtime.device) for n,v in record['K'].items()},
                raw_frames=record['provenance']['raw_frames'])
            records.append(dict(task=task,teacher=teacher,path=str(path),provenance=record['provenance'],
                parameter_sources=record['parameter_sources'], frame_indices=record['frame_indices'].tolist()))
    if create:
        write_json_atomic(ROOT/'native/manifest.json', dict(records=records, new_public_native_reads=6,
            reused_public_native=2, feature_owner='live', same_features_both_arms=True))
        print(json.dumps(dict(event='native_ready', new_reads=6, reused=2)), flush=True)
    return cached


def step(runtime, gamma, cached, data, entry, arm, opt, params, micro):
    started = time.monotonic()
    opt.zero_grad(set_to_none=True)
    losses = []
    for task in TASKS:
        q = entry[str(task)]
        batch = runtime.processor.training_batch(raw_batch(data, task, q['queries']))
        for teacher in TEACHERS[task]:
            state,z = compile_content(runtime.writer,cached[(task,teacher)])
            with autocast(runtime.device):
                credit = coupled_credit(runtime,state,z,gamma,batch,seed=q['flow_seed'],
                    microbatch=micro,live=arm=='live',backward=True)
            writer_vjp(runtime.writer,state,z,credit)
            losses.append(dict(task=task,teacher=teacher,L_F=credit['L_F'],L_R=credit['L_R'],weight=1/8))
            del state,z,credit
    norm = torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
    opt.step()
    torch.cuda.synchronize()
    return dict(losses=losses, unclipped_gradient_norm=float(norm), seconds=time.monotonic()-started,
        microbatch=micro, allocated_peak_GiB=torch.cuda.max_memory_allocated()/2**30,
        reserved_peak_GiB=torch.cuda.max_memory_reserved()/2**30)


def initial_readback(runtime, cached):
    records = []
    with torch.no_grad():
        for (task,teacher), features in cached.items():
            state,z = compile_content(runtime.writer,features,checkpoint_targets=False)
            retained = load_file(str(OLD/f'parent/bank/{condition_id(task,teacher)}.safetensors'), device='cuda:0')
            native_error = error([(state[n+LORA_B_SUFFIX],retained[n+LORA_B_SUFFIX]) for n in runtime.writer.names])
            closure = error([(runtime.writer.writes[i].o(z[i].T).T,
                state[n+LORA_B_SUFFIX]-runtime.writer.public_state()[n+LORA_B_SUFFIX]) for i,n in enumerate(runtime.writer.names)])
            if native_error['relative_L2']>.1 or closure['relative_L2']>.01:
                raise ValueError('material initial reconstruction discrepancy; no numerical sweep')
            records.append(dict(task=task,teacher=teacher,parent_B=native_error,M_equals_OZ=closure))
    return records


def functional_readback(runtime, gamma, cached, data, panel, out, micro):
    records = []
    with torch.no_grad():
        for task in TASKS:
            p = panel[str(task)]
            for teacher in TEACHERS[task]:
                state,z = compile_content(runtime.writer,cached[(task,teacher)],checkpoint_targets=False)
                validate_lora_state(state,runtime.lora)
                ident = condition_id(task,teacher)
                factorpath = out/'bank'/f'{ident}.safetensors'
                factorpath.parent.mkdir(parents=True,exist_ok=True)
                save_file({k:v.cpu().contiguous() for k,v in state.items()},str(factorpath))
                zpath = out/'bank'/f'{ident}_Z.pt'
                save_pt(zpath,z.cpu())
                functional = {}
                for label in ('A','B'):
                    raw = raw_batch(data,task,p[label])
                    valid = (~raw['action_is_pad']).sum(1).to(torch.int64)
                    batch = runtime.processor.training_batch(raw)
                    refpath = PRIOR/f'task{task:03d}_{label}_query_{"flow" if label=="A" else "noise"}_target.pt'
                    ref = torch.load(refpath,map_location='cpu',weights_only=False,mmap=True)
                    seed = p['A_flow_seed'] if label=='A' else p['B_flow_seed']
                    with autocast(runtime.device):
                        fm = coupled_credit(runtime,state,z,gamma,batch,seed=seed,microbatch=micro,live=True,backward=False)
                    row = dict(queries=p[label],flow_seed=seed,query_offset=1,target_ref=str(refpath),
                        FM_target=fm['target'],prediction_FM_student=fm['student'],prediction_FM_reader=fm['reader'],
                        action_target=batch['action'].cpu(), valid_future_lengths=valid,L_F=fm['L_F'],L_R=fm['L_R'])
                    if label=='A' and not torch.equal(fm['target'],ref['FM_target'][...,:7]):
                        raise ValueError('fixed A28 flow/noise/time target pairing changed')
                    if label=='B':
                        if (ref['queries']!=p[label] or not torch.equal(row['action_target'],ref['target'])
                                or not torch.equal(valid,ref['valid_future_lengths'])):
                            raise ValueError('original B20 identity/normalization/valid changed')
                        obs = {k:v for k,v in batch.items() if k!='action'}
                        copy_task_lora_state_(runtime.policy,state,runtime.lora)
                        with autocast(runtime.device):
                            row['action_generated_student'] = runtime.policy.predict_action_chunk(obs,
                                noise=ref['noise'].to(runtime.device),num_steps=10).float().cpu()[...,:7]
                            with residual_output(runtime.policy,gamma,z):
                                row['action_generated_reader'] = runtime.policy.predict_action_chunk(obs,
                                    noise=ref['noise'].to(runtime.device),num_steps=10).float().cpu()[...,:7]
                        runtime.restore_identity()
                    path = out/'functional'/f'{ident}_{label}.pt'
                    save_pt(path,row)
                    functional[label] = str(path)
                records.append(dict(condition_id=ident,task=task,teacher=teacher,factors=file_record(factorpath),
                    Z=file_record(zpath),functional=functional,raw_frames=cached[(task,teacher)]['raw_frames']))
    return records


def main(arm):
    started = time.monotonic()
    out = ROOT/arm
    out.mkdir(parents=True,exist_ok=True)
    prepare_libero_config(ROOT/'libero_config')
    identity = git_state(Path(__file__).resolve().parents[1])
    if identity['branch'] or not git_state_is_clean_pushed_or_frozen_authority(identity):
        raise ValueError('actual computation requires clean pushed detached source')
    torch.cuda.set_device(0)
    affinity = bind_current_process_to_cuda_numa(0)
    torch.manual_seed(7); torch.cuda.manual_seed(7); np.random.seed(7); random.seed(7)
    torch.backends.cuda.matmul.allow_tf32 = True
    spec = specification(CONTINUATION2340_SPEC_PATH)
    runtime = build_runtime(ASSET,spec,torch.device('cuda:0'),'T',evaluation=True)
    runtime.writer.load_state_dict(load_file(str(ECP),device='cuda:0'),strict=True)
    runtime.writer.common.requires_grad_(False)
    runtime.writer.train(); runtime.policy.eval(); runtime.restore_identity()
    gamma = CreditReader().to(runtime.device)
    data = FormalData(ASSET,spec,query_labels=True,task_ids=TASKS)
    manifest = read_json(OLD/'query_manifest.json')
    panel = read_json(PRIOR/'fixed_panels.json')
    if len(manifest['steps'])!=64 or any(tuple(panel[str(t)]['teachers'])!=TEACHERS[t] for t in TASKS):
        raise ValueError('fixed learning stream/teacher mapping changed')
    contract = dict(schema='ember_state_credit_v1',study=ROOT.name,arm=arm,git=identity,
        parent=file_record(ECP),parent_training_git='e2afbfd7c997e3f792921600608efa2fa3c1b25a',
        original_learning_git='092a0ae83080f85ddff0e4129c92fb42aa527c0c',
        original_reading_git='5313257c8070a792bbec0c6345c98376ed5cd1ea',
        query_manifest=file_record(OLD/'query_manifest.json'),query_offset=1,updates=64,
        optimizer=dict(kind='AdamW',lr=1e-4,betas=[.9,.95],eps=1e-8,weight_decay=1e-4,clip=1.,scheduler=None),
        condition_weight=1/8,source_trainable=0,aux_weight=1,aux_hidden_gradient=arm=='live',
        topology=dict(host=socket.gethostname(),world_size=1,device=os.environ['CUDA_VISIBLE_DEVICES'],affinity=affinity),
        gamma_seed=7,logical_queries_per_update=224,condition_query_uses=14336)
    write_json_atomic(out/'training_contract.json',contract)
    try:
        cached = native_features(runtime,data,create=arm=='live')
        write_json_atomic(out/'initial_reconstruction.json',dict(records=initial_readback(runtime,cached)))
        initial_writer = {k:v.cpu().clone() for k,v in runtime.writer.state_dict().items()}
        initial_gamma = {k:v.cpu().clone() for k,v in gamma.state_dict().items()}
        torch.manual_seed(7); torch.cuda.manual_seed(7); np.random.seed(7); random.seed(7)
        initial_rng = rng_state()
        profiles = []
        for micro in (14,28):
            runtime.writer.load_state_dict(initial_writer); gamma.load_state_dict(initial_gamma)
            restore_rng(initial_rng)
            opt,params = optimizer(runtime,gamma)
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            result = step(runtime,gamma,cached,data,manifest['steps'][0],arm,opt,params,micro)
            profiles.append(result)
            write_json_atomic(out/'profile.json',dict(disposable_updates=len(profiles),rows=profiles,initial_state_restored_before_each=True))
            del opt
        micro = min(profiles,key=lambda r:r['seconds'])['microbatch']
        runtime.writer.load_state_dict(initial_writer); gamma.load_state_dict(initial_gamma)
        restore_rng(initial_rng)
        opt,params = optimizer(runtime,gamma)
        del initial_writer,initial_gamma
        contract.update(microbatch=micro,profile=profiles,
            expected_training_seconds=64*next(r['seconds'] for r in profiles if r['microbatch']==micro),
            packing_reason='actual faster of registered full updates at micro14/28; 28 is all queries in one condition')
        write_json_atomic(out/'training_contract.json',contract)
        print(json.dumps(dict(event='profile_complete',arm=arm,microbatch=micro,
            expected_training_seconds=contract['expected_training_seconds'],profiles=profiles)),flush=True)
        if input().strip() != 'RUN64':
            raise RuntimeError('batch owner withheld formal run because of actual resource budget')
        for update,entry in enumerate(manifest['steps'],1):
            record = step(runtime,gamma,cached,data,entry,arm,opt,params,micro)
            with (out/'metrics.jsonl').open('a') as log:
                log.write(json.dumps(dict(update=update,**record))+'\n')
        checkpoint = out/'checkpoints/update64'
        checkpoint.mkdir(parents=True)
        save_file({k:v.cpu().contiguous() for k,v in runtime.writer.state_dict().items()},str(checkpoint/'writer.safetensors'))
        save_file({k:v.cpu().contiguous() for k,v in gamma.state_dict().items()},str(out/'gamma.safetensors'))
        save_pt(checkpoint/'trainer_state.pt',dict(schema=contract['schema'],writer=runtime.writer.state_dict(),
            gamma=gamma.state_dict(),optimizer=opt.state_dict(),scheduler=None,scaler=None,next_update=64,
            sampler=dict(manifest=str(OLD/'query_manifest.json'),cursor=64,condition_order=list(cached),query_offset=1),
            RNG=rng_state(),topology=contract['topology'],contract=contract))
        write_json_atomic(checkpoint/'checkpoint_manifest.json',dict(complete=True,next_update=64,schema=contract['schema'],
            files={p.name:file_record(p) for p in checkpoint.iterdir() if p.is_file()},gamma=file_record(out/'gamma.safetensors')))
        gamma.eval().requires_grad_(False); runtime.writer.eval().requires_grad_(False)
        records = functional_readback(runtime,gamma,cached,data,panel,out,micro)
        original = read_json(OLD/'parent/evaluation/teacher0/run_contract.json')
        keys = tuple((t['suite'],t['task_id']) for t in original['tasks'])
        bank = inspect_bank(manifest_path=OLD/'parent/bank/panel_teacher0.json',source=original['adapter']['source'],
            task_keys=keys,evaluation_role='development_train',require_formal=True,
            task_init_state_ids={(t['suite'],t['task_id']):tuple(t['init_state_ids']) for t in original['tasks']})
        bank = copy.deepcopy(bank)
        bank.pop('learning_limit_panel',None)
        bank.update(mode='state_coupled_credit',condition_factors='complete_A0_plus_S_B0_plus_M',conditions=records,
            state_coupled_credit=dict(study=ROOT.name,arm=arm,training_contract=file_record(out/'training_contract.json')))
        write_json_atomic(out/'bank/manifest.json',bank)
        bank['manifest'] = file_record(out/'bank/manifest.json')
        data.close(); del gamma,cached,opt,params
        runtime.writer = None  # No Writer/gamma/Z survives into the student consumer.
        gc.collect(); torch.cuda.empty_cache()
        with torch.no_grad(),autocast(runtime.device):
            rows = evaluate(ROOT,arm,runtime,bank,original)
        write_json_atomic(out/'completion.json',dict(complete=True,updates=64,rows=len(rows),full=12,compact=52,
            seconds=time.monotonic()-started,git=identity,source_loading_count=1,profiles=profiles,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30))
    finally:
        data.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('live','stop'),required=True)
    args=parser.parse_args()
    torch.set_num_threads(6)
    main(args.arm)
