#!/usr/bin/env python3
"""Unique bounded §116 B-only learning and fixed readback; retired on delivery."""
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

from ember.pi05_evaluation import rollout_shard  # Canonical facade initializes internal imports.
from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX
from ember.operator_writer.run import build_runtime
from ember.operator_writer.data import FormalData
from ember.operator_writer.bank import inspect_bank
from ember.operator_writer.effect_labels import ROOT, ASSET, OLD, TEACHERS
from ember.operator_writer.effect_learning import (freeze_except_B, native_cache, compile_B, step, B_GROUPS)
from ember.operator_writer.effect_readout import functional
from ember.operator_writer.effect_evaluation import evaluate
from ember.pi05_assets import prepare_libero_config
from ember.pi05_eval_contract import git_state, git_state_is_clean_pushed_or_frozen_authority
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record
from ember.writer.runtime import autocast
from ember.writer.topology import bind_current_process_to_cuda_numa

PARENT_ROOT = ROOT.parent/'conditional_A_reexpression_diagnostic_20261002'
ECP = ROOT.parent/'conditional_read_write_continuation900_20261002/conditional_read_write/train/attempts/continuation/checkpoints/macro_00000900/ecp.safetensors'


def rng_state():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy']); random.setstate(state['python'])


def optimizer(parameters):
    return torch.optim.AdamW(parameters,lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=1e-4)


def reconstruction(runtime,cached):
    numerator, denominator, maximum = 0.,0.,0.
    records = []
    for (task,teacher),fixed in cached.items():
        state = compile_B(runtime,fixed)
        old = load_file(str(PARENT_ROOT/'Original/bank'/f'task{task:03d}_teacher{teacher:02d}.safetensors'),device='cuda:0')
        row = dict(task=task,teacher=teacher)
        for label,suffix in [('A',LORA_A_SUFFIX),('B',LORA_B_SUFFIX)]:
            n = sum(float((state[k]-old[k]).double().square().sum()) for k in state if k.endswith(suffix))
            d = sum(float(old[k].double().square().sum()) for k in state if k.endswith(suffix))
            row[label+'_relative_L2'] = (n/max(d,1e-30))**.5
            numerator += n; denominator += d
        maximum = max(maximum,row['A_relative_L2'],row['B_relative_L2']); records.append(row)
    if maximum > .1:
        raise ValueError('material initial parent reconstruction discrepancy; no precision sweep authorized')
    return dict(records=records,aggregate_relative_L2=(numerator/max(denominator,1e-30))**.5,
                normal_BF16_TF32_and_reduction_variation_accepted=True)


def save_checkpoint(runtime,opt,contract):
    path = ROOT/contract['arm']/'checkpoint64'; path.mkdir(exist_ok=False)
    save_file({k:v.cpu().contiguous() for k,v in runtime.writer.state_dict().items()},str(path/'writer.safetensors'))
    state = dict(schema=contract['schema'],writer_file=file_record(path/'writer.safetensors'),
        optimizer=opt.state_dict(),scheduler=None,scaler=None,next_update=64,
        sampler=dict(manifest=str(OLD/'query_manifest.json'),cursor=64,query_offset=1,
                     condition_order=[[t,d] for t,ds in TEACHERS.items() for d in ds]),
        RNG=rng_state(),topology=contract['topology'],contract=contract)
    torch.save(state,path/'trainer_state.pt')
    write_json_atomic(path/'checkpoint_manifest.json',dict(complete=True,schema=contract['schema'],next_update=64,
        files={p.name:file_record(p) for p in path.iterdir() if p.is_file()},Writer_stored_once=True))


def main(arm,stage):
    started = time.monotonic(); out = ROOT/arm; out.mkdir(exist_ok=True)
    identity = git_state(Path(__file__).resolve().parents[1])
    if identity['branch'] or not git_state_is_clean_pushed_or_frozen_authority(identity):
        raise ValueError('actual consumer must use clean pushed detached source')
    prepare_libero_config(ROOT/'libero_config')
    torch.cuda.set_device(0); affinity = bind_current_process_to_cuda_numa(0)
    torch.manual_seed(7); torch.cuda.manual_seed(7); np.random.seed(7); random.seed(7)
    torch.backends.cuda.matmul.allow_tf32 = True
    original = read_json(PARENT_ROOT/'Original/evaluation/teacher0/run_contract.json')
    spec = read_json(Path(original['adapter']['spec']['path']))
    runtime = build_runtime(ASSET,spec,torch.device('cuda:0'),'conditional_read_write',evaluation=True)
    runtime.writer.load_state_dict(load_file(str(ECP),device='cuda:0'),strict=True)
    runtime.policy.eval(); runtime.writer.train(); parameters = freeze_except_B(runtime.writer)
    data = FormalData(ASSET,spec,query_labels=True,task_ids=tuple(TEACHERS))
    manifest = read_json(OLD/'query_manifest.json'); panels = read_json(Path(manifest['panel_source']))
    positions = torch.load(ROOT/'labels/query_positions.pt',map_location='cpu',weights_only=False)
    try:
        if arm=='J' and stage=='all':
            # This fixed reader is independent of native preparation; do useful work on the second device.
            parent_states={(t,d):load_file(str(PARENT_ROOT/'Original/bank'/f'task{t:03d}_teacher{d:02d}.safetensors'),device='cuda:0')
                           for t,ds in TEACHERS.items() for d in ds}
            functional(runtime,data,positions,panels,None,ROOT/'parent',28,arm='parent',states=parent_states)
            del parent_states
            print(json.dumps(dict(event='parent_functional_complete',arm=arm)),flush=True)
            if input().strip()!='NATIVE_READY':raise RuntimeError('native preparation did not complete')
        cached = native_cache(runtime,data,create=arm=='A' and stage=='all',frame_chunk=128)
        print(json.dumps(dict(event='native_ready',arm=arm,cache_records=len(cached))),flush=True)
        if stage == 'all':
            if (out/'checkpoint64').exists():
                raise RuntimeError('complete fixed64 point exists; do not repeat training')
            contract = dict(schema='ember_joint_action_effect_training_v1',study=ROOT.name,arm=arm,git=identity,
                parent=file_record(ECP),parent_training_git='a0e0248d96568e42b24a3d1c4102e2ca6e35a40c',
                source_training_git='b8ea00e9fbb86742ef076bac9dd35c5314cd5aed',native_manifest=file_record(ROOT/'native/manifest.json'),
                query_manifest=file_record(OLD/'query_manifest.json'),labels=file_record(ROOT/'labels/manifest.json'),
                query_offset=1,updates=64,condition_weight=1/8,condition_query_uses=14336,
                trainable_names=[n for n,p in runtime.writer.named_parameters() if p.requires_grad],
                frozen=['source','common_A0_B0','native','interpreter','all_A_heads'],source_trainable=0,
                action_coordinates=list(range(7)),effect_coordinates=list(range(7,10)) if arm=='J' else [],
                unpenalized_padding=list(range(10 if arm=='J' else 7,32)),effect_weight=1 if arm=='J' else 0,
                optimizer=dict(kind='AdamW',lr=1e-4,betas=[.9,.95],eps=1e-8,weight_decay=1e-4,clip=1.,scheduler=None),
                topology=dict(host=socket.gethostname(),world_size=1,physical_device=os.environ['CUDA_VISIBLE_DEVICES'],affinity=affinity))
            write_json_atomic(out/'training_contract.json',contract)
            write_json_atomic(out/'initial_reconstruction.json',reconstruction(runtime,cached))
            initial = {n:p.detach().cpu().clone() for n,p in runtime.writer.named_parameters() if p.requires_grad}
            torch.manual_seed(7);torch.cuda.manual_seed(7);np.random.seed(7);random.seed(7);initial_rng=rng_state()
            profiles = []
            for micro in (14,28):
                with torch.no_grad():
                    for n,p in runtime.writer.named_parameters():
                        if n in initial:p.copy_(initial[n])
                restore_rng(initial_rng);opt=optimizer(parameters)
                gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                try:
                    result = step(runtime,cached,data,positions,manifest['steps'][0],arm,opt,parameters,micro)
                except torch.cuda.OutOfMemoryError as error:
                    result = dict(microbatch=micro,failed=True,error=str(error),seconds=None,
                        allocated_peak_GiB=torch.cuda.max_memory_allocated()/2**30,
                        reserved_peak_GiB=torch.cuda.max_memory_reserved()/2**30)
                profiles.append(result)
                write_json_atomic(out/'profile.json',dict(disposable_updates=len(profiles),rows=profiles,
                    same_initial_parameters_RNG_restored=True,only_registered_first_update=True))
                del opt
            usable = [p for p in profiles if not p.get('failed')]
            if not usable:raise RuntimeError('both authorized packing profiles exceeded physical capacity')
            micro = min(usable,key=lambda p:p['seconds'])['microbatch']
            with torch.no_grad():
                for n,p in runtime.writer.named_parameters():
                    if n in initial:p.copy_(initial[n])
            restore_rng(initial_rng);opt=optimizer(parameters);del initial
            contract.update(microbatch=micro,profile=profiles,expected_training_seconds=64*min(p['seconds'] for p in usable),
                packing_reason='measured faster full update; two teacher conditions vmapped at same query prefix; micro28 packs56 suffix queries and exhausts that task logical inputs; independent arms parallel')
            write_json_atomic(out/'training_contract.json',contract)
            print(json.dumps(dict(event='profile_complete',arm=arm,expected_training_seconds=contract['expected_training_seconds'],
                profiles=[{k:v for k,v in p.items() if k!='rows'} for p in profiles])),flush=True)
            if input().strip()!='RUN64':raise RuntimeError('owner withheld run because actual cost cannot fit budget')
            for update,entry in enumerate(manifest['steps'],1):
                result=step(runtime,cached,data,positions,entry,arm,opt,parameters,micro)
                with (out/'training_metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(update=update,**result))+'\n')
            save_checkpoint(runtime,opt,contract);del opt
            print(json.dumps(dict(event='checkpoint64_complete',arm=arm)),flush=True)
        else:
            contract=read_json(out/'training_contract.json');micro=contract['microbatch']
            runtime.writer.load_state_dict(load_file(str(out/'checkpoint64/writer.safetensors'),device='cuda:0'),strict=True)
        runtime.writer.eval().requires_grad_(False)
        records=functional(runtime,data,positions,panels,cached,out,micro,arm=arm)
        keys=tuple((t['suite'],t['task_id']) for t in original['tasks'])
        bank=inspect_bank(manifest_path=PARENT_ROOT/'Original/bank/panel_teacher0.json',source=original['adapter']['source'],
            task_keys=keys,evaluation_role='development_train',require_formal=True,
            task_init_state_ids={key:(0,1,2,3) for key in keys})
        bank=copy.deepcopy(bank);bank.pop('reexpression_panel',None)
        bank.update(mode='joint_action_effect_credit',condition_factors='complete_A0_plus_S_B0_plus_M',conditions=records,
            checkpoint=str(out/'checkpoint64'),training_git=contract['git']['commit'],reading_git=identity['commit'],
            joint_action_effect_credit=dict(study=ROOT.name,arm=arm,training_contract=file_record(out/'training_contract.json')))
        write_json_atomic(out/'bank/manifest.json',bank);bank['manifest']=file_record(out/'bank/manifest.json')
        data.close();del cached,parameters,positions;runtime.writer=None;gc.collect();torch.cuda.empty_cache()
        with torch.no_grad(),autocast(runtime.device):rows=evaluate(runtime,arm,bank,original)
        write_json_atomic(out/'completion.json',dict(complete=True,updates=64,rows=len(rows),full=6,compact=26,
            seconds=time.monotonic()-started,training_git=contract['git'],reading_git=identity,source_loading_count=1,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30))
        print(json.dumps(dict(event='arm_complete',arm=arm,rows=len(rows))),flush=True)
    finally:
        data.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=('A','J'),required=True)
    p.add_argument('--stage',choices=('all','readback'),default='all');args=p.parse_args()
    torch.set_num_threads(6);main(args.arm,args.stage)
