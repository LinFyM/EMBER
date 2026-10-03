#!/usr/bin/env python3
"""Unique fixed64 two-arm coordinate diagnostic; no automatic scientific successor."""
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
from safetensors.torch import load_file,save_file
from ember.pi05_evaluation import rollout_shard  # canonical public import order
from ember.operator_writer.run import build_runtime
from ember.operator_writer.data import FormalData
from ember.operator_writer.role_labels import ROOT,ASSET,OLD,TEACHERS
from ember.operator_writer.role_learning import freeze_heads,native_cache,compile_fixed,step,GROUPS
from ember.operator_writer.role_readout import readout
from ember.operator_writer.role_evaluation import evaluate,TASK16_REFERENCE,TASK16_TEACHERS
from ember.pi05_assets import prepare_libero_config
from ember.pi05_eval_contract import git_state,git_state_is_clean_pushed_or_frozen_authority
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.writer.materialization import file_record
from ember.writer.runtime import autocast
from ember.writer.topology import bind_current_process_to_cuda_numa

PARENT=ROOT.parent/'conditional_A_reexpression_diagnostic_20261002'
ECP=ROOT.parent/'conditional_read_write_continuation900_20261002/conditional_read_write/train/attempts/continuation/checkpoints/macro_00000900/ecp.safetensors'


def rng_state():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())


def restore_rng(state):
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy']);random.setstate(state['python'])


def optimizer(parameters):
    return torch.optim.AdamW(parameters,lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=1e-4)


def reconstruction(runtime,cached):
    numerator=denominator=0.;rows=[]
    for (task,teacher),fixed in cached.items():
        state=compile_fixed(runtime,fixed)
        parent=load_file(str(PARENT/'Original/bank'/f'task{task:03d}_teacher{teacher:02d}.safetensors'),device='cuda:0')
        n=sum(float((state[k]-parent[k]).double().square().sum()) for k in state)
        d=sum(float(parent[k].double().square().sum()) for k in state)
        numerator+=n;denominator+=d;rows.append(dict(task=task,teacher=teacher,relative_L2=(n/max(d,1e-30))**.5))
    if max(r['relative_L2'] for r in rows)>.1:
        raise ValueError('material initial reconstruction mismatch, no dtype sweep authorized')
    return dict(rows=rows,aggregate_relative_L2=(numerator/max(denominator,1e-30))**.5,
                normal_BF16_TF32_reduction_differences_accepted=True)


def checkpoint(runtime,opt,contract):
    path=ROOT/contract['arm']/'checkpoint64';path.mkdir(exist_ok=False)
    save_file({k:v.cpu().contiguous() for k,v in runtime.writer.state_dict().items()},str(path/'writer.safetensors'))
    torch.save(dict(schema=contract['schema'],optimizer=opt.state_dict(),scheduler=None,scaler=None,next_update=64,
        sampler=dict(manifest=str(OLD/'query_manifest.json'),cursor=64,query_offset=1,
            condition_order=[[t,d] for t,ds in TEACHERS.items() for d in ds]),
        RNG=rng_state(),topology=contract['topology'],contract=contract),path/'trainer_state.pt')
    write_json_atomic(path/'checkpoint_manifest.json',dict(complete=True,schema=contract['schema'],next_update=64,
        files={p.name:file_record(p) for p in path.iterdir() if p.is_file()},Writer_stored_once=True))


@torch.no_grad()
def task16_banks(runtime,spec,out):
    data=FormalData(ASSET,spec,query_labels=False,task_ids=(16,),role='validation');records=[]
    try:
        for teacher in TASK16_TEACHERS:
            condition,raw,sampled=data.condition(runtime,16,teacher)
            state,_=runtime.compile(condition,frame_chunk=128)
            identifier=f'task016_teacher{teacher:02d}'
            factor=out/'bank'/f'{identifier}.safetensors'
            save_file({k:v.cpu().contiguous() for k,v in state.items()},str(factor))
            records.append(dict(condition_id=identifier,global_task_id=16,teacher_demo=teacher,
                factors=file_record(factor),raw_frames=raw,sampled_frames=sampled,native_reads=1,
                layouts_reuse_one_LoRA=True,teacher_privileged_fields_read=False))
    finally:data.close()
    return records


def main(arm,stage):
    started=time.monotonic();out=ROOT/arm;out.mkdir(exist_ok=True)
    identity=git_state(Path(__file__).resolve().parents[1])
    if identity['branch'] or not git_state_is_clean_pushed_or_frozen_authority(identity):
        raise ValueError('actual consumers require clean pushed detached code')
    prepare_libero_config(ROOT/'tmp/libero_config')
    torch.cuda.set_device(0);affinity=bind_current_process_to_cuda_numa(0)
    torch.manual_seed(7);torch.cuda.manual_seed(7);np.random.seed(7);random.seed(7)
    torch.backends.cuda.matmul.allow_tf32=True
    original=read_json(PARENT/'Original/evaluation/teacher0/run_contract.json')
    spec=read_json(Path(original['adapter']['spec']['path']))
    runtime=build_runtime(ASSET,spec,torch.device('cuda:0'),'conditional_read_write',evaluation=True)
    runtime.writer.load_state_dict(load_file(str(ECP),device='cuda:0'),strict=True)
    runtime.policy.eval();runtime.writer.train();parameters=freeze_heads(runtime.writer)
    data=FormalData(ASSET,spec,query_labels=True,task_ids=tuple(TEACHERS))
    manifest=read_json(OLD/'query_manifest.json');panels=read_json(Path(manifest['panel_source']))
    labels=torch.load(ROOT/'labels/role_labels.pt',map_location='cpu',weights_only=False)
    try:
        cached=native_cache(runtime,data,create=arm=='F' and stage=='all',frame_chunk=128)
        print(json.dumps(dict(event='native_ready',arm=arm)),flush=True)
        if stage=='all':
            if (out/'checkpoint64').exists():raise ValueError('fixed endpoint already exists; do not duplicate')
            contract=dict(schema='ember_role_coordinate_credit_training_v1',study=ROOT.name,arm=arm,git=identity,
                parent=file_record(ECP),parent_training_git='85919994aef11c17b49b7d0e70a2c110158bff61',
                original_reading_git='923ff89b60f4f8a269e5352d8d82189d72a9cd0a',
                source_training_git='b8ea00e9fbb86742ef076bac9dd35c5314cd5aed',
                native_manifest=file_record(ROOT/'native/manifest.json'),query_manifest=file_record(OLD/'query_manifest.json'),
                labels=file_record(ROOT/'labels/manifest.json'),query_offset=1,updates=64,condition_weight=1/8,
                condition_query_uses=14336,teacher_origin_geometry_uses=64*256 if arm=='G' else 0,
                self_geometry_query_uses=14336 if arm=='G' else 0,coordinate_scale_m=1,
                trainable_names=[n for n,p in runtime.writer.named_parameters() if p.requires_grad],
                frozen=['source','public_A0_B0','native','four_layer_interpreter'],source_trainable=0,
                action_dimensions=list(range(7)),unpenalized_padding=list(range(7,32)),
                teacher_geometry_weight=.5 if arm=='G' else 0,self_geometry_weight=.5 if arm=='G' else 0,
                direct_A_and_live_self_h_credit=True,compiled_M_A_dependency=True,teacher_geometry_replay_once=True,
                cached=['X','H','c','d','frame_indices'],recomputed=['A','S','K','delta_z','Value','M'],
                optimizer=dict(kind='AdamW',lr=1e-4,betas=[.9,.95],eps=1e-8,weight_decay=1e-4,clip=1,scheduler=None),
                topology=dict(host=socket.gethostname(),world_size=1,physical_device=os.environ['CUDA_VISIBLE_DEVICES'],affinity=affinity))
            write_json_atomic(out/'training_contract.json',contract)
            write_json_atomic(out/'initial_reconstruction.json',reconstruction(runtime,cached))
            initial={n:p.detach().cpu().clone() for n,p in runtime.writer.named_parameters() if p.requires_grad}
            torch.manual_seed(7);torch.cuda.manual_seed(7);np.random.seed(7);random.seed(7);initial_rng=rng_state()
            profiles=[]
            for micro in (14,28):
                with torch.no_grad():
                    for n,p in runtime.writer.named_parameters():
                        if n in initial:p.copy_(initial[n])
                restore_rng(initial_rng);opt=optimizer(parameters)
                gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                profile_started=time.monotonic()
                try:result=step(runtime,cached,data,labels,manifest['steps'][0],arm,opt,parameters,micro)
                except torch.cuda.OutOfMemoryError as error:
                    optimizer(parameters).zero_grad(set_to_none=True)
                    result=dict(microbatch=micro,failed=True,error=str(error),seconds=time.monotonic()-profile_started,
                        allocated_peak_GiB=torch.cuda.max_memory_allocated()/2**30,
                        reserved_peak_GiB=torch.cuda.max_memory_reserved()/2**30)
                profiles.append(result)
                write_json_atomic(out/'profile.json',dict(disposable_updates=len(profiles),rows=profiles,
                    initial_parameters_RNG_restored=True,only_registered_first_update=True))
                del opt
            usable=[p for p in profiles if not p.get('failed')]
            if not usable:raise RuntimeError('authorized profiles could not fit device capacity')
            selected=min(usable,key=lambda p:p['seconds']);micro=selected['microbatch']
            with torch.no_grad():
                for n,p in runtime.writer.named_parameters():
                    if n in initial:p.copy_(initial[n])
            restore_rng(initial_rng);opt=optimizer(parameters);del initial
            contract.update(microbatch=micro,expected_training_seconds=64*selected['seconds'],
                packing_reason='real full update throughput; max28 query times2 teacher suffixes exhausts task inputs; full longest51-frame teacher in one native chunk; independent arms parallel',profile=profiles)
            write_json_atomic(out/'training_contract.json',contract)
            print(json.dumps(dict(event='profile_complete',arm=arm,expected_training_seconds=contract['expected_training_seconds'],
                profiles=[{k:v for k,v in p.items() if k!='rows'} for p in profiles])),flush=True)
            if input().strip()!='RUN64':raise RuntimeError('actual budget admission did not permit64 updates')
            for update,entry in enumerate(manifest['steps'],1):
                result=step(runtime,cached,data,labels,entry,arm,opt,parameters,micro)
                with (out/'training_metrics.jsonl').open('a') as handle:handle.write(json.dumps(dict(update=update,**result))+'\n')
            checkpoint(runtime,opt,contract);del opt
            print(json.dumps(dict(event='checkpoint64_complete',arm=arm)),flush=True)
        else:
            contract=read_json(out/'training_contract.json')
            runtime.writer.load_state_dict(load_file(str(out/'checkpoint64/writer.safetensors'),device='cuda:0'),strict=True)
        runtime.writer.eval().requires_grad_(False)
        records=readout(runtime,data,labels,panels,cached,out)
        with autocast(runtime.device):records+=task16_banks(runtime,spec,out)
        bank=copy.deepcopy(original['adapter']);bank.pop('reexpression_panel',None)
        bank.update(mode='conditional_read_write',condition_factors='complete_A0_plus_S_B0_plus_M',conditions=records,
            checkpoint=str(out/'checkpoint64'),training_git=contract['git']['commit'],reading_git=identity['commit'],
            role_coordinate_credit=dict(study=ROOT.name,arm=arm),native_reading=read_json(TASK16_REFERENCE)['adapter']['native_reading'])
        task16=copy.deepcopy(read_json(TASK16_REFERENCE)['adapter']['tasks'][0])
        task16['episodes']=[dict(init_state_id=i,condition_id=f'task016_teacher{d:02d}',teacher_demo_indices=[d],video_ordinal=i)
                            for i,d in enumerate(TASK16_TEACHERS)]
        bank['tasks'].append(task16)
        write_json_atomic(out/'bank/manifest.json',bank);bank['manifest']=file_record(out/'bank/manifest.json')
        data.close();del cached,parameters,labels;runtime.writer=None;gc.collect();torch.cuda.empty_cache()
        original['git']=identity
        with torch.no_grad(),autocast(runtime.device):rows=evaluate(runtime,arm,bank,original)
        write_json_atomic(out/'completion.json',dict(complete=True,updates=64,rows=len(rows),full=8,compact=40,
            seconds=time.monotonic()-started,training_git=contract['git'],reading_git=identity,source_loading_count=1,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30))
        print(json.dumps(dict(event='arm_complete',arm=arm,rows=len(rows))),flush=True)
    finally:data.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--arm',choices=('F','G'),required=True)
    p.add_argument('--stage',choices=('all','readback'),default='all');args=p.parse_args()
    torch.set_num_threads(6);main(args.arm,args.stage)
