"""Temporary fixed12-case/three-arm consumer of the canonical evaluator."""
from contextlib import contextmanager
from copy import deepcopy
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch

import ember.pi05_evaluation as evaluation
from ember.operator_writer.bank import PreparedOperatorLoRA, episode_evidence
from ember.pi05_assets import prepare_libero_config, write_json_atomic
from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_eval.worker_setup import load_policy, validate_worker_assets
from ember.writer.topology import bind_current_process_to_cuda_numa
from .visible_object_attention import VisibleObjectAdapter, historical_module, visible_coverage

ROOT = Path('/data1/user/ymdai/ember_runs/visible_object_readout_intervention_20261005')
REFERENCES = {
    'C900':Path('/data1/user/ymdai/ember_runs/task16_condition_context_crossover_20261004/run_contract.json'),
    'T2340':Path('/data1/user/ymdai/ember_runs/object_position_transport_20261004/evaluation/T2340/original/run_contract.json')}
ARMS = ('parent','route_butter','route_orange_juice')


def panel():
    cases = [dict(case_id=f'C900_scene{i}_teacher{j}_noise{k}',model='C900',
        physical_init_state_id=i,teacher_demo=j,noise_stream_init_state_id=k,layout='original',
        source_condition_init_state_id=0 if j == 47 else 46,condition_id=f'task_16_demos_{j}')
        for i,j,k in product((0,46),(47,40),(0,46))]
    cases.extend(dict(case_id=f'T2340_init{i}_{layout}',model='T2340',physical_init_state_id=i,
        teacher_demo=j,noise_stream_init_state_id=i,layout=layout,
        source_condition_init_state_id=i,condition_id=f'task_16_demos_{j}')
        for i,j in [(2,28),(4,32)] for layout in ('original','swapped'))
    return [{**c,'panel_ordinal':n} for n,c in enumerate(cases)]


def contracts(arm):
    originals = {k:json.loads(p.read_text()) for k,p in REFERENCES.items()}
    assert all(originals['C900'][k] == originals['T2340'][k] for k in ['model','policy','environment','rng'])
    norms = [v['normalization'] for v in originals.values()]
    assert {k:v for k,v in norms[0].items() if k!='path'} == {k:v for k,v in norms[1].items() if k!='path'}
    assert json.loads(Path(norms[0]['path']).read_text()) == json.loads(Path(norms[1]['path']).read_text())
    run = deepcopy(originals['C900'])
    run.pop('task16_condition_context_crossover',None)
    run['git'] = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty=False)
    run.update(schema_version='ember_visible_object_readout_intervention_v1',
        output_dir=str(ROOT/'evaluation'/arm),mode='frozen_diagnostic',role='privileged_analysis',
        contract_reference=dict(path=str(ROOT/'evaluation'/arm/'run_contract.json')),
        visible_object_readout_intervention=dict(arm=arm,cases=panel(),gradient_use=False,qualification=False),
        command=['scripts/visible_object_readout.py','--arm',arm])
    run['parallel'].update(envs_per_replica=12,worker_count=1,replicas_per_gpu=1,physical_gpu_count=1,
                           physical_gpu_ids=[int(os.environ['CUDA_VISIBLE_DEVICES'])])
    run['diagnostic_occupancy_capture'].update(mode='full',full_conditions=[],
        trajectory_root=str(ROOT/'evaluation'/arm/'trajectories'),
        passive_trace=dict(schema_version='ember_visible_object_readout_v1',trace_root=str(ROOT/'evaluation'/arm/'continuous')))
    run['diagnostic_stage_predicates'] = dict(schema_version='ember_pi05_stage_predicate_capture_v1',
        capture='all_rows_post_settling_then_every_executed_control_step',full_conditions_only=False)
    assert run['tasks'][0]['horizon'] == 280
    banks = {k:deepcopy(v['adapter']) for k,v in originals.items()}
    for model,bank in banks.items():
        keys = {c['condition_id'] for c in panel() if c['model']==model}
        bank['conditions'] = [c for c in bank['conditions'] if c['condition_id'] in keys]
        bank['tasks'] = [t for t in bank['tasks'] if (t['suite'],t['task_id'])==('libero_object',6)]
        assert {c['condition_id'] for c in bank['conditions']} == keys
        assert bank['source'] == run['model'] and bank['lora'] == banks['C900']['lora']
    return run,banks


@contextmanager
def case_context(adapter,run,banks,arm):
    from ember.pi05_eval import trajectory_capture
    xy = historical_module('object_position_transport',Path('/data1/user/ymdai/ember_runs/object_position_transport_20261004/frozen_run2'))
    start,noise,finish,plan,passive = (evaluation.start_fixed_episode,evaluation.make_policy_noise,
        evaluation.finish_episode_row,evaluation._plan_action_chunks,trajectory_capture.record_passive_step)
    cases = panel()

    def start_case(**args):
        case = cases[int(args['init_state_id'])]
        bank = banks[case['model']]
        task = bank['tasks'][0]
        ep = next(e for e in task['episodes'] if e['init_state_id']==case['source_condition_init_state_id'])
        assert ep['condition_id']==case['condition_id'] and ep['teacher_demo_indices']==[case['teacher_demo']]
        prepared = PreparedOperatorLoRA(case['case_id'],episode_evidence(bank,task,ep))
        args.update(init_state_id=case['physical_init_state_id'],
                    task_adapter=SimpleNamespace(prepare_episode=lambda **kw:prepared))
        local = deepcopy(args['contract'])
        local['output_dir'] = str(ROOT/'cases'/arm/case['case_id'])
        local['operator_read_write_scene'] = json.loads(REFERENCES[case['model']].read_text())['operator_read_write_scene']
        args['contract'] = local
        slot = start(**args)
        slot['readout_case'],slot['readout_env'] = case,args['env']
        if case['model']=='T2340':
            local['object_position_transport'] = dict(model='T2340',layout=case['layout'],schema_version=xy.TAG)
            slot['obs'],slot['object_position_transport'] = xy.apply(args['env'],slot['obs'],local,case['physical_init_state_id'])
            # Existing trace/capture began before the physical swap: reset the initial sample only.
        states = (('In','butter_1','basket_1_contain_region'),('In','orange_juice_1','basket_1_contain_region'))
        slot['stage_predicate_states'] = states
        initial = tuple(bool(args['env'].env._eval_predicate(s)) for s in states)
        slot.update(stage_predicate_last=initial,stage_predicate_ever=initial,
                    stage_predicate_peak=sum(initial),stage_predicate_transitions=[])
        trajectory_capture.start_passive_trace(args['env'],slot,local['diagnostic_occupancy_capture'])
        return slot

    def trace(env,slot,action,capture):
        passive(env,slot,action,capture)
        if 'readout_case' in slot:
            registry = slot['passive_trace']['body_registry']
            slot.setdefault('readout_body_quaternions',[]).append(np.stack([
                env.env.sim.data.body_xquat[b['body_id']].copy() for b in registry]))

    def paired_noise(slots,**kw):
        return noise([dict(s,init_state_id=s['readout_case']['noise_stream_init_state_id']) for s in slots],**kw)

    def plan_case(slots,**kw):
        planning = [s for s in slots if s is not None and not s['action_plan'] and not s.get('prefix_terminal',False)]
        if planning:
            coverage,names = [],None
            for slot in planning:
                first = slot['replan_index']==0
                ns,f,info = visible_coverage(slot['readout_env'],slot['obs'],verify=first)
                assert names is None or names==ns
                names = ns
                coverage.append(f)
                if first:
                    output = ROOT/'cases'/arm/slot['readout_case']['case_id'];output.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(output/'first_visible_coverage.npz',coverage=f.numpy(),
                        entity_names=np.asarray(ns),canonical_rgb=info.pop('pixels'),raw_masks=info.pop('raw_masks'))
                    write_json_atomic(output/'segmentation_mapping.json',info)
            adapter.set_visibility(planning,names,coverage)
        return plan(slots,**kw)

    def finish_case(**args):
        slot,case = args['slot'],args['slot']['readout_case']
        root = ROOT/'cases'/arm/case['case_id'];root.mkdir(parents=True,exist_ok=True)
        local = deepcopy(args['contract'])
        local['diagnostic_occupancy_capture']['trajectory_root'] = str(root/'trajectory')
        local['diagnostic_occupancy_capture']['passive_trace']['trace_root'] = str(root/'continuous')
        args['contract'] = local
        row = finish(**args)
        quats = np.stack(slot['readout_body_quaternions'])
        assert len(quats)==slot['steps']+1
        np.savez_compressed(root/'body_quaternions.npz',body_quaternions=quats)
        row.update(case,arm=arm,attention_effects=adapter.save_effects(case['case_id'],root),
            body_quaternions=str(root/'body_quaternions.npz'))
        if 'object_position_transport' in slot:row['object_position_transport']=slot['object_position_transport']
        write_json_atomic(root/'results.json',row)
        return row

    evaluation.start_fixed_episode,evaluation.make_policy_noise,evaluation.finish_episode_row,evaluation._plan_action_chunks = start_case,paired_noise,finish_case,plan_case
    trajectory_capture.record_passive_step,evaluation.record_passive_step = trace,trace
    try:yield
    finally:
        evaluation.start_fixed_episode,evaluation.make_policy_noise,evaluation.finish_episode_row,evaluation._plan_action_chunks = start,noise,finish,plan
        trajectory_capture.record_passive_step,evaluation.record_passive_step = passive,passive


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS,required=True)
    args=parser.parse_args();out=ROOT/'evaluation'/args.arm
    assert not (out/'completion.json').exists()
    assert subprocess.check_output(['git','status','--porcelain'],text=True)==''
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==''
    out.mkdir(parents=True,exist_ok=True)
    run,banks=contracts(args.arm);write_json_atomic(out/'run_contract.json',run)
    torch.set_num_threads(8);torch.manual_seed(7);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    affinity=bind_current_process_to_cuda_numa(0);assert affinity is not None
    os.environ['EMBER_LIBERO_ASSETS_ROOT']=run['libero_paths']['assets']
    prepare_libero_config(ROOT/'cache'/args.arm/'libero_config')
    started=time.monotonic();adapter=pool=None
    try:
        model,norm,tokenizer=validate_worker_assets(run)
        policy,preprocess,postprocess=load_policy(model,norm['stats'],tokenizer,run['policy'])
        adapter=VisibleObjectAdapter(policy=policy,source=run['model'],evaluation_adapter=banks['C900'],
            task_keys=(('libero_object',6),),device=torch.device('cuda:0'),require_formal=True,
            banks=banks,cases=panel(),arm=args.arm)
        pool=PersistentTaskEnvironmentPool(run,physical_gpu_id=int(os.environ['CUDA_VISIBLE_DEVICES']))
        envs,init_states=pool.switch(run['tasks'][0])
        rollout=time.monotonic()
        with case_context(adapter,run,banks,args.arm):
            rows=evaluation.rollout_shard(envs=envs,init_states=init_states,task=run['tasks'][0],
                state_ids=list(range(12)),contract=run,policy=policy,preprocess=preprocess,
                postprocess=postprocess,task_adapter=adapter)
        assert len(rows)==12 and all(r['occupancy_trajectory']['capture_level']=='full' for r in rows)
        write_json_atomic(out/'results.json',dict(episodes=sorted(rows,key=lambda r:r['panel_ordinal'])))
        write_json_atomic(out/'completion.json',dict(complete=True,rows=12,full=12,arm=args.arm,
            reading_git=run['git']['commit'],model_loads=1,rollout_seconds=time.monotonic()-rollout,
            total_seconds=time.monotonic()-started,packing=adapter.packing,max_actual_batch=adapter.max_inference_batch,
            allocated_peak_GiB=torch.cuda.max_memory_allocated()/2**30,reserved_peak_GiB=torch.cuda.max_memory_reserved()/2**30,
            GPU_local_CPU_affinity=affinity,source_bank_copies=0,teacher_privileged_reads=0,model_gradients=0))
    finally:
        if adapter is not None:adapter.close()
        if pool is not None:pool.close()
        torch.cuda.empty_cache()


if __name__=='__main__':main()
