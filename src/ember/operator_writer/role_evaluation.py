"""Fixed96 cases through canonical episode/adapter/persistent environment owners."""
from __future__ import annotations
import copy
import os
import numpy as np
import torch
from ember.lora import LORA_A_SUFFIX
from ember.pi05_evaluation import rollout_shard,_validate_episode_row
from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_eval_queue import EvaluationShard
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.operator_writer.bank import FrozenOperatorAdapter,PreparedOperatorLoRA,episode_evidence
from .role_labels import ROOT,POINTS,TEACHERS
from .role_learning import FINAL
from .role_readout import capture_projection

TASK16_TEACHERS=(47,33,28,46,32,1,24,43)
TASK16_REFERENCE=ROOT.parent/'object_position_transport_20261004/evaluation/C900/original/run_contract.json'


def validate_cases(cases,task,state_ids,contract):
    registration=contract.get('role_coordinate_credit',{})
    t=task['global_task_id']
    expected=([(d,i,'original') for d in TEACHERS[t] for i in range(4)] if t!=16 else
              [(d,i,layout) for layout in ('original','swapped') for i,d in enumerate(TASK16_TEACHERS)])
    actual=[(c['evidence']['teacher'],c['evidence']['init_state_id'],c['evidence']['layout']) for c in cases]
    if (registration.get('study')!=ROOT.name or registration.get('arm') not in ('F','G')
            or actual!=expected or list(state_ids)!=[e[1] for e in expected]
            or len({c['evidence']['case_id'] for c in cases})!=len(expected)):
        raise ValueError('registered role-coordinate finite panel changed')


def evaluate(runtime,arm,bank,original):
    out=ROOT/arm;rows=[];forwards=[]
    contract=copy.deepcopy(original)
    contract.update(output_dir=str(out),role='development_train',mode='screen',analysis_only=True,adapter=bank,
                    role_coordinate_credit=dict(study=ROOT.name,arm=arm))
    contract['parallel'].update(envs_per_replica=16,physical_gpu_count=1,
        physical_gpu_ids=[int(os.environ['CUDA_VISIBLE_DEVICES'])],replicas_per_gpu=1,worker_count=1)
    runtime.restore_identity()
    adapter=FrozenOperatorAdapter(policy=runtime.policy,source=bank['source'],evaluation_adapter=bank,
        task_keys=tuple((t['suite'],t['task_id']) for t in bank['tasks']),device=runtime.device,
        require_formal=True,reuse_injected=True)
    predict_base=adapter.predict_action_chunk;last=None
    def predict(prepared,batch,**kwargs):
        nonlocal last
        matrices=torch.stack([adapter._state(item.key)[FINAL+LORA_A_SUFFIX][:3] for item in prepared])
        start,stop=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
        with capture_projection(runtime.policy,matrices) as (z,tau):
            result=predict_base(prepared,batch,**kwargs)
        stop.record();stop.synchronize()
        if len(z)!=10 or abs(float(tau[0][0])-1)>1e-6:
            raise ValueError('actual evaluator ten-flow projection capture changed')
        last=(torch.stack(z,1).numpy(),torch.stack(tau,1).numpy())
        forwards.append(dict(batch=len(prepared),seconds=start.elapsed_time(stop)/1000,
            allocated_GiB=torch.cuda.memory_allocated()/2**30,reserved_GiB=torch.cuda.memory_reserved()/2**30))
        return result
    def record(slots):
        if last[0].shape!=(len(slots),10,50,3):raise ValueError('actual passive projection row routing mismatch')
        for i,slot in enumerate(slots):
            trace=slot['passive_trace'];trace['role_projections'].append(last[0][i].copy())
            trace['role_taus'].append(last[1][i].copy());trace['role_replan_steps'].append(slot['steps'])
    adapter.predict_action_chunk=predict;adapter.record_role_prediction=record
    pool=PersistentTaskEnvironmentPool(contract,physical_gpu_id=int(os.environ['CUDA_VISIBLE_DEVICES']))
    try:
        for t in (32,20,16,12,0):
            base=read_json(TASK16_REFERENCE) if t==16 else original
            identity=next(x for x in bank['tasks'] if x['global_task_id']==t)
            task=copy.deepcopy(next(x for x in base['tasks'] if (x['suite'],x['task_id'])==(identity['suite'],identity['task_id'])))
            task['global_task_id']=t
            task['init_state_ids']=([i for d in TEACHERS[t] for i in range(4)] if t!=16 else list(range(8))*2)
            current=copy.deepcopy(contract)
            for key in ('environment','policy','rng','operator_read_write_scene'):
                if key in base:current[key]=copy.deepcopy(base[key])
            cases=[]
            planned=([(d,i,'original') for d in TEACHERS[t] for i in range(4)] if t!=16 else
                     [(d,i,l) for l in ('original','swapped') for i,d in enumerate(TASK16_TEACHERS)])
            for teacher,init,layout in planned:
                key=f'task{t:03d}_teacher{teacher:02d}'
                identifier=f'{arm}_{key}_init{init}_{layout}'
                full=((t in (0,12,20) and teacher==TEACHERS[t][0] and init==0) or
                      (t==32 and ((teacher==17 and init==2) or (teacher==43 and init in (2,3)))) or
                      (t==16 and init==0))
                evidence=dict(case_id=identifier,arm=arm,task=t,teacher=teacher,init_state_id=init,
                    layout=layout,condition_id=key,full_capture=full)
                case_contract=copy.deepcopy(current);path=out/'cases'/identifier
                case_contract['output_dir']=str(path)
                if t==16:
                    for field in ('scene_root','scene_manifest'):
                        case_contract['adapter'][field]=copy.deepcopy(base['adapter'][field])
                capture=case_contract['diagnostic_occupancy_capture']
                capture.update(mode='compact',full_conditions=[dict(suite=task['suite'],task_id=task['task_id'],
                    init_state_id=init)] if full else [],trajectory_root=str(path/'trajectories'))
                capture['passive_trace']['trace_root']=str(path/'continuous_traces')
                kind,name=POINTS[t] if t!=16 else ('body','butter_1_main')
                capture['role_coordinates']=dict(point_kind=kind,point_name=name,passive_In=t==16,
                    scale_m=1,prediction_dimensions=[0,1,2],actual_sampling='T+1_control_steps')
                if t==16:
                    case_contract['role_coordinate_layout']=dict(study=ROOT.name,arm=arm,layout=layout,
                        reference_contract=str(TASK16_REFERENCE),teacher=teacher,environment_steps_added=0)
                bank_task=next(x for x in case_contract['adapter']['tasks'] if x['global_task_id']==t)
                bank_task['episodes']=[dict(init_state_id=i,condition_id=key,teacher_demo_indices=[teacher],video_ordinal=0)
                    for i in (range(8) if t==16 else range(4))]
                episode=dict(init_state_id=init,condition_id=key,teacher_demo_indices=[teacher],video_ordinal=0)
                prepared=PreparedOperatorLoRA(key,episode_evidence(case_contract['adapter'],bank_task,episode))
                cases.append(dict(evidence=evidence,contract=case_contract,prepared_adapter=prepared))
                write_json_atomic(path/'run_contract.json',case_contract)
            validate_cases(cases,task,task['init_state_ids'],current)
            envs,initial_states=pool.switch(task)
            result=rollout_shard(envs=envs,init_states=initial_states,task=task,state_ids=tuple(task['init_state_ids']),
                contract=current,policy=runtime.policy,preprocess=runtime.processor,
                postprocess=runtime.processor.unnormalize_action,task_adapter=adapter,episode_contexts=cases)
            for row in result:
                case=next(c for c in cases if c['evidence']['case_id']==row['role_coordinate_case']['case_id'])
                shard=EvaluationShard(job_id=case['evidence']['case_id'],ordinal=0,suite=task['suite'],task_id=task['task_id'],
                    init_state_ids=(row['init_state_id'],),horizon=task['horizon'],estimated_cost=task['horizon'],preferred_gpu=None)
                _validate_episode_row(row,contract=case['contract'],shard=shard,task=task)
            rows.extend(result)
            write_json_atomic(out/'results.json',dict(rows=rows,forward_records=forwards,Writer_removed=True,
                source_loading_count=1,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30))
        if len(rows)!=48 or sum(r['role_coordinate_case']['full_capture'] for r in rows)!=8:
            raise ValueError('registered arm must produce48 rows and8 full')
    finally:
        pool.close();adapter.close()
    return rows
