"""Bounded 32-row arm on the canonical evaluator and existing paired scenes."""
from __future__ import annotations
import copy
import os
from pathlib import Path

import numpy as np
import torch

from ember.pi05_evaluation import rollout_shard, _validate_episode_row
from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_eval_queue import EvaluationShard
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import write_json_atomic
from ember.operator_writer.bank import FrozenOperatorAdapter, PreparedOperatorLoRA, episode_evidence
from .effect_labels import ROOT, POINTS, TEACHERS
from .effect_readout import capture_full_sample


def validate_cases(cases, task, state_ids, contract):
    registration = contract.get('joint_action_effect_cases', {})
    global_task = int(task['global_task_id'])
    expected = [(teacher, init) for teacher in TEACHERS[global_task] for init in range(4)]
    if (registration.get('study') != ROOT.name or registration.get('arm') not in ('A','J')
            or len(cases) != 8 or list(state_ids) != [s for _,s in expected]
            or [(c['evidence']['teacher'],c['evidence']['init_state_id']) for c in cases] != expected
            or any(c['evidence']['task'] != global_task for c in cases)
            or len({c['evidence']['case_id'] for c in cases}) != 8):
        raise ValueError('joint-effect exact finite task/teacher/init pairing changed')


def evaluate(runtime, arm, bank, original):
    out = ROOT / arm
    contract = copy.deepcopy(original)
    contract.update(git=git_state(Path(__file__).resolve().parents[3]), output_dir=str(out),
                    role='development_train', mode='screen', analysis_only=True, adapter=bank)
    contract['parallel'].update(envs_per_replica=8, physical_gpu_count=1,
        physical_gpu_ids=[int(os.environ['CUDA_VISIBLE_DEVICES'])], replicas_per_gpu=1, worker_count=1)
    runtime.restore_identity()
    adapter = FrozenOperatorAdapter(policy=runtime.policy,source=bank['source'],evaluation_adapter=bank,
        task_keys=tuple((t['suite'],t['task_id']) for t in bank['tasks']),device=runtime.device,
        require_formal=True,reuse_injected=True)
    rows, forwards = [], []
    base_predict = adapter.predict_action_chunk
    last_effect = None

    def predict(prepared, batch, **kwargs):
        nonlocal last_effect
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        with capture_full_sample(runtime.policy) as sampled:
            result = base_predict(prepared,batch,**kwargs)
        stop.record(); stop.synchronize()
        if len(sampled) != 1:
            raise ValueError('one actual native integration per paired replan')
        last_effect = sampled[0][...,7:10].float().cpu().numpy() if arm == 'J' else None
        forwards.append(dict(batch=len(prepared),seconds=start.elapsed_time(stop)/1000,
            allocated_GiB=torch.cuda.memory_allocated()/2**30,reserved_GiB=torch.cuda.memory_reserved()/2**30))
        return result

    def record_prediction(slots):
        if last_effect is None or last_effect.shape != (len(slots),50,3):
            raise ValueError('passive joint-effect prediction mismatched actual replan')
        for index, slot in enumerate(slots):
            trace = slot['passive_trace']
            trace['displacement_predictions'].append(last_effect[index].copy())
            trace['displacement_replan_steps'].append(int(slot['steps']))

    adapter.predict_action_chunk = predict
    if arm == 'J':
        adapter.record_effect_prediction = record_prediction
    pool = PersistentTaskEnvironmentPool(contract, physical_gpu_id=int(os.environ['CUDA_VISIBLE_DEVICES']))
    try:
        # Same canonical long-first task queue, one source and persistent task workers.
        for global_task in (32,20,12,0):
            task = copy.deepcopy(next(t for t in contract['tasks'] if t['global_task_id'] == global_task))
            task['init_state_ids'] = [i for _ in range(2) for i in range(4)]
            current = copy.deepcopy(contract)
            cases = []
            for slot, teacher in enumerate(TEACHERS[global_task]):
                for init in range(4):
                    key = f'task{global_task:03d}_teacher{teacher:02d}'
                    identifier = f'{arm}_{key}_init{init}'
                    full = (global_task != 32 and slot == 0 and init == 0) or (global_task == 32 and
                        ((teacher == 17 and init == 2) or (teacher == 43 and init in (2,3))))
                    evidence = dict(case_id=identifier,arm=arm,task=global_task,teacher=teacher,
                        init_state_id=init,condition_id=key,full_capture=full)
                    per_case = copy.deepcopy(current)
                    path = out/'cases'/identifier; per_case['output_dir'] = str(path)
                    capture = per_case['diagnostic_occupancy_capture']
                    capture.update(mode='compact',full_conditions=[dict(suite=task['suite'],
                        task_id=task['task_id'],init_state_id=init)] if full else [],trajectory_root=str(path/'trajectories'))
                    capture['passive_trace']['trace_root'] = str(path/'continuous_traces')
                    if arm == 'J':
                        kind,name = POINTS[global_task]
                        capture['joint_effect'] = dict(point_kind=kind,point_name=name,scale_m=.1,
                            prediction_dimensions=[7,8,9],actual_sampling='T+1_actual_control_steps')
                    bank_task = next(t for t in per_case['adapter']['tasks'] if t['global_task_id']==global_task)
                    bank_task['episodes'] = [dict(init_state_id=i,condition_id=key,
                        teacher_demo_indices=[teacher],video_ordinal=slot) for i in range(4)]
                    prepared = PreparedOperatorLoRA(key,episode_evidence(bank,bank_task,bank_task['episodes'][init]))
                    cases.append(dict(evidence=evidence,contract=per_case,prepared_adapter=prepared))
            current['joint_action_effect_cases'] = dict(study=ROOT.name,arm=arm,case_ids=[c['evidence']['case_id'] for c in cases])
            for case in cases:
                case['contract']['joint_action_effect_cases'] = current['joint_action_effect_cases']
                write_json_atomic(Path(case['contract']['output_dir'])/'run_contract.json',case['contract'])
            envs, initial_states = pool.switch(task)
            result = rollout_shard(envs=envs,init_states=initial_states,task=task,state_ids=tuple(task['init_state_ids']),
                contract=current,policy=runtime.policy,preprocess=runtime.processor,
                postprocess=runtime.processor.unnormalize_action,task_adapter=adapter,episode_contexts=cases)
            for row in result:
                case = next(c for c in cases if c['evidence']['case_id']==row['joint_action_effect_case']['case_id'])
                shard = EvaluationShard(job_id=case['evidence']['case_id'],ordinal=0,suite=task['suite'],task_id=task['task_id'],
                    init_state_ids=(row['init_state_id'],),horizon=task['horizon'],estimated_cost=task['horizon'],preferred_gpu=None)
                _validate_episode_row(row,contract=case['contract'],shard=shard,task=task)
            rows.extend(result)
            write_json_atomic(out/'results.json',dict(rows=rows,forward_records=forwards,
                Writer_removed=True,source_loading_count=1,consumer_git=contract['git'],
                peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
        if len(rows)!=32 or sum(r['joint_action_effect_case']['full_capture'] for r in rows)!=6:
            raise ValueError('registered arm requires exact32 rows and6 full')
    finally:
        pool.close(); adapter.close()
    return rows
