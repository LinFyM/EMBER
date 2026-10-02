"""Task-only frozen readback matrix on the canonical environment consumer."""
import copy
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

# Canonical facade must initialize before its internal helpers.
from ember.pi05_evaluation import rollout_shard, _validate_episode_row
from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_eval_queue import EvaluationShard
from ember.pi05_eval_contract import git_state
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.operator_writer.bank import FrozenOperatorAdapter, PreparedOperatorLoRA, episode_evidence
from ember.operator_writer.state_credit_diagnostic import CreditReader, residual_output

TASKS = (32, 20, 12, 0)
TEACHERS = {0: (40, 11), 12: (25, 14), 20: (38, 42), 32: (17, 43)}
STUDY = 'state_coupled_functional_credit_20261002'


def condition_id(task, teacher):
    return f'task{task:03d}_teacher{teacher:02d}'


def evaluate(root, arm, runtime, bank, original):
    out = root / arm
    contract = copy.deepcopy(original)
    contract.update(git=git_state(Path(__file__).resolve().parents[3]), output_dir=str(out),
                    role='development_train', mode='screen', analysis_only=True, adapter=bank)
    contract['parallel'].update(envs_per_replica=8, physical_gpu_count=1,
        physical_gpu_ids=[int(os.environ['CUDA_VISIBLE_DEVICES'])], replicas_per_gpu=1, worker_count=1)
    runtime.restore_identity()
    adapter = FrozenOperatorAdapter(policy=runtime.policy, source=bank['source'], evaluation_adapter=bank,
        task_keys=tuple((t['suite'], t['task_id']) for t in bank['tasks']), device=runtime.device,
        require_formal=True, reuse_injected=True)
    rows, forwards = [], []
    base_predict = adapter.predict_action_chunk
    gamma, z_values = None, {}

    def predict(prepared, batch, **kwargs):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        is_reader = [p.evidence['consumer'] == 'reader' for p in prepared]
        if any(is_reader):
            if not all(is_reader) or gamma is None:
                raise ValueError('student runtime must have no Reader/gamma/Z')
            z = torch.stack([z_values[p.key] for p in prepared])
            with residual_output(runtime.policy, gamma, z):
                result = base_predict(prepared, batch, **kwargs)
        else:
            if gamma is not None or z_values:
                raise ValueError('student runtime retained gamma/Z')
            result = base_predict(prepared, batch, **kwargs)
        stop.record()
        stop.synchronize()
        forwards.append(dict(consumer='reader' if any(is_reader) else 'student', batch=len(prepared),
            seconds=start.elapsed_time(stop)/1000, allocated_GiB=torch.cuda.memory_allocated()/2**30,
            reserved_GiB=torch.cuda.memory_reserved()/2**30))
        return result

    adapter.predict_action_chunk = predict
    pool = PersistentTaskEnvironmentPool(contract, physical_gpu_id=int(os.environ['CUDA_VISIBLE_DEVICES']))
    try:
        for consumer in ('student', 'reader'):
            if consumer == 'reader':
                gamma = CreditReader().to(runtime.device).eval().requires_grad_(False)
                gamma.load_state_dict(load_file(str(out/'gamma.safetensors'), device='cuda:0'), strict=True)
                z_values = {r['condition_id']: torch.load(r['Z']['path'], map_location='cuda:0',
                    weights_only=True) for r in bank['conditions']}
            for global_task in TASKS:
                task = copy.deepcopy(next(t for t in contract['tasks'] if t['global_task_id'] == global_task))
                task['init_state_ids'] = [i for _ in range(2) for i in range(4)]
                cases = []
                current = copy.deepcopy(contract)
                for slot, teacher in enumerate(TEACHERS[global_task]):
                    for init in range(4):
                        identifier = f'{arm}_{consumer}_{condition_id(global_task,teacher)}_init{init}'
                        full = (init == 0 and slot == 0) or (global_task == 32 and teacher == 43 and init in (2, 3))
                        evidence = dict(case_id=identifier, arm=arm, consumer=consumer, task=global_task,
                            teacher=teacher, init_state_id=init, condition_id=condition_id(global_task, teacher),
                            full_capture=full, deployment_form=consumer == 'student')
                        per_case = copy.deepcopy(current)
                        path = out/'cases'/identifier
                        per_case['output_dir'] = str(path)
                        capture = per_case['diagnostic_occupancy_capture']
                        capture.update(mode='compact', full_conditions=[dict(suite=task['suite'],
                            task_id=task['task_id'], init_state_id=init)] if full else [],
                            trajectory_root=str(path/'trajectories'))
                        capture['passive_trace']['trace_root'] = str(path/'continuous_traces')
                        bank_task = next(t for t in per_case['adapter']['tasks'] if t['global_task_id'] == global_task)
                        bank_task['episodes'] = [dict(init_state_id=i, condition_id=evidence['condition_id'],
                            teacher_demo_indices=[teacher], video_ordinal=slot) for i in range(4)]
                        ep = bank_task['episodes'][init]
                        prepared = PreparedOperatorLoRA(evidence['condition_id'],
                            {**episode_evidence(bank, bank_task, ep), 'consumer': consumer})
                        cases.append(dict(evidence=evidence, contract=per_case, prepared_adapter=prepared))
                current['state_coupled_credit_cases'] = dict(study=STUDY,
                    case_ids=[c['evidence']['case_id'] for c in cases], physical_init_states=4,
                    registered_case_count=8, consumer=consumer, arm=arm)
                for c in cases:
                    c['contract']['state_coupled_credit_cases'] = current['state_coupled_credit_cases']
                    write_json_atomic(Path(c['contract']['output_dir'])/'run_contract.json', c['contract'])
                envs, initial_states = pool.switch(task)
                result = rollout_shard(envs=envs, init_states=initial_states, task=task,
                    state_ids=tuple(task['init_state_ids']), contract=current, policy=runtime.policy,
                    preprocess=runtime.processor, postprocess=runtime.processor.unnormalize_action,
                    task_adapter=adapter, episode_contexts=cases)
                for row in result:
                    case = next(c for c in cases if c['evidence']['case_id'] == row['state_coupled_credit_case']['case_id'])
                    shard = EvaluationShard(job_id=case['evidence']['case_id'], ordinal=0,
                        suite=task['suite'], task_id=task['task_id'], init_state_ids=(row['init_state_id'],),
                        horizon=task['horizon'], estimated_cost=task['horizon'], preferred_gpu=None)
                    _validate_episode_row(row, contract=case['contract'], shard=shard, task=task)
                rows.extend(result)
                write_json_atomic(out/'results.json', dict(rows=rows, forward_records=forwards,
                    student_Writer_gamma_Z_removed=True, source_loading_count=1,
                    peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30, consumer_git=contract['git']))
        if len(rows) != 64:
            raise ValueError('fixed arm readback must contain all64 rows')
    finally:
        pool.close()
        adapter.close()
    return rows
