from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.prefix_replay import plan_saved_prefix, validate_cases
from ember.pi05_eval_contract import policy_noise_seed
from ember.pi05_evaluation import rollout_shard


def geometry(tmp_path):
    task = dict(suite='libero_10', task_id=2, split_role='train', language='task', horizon=15)
    contract = dict(role='development_train', policy=dict(replan_steps=5, num_inference_steps=10),
        rng=dict(inference_seed=7), environment=dict(dummy_action=[0.]*7,dummy_settling_steps=10),
        operator_read_write_scene=None, output_dir=str(tmp_path),
        diagnostic_occupancy_capture=None)
    ids = ['17_17','17_43','43_17','43_43']
    contract['frozen_case_matrix'] = dict(schema_version='ember_saved_action_case_matrix_v1',
        case_ids=ids, cut_control_steps=10, task=['libero_10',2],init_state_id=2)
    noise = tuple(policy_noise_seed(7,'libero_10',2,2,k) for k in range(3))
    cases = [dict(evidence=dict(case_id=k),contract=contract,
        prepared_adapter=SimpleNamespace(key=k),physical_prefix=np.full((10,7),float(n//2+1),dtype=np.float32),
        archived_chunks=(torch.zeros(1,50,7),)*2,noise_seeds=noise) for n,k in enumerate(ids)]
    return task,contract,cases


def test_four_paired_cases_use_actual_rollout_without_cut_reset_or_extra_settling(monkeypatch,tmp_path):
    task,contract,cases = geometry(tmp_path)
    class Env:
        def __init__(self):self.actions=[]
        def seed(self,seed):assert seed==7
        def reset(self):self.actions=[]
        def set_init_state(self,state):return {'state':torch.zeros(1,8)}
        def step(self,action):
            self.actions.append(np.asarray(action).copy())
            return {'state':torch.full((1,8),float(len(self.actions)))},0,False,{}
    class Policy:
        config=SimpleNamespace(chunk_size=50,max_action_dim=32)
        def __init__(self):self.calls=[];self.resets=0
        def reset(self):self.resets+=1
        def predict_action_chunk(self,batch,*,noise,num_steps):
            self.calls.append((batch,noise,num_steps))
            return torch.ones(4,50,7)
    monkeypatch.setattr('ember.pi05_evaluation.libero_policy_input',lambda obs,language:{'observation.state':obs['state']})
    policy=Policy();envs=[Env() for _ in cases]
    rows=rollout_shard(envs=envs,init_states=[0,1,2],task=task,state_ids=[2]*4,contract=contract,
        policy=policy,preprocess=lambda x:x,postprocess=lambda x:x,episode_contexts=cases)
    assert policy.resets==1 and len(policy.calls)==1
    assert policy.calls[0][0]['observation.state'].shape==(4,8)
    assert torch.equal(policy.calls[0][0]['observation.state'],torch.full((4,8),20.))
    assert policy.calls[0][2]==10
    assert len(rows)==4 and all(r['steps']==15 for r in rows)
    assert all(tuple(r['policy_noise_seeds'])==cases[0]['noise_seeds'] for r in rows)
    for env,case in zip(envs,cases):
        assert len(env.actions)==25  # ten initial dummy + ten archived + five generated
        np.testing.assert_array_equal(env.actions[10:20],case['physical_prefix'])


def test_duplicate_states_require_registered_train_case_matrix(tmp_path):
    task,contract,cases=geometry(tmp_path)
    validate_cases(cases,[2]*4,task,contract)
    task['split_role']='validation'
    with pytest.raises(Pi05EvaluationError,match='scope'):validate_cases(cases,[2]*4,task,contract)


def test_saved_prefix_rejects_restarted_noise_index(tmp_path):
    task,contract,cases=geometry(tmp_path)
    slot=dict(steps=5,replan_index=0,init_state_id=2,saved_prefix=cases[0],action_plan=deque())
    with pytest.raises(Pi05EvaluationError,match='absolute noise index'):
        plan_saved_prefix([slot],[{}],[{}],task=task,contract=contract)
