"""§15 metadata events, weighted actual credit entry and complete-parent contracts."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import json
import math
from collections import Counter

import pytest
import torch

from ember.operator_writer import data, credit, joint_training as study, specification as specs
from ember.operator_writer import support_diversity as fork, run
from ember.pi05_source_checkpoint import read_json

REPO = Path(__file__).resolve().parents[2]


def metadata_data(monkeypatch, spec):
    monkeypatch.setattr(data, 'RawTeacherVideoStore', lambda *a, **kw: SimpleNamespace(close=lambda: None))
    tasks = data.TASKS[:24] + fork.SOURCE_TASKS if spec['task'] == fork.TASK else data.TASKS
    return data.FormalData(REPO, spec, query_labels=False, task_ids=tasks)


def test_full_window_real_event_consumer_preserves_target_and_corrects_source(monkeypatch):
    spec = specs.specification(specs.SUPPORT_DIVERSITY_SPEC_PATH)
    current = metadata_data(monkeypatch, spec)
    old = metadata_data(monkeypatch, specs.specification(specs.CONDITIONAL_CONTINUATION_SPEC_PATH))
    counts, weight, teacher = Counter(), Counter(), {}
    target_count = 0
    for step in range(450, 630):
        originals = {t: old.event(step, t) for t in old.tasks_for_step(step)}
        jobs = [current.event(step, task) for task in current.tasks_for_step(step)]
        assert len(jobs) == len({j['task'] for j in jobs}) == 4
        for j in jobs:
            assert len({q['demo'] for q in j['queries']}) == 28
            assert j['teacher_demo'] not in {q['demo'] for q in j['queries']}
            assert all(0 <= q['frame'] < current.tasks[j['task']].episode_lengths[q['demo']] - 1 for q in j['queries'])
            if j['group'] == 'target':
                assert {k:v for k,v in j.items() if k not in ('weight','group','original_task')} == originals[j['task']]
                assert j['weight'] == 1
                target_count += 1
            else:
                counts[j['task']] += 1; weight[j['task']] += j['weight']
                teacher.setdefault(j['task'], []).append(j['teacher_demo'])
                assert j['task'] >= 40
    assert target_count == 480 and sum(counts.values()) == 240
    assert Counter(counts.values()) == {3:44,4:27}
    assert all(math.isclose(v,240/71) for v in weight.values())
    assert all(len(set(v)) == len(v) for v in teacher.values())
    assert math.isclose(sum(weight.values()),240)
    original_sampler = read_json(study.CONDITIONAL_PARENT_CHECKPOINT.parent.parent/'run_contract.json')['sampler']
    migration = current.restore({**original_sampler,'next_step':450})
    assert 'not_exact_trajectory' in migration['migration']
    assert current.sampler_state()['parent_sampler'] == {**original_sampler,'next_step':450}
    current.next_step = 540
    saved = current.sampler_state(); current.next_step = 0
    assert current.restore(saved) is None and current.next_step == 540
    bad = deepcopy(saved); bad['support_slots'][0]['weight'] *= 2
    with pytest.raises(ValueError): current.restore(bad)


@pytest.mark.parametrize('weight',[1.,80/71,60/71])
def test_real_one_job_weighted_cotangent_replay_no_rank_average(monkeypatch, weight):
    parameter = torch.tensor(1.,requires_grad=True)
    runtime = SimpleNamespace(device=torch.device('cpu'), policy=None,lora=None, writer=None,
                              processor=SimpleNamespace(training_batch=lambda b:b))
    runtime.compile = lambda *a,**kw: ({'factor':parameter*2}, {})
    queries = [{'demo':i,'frame':0} for i in range(28)]
    dataset = SimpleNamespace(condition=lambda *a: ((),5,2),batch=lambda e:{})
    def functional(policy,state,contract,batch,**kw):
        assert kw['condition_weight'] == weight/4
        return {'lora_cotangent':{'factor':(2*state['factor']*kw['condition_weight']).detach()},'flow_loss':4.}
    monkeypatch.setattr(credit,'paired_functional_credit',functional)
    monkeypatch.setattr(credit,'autocast',lambda device:nullcontext())
    monkeypatch.setattr(credit,'native_credit',lambda native:{})
    monkeypatch.setattr(torch.cuda,'synchronize',lambda device:None)
    event={'task':40,'visit':0,'teacher_demo':49,'queries':queries,'flow_seed':42,'weight':weight}
    result=credit.one_job(runtime,dataset,event,28,32,'full')
    assert parameter.grad.item() == pytest.approx(2*weight)
    assert result['weight'] == weight and result['public_flow_loss'] is None


def test_real450_scientific_fork_source_and_optimizer_preserved():
    spec = specs.specification(specs.SUPPORT_DIVERSITY_SPEC_PATH)
    parent = read_json(study.CONDITIONAL_PARENT_CHECKPOINT.parent.parent/'run_contract.json')
    current = {**deepcopy(parent),'git':{'commit':'fixture-new-reading','branch':'','dirty_paths':[],
               'pushed_ref':'origin/main'},'spec':str(specs.SUPPORT_DIVERSITY_SPEC_PATH),
               'events':spec['events'],'continuation':spec['continuation']}
    parent_state={**parent['sampler'],'next_step':450}
    current['sampler']={k:v for k,v in fork.sampler_state(parent_state,450).items() if k!='next_step'}
    study.register_conditional_resume(spec,SimpleNamespace(resume=study.CONDITIONAL_PARENT_CHECKPOINT),current)
    assert current['source_resume']['migration']=='registered_support_distribution_fork'
    assert run.resume_contract_compatible(parent,current,allow_topology_change=True)
    for field in ('operator','optimizer'):
        bad=deepcopy(current);bad[field]['extra']=True
        assert not run.resume_contract_compatible(parent,bad,allow_topology_change=True)
    bad=deepcopy(current);bad['events']['support_weight']='unweighted'
    assert not run.resume_contract_compatible(parent,bad,allow_topology_change=True)
    args=SimpleNamespace(mode=study.CONDITIONAL_MODE,attempt='diversity',resume=study.CONDITIONAL_PARENT_CHECKPOINT,
                         microbatch=28,frame_chunk=32,stop_after_macro=None,pilot_arm=None)
    study.validate_request(spec,args)
    for stop in (450,630,720,810,900):
        args.stop_after_macro=stop
        with pytest.raises(ValueError):study.validate_request(spec,args)
