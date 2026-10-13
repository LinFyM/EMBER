"""Physical migration keeps the complete selector state and inactive rank RNG."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.proposal_writer import shared_checkpoint as saved


def context(world, rank=0):
    return SimpleNamespace(world_size=world, rank=rank, local_rank=rank,
                           is_main=rank==0, device=torch.device('cpu'))


def test_complete_boundary_migration_restores_the_saved_rank(monkeypatch, tmp_path):
    original=[dict(marker=i) for i in range(4)]
    state=dict(topology=dict(world_size=4,logical_condition_batch=4),rank_rng=original,
               stage='local',next_update=180)
    runtime=SimpleNamespace(root=tmp_path,G_sampler=dict(next_update=38349),supervision_version=9)
    restored=[]
    monkeypatch.setattr(saved,'restore_rng',lambda rng,ctx:restored.append((rng,ctx.rank)))
    saved.restore_physical_rng(tmp_path/'A180',state,runtime,context(3,2),stage='RL')
    assert restored==[(original[2],2)]
    assert runtime.retained_rank_rng==original
    assert runtime.physical_rng_migration['source_unassigned_ranks']==[3]
    assert runtime.physical_rng_migration['logical_batch']==4
    assert not runtime.physical_rng_migration['bitwise_exact_claim']


def test_checkpoint_preserves_inactive_rng_and_resumes_both_Adam_states(monkeypatch,tmp_path):
    generator=nn.Module();generator.endpoint=nn.Linear(1,1)
    actor=nn.Linear(1,1);baseline=nn.Linear(1,1)
    theta=torch.optim.AdamW(actor.parameters());omega=torch.optim.AdamW(baseline.parameters())
    actor(torch.ones(1,1)).sum().backward();theta.step()
    baseline(torch.ones(1,1)).sum().backward();omega.step()
    migration=dict(source_checkpoint=str(tmp_path/'original_RL9'))
    write_json_atomic(tmp_path/'run_contract.json',dict(learning_migration=migration,training_tasks=[0,1,40,41]))
    runtime=SimpleNamespace(root=tmp_path,generator=generator,actor=actor,policy=nn.Linear(1,1),
        generator_optimizer_state=torch.optim.AdamW(generator.endpoint.parameters()).state_dict(),
        G_sampler=dict(next_update=38349),supervision_version=9,learning_migration=migration,
        source_rank_rng=[dict(marker=i) for i in range(4)],source_theta_sampler=dict(next_update=9),
        retained_rank_rng=[dict(marker=i) for i in range(4)],physical_rng_migration=dict(current_world_size=3),
        lora=SimpleNamespace(parameter_count=2),bank_scale_path=tmp_path/'scale.json',
        objective_migration=None,frozen_frontend_checkpoint='frozen_phi432',
        component_migration={},component_rng_sources={},component_samplers={},
        training_baseline=baseline,training_baseline_optimizer=omega)
    runtime.freeze_psi=lambda:runtime.generator.requires_grad_(False)
    monkeypatch.setattr(saved,'capture_rng',lambda _:dict(marker=10))
    monkeypatch.setattr(saved.dist,'gather_object',lambda rng,ranks,dst:ranks.__setitem__(slice(None),
        [dict(marker=10+i) for i in range(3)]))
    monkeypatch.setattr(saved.dist,'barrier',lambda **_:None)
    sampler=dict(next_update=1,task_order=[0,1,40,41],logical_condition_batch=4)
    path=saved.checkpoint(tmp_path,'RL',1,runtime,theta,context(3),sampler=sampler)
    state=torch.load(path/'state.pt',weights_only=False)
    assert state['rank_rng']==[dict(marker=10+i) for i in range(3)]
    assert state['retained_rank_rng']==state['rank_rng']+[dict(marker=3)]
    assert state['topology']['world_size']==3 and state['sampler']==sampler

    fresh=deepcopy(runtime)
    fresh.freeze_psi=lambda:fresh.generator.requires_grad_(False)
    new_theta=torch.optim.AdamW(fresh.actor.parameters());new_omega=torch.optim.AdamW(fresh.training_baseline.parameters())
    restored=[]
    monkeypatch.setattr(saved,'restore_rng',lambda rng,ctx:restored.append((rng,ctx.rank)))
    update=saved.resume_checkpoint(path,'RL',fresh,new_theta,context(4,3),
        baseline=fresh.training_baseline,baseline_optimizer=new_omega)
    assert update==1 and restored==[(dict(marker=3),3)]
    assert all(int(s['step'])==1 for s in new_theta.state.values())
    assert all(int(s['step'])==1 for s in new_omega.state.values())
    assert fresh.G_sampler==dict(next_update=38349) and fresh.supervision_version==9
    assert not any(p.requires_grad for p in fresh.generator.parameters())
    assert read_json(path/'manifest.json')['complete']


def test_migration_rejects_an_unregistered_new_rank(monkeypatch,tmp_path):
    state=dict(topology=dict(world_size=2,logical_condition_batch=4),rank_rng=[{},{}],
               stage='RL',next_update=1)
    monkeypatch.setattr(saved,'restore_rng',lambda *_:pytest.fail('invalid migration must not consume RNG'))
    with pytest.raises(ValueError,match='registered retained RNG'):
        saved.restore_physical_rng(tmp_path,state,SimpleNamespace(),context(3,2),stage='RL')
