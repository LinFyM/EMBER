"""Retained numerical-coordinate and complete-score protocol regressions."""
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from ember.pi05_lora import load_pi05_lora_contract, derive_pi05_lora_rank
from ember.lora import expected_lora_state_shapes
from ember.proposal_writer.model import FactorLayout
from ember.proposal_writer.path import score_log_probability
from ember.proposal_writer.contract import pilot_panel
from ember.writer.practice.execution import action_chunk


def test_complete_original_coordinate_roundtrip():
    root=Path(__file__).resolve().parents[2]
    contract=derive_pi05_lora_rank(load_pi05_lora_contract(root/'configs/pi05_lora_rank128_aligned.json'),rank=8)
    generator=torch.Generator().manual_seed(11)
    state={k:torch.randn(shape,generator=generator) for k,shape in expected_lora_state_shapes(contract).items()}
    layout=FactorLayout(contract,state);blocks=layout.pack(state)
    # Arbitrary padding cannot change any original A row/B column.
    blocks[~layout.valid]=123.
    restored=layout.unpack(blocks)
    assert layout.count==643584 and len(blocks)==10064
    assert all(torch.equal(restored[k],v) for k,v in state.items())


def test_condition_sum_and_forced_zero_credit():
    logits=torch.tensor([.2,.6],requires_grad=True)
    score=score_log_probability(logits,0)+score_log_probability(logits,1)
    forced=score_log_probability(logits[:1],0,forced=True)
    (score+forced).backward()
    torch.testing.assert_close(logits.grad,torch.ones(2)-2*logits.detach().softmax(0))
    assert float(forced.detach())==0


def test_distributed_missing_label_batch_has_zero_gradient(monkeypatch):
    from ember.proposal_writer import learning
    module=nn.Linear(2,1)
    context=SimpleNamespace(world_size=2,device=torch.device('cpu'))
    reductions=[]
    monkeypatch.setattr(learning.dist,'all_reduce',lambda value,**_: reductions.append(value.clone()))
    learning.synchronize_gradients(module,context)
    assert len(reductions)==1 and int(reductions[0].sum())==0
    assert all(parameter.grad is None for parameter in module.parameters())


def test_roles_are_disjoint_full_fifty_panels():
    for row in pilot_panel()['tasks']:
        roles=['selection_states','audit_states','practice_states','local_query_states','rl_query_states','report_states']
        states=[s for role in roles for s in row[role]]
        assert len(states)==50 and set(states)==set(range(50))
        assert len(set(row['video_order']))==50


def test_locked_queries_capture_function_without_appending_history():
    from ember.proposal_writer.path import evaluate_fixed
    from ember.writer.practice import History
    class Runner:
        def run(self,requests):
            for request in requests:
                first=dict(normalized_actions=torch.ones(5,7),noise_seed=1729,executed=torch.ones(5,dtype=torch.bool))
                yield dict(request=request,row=dict(success=True,init_state_id=request['state_id']),
                           history=History(records=[first]))
    own=History()
    path=SimpleNamespace(selected='G',states={'G':{},'MT300':{}},identity='condition',demo=0,task=12,history=own)
    rows=evaluate_fixed(None,Runner(),path,{'horizon':280},[3],1234)
    assert len(rows)==2 and not own.records and not own.episodes
    assert {row['arm'] for row in rows}=={'MT','selected'}
    assert all(row['query_excluded_from_H'] and row['first_noise_seed']==1729 for row in rows)
    assert all(torch.tensor(row['first_native_normalized_actions']).shape==(5,7) for row in rows)


def test_full_native_latent_and_same_call_hidden_times():
    class Velocity(nn.Module):
        def __init__(self):
            super().__init__();self.policy=SimpleNamespace(model=SimpleNamespace(action_out_proj=nn.Identity()))
            self.calls=[]
        def forward(self,value,time):
            assert value.shape==(2,50,32)
            self.calls.append(time)
            self.policy.model.action_out_proj(torch.full((2,50,1024),time))
            return torch.ones_like(value)
    velocity=Velocity();actions,hidden=action_chunk(velocity,torch.ones(2,50,32))
    assert len(velocity.calls)==10 and actions.shape==(2,50,7) and hidden.shape==(2,2,50,1024)
    torch.testing.assert_close(actions,torch.zeros_like(actions),atol=1e-6,rtol=0)
    assert float(hidden[0,0,0,0])==1. and abs(float(hidden[0,1,0,0])-.1)<.001


def test_frozen_response_cache_keeps_fresh_actor_gradient():
    from ember.proposal_writer.path import decision_logits
    from ember.writer.practice import History
    class Reader(nn.Module):
        def __init__(self):super().__init__();self.weight=nn.Parameter(torch.tensor(2.));self.calls=0
        def forward(self,kind,features,rows,states,candidates,responses,budget):
            self.calls+=1
            return self.weight*responses['MT300'][0,:1]
    class Runtime:
        def __init__(self):
            self.device=torch.device('cpu');self.actor=Reader();self.calls=0
            self.tasks={12:SimpleNamespace(authority=SimpleNamespace(language='real target language'))}
        def observation_features(self,observations):return {}
        def responses(self,states,history,language,*,parent=None):
            self.calls+=1;return {'MT300':torch.ones(1,70)}
    runtime=Runtime();state={'factor':torch.ones(1)}
    path=SimpleNamespace(task=12,history=History(states={'MT300':state}),states={'MT300':state})
    decision=dict(kind='final',candidates=['MT300'],records=0,episodes=0,budget=[1.,1.,0.,0.])
    first=decision_logits(runtime,path,decision,{})
    first.sum().backward()
    assert float(runtime.actor.weight.grad)==1
    with torch.no_grad():runtime.actor.weight.add_(1.)
    runtime.actor.zero_grad(set_to_none=True)
    second=decision_logits(runtime,path,decision,{})
    second.sum().backward()
    assert runtime.calls==1 and runtime.actor.calls==2
    assert float(second.detach()[0])==3 and float(runtime.actor.weight.grad)==1


def test_candidate_retirement_requires_consumption_and_preserves_facts(tmp_path):
    import pytest
    from ember.proposal_writer.path import Compilation,retire_consumed_parameters,load_history
    from ember.writer.practice import History
    from ember.pi05_source_checkpoint import write_json_atomic
    names=['layer.lora_A.default.weight','layer.lora_B.default.weight']
    mt={k:torch.ones(2,2) for k in names}
    states={'MT300':mt,'G_000':{k:v+1 for k,v in mt.items()},'G_001':{k:v+2 for k,v in mt.items()}}
    raw=torch.arange(24,dtype=torch.uint8).reshape(2,3,2,2)
    hidden=torch.ones(2,50,3)
    h=History(records=[dict(parameter_ref='G_000',hidden=hidden,actions=torch.ones(5,7))],
        observations={'seen':dict(images=raw)},states=states,image_features={'seen':torch.ones(2,3)})
    path=Compilation(12,42,'owned_test',0,7,h,states,practiced=list(states),valid=list(states),
        selected='G_001',kernel_identity={'checkpoint':'retained_fixed_psi'})
    path.save(tmp_path)
    proof=tmp_path/'consumer.json';write_json_atomic(proof,{'complete':False})
    with pytest.raises(ValueError,match='complete consumer'):retire_consumed_parameters(tmp_path,proof,mt)
    assert (tmp_path/'G_000.safetensors').exists()
    write_json_atomic(proof,{'complete':True})
    result=retire_consumed_parameters(tmp_path,proof,mt)
    saved=load_history(tmp_path/'history.pt.gz')
    assert not (tmp_path/'G_000.safetensors').exists() and (tmp_path/'G_001.safetensors').exists()
    assert torch.equal(saved.observations['seen']['images'],raw) and torch.equal(saved.records[0]['hidden'],hidden)
    assert set(saved.states)=={'MT300'} and not saved.image_features
    assert result['deleted_parameter_bytes']>0 and result['full_compilation_replay_available'] is False


def test_cost_forecast_counts_all_receipts_without_gpu_hour_gate(tmp_path,monkeypatch):
    import datetime
    import pytest
    import time
    from ember.proposal_writer import batch
    from ember.pi05_source_checkpoint import write_json_atomic
    # Historical numeric limits do not reactivate an Owner-cancelled launch cap.
    write_json_atomic(tmp_path/'run_contract.json',{'limits':{'GPU_hours':96,'carry_in_GPU_hours':18.640412103864882}})
    monkeypatch.setattr(batch.subprocess,'check_output',lambda command,**_: 'pushed_commit' if command[1]=='rev-parse' else '')
    coordinator=batch.Batch(tmp_path,[('gpu01',1),('gpu01',3)],root=tmp_path)
    write_json_atomic(tmp_path/'profile_charged/exit.json',{'GPU_hours':1000.,'exit_code':1})
    outside=tmp_path/'jobs/old_teacher/attempt_001'
    write_json_atomic(outside/'launch.json',dict(job='old_teacher',command=['teacher','--stop','160'],
        start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),estimate_seconds=160*10.9))
    fast=batch.Job('fast',['teacher'],estimate_seconds=4*3600,device_estimates={('gpu01',1):3600})
    forecast=coordinator.track_cost_estimate(fast,'gpu01',1)
    assert forecast['settled_GPU_hours']==pytest.approx(1018.6404121038649,abs=1e-10)
    assert forecast['inflight_GPU_hours']>=batch.UNMEASURED_TEACHER_SECONDS*160/3600
    assert forecast['added_device_GPU_hours']==1 and forecast['GPU_hours_hard_cap_active'] is False
    assert coordinator.running[('gpu01',1)]['estimate_seconds']==3600
    write_json_atomic(outside/'launch.json',dict(job='outside_G',command=['G'],physical_GPU_count=2,
        start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),estimate_seconds=3600,budget_GPUh=96))
    coordinator.running[('gpu01',1)]['start']=time.time()-2*3600
    forecast=coordinator.track_cost_estimate(batch.Job('additional',['audit'],estimate_seconds=3*3600),'gpu01',3)
    assert forecast['inflight_GPU_hours']>=4 and forecast['projected_GPU_hours']>=1025.6404121038649


def test_capless_admission_preserves_physical_and_quota_guards(tmp_path,monkeypatch):
    import json
    import pytest
    from ember.proposal_writer import batch
    from ember.pi05_source_checkpoint import write_json_atomic
    write_json_atomic(tmp_path/'run_contract.json',{'limits':{'GPU_hours':None,'GPU_hours_hard_cap_active':False}})
    write_json_atomic(tmp_path/'profile_charged/exit.json',{'GPU_hours':1000.,'exit_code':0})
    monkeypatch.setattr(batch.subprocess,'check_output',lambda command,**_: 'pushed' if command[1]=='rev-parse' else '')
    coordinator=batch.Batch(tmp_path,[('gpu01',0)],root=tmp_path)
    gpus=[dict(index=i,processes=[{'owner':'ymdai'}],mem_used_mib=10000,mem_free_mib=30000,
               compute_mode='Default',uuid=str(i),util_gpu_pct=100) for i in range(7)]
    quota='/dev/data1 100 2147483648'
    def run(command,**_):
        return SimpleNamespace(stdout=json.dumps(dict(bad_nodes=[],node_results={'gpu01':{'gpus':gpus},
            'gpu02':{'gpus':[]}})) if command[0]=='python3' else quota)
    monkeypatch.setattr(batch.subprocess,'run',run)
    job=batch.Job('G240',['G'],estimate_seconds=3600)
    with pytest.raises(RuntimeError,match='outside project allocation'):
        coordinator.admit(job,'gpu01',0,tmp_path)
    for gpu in gpus:gpu['processes']=[]
    quota='/dev/data1 2097152000 2147483648'
    with pytest.raises(RuntimeError,match='independent quota'):
        coordinator.admit(job,'gpu01',0,tmp_path)
    quota='/dev/data1 100 2147483648'
    admission=coordinator.admit(job,'gpu01',0,tmp_path)
    assert admission['concurrency_cap']==6 and admission['cost_forecast']['projected_GPU_hours']==1001


def test_external_checkpoint_exit_event_releases_deferred_device(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from ember.proposal_writer import batch
    from ember.pi05_source_checkpoint import write_json_atomic
    write_json_atomic(tmp_path/'run_contract.json',{'limits':{'GPU_hours':52}})
    monkeypatch.setattr(batch.subprocess,'check_output',lambda command,**_: 'pushed_commit' if command[1]=='rev-parse' else '')
    coordinator=batch.Batch(tmp_path,[('gpu01',1)],root=tmp_path)
    launches=[]
    monkeypatch.setattr(coordinator,'execute',lambda job,device: launches.append((job.identity,device)) or job.identity)
    parent=tmp_path/'jobs/teacher160/latest.json'
    # The receipt can precede registration: watch-before-read also handles this race.
    write_json_atomic(parent,{'exit_code':0})
    completed=coordinator.run([batch.Job('resume480',['teacher'],('teacher160',),10)],
        deferred_devices={('gpu01',1):('teacher160',)})
    assert launches==[('resume480',('gpu01',1))] and set(completed)=={'teacher160','resume480'}
    parent.unlink()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future=executor.submit(batch.dependency_completion,tmp_path,{'teacher160'})
        write_json_atomic(parent,{'exit_code':0})
        assert future.result(timeout=3)=={'teacher160'}


def test_merged_base_task_source_expert_and_backward_ownership():
    from ember.lora import LoRATarget,inject_task_lora,identity_lora_state
    from ember.proposal_writer.native import PolicyContexts
    from torch.utils.checkpoint import checkpoint
    class Policy(nn.Module):
        def __init__(self):super().__init__();self.layer=nn.Linear(3,2,bias=False)
        def forward(self,x):return self.layer(x)
    task=SimpleNamespace(targets=(LoRATarget('layer',3,2),),rank=2,alpha=2,dropout=0.,identity_seed=7,parameter_count=10)
    expert=SimpleNamespace(targets=task.targets,rank=4,alpha=4,dropout=0.,identity_seed=7,parameter_count=20)
    policy=inject_task_lora(Policy(),task).requires_grad_(False)
    source=policy.layer.weight.detach().clone()
    mt={k:torch.full(shape,.1) for k,shape in expected_lora_state_shapes(expert).items()}
    initial=identity_lora_state(task);context=PolicyContexts(policy,task,expert,mt,initial)
    x=torch.ones(1,3)
    expected=nn.functional.linear(x,source+mt['layer.lora_B.default.weight']@mt['layer.lora_A.default.weight'])
    with context.activate([initial]):torch.testing.assert_close(policy(x),expected)
    torch.testing.assert_close(policy(x),nn.functional.linear(x,source))
    with context.source_expert():
        with context.activate([mt]):torch.testing.assert_close(policy(x),expected)
    factors={k:v.clone().requires_grad_() for k,v in initial.items()}
    def call(value,*values):
        with context.activate([dict(zip(factors,values,strict=True))]):return policy(value)
    output=checkpoint(call,x,*factors.values(),use_reentrant=False)
    output.sum().backward()
    assert factors['layer.lora_B.default.weight'].grad.norm()>0
    assert all(p.grad is None for p in policy.parameters())
    torch.testing.assert_close(policy(x),nn.functional.linear(x,source))
    assert context._active_state is None and not context.expert_active.get()
    context.close()


def test_multistart_dependencies_and_same_node_shared_allocation():
    from ember.proposal_writer.batch import initial_jobs,Job,Batch
    jobs={j.identity:j for j in initial_jobs(28,16,[('gpu01',1)],root=Path('/unused'))}
    assert len(jobs)==48
    assert jobs['collect_0012_1'].dependencies==('teacher_0012_0_160',)
    assert jobs['collect_0012_2'].dependencies==('teacher_0012_0_480',)
    assert jobs['collect_0012_3'].dependencies==('collect_0012_1',)
    assert jobs['teacher_0012_3_480'].dependencies==('collect_0012_3',)
    shared=Job('G',['G'],gpu_count=2)
    assert Batch.allocation(shared,[('gpu01',1),('gpu02',4),('gpu02',6)])==[('gpu02',4),('gpu02',6)]
    assert Batch.allocation(shared,[('gpu01',1),('gpu02',4)]) is None


def test_batch_prioritizes_unlocking_chain_and_measured_device(tmp_path,monkeypatch):
    from ember.proposal_writer import batch
    from ember.pi05_source_checkpoint import write_json_atomic
    write_json_atomic(tmp_path/'run_contract.json',{'limits':{'GPU_hours':96}})
    monkeypatch.setattr(batch.subprocess,'check_output',lambda command,**_: 'pushed' if command[1]=='rev-parse' else '')
    coordinator=batch.Batch(tmp_path,[('gpu01',1)],root=tmp_path)
    launches=[]
    monkeypatch.setattr(coordinator,'execute',lambda job,device: launches.append(job.identity) or job.identity)
    jobs=[batch.Job('leaf',[],estimate_seconds=15),batch.Job('seed',[],estimate_seconds=2),
          batch.Job('collect',[],('seed',),2),batch.Job('late',[],('collect',),20)]
    coordinator.run(jobs)
    assert launches==['seed','collect','late','leaf']
    measured=batch.Job('measured',[],estimate_seconds=20,device_estimates={('gpu01',5):10})
    assert batch.Batch.allocation(measured,[('gpu01',1),('gpu01',5)])==('gpu01',5)


def test_located_device_failure_keeps_independent_work_and_blocks_descendant(tmp_path,monkeypatch):
    import pytest
    from ember.proposal_writer import batch
    from ember.pi05_source_checkpoint import write_json_atomic
    write_json_atomic(tmp_path/'run_contract.json',{'limits':{'GPU_hours':96}})
    monkeypatch.setattr(batch.subprocess,'check_output',lambda command,**_: 'pushed' if command[1]=='rev-parse' else '')
    coordinator=batch.Batch(tmp_path,[('gpu01',1),('gpu01',5)],root=tmp_path)
    launches=[]
    def execute(job,device):
        launches.append((job.identity,device))
        if job.identity=='renderer':raise batch.DeviceWorkloadFailure('located framebuffer failure')
        return job.identity
    monkeypatch.setattr(coordinator,'execute',execute)
    jobs=[batch.Job('renderer',[],estimate_seconds=20),batch.Job('dependent',[],('renderer',),30),
          batch.Job('independent',[],estimate_seconds=10),batch.Job('independent_next',[],('independent',),5)]
    with pytest.raises(RuntimeError,match='framebuffer'):
        coordinator.run(jobs)
    assert [name for name,_ in launches]==['renderer','independent','independent_next']
    assert launches[1][1]==launches[2][1]==('gpu01',5)


def test_unknown_batch_failure_stops_new_dispatch(tmp_path,monkeypatch):
    import pytest
    from ember.proposal_writer import batch
    from ember.pi05_source_checkpoint import write_json_atomic
    write_json_atomic(tmp_path/'run_contract.json',{'limits':{'GPU_hours':96}})
    monkeypatch.setattr(batch.subprocess,'check_output',lambda command,**_: 'pushed' if command[1]=='rev-parse' else '')
    coordinator=batch.Batch(tmp_path,[('gpu01',1)],root=tmp_path)
    launches=[]
    def execute(job,device):
        launches.append(job.identity)
        raise RuntimeError('unknown data/gradient failure')
    monkeypatch.setattr(coordinator,'execute',execute)
    with pytest.raises(RuntimeError,match='unknown data'):
        coordinator.run([batch.Job('unknown',[],estimate_seconds=20),batch.Job('later',[],estimate_seconds=10)])
    assert launches==['unknown']


def test_shared_launch_uses_measured_headroom_chunks_without_changing_logical_flow(tmp_path,monkeypatch):
    from ember.proposal_writer import batch
    from ember.pi05_source_checkpoint import write_json_atomic
    write_json_atomic(tmp_path/'run_contract.json',{'limits':{'GPU_hours':96}})
    monkeypatch.setattr(batch.subprocess,'check_output',lambda command,**_: 'pushed' if command[1]=='rev-parse' else '')
    devices=[('gpu01',gpu) for gpu in (1,3,4,5)]
    coordinator=batch.Batch(tmp_path,devices,root=tmp_path)
    memory={1:39572,3:20118,4:20118,5:21262}
    monkeypatch.setattr(coordinator,'admit',lambda job,node,gpu,destination: {'free_MiB':memory[gpu]})
    def launch(job,selected,destination,command,env,admissions):
        assert selected==devices and env['EMBER_META_FRAME_CHUNKS']=='16,4,4,8'
        assert env['CUDA_VISIBLE_DEVICES']=='1,3,4,5'
        assert command[command.index('--stop')+1]=='240'
        return job.identity
    monkeypatch.setattr(coordinator,'launch',launch)
    assert coordinator.execute(batch.Job('G240',['G','--stop','240','--frame-chunk','4'],gpu_count=4),devices)=='G240'


def test_condition_queue_preserves_full_panels_and_parallel_RL_report_dependencies(tmp_path):
    from ember.proposal_writer.batch import stage_jobs
    args=SimpleNamespace(root=tmp_path,stage='decision-report',shared_gpus=4,shared_frame_chunk=4,slots=16)
    jobs={job.identity:job for job in stage_jobs(args)}
    local=[job for job in jobs.values() if job.arguments[0]=='local-data']
    report=[job for job in jobs.values() if job.arguments[0]=='report' and '--video-arm' not in job.arguments]
    pair=lambda job:(int(job.arguments[job.arguments.index('--task')+1]),
                     int(job.arguments[job.arguments.index('--condition-ordinal')+1]))
    assert {pair(job) for job in local}=={(task,ordinal) for task in (12,29,32,38) for ordinal in range(4)}
    assert len(local)==16 and len(report)==16
    for version in ('local128','RL16'):
        rows=[job for job in report if job.dependencies==(version,)]
        assert {pair(job) for job in rows}=={(task,ordinal) for task in (12,29,32,38) for ordinal in (0,1)}
    assert set(jobs['local128'].dependencies)=={job.identity for job in local}
    assert jobs['RL16'].dependencies==('local128',)
    for task in (12,32):
        for arm in ('other','wrong'):
            assert jobs[f'control_{arm}_{task:04d}'].dependencies==(f'report_RL16_{task:04d}_00',)


def test_empty_seed_events_and_true_producer_parent_identity(tmp_path):
    from safetensors.torch import save_file
    from ember.proposal_writer.contract import pilot_panel
    from ember.proposal_writer.run import seed_events,event_record
    from ember.proposal_writer.path import load_history
    from ember.pi05_source_checkpoint import read_json
    panel=pilot_panel()
    save_file({'A':torch.ones(1),'B':torch.zeros(1)},str(tmp_path/'initial.safetensors'))
    seed_events(tmp_path,panel)
    for row in panel['tasks']:
        task=row['task_id'];directory=tmp_path/'events'/f'task_{task:04d}_event_00'
        event=read_json(directory/'event.json');history=load_history(event['history_path'])
        assert event['complete'] and event['parent_ref']=='MT300' and event['history_producer'] is None
        assert not history.records and set(history.states)=={'MT300'}
        mid,_=event_record(tmp_path,row,task,1);late,_=event_record(tmp_path,row,task,2)
        returned,_=event_record(tmp_path,row,task,3)
        assert mid['parent_ref']==returned['history_producer']==f'seed160_task_{task:04d}'
        assert returned['parent_ref']=='MT300' and late['parent_ref']==f'seed480_task_{task:04d}'
        assert mid['teacher_demo']==late['teacher_demo'] and event['teacher_demo']==returned['teacher_demo']
