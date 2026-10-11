"""Retained numerical-coordinate and complete-score protocol regressions."""
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from ember.pi05_lora import load_pi05_lora_contract, derive_pi05_lora_rank
from ember.lora import expected_lora_state_shapes
from ember.proposal_writer.model import FactorLayout
from ember.proposal_writer.path import score_log_probability
from ember.proposal_writer.contract import materialize_panel,TASKS
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


def test_training_roles_respect_independent_teacher_report_and_query(tmp_path):
    panel=materialize_panel(tmp_path,selection_count=16,report_count=16)
    assert len(panel['tasks'])==36
    assert panel['selection_uses']==['teacher_selection'] and panel['selection_folds_are_historical_metadata']
    for row in panel['tasks']:
        selection=set(row['teacher_selection_states']);report=set(row['teacher_report_states'])
        practice=set(row['practice_states']);query=set(row['query_states']);limited=set(row['report_states'])
        assert len(practice)==20 and len(query)==10 and row['local_query_states']==row['rl_query_states']
        assert selection<=practice and not report&(practice|query|limited|selection)
        assert not query&(practice|limited) and not limited&practice
        assert len(report|practice|query|limited)==50 and len(set(row['video_order']))==50
        assert set(row['teacher_query_demos'])==set(range(46)) and set(row['teacher_videos'])==set(range(46,50))
        assert row['bootstrap_state_id']==row['teacher_selection_states'][0]


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
        def forward(self,kind,features,rows,states,candidates,responses):
            self.calls+=1
            return self.weight*responses['MT300'][0,:1]
    class Runtime:
        def __init__(self):
            self.components={}
            self.device=torch.device('cpu');self.actor=Reader();self.calls=0
            self.tasks={12:SimpleNamespace(authority=SimpleNamespace(language='real target language'))}
        def observation_features(self,observations):return {}
        def responses(self,states,history,language,*,parent=None,absolute_cache=None):
            self.calls+=1;return {'MT300':torch.ones(1,70)}
    runtime=Runtime();state={'factor':torch.ones(1)}
    path=SimpleNamespace(task=12,history=History(states={'MT300':state}),states={'MT300':state})
    decision=dict(kind='final',candidates=['MT300'],records=0,episodes=0)
    first=decision_logits(runtime,path,decision,{})
    first.sum().backward()
    assert float(runtime.actor.weight.grad)==1
    with torch.no_grad():runtime.actor.weight.add_(1.)
    runtime.actor.zero_grad(set_to_none=True)
    second=decision_logits(runtime,path,decision,{})
    second.sum().backward()
    assert runtime.calls==1 and runtime.actor.calls==2
    assert float(second.detach()[0])==3 and float(runtime.actor.weight.grad)==1




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
    write_json_atomic(tmp_path/'run_contract.json',{'limits':{'GPU_hours':None,'GPU_hours_hard_cap_active':False,'added_peak_GiB':128}})
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


def test_common_teacher_dependencies_and_same_node_shared_allocation(tmp_path):
    from ember.proposal_writer.batch import initial_jobs,Job,Batch
    from ember.pi05_source_checkpoint import write_json_atomic
    write_json_atomic(tmp_path/'run_contract.json',dict(teacher=dict(readout_updates=[0,200,400,600,800],selection_per_node=16,report_per_arm=16),
        environment=dict(tasks=[dict(global_task_id=t,horizon=400 if t>=40 else 520) for t in TASKS])))
    jobs={j.identity:j for j in initial_jobs(28,16,[('gpu01',1)],root=tmp_path)}
    assert len(jobs)==360
    assert jobs['teacher_0012_00000400'].dependencies==('readout_0012_00000200',)
    assert jobs['readout_0012_00000200'].dependencies==('teacher_0012_00000200','readout_0012_00000000')
    assert not jobs['readout_0012_00000000'].dependencies
    assert len(jobs['teacher_report_0012'].dependencies)==5
    assert not any('100'==j.arguments[j.arguments.index('--stop')+1] for j in jobs.values() if '--stop' in j.arguments)
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


def test_complete_saturation_prunes_future_jobs_without_faking_completion(tmp_path,monkeypatch):
    from ember.proposal_writer import batch
    from ember.pi05_source_checkpoint import read_json,write_json_atomic
    monkeypatch.setattr(batch.subprocess,'check_output',lambda command,**_: 'pushed' if command[1]=='rev-parse' else '')
    for successes in (15,16):
        root=tmp_path/str(successes)
        write_json_atomic(root/'run_contract.json',dict(teacher=dict(saturation_stop=16,readout_updates=[0,200,400])))
        completed_teacher='teacher_0012_00000200'
        write_json_atomic(root/'jobs'/completed_teacher/'latest.json',dict(exit_code=0))
        coordinator=batch.Batch(root,[('gpu01',1)],root=root);launches=[]
        def execute(job,device):
            launches.append(job.identity)
            if job.identity.startswith('readout_'):
                update=int(job.identity.split('_')[-1])
                write_json_atomic(root/'teachers/task_0012/selection'/f'update_{update:08d}/complete.json',
                    dict(complete=True,rows=[dict(success=i<successes) for i in range(16)]))
            return job.identity
        monkeypatch.setattr(coordinator,'execute',execute)
        jobs=[batch.Job('readout_0012_00000000',[],estimate_seconds=2),
              batch.Job(completed_teacher,[],('readout_0012_00000000',),2),
              batch.Job('readout_0012_00000200',[],(completed_teacher,'readout_0012_00000000'),2),
              batch.Job('teacher_0012_00000400',[],('readout_0012_00000200',),2),
              batch.Job('readout_0012_00000400',[],('teacher_0012_00000400','readout_0012_00000200'),2),
              batch.Job('teacher_report_0012',[],tuple(f'readout_0012_{u:08d}' for u in (0,200,400)),2)]
        done=coordinator.run(jobs)
        assert completed_teacher in done and completed_teacher not in launches
        if successes==16:
            assert launches==['readout_0012_00000000','teacher_report_0012']
            omission=read_json(root/'scientific_omissions/task_0012.json')
            assert omission['incumbent_update']==0 and omission['selection_rows']==16
            assert not set(omission['omitted_jobs'])&set(done)
            assert not (root/'teachers/task_0012/selection/update_00000200/complete.json').exists()
        else:
            assert len(launches)==5 and not (root/'scientific_omissions/task_0012.json').exists()


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
    def readout_launch(job,selected,destination,command,env,admissions):
        assert selected==[devices[-1]] and command[command.index('--slots')+1]=='8'
        assert command[command.index('--stop')+1]=='0' and '--microbatch' not in command
        return job.identity
    monkeypatch.setattr(coordinator,'launch',readout_launch)
    job=batch.Job('MT101',['readout','--task','101','--stop','0','--slots','16'])
    assert coordinator.execute(job,devices[-1])=='MT101' and job.arguments[-1]=='16'


def test_conditional_parent_uses_complete_logits_and_exact_normalization():
    from ember.proposal_writer.path import categorical_record,conditional_parent_log_probability
    from ember.writer.practice import History
    class Runtime:
        device=torch.device('cpu')
        components={}
        tasks={12:SimpleNamespace(authority=SimpleNamespace(language='legitimate language'))}
        def observation_features(self,observations):return {}
        def responses(self,*args,**kwargs):return {}
        def actor(self,kind,features,rows,states,candidates,responses):
            assert candidates==['MT300','G','STOP']  # STOP participates in the set-attention forward.
            return torch.tensor([.4,-.3,2.])
    path=SimpleNamespace(task=12,history=History(),states={'MT300':{},'G':{}},decisions=[])
    chosen=categorical_record(Runtime(),path,'parent',['MT300','G','STOP'],{},
        rng=torch.Generator().manual_seed(2),uniform=False,conditional_continue=True,round_index=0)
    row=path.decisions[0]
    assert chosen!='STOP' and row['sampling_probabilities'][-1]==0
    logits=torch.tensor([.4,-.3,2.],requires_grad=True)
    logp=conditional_parent_log_probability(logits,row['selected_index'],row['candidates'])
    torch.testing.assert_close(logp,logits[:2].log_softmax(0)[row['selected_index']])
    assert abs(float(logp.detach())-row['log_probability'])<1e-6


def test_stopping_recursion_adjoints_and_full_finite_tree_gradient():
    import itertools
    from ember.proposal_writer.learning import continuation_values
    theta=torch.tensor([.15,-.24,.31,.1],dtype=torch.float64,requires_grad=True)
    def node(history):
        index=1+sum((i+2)*(a+1) for i,a in enumerate(history))
        logits=torch.stack((theta[0]*index,theta[1]-.07*index,theta[2]+.04*index))
        full=logits.softmax(0);mu=logits[:2].softmax(0)
        practice=torch.stack((theta[3]+.02*index,-theta[3])).softmax(0)
        query=torch.stack((theta[0]+.03*index,theta[1]-.04*index)).softmax(0)
        q=query@theta.new_tensor([.1+.02*index,.85-.01*index])
        return full[-1],mu,practice,q
    actual=theta.sum()*0;estimator=theta.sum()*0
    for choices in itertools.product(range(4),repeat=2):
        history=[];p=[];q=[];scores=[];probability=theta.new_tensor(1.)
        for choice in choices:
            stop,mu,practice,value=node(history);p.append(stop);q.append(value)
            likelihood=mu[choice//2]*practice[choice%2]
            probability=probability*likelihood;scores.append(likelihood.log());history.append(choice)
        q.append(node(history)[3]);values,q_adj,p_adj=continuation_values(p,q,cost=.001)
        F=sum(values)/2
        actual=actual+probability*F
        score=sum(values[d].detach()*sum(scores[:d+1]) for d in range(2))/2
        direct=sum(a.detach()*value for a,value in zip(q_adj,q))+sum(a.detach()*stop for a,stop in zip(p_adj,p))
        estimator=estimator+probability.detach()*(score+direct)
    expected=torch.autograd.grad(actual,theta,retain_graph=True)[0]
    measured=torch.autograd.grad(estimator,theta)[0]
    torch.testing.assert_close(measured,expected,atol=1e-12,rtol=1e-10)
    # STOP absorbs unchanged value and tiny continuation cost encourages earlier delivery.
    p=torch.tensor([1.,.3]);q=torch.tensor([.8,.4,.9])
    values,q_adj,p_adj=continuation_values(p,q,cost=.001)
    assert all(abs(float(value)-.8)<1e-7 for value in values)
    assert float(q_adj[0])==1 and float(q_adj[1])==float(q_adj[2])==0


def test_five_real_continue_rounds_stop_deployment_and_branch_rng(monkeypatch,tmp_path):
    from ember.proposal_writer import path as owner
    from ember.writer.practice import History
    class Runner:
        environment_steps=0
        def run(self,requests,*,on_completion=None):
            pending=list(requests)
            while pending:
                request=pending.pop(0)
                self.environment_steps+=20;episode=dict(environment_steps=20,init_state_id=request['state_id'],success=False)
                key=request['episode_id'];record=dict(pre=key,post=key,parameter_ref=request['parameter_ref'],
                    feedback=torch.zeros(5),episode=0)
                history=History(records=[record],observations={key:{}},episodes=[episode],states={request['parameter_ref']:request['state']})
                output=dict(row=episode,history=history,request=request)
                if on_completion:pending.extend(on_completion(output))
                yield output
    class Runtime:
        psi_frozen=True;root=tmp_path;device=torch.device('cpu');components={};mt={'factor':torch.ones(1)}
        tasks={12:SimpleNamespace(authority=SimpleNamespace(language='correct language'))}
        def teaching(self,*args,**kwargs):return {}
    def logits(runtime,path,decision,features):
        assert decision['records']<=len(path.history.records)
        # STOP dominates in deployment, yet all continuation rounds must be physically executed.
        return torch.tensor([0. if name!='STOP' else 1000. for name in decision['candidates']])
    def generate(runtime,path,parent,features):
        ref=f'G_{len(path.attempts):03d}';path.states[ref]={'factor':torch.ones(1)};path.valid.append(ref)
        path.attempts.append(dict(reference=ref,parent=parent,generation_and_materialization_seconds=0.))
        return [ref]
    monkeypatch.setattr(owner,'decision_logits',logits);monkeypatch.setattr(owner,'generate_candidates',generate)
    task=dict(global_task_id=12,horizon=100);panel=dict(practice_states=list(range(20)))
    keywords=dict(identity='test',path_ordinal=0,noise_root=7,max_rounds=5)
    first=owner.compile_condition(Runtime(),Runner(),task,panel,0,conditional_continue=True,**keywords)
    assert first.completed_rounds==5 and len(first.outputs)==6 and len(first.history.episodes)==6
    assert all(row['sampling_probabilities'][-1]==0 for row in first.decisions if row['kind']=='parent')
    emit=owner.emit_output
    def altered_branch(*args,**kwargs):
        emit(*args,**kwargs)
        args[1].selected='MT300'  # Final outlet does not govern continuation actions.
    monkeypatch.setattr(owner,'emit_output',altered_branch)
    second=owner.compile_condition(Runtime(),Runner(),task,panel,0,conditional_continue=True,**keywords)
    assert [(d['kind'],d['selected']) for d in first.decisions if d['kind']!='final']==[(d['kind'],d['selected']) for d in second.decisions if d['kind']!='final']
    deployed=owner.compile_condition(Runtime(),Runner(),task,panel,0,**keywords)
    assert deployed.completed_rounds==0 and len(deployed.history.episodes)==1 and deployed.selected=='MT300'
    specifications=[dict(task=task,panel=panel,demo=0,conditional_continue=True,**{**keywords,'identity':name}) for name in ('a','b')]
    together=owner.compile_conditions(Runtime(),Runner(),specifications)
    assert [p.identity for p in together]==['a','b']
    for path in together:
        assert path.completed_rounds==5 and len(path.history.episodes)==6
        assert [(d['kind'],d['selected']) for d in path.decisions]==[(d['kind'],d['selected']) for d in second.decisions]
        assert path.timings['shared_compilation_group']['simultaneous_conditions']==2


def test_query_labels_cached_once_and_never_added_to_history(tmp_path):
    from ember.proposal_writer.supervision import query_once,query_conditions
    from ember.writer.practice import History
    class Runner:
        environment_steps=0;requests=[]
        def run(self,requests):
            self.requests.extend(requests)
            for request in requests:
                self.environment_steps+=12
                yield dict(request=request,row=dict(success=True,init_state_id=request['state_id'],environment_steps=12,
                    env_seed=request['noise_root'],policy_seed_root=request['noise_root'],
                    suite='libero_10',task_id=2,language='exact'),history=History())
    path=SimpleNamespace(selected='G',states={'MT300':{},'G':{},'unpracticed':{}},identity='condition',demo=0,
        task=12,history=History(episodes=[dict(init_state_id=0)]),
        outputs=[dict(candidates=['MT300']),dict(candidates=['MT300','G']),dict(candidates=['MT300','G'])])
    panel=dict(query_states=list(range(10,20)));runner=Runner();task=dict(horizon=100,suite='libero_10',task_id=2,language='exact')
    destination=tmp_path/'student_data/G_init/task_0012/query'
    first=query_once(None,runner,path,task,panel,0,destination)
    second=query_once(None,runner,path,task,panel,0,destination)
    assert first==second=={'MT300':1.,'G':1.} and len(runner.requests)==8
    assert not path.history.records and len(path.history.episodes)==1
    assert all(request['parameter_ref']!='unpracticed' for request in runner.requests)
    path.identity='first_RL';runtime=SimpleNamespace(root=tmp_path)
    output=tmp_path/'RL_query'
    rewards=query_conditions(runtime,runner,[(path,task,panel,0,output)],mt_reference_version='G_init')
    assert rewards==[first] and len(runner.requests)==12
    import json
    complete=json.loads((output/'complete.json').read_text())
    assert len(complete['row_reuse'])==4 and complete['actual_new_episodes_this_attempt']==4
    assert all(row['condition']=='condition' for row in complete['rows'] if row['reference']=='MT300')
    query_conditions(runtime,runner,[(path,task,panel,1,tmp_path/'visit1')],mt_reference_version='G_init')
    assert len(runner.requests)==20  # Later RNG/state panels execute all eight real queries.


def test_native_absolute_response_cross_head_reuse_and_support_invalidation(monkeypatch):
    from contextvars import ContextVar
    from ember.proposal_writer.runtime import Runtime
    runtime=Runtime.__new__(Runtime);runtime.device=torch.device('cpu');runtime.root=Path('/data1/test_only')
    runtime.response_microbatch=16;runtime.components=dict(response_seconds=0.,response_observations=0,response_cache_hits=0)
    runtime.execution=SimpleNamespace(expert_active=ContextVar('expert',default=False))
    runtime.generator=SimpleNamespace(meta=SimpleNamespace(active=ContextVar('meta',default=None)))
    runtime.lora=SimpleNamespace(targets=[SimpleNamespace(name='model.gemma_expert.layer')])
    prefixes=[]
    from ember.proposal_writer import runtime as owner
    monkeypatch.setattr(owner,'NativeVelocity',lambda *args,**kwargs:prefixes.append(1) or object())
    runtime.policy=object()
    calls=[]
    def native(states,batch,noise,**kwargs):
        calls.append(1);return torch.full((2,5,7),float(states[0]['value']))
    runtime.native_actions=native
    records=[dict(pre='real_observation0',noise_seed=7)]
    history=SimpleNamespace(records=records,response_support=lambda *_:dict(
        positions=[0],noise=torch.ones(2,50,32),batch={'legitimate':torch.ones(2,1)}))
    cache={};states={'MT300':{'value':2.},'G':{'value':6.}}
    runtime.responses({'MT300':states['MT300']},history,'exact language',absolute_cache=cache)
    practice=runtime.responses(states,history,'exact language',parent='MT300',absolute_cache=cache)
    final=runtime.responses(states,history,'exact language',absolute_cache=cache)
    assert len(calls)==2 and len(prefixes)==1 and torch.all(practice['G'][:,35:]==4) and torch.all(final['G'][:,35:]==0)
    records[0]['noise_seed']=8
    runtime.responses(states,history,'exact language',absolute_cache=cache)
    assert len(calls)==4 and len(prefixes)==2


def test_joint_bank_retains_parameters_and_excludes_future_prefixes(tmp_path):
    import json
    from safetensors.torch import save_file
    from ember.proposal_writer.supervision import register_students
    from ember.proposal_writer.learning import CFMTrainer,HistoryCache
    from ember.proposal_writer.path import Compilation,save_history
    from ember.writer.practice import History
    root=tmp_path;directory=root/'RL_paths'/'batch1_task12';directory.mkdir(parents=True)
    state={'factor':torch.ones(1)};save_file(state,str(root/'initial.safetensors'))
    history=History(records=[dict(pre='a',post='a',parameter_ref='MT300'),
        dict(pre='b',post='b',parameter_ref='G_000')],observations={'a':{},'b':{}},
        episodes=[dict(init_state_id=0),dict(init_state_id=1)],states={'MT300':state,'G_000':state})
    save_history(directory/'history.pt.gz',history);save_file(state,str(directory/'G_000.safetensors'))
    (directory/'path.json').write_text('{}')
    path=Compilation(12,0,directory.name,0,7,history,history.states,
        boundaries=[dict(records=1,episodes=1,actual_parameter_ref='MT300',row=dict(success=False)),
                    dict(records=2,episodes=2,actual_parameter_ref='G_000',row=dict(success=False))])
    context=SimpleNamespace(world_size=1,is_main=True,rank=0)
    register_students(root,[path],1,context)
    register_students(root,[path],1,context)  # Same unconsumed batch may be recovered.
    rows=json.loads((root/'G_student_bank.json').read_text())['rows'];assert len(rows)==2
    cache=HistoryCache();early=cache.get(rows[0]['parent']['histories'][0])
    assert len(early.records)==1 and 'b' not in early.observations and 'G_000' not in early.states
    (root/'run_contract.json').write_text('{}')
    for task in TASKS:
        folder=root/'teachers'/f'task_{task:04d}';folder.mkdir(parents=True)
        (folder/'E.json').write_text(json.dumps(dict(reference='E',weights='teacher')))
        bootstrap=root/'bootstrap'/f'task_{task:04d}';bootstrap.mkdir(parents=True)
        (bootstrap/'complete.json').write_text(json.dumps(dict(complete=True,reference='MT300',history='original')))
    trainer=CFMTrainer.__new__(CFMTrainer);trainer.root=root;trainer.runtime=SimpleNamespace()
    trainer.refresh_bank(0);assert len(trainer.bank[12]['parents'])==1
    trainer.refresh_bank(1);assert len(trainer.bank[12]['parents'])==2
    assert len(trainer.bank[12]['parents'][0]['histories'])==2
    assert (directory/'G_000.safetensors').exists() and (directory/'history.pt.gz').exists()
    path.bootstrap_provenance=dict(reused=True,source='original')
    register_students(root,[path],1,context);trainer.refresh_bank(1)
    assert trainer.bank[12]['parents'][0]['histories']==['original']  # Reuse does not multiply actual H weight.


def test_joint_resume_restores_both_optimizers_and_data_version(monkeypatch,tmp_path):
    from ember.proposal_writer import learning
    from ember.proposal_writer.contract import CHECKPOINT_SCHEMA
    generator=nn.Linear(1,1);actor=nn.Linear(1,1);baseline=nn.Linear(1,1)
    theta_optimizer=torch.optim.AdamW(actor.parameters());psi_optimizer=torch.optim.AdamW(generator.parameters())
    actor(torch.ones(1,1)).sum().backward();theta_optimizer.step()
    generator(torch.ones(1,1)).sum().backward();psi_optimizer.step()
    state=dict(schema_version=CHECKPOINT_SCHEMA,stage='RL',topology=dict(world_size=1),
        psi=generator.state_dict(),theta=actor.state_dict(),baseline=baseline.state_dict(),
        optimizer=theta_optimizer.state_dict(),generator_optimizer=psi_optimizer.state_dict(),
        psi_version=11,G_sampler=dict(next_update=11),supervision_version=9,next_update=9,rank_rng=[{}])
    torch.save(state,tmp_path/'state.pt')
    fresh=SimpleNamespace(generator=nn.Linear(1,1),actor=nn.Linear(1,1))
    new_theta=torch.optim.AdamW(fresh.actor.parameters());new_psi=torch.optim.AdamW(fresh.generator.parameters())
    new_baseline=nn.Linear(1,1);monkeypatch.setattr(learning,'restore_rng',lambda *_:None)
    result=learning.resume_checkpoint(tmp_path,'RL',fresh,new_theta,SimpleNamespace(world_size=1,rank=0),
        baseline=new_baseline,generator_optimizer=new_psi)
    assert result==9 and fresh.version==11 and fresh.G_sampler['next_update']==11 and fresh.supervision_version==9
    assert len(new_theta.state)==2 and len(new_psi.state)==2
    torch.testing.assert_close(fresh.generator.weight,generator.weight)
    torch.testing.assert_close(new_baseline.weight,baseline.weight)


def test_teacher_report_same_MT_alias_uses_one_real_episode(monkeypatch,tmp_path):
    import json
    from safetensors.torch import save_file
    from ember.proposal_writer import run
    from ember.writer.practice import History
    state={'factor':torch.ones(1)};weights=tmp_path/'initial.safetensors';save_file(state,str(weights))
    (tmp_path/'teacher_report_references.json').write_text(json.dumps(dict(references={})))
    monkeypatch.setattr(run,'select_teacher',lambda *_:dict(weights=str(weights),selected_update=0,reference='MT300'))
    monkeypatch.setattr(run,'task_panel',lambda *_:dict(teacher_report_states=[4]))
    class Runner:
        environment_steps=0;components={};requests=[]
        def run(self,requests):
            self.requests.extend(requests)
            for request in requests:
                self.environment_steps+=12
                yield dict(request=request,row=dict(success=True,init_state_id=4,environment_steps=12,
                    actual_parameter_ref='MT300'),history=History(records=[dict(normalized_actions=torch.zeros(5,7))]))
    runner=Runner();args=SimpleNamespace(root=tmp_path,task=12)
    runtime=SimpleNamespace(device=torch.device('cpu'),mt=state)
    run.teacher_report(args,runtime,runner,dict(tasks=[dict(global_task_id=12)]))
    record=json.loads((tmp_path/'teachers/task_0012/report/complete.json').read_text())
    assert len(runner.requests)==1 and record['logical_rows']==2 and record['actual_new_episodes_this_attempt']==1
    assert all(r['query_excluded_from_H'] and r['reference_provenance']['same_policy_E_MT_alias'] for r in record['rows'])


def test_formal_persistent_worker_claims_all400_without_reloading_kernel(monkeypatch,tmp_path):
    from ember.pi05_source_checkpoint import write_json_atomic
    from ember.proposal_writer import evaluation,pipeline
    checkpoint=tmp_path/'checkpoint';tasks=[dict(global_task_id=i,task_id=i,suite='fixture',horizon=220+100*(i%2)) for i in range(8)]
    panel=dict(checkpoint=str(checkpoint),specification=dict(version=['RL',9]),environment=dict(tasks=tasks),
        rows=[dict(task=i,init_state_id=j) for i in range(8) for j in range(50)])
    write_json_atomic(tmp_path/'formal_panel.json',panel)
    write_json_atomic(tmp_path/'run_contract.json',dict(condition_estimate_seconds=dict(formal=10)))
    base=evaluation.formal_queue(tmp_path,workers=6,condition_batch=4)
    plan=(base/'queue_plan.json').read_text()
    evaluation.formal_queue(tmp_path,workers=4,condition_batch=2)  # New physical workers/chunks reuse immutable shards.
    assert (base/'queue_plan.json').read_text()==plan
    from ember.proposal_writer.batch import stage_jobs
    jobs=stage_jobs(tmp_path,'formal',microbatch=28,slots=16,teacher_devices=(),estimates={},shared_gpus=None,
        stop=None,physical_workers=4,condition_batch=2)
    assert len(jobs)==4 and len({j.identity for j in jobs})==4
    assert all('--worker-id' in j.arguments and j.estimate_seconds==1000 for j in jobs)
    loaded=[];groups=[]
    monkeypatch.setattr(pipeline,'load_version',lambda runtime,path:loaded.append(path) or dict(stage='RL',next_update=9))
    def consume(args,runtime,runner,panel,rows):
        groups.append([(r['task'],r['init_state_id']) for r in rows])
        return [dict(success=False) for _ in rows]
    monkeypatch.setattr(evaluation,'formal_conditions',consume)
    args=SimpleNamespace(root=tmp_path,checkpoint=checkpoint,video_arm='correct',worker_id='fixture-persistent',condition_batch=2)
    result=evaluation.formal_worker(args,SimpleNamespace(actor=SimpleNamespace(eval=lambda:None)),None)
    assert loaded==[checkpoint] and result['completed_rows']==400
    assert max(map(len,groups))<=2 and len({row for group in groups for row in group})==400


def test_formal_group_keeps_paired_rng_and_reuses_selected_MT(monkeypatch,tmp_path):
    from ember.pi05_source_checkpoint import read_json
    from ember.proposal_writer import evaluation
    from ember.writer.practice import History
    checkpoint=tmp_path/'checkpoint';compiled=[]
    tasks=[dict(global_task_id=i,task_id=i,suite='fixture',horizon=220) for i in (0,1)]
    rows=[dict(task=i,init_state_id=i,teacher_demo=i+10,video_ordinal=i,practice_states=[3],noise_root=7,
        registered_scene=dict(root='registered'),MT_source='historical-MT',MT_row=dict(success=False,
            init_state_id=i,environment_steps=220)) for i in (0,1)]
    panel=dict(specification=dict(version=['RL',9],max_rounds=5,environment_multiplier=6),environment=dict(tasks=tasks))
    def compile_group(runtime,runner,specifications):
        compiled.extend(specifications);paths=[]
        for index,spec in enumerate(specifications):
            def save(destination,**_):destination.mkdir(parents=True,exist_ok=True)
            selected='G' if index==0 else 'MT300'
            paths.append(SimpleNamespace(identity=spec['identity'],selected=selected,states={selected:{}},
                history=History(episodes=[dict(init_state_id=3)]),environment_steps=100,
                timings=dict(new_environment_steps=100),save=save))
        return paths
    monkeypatch.setattr(evaluation,'compile_conditions',compile_group)
    class Runner:
        requests=[]
        def run(self,requests):
            self.requests.extend(requests)
            for request in requests:
                yield dict(request=request,row=dict(success=True,init_state_id=request['state_id'],environment_steps=14),
                    history=History(records=[dict(normalized_actions=torch.ones(5,7))]))
    runner=Runner();args=SimpleNamespace(root=tmp_path,checkpoint=checkpoint,video_arm='correct')
    results=evaluation.formal_conditions(args,None,runner,panel,rows)
    assert len(compiled)==2 and len(runner.requests)==1
    request=runner.requests[0]
    assert request['state_id']==0 and request['noise_root']==7 and request['registered_scene']==rows[0]['registered_scene']
    assert not request['capture_history'] and not request['capture_hidden']
    assert all(r['query_excluded_from_H'] and r['fixed_unique_LoRA'] for r in results)
    assert results[1]['fixed_parameter_result_reused_from']=='historical-MT'
    records=[read_json(tmp_path/'validation/RL_00000009/correct'/f'task_{i:04d}_state_{i:03d}'/'complete.json') for i in (0,1)]
    assert [r['actual_new_environment_steps'] for r in records]==[114,100]
    assert evaluation.formal_conditions(args,None,runner,panel,rows)==results and len(runner.requests)==1


def test_analysis_exception_requires_registered_live_child_and_preserves_ordinary_cap(monkeypatch,tmp_path):
    import json,threading,pytest
    from ember.pi05_source_checkpoint import write_json_atomic
    from ember.proposal_writer import batch as owner
    from ember.proposal_writer.batch import Batch,Job
    batch=object.__new__(Batch);batch.root=tmp_path
    folder=tmp_path/'profile_cost';folder.mkdir()
    launch=dict(analysis_exception_only=True,PID=73,node='gpu02',physical_gpu=4)
    write_json_atomic(folder/'launch.json',launch)
    devices=[('gpu02',dict(index=4,processes=[dict(pid=73,owner='ymdai')]))]
    assert batch.bounded_analysis_devices(devices)=={('gpu02',4)}
    assert not batch.bounded_analysis_devices([('gpu02',dict(index=4,processes=[dict(pid=74,owner='ymdai')]))])
    def gpu(index,pid=None):
        return dict(index=index,processes=[] if pid is None else [dict(pid=pid,owner='ymdai')],
            mem_used_mib=0 if pid is None else 15000,mem_free_mib=32000,compute_mode='Default',uuid=str(index),util_gpu_pct=100)
    snapshot=dict(bad_nodes=[],node_results=dict(gpu01=dict(gpus=[gpu(i,100+i if i<5 else None) for i in range(6)]),
        gpu02=dict(gpus=[gpu(4,73)])))
    def command(arguments,**_):
        return SimpleNamespace(stdout=json.dumps(snapshot) if 'gpu_preflight.py' in arguments[1] else '/dev/fixture 1000 2147483648 2147483648\n')
    monkeypatch.setattr(owner.subprocess,'run',command)
    batch.running={('gpu01',i):{} for i in range(5)};batch.required_free_MiB=18000;batch.lock=threading.Lock()
    batch.track_cost_estimate=lambda *_:dict(GPU_hours_hard_cap_active=False)
    write_json_atomic(tmp_path/'run_contract.json',dict(limits=dict(added_peak_GiB=256)))
    destination=tmp_path/'admit';destination.mkdir()
    admission=batch.admit(Job('actual-normal',['teacher'],estimate_seconds=10),'gpu01',5,destination)
    assert admission['concurrency_cap']==6 and admission['ordinary_physical_GPU_count']==6
    assert admission['physical_GPU_count_including_analysis']==7 and admission['active_bounded_analysis_devices']==['gpu02:4']
    write_json_atomic(folder/'exit.json',dict(exit_code=0))
    assert not batch.bounded_analysis_devices(devices)
    with pytest.raises(RuntimeError,match='outside project allocation'):
        batch.admit(Job('unregistered-extra',['teacher'],estimate_seconds=10),'gpu01',5,destination)
