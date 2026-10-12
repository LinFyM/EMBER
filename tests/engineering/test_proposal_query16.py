"""Finite query credit, immutable MT reuse and score-only E consumer regressions."""
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.proposal_writer import supervision as q


def condition(root, task_id=13):
    path=SimpleNamespace(task=task_id,identity=f'task_{task_id:04d}',demo=47,selected='G_000',
        states={'MT300':{'factor':torch.zeros(1)},'G_000':{'factor':torch.ones(1)}},
        history=SimpleNamespace(episodes=[{'init_state_id':100}],records=[]),
        outputs=[{'candidates':['MT300','G_000']},{'candidates':['MT300','G_000']}])
    task=dict(suite='libero_object',task_id=task_id-10,language='exact task language',horizon=280)
    panel=dict(task_id=task_id,query_states=list(range(16)),query_count=16,teacher_selection_states=[14,15])
    return SimpleNamespace(root=root),path,task,panel


def native_evidence(task,state,noise):
    trace=[q.policy_noise_seed(noise,task['suite'],task['task_id'],state,i) for i in range(3)]
    return dict(first_noise_seed=trace[0],policy_noise_seeds=trace,initial_proprio=[0.]*8)


class Runner:
    def __init__(self):self.requests=[]
    def run(self,requests):
        for request in requests:
            self.requests.append(request)
            state=request['state_id'];ref=request['parameter_ref'];task=request['task']
            success=state%2==0 if ref=='MT300' else state%3==0
            row=dict(init_state_id=state,success=success,
                env_seed=request['noise_root'],policy_seed_root=request['noise_root'],environment_steps=17,
                **native_evidence(task,state,request['noise_root']),
                **{key:task[key] for key in ('suite','task_id','language')})
            yield dict(request=request,row=row,history=SimpleNamespace(records=[]))


def test_query_uses_real_cardinality_once_and_restores(tmp_path):
    runtime,path,task,panel=condition(tmp_path);runner=Runner();destination=tmp_path/'student/query'
    rewards=q.query_once(runtime,runner,path,task,panel,0,destination)
    assert len(runner.requests)==32 and rewards=={'MT300':8/16,'G_000':6/16}
    assert len({(r['parameter_ref'],r['state_id']) for r in runner.requests})==32
    assert all(not r['capture_history'] and not r['capture_hidden'] for r in runner.requests)
    saved=read_json(destination/'complete.json')
    assert saved['query_count']==16 and saved['actual_new_episodes_total']==32
    assert q.query_once(runtime,runner,path,task,panel,0,destination)==rewards and len(runner.requests)==32
    saved['rows']=saved['rows'][:8];saved['rewards']={'MT300':.25,'G_000':.25}
    write_json_atomic(destination/'complete.json',saved)
    assert q.query_once(runtime,runner,path,task,panel,0,destination)==rewards and len(runner.requests)==32
    with pytest.raises(ValueError,match='distinct'):q.query_panel(dict(query_states=[1,1],query_count=2),0)
    assert q.query_panel(dict(query_states=list(range(16))),7)==list(range(16))


def test_MT_external_reuse_checks_sources_and_rng(tmp_path):
    root,source=tmp_path/'new',tmp_path/'previous';root.mkdir();source.mkdir()
    contract=dict(source={'checkpoint':'source'},mt_checkpoint='/existing/MT',environment={'scene':'official'})
    for directory in (root,source):write_json_atomic(directory/'run_contract.json',contract)
    for relative in ('initial.safetensors','base/merged_targets.safetensors'):
        original=source/relative;original.parent.mkdir(exist_ok=True);original.write_bytes(b'identity')
        target=root/relative;target.parent.mkdir(exist_ok=True);target.symlink_to(original)
    runtime,path,task,panel=condition(root);noise=q.seed(70,path.task,0)
    for state in range(4):
        write_json_atomic(source/'student_data/previous/task_0013/query'/f'MT300_{state:03d}.json',
            dict(global_task_id=13,init_state_id=state,reference='MT300',success=state%2==0,
                env_seed=noise,policy_seed_root=noise,environment_steps=17,
                **native_evidence(task,state,noise),
                **{key:task[key] for key in ('suite','task_id','language')}))
    bad=source/'student_data/previous/task_0013/query/MT300_003.json'
    row=read_json(bad);row['policy_seed_root']+=1;write_json_atomic(bad,row)
    runner=Runner();destination=root/'student/query'
    q.query_once(runtime,runner,path,task,panel,0,destination,mt_reference_version='previous',mt_reference_root=source)
    saved=read_json(destination/'complete.json')
    assert len(runner.requests)==29 and len(saved['row_reuse'])==3
    assert saved['actual_new_episodes_total']==29 and saved['incompatible_MT_references'][0]['fields']==['policy_seed_root']
    assert all(r['parameter_ref']=='MT300' or not read_json(destination/f"{r['parameter_ref']}_{r['state_id']:03d}.json").get('reused')
               for r in runner.requests)
    (root/'initial.safetensors').unlink();(root/'initial.safetensors').write_bytes(b'new asset')
    assert q.MT_source_compatibility(root,source)==['initial.safetensors']


def test_readonly_E_reuses_MT_without_entering_credit_or_H(tmp_path):
    runtime,path,task,panel=condition(tmp_path);runner=Runner();destination=tmp_path/'student'
    q.query_once(runtime,runner,path,task,panel,0,destination/'query')
    refs=list(path.states);history=list(path.history.episodes)
    teacher=dict(reference='MT300',weights='/same/initial.safetensors')
    result=q.readonly_E_query(runtime,runner,path,task,panel,0,destination/'readonly_E',teacher)
    assert len(runner.requests)==32 and result['actual_new_episodes_total']==0
    assert result['rewards']=={'E':.5} and len(result['rows'])==16 and result['excluded_from_Actor_credit']
    assert set(read_json(destination/'query/complete.json')['rewards'])=={'MT300','G_000'}
    assert list(path.states)==refs and path.history.episodes==history
    assert all(r['source_row'] for r in result['rows'])
    write_json_atomic(destination/'parent.json',dict(complete=True,actual_generated_parameter_ref='G_000',common_E_reference='MT300',common_E_weights='/same/initial.safetensors'))
    assert q.student_data_complete(destination,panel,13,require_E=True)
    (destination/'readonly_E/complete.json').unlink()
    assert not q.student_data_complete(destination,panel,13,require_E=True)
    assert q.student_data_complete(destination,panel,13,require_E=False)
    q.readonly_E_query(runtime,runner,path,task,panel,0,destination/'readonly_E',teacher)
    assert q.student_data_complete(destination,panel,13,require_E=True) and len(runner.requests)==32


def test_nonzero_E_uses_identical_panel_without_practice_capture(tmp_path):
    runtime,path,task,panel=condition(tmp_path);runner=Runner();destination=tmp_path/'student'
    q.query_once(runtime,runner,path,task,panel,0,destination/'query')
    file=tmp_path/'E.safetensors';save_file({'factor':torch.full((1,),2.)},str(file))
    teacher=dict(reference='E_fixed',weights=str(file));refs=set(path.states)
    result=q.readonly_E_query(runtime,runner,path,task,panel,0,destination/'readonly_E',teacher)
    assert len(runner.requests)==48 and result['actual_new_episodes_total']==16
    assert result['actual_new_environment_steps_total']==16*17 and result['rewards']['E']==6/16
    assert set(path.states)==refs and 'E_fixed' not in refs
    mt={r['state_id']:r for r in runner.requests if r['parameter_ref']=='MT300'}
    for request in runner.requests[32:]:
        assert request['noise_root']==mt[request['state_id']]['noise_root']
        assert not request['capture_history'] and not request['capture_hidden'] and not request['uses_teaching']
    q.readonly_E_query(runtime,runner,path,task,panel,0,destination/'readonly_E',teacher)
    assert len(runner.requests)==48


def test_first_student_summary_preserves_category_distribution(tmp_path):
    from ember.proposal_writer.analysis import student_summary
    runtime,path,task,panel=condition(tmp_path);version='first';destination=tmp_path/'student_data'/version/'task_0013'
    runner=Runner();q.query_once(runtime,runner,path,task,panel,0,destination/'query')
    q.readonly_E_query(runtime,runner,path,task,panel,0,destination/'readonly_E',dict(reference='MT300',weights='initial'))
    write_json_atomic(destination/'parent.json',dict(complete=True,actual_generated_parameter_ref='G_000',common_E_reference='MT300',common_E_weights='initial'))
    write_json_atomic(tmp_path/'panel.json',dict(tasks=[panel]))
    write_json_atomic(tmp_path/'run_contract.json',dict(student_refresh=dict(version=version,query_states_per_parameter=16),limits={}))
    result=student_summary(tmp_path,tasks=[13])
    assert result['MT_G']['rows']==16 and result['MT_G']['gained']==3 and result['MT_G']['lost']==5
    assert result['MT_E']['net']==0 and result['G_by_teacher_MT_category']['E_only']['rows']==0
    assert result['G_by_teacher_MT_category']['both_success']=={'rows':8,'G_success':3}
    assert result['logical_rows']==48 and result['query_cost']['unique_new_episodes']==32
    assert result['query_cost']['reused_logical_rows']==16 and result['tasks'][0]['teacher_selection_overlap_states']==[14,15]
    assert (tmp_path/'analysis/first_student_query16_rows.jsonl').is_file()


def test_refresh_resumes_missing_E_despite_parent_receipt(tmp_path,monkeypatch):
    from ember.proposal_writer import pipeline
    runtime,path,task,panel=condition(tmp_path);version='first';destination=tmp_path/'student_data'/version/'task_0013'
    runner=Runner();q.query_once(runtime,runner,path,task,panel,0,destination/'query')
    write_json_atomic(destination/'parent.json',dict(complete=True,actual_generated_parameter_ref='G_000',common_E_reference='MT300',common_E_weights='initial'))
    write_json_atomic(destination/'path.json',{})
    write_json_atomic(tmp_path/'teachers/task_0013/E.json',dict(reference='MT300',weights='initial'))
    write_json_atomic(tmp_path/'panel.json',dict(tasks=[panel]))
    write_json_atomic(tmp_path/'run_contract.json',dict(student_refresh=dict(version=version,G_update=216,readonly_E_query=True)))
    path.completed_rounds=1;path.boundaries=[dict(actual_parameter_ref='G_000')]
    path.history.states=path.states;path.kernel_identity={};path.timings={}
    monkeypatch.setattr(pipeline,'load_version',lambda *args,**kwargs:dict(stage='G',next_update=216))
    monkeypatch.setattr(pipeline,'restore_compilation',lambda *args:path)
    args=SimpleNamespace(root=tmp_path,task=13,checkpoint=tmp_path/'G216')
    pipeline.refresh(args,runtime,runner,dict(tasks=[dict(global_task_id=13,**task)]))
    assert (destination/'readonly_E/complete.json').exists() and len(runner.requests)==32
    assert read_json(destination/'parent.json')['readonly_E_complete']
    monkeypatch.setattr(pipeline,'restore_compilation',lambda *args:pytest.fail('complete student must not be recompiled'))
    pipeline.refresh(args,runtime,runner,dict(tasks=[dict(global_task_id=13,**task)]))


def previous_control_fixture(tmp_path,*,teacher_MT=False):
    root,source=tmp_path/'new',tmp_path/'previous';root.mkdir();source.mkdir()
    shared=dict(source={'checkpoint':'source'},mt_checkpoint='/existing/MT',environment={'scene':'official'})
    for relative in ('initial.safetensors','base/merged_targets.safetensors'):
        original=source/relative;original.parent.mkdir(exist_ok=True);save_file({'factor':torch.zeros(1)},str(original))
        target=root/relative;target.parent.mkdir(exist_ok=True);target.symlink_to(original)
    write_json_atomic(source/'run_contract.json',dict(**shared,limits={},
        student_refresh=dict(version='previous',query_states_per_parameter=16)))
    write_json_atomic(root/'run_contract.json',dict(**shared,limits={},previous_root=str(source),previous_student_version='previous',
        student_refresh=dict(version='endpoint',query_states_per_parameter=16,readonly_previous_G_query=True)))
    runtime,path,task,panel=condition(source);previous=source/'student_data/previous/task_0013'
    runner=Runner();q.query_once(runtime,runner,path,task,panel,0,previous/'query')
    weights=source/'E.safetensors';save_file({'factor':torch.full((1,),2.)},str(weights))
    teacher=dict(reference='MT300' if teacher_MT else 'E_fixed',
        weights=str(source/'initial.safetensors' if teacher_MT else weights))
    q.readonly_E_query(runtime,runner,path,task,panel,0,previous/'readonly_E',teacher)
    weights=previous/'G_000.safetensors';save_file({'factor':torch.ones(1)},str(weights))
    write_json_atomic(previous/'parent.json',dict(complete=True,actual_generated_parameter_ref='G_000',
        common_E_reference=teacher['reference'],common_E_weights=teacher['weights'],
        parent={'weights':str(weights)},kernel_identity={'checkpoint':'actual_previous_G216'}))
    meta=dict(task_id=13,teacher_demo=path.demo,path_ordinal=0,noise_root=123,identity=path.identity)
    write_json_atomic(previous/'path.json',meta)
    write_json_atomic(root/'panel.json',dict(tasks=[panel]))
    runtime,path,task,panel=condition(root)
    return runtime,path,task,panel,teacher,source,meta


@pytest.mark.parametrize('teacher_MT',[False,True])
def test_endpoint_reuses_full_MT_E_and_reads_real_previous_G(tmp_path,teacher_MT):
    from ember.proposal_writer.analysis import student_summary
    runtime,path,task,panel,teacher,source,meta=previous_control_fixture(tmp_path,teacher_MT=teacher_MT)
    destination=runtime.root/'student_data/endpoint/task_0013';runner=Runner()
    q.query_once(runtime,runner,path,task,panel,0,destination/'query',mt_reference_root=source,mt_reference_version='previous')
    result=q.readonly_E_query(runtime,runner,path,task,panel,0,destination/'readonly_E',teacher)
    assert len(runner.requests)==16 and {r['parameter_ref'] for r in runner.requests}=={'G_000'}
    assert result['actual_new_episodes_total']==0 and len(result['row_reuse'])==16
    assert not result['incompatible_E_references']
    write_json_atomic(destination/'parent.json',dict(complete=True,actual_generated_parameter_ref='G_000',
        common_E_reference=teacher['reference'],common_E_weights=teacher['weights']))
    write_json_atomic(destination/'path.json',meta)
    summary=student_summary(runtime.root,tasks=[13])
    assert summary['logical_rows']==64 and summary['MT_oldG']['rows']==16 and summary['oldG_G']['net']==0
    assert summary['query_cost']['unique_new_episodes']==16 and summary['query_cost']['unique_new_environment_steps']==16*17
    assert summary['query_cost']['reused_logical_rows']==32 and summary['query_cost']['historical_comparator_rows']==16
    import json
    rows=[json.loads(line) for line in (runtime.root/'analysis/first_student_query16_rows.jsonl').read_text().splitlines()]
    previous=[row for row in rows if row['readout_arm']=='oldG']
    assert len(previous)==16 and all(row['source_kernel_identity']=={'checkpoint':'actual_previous_G216'} for row in previous)
    assert all(str(source/'student_data/previous/task_0013/query') in row['actual_episode_row_file'] for row in previous)
    assert all(row['excluded_from_Actor_credit'] and row['excluded_from_H'] for row in previous)
    assert set(read_json(destination/'query/complete.json')['rewards'])=={'MT300','G_000'}
    assert set(path.states)=={'MT300','G_000'}
    if teacher_MT:
        assert all(row['source_row']==str(destination/'query'/f"MT300_{row['init_state_id']:03d}.json") for row in result['rows'])
    q.query_once(runtime,runner,path,task,panel,0,destination/'query',mt_reference_root=source,mt_reference_version='previous')
    (destination/'readonly_E/complete.json').unlink()
    q.readonly_E_query(runtime,runner,path,task,panel,0,destination/'readonly_E',teacher)
    assert len(runner.requests)==16


def test_direct_endpoint_query_reuses_controls_without_practice_or_actor(tmp_path):
    runtime,path,task,panel,teacher,source,_=previous_control_fixture(tmp_path)
    path.sampling_identity='endpoint_query_only';path.completed_rounds=0
    path.boundaries=[];path.decisions=[];path.outputs=[];path.practiced=['MT300']
    path.valid=['MT300','G_000'];path.attempts=[dict(reference='G_000')]
    destination=runtime.root/'student_data/endpoint/task_0013';runner=Runner()
    reward=q.query_once(runtime,runner,path,task,panel,0,destination/'query',
        mt_reference_root=source,mt_reference_version='previous',endpoint_only=True)
    q.readonly_E_query(runtime,runner,path,task,panel,0,destination/'readonly_E',teacher)
    assert len(runner.requests)==16 and {r['parameter_ref'] for r in runner.requests}=={'G_000'}
    assert {r['state_id'] for r in runner.requests}==set(panel['query_states'])
    assert all(r['noise_root']==q.seed(70,path.task,0) and not r['uses_teaching'] for r in runner.requests)
    receipt=read_json(destination/'query/complete.json')
    assert receipt['unpracticed_fixed_endpoint_query'] and receipt['new_G_practice_episodes']==0
    assert not receipt['unpracticed_candidates_not_queried'] and reward['G_000']==6/16
    q.query_once(runtime,runner,path,task,panel,0,destination/'query',endpoint_only=True)
    assert len(runner.requests)==16  # Recovery consumes the saved endpoint's missing rows only.
    path.boundaries=[dict(actual_parameter_ref='G_000')]
    with pytest.raises(ValueError,match='unpracticed endpoint'):
        q.query_once(runtime,runner,path,task,panel,0,destination/'query',endpoint_only=True)


def test_direct_endpoint_queue_keeps_full36_query16_and_omits_practice(tmp_path):
    from ember.proposal_writer.evaluation import student_queue
    from ember.proposal_writer.contract import TASKS
    from ember.pi05_eval_queue import queue_summary
    tasks=[dict(global_task_id=t,suite='libero_90',task_id=t,horizon=400) for t in TASKS]
    panel=[dict(task_id=t,query_states=list(range(16)),query_count=16) for t in TASKS]
    write_json_atomic(tmp_path/'panel.json',dict(tasks=panel))
    write_json_atomic(tmp_path/'run_contract.json',dict(git={'commit':'frozen'},environment={'tasks':tasks},
        student_refresh=dict(version='direct',G_update=864,query_states_per_parameter=16,
            readout_mode='endpoint_query_only',post_generation_practice=False,condition_mapping={})))
    base=student_queue(tmp_path);plan=read_json(base/'queue_plan.json')
    assert plan['new_episodes']==576 and plan['new_G_practice_episodes']==0
    assert len(plan['shards'])==36 and all(len(s['init_state_ids'])==16 for s in plan['shards'])
    assert queue_summary(base/'jobs.sqlite')['status_counts']=={'pending':36}


def test_endpoint_recomputes_only_incompatible_E_row(tmp_path):
    runtime,path,task,panel,teacher,source,_=previous_control_fixture(tmp_path)
    file=source/'student_data/previous/task_0013/readonly_E/E_003.json'
    row=read_json(file);row['policy_noise_seeds'][1]+=1;write_json_atomic(file,row)
    destination=runtime.root/'student_data/endpoint/task_0013';runner=Runner()
    q.query_once(runtime,runner,path,task,panel,0,destination/'query',mt_reference_root=source,mt_reference_version='previous')
    result=q.readonly_E_query(runtime,runner,path,task,panel,0,destination/'readonly_E',teacher)
    assert len(runner.requests)==17 and result['actual_new_episodes_total']==1
    assert result['actual_new_environment_steps_total']==17 and len(result['row_reuse'])==15
    assert result['incompatible_E_references']==[dict(source=str(file),fields=['policy_noise_seeds','policy_noise_common_prefix'])]
