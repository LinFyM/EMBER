"""Actual student prefixes and once-per-parameter independent query labels."""
from __future__ import annotations

from pathlib import Path
import time

import torch.distributed as dist
from safetensors.torch import load_file

from ember.pi05_source_checkpoint import read_json,write_json_atomic
from .contract import TASKS,seed
from .path import fixed_requests,fixed_row,restore_compilation


def register_students(root,paths,update,context):
    """Index true visited P/H prefixes without copying their histories or querying labels."""
    root=Path(root);local=[]
    for path in paths:
        directory=root/'RL_paths'/path.identity
        if not (directory/'path.json').exists():raise ValueError('student supervision needs retained actual source path')
        nodes=list(path.boundaries)
        if not any(node['episodes']==1 for node in nodes):
            bootstrap=[row for row in path.history.records if row['episode']==0]
            if not bootstrap or any(row['parameter_ref']!='MT300' for row in bootstrap):
                raise ValueError('student supervision needs the actual MT bootstrap prefix')
            nodes.insert(0,dict(records=len(bootstrap),episodes=1,actual_parameter_ref='MT300',
                row=path.history.episodes[0]))
        for node in nodes:
            ref=node['actual_parameter_ref']
            if node['row'].get('model_failure') or node['row'].get('budget_truncated'):
                raise ValueError('incomplete practice cannot enter complete joint supervision')
            descriptor=dict(path=str(directory/'history.pt.gz'),records=node['records'],episodes=node['episodes'])
            if ref=='MT300' and node['episodes']==1 and path.bootstrap_provenance.get('reused'):
                descriptor=path.bootstrap_provenance['source']
            weights=root/'initial.safetensors' if ref=='MT300' else directory/f'{ref}.safetensors'
            if not weights.exists():raise ValueError('student parameters must survive selector consumption')
            parent_ref='MT300' if ref=='MT300' else f'{path.identity}:{ref}'
            local.append(dict(task=path.task,joint_update=update,source_path=str(directory/'path.json'),
                source_parameter_ref=ref,kernel_identity=path.kernel_identity,
                parent=dict(reference=parent_ref,weights=str(weights),histories=[descriptor])))
    gathered=[None]*context.world_size if context.is_main else None
    if context.world_size>1:dist.gather_object(local,gathered,dst=0)
    else:gathered=[local]
    if context.is_main:
        file=root/'G_student_bank.json';previous=read_json(file)['rows'] if file.exists() else []
        # Recovered, unconsumed batches retain their original rows; no duplicate parent weight.
        retained=[r for r in previous if r['joint_update']!=update]
        rows=retained+[row for rank in gathered for row in rank]
        write_json_atomic(file,dict(complete=True,version=update,rows=rows,
            common_E_unchanged=True,scale_reestimated=False,query_not_in_H=True,
            history_storage='one original path; prefix views by actual records/episodes',
            parent_sampling='equal actual parameter; H uniform within parent'))
    if context.world_size>1:dist.barrier(device_ids=[context.local_rank])
    return update


def panels(root):
    return {row['task_id']:row for row in read_json(Path(root)/'panel.json')['tasks']}


def bootstrap_source(root,task,panel,ordinal,cache):
    """First visit reuses registered MT practice; later visits really execute MT."""
    if ordinal!=0:return None,None
    row=read_json(Path(root)/'bootstrap'/f'task_{task:04d}'/'complete.json')
    if not row['complete'] or row['reference']!='MT300' or row['state_id']!=panel['bootstrap_state_id']:
        raise ValueError('fixed initial MT bootstrap identity changed')
    return cache.get(row['history']),row['history']


def query_panel(panel,visit):
    pool=panel['query_states']
    count=panel.get('query_count',len(pool))
    if not isinstance(count,int) or isinstance(count,bool) or count<1 or count>len(pool) or len(set(pool))!=len(pool):
        raise ValueError('query requires a positive count of distinct registered initial states')
    return [pool[(count*visit+i)%len(pool)] for i in range(count)]


def query_once(runtime,runner,path,task,panel,visit,destination,**options):
    return query_conditions(runtime,runner,[(path,task,panel,visit,destination)],**options)[0]


def complete_query(saved,states,noise_root,references):
    expected={(ref,state) for ref in references for state in states}
    rows=saved.get('rows',[])
    actual={(row.get('reference'),row.get('init_state_id')) for row in rows}
    return (saved.get('complete') and saved.get('query_states')==states
        and saved.get('policy_env_noise_root')==noise_root and set(saved.get('rewards',{}))==set(references)
        and len(rows)==len(expected) and actual==expected
        and all((row.get('env_seed'),row.get('policy_seed_root'))==(noise_root,noise_root) for row in rows)
        and all(abs(saved['rewards'][ref]-sum(bool(row['success']) for row in rows if row['reference']==ref)/len(states))<1e-12
                for ref in references))


def query_row(file):
    visited=set();file=Path(file)
    while True:
        if file in visited:raise ValueError('cyclic query row reference')
        visited.add(file);row=read_json(file)
        if 'source_row' not in row:return row
        file=Path(row['source_row'])


def MT_source_compatibility(root,source_root):
    """Immutable asset references and the official scene contract, without tensor hashes."""
    root,source_root=Path(root),Path(source_root)
    if root.resolve()==source_root.resolve():return []
    current,previous=read_json(root/'run_contract.json'),read_json(source_root/'run_contract.json')
    changed=[key for key in ('source','mt_checkpoint','environment') if current.get(key)!=previous.get(key)]
    for relative in ('initial.safetensors','base/merged_targets.safetensors'):
        left,right=root/relative,source_root/relative
        if not left.is_file() or not right.is_file() or left.resolve()!=right.resolve():changed.append(relative)
    return changed


def reuse_initial_MT(entry,task,root,version,*,source_root=None):
    """A single registered second consumer references compatible initial MT rows."""
    path=entry['path'];source_root=Path(source_root) if source_root is not None else Path(root)
    source=source_root/'student_data'/version/f'task_{path.task:04d}'/'query'
    changed=MT_source_compatibility(root,source_root)
    if changed:
        entry['incompatible_references'].append(dict(source=str(source),fields=changed));return
    for state in entry['states']:
        target=entry['destination']/f'MT300_{state:03d}.json';original=source/target.name
        if target.exists() or not original.exists():continue
        row=query_row(original)
        expected=dict(global_task_id=path.task,init_state_id=state,reference='MT300',
            env_seed=entry['noise_root'],policy_seed_root=entry['noise_root'],
            **{key:task[key] for key in ('suite','task_id','language')})
        changed=[key for key,value in expected.items() if row.get(key)!=value]
        if changed:
            entry['incompatible_references'].append(dict(source=str(original),fields=changed));continue
        write_json_atomic(target,dict(reused=True,source_row=str(original),
            logical_consumer=path.identity,actual_episode_identity_unchanged=True,
            reference='MT300',init_state_id=state,actual_new_episodes=0,actual_new_environment_steps=0,
            equivalence='same immutable MT, task/state/official scene/env-policy RNG; identity only names output'))


def query_conditions(runtime,runner,conditions,*,mt_reference_version=None,mt_reference_root=None):
    """Batch only missing independent λ/state/RNG queries, never their histories."""
    tick=time.monotonic();entries=[];requests=[];owners={}
    for path,task,panel,visit,destination in conditions:
        destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
        states=query_panel(panel,visit);noise_root=seed(70,path.task,visit)
        references=list(dict.fromkeys(ref for node in path.outputs for ref in node['candidates']))
        if {r['init_state_id'] for r in path.history.episodes}.intersection(states):raise ValueError('query initial states leaked into practice')
        entry=dict(path=path,destination=destination,states=states,noise_root=noise_root,references=references,
            new_steps=0,new_episodes=0,reused_rows=[],incompatible_references=[])
        entries.append(entry)
        complete=destination/'complete.json'
        if complete.exists():
            saved=read_json(complete)
            if complete_query(saved,states,noise_root,references):
                entry['completed_query']=saved
                continue  # Restore the immutable panel and its original environment ledger.
        if visit==0 and mt_reference_version is not None and 'MT300' in references:
            reuse_initial_MT(entry,task,runtime.root,mt_reference_version,source_root=mt_reference_root)
        missing=fixed_requests(path,task,states,noise_root,references=references,
            existing_rows={(ref,state) for ref in references for state in states if (destination/f'{ref}_{state:03d}.json').exists()})
        for request in missing:
            if request['episode_id'] in owners:raise ValueError('query episode identity reused across conditions')
            owners[request['episode_id']]=entry
        requests.extend(missing)
    for output in runner.run(sorted(requests,key=lambda r:r['task']['horizon'],reverse=True)):
        entry=owners[output['request']['episode_id']];row=fixed_row(entry['path'],output)
        write_json_atomic(entry['destination']/f"{row['reference']}_{row['init_state_id']:03d}.json",row)
        entry['new_steps']+=int(row.get('environment_steps',0));entry['new_episodes']+=1
    results=[]
    for entry in entries:
        if 'completed_query' in entry:
            results.append(entry['completed_query']['rewards']);continue
        destination=entry['destination'];references=entry['references'];states=entry['states']
        rows=[];new_rows=[]
        for ref in references:
            for state in states:
                file=destination/f'{ref}_{state:03d}.json';row=read_json(file)
                if 'source_row' in row:
                    entry['reused_rows'].append(dict(destination=str(file),**row));row=query_row(file)
                else:new_rows.append(row)
                rows.append(row)
        expected={(ref,state) for ref in references for state in states}
        if len(rows)!=len(expected) or {(r['reference'],r['init_state_id']) for r in rows}!=expected:
            raise ValueError('query cache must contain each immutable parameter/state exactly once')
        if any((r.get('env_seed'),r.get('policy_seed_root'))!=(entry['noise_root'],entry['noise_root']) for r in rows):
            raise ValueError('query cache changed registered environment/policy RNG')
        rewards={ref:sum(bool(r['success']) for r in rows if r['reference']==ref)/len(states) for ref in references}
        write_json_atomic(destination/'complete.json',dict(complete=True,rewards=rewards,rows=rows,query_states=states,
            query_count=len(states),logical_rows=len(rows),
            policy_env_noise_root=entry['noise_root'],query_excluded_from_H=True,each_parameter_query_once_across_prefixes=True,
            unpracticed_candidates_not_queried=True,actual_new_environment_steps_this_attempt=entry['new_steps'],
            actual_new_episodes_this_attempt=entry['new_episodes'],shared_query_group_seconds=time.monotonic()-tick,
            actual_new_episodes_total=len(new_rows),actual_new_environment_steps_total=sum(int(r.get('environment_steps',0)) for r in new_rows),
            row_reuse=entry['reused_rows'],incompatible_MT_references=entry['incompatible_references'],
            shared_group_members=[p.identity for p,*_ in conditions]))
        results.append(rewards)
    return results


def readonly_E_query(runtime,runner,path,task,panel,visit,destination,teacher):
    """Score a fixed teacher on the same panel, outside all compiler/Actor states and credit."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    states=query_panel(panel,visit);noise_root=seed(70,path.task,visit);tick=time.monotonic()
    credit=read_json(destination.parent/'query/complete.json')
    if not complete_query(credit,states,noise_root,credit['rewards']) or 'MT300' not in credit['rewards']:
        raise ValueError('readonly E requires a complete matching MT/G credit panel')
    receipt=destination/'complete.json'
    if receipt.exists():
        saved=read_json(receipt)
        if (complete_query(saved,states,noise_root,['E']) and saved.get('teacher_reference')==teacher['reference']
                and saved.get('teacher_weights')==teacher['weights'] and saved.get('excluded_from_Actor_credit')):
            return saved
    mt_rows={row['init_state_id']:row for row in credit['rows'] if row['reference']=='MT300'}
    requests=[];reused=[];new_steps=0;new_episodes=0
    weights=None
    for state in states:
        file=destination/f'E_{state:03d}.json'
        if file.exists():continue
        if teacher['reference']=='MT300':
            source=destination.parent/'query'/f'MT300_{state:03d}.json'
            write_json_atomic(file,dict(reused=True,source_row=str(source),reference='E',init_state_id=state,
                teacher_parameter_reference='MT300',actual_new_episodes=0,actual_new_environment_steps=0))
            reused.append(str(source))
        else:
            if weights is None:weights=load_file(teacher['weights'])
            requests.append(dict(task=task,state=weights,parameter_ref=teacher['reference'],state_id=state,
                noise_root=noise_root,snapshots=False,uses_teaching=False,
                episode_id=f'{path.identity}_readonly_E_query_{state}',remaining=task['horizon']+10,
                arm='E',capture_history=False,capture_hidden=False))
    for output in runner.run(requests):
        row=fixed_row(path,output);state=row['init_state_id']
        row.update(reference='E',teacher_parameter_reference=teacher['reference'],readonly_teacher_control=True,
            excluded_from_Actor_credit=True,not_a_compiler_candidate=True)
        write_json_atomic(destination/f'E_{state:03d}.json',row)
        new_steps+=int(row.get('environment_steps',0));new_episodes+=1
    rows=[];new_rows=[]
    for state in states:
        row=read_json(destination/f'E_{state:03d}.json')
        if 'source_row' in row:
            row=dict(mt_rows[state],reference='E',arm='E',teacher_parameter_reference='MT300',
                readonly_teacher_control=True,excluded_from_Actor_credit=True,not_a_compiler_candidate=True,
                source_row=row['source_row'],reused=True)
        else:new_rows.append(row)
        rows.append(row)
    result=dict(complete=True,schema_version='ember_proposal_readonly_E_query_v1',rows=rows,query_states=states,
        query_count=len(states),rewards={'E':sum(bool(row['success']) for row in rows)/len(states)},
        policy_env_noise_root=noise_root,teacher_reference=teacher['reference'],teacher_weights=teacher['weights'],
        excluded_from_Actor_credit=True,excluded_from_H=True,not_a_compiler_candidate=True,
        score_did_not_select_teacher_or_model=True,actual_new_episodes_this_attempt=new_episodes,
        actual_new_environment_steps_this_attempt=new_steps,reused_MT_source_rows=reused,
        actual_new_episodes_total=len(new_rows),actual_new_environment_steps_total=sum(int(r.get('environment_steps',0)) for r in new_rows),
        wall_seconds_this_attempt=time.monotonic()-tick)
    if any(row.get('teacher_parameter_reference')!=teacher['reference'] for row in rows):
        raise ValueError('readonly E rows belong to a different fixed teacher')
    if not complete_query(result,states,noise_root,['E']):raise ValueError('readonly E panel incomplete or unpaired')
    write_json_atomic(receipt,result)
    return result


def student_data_complete(destination,panel,task,*,require_E):
    """A parent receipt alone cannot skip an incomplete query or readonly control."""
    destination=Path(destination);states=query_panel(panel,0);noise_root=seed(70,task,0)
    parent_file,query_file=destination/'parent.json',destination/'query/complete.json'
    if not parent_file.exists() or not query_file.exists():return False
    parent,query=read_json(parent_file),read_json(query_file)
    if not parent.get('complete') or not complete_query(query,states,noise_root,['MT300',parent['actual_generated_parameter_ref']]):
        return False
    if require_E:
        file=destination/'readonly_E/complete.json'
        if not file.exists():return False
        saved=read_json(file)
        if (not complete_query(saved,states,noise_root,['E']) or not saved.get('excluded_from_Actor_credit')
                or saved.get('teacher_reference')!=parent.get('common_E_reference')
                or saved.get('teacher_weights')!=parent.get('common_E_weights')):return False
    return True


class LocalExamples:
    """All36 final warmstart uses only the same actual initial MT/G student pairs."""
    def __init__(self,root,runtime):
        self.root,self.runtime=Path(root),runtime
        contract=read_json(self.root/'run_contract.json')
        self.data_version=contract['student_refresh']['version']
        self.current={task:self.root/'student_data'/self.data_version/f'task_{task:04d}' for task in TASKS}
        registered=panels(self.root)
        for task,directory in self.current.items():
            if not student_data_complete(directory,registered[task],task,
                    require_E=contract['student_refresh'].get('readonly_E_query',False)):
                raise ValueError('same initial student practice/query must cover all36 before final warmstart')

    def __call__(self,task,update):
        directory=self.current[task];path=restore_compilation(directory,self.runtime)
        rewards=read_json(directory/'query/complete.json')['rewards']
        prefixes=[dict(records=node['records'],episodes=node['episodes'],practiced=node['candidates'],
            reward={ref:rewards[ref] for ref in node['candidates']}) for node in path.outputs]
        return dict(path=path,prefixes=prefixes,source=dict(kind='initial_actual_student_pair',
            file=str(directory/'path.json'),data_version=self.data_version,additional_uniform_R5=False))
