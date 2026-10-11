"""Actual student prefixes and once-per-parameter independent query labels."""
from __future__ import annotations

from pathlib import Path
import time

import torch.distributed as dist

from ember.pi05_source_checkpoint import read_json,write_json_atomic
from .contract import TASKS,seed
from .path import fixed_requests,fixed_row,restore_compilation


def register_students(root,paths,update,context):
    """Index true visited P/H prefixes without copying their histories or querying labels."""
    root=Path(root);local=[]
    for path in paths:
        directory=root/'RL_paths'/path.identity
        if not (directory/'path.json').exists():raise ValueError('student supervision needs retained actual source path')
        for node in path.boundaries:
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
    return [pool[(4*visit+i)%len(pool)] for i in range(4)]


def query_once(runtime,runner,path,task,panel,visit,destination):
    return query_conditions(runtime,runner,[(path,task,panel,visit,destination)])[0]


def reuse_initial_MT(entry,task,root,version):
    """A single registered second consumer references compatible initial MT rows."""
    path=entry['path'];source=Path(root)/'student_data'/version/f'task_{path.task:04d}'/'query'
    for state in entry['states']:
        target=entry['destination']/f'MT300_{state:03d}.json';original=source/target.name
        if target.exists() or not original.exists():continue
        row=read_json(original)
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


def query_conditions(runtime,runner,conditions,*,mt_reference_version=None):
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
        if visit==0 and mt_reference_version is not None and 'MT300' in references:
            reuse_initial_MT(entry,task,runtime.root,mt_reference_version)
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
        destination=entry['destination'];references=entry['references'];states=entry['states']
        rows=[]
        for ref in references:
            for state in states:
                file=destination/f'{ref}_{state:03d}.json';row=read_json(file)
                if 'source_row' in row:
                    entry['reused_rows'].append(dict(destination=str(file),**row));row=read_json(Path(row['source_row']))
                rows.append(row)
        rewards={ref:sum(bool(r['success']) for r in rows if r['reference']==ref)/4 for ref in references}
        write_json_atomic(destination/'complete.json',dict(complete=True,rewards=rewards,rows=rows,query_states=states,
            policy_env_noise_root=entry['noise_root'],query_excluded_from_H=True,each_parameter_query_once_across_prefixes=True,
            unpracticed_candidates_not_queried=True,actual_new_environment_steps_this_attempt=entry['new_steps'],
            actual_new_episodes_this_attempt=entry['new_episodes'],shared_query_group_seconds=time.monotonic()-tick,
            row_reuse=entry['reused_rows'],incompatible_MT_references=entry['incompatible_references'],
            shared_group_members=[p.identity for p,*_ in conditions]))
        results.append(rewards)
    return results


class LocalExamples:
    """All36 final warmstart uses only the same actual initial MT/G student pairs."""
    def __init__(self,root,runtime):
        self.root,self.runtime=Path(root),runtime
        contract=read_json(self.root/'run_contract.json')
        self.data_version=contract['student_refresh']['version']
        self.current={task:self.root/'student_data'/self.data_version/f'task_{task:04d}' for task in TASKS}
        for directory in self.current.values():
            if not read_json(directory/'parent.json')['complete'] or not read_json(directory/'query/complete.json')['complete']:
                raise ValueError('same initial student practice/query must cover all36 before final warmstart')

    def __call__(self,task,update):
        directory=self.current[task];path=restore_compilation(directory,self.runtime)
        rewards=read_json(directory/'query/complete.json')['rewards']
        prefixes=[dict(records=node['records'],episodes=node['episodes'],practiced=node['candidates'],
            reward={ref:rewards[ref] for ref in node['candidates']}) for node in path.outputs]
        return dict(path=path,prefixes=prefixes,source=dict(kind='initial_actual_student_pair',
            file=str(directory/'path.json'),data_version=self.data_version,additional_uniform_R5=False))
