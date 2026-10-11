"""Bounded teacher-prefix replay and once-per-parameter independent query labels."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch.distributed as dist
from safetensors.torch import load_file

from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.writer.practice import History

from .contract import TASKS,seed
from .learning import HistoryCache
from .path import Compilation,cpu_state,fixed_requests,fixed_row,restore_compilation


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
    """A real MT selection episode, with actual producer and logical cost retained."""
    entry=read_json(Path(root)/'teachers'/f'task_{task:04d}'/'E.json')
    parent=next(p for p in entry['parents'] if p['update']==0)
    legal=[r for r in parent['rows'] if r['init_state_id'] in panel['practice_states']]
    row=legal[ordinal%len(legal)]
    return cache.get(row['history']),row['history']


def query_panel(panel,visit):
    pool=panel['query_states']
    return [pool[(4*visit+i)%len(pool)] for i in range(4)]


def query_once(runtime,runner,path,task,panel,visit,destination):
    return query_conditions(runtime,runner,[(path,task,panel,visit,destination)])[0]


def query_conditions(runtime,runner,conditions):
    """Batch only missing independent λ/state/RNG queries, never their histories."""
    tick=time.monotonic();entries=[];requests=[];owners={}
    for path,task,panel,visit,destination in conditions:
        destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
        states=query_panel(panel,visit);noise_root=seed(70,path.task,visit)
        references=list(dict.fromkeys(ref for node in path.outputs for ref in node['candidates']))
        if {r['init_state_id'] for r in path.history.episodes}.intersection(states):raise ValueError('query initial states leaked into practice')
        entry=dict(path=path,destination=destination,states=states,noise_root=noise_root,references=references,new_steps=0,new_episodes=0)
        entries.append(entry)
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
        rows=[read_json(destination/f'{ref}_{state:03d}.json') for ref in references for state in states]
        rewards={ref:sum(bool(r['success']) for r in rows if r['reference']==ref)/4 for ref in references}
        write_json_atomic(destination/'complete.json',dict(complete=True,rewards=rewards,rows=rows,query_states=states,
            policy_env_noise_root=entry['noise_root'],query_excluded_from_H=True,each_parameter_query_once_across_prefixes=True,
            unpracticed_candidates_not_queried=True,actual_new_environment_steps_this_attempt=entry['new_steps'],
            actual_new_episodes_this_attempt=entry['new_episodes'],shared_query_group_seconds=time.monotonic()-tick,
            shared_group_members=[p.identity for p,*_ in conditions]))
        results.append(rewards)
    return results


class LocalExamples:
    """Lazy all36 supervision: bounded decoded H, teacher warmstart then current G paths."""
    def __init__(self,root,runtime):
        self.root,self.runtime=Path(root),runtime
        self.panel=panels(root);self.cache=HistoryCache(max_entries=2)
        contract=read_json(self.root/'run_contract.json')
        self.teacher_updates=contract['local_supervision']['teacher_warm_updates']
        self.data_version=contract['student_refresh']['version']
        self.current={task:self.root/'student_data'/self.data_version/f'task_{task:04d}' for task in TASKS}
        for directory in self.current.values():
            if not read_json(directory/'parent.json')['complete'] or not read_json(directory/'query/complete.json')['complete']:
                raise ValueError('same initial student practice/query must cover all36 before final warmstart')

    def __call__(self,task,update):
        visit=update//9
        if update<self.teacher_updates:return self.teacher(task,visit)
        directory=self.current[task];path=restore_compilation(directory,self.runtime)
        rewards=read_json(directory/'query/complete.json')['rewards']
        prefixes=[dict(records=node['records'],episodes=node['episodes'],practiced=node['candidates'],
            reward={ref:rewards[ref] for ref in node['candidates']}) for node in path.outputs]
        return dict(path=path,prefixes=prefixes,source=dict(kind='initial_actual_student_pair',
            file=str(directory/'path.json'),data_version=self.data_version,additional_uniform_R5=False))

    def teacher(self,task,visit):
        panel=self.panel[task];entry=read_json(self.root/'teachers'/f'task_{task:04d}'/'E.json')
        fold=visit%2;input_states=panel['selection_folds'][fold];query_states=set(panel['selection_folds'][1-fold])
        mt=next(p for p in entry['parents'] if p['update']==0)
        # Order independent of outcomes; one true episode per producer, never an offline demonstration.
        rest=[p for p in entry['parents'] if p['update']!=0]
        order=[mt,*np.random.default_rng(seed(61,task,visit)).permutation(rest).tolist()]
        state={'MT300':cpu_state(self.runtime.mt)};history=History(states=state.copy())
        path=Compilation(task,panel['policy_videos'][visit%4],f'teacher_replay_{task}_{visit}',visit,
            seed(62,task,visit),history,state,sampling_identity='teacher_replay')
        practiced=[];prefixes=[];sources=[]
        for position,parent in enumerate(order):
            rows=[r for r in parent['rows'] if r['init_state_id'] in input_states]
            row=rows[(visit//2+position)%len(rows)];actual=self.cache.get(row['history'])
            if not actual.records or len(actual.episodes)!=1:raise ValueError('teacher prefix needs one actual complete episode')
            ref=parent['reference'];path.states[ref]=load_file(parent['weights']);history.append(actual)
            practiced.append(ref);sources.append(row['history'])
            reward={p['reference']:sum(bool(r['success']) for r in p['rows'] if r['init_state_id'] in query_states)/len(query_states)
                    for p in order[:position+1]}
            prefixes.append(dict(records=len(history.records),episodes=len(history.episodes),
                practiced=practiced.copy(),reward=reward))
        path.practiced=practiced;path.valid=practiced.copy();path.selected=practiced[-1]
        self.cache.trim()
        return dict(path=path,prefixes=prefixes,source=dict(kind='teacher_replay',input_fold=fold,
            query_fold=1-fold,actual_history_files=sources,supervised_replay_not_on_policy=True))
