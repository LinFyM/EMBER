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
from .path import Compilation,cpu_state,evaluate_fixed,restore_compilation


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
    """No query H enters path; identical λ/state/RNG outcomes serve all its prefixes."""
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    states=query_panel(panel,visit);noise_root=seed(70,path.task,visit)
    references=list(dict.fromkeys(ref for node in path.outputs for ref in node['candidates']))
    practice_states={row['init_state_id'] for row in path.history.episodes}
    if practice_states.intersection(states):raise ValueError('query initial states leaked into practice')
    tick=time.monotonic();before=runner.environment_steps
    for row in evaluate_fixed(runtime,runner,path,task,states,noise_root,references=references,
                             existing_rows={(ref,state) for ref in references for state in states
                                 if (destination/f'{ref}_{state:03d}.json').exists()}):
        write_json_atomic(destination/f"{row['reference']}_{row['init_state_id']:03d}.json",row)
    rows=[read_json(destination/f'{ref}_{state:03d}.json') for ref in references for state in states]
    rewards={ref:sum(bool(row['success']) for row in rows if row['reference']==ref)/4 for ref in references}
    write_json_atomic(destination/'complete.json',dict(complete=True,rewards=rewards,rows=rows,
        query_states=states,policy_env_noise_root=noise_root,query_excluded_from_H=True,
        each_parameter_query_once_across_prefixes=True,unpracticed_candidates_not_queried=True,
        actual_new_environment_steps_this_attempt=runner.environment_steps-before,
        query_seconds_this_attempt=time.monotonic()-tick))
    return rewards


class LocalExamples:
    """Lazy all36 supervision: bounded decoded H, teacher warmstart then current G paths."""
    def __init__(self,root,runtime):
        self.root,self.runtime=Path(root),runtime
        self.panel=panels(root);self.cache=HistoryCache(max_entries=2)
        contract=read_json(self.root/'run_contract.json')
        self.teacher_updates=contract['local_supervision']['teacher_warm_updates']
        self.data_version=contract['local_data_version']
        self.current={task:[] for task in TASKS}
        for file in sorted((self.root/'local_data'/self.data_version).glob('*/local_labels.json')):
            row=read_json(file)
            if not row['complete']:raise ValueError('current G local examples incomplete')
            self.current[row['task']].append(file)
        if any(not self.current[t] for t in TASKS):raise ValueError('current G local data must cover all36')

    def __call__(self,task,update):
        visit=update//9
        if update<self.teacher_updates:return self.teacher(task,visit)
        file=self.current[task][visit%len(self.current[task])]
        record=read_json(file);path=restore_compilation(file.parent,self.runtime)
        return dict(path=path,prefixes=record['prefixes'],source=dict(kind='current_G',file=str(file),data_version=self.data_version))

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
