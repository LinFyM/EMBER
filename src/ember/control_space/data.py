"""Legal past-panel prefixes and the continued cross-episode functional stream."""
from pathlib import Path

import numpy as np

from ember.pi05_lora import load_pi05_lora_contract,derive_pi05_lora_rank
from ember.pi05_source_checkpoint import read_json
from ember.reference.contract import ASSETS,TASKS
from ember.writer.learning_data import load_learning_tasks,_episode_queries

BASE=Path('/data1/user/ymdai/ember_runs/source_policy_space_compiler_20261015')
HISTORY_ROOT=BASE/'incumbent_compilation_seed7'
VIDEO_ROOT=BASE/'video_initial_seed7'
INITIAL_READER=VIDEO_ROOT/'checkpoints/update_00000900/state.pt'
ARMS=('video','no_video')


def topology():
    return derive_pi05_lora_rank(load_pi05_lora_contract(ASSETS/'configs/pi05_lora_rank128_aligned.json'),rank=8)


def permutation(task,domain,cycle,count):
    return np.random.default_rng(np.random.SeedSequence([7,int(task),int(domain),int(cycle)])).permutation(count).tolist()


def original_demo(task,visit):return permutation(task,1,visit//50,50)[visit%50]


def feature_entries():
    return [e for e in read_json(VIDEO_ROOT/'features/manifest.json')['entries'] if e['kind']=='original']


def _prefixes(task,arm,condition,spec):
    """Consume panels only; the old final incumbent and query have no data edge."""
    from .comparison import _validate_episode
    file=HISTORY_ROOT/'conditions'/condition['id']/'path.json'
    panels=read_json(file)['panels'];result=[];histories=[];spent=0;incumbent=None
    for ordinal,panel in enumerate(panels):
        rows=panel['rows'];y=panel['y']
        if panel['ordinal']!=ordinal or len(rows)!=4 or [r['init_state_id'] for r in rows]!=condition['practice_states']:
            raise ValueError('offline history requires every actual complete panel in original state order')
        if len(y)!=64 or not all(np.isfinite(y)):raise ValueError('actual behavior code must be finite64')
        for row in rows:
            _validate_episode(row,spec,condition['practice_states'],202610161)
            if row['behavior_y']!=y or not Path(row['history_path']).is_file():
                raise ValueError('behavior code/raw History no longer identify the actual panel')
        successes=sum(r['success'] for r in rows)
        if panel['successes']!=successes:raise ValueError('past observed success count changed')
        improve=incumbent is None or successes>incumbent['successes']
        if bool(panel['accepted'])!=improve:raise ValueError('prefix incumbent must follow strict improvement and keep ties')
        if improve:incumbent=dict(y=y,successes=successes,ordinal=ordinal)
        histories.extend(r['history_path'] for r in rows);spent+=sum(r['environment_steps'] for r in rows)
        if spent>8192:raise ValueError('old actual panel prefix exceeded its condition budget')
        result.append(dict(task=task,arm=arm,path=str(file),condition_id=condition['id'],panel_ordinal=ordinal,
            panel_identity=panel['identity'],m=list(incumbent['y']),best_score=incumbent['successes']/4,
            successes=incumbent['successes'],incumbent_ordinal=incumbent['ordinal'],remaining_steps=8192-spent,
            observed_steps=spent,history_paths=list(histories),logical_practice_episodes=len(histories),
            terminal=incumbent['successes']==4 or 8192-spent<4*(spec['horizon']+10)))
    if not panels or panels[0]['kind']!='initial' or panels[0]['y']!=condition['initial_y']:
        raise ValueError('prefix stream requires the actually practiced registered initial code')
    return result


def prefix_manifest():
    """Metadata only: no History tensors, future panels in a prefix, or readout labels."""
    run=read_json(HISTORY_ROOT/'run_contract.json')
    if run['schema']!='ember_source_incumbent_compilation_r8_seed7_v1':raise ValueError('wrong actual history phase')
    conditions={(c['task'],c['arm']):c for c in run['conditions']}
    specs={t['global_task_id']:t for t in run['environment']['tasks']};prefixes={};counts={a:0 for a in ARMS}
    for task in TASKS:
        prefixes[task]={}
        for arm in ARMS:
            pool=_prefixes(task,arm,conditions[task,arm],specs[task]);prefixes[task][arm]=pool;counts[arm]+=len(pool)
    if counts!=dict(video=156,no_video=158):raise ValueError('only the314 actually observed complete panels are authorized')
    return dict(tasks=prefixes,panels=counts,logical_practice_episodes=4*sum(counts.values()),
        source=str(HISTORY_ROOT),independent_tasks=36,uses_final_query=False,uses_final_incumbent=False)


def _initial_stream():
    import torch
    saved=torch.load(INITIAL_READER,map_location='cpu',weights_only=False,mmap=True)
    if saved['phase']!='video_initial' or saved['next_update']!=900:raise ValueError('functional stream requires actual Reader900 origin')
    stream=saved['stream']
    if stream['next_update']!=900 or stream['seed']!=7 or any(stream['visits'][t]!=100 for t in TASKS):
        raise ValueError('Reader900 task/video cursor origin changed')
    return stream


class Stream:
    """Task IDs/paths are orchestration metadata, never neural conditioning tokens."""
    def __init__(self,root,state=None):
        self.root=Path(root);self.prefixes=prefix_manifest()['tasks']
        self.bykey={e['key']:e for e in feature_entries()}
        if len(self.bykey)!=1800:raise ValueError('reuse the complete legal original feature pool')
        origin=_initial_stream();self.origin_query_cursor=dict(origin['query_cursor'])
        self.visits={int(k):int(v) for k,v in origin['visits'].items()}
        self.query_cursor={int(k):int(v) for k,v in origin['query_cursor'].items()}
        self.new_visits={t:0 for t in TASKS};self.next_update=0
        self.tasks=load_learning_tasks(ASSETS,TASKS,protocol_path='configs/libero_24_8_8_coverage_v1/protocol.json')
        if state:self._restore(state)

    def _restore(self,state):
        if state['seed']!=7 or state.get('logical_origin')!=900:raise ValueError('functional stream origin/seed changed')
        self.next_update=int(state['next_update'])
        for name in ('visits','query_cursor','new_visits'):
            values={int(k):int(v) for k,v in state[name].items()}
            if set(values)!=set(TASKS):raise ValueError('complete36-task sampler cursors are required')
            setattr(self,name,values)
        if not 0<=self.next_update<=180 or sum(self.new_visits.values())!=4*self.next_update:
            raise ValueError('fresh functional update/visit count changed')
        if any(self.visits[t]!=100+self.new_visits[t] or self.query_cursor[t]<self.origin_query_cursor[t] for t in TASKS):
            raise ValueError('resume must continue actual Reader900 labels without resetting their cursors')

    def state_dict(self):
        return dict(next_update=self.next_update,visits=dict(self.visits),query_cursor=dict(self.query_cursor),
            new_visits=dict(self.new_visits),seed=7,logical_origin=900)

    def _event(self,task):
        visit=self.visits[task];k=self.new_visits[task];arm_index=k%2;arm=ARMS[arm_index]
        pool=self.prefixes[task][arm];q=k//2;ordinal=permutation(task,17+arm_index,q//len(pool),len(pool))[q%len(pool)]
        prefix=pool[ordinal];demo=original_demo(task,visit);cursor=self.query_cursor[task]
        demos,frames,cursor=_episode_queries(task,self.tasks[task].episode_lengths,permutation(task,3,0,50),
            seed=7,cursor=cursor,count=28,teacher_demo=demo)
        event=dict(task=task,visit=visit,new_visit=k,original=self.bykey[f'original_t{task:04d}_d{demo:02d}'],
            demos=demos,frames=frames,m=prefix['m'],best_y=prefix['m'],best_score=prefix['best_score'],
            remaining_steps=prefix['remaining_steps'],history_paths=prefix['history_paths'],prefix=dict(prefix),
            query_cursor_before=self.query_cursor[task],query_cursor_after=cursor,
            off_policy_history=True,history_produced_by_current_teacher_or_psi=False)
        self.visits[task]+=1;self.new_visits[task]+=1;self.query_cursor[task]=cursor
        return event

    def next(self):
        if self.next_update>=180:raise StopIteration('the unique180-update functional window is complete')
        cycle,slot=divmod(900+self.next_update,9);order=permutation(0,0,cycle,36)
        events=[self._event(TASKS[index]) for index in order[slot*4:slot*4+4]]
        self.next_update+=1;return events
