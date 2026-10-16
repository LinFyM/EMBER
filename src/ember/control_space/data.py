"""Immutable36-reference authority, original-video stream and final976 panel."""
from pathlib import Path
import numpy as np
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.writer.learning_data import load_learning_tasks,_episode_queries
from ember.expert_manifold.video_schedule import condition_demo_index
from ember.reference.contract import environment_contract
from .contract import ASSETS,TASKS,VALIDATION_TASKS,VIDEO_ROOT,BANK_ROOT,SCHEMA,PHASE,LIMITS,SOURCE,NORMALIZATION,INITIAL_READER


def teacher_demo(task,visit):
    cycle,slot=divmod(visit,50)
    return int(np.random.default_rng(np.random.SeedSequence([7,202610173,task,cycle])).permutation(50)[slot])


def registered_panels():
    panels=read_json(VIDEO_ROOT/'panels.json')
    if set(panels)!= {'seen','validation'}:raise ValueError('only registered seen576/strict400')
    for role,tasks,count,noise in (('seen',TASKS,16,202610151),('validation',VALIDATION_TASKS,50,7)):
        rows=panels[role]
        if len(rows)!=len(tasks)*count or {(r['task'],r['init_state_id']) for r in rows}!={(t,i) for t in tasks for i in range(count)}:
            raise ValueError('whole fixed task/state panel changed')
        for row in rows:
            task=row['task'];suite=('libero_spatial','libero_object','libero_goal','libero_10')[task//10] if task<40 else 'libero_90'
            local=task%10 if task<40 else task-40
            demo=condition_demo_index(7,suite,local,row['init_state_id'],condition='correct',demo_count=50,sampling_mode='without_replacement')
            if row['noise_root']!=noise or row['teacher_demo']!=demo:raise ValueError('official state-video/noise mapping changed')
        if count==50 and any(len({r['teacher_demo'] for r in rows if r['task']==t})!=50 for t in tasks):raise ValueError('held teaching must use all50 videos once')
    return panels


def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    bank=read_json(BANK_ROOT/'bank.json')
    if len(bank['tasks'])!=36 or {t['task'] for t in bank['tasks']}!=set(TASKS):raise ValueError('audited36 reference bank changed')
    refs=[dict(task=t['task'],selected_update=t['selected_update'],path=t['selected_path']) for t in bank['tasks']]
    if sum(r['path'] is None for r in refs)!=5:raise ValueError('five selected source identities keep equal mass')
    old=read_json(VIDEO_ROOT/'run_contract.json')
    from ember.proposal_writer.contract import environment_contract as environment
    environments=dict(seen=old['environment'],validation=environment(ASSETS,root=root,role='validation'))
    contract=dict(schema=SCHEMA,phase=PHASE,limits=LIMITS,tasks=list(TASKS),references=refs,
        environments=environments,panels=str(root/'panels.json'),source_checkpoint=str(ASSETS/SOURCE),normalization=str(ASSETS/NORMALIZATION),
        frozen_reader=str(INITIAL_READER),rank=8,alpha=8,dropout=0.,targets=38,seed=7,
        stages=dict(BA=dict(updates=1000,conditions=36000,batch=36),joint=dict(updates=180,conditions=720,queries=20160,batch=112,BA_weight=.1)),
        final_update=1180,eval_rows=976,interaction_adaptation=False,Test=False)
    for name,value in (('run_contract.json',contract),('panels.json',registered_panels())):
        path=root/name
        if path.exists() and read_json(path)!=value:raise ValueError('immutable source-weight contract changed')
        if not path.exists():write_json_atomic(path,value)
    return contract


class Stream:
    """Logical task/video/query identities are independent of every physical rank."""
    def __init__(self,state=None):
        self.next_update=0;self.visits={t:0 for t in TASKS};self.query_cursor={t:0 for t in TASKS}
        self.query_seed={t:int(np.random.SeedSequence([7,202610175,t]).generate_state(1)[0]) for t in TASKS}
        self.query_order={t:np.random.default_rng(self.query_seed[t]).permutation(50).tolist() for t in TASKS}
        self.tasks=None
        if state:
            if state['schema']!=SCHEMA or state['seed']!=7:raise ValueError('new stream cannot load historical Reader/psi state')
            self.next_update=int(state['next_update'])
            for field in ('visits','query_cursor'):
                values={int(k):int(v) for k,v in state[field].items()}
                if set(values)!=set(TASKS):raise ValueError('complete36 stream required')
                setattr(self,field,values)
            if ({int(k):v for k,v in state['query_seed'].items()}!=self.query_seed or {int(k):v for k,v in state['query_order'].items()}!=self.query_order):
                raise ValueError('fresh task-derived query RNG changed')
            expected=36*min(1000,self.next_update)+4*max(0,self.next_update-1000)
            if not 0<=self.next_update<=1180 or sum(self.visits.values())!=expected:raise ValueError('logical update/visit boundary changed')

    def state_dict(self):
        return dict(schema=SCHEMA,seed=7,next_update=self.next_update,visits=dict(self.visits),query_cursor=dict(self.query_cursor),query_seed=self.query_seed,query_order=self.query_order)

    def next(self):
        update=self.next_update
        if update>=1180:raise StopIteration('one registered1180 trajectory ended')
        selected=list(TASKS)
        if update>=1000:
            if self.tasks is None:self.tasks=load_learning_tasks(ASSETS,TASKS,protocol_path='configs/libero_24_8_8_coverage_v1/protocol.json')
            cycle,slot=divmod(update-1000,9)
            selected=np.random.default_rng(np.random.SeedSequence([7,202610174,cycle])).permutation(TASKS)[slot*4:slot*4+4].tolist()
        events=[]
        for task in selected:
            visit=self.visits[task];demo=teacher_demo(task,visit);event=dict(task=task,visit=visit,teacher_demo=demo)
            if update>=1000:
                cursor=self.query_cursor[task]
                demos,frames,next_cursor=_episode_queries(task,self.tasks[task].episode_lengths,self.query_order[task],
                    seed=self.query_seed[task],cursor=cursor,count=28,teacher_demo=demo)
                event.update(demos=demos,frames=frames,query_seed=self.query_seed[task],query_cursor_before=cursor,query_cursor_after=next_cursor)
                self.query_cursor[task]=next_cursor
            events.append(event);self.visits[task]+=1
        self.next_update+=1;return events
