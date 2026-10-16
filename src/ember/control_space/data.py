"""Explicit condition metadata and continuation of the selected900 logical stream."""
from pathlib import Path
import json
import numpy as np
import torch
from ember.pi05_lora import load_pi05_lora_contract,derive_pi05_lora_rank
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.reference.contract import ASSETS,environment_contract
from ember.writer.learning_data import load_learning_tasks,_episode_queries
from .contract import (SCHEMA,LIMITS,TASKS,REFERENCE_ROOT,PARENT_ROOT,OPTIMIZER,BASIS,
    INITIAL_READER,VIDEO_ROOT,ROOT,PRACTICE_ROOT,QUERY_ROOT)


def topology():return derive_pi05_lora_rank(load_pi05_lora_contract(ASSETS/'configs/pi05_lora_rank128_aligned.json'),rank=8)
def permutation(task,domain,cycle,count):
    return np.random.default_rng(np.random.SeedSequence([7,int(task),int(domain),int(cycle)])).permutation(count).tolist()

def make_runtime(root,device):
    from ember.reference.training import Runtime
    from ember.writer.data import RawTeacherVideoStore
    runtime=Runtime(root,device)
    runtime.videos=RawTeacherVideoStore([t.authority for t in runtime.tasks.values()],frame_stride=5,camera_view='dual')
    return runtime

def feature_entries(root):return read_json(Path(root)/'features/manifest.json')['entries']

def prepare(root):
    from ember.expert_manifold.video_schedule import condition_demo_index
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    environment=environment_contract(root)
    old=torch.load(INITIAL_READER,map_location='cpu',mmap=True,weights_only=False)
    if old['phase']!='video_initial' or old['next_update']!=900 or set(old['stream']['visits'].values())!={100}:
        raise ValueError('only the strict400 selected900 Reader is a legal initializer')
    bytask={t['global_task_id']:t for t in environment['tasks']}
    oldentries=read_json(VIDEO_ROOT/'features/manifest.json')['entries'];features=[];conditions={}
    for task in TASKS:
        spec=bytask[task]
        demo=condition_demo_index(7,spec['suite'],spec['task_id'],0,condition='correct',demo_count=50,sampling_mode='without_replacement')
        feature=next(e for e in oldentries if e['kind']=='original' and e['task']==task and e['demo']==demo)
        if not Path(feature['cache_path']).is_file():raise FileNotFoundError('required existing original prefix is absent')
        features.append(feature)
        practice=np.random.default_rng(np.random.SeedSequence([7,PRACTICE_ROOT,task,0])).choice(np.arange(16,50),4,replace=False).tolist()
        conditions[task]=dict(task=task,v=0,teacher_demo=demo,original=feature,practice_states=practice,
            practice_noise_root=PRACTICE_ROOT,query_states=[0,4,8,12],query_noise_root=QUERY_ROOT)
    result=read_json(VIDEO_ROOT/'analysis/results.json')['checkpoints']['900']['seen']['tasks']
    baselines={t['task']:len(t['success_sets']['generated'])/16 for t in result}
    contract=dict(schema=SCHEMA,phase='feedback_proposals',limits=LIMITS,seed=7,environment=environment,
        frozen_basis=str(BASIS),initial_reader_checkpoint=str(INITIAL_READER),reference_root=str(REFERENCE_ROOT),
        video_root=str(VIDEO_ROOT),bank_root=str(PARENT_ROOT),tasks=list(TASKS),conditions=36,meta_updates=9,
        logical=dict(tasks=4,FM_queries_per_task=28,query_initial_states=4),optimizer=OPTIMIZER,
        loss=dict(FM=1.,PG='all_actual_proposals_joint64_sum',beta='first_nonzero_global_norm_once'),
        initial_baselines=baselines,initial_stream=old['stream'],source_frozen=True,MT_loaded=False,
        sigma=.25,practice_steps_per_condition=8192,held=False,Test=False)
    for name,value in [('run_contract.json',contract),('conditions.json',conditions),('features/manifest.json',dict(entries=features))]:
        value=json.loads(json.dumps(value));path=root/name
        if path.exists() and read_json(path)!=value:raise ValueError('immutable current phase inputs changed: '+name)
        if not path.exists():write_json_atomic(path,value)
    return contract


class Stream:
    def __init__(self,root,state=None):
        self.root=Path(root);self.conditions={int(t):v for t,v in read_json(self.root/'conditions.json').items()}
        initial=read_json(self.root/'run_contract.json')['initial_stream'];self.origin_update=initial['next_update']
        self.visits={int(k):int(v) for k,v in initial['visits'].items()}
        self.query_cursor={int(k):int(v) for k,v in initial['query_cursor'].items()};self.next_update=0
        self.tasks=load_learning_tasks(ASSETS,TASKS,protocol_path='configs/libero_24_8_8_coverage_v1/protocol.json')
        if state:
            self.visits={int(k):int(v) for k,v in state['visits'].items()};self.query_cursor={int(k):int(v) for k,v in state['query_cursor'].items()}
            self.next_update=state['next_update']
    def state_dict(self):return dict(next_update=self.next_update,origin_update=self.origin_update,visits=self.visits,query_cursor=self.query_cursor,seed=7)
    def next(self):
        if self.next_update>=9:raise ValueError('only the first complete36 task cycle is authorized')
        cycle,slot=divmod(self.origin_update+self.next_update,9);order=permutation(0,0,cycle,36);events=[]
        for index in order[slot*4:slot*4+4]:
            task=TASKS[index];condition=self.conditions[task];demo=condition['teacher_demo'];visit=self.visits[task]
            demos,frames,cursor=_episode_queries(task,self.tasks[task].episode_lengths,permutation(task,3,0,50),
                seed=7,cursor=self.query_cursor[task],count=28,teacher_demo=demo)
            events.append(dict(condition,visit=visit,demos=demos,frames=frames,meta_update=self.next_update))
            self.visits[task]+=1;self.query_cursor[task]=cursor
        self.next_update+=1;return events
