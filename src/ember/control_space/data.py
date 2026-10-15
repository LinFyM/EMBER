"""Explicit RGB authorities and rank-independent original/producer/query streams."""
from pathlib import Path
import json
import numpy as np
from ember.pi05_lora import load_pi05_lora_contract,derive_pi05_lora_rank
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.reference.contract import ASSETS
from ember.writer.learning_data import load_learning_tasks,_episode_queries
from .contract import (SCHEMA,LIMITS,TASKS,VALIDATION,REFERENCE_ROOT,PARENT_ROOT,OPERATOR_ROOT,OPTIMIZER,
    LINEAR_ROOT,BASIS,LEGACY_IMAGE_ROOT,LEGACY_IMAGE_CACHE)


def topology():
    return derive_pi05_lora_rank(load_pi05_lora_contract(ASSETS/'configs/pi05_lora_rank128_aligned.json'),rank=8)


def permutation(task,domain,cycle,count):
    return np.random.default_rng(np.random.SeedSequence([7,int(task),int(domain),int(cycle)])).permutation(count).tolist()


def original_demo(task,visit):return permutation(task,1,visit//50,50)[visit%50]


def producer_rows():
    path=REFERENCE_ROOT/'analysis/engineering_final_four_20261016/producer_videos.json'
    rows=json.loads(path.read_text());eligible=[r for r in rows if r['success'] and r['correct_auxiliary_teaching_eligible']]
    if len(rows)!=648 or len(eligible)!=545:raise ValueError('original 545 success/103 failure producer bank changed')
    return sorted(eligible,key=lambda r:(r['task_id'],r['init_state_id']))


def feature_entries(root):return read_json(Path(root)/'features/manifest.json')['entries']


def feature_manifest(root):
    entries=[];tasks=load_learning_tasks(ASSETS,TASKS,protocol_path='configs/libero_24_8_8_coverage_v1/protocol.json')
    for task in TASKS:
        for demo in range(50):
            key=f'original_t{task:04d}_d{demo:02d}'
            old=LEGACY_IMAGE_CACHE/f'task_{task:04d}_demo_{demo:02d}.pt'
            cache=old if demo in (46,47,48,49) and old.is_file() else Path(root)/'features/cache'/f'{key}.pt'
            entries.append(dict(key=key,kind='original',task=task,demo=demo,cache_path=str(cache),
                first_visit=next(v for v in range(50) if original_demo(task,v)==demo),
                frames=(tasks[task].episode_lengths[demo]-1)//5+1+int((tasks[task].episode_lengths[demo]-1)%5!=0)))
    auxiliary=producer_rows()
    bank={r['task']:r for r in read_json(PARENT_ROOT/'bank.json')['tasks']}
    for row in auxiliary:
        expected=bank[row['task_id']]['selected_path'] or 'raw_source_identity'
        if row['producer_LoRA']!=expected:raise ValueError('auxiliary code label would identify another producer')
    for task in TASKS:
        pool=[r for r in auxiliary if r['task_id']==task]
        order=permutation(task,2,0,len(pool))
        for index,row in enumerate(pool):
            key=f'producer_t{task:04d}_s{row["init_state_id"]:03d}'
            entries.append(dict(key=key,kind='producer',task=task,path=row['path'],frame_indices=row['frame_indices'],
                cache_path=str(Path(root)/'features/cache'/f'{key}.pt'),first_visit=order.index(index),
                frames=len(row['frame_indices']),producer_LoRA=row['producer_LoRA']))
    entries.sort(key=lambda r:(r['first_visit'],r['task'],r['kind']))
    if len(entries)!=2345 or sum(r['frames'] for r in entries)!=75873:raise ValueError('complete video pool/terminal frame count changed')
    return dict(entries=entries,original_videos=1800,producer_videos=545,frames=75873,feature_shape=[512,2048],dtype='bfloat16')


def make_runtime(root,device,*,include_validation=False):
    from ember.reference.training import Runtime
    from ember.writer.data import RawTeacherVideoStore
    runtime=Runtime(root,device)
    if include_validation:
        runtime.tasks.update(load_learning_tasks(ASSETS,VALIDATION,role='validation',
            protocol_path='configs/libero_24_8_8_coverage_v1/protocol.json'))
    runtime.videos=RawTeacherVideoStore([t.authority for t in runtime.tasks.values()],frame_stride=5,camera_view='dual')
    return runtime


def panels(environment):
    from ember.expert_manifold.video_schedule import condition_demo_index
    bytask={t['global_task_id']:t for t in environment['tasks']}
    seen=[dict(task=task,init_state_id=init,teacher_demo=condition_demo_index(7,bytask[task]['suite'],
        bytask[task]['task_id'],init,condition='correct',demo_count=50,sampling_mode='without_replacement'),noise_root=202610151)
        for task in TASKS for init in range(16)]
    validation=read_json(LEGACY_IMAGE_ROOT/'readout_panels/validation.json')['rows']
    if len(validation)!=400 or {r['task'] for r in validation}!=set(VALIDATION):raise ValueError('strict held panel changed')
    for task in VALIDATION:
        selected=[r for r in validation if r['task']==task]
        if {r['init_state_id'] for r in selected}!=set(range(50)) or {r['teacher_demo'] for r in selected}!=set(range(50)):
            raise ValueError('held400 must use each of all50 videos once')
        for r in selected:
            expected=condition_demo_index(7,bytask[task]['suite'],bytask[task]['task_id'],r['init_state_id'],condition='correct',demo_count=50,sampling_mode='without_replacement')
            if r['teacher_demo']!=expected or r['noise_root']!=7:raise ValueError('held registered pairing differs from official seed7')
    return dict(seen=seen,validation=validation)


def prepare(root):
    from ember.proposal_writer.contract import environment_contract
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if not read_json(LINEAR_ROOT/'completion.json')['complete']:raise ValueError('one completed linear64 basis required')
    train=read_json(PARENT_ROOT/'run_contract.json')['environment']
    held=environment_contract(ASSETS,root=root,role='validation')
    environment=dict(train,tasks=train['tasks']+held['tasks'])
    contract=dict(schema=SCHEMA,phase='video_initial',frozen_basis=str(BASIS),linear_root=str(LINEAR_ROOT),
        parent_root=str(PARENT_ROOT),operator_root=str(OPERATOR_ROOT),reference_root=str(REFERENCE_ROOT),
        limits=LIMITS,seed=7,environment=environment,panels=str(root/'panels.json'),
        tasks=list(TASKS),validation=list(VALIDATION),checkpoints=[900,1800],logical=dict(tasks=4,FM_queries_per_task=28,auxiliary_per_task=1),
        loss=dict(FM=1.,producer_code_MSE=.1),optimizer=OPTIMIZER,source_frozen=True,MT_loaded=False,
        practice_interactions=0,feature_schema='ember_source_image_prefix_features_v1')
    values={'run_contract.json':contract,'bank.json':read_json(PARENT_ROOT/'bank.json'),
        'units.json':read_json(PARENT_ROOT/'units.json'),'panels.json':panels(environment),
        'features/manifest.json':feature_manifest(root)}
    for name,value in values.items():
        path=root/name
        # JSON serializes optimizer betas as a list; compare the serialized contract.
        value=json.loads(json.dumps(value))
        if path.exists() and read_json(path)!=value:raise ValueError(f'immutable video contract changed: {name}')
        if not path.exists():write_json_atomic(path,value)
    return contract


class Stream:
    """Only cursors and orchestration identities; none enters Reader's conditions."""
    def __init__(self,root,state=None):
        self.entries=feature_entries(root);self.bykey={e['key']:e for e in self.entries}
        self.pools={t:[e for e in self.entries if e['kind']=='producer' and e['task']==t] for t in TASKS}
        # Manifest order is first-visit sorted; freeze the producer pool by init identity.
        self.pools={t:sorted(v,key=lambda e:e['key']) for t,v in self.pools.items()}
        self.visits={t:0 for t in TASKS};self.query_cursor={t:0 for t in TASKS};self.next_update=0
        self.tasks=load_learning_tasks(ASSETS,TASKS,protocol_path='configs/libero_24_8_8_coverage_v1/protocol.json')
        if state:
            self.visits={int(k):int(v) for k,v in state['visits'].items()};self.query_cursor={int(k):int(v) for k,v in state['query_cursor'].items()}
            self.next_update=int(state['next_update'])

    def state_dict(self):return dict(next_update=self.next_update,visits=self.visits,query_cursor=self.query_cursor,seed=7)

    def next(self):
        cycle,slot=divmod(self.next_update,9);order=permutation(0,0,cycle,36);result=[]
        for index in order[slot*4:slot*4+4]:
            task=TASKS[index];visit=self.visits[task];demo=original_demo(task,visit);pool=self.pools[task]
            auxiliary=pool[permutation(task,2,visit//len(pool),len(pool))[visit%len(pool)]]
            qorder=permutation(task,3,0,50)
            demos,frames,cursor=_episode_queries(task,self.tasks[task].episode_lengths,qorder,seed=7,
                cursor=self.query_cursor[task],count=28,teacher_demo=demo)
            result.append(dict(task=task,visit=visit,original=self.bykey[f'original_t{task:04d}_d{demo:02d}'],
                auxiliary=auxiliary,demos=demos,frames=frames))
            self.visits[task]+=1;self.query_cursor[task]=cursor
        self.next_update+=1;return result
