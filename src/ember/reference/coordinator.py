"""Event-driven dynamic work, bounded same-trajectory selection and one owner."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
import datetime
import json
import sqlite3
import threading
import time

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_eval_queue import EvaluationShard, initialize_queue, claim_next
from ember.proposal_writer.batch import Batch, Job
from .contract import (SCHEMA,TASKS,BASE,EXTENSIONS,LIMITS,SOURCE,NORMALIZATION,SHARD,
                       environment_contract,plan,selection_decision)


def prepare(root):
    path=root/'run_contract.json'
    if path.exists():
        run=read_json(path)
        if run['schema']!=SCHEMA or run['tasks']!=list(TASKS) or run['limits']!=LIMITS:
            raise ValueError('reference run contract changed')
        return run
    environment=environment_contract(root)
    previous=Path('/data1/user/ymdai/ember_runs/video_effect_parameter_compiler_20261015/corrected_results.json')
    mt=[r for r in read_json(previous)['rows'] if r['arm']=='MT']
    if len(mt)!=128 or sum(r['success'] for r in mt)!=67:
        raise ValueError('unique historical MT128/67 identity changed')
    write_json_atomic(root/'inputs'/'MT128_reference.json',dict(source=str(previous),rows=mt,
        comparison_only=True,new_MT_execution=False,pairing_requires_actual_rows=True))
    run=dict(schema=SCHEMA,tasks=list(TASKS),source=SOURCE,normalization=NORMALIZATION,
        environment=environment,limits=LIMITS,logical_batch=28,rank=8,targets=38,
        candidates=list(BASE),conditional_extensions=list(EXTENSIONS),selection_states=list(range(32)),
        report_states=list(range(32,50)),noise_roots=dict(selection=202610151,report=202610152),
        first_dispatch=read_json(root/'inputs/dispatch.json')['first_dispatch_UTC'],
        created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        deployment='one complete source-relative LoRA; no MT; frozen source',
        scope='eight nonheld reference trajectories only; no compiler/held/Test',
        environment_lifecycle='lazy spawn; full close after each shard and before training',
        physical_execution='two useful updates each7/14/28; useful evaluation slots4/8/16 with aggregate CPU admission')
    write_json_atomic(path,run)
    return run


class Scheduler:
    def __init__(self,root,panel,workers):
        self.root,self.panel,self.workers=root,panel,workers
        self.lock=threading.Lock();self.waiting={};self.running={};self.scores={t:{} for t in TASKS}
        self.decisions={};self.failed=False;self.eval_counts={};self.latest_resources={}
        self.running_slots={};self.environment_GiB=4.6;self.slot_cap=64;self.parent_GiB=16
        self.evaluation_measures={}

    def statuses(self):
        with closing(sqlite3.connect(self.root/'work.sqlite',timeout=60)) as db:
            return dict(db.execute('SELECT job_id,status FROM jobs'))

    def evaluate_selection(self,statuses):
        for task in TASKS:
            for update in (0,*BASE,*EXTENSIONS):
                if update in self.scores[task]:continue
                identities=[f'selection_t{task:04d}_u{update:08d}_s{s:02d}' for s in range(0,32,SHARD)]
                if not all(statuses[i]=='complete' for i in identities):continue
                self.scores[task][update]=sum(sum(r['success'] for r in read_json(self.root/'shards'/f'{i}.json')['output']['rows']) for i in identities)
            decision=selection_decision(self.scores[task])
            if decision:
                self.decisions[task]=decision
                if decision['settled']:
                    path=self.root/'selection'/f'task_{task:04d}'/'selected.json'
                    if path.exists():
                        if read_json(path)['selected_update']!=decision['selected_update']:
                            raise ValueError('independent report cannot change frozen selection')
                    else:write_json_atomic(path,dict(decision,frozen_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))

    def eligible(self,statuses):
        ready=[]
        for identity,work in self.panel.items():
            if statuses[identity]!='pending':continue
            task=work['task'];kind=work['kind'];decision=self.decisions.get(task,{})
            if kind=='train':
                stop=work['stop'];start=work['start']
                parent=(start==0 or statuses[f'train_t{task:04d}_u{start:08d}']=='complete')
                allowed=(stop in BASE or (not decision.get('settled') and decision.get('next_update')==stop))
                if parent and allowed:ready.append(identity)
            elif kind=='selection':
                u=work['update']
                if u==0 or statuses[f'train_t{task:04d}_u{u:08d}']=='complete':ready.append(identity)
            elif decision.get('settled'):
                if work['arm']=='selected' and decision['selected_update']==0:
                    source=identity.replace('_selected_','_source_')
                    if statuses[source]!='complete':continue
                ready.append(identity)
        return ready

    def terminal(self,statuses):
        if len(self.decisions)!=len(TASKS):return False
        for task,decision in self.decisions.items():
            if not decision.get('settled') and not decision.get('window_exhausted'):return False
            if decision.get('settled'):
                if any(statuses[i]!='complete' for i,w in self.panel.items() if w['task']==task and w['kind']=='report'):return False
        return True

    def choose_work(self,ready):
        training=[i for i in ready if self.panel[i]['kind']=='train']
        evaluating=[i for i in ready if self.panel[i]['kind']!='train']
        active_training=sum(self.panel[i]['kind']=='train' for i in self.running.values())
        if training and (active_training<max(1,self.workers//2) or not evaluating):return training
        return evaluating if sum(self.running_slots.values())<self.slot_cap else []

    def physical_slots(self,worker,work):
        if work['kind']=='train':return 0
        resources=self.latest_resources.get(worker,{})
        available=resources.get('host_memory_KiB',{}).get('MemAvailable',0)/1024**2
        count=self.eval_counts.get(worker,0);desired=(4,8,16)[min(count,2)]
        measured=self.evaluation_measures.get(worker,[])
        if count>=3 and measured:desired=max(measured,key=lambda m:m['native_plans_per_second'])['slots']
        allowed=max(4,self.slot_cap-sum(self.running_slots.values()))
        slots=max(s for s in (4,8,16) if s<=desired and s<=allowed)
        if available<64+self.environment_GiB*slots:slots=4
        self.eval_counts[worker]=count+1
        return slots

    def dispatch(self):
        statuses=self.statuses();self.failed |= any(s=='failed' for s in statuses.values())
        self.evaluate_selection(statuses)
        for worker,stream in list(self.waiting.items()):
            ready=[] if self.failed else self.eligible(statuses)
            if ready:
                # Keep ready learning and evaluation concurrent; each evaluation
                # claim still uses the existing long-first cost queue.
                eligible=self.choose_work(ready)
                if not eligible:continue
                claim=claim_next(self.root/'work.sqlite',worker_id=worker,eligible_job_ids=eligible)
                if claim is None:continue
                identity=claim.shard.job_id;work=self.panel[identity]
                slots=self.physical_slots(worker,work)
                message=dict(command='work',identity=identity,work=work,claim_token=claim.claim_token,slots=slots)
                self.running[worker]=identity;self.running_slots[worker]=slots;statuses[identity]='claimed'
            elif self.failed or self.terminal(statuses):
                message=dict(command='stop')
            else:continue
            stream.write(json.dumps(message)+'\n');stream.flush();del self.waiting[worker]
            with (self.root/'owner_events.jsonl').open('a') as f:
                f.write(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),worker=worker,**message))+'\n')

    def event(self,worker,event,stream):
        with self.lock:
            if event['kind']=='failed':self.failed=True
            self.latest_resources[worker]=event.get('resources',self.latest_resources.get(worker,{}))
            resource=self.latest_resources.get(worker,{})
            parent=next((p for p in resource.get('per_process',[]) if p['PID']==resource.get('PID')),None)
            if parent:self.parent_GiB=max(self.parent_GiB,parent['PSS_KiB']/1024**2*1.2)
            with (self.root/'worker_resources.jsonl').open('a') as f:f.write(json.dumps(dict(worker=worker,**event))+'\n')
            if event['kind']=='ready':
                previous=self.running.pop(worker,None);slots=self.running_slots.pop(worker,None)
                if previous and self.panel[previous]['kind']!='train':
                    receipt=read_json(self.root/'shards'/f'{previous}.json')
                    resources=receipt['output']['resources'];peak=resources.get('peak_environment_resources')
                    if peak:
                        pids=peak['environment_PIDs']
                        sizes=[p['PSS_KiB']/1024**2 for p in peak['per_process'] if p['PID'] in pids]
                        if sizes:
                            self.environment_GiB=max(sizes)*1.2
                            total=peak['host_memory_KiB']['MemTotal']/1024**2
                            self.slot_cap=max(4,min(self.workers*16,int((total-self.workers*self.parent_GiB-64)/self.environment_GiB)//4*4))
                    if resources.get('new_episodes',0)>=8:
                        measurement=dict(slots=slots,episodes_per_second=resources['episodes_per_second'],
                            native_plans_per_second=resources['runner']['plans']/max(1e-6,resources['runner']['native_prefix_seconds']+resources['runner']['native_flow_seconds']),
                            peak_allocated_MiB=receipt['peak_allocated_MiB'],environment_GiB=self.environment_GiB)
                        self.evaluation_measures.setdefault(worker,[]).append(measurement)
                self.waiting[worker]=stream
            self.dispatch()


def owner(args):
    root=args.root;run=prepare(root);panel=plan(root,run['environment'])
    devices=[(value.split(':')[0],int(value.split(':')[1])) for value in args.devices]
    shards=[EvaluationShard(identity,index,w['kind'],w['task'],
        next(t['horizon'] for t in run['environment']['tasks'] if t['global_task_id']==w['task']),
        tuple(w['states']),w['cost']) for index,(identity,w) in enumerate(panel.items())]
    initialize_queue(root/'work.sqlite',shards,contract_reference=str(root/'run_contract.json'),recover_claims=args.resume,retry_failed=args.resume)
    scheduler=Scheduler(root,panel,len(devices))
    batch=Batch(Path.cwd(),devices,root=root,required_free_MiB=34000,work_event=scheduler.event)
    jobs=[Job(f'reference_worker_{i:02d}', ['source-reference','worker','--worker-id',f'reference_worker_{i:02d}','--slots','8'],
              eligible_devices=(device,),estimate_seconds=6*3600,required_free_MiB=34000) for i,device in enumerate(devices)]
    started=time.monotonic();failure=None
    # Each future is the actual persistent GPU child. A startup exception must
    # also wake idle peers so no worker waits forever for a dead producer.
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures=[executor.submit(batch.execute,job,device) for job,device in zip(jobs,devices)]
        for future in as_completed(futures):
            try:future.result()
            except BaseException as error:
                failure=repr(error) if failure is None else failure+'; '+repr(error)
                with scheduler.lock:
                    scheduler.failed=True;scheduler.dispatch()
    from .analysis import analyze
    result=analyze(root)
    result.update(owner_elapsed_seconds=time.monotonic()-started,owner_failure=failure,
                  end_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json_atomic(root/'completion.json',result)
    if failure:raise RuntimeError(failure)
    if not scheduler.terminal(scheduler.statuses()):raise RuntimeError('workers exited without a completed bounded window')
