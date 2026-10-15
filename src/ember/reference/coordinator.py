"""Event-driven dynamic work, bounded same-trajectory selection and one owner."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
import datetime
import json
import sqlite3
import statistics
import threading
import time

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_eval_queue import EvaluationShard, initialize_queue, claim_next
from ember.proposal_writer.batch import Batch, Job
from .contract import TASKS,ACTIVE_TASKS,BASE,SHARD,plan,selection_decision,queue_path,nodes


from .revision import prepare_window as prepare, import_previous_receipts


class Scheduler:
    def __init__(self,root,panel,workers,worker_nodes=None):
        self.root,self.panel,self.workers=root,panel,workers
        self.lock=threading.Lock();self.waiting={};self.running={};self.scores={t:{} for t in TASKS}
        self.decisions={};self.failed=False;self.eval_counts={};self.latest_resources={}
        self.running_slots={};self.environment_GiB={};self.parent_GiB={};self.node_resources={}
        self.worker_node={w:n.split('.')[0].lower() for w,n in (worker_nodes or {}).items()}
        self.resident_slots={};self.environment_PIDs={};self.memory_reserve_GiB=64
        self.evaluation_measures={};self.idled=set();self.load_evaluation_reference()

    def load_evaluation_reference(self):
        source=self.root/'analysis/engineering_completion_window_20261015/evaluation_measurements.json'
        measures=read_json(source);summary={};environment=4.6;shards=[]
        for slots,rows in measures.items():
            summary[slots]=dict(shards=len(rows),native_plans_per_second=statistics.median(
                r['native_plans_per_second'] for r in rows),end_to_end_plans_per_second=statistics.median(
                r['end_to_end_plans_per_second'] for r in rows))
            identity=max(rows,key=lambda r:r['PSS_KiB'])['identity']
            path=self.root/'shards'/f'{identity}.json';shards.append(str(path))
            peak=read_json(path)['output']['resources']['peak_environment_resources']
            environment=max(environment,max(p['PSS_KiB']/1024**2*1.2 for p in peak['per_process']
                if p['PID'] in peak['environment_PIDs']))
        self.reference_environment_GiB=environment
        self.evaluation_reference=dict(source=str(source),summary=summary,memory_shards=shards,
            default_slots=16,reason='reuse measured throughput; reduce only for node CPU admission')

    def statuses(self):
        with closing(sqlite3.connect(queue_path(self.root),timeout=60)) as db:
            return dict(db.execute('SELECT job_id,status FROM jobs'))

    def evaluate_selection(self,statuses):
        for task in TASKS:
            for update in nodes(task):
                if update in self.scores[task]:continue
                identities=[f'selection_t{task:04d}_u{update:08d}_s{s:02d}' for s in range(0,32,SHARD)]
                if not all(statuses[i]=='complete' for i in identities):continue
                self.scores[task][update]=sum(sum(r['success'] for r in read_json(self.root/'shards'/f'{i}.json')['output']['rows']) for i in identities)
            decision=selection_decision(self.scores[task],task)
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
            if task not in ACTIVE_TASKS:continue
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

    def node_budget(self,worker):
        node=self.worker_node[worker];resources=self.node_resources.get(node,{})
        memory=resources.get('host_memory_KiB',{});members=[w for w,n in self.worker_node.items() if n==node]
        environment=self.environment_GiB.get(node,self.reference_environment_GiB)
        parent=self.parent_GiB.get(node,16.);reserved=0.;occupied={};observed={}
        for member in members:
            details=self.latest_resources.get(member,{})
            processes=details.get('per_process',[]);pids=self.environment_PIDs.get(member,set())
            parent_pss=sum(p['PSS_KiB']/1024**2 for p in processes if p['PID']==details.get('PID'))
            observed[member]=sum(p['PSS_KiB']/1024**2 for p in processes if p['PID'] in pids)
            occupied[member]=max(self.running_slots.get(member,0),self.resident_slots.get(member,0))
            reserved+=max(0.,parent-parent_pss)+max(0.,occupied[member]*environment-observed[member])
        total=memory.get('MemTotal',0)/1024**2;available=memory.get('MemAvailable',0)/1024**2
        cap=max(0,min(len(members)*16,int((total-self.memory_reserve_GiB-len(members)*parent)/environment)//4*4))
        return dict(node=node,slot_cap=cap,occupied=occupied,observed=observed,
            available_GiB=available,unreflected_GiB=reserved,environment_GiB=environment,
            workers=len(members),parent_GiB=parent,reserve_GiB=self.memory_reserve_GiB)

    def training_allowed(self,worker):
        budget=self.node_budget(worker)
        released=max(budget['occupied'][worker]*budget['environment_GiB'],budget['observed'][worker])
        return budget['available_GiB']-budget['unreflected_GiB']+released>=self.memory_reserve_GiB

    def choose_work(self,worker,ready):
        training=[i for i in ready if self.panel[i]['kind']=='train' and self.training_allowed(worker)]
        evaluating=[i for i in ready if self.panel[i]['kind']!='train' and self.physical_slots(worker,self.panel[i])]
        active_training=sum(self.panel[i]['kind']=='train' for i in self.running.values())
        if training and (active_training<max(1,self.workers//2) or not evaluating):return training
        return evaluating or training

    def physical_slots(self,worker,work):
        if work['kind']=='train':return 0
        budget=self.node_budget(worker);occupied=budget['occupied'];current=occupied[worker]
        others=sum(count for member,count in occupied.items() if member!=worker)
        for slots in (16,8,4):
            useful=min(slots,len(work['states']))
            extra=max(0,useful-current)*budget['environment_GiB']
            if (others+useful<=budget['slot_cap'] and budget['available_GiB']-
                    budget['unreflected_GiB']-extra>=self.memory_reserve_GiB):return slots
        return 0

    def dispatch(self):
        statuses=self.statuses();self.failed |= any(s=='failed' for s in statuses.values())
        self.evaluate_selection(statuses)
        for worker,stream in list(self.waiting.items()):
            ready=[] if self.failed else self.eligible(statuses)
            message=None
            if ready:
                # Keep ready learning and evaluation concurrent; each evaluation
                # claim still uses the existing long-first cost queue.
                eligible=self.choose_work(worker,ready)
                claim=claim_next(queue_path(self.root),worker_id=worker,eligible_job_ids=eligible) if eligible else None
                if claim is not None:
                    identity=claim.shard.job_id;work=self.panel[identity];slots=self.physical_slots(worker,work)
                    message=dict(command='work',identity=identity,work=work,claim_token=claim.claim_token,slots=slots)
                    self.running[worker]=identity;self.running_slots[worker]=min(slots,len(work['states']))
                    message.update(CPU_admission=self.node_budget(worker),
                        evaluation_reference=self.evaluation_reference['source'])
                    statuses[identity]='claimed'
            if message is None:
                if self.failed or self.terminal(statuses):message=dict(command='stop')
                elif worker not in self.idled:
                    self.idled.add(worker);message=dict(command='idle')
                else:continue
            stream.write(json.dumps(message)+'\n');stream.flush()
            if message['command']!='idle':
                del self.waiting[worker];self.idled.discard(worker)
            with (self.root/'owner_events.jsonl').open('a') as f:
                f.write(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),worker=worker,**message))+'\n')

    def record_resources(self,worker,resource):
        node=resource['hostname'].split('.')[0].lower()
        if worker in self.worker_node and self.worker_node[worker]!=node:
            raise ValueError('worker hostname differs from its configured physical node')
        self.worker_node[worker]=node;self.latest_resources[worker]=resource;self.node_resources[node]=resource
        parent=next((p for p in resource.get('per_process',[]) if p['PID']==resource.get('PID')),None)
        if parent:self.parent_GiB[node]=max(self.parent_GiB.get(node,16.),parent['PSS_KiB']/1024**2*1.2)

    def record_evaluation(self,worker,identity,slots):
        receipt=read_json(self.root/'shards'/f'{identity}.json');resources=receipt['output']['resources']
        peak=resources.get('peak_environment_resources');node=self.worker_node[worker]
        if peak:
            self.environment_PIDs[worker]=set(peak['environment_PIDs'])
            sizes=[p['PSS_KiB']/1024**2 for p in peak['per_process'] if p['PID'] in self.environment_PIDs[worker]]
            if sizes:self.environment_GiB[node]=max(self.environment_GiB.get(node,self.reference_environment_GiB),max(sizes)*1.2)
            current={p['PID'] for p in self.latest_resources[worker].get('per_process',[])}
            self.resident_slots[worker]=len(current & self.environment_PIDs[worker])
        if resources.get('new_episodes',0)>=8:
            measurement=dict(slots=slots,episodes_per_second=resources['episodes_per_second'],
                native_plans_per_second=resources['runner']['plans']/max(1e-6,resources['runner']['native_prefix_seconds']+
                    resources['runner']['native_flow_seconds']),peak_allocated_MiB=receipt['peak_allocated_MiB'],
                environment_GiB=self.environment_GiB.get(node,self.reference_environment_GiB))
            self.evaluation_measures.setdefault(worker,[]).append(measurement)

    def event(self,worker,event,stream):
        with self.lock:
            if event['kind']=='failed':self.failed=True
            if event.get('resources'):self.record_resources(worker,event['resources'])
            with (self.root/'worker_resources.jsonl').open('a') as f:f.write(json.dumps({**event,'worker':worker})+'\n')
            if event['kind']=='environments_closed':
                if event['remaining_environment_PIDs']:raise RuntimeError('idle worker retained environment processes')
                self.resident_slots.pop(worker,None);self.environment_PIDs.pop(worker,None)
            if event['kind']=='ready':
                previous=self.running.pop(worker,None);slots=self.running_slots.pop(worker,0)
                if previous and self.panel[previous]['kind']!='train':self.record_evaluation(worker,previous,slots)
                self.waiting[worker]=stream
            self.dispatch()


def owner(args):
    root=args.root;run=prepare(root);panel=plan(root,run['environment'])
    devices=[(value.split(':')[0],int(value.split(':')[1])) for value in args.devices]
    shards=[EvaluationShard(identity,index,w['kind'],w['task'],
        next(t['horizon'] for t in run['environment']['tasks'] if t['global_task_id']==w['task']),
        tuple(w['states']),w['cost']) for index,(identity,w) in enumerate(panel.items())]
    initialize_queue(queue_path(root),shards,contract_reference=str(root/'run_contract.json'),recover_claims=args.resume,retry_failed=args.resume)
    imported=import_previous_receipts(root)
    write_json_atomic(root/'windows'/run['window']/'queue_import.json',imported)
    scheduler=Scheduler(root,panel,len(devices),worker_nodes={f'reference_worker_{i:02d}':node for i,(node,_) in enumerate(devices)})
    write_json_atomic(root/'windows'/run['window']/'evaluation_execution.json',scheduler.evaluation_reference)
    batch=Batch(Path.cwd(),devices,root=root,required_free_MiB=34000,work_event=scheduler.event)
    jobs=[Job(f'reference_worker_{i:02d}', ['source-reference','worker','--worker-id',f'reference_worker_{i:02d}','--slots','8'],
              eligible_devices=(device,),estimate_seconds=2*3600,required_free_MiB=34000) for i,device in enumerate(devices)]
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
