"""One bounded owner for saved-controller comparison, with no learning entrypoint."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,datetime,fcntl,json,math,os,sqlite3,subprocess,time
from ember.pi05_source_checkpoint import write_json_atomic,read_json
from ember.pi05_eval_queue import EvaluationShard,initialize_queue,claim_next,complete_job,fail_job
from ember.proposal_writer.batch import Batch,Job,live_gpu_snapshot
from ember.reference.coordinator import Scheduler
from .contract import REFERENCE_ROOT
from .practice import case_id,paths


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def queue_path(root):return Path(root)/'evaluation.sqlite'


def device_pool(args):
    folder=args.root/'admission'/f'pool_{time.time_ns()}';folder.mkdir(parents=True)
    snap=live_gpu_snapshot(folder);rows=[(n,g) for n,v in snap['node_results'].items() for g in v['gpus']]
    idle=sum(not g['processes'] and g['mem_used_mib']<1000 for _,g in rows);cap=6 if idle<=10 else 8
    outside={(n,g['index']) for n,g in rows if any(p['owner']=='ymdai' for p in g['processes'])}
    groups={}
    for n,g in sorted(rows,key=lambda r:-r[1]['mem_free_mib']):
        if (f'{n}:{g["index"]}' in args.devices and g['compute_mode']!='Prohibited'
                and g['mem_free_mib']>=14000 and (n,g['index']) not in outside):groups.setdefault(n,[]).append(g)
    #24 is the highest actually completed renderer pack;32 failed at EGL
    #creation. Avoid loading replicas that cannot shorten the remaining work.
    with sqlite3.connect(queue_path(args.root)) as db:
        pending=db.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()[0]
    available=min(6,cap-len(outside),math.ceil(pending/24));selected=[]
    while groups and len(selected)<available:
        for node in list(groups):
            g=groups[node].pop(0);selected.append((node,g['index'],g['mem_free_mib']))
            if not groups[node]:del groups[node]
            if len(selected)==available:break
    if not selected:raise RuntimeError('no live device fits the measured saved-controller evaluator')
    return selected,cap


class WorkQueue(Scheduler):
    """Bundle existing atomic row claims; reuse node PSS admission and queue costs."""
    def __init__(self,root,requests,workers,ceilings,capacities):
        super().__init__(root,{},len(workers),worker_nodes={f'query_worker_{i:02d}':n for i,(n,_) in enumerate(workers)},
            queue_path_fn=queue_path,evaluation_reference_root=REFERENCE_ROOT)
        self.requests={case_id(s):s for s in requests};self.inflight={};self.ceilings=ceilings;self.capacities=capacities;self.measures=[]
        self.task_horizon={s['global_task_id']:s['horizon'] for s in read_json(root/'run_contract.json')['environment']['tasks']}

    def send(self,stream,message):stream.write(json.dumps(message)+'\n');stream.flush()

    def node_budget(self,worker):
        value=super().node_budget(worker)
        # The old reference pool capped every replica at16; this query pool can
        # use32 only when the same measured node memory calculation admits it.
        total=self.node_resources.get(value['node'],{}).get('host_memory_KiB',{}).get('MemTotal',0)/2**20
        value['slot_cap']=max(0,int((total-64-value['workers']*value['parent_GiB'])/value['environment_GiB']))
        return value

    def slots(self,worker,pending):
        budget=self.node_budget(worker);preferred=self.ceilings[worker]
        if worker in self.eval_counts and self.measures:
            preferred=max(self.measures,key=lambda r:r['native_plans_per_second'])['slots']
        preferred=min(preferred,self.capacities[worker])
        candidates=[s for s in (24,16,8,4) if s<=preferred]
        others=sum(v for k,v in budget['occupied'].items() if k!=worker)
        for slots in candidates:
            useful=min(slots,pending);extra=max(0,useful-budget['occupied'][worker])*budget['environment_GiB']
            if (others+useful<=budget['slot_cap'] and budget['available_GiB']-budget['unreflected_GiB']-extra>=64):return useful
        return 0

    def dispatch(self):
        statuses=self.statuses();pending=sum(v=='pending' for v in statuses.values())
        for worker,stream in list(self.waiting.items()):
            if self.failed or not pending:
                self.send(stream,dict(command='stop'));del self.waiting[worker];continue
            slots=self.slots(worker,pending)
            if not slots:
                if worker not in self.idled:self.send(stream,dict(command='idle'));self.idled.add(worker)
                continue
            claims=[claim_next(self.queue_path(),worker_id=worker) for _ in range(slots)]
            claims=[c for c in claims if c is not None];pending-=len(claims)
            if not claims:continue
            self.inflight[worker]=claims;self.running_slots[worker]=len(claims);self.idled.discard(worker)
            self.send(stream,dict(command='panel',identity=f'pack_{time.time_ns()}',rows=[self.requests[c.shard.job_id] for c in claims],
                slots=len(claims),archive_workers=2));del self.waiting[worker]

    def complete(self,worker,event):
        claims=self.inflight.pop(worker)
        if len(claims)!=len(event['rows']):raise ValueError('worker lost registered query rows')
        for claim,row in zip(claims,event['rows'],strict=True):
            spec=self.requests[claim.shard.job_id]
            if (spec['task'],spec['state_id'],spec['arm'])!=(row['global_task_id'],row['init_state_id'],row['arm']):
                raise ValueError('actual query changed task/state/controller identity')
            path=self.root/'shards'/f'{claim.shard.job_id}.json';write_json_atomic(path,dict(row=row,episode_path=str(paths(self.root,spec)[0])))
            complete_job(self.queue_path(),job_id=claim.shard.job_id,worker_id=worker,claim_token=claim.claim_token,
                rows_path=str(path.relative_to(self.root)),rows_bytes=path.stat().st_size,row_count=1,successes=int(row['success']))
        resource=event['resources'];peak=resource.get('peak_environment_resources')
        if peak:
            self.environment_PIDs[worker]=set(peak['environment_PIDs']);self.resident_slots[worker]=len(peak['environment_PIDs'])
            sizes=[p['PSS_KiB']/2**20 for p in peak['per_process'] if p['PID'] in self.environment_PIDs[worker]]
            if sizes:self.environment_GiB[self.worker_node[worker]]=max(self.reference_environment_GiB,max(sizes)*1.2)
        runner=resource['runner'];seconds=resource['seconds'];new=resource['new_episodes']
        measure=dict(worker=worker,slots=event['slots'],new_episodes=new,seconds=seconds,
            environment_steps=sum(r['environment_steps'] for r in event['rows']),
            native_plans_per_second=runner['plans']/max(1e-6,runner['native_prefix_seconds']+runner['native_flow_seconds']),
            peak_allocated_MiB=event['peak_allocated_MiB'],peak_reserved_MiB=event['peak_reserved_MiB'],
            PSS_KiB=(peak or {}).get('PSS_KiB'))
        if new>=8:self.measures.append(measure)
        self.eval_counts[worker]=self.eval_counts.get(worker,0)+1
        write_json_atomic(self.root/'packs'/f'{event["identity"]}.json',dict(**event,measure=measure))
        print('EMBER_USEFUL_OBSERVATION '+json.dumps(measure),flush=True)

    def event(self,worker,event,stream):
        with self.lock:
            with (self.root/'worker_resources.jsonl').open('a') as f:f.write(json.dumps(dict(event,worker=worker,UTC=utc()))+'\n')
            resource=event.get('resources',{})
            if resource:self.record_resources(worker,resource if 'hostname' in resource else resource['after_panel_resources'])
            if event['kind']=='failed':
                self.failed=True
                for claim in self.inflight.get(worker,[]):
                    fail_job(self.queue_path(),job_id=claim.shard.job_id,worker_id=worker,claim_token=claim.claim_token,error=event['error'])
            elif event['kind']=='environments_closed':
                if event['remaining_environment_PIDs']:raise RuntimeError('idle/exit retained environment children')
                self.environment_PIDs.pop(worker,None);self.resident_slots.pop(worker,None)
            elif event['kind']=='complete':self.complete(worker,event)
            elif event['kind']=='ready':
                self.running_slots.pop(worker,None);self.waiting[worker]=stream
            self.dispatch()


def owner(args):
    from .comparison import prepare,analyze,execution_contract
    args.root.mkdir(parents=True,exist_ok=True)
    with (args.root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.root/'completion.json').exists():raise ValueError('completed finite comparison cannot restart')
        for name in ('shards','packs'):(args.root/name).mkdir(exist_ok=True)
        prepare(args.root);run=execution_contract(args.root);tasks={s['global_task_id']:s for s in run['environment']['tasks']}
        shards=[EvaluationShard(case_id(s),i,tasks[s['task']]['suite'],s['task'],tasks[s['task']]['horizon'],
            (s['state_id'],),tasks[s['task']]['horizon']+10) for i,s in enumerate(run['requests'])]
        initialize_queue(queue_path(args.root),shards,contract_reference=str(args.root/'run_contract.json'),
            recover_claims=args.resume,retry_failed=args.resume)
        choices,cap=device_pool(args);workers=[(n,g) for n,g,_ in choices]
        capacities={f'query_worker_{i:02d}':24 if free>=40000 else 16 if free>=28000 else 8 if free>=18000 else 4 for i,(_,_,free) in enumerate(choices)}
        ceilings=dict(capacities)
        scheduler=WorkQueue(args.root,run['requests'],workers,ceilings,capacities)
        write_json_atomic(args.root/'execution.json',dict(devices=choices,total_cap=cap,initial_slot_ceilings=ceilings,
            reused_evaluation_measurements=scheduler.evaluation_reference,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        batch=Batch(Path.cwd(),workers,root=args.root,required_free_MiB=14000,work_event=scheduler.event)
        jobs=[Job(f'query_worker_{i:02d}',['control-space','worker','--worker-id',f'query_worker_{i:02d}','--slots','24'],
            eligible_devices=(device,),estimate_seconds=1500,
            required_free_MiB={24:40000,16:28000,8:18000,4:14000}[capacities[f'query_worker_{i:02d}']]) for i,device in enumerate(workers)]
        failure=None
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures=[pool.submit(batch.execute,j,d) for j,d in zip(jobs,workers)]
            for future in as_completed(futures):
                try:future.result()
                except BaseException as error:
                    failure=repr(error)
                    with scheduler.lock:scheduler.failed=True;scheduler.dispatch()
        if failure:raise RuntimeError(failure)
        if any(v!='complete' for v in scheduler.statuses().values()):raise RuntimeError('124 registered new rows not complete')
        return analyze(args.root)


def main():
    from .comparison import ROOT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['owner','worker'])
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--devices',nargs='+')
    parser.add_argument('--worker-id');parser.add_argument('--slots',type=int,default=24)
    parser.add_argument('--physical-gpu',type=int,default=0);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip() or subprocess.check_output(['git','branch','--show-current'],text=True).strip():
        raise ValueError('formal comparison requires clean pushed detached source')
    subprocess.run(['git','merge-base','--is-ancestor','HEAD','origin/main'],check=True)
    if args.command=='worker':
        from .readout import worker
        worker(args)
    else:owner(args)

if __name__=='__main__':main()
