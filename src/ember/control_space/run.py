"""One persistent owner for independent, budgeted incumbent compilation paths."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,datetime,fcntl,json,os,sqlite3,subprocess,time
import torch
from ember.pi05_source_checkpoint import write_json_atomic,read_json
from ember.pi05_eval_queue import EvaluationShard,initialize_queue,claim_next,complete_job,fail_job
from ember.proposal_writer.batch import Batch,Job,live_gpu_snapshot
from ember.reference.coordinator import Scheduler
from .contract import REFERENCE_ROOT,BASIS,INITIAL_READER
from .practice import specifications
from .compilation import Compilation,maximum_panels
from .model import load_basis


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def queue_path(root):return Path(root)/'evaluation.sqlite'


def device_pool(args):
    folder=args.root/'admission'/f'pool_{time.time_ns()}';folder.mkdir(parents=True)
    snap=live_gpu_snapshot(folder);rows=[(n,g) for n,v in snap['node_results'].items() for g in v['gpus']]
    idle=sum(not g['processes'] and g['mem_used_mib']<1000 for _,g in rows);cap=6 if idle<=10 else 8
    outside={(n,g['index']) for n,g in rows if any(p['owner']=='ymdai' for p in g['processes'])};groups={}
    for node,gpu in sorted(rows,key=lambda r:-r[1]['mem_free_mib']):
        if (f'{node}:{gpu["index"]}' in args.devices and gpu['compute_mode']!='Prohibited'
                and gpu['mem_free_mib']>=18000 and (node,gpu['index']) not in outside):groups.setdefault(node,[]).append(gpu)
    selected=[];available=min(6,cap-len(outside))
    while groups and len(selected)<available:
        for node in list(groups):
            gpu=groups[node].pop(0);selected.append((node,gpu['index'],gpu['mem_free_mib']))
            if not groups[node]:del groups[node]
            if len(selected)==available:break
    if not selected:raise RuntimeError('no live shared device fits actual native/raw-H evaluation')
    return selected,cap


def panel_queue(root,run,tasks):
    work={};shards=[]
    for c in run['conditions']:
        task=tasks[c['task']]
        nodes=[('practice',i) for i in range(maximum_panels(task['horizon'])) if not(c['arm']=='no_video' and i==0)]+[('query',0)]
        for role,ordinal in nodes:
            identity=c['id']+(f'_practice_{ordinal:03d}' if role=='practice' else '_query')
            states=c['practice_states'] if role=='practice' else c['query_states']
            work[identity]=dict(condition=c['id'],role=role,ordinal=ordinal)
            shards.append(EvaluationShard(identity,len(shards),task['suite'],c['task'],task['horizon'],tuple(states),4*(task['horizon']+10)))
    return work,shards


class WorkQueue(Scheduler):
    """Whole four-case decisions share useful16/24 packs, never an arm's state."""
    def __init__(self,root,run,workers,capacities):
        super().__init__(root,{},len(workers),worker_nodes={f'worker_{i:02d}':n for i,(n,_) in enumerate(workers)},
            queue_path_fn=queue_path,evaluation_reference_root=REFERENCE_ROOT)
        self.run=run;self.capacities=capacities;self.inflight={};self.measures=[]
        self.tasks={s['global_task_id']:s for s in run['environment']['tasks']};self.work,_=panel_queue(root,run,self.tasks)
        decoder=load_basis(BASIS).requires_grad_(False);saved=torch.load(INITIAL_READER,map_location='cpu',mmap=True,weights_only=False)
        units=saved['code_units'].float();del saved
        if units.shape!=(64,) or not bool(units.isfinite().all()) or bool((units<=0).any()):raise ValueError('fixed original code units changed')
        self.compilers={c['id']:Compilation(root,c,decoder,units,self.tasks[c['task']]) for c in run['conditions']}
        self.ready={};self.alive=set(self.worker_node);self.refresh()

    def refresh(self):
        self.ready={}
        for compiler in self.compilers.values():
            request=compiler.next()
            if request:self.ready[request['identity']]=request

    def terminal(self,statuses=None):return all(c.state['query_complete'] for c in self.compilers.values())
    def send(self,stream,value):stream.write(json.dumps(value)+'\n');stream.flush()

    def node_budget(self,worker):
        value=super().node_budget(worker);total=self.node_resources.get(value['node'],{}).get('host_memory_KiB',{}).get('MemTotal',0)/2**20
        value['slot_cap']=max(0,int((total-64-value['workers']*value['parent_GiB'])/value['environment_GiB']))
        return value

    def slots(self,worker):
        preferred=24 if self.eval_counts.get(worker,0) else (16 if int(worker[-2:])%2 else 24)
        if self.eval_counts.get(worker,0) and self.measures:preferred=max(self.measures,key=lambda r:r['end_to_end_plans_per_second'])['slots']
        preferred=min(preferred,self.capacities[worker]);budget=self.node_budget(worker)
        others=sum(v for k,v in budget['occupied'].items() if k!=worker)
        for slots in (24,16,8,4):
            extra=max(0,slots-budget['occupied'][worker])*budget['environment_GiB']
            if (slots<=preferred and others+slots<=budget['slot_cap'] and budget['available_GiB']-budget['unreflected_GiB']-extra>=64):return slots
        return 0

    def commit(self,claim,request,rows,worker):
        path=self.root/'shards'/f'{claim.shard.job_id}.json'
        write_json_atomic(path,dict(rows=rows,role=request['role'],condition_id=request['condition_id']))
        complete_job(self.queue_path(),job_id=claim.shard.job_id,worker_id=worker,claim_token=claim.claim_token,
            rows_path=str(path.relative_to(self.root)),rows_bytes=path.stat().st_size,row_count=4,successes=sum(r['success'] for r in rows))

    def adopt_committed(self):
        # Close a crash gap between an atomic condition decision and queue receipt.
        statuses=self.statuses()
        for c in self.compilers.values():
            entries=[(p['identity'],'practice',p['rows']) for p in c.state['panels'] if not p['reused']]
            if c.state['query_complete'] and not c.state['query_reused']:entries.append((c.id+'_query','query',c.state['query_rows']))
            for identity,role,rows in entries:
                if statuses[identity]=='complete':continue
                claim=claim_next(self.queue_path(),worker_id='owner_adopt',eligible_job_ids=[identity])
                if claim is None:raise RuntimeError('committed decision lost its recoverable queue ownership')
                self.commit(claim,dict(role=role,condition_id=c.id),rows,'owner_adopt')

    def dispatch(self):
        statuses=self.statuses();eligible=[i for i in self.ready if statuses[i]=='pending']
        for worker,stream in list(self.waiting.items()):
            unfinished=sum(not c.state['query_complete'] for c in self.compilers.values())
            if self.failed or self.terminal() or len(self.alive)>unfinished:
                self.send(stream,dict(command='stop'));del self.waiting[worker];self.alive.discard(worker);continue
            slots=self.slots(worker);claims=[]
            for _ in range(slots//4):
                claim=claim_next(self.queue_path(),worker_id=worker,eligible_job_ids=eligible) if eligible else None
                if claim is None:break
                claims.append(claim);eligible.remove(claim.shard.job_id)
            if not claims:
                if worker not in self.idled:self.send(stream,dict(command='idle'));self.idled.add(worker)
                continue
            rows=[]
            for claim in claims:rows.extend(specifications(self.ready[claim.shard.job_id]))
            self.inflight[worker]=claims;self.running_slots[worker]=len(rows);self.idled.discard(worker)
            message=dict(command='panel',identity=f'pack_{time.time_ns()}',rows=rows,slots=len(rows),archive_workers=2)
            self.send(stream,message);del self.waiting[worker]
            with (self.root/'owner_events.jsonl').open('a') as f:f.write(json.dumps(dict(UTC=utc(),worker=worker,CPU_admission=self.node_budget(worker),**message))+'\n')

    def complete(self,worker,event):
        claims=self.inflight.pop(worker);groups={}
        for row in event['rows']:groups.setdefault(row['identity'],[]).append(row)
        if set(groups)!={c.shard.job_id for c in claims}:raise ValueError('pack changed a whole panel identity')
        for claim in claims:
            identity=claim.shard.job_id;request=self.ready[identity];rows=groups[identity]
            compiler=self.compilers[request['condition_id']];compiler.accept(dict(rows=rows))
            self.commit(claim,request,rows,worker)
        resource=event['resources'];peak=resource.get('peak_environment_resources')
        if peak:
            self.environment_PIDs[worker]=set(peak['environment_PIDs']);self.resident_slots[worker]=len(peak['environment_PIDs'])
            sizes=[p['PSS_KiB']/2**20 for p in peak['per_process'] if p['PID'] in self.environment_PIDs[worker]]
            if sizes:self.environment_GiB[self.worker_node[worker]]=max(self.reference_environment_GiB,max(sizes)*1.2)
        runner=resource['runner'];seconds=resource['seconds'];new=resource['new_episodes']
        measure=dict(worker=worker,slots=event['slots'],new_episodes=new,seconds=seconds,arms={a:sum(r['arm']==a for r in event['rows']) for a in ('video','no_video')},
            environment_steps=sum(r['environment_steps'] for r in event['rows']),end_to_end_plans_per_second=runner['plans']/max(1e-6,seconds),native_plans_per_second=runner['plans']/max(1e-6,runner['native_prefix_seconds']+runner['native_flow_seconds']),
            peak_allocated_MiB=event['peak_allocated_MiB'],peak_reserved_MiB=event['peak_reserved_MiB'],PSS_KiB=(peak or {}).get('PSS_KiB'),
            H_archive_bytes=sum(r.get('history_bytes',0) for r in event['rows']),H_primary_bytes=sum(r.get('history_primary_raw_bytes',0) for r in event['rows']))
        if new>=8 and event['slots'] in (16,24):self.measures.append(measure)
        self.eval_counts[worker]=self.eval_counts.get(worker,0)+1
        write_json_atomic(self.root/'packs'/f'{event["identity"]}.json',dict(**event,measure=measure))
        if not self.failed:self.refresh()
        print('EMBER_USEFUL_OBSERVATION '+json.dumps(measure),flush=True)

    def event(self,worker,event,stream):
        with self.lock:
            with (self.root/'worker_resources.jsonl').open('a') as f:f.write(json.dumps(dict(event,worker=worker,UTC=utc()))+'\n')
            resource=event.get('resources',{})
            if resource:self.record_resources(worker,resource if 'hostname' in resource else resource['after_panel_resources'])
            if event['kind']=='failed':
                self.failed=True
                for claim in self.inflight.get(worker,[]):fail_job(self.queue_path(),job_id=claim.shard.job_id,worker_id=worker,claim_token=claim.claim_token,error=event['error'])
            elif event['kind']=='environments_closed':
                if event['remaining_environment_PIDs']:raise RuntimeError('idle/exit retained environment children')
                self.environment_PIDs.pop(worker,None);self.resident_slots.pop(worker,None)
            elif event['kind']=='complete':self.complete(worker,event)
            elif event['kind']=='ready':self.running_slots.pop(worker,None);self.waiting[worker]=stream
            self.dispatch()


def owner(args):
    from .comparison import prepare,analyze
    args.root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(8)
    with (args.root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.root/'completion.json').exists():raise ValueError('completed finite compilation cannot restart')
        for name in ('shards','packs'):(args.root/name).mkdir(exist_ok=True)
        run=prepare(args.root);tasks={s['global_task_id']:s for s in run['environment']['tasks']};work,shards=panel_queue(args.root,run,tasks)
        initialize_queue(queue_path(args.root),shards,contract_reference=str(args.root/'run_contract.json'),recover_claims=args.resume,retry_failed=args.resume)
        choices,cap=device_pool(args);workers=[(n,g) for n,g,_ in choices]
        capacities={f'worker_{i:02d}':24 if free>=40000 else 16 if free>=28000 else 8 for i,(_,_,free) in enumerate(choices)}
        scheduler=WorkQueue(args.root,run,workers,capacities);scheduler.adopt_committed()
        write_json_atomic(args.root/'execution.json',dict(devices=choices,total_cap=cap,slot_capacities=capacities,maximum_conditional_panel_jobs=len(work),
            reused_evaluation_measurements=scheduler.evaluation_reference,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        batch=Batch(Path.cwd(),workers,root=args.root,required_free_MiB=18000,work_event=scheduler.event)
        jobs=[Job(f'worker_{i:02d}',['control-space','worker','--worker-id',f'worker_{i:02d}','--slots','24'],eligible_devices=(device,),estimate_seconds=6000,
            required_free_MiB={24:40000,16:28000,8:18000}[capacities[f'worker_{i:02d}']]) for i,device in enumerate(workers)]
        failure=None
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures=[pool.submit(batch.execute,j,d) for j,d in zip(jobs,workers)]
            for future in as_completed(futures):
                try:future.result()
                except BaseException as error:
                    failure=repr(error)
                    with scheduler.lock:scheduler.failed=True;scheduler.dispatch()
        if failure:raise RuntimeError(failure)
        if not scheduler.terminal():raise RuntimeError('all72 conditions must finish, including each independent query')
        with sqlite3.connect(queue_path(args.root)) as db:db.execute("UPDATE jobs SET status='not_triggered' WHERE status='pending'")
        return analyze(args.root)


def main():
    from .comparison import ROOT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['owner','worker']);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--devices',nargs='+');parser.add_argument('--worker-id');parser.add_argument('--slots',type=int,default=24)
    parser.add_argument('--physical-gpu',type=int,default=0);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip() or subprocess.check_output(['git','branch','--show-current'],text=True).strip():raise ValueError('formal compilation requires clean pushed detached source')
    subprocess.run(['git','merge-base','--is-ancestor','HEAD','origin/main'],check=True)
    if args.command=='worker':
        from .readout import worker
        worker(args)
    else:owner(args)

if __name__=='__main__':main()
