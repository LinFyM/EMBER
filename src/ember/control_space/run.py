"""One owner for fixed feedback180 validation400 and audited raw-H retirement."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import closing
from itertools import combinations
from dataclasses import replace
import argparse,datetime,fcntl,json,os,shlex,sqlite3,subprocess,time
import torch
from ember.pi05_source_checkpoint import write_json_atomic,read_json
from ember.pi05_eval_queue import EvaluationShard,initialize_queue,claim_next,complete_job,fail_job
from ember.proposal_writer.batch import Batch,Job,DeviceWorkloadFailure,live_gpu_snapshot,quota_headroom
from ember.reference.coordinator import Scheduler
from .contract import REFERENCE_ROOT,BASIS,INITIAL_READER,request_identity
from .practice import specifications
from .compilation import Compilation,maximum_panels
from .model import load_basis
from .lifecycle import receipt_path


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def queue_path(root):return Path(root)/'evaluation.sqlite'


def storage_admission(root):
    quota=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','strg01',"xfs_quota -c 'quota -u -b ymdai' /data1"],text=True)
    used,limit=quota_headroom(quota,peak_GiB=96)
    shared=os.statvfs('/data1');free=shared.f_bavail*shared.f_frsize
    if free<96*2**30:raise RuntimeError('shared data1 free capacity cannot cover96GiB peak')
    relevant=subprocess.check_output(['du','-sk',str(REFERENCE_ROOT.parent)],text=True)
    root.mkdir(parents=True,exist_ok=True)
    (root/'admission').mkdir(exist_ok=True)
    write_json_atomic(root/'admission'/f'storage_{time.time_ns()}.json',dict(UTC=utc(),quota=quota,used_KiB=used,quota_KiB=limit,
        shared_free_bytes=free,relevant_usage=relevant,additional_peak_GiB=96,new_outputs_filesystem='data1'))


def cpu_snapshot(root,label):
    code="""import os,json,pathlib
pss=threads=processes=zombies=0
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  status=(p/'status').read_text().splitlines();processes+=1
  threads+=next(int(s.split()[1]) for s in status if s.startswith('Threads:'))
  zombies+=next(s.split()[1] for s in status if s.startswith('State:'))=='Z'
  try:pss+=next(int(s.split()[1]) for s in (p/'smaps_rollup').read_text().splitlines() if s.startswith('Pss:'))
  except OSError:pass
 except (OSError,StopIteration):pass
memory={s.split(':')[0]:int(s.split()[1]) for s in pathlib.Path('/proc/meminfo').read_text().splitlines() if s.startswith(('MemTotal:','MemAvailable:','SwapFree:'))}
print(json.dumps(dict(host=os.uname().nodename,uid_PSS_KiB=pss,uid_threads=threads,uid_processes=processes,uid_zombies=zombies,memory_KiB=memory)))"""
    results={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(subprocess.check_output,['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',node,
            'python3 -c '+shlex.quote(code)],text=True):node for node in ('gpu01','gpu02')}
        for future in as_completed(futures):results[futures[future]]=json.loads(future.result())
    write_json_atomic(root/'admission'/f'CPU_{label}_{time.time_ns()}.json',dict(UTC=utc(),nodes=results));return results


def gpu_candidates(args,snap,minimum):
    rows=[(n,g) for n,v in snap['node_results'].items() for g in v['gpus']]
    idle=sum(not g['processes'] and g['mem_used_mib']<1000 for _,g in rows);cap=6 if idle<=10 else 8
    outside={(n,g['index']) for n,g in rows if any(p['owner']=='ymdai' for p in g['processes'])};groups={}
    for node,gpu in sorted(rows,key=lambda r:-r[1]['mem_free_mib']):
        if (f'{node}:{gpu["index"]}' in args.devices and gpu['compute_mode']!='Prohibited'
                and gpu['mem_free_mib']>=minimum and (node,gpu['index']) not in outside):groups.setdefault(node,[]).append(gpu)
    return groups,cap,len(outside)


def device_pool(args):
    folder=args.root/'admission'/f'pool_{time.time_ns()}';folder.mkdir(parents=True)
    snap=live_gpu_snapshot(folder);groups,cap,outside=gpu_candidates(args,snap,18000)
    resources=cpu_snapshot(args.root,'compile');selected=[]
    groups={n:gs[:min(6,max(0,int((resources[n]['memory_KiB']['MemAvailable']/2**20-64)//32)))] for n,gs in groups.items()}
    groups={n:gs for n,gs in groups.items() if gs};available=max(0,cap-outside)
    candidates=[(n,g) for n,gs in groups.items() for g in gs]
    # Reuse the measured environment footprint owned by the existing scheduler.
    guide=Scheduler(args.root,{},1,evaluation_reference_root=REFERENCE_ROOT)
    env=guide.reference_environment_GiB
    def predicted(plan):
        slots=0
        for node in groups:
            here=[g for n,g in plan if n==node];workers=len(here)
            memory=resources[node]['memory_KiB']['MemAvailable']/2**20
            cpu_slots=max(0,int((memory-64-workers*16)/env)//4*4)
            slots+=min(sum(16 if g['mem_free_mib']>=28000 else 8 for g in here),cpu_slots)
        return slots,sum(g['mem_free_mib'] for _,g in plan)
    plans=(p for count in range(1,min(available,len(candidates))+1) for p in combinations(candidates,count))
    best=max(plans,key=predicted,default=())
    selected=[(n,g['index'],g['mem_free_mib']) for n,g in best]
    write_json_atomic(folder/'allocation.json',dict(predicted_useful_episode_slots=predicted(best)[0],
        environment_GiB=env,parent_GiB=16,reserve_GiB=64,devices=selected,
        reason='joint native VRAM capacity and measured per-node CPU footprint; useful work refines throughput'))
    if not selected:raise RuntimeError('no live shared device fits actual native/raw-H evaluation')
    return selected,cap


def panel_queue(root,run,tasks):
    work={};shards=[]
    for c in run['conditions']:
        task=tasks[c['task']]
        nodes=[('practice',i) for i in range(maximum_panels(task['horizon']))]+[('query',0)]
        for role,ordinal in nodes:
            identity=request_identity(c,role,ordinal)
            states=c['practice_states'] if role=='practice' else c['query_states']
            work[identity]=dict(condition=c['id'],role=role,ordinal=ordinal)
            shards.append(EvaluationShard(identity,len(shards),task['suite'],c['task'],task['horizon'],tuple(states),len(states)*(task['horizon']+10)))
    return work,shards


class WorkQueue(Scheduler):
    """Causal proposals and whole four-case panels share one persistent runtime."""
    def __init__(self,root,run,workers,capacities):
        super().__init__(root,{},len(workers),worker_nodes={f'worker_{i:02d}':n for i,(n,_) in enumerate(workers)},
            queue_path_fn=queue_path,evaluation_reference_root=REFERENCE_ROOT)
        self.run=run;self.capacities=capacities;self.inflight={};self.retiring={};self.measures=[]
        self.tasks={s['global_task_id']:s for s in run['environment']['tasks']};self.work,_=panel_queue(root,run,self.tasks)
        decoder=load_basis(BASIS).requires_grad_(False);saved=torch.load(INITIAL_READER,map_location='cpu',mmap=True,weights_only=False)
        units=saved['code_units'].float();del saved
        if units.shape!=(64,) or not bool(units.isfinite().all()) or bool((units<=0).any()):raise ValueError('fixed original code units changed')
        self.compilers={c['id']:Compilation(root,dict(c,original=c['teaching']),decoder,units,self.tasks[c['task']]) for c in run['conditions']}
        self.retired={i for i in self.compilers if receipt_path(self.root,i).exists()}
        self.raw_bytes=sum(r.get('history_bytes',0) for c in self.compilers.values() for p in c.state['panels']
            for r in p['rows'] if Path(r['history_path']).is_file())
        self.raw_peak=self.raw_bytes;self.pause_admission=False
        self.ready={};self.alive=set(self.worker_node);self.refresh()

    def refresh(self):
        active={self.work[claim.shard.job_id]['condition'] for claims in self.inflight.values() for claim in claims}
        admitted=[c for c in self.compilers.values() if c.state.get('admitted') and c.id not in self.retired]
        waiting=[c for c in self.compilers.values() if not c.state.get('admitted')]
        # Full-H admission is independent of worker count; completion immediately
        # makes room after its atomic audit/retirement, never400 initial panels.
        waiting.sort(key=lambda c:(-c.task['horizon'],c.condition['final_init'],c.condition['task']))
        for compiler in waiting[:0 if self.pause_admission else max(0,32-len(admitted))]:
            compiler.state.update(admitted=True,admitted_UTC=utc());compiler.save();admitted.append(compiler)
        self.ready={}
        for compiler in admitted:
            identity=compiler.id
            request=(compiler.state.get('pending') or compiler.state.get('pending_proposal')) if identity in active else compiler.next()
            if request:self.ready[request['identity']]=request

    def record_storage(self):
        self.raw_peak=max(self.raw_peak,self.raw_bytes)
        # 44GiB raw leaves the registered19.3GiB candidates,17GiB caches and
        # atomic/fault/metadata allowance inside96GiB. Unexpected growth stops
        # only new admission; admitted complete-H work is never truncated.
        self.pause_admission=self.raw_bytes>44*2**30
        write_json_atomic(self.root/'storage_usage.json',dict(raw_bytes=self.raw_bytes,raw_peak_bytes=self.raw_peak,
            pause_new_admission=self.pause_admission,inflight_limit=32,registered_peak_GiB=96,UTC=utc()))
        if self.pause_admission:raise RuntimeError('actual raw H exceeds44GiB allocation; preserve active progress and report resource boundary')

    def terminal(self,statuses=None):
        return all(c.state['query_complete'] and c.id in self.retired for c in self.compilers.values())
    def send(self,stream,value):stream.write(json.dumps(value)+'\n');stream.flush()

    def node_budget(self,worker):
        value=super().node_budget(worker);total=self.node_resources.get(value['node'],{}).get('host_memory_KiB',{}).get('MemTotal',0)/2**20
        value['slot_cap']=max(0,int((total-64-value['workers']*value['parent_GiB'])/value['environment_GiB']))
        return value

    def slots(self,worker):
        # §16 measured16 faster end-to-end than24 with actual H; no32 reprobe.
        preferred=min(16,self.capacities[worker]);budget=self.node_budget(worker)
        others=sum(v for k,v in budget['occupied'].items() if k!=worker)
        for slots in (16,8,4,2,1):
            extra=max(0,slots-budget['occupied'][worker])*budget['environment_GiB']
            if (slots<=preferred and others+slots<=budget['slot_cap'] and budget['available_GiB']-budget['unreflected_GiB']-extra>=64):return slots
        return 0

    def commit(self,claim,request,rows,worker):
        path=self.root/'shards'/f'{claim.shard.job_id}.json'
        value=dict(rows=rows,role=request['role'],condition_id=request['condition_id'])
        if path.exists():
            if read_json(path)!=value:raise ValueError('adopted shard differs from its original committed rows')
        else:write_json_atomic(path,value)
        complete_job(self.queue_path(),job_id=claim.shard.job_id,worker_id=worker,claim_token=claim.claim_token,
            rows_path=str(path.relative_to(self.root)),rows_bytes=path.stat().st_size,row_count=len(rows),successes=sum(r['success'] for r in rows))

    def adopt_committed(self):
        # Close a crash gap between an atomic condition decision and queue receipt.
        statuses=self.statuses()
        for c in self.compilers.values():
            entries=[(p['identity'],'practice',p['rows']) for p in c.state['panels'] if not p['reused']]
            if c.state['query_complete'] and not c.state['query_reused']:entries.append((c.state['query_rows'][0]['identity'],'query',c.state['query_rows']))
            for identity,role,rows in entries:
                if statuses[identity]=='complete':continue
                claim=claim_next(self.queue_path(),worker_id='owner_adopt',eligible_job_ids=[identity])
                if claim is None:raise RuntimeError('committed decision lost its recoverable queue ownership')
                self.commit(claim,dict(role=role,condition_id=c.id),rows,'owner_adopt')

    def send_claims(self,worker,stream,claims):
        requests=[self.ready[c.shard.job_id] for c in claims];kind=requests[0]['command']
        if any(r['command']!=kind for r in requests):raise ValueError('proposal and actual-panel work cannot mix')
        message=dict(command=kind,identity=f'pack_{time.time_ns()}')
        if kind=='propose':message['requests']=requests
        else:
            rows=[s for r in requests for s in specifications(r)]
            message.update(rows=rows,slots=len(rows),archive_workers=2)
            self.running_slots[worker]=len(rows)
        self.inflight[worker]=claims;self.idled.discard(worker)
        self.send(stream,message);del self.waiting[worker]
        with (self.root/'owner_events.jsonl').open('a') as f:f.write(json.dumps(dict(UTC=utc(),worker=worker,CPU_admission=self.node_budget(worker),**message))+'\n')

    def retirement_candidates(self):
        inflight={i for values in self.retiring.values() for i in values}
        return [c.id for c in self.compilers.values() if c.state['query_complete']
            and c.id not in self.retired and c.id not in inflight]

    def claim_pack(self,worker,slots,eligible):
        claims=[];room=slots;kind=None
        while room:
            candidates=[i for i in eligible if (kind is None or self.ready[i]['command']==kind)
                and (4 if self.ready[i]['command']=='propose' else len(self.ready[i]['states']))<=room]
            claim=claim_next(self.queue_path(),worker_id=worker,eligible_job_ids=candidates) if candidates else None
            if claim is None:break
            claims.append(claim);identity=claim.shard.job_id;eligible.remove(identity)
            kind=self.ready[identity]['command'];room-=4 if kind=='propose' else len(self.ready[identity]['states'])
        return claims

    def dispatch(self):
        statuses=self.statuses();eligible=[i for i in self.ready if statuses[i]=='pending']
        completed=self.retirement_candidates()
        for worker,stream in list(self.waiting.items()):
            if worker in self.inflight:
                if not self.failed:self.send_claims(worker,stream,self.inflight[worker])
                else:self.send(stream,dict(command='stop'));del self.waiting[worker]
                continue
            if completed and not self.failed:
                chosen=completed[:4];del completed[:4];self.retiring[worker]=chosen
                self.send(stream,dict(command='retire',conditions=chosen));del self.waiting[worker];continue
            unfinished=sum(not c.state['query_complete'] for c in self.compilers.values())
            remaining=max(1,unfinished+len(completed)+len(self.retiring))
            if self.failed or self.terminal() or len(self.alive)>remaining:
                self.send(stream,dict(command='stop'));del self.waiting[worker];self.alive.discard(worker);continue
            claims=self.claim_pack(worker,self.slots(worker),eligible)
            if not claims:
                if worker not in self.idled:self.send(stream,dict(command='idle'));self.idled.add(worker)
                continue
            self.send_claims(worker,stream,claims)

    def proposed(self,worker,event):
        claims=self.inflight.pop(worker);receipts={r['request']['identity']:r for r in event['receipts']}
        if set(receipts)!={c.shard.job_id for c in claims}:raise ValueError('proposal pack changed saved causal decisions')
        pending=[]
        for claim in claims:
            identity=claim.shard.job_id;compiler=self.compilers[self.ready[identity]['condition_id']]
            if compiler.accept_proposal(receipts[identity]):pending.append(claim)
            else:
                with closing(sqlite3.connect(self.queue_path())) as db,db:
                    changed=db.execute("UPDATE jobs SET status='not_triggered' WHERE job_id=? AND status='claimed' AND worker_id=? AND claim_token=?",
                        (identity,worker,claim.claim_token)).rowcount
                    if changed!=1:raise ValueError('repeated-code stop lost its queue claim')
        if pending:self.inflight[worker]=pending
        write_json_atomic(self.root/'packs'/f'{event["identity"]}.json',event)
        self.refresh()

    def complete(self,worker,event):
        claims=self.inflight.pop(worker);groups={}
        for row in event['rows']:groups.setdefault(row['identity'],[]).append(row)
        if set(groups)!={c.shard.job_id for c in claims}:raise ValueError('pack changed a whole panel identity')
        for claim in claims:
            identity=claim.shard.job_id;request=self.ready[identity];rows=groups[identity]
            compiler=self.compilers[request['condition_id']];compiler.accept(dict(rows=rows))
            self.commit(claim,request,rows,worker)
        self.raw_bytes+=sum(r.get('history_bytes',0) for r in event['rows'] if r['role']=='practice')
        self.record_storage()
        resource=event['resources'];peak=resource.get('peak_environment_resources')
        if peak:
            self.environment_PIDs[worker]=set(peak['environment_PIDs']);self.resident_slots[worker]=len(peak['environment_PIDs'])
            sizes=[p['PSS_KiB']/2**20 for p in peak['per_process'] if p['PID'] in self.environment_PIDs[worker]]
            if sizes:self.environment_GiB[self.worker_node[worker]]=max(self.reference_environment_GiB,max(sizes)*1.2)
        runner=resource['runner'];seconds=resource['seconds'];new=resource['new_episodes']
        measure=dict(worker=worker,slots=event['slots'],new_episodes=new,seconds=seconds,
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
            elif event['kind']=='retired':
                expected=set(self.retiring.pop(worker));actual={r['condition_id'] for r in event['receipts']}
                if actual!=expected or any(not receipt_path(self.root,i).exists() for i in actual):raise ValueError('retirement lost completed conditions')
                self.raw_bytes-=sum(r['deleted_raw_bytes'] for r in event['receipts'])
                self.record_storage()
                self.retired.update(actual)
                for i in actual:
                    compiler=self.compilers[i];compiler.state['raw_H_lifecycle']=str(receipt_path(self.root,i));compiler.save()
                self.refresh()
            elif event['kind']=='complete':self.complete(worker,event)
            elif event['kind']=='proposed':self.proposed(worker,event)
            elif event['kind']=='ready':self.running_slots.pop(worker,None);self.waiting[worker]=stream
            self.dispatch()


def execute_worker(batch,job,device,scheduler):
    """A prelaunch shared-device change must not stop unrelated ready work."""
    def launched():
        attempts=sorted((scheduler.root/'jobs'/job.identity).glob('attempt_*'))
        return bool(attempts and (attempts[-1]/'launch.json').exists())
    error=None
    try:return batch.execute(job,device)
    except DeviceWorkloadFailure as failure:
        if launched():raise
        error=repr(failure)
    if scheduler.capacities[job.identity]>8:
        with scheduler.lock:scheduler.capacities[job.identity]=8
        arguments=list(job.arguments);arguments[arguments.index('--slots')+1]='8'
        job=replace(job,arguments=arguments,required_free_MiB=18000)
        write_json_atomic(scheduler.root/'admission'/f'{job.identity}_reduced_before_launch_{time.time_ns()}.json',
            dict(worker=job.identity,error=error,device=device,physical_slots=8,required_free_MiB=18000,
                no_child_launched=True,no_environment_submission=True,UTC=utc()))
        try:return batch.execute(job,device)
        except DeviceWorkloadFailure as failure:
            if launched():raise
            error=repr(failure)
    with scheduler.lock:
        if job.identity in scheduler.inflight or job.identity in scheduler.retiring:
            raise RuntimeError('prelaunch withdrawal cannot own scientific work')
        scheduler.alive.discard(job.identity);scheduler.worker_node.pop(job.identity,None)
        scheduler.waiting.pop(job.identity,None)
        write_json_atomic(scheduler.root/'admission'/f'{job.identity}_unavailable_{time.time_ns()}.json',
            dict(worker=job.identity,error=error,device=device,no_child_launched=True,no_environment_submission=True,UTC=utc()))
        scheduler.dispatch()
    return job.identity


def owner(args):
    from .comparison import prepare,analyze
    storage_admission(args.root);torch.set_num_threads(8)
    with (args.root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.root/'completion.json').exists():raise ValueError('completed finite compilation cannot restart')
        for name in ('shards','packs','proposals','admission'):(args.root/name).mkdir(exist_ok=True)
        run=prepare(args.root)
        tasks={s['global_task_id']:s for s in run['environment']['tasks']};work,shards=panel_queue(args.root,run,tasks)
        reference=args.root/'run_contract.json'
        initialize_queue(queue_path(args.root),shards,contract_reference=str(reference),recover_claims=args.resume,retry_failed=args.resume)
        choices,cap=device_pool(args);workers=[(n,g) for n,g,_ in choices]
        capacities={f'worker_{i:02d}':16 if free>=28000 else 8 for i,(_,_,free) in enumerate(choices)}
        scheduler=WorkQueue(args.root,run,workers,capacities);scheduler.adopt_committed()
        write_json_atomic(args.root/'execution.json',dict(devices=choices,total_cap=cap,slot_capacities=capacities,maximum_conditional_panel_jobs=len(work),
            reused_evaluation_measurements=scheduler.evaluation_reference,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        batch=Batch(Path.cwd(),workers,root=args.root,required_free_MiB=18000,work_event=scheduler.event)
        jobs=[Job(f'worker_{i:02d}',['control-space','worker','--worker-id',f'worker_{i:02d}','--slots','16'],eligible_devices=(device,),estimate_seconds=6000,
            required_free_MiB={24:40000,16:28000,8:18000}[capacities[f'worker_{i:02d}']]) for i,device in enumerate(workers)]
        failure=None
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures=[pool.submit(execute_worker,batch,j,d,scheduler) for j,d in zip(jobs,workers)]
            for future in as_completed(futures):
                try:future.result()
                except BaseException as error:
                    failure=repr(error)
                    with scheduler.lock:scheduler.failed=True;scheduler.dispatch()
        if failure:raise RuntimeError(failure)
        if not scheduler.terminal():raise RuntimeError('all400 conditions must finish one final and the complete raw-H audit')
        with closing(sqlite3.connect(queue_path(args.root))) as db,db:db.execute("UPDATE jobs SET status='not_triggered' WHERE status='pending'")
        return analyze(args.root)


def main():
    from .comparison import ROOT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['owner','worker']);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--devices',nargs='+');parser.add_argument('--worker-id');parser.add_argument('--slots',type=int,default=24)
    parser.add_argument('--physical-gpu',type=int,default=0);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip() or subprocess.check_output(['git','branch','--show-current'],text=True).strip():raise ValueError('formal compilation requires clean pushed detached source')
    subprocess.run(['git','merge-base','--is-ancestor','HEAD','origin/main'],check=True)
    if args.command=='worker':
        from .readout import worker
        worker(args)
    else:owner(args)

if __name__=='__main__':main()
