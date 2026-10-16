"""One owner for fresh functional H learning and fixed terminal compilation."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import closing
import argparse,datetime,fcntl,json,os,shlex,sqlite3,subprocess,time
import torch
from ember.pi05_source_checkpoint import write_json_atomic,read_json
from ember.pi05_eval_queue import EvaluationShard,initialize_queue,claim_next,complete_job,fail_job
from ember.proposal_writer.batch import Batch,Job,live_gpu_snapshot,quota_headroom
from ember.reference.coordinator import Scheduler
from .contract import REFERENCE_ROOT,BASIS,INITIAL_READER
from .practice import specifications
from .compilation import Compilation,maximum_panels
from .model import load_basis
from .data import feature_entries


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def queue_path(root):return Path(root)/'evaluation.sqlite'


def storage_admission(root):
    quota=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','strg01',"xfs_quota -c 'quota -u -b ymdai' /data1"],text=True)
    used,limit=quota_headroom(quota,peak_GiB=64)
    shared=os.statvfs('/data1');free=shared.f_bavail*shared.f_frsize
    if free<64*2**30:raise RuntimeError('shared data1 free capacity cannot cover64GiB peak')
    relevant=subprocess.check_output(['du','-sk',str(REFERENCE_ROOT.parent)],text=True)
    root.mkdir(parents=True,exist_ok=True)
    (root/'admission').mkdir(exist_ok=True)
    write_json_atomic(root/'admission'/f'storage_{time.time_ns()}.json',dict(UTC=utc(),quota=quota,used_KiB=used,quota_KiB=limit,
        shared_free_bytes=free,relevant_usage=relevant,additional_peak_GiB=64,new_outputs_filesystem='data1'))


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


def device_pool(args,*,training=False):
    folder=args.root/'admission'/f'pool_{time.time_ns()}';folder.mkdir(parents=True)
    snap=live_gpu_snapshot(folder)
    groups,cap,outside=gpu_candidates(args,snap,getattr(args,'training_free_MiB',36000) if training else 18000)
    resources=cpu_snapshot(args.root,'train' if training else 'compile')
    if not training:
        groups={n:gs[:max(0,int((resources[n]['memory_KiB']['MemAvailable']/2**20-64)//32))] for n,gs in groups.items()}
        groups={n:gs for n,gs in groups.items() if gs}
    selected=[];available=min(6,cap-outside)
    if training and groups:
        node=max(groups,key=lambda n:(min(available,len(groups[n])),sum(g['mem_free_mib'] for g in groups[n][:available])))
        cpu_cap=max(0,int((resources[node]['memory_KiB']['MemAvailable']/2**20-64)//32))
        selected=[(node,g['index'],g['mem_free_mib']) for g in groups[node][:min(available,cpu_cap)]]
        groups={}
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
        nodes=[('practice',i) for i in range(1,maximum_panels(task['horizon']))]+[('query',0)]
        for role,ordinal in nodes:
            identity=c['id']+(f'_practice_{ordinal:03d}' if role=='practice' else '_query')
            states=c['practice_states'] if role=='practice' else c['query_states']
            work[identity]=dict(condition=c['id'],role=role,ordinal=ordinal)
            shards.append(EvaluationShard(identity,len(shards),task['suite'],c['task'],task['horizon'],tuple(states),4*(task['horizon']+10)))
    return work,shards


class WorkQueue(Scheduler):
    """Causal proposals and whole four-case panels share one persistent runtime."""
    def __init__(self,root,run,workers,capacities):
        super().__init__(root,{},len(workers),worker_nodes={f'worker_{i:02d}':n for i,(n,_) in enumerate(workers)},
            queue_path_fn=queue_path,evaluation_reference_root=REFERENCE_ROOT)
        self.run=run;self.capacities=capacities;self.inflight={};self.measures=[]
        self.tasks={s['global_task_id']:s for s in run['environment']['tasks']};self.work,_=panel_queue(root,run,self.tasks)
        decoder=load_basis(BASIS).requires_grad_(False);saved=torch.load(INITIAL_READER,map_location='cpu',mmap=True,weights_only=False)
        units=saved['code_units'].float();del saved
        if units.shape!=(64,) or not bool(units.isfinite().all()) or bool((units<=0).any()):raise ValueError('fixed original code units changed')
        videos={(e['task'],e['demo']):e for e in feature_entries()}
        self.compilers={c['id']:Compilation(root,dict(c,original=videos[c['task'],c['teacher_demo']]),decoder,units,self.tasks[c['task']]) for c in run['conditions']}
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
        # §16 measured16 faster end-to-end than24 with actual H; no32 reprobe.
        preferred=min(16,self.capacities[worker]);budget=self.node_budget(worker)
        others=sum(v for k,v in budget['occupied'].items() if k!=worker)
        for slots in (16,8,4):
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

    def dispatch(self):
        statuses=self.statuses();eligible=[i for i in self.ready if statuses[i]=='pending']
        for worker,stream in list(self.waiting.items()):
            if worker in self.inflight:
                if not self.failed:self.send_claims(worker,stream,self.inflight[worker])
                else:self.send(stream,dict(command='stop'));del self.waiting[worker]
                continue
            unfinished=sum(not c.state['query_complete'] for c in self.compilers.values())
            if self.failed or self.terminal() or len(self.alive)>unfinished:
                self.send(stream,dict(command='stop'));del self.waiting[worker];self.alive.discard(worker);continue
            slots=self.slots(worker);claims=[]
            candidates=list(eligible)
            for _ in range(slots//4):
                claim=claim_next(self.queue_path(),worker_id=worker,eligible_job_ids=candidates) if candidates else None
                if claim is None:break
                claims.append(claim);eligible.remove(claim.shard.job_id)
                kind=self.ready[claim.shard.job_id]['command']
                candidates=[i for i in candidates if i in eligible and self.ready[i]['command']==kind]
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
            elif event['kind']=='complete':self.complete(worker,event)
            elif event['kind']=='proposed':self.proposed(worker,event)
            elif event['kind']=='ready':self.running_slots.pop(worker,None);self.waiting[worker]=stream
            self.dispatch()


def segment_measurement(file,devices,cap):
    import statistics
    segment=read_json(file)
    if not segment['complete']:raise RuntimeError('training child did not complete its segment')
    measure=dict(world_size=segment['world_size'],devices=devices,total_cap=cap,start=segment['start'],stop=segment['stop'],receipt=str(file),
        median_update_seconds=statistics.median(m['seconds'] for m in segment['measurements']),
        limitation='sequential useful cycles have different complete H lengths/cache hits; physical throughput evidence, not a scientific comparison')
    valid=[m for m in segment['measurements'] if not m['OOM_attempts']]
    if valid:
        best=min(valid,key=lambda m:m['seconds'])
        measure['next_live_headroom_MiB']=max(18000,int(best['peak_allocated_MiB'])+1024)
    return measure


def train_trajectory(args):
    from .checkpoint import latest,path
    if (args.root/'training_completion.json').is_file():
        receipt=read_json(args.root/'training_completion.json')
        if not receipt['complete'] or receipt['next_update']!=180:raise ValueError('incompatible terminal training receipt')
        return
    completed=latest(args.root);measure_file=args.root/'training/topology_comparison.json'
    measures=read_json(measure_file) if measure_file.exists() else []
    while completed<180:
        choices,cap=device_pool(args,training=True);available=len(choices)
        # Every comparison advances the one valid trajectory at a full9-update boundary.
        pending=[n for n in (min(4,available),min(2,available),min(6,available)) if n not in {m['world_size'] for m in measures}]
        if pending:
            world=pending[0];stop=min(180,completed+9)
        else:
            eligible=[m for m in measures if m['world_size']<=available]
            if not eligible:raise RuntimeError('no measured train topology remains live')
            world=min(eligible,key=lambda m:m['median_update_seconds'])['world_size'];stop=180
        selected=[(n,g) for n,g,_ in choices[:world]]
        if not selected:raise RuntimeError('no same-node useful training allocation')
        def observation(identity,event,stream):
            if event['kind']!='training_observation':raise ValueError('unexpected training-owner event')
            print('EMBER_TRAIN_OBSERVATION '+json.dumps({k:event[k] for k in ('update','world_size','seconds','queries_per_second',
                'microbatch','actual_max_FM_batch','event_chunk','image_chunk','peak_allocated_MiB','peak_reserved_MiB','FM','head_gradient_norm','body_gradient_norm')}),flush=True)
        required=getattr(args,'training_free_MiB',36000)
        batch=Batch(Path.cwd(),selected,root=args.root,required_free_MiB=required,work_event=observation)
        job=Job(f'train_{completed:08d}_{stop:08d}', ['control-space','train','--start',str(completed),'--stop',str(stop)],
            eligible_devices=tuple(selected),gpu_count=world,estimate_seconds=max(600,(stop-completed)*60),required_free_MiB=required)
        batch.execute(job,selected)
        segment_file=args.root/'training'/f'segment_{completed:08d}_{stop:08d}.json';segment=read_json(segment_file)
        if not segment['complete'] or segment['stop']!=stop or latest(args.root)!=stop:raise RuntimeError('training child did not save its actual complete boundary')
        measure=segment_measurement(segment_file,selected,cap)
        args.training_free_MiB=measure.get('next_live_headroom_MiB',required)
        measures.append(measure);write_json_atomic(measure_file,measures);completed=stop
    saved=torch.load(path(args.root,180),map_location='cpu',weights_only=False,mmap=True)
    stream=saved['stream']
    if stream['next_update']!=180 or any(v!=20 for v in stream['new_visits'].values()):raise ValueError('terminal stream must contain20 visits/task')
    segments=[str(p) for p in sorted((args.root/'training').glob('segment_*.json'))]
    write_json_atomic(args.root/'training_completion.json',dict(complete=True,next_update=180,checkpoint=str(path(args.root,180)),stream=stream,
        cumulative_training_seconds=saved['training_seconds'],segments=segments,FM_queries=20160,history_conditions=720,end_UTC=utc()))


def owner(args):
    from .comparison import prepare,analyze
    storage_admission(args.root);torch.set_num_threads(8)
    with (args.root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.root/'completion.json').exists():raise ValueError('completed finite compilation cannot restart')
        for name in ('shards','packs','proposals','admission'):(args.root/name).mkdir(exist_ok=True)
        run=prepare(args.root);train_trajectory(args)
        tasks={s['global_task_id']:s for s in run['environment']['tasks']};work,shards=panel_queue(args.root,run,tasks)
        initialize_queue(queue_path(args.root),shards,contract_reference=str(args.root/'run_contract.json'),recover_claims=args.resume,retry_failed=args.resume)
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
            futures=[pool.submit(batch.execute,j,d) for j,d in zip(jobs,workers)]
            for future in as_completed(futures):
                try:future.result()
                except BaseException as error:
                    failure=repr(error)
                    with scheduler.lock:scheduler.failed=True;scheduler.dispatch()
        if failure:raise RuntimeError(failure)
        if not scheduler.terminal():raise RuntimeError('all36 conditions must finish, including each independent query')
        with closing(sqlite3.connect(queue_path(args.root))) as db,db:db.execute("UPDATE jobs SET status='not_triggered' WHERE status='pending'")
        return analyze(args.root)


def main():
    from .comparison import ROOT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['owner','worker','train']);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--devices',nargs='+');parser.add_argument('--worker-id');parser.add_argument('--slots',type=int,default=24)
    parser.add_argument('--physical-gpu',type=int,default=0);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int,default=180);args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip() or subprocess.check_output(['git','branch','--show-current'],text=True).strip():raise ValueError('formal compilation requires clean pushed detached source')
    subprocess.run(['git','merge-base','--is-ancestor','HEAD','origin/main'],check=True)
    if args.command=='train':
        from .training import train
        train(args)
    elif args.command=='worker':
        from .readout import worker
        worker(args)
    else:owner(args)

if __name__=='__main__':main()
