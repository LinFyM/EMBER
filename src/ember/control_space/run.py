"""One first-cycle owner: fixed-theta collection, complete-path update, real exit."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse,datetime,fcntl,json,os,shutil,subprocess,time
from ember.pi05_source_checkpoint import write_json_atomic,read_json
from ember.proposal_writer.batch import Batch,Job,live_gpu_snapshot
from ember.reference.coordinator import Scheduler
from .contract import ROOT,REFERENCE_ROOT,condition_id,condition_root,checkpoint


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def cpu_admit(root,node,replicas,slots):
    import shlex
    code="import json;from pathlib import Path;m={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines()};print(json.dumps(m))"
    result=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',node,shlex.join(['python3','-c',code])],text=True))
    estimate=24*replicas+5*slots+16
    result.update(node=node,replicas=replicas,slots=slots,estimated_added_GiB=estimate,UTC=utc())
    folder=root/'admission';folder.mkdir(exist_ok=True);write_json_atomic(folder/f'CPU_{node}_{time.time_ns()}.json',result)
    if result['MemAvailable']/2**20<64+estimate:raise RuntimeError('actual node CPU cannot admit shared replicas/environments/H cache')


def device_pool(args,*,same_node=False):
    folder=args.root/'admission'/f'pool_{time.time_ns()}';folder.mkdir(parents=True)
    snapshot=live_gpu_snapshot(folder);rows=[(node,g) for node,value in snapshot['node_results'].items() for g in value['gpus']]
    idle=sum(not g['processes'] and g['mem_used_mib']<1000 for _,g in rows);cap=6 if idle<=10 else 8
    outside={(n,g['index']) for n,g in rows if any(p['owner']=='ymdai' for p in g['processes'])}
    eligible=[(n,g) for n,g in rows if f'{n}:{g["index"]}' in args.devices and g['compute_mode']!='Prohibited'
        and g['mem_free_mib']>=22000 and (n,g['index']) not in outside]
    eligible.sort(key=lambda v:(-v[1]['mem_free_mib'],v[1]['util_gpu_pct']))
    if not eligible:raise RuntimeError('no actual shared GPU can fit source plus useful FM/H work')
    if same_node:
        nodes={n for n,_ in eligible};chosen=max(nodes,key=lambda n:sum(node==n for node,_ in eligible))
        eligible=[row for row in eligible if row[0]==chosen]
    available=cap-len(outside);pool=[]
    for n,g in eligible:
        if sum(x==n for x,_ in pool)>=6:continue
        pool.append((n,g['index']))
        if len(pool)>=available:break
    if len(pool)<(1 if same_node else 2):raise RuntimeError('insufficient useful devices for the current actual phase')
    return pool,cap


class WorkQueue(Scheduler):
    """Reuse measured node memory accounting; pending actual panels are dynamic."""
    def __init__(self,root,workers):
        super().__init__(root,{},len(workers),worker_nodes={f'episode_{i:02d}':n for i,(n,_) in enumerate(workers)},evaluation_reference_root=REFERENCE_ROOT)
        self.pending=[];self.panels={};self.inflight={};self.controller=None;self.stopping=False;self.error=None;self.pack=0
        self.first_observation=False;self.archive_workers=2;self.resources_file=root/'worker_resources.jsonl';self.max_pack=max(1,16//len(workers))

    def log(self,worker,value):
        with self.resources_file.open('a') as f:f.write(json.dumps({**value,'UTC':utc(),'worker':worker})+'\n')

    def send(self,stream,message):stream.write(json.dumps(message)+'\n');stream.flush()

    def add(self,work):
        identity=work['identity']
        if identity in self.panels:raise ValueError('one behavior panel request must not be duplicated')
        self.panels[identity]=dict(work=work,rows={})
        for state in work['states']:
            self.pending.append(dict(identity=identity,task=work['task'],state_id=state,noise_root=work['noise_root'],
                parameter_ref=work['parameter_ref'],y=work['y'],role=work['role'],cache_root=work['cache_root'],
                **{k:work[k] for k in ['behavior_model','meta_update','condition_id','panel_ordinal']}))

    def complete_pack(self,worker,value):
        specs=self.inflight.pop(worker)
        if len(specs)!=len(value['rows']):raise ValueError('episode response lost pack rows')
        for spec,row in zip(specs,value['rows'],strict=True):
            if (spec['task'],spec['state_id'])!=(row['global_task_id'],row['init_state_id']):raise ValueError('worker response changed task/state ordering')
            panel=self.panels[spec['identity']];panel['rows'][spec['state_id']]=row
            if len(panel['rows'])==4:
                output=dict(identity=spec['identity'],rows=[panel['rows'][s] for s in panel['work']['states']])
                self.send(self.controller,dict(command='panel_done',output=output))
        resource=value.get('resources',{});peak=resource.get('peak_environment_resources',{})
        if not self.first_observation:
            self.first_observation=True
            print('EMBER_USEFUL_OBSERVATION '+json.dumps(dict(first_actual_pack=True,rows=len(value['rows']),
                seconds=value.get('seconds'),slots=value.get('slots'),
                history_bytes=sum(r.get('history_bytes',0) for r in value['rows']),
                history_primary_raw_bytes=sum(r.get('history_primary_raw_bytes',0) for r in value['rows']),
                peak_allocated_MiB=value.get('peak_allocated_MiB'),peak_environment_PSS_KiB=(peak or {}).get('PSS_KiB'),
                resources=resource)),flush=True)
        if peak:
            self.environment_PIDs[worker]=set(peak.get('environment_PIDs',[]));self.resident_slots[worker]=len(self.environment_PIDs[worker])
            node=self.worker_node[worker];sizes=[p['PSS_KiB']/2**20 for p in peak.get('per_process',[]) if p['PID'] in self.environment_PIDs[worker]]
            if sizes:self.environment_GiB[node]=max(self.environment_GiB.get(node,self.reference_environment_GiB),max(sizes)*1.2)

    def dispatch(self):
        for worker,stream in list(self.waiting.items()):
            if self.failed or self.stopping:
                self.send(stream,dict(command='stop'));del self.waiting[worker];continue
            if not self.pending:
                if worker not in self.idled:self.send(stream,dict(command='idle'));self.idled.add(worker)
                continue
            budget=self.node_budget(worker);size=min(self.max_pack,len(self.pending))
            extra=max(0,size-budget['occupied'][worker])*budget['environment_GiB']
            if budget['available_GiB']-budget['unreflected_GiB']-extra<64:continue
            self.pending.sort(key=lambda row:-self.task_horizon[row['task']])
            specs=self.pending[:size];del self.pending[:size];self.pack+=1
            self.inflight[worker]=specs;self.running_slots[worker]=size;self.idled.discard(worker)
            self.send(stream,dict(command='panel',identity=f'pack_{self.pack:06d}',request_id=self.pack,rows=specs,slots=size,archive_workers=self.archive_workers))
            del self.waiting[worker]

    def event(self,worker,value,stream):
        with self.lock:
            self.log(worker,value);kind=value['kind']
            if worker=='controller':
                if kind=='controller_ready':self.controller=stream
                elif kind=='panel_request':self.add(value['work'])
                elif kind=='panel_requests':
                    for work in value['work']:self.add(work)
                elif kind=='collection_complete':self.stopping=True
            else:
                resource=value.get('resources')
                if kind=='ready' and resource:self.record_resources(worker,resource)
                if kind=='environments_closed':
                    if value['remaining_environment_PIDs']:raise RuntimeError('idle kept actual environment children')
                    self.resident_slots.pop(worker,None);self.environment_PIDs.pop(worker,None)
                elif kind=='complete':self.complete_pack(worker,value)
                elif kind=='ready':
                    self.running_slots.pop(worker,None);self.waiting[worker]=stream
                elif kind=='failed':self.failed=True;self.error=value
            self.dispatch()

    def failed_future(self,future):
        if future.exception():
            with self.lock:
                self.failed=True;self.error=repr(future.exception())
                if self.controller:
                    try:self.send(self.controller,dict(command='stop'))
                    except (BrokenPipeError,OSError):pass
                self.dispatch()


def collect_batch(args,update,pool,worker_count):
    receipt=args.root/'batches'/f'collection_{update:08d}.json'
    if receipt.is_file():
        saved=read_json(receipt)
        if saved['complete'] and 'owner_seconds' in saved:return saved
    controller=pool[0];workers=pool[1:1+worker_count]
    specs=read_json(args.root/'run_contract.json')['environment']['tasks']
    scheduler=WorkQueue(args.root,workers);scheduler.archive_workers=2 if update==0 else 4
    scheduler.task_horizon={s['global_task_id']:s['horizon'] for s in specs}
    for node in {n for n,_ in [controller,*workers]}:
        count=sum(n==node for n,_ in [controller,*workers]);slots=16 if any(n==node for n,_ in workers) else 0
        cpu_admit(args.root,node,count,slots)
    batch=Batch(Path.cwd(),[controller,*workers],root=args.root,required_free_MiB=22000,work_event=scheduler.event)
    jobs=[('controller',Job(f'collect_{update:08d}', ['control-space','collect','--start',str(update)],estimate_seconds=1800),controller)]
    jobs.extend((f'episode_{i:02d}',Job(f'episode_{update:08d}_{i:02d}',
        ['control-space','worker','--worker-id',f'episode_{i:02d}','--slots','16'],estimate_seconds=1800),device) for i,device in enumerate(workers))
    # Batch callback identity differs from stable per-phase worker identity.
    identities={job.identity:name for name,job,_ in jobs}
    batch.work_event=lambda identity,value,stream:scheduler.event(identities[identity],value,stream)
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures=[executor.submit(batch.execute,job,device) for _,job,device in jobs]
        for future in futures:future.add_done_callback(scheduler.failed_future)
        for future in as_completed(futures):future.result()
    if scheduler.failed:raise RuntimeError(str(scheduler.error))
    record=read_json(args.root/'batches'/f'collection_{update:08d}.json')
    record.update(owner_seconds=time.monotonic()-started,episode_workers=len(workers),physical_devices=[controller,*workers])
    write_json_atomic(args.root/'batches'/f'collection_{update:08d}.json',record)
    return record


def train_batch(args,update,pool,ranks):
    nodes={n for n,_ in pool};node=max(nodes,key=lambda n:sum(v==n for v,_ in pool))
    devices=[d for d in pool if d[0]==node][:ranks]
    cpu_admit(args.root,node,len(devices),0)
    batch=Batch(Path.cwd(),devices,root=args.root,required_free_MiB=22000)
    job=Job(f'meta_{update+1:08d}_w{len(devices)}',['control-space','train','--start',str(update),'--stop',str(update+1)],
        gpu_count=len(devices),estimate_seconds=180)
    batch.execute(job,devices if len(devices)>1 else devices[0])
    return read_json(args.root/'training'/f'meta_{update+1:08d}.json')


def owner(args):
    from .data import prepare,Stream
    from .checkpoint import latest,load
    from .training import initialize_fresh
    from .analysis import analyze
    args.root.mkdir(parents=True,exist_ok=True)
    lock=(args.root/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (args.root/'completion.json').exists():raise ValueError('completed current cycle must not restart')
    prepare(args.root);initialize_fresh(args.root);collection=[];learning=[]
    for update in range(latest(args.root),9):
        saved=load(args.root,update);stream=Stream(args.root,saved['stream']);events=stream.next()
        spec={t['global_task_id']:t for t in read_json(args.root/'run_contract.json')['environment']['tasks']}
        events=[dict(e,horizon=spec[e['task']]['horizon']) for e in events]
        destination=args.root/'batches'/f'meta_{update:08d}.json'
        value=dict(events=events,behavior_checkpoint=str(checkpoint(args.root,update)))
        if destination.exists() and read_json(destination)!=value:raise ValueError('fixed on-policy batch changed')
        if not destination.exists():write_json_atomic(destination,value)
        pool,cap=device_pool(args)
        if update<4:worker_count=min((1,2,4,6)[update],len(pool)-1)
        else:
            previous=[read_json(p) for p in (args.root/'batches').glob('collection_*.json')]
            best=min(previous,key=lambda v:v['owner_seconds']/max(1,v.get('environment_steps',1)))
            worker_count=min(best['episode_workers'],len(pool)-1)
        collection.append(collect_batch(args,update,pool,worker_count))
        pool,_=device_pool(args,same_node=True);available=max(sum(n==node for n,_ in pool) for node in {n for n,_ in pool})
        if update<4:ranks=min((1,2,4,6)[update],available)
        else:
            previous=[read_json(p) for p in (args.root/'training').glob('meta_????????.json')]
            ranks=min(min(previous,key=lambda v:v['seconds'])['world_size'],available)
        learning.append(train_batch(args,update,pool,ranks))
        # Only reproducible dense image inputs retire; full actual H stays archived.
        cache=args.root/'cache/H'
        if cache.exists():shutil.rmtree(cache)
        print('EMBER_USEFUL_OBSERVATION '+json.dumps(dict(meta_update=update+1,collection_seconds=collection[-1]['owner_seconds'],
            episode_workers=worker_count,train_ranks=ranks,training_seconds=learning[-1]['seconds'])),flush=True)
    result=analyze(args.root);write_json_atomic(args.root/'owner_result.json',dict(complete=True,UTC=utc()))
    lock.close();return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['owner','collect','train','worker']);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--devices',nargs='+');parser.add_argument('--worker-id');parser.add_argument('--slots',type=int,default=16)
    parser.add_argument('--physical-gpu',type=int,default=0);parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int,default=1)
    args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip() or subprocess.check_output(['git','branch','--show-current'],text=True).strip():
        raise ValueError('formal current work requires a clean pushed detached source')
    subprocess.run(['git','merge-base','--is-ancestor','HEAD','origin/main'],check=True)
    if args.command=='worker':
        from .readout import worker
        worker(args)
    elif args.command=='collect':
        from .compilation import collect
        collect(args)
    elif args.command=='train':
        from .training import train
        train(args)
    else:owner(args)

if __name__=='__main__':main()
