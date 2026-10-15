"""One feature/train/readout owner, useful physical comparisons, one real exit."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,datetime,fcntl,json,statistics,subprocess,threading,time
from ember.pi05_source_checkpoint import write_json_atomic,read_json
from ember.proposal_writer.batch import Batch,Job,live_gpu_snapshot
from .contract import ROOT,SCHEMA,CHECKPOINTS


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


class FeatureState:
    def __init__(self,root,entries):
        self.condition=threading.Condition();self.failures=[]
        self.ready={e['key'] for e in entries if Path(e['cache_path']).is_file()}
        self.root=root

    def event(self,worker,value,stream):
        with self.condition:
            if value['kind']=='feature_complete':self.ready.add(value['key'])
            elif value['kind'].endswith('_observation'):
                print('EMBER_USEFUL_OBSERVATION '+json.dumps(dict(worker=worker,**value)),flush=True)
            self.condition.notify_all()

    def completed(self,future):
        with self.condition:
            error=future.exception()
            if error:self.failures.append(repr(error))
            self.condition.notify_all()

    def require(self,keys):
        with self.condition:
            while not set(keys)<=self.ready:
                if self.failures:raise RuntimeError('; '.join(self.failures))
                self.condition.wait()


def device_pool(args,root):
    folder=root/'admission';folder.mkdir(exist_ok=True)
    snapshot=live_gpu_snapshot(folder);rows=[(node,g) for node,value in snapshot['node_results'].items() for g in value['gpus']]
    allowed=set(args.devices)
    eligible=lambda n,g,minimum:f'{n}:{g["index"]}' in allowed and g['compute_mode']!='Prohibited' and g['mem_free_mib']>=minimum
    train=sorted([(n,g) for n,g in rows if eligible(n,g,32000)],key=lambda r:(r[1]['util_gpu_pct'],-r[1]['mem_free_mib']))
    nodes={n for n,g in train};node=max(nodes,key=lambda n:sum(v==n for v,_ in train)) if nodes else None
    train=[(n,g['index']) for n,g in train if n==node][:6]
    other=sorted([(n,g) for n,g in rows if n!=node and eligible(n,g,26000)],key=lambda r:(r[1]['util_gpu_pct'],-r[1]['mem_free_mib']))
    readout=[(n,g['index']) for n,g in other[:2]]
    if len(train)<1 or len(readout)<1:raise RuntimeError('two-node useful source/train/readout headroom unavailable')
    return train,readout


def required_features(root,start,stop):
    from .data import Stream
    from .checkpoint import path
    state=None
    if start:
        import torch
        state=torch.load(path(root,start),map_location='cpu',weights_only=False,mmap=True)['stream']
    stream=Stream(root,state);keys=set()
    for _ in range(start,stop):
        for e in stream.next():keys.update((e['original']['key'],e['auxiliary']['key']))
    return keys


def cpu_admit(root,node,ranks):
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',node,'python3','-c',
        "import json;from pathlib import Path;m={l.split(':')[0]:int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines()};print(json.dumps(m))"]
    # ssh concatenates argv; quote the single Python program explicitly.
    import shlex
    command=command[:6]+[shlex.join(command[6:])]
    record=json.loads(subprocess.check_output(command,text=True))
    record.update(node=node,ranks=ranks,estimated_added_GiB=ranks*24,UTC=utc())
    folder=root/'admission';folder.mkdir(exist_ok=True)
    write_json_atomic(folder/f'CPU_{node}_{time.time_ns()}.json',record)
    if record['MemAvailable']/2**20<64+24*ranks:raise RuntimeError('node CPU cannot admit source replicas plus64GiB reserve')


def training_segment(args,batch,pool,start,stop):
    from .checkpoint import path
    if path(args.root,stop).is_file():
        receipt=args.root/'training'/f'segment_{start:08d}_{stop:08d}.json'
        if receipt.is_file():return read_json(receipt)
        import torch
        saved=torch.load(path(args.root,stop),map_location='cpu',weights_only=False,mmap=True)
        measures=[json.loads(line) for line in (args.root/'training/updates.jsonl').read_text().splitlines() if start<json.loads(line)['update']<=stop]
        return dict(start=start,stop=stop,world_size=saved['topology']['world_size'],measurements=measures,job_wall_seconds=None,reused_complete_checkpoint=True)
    cpu_admit(args.root,pool[0][0],len(pool))
    job=Job(f'train_{start:08d}_{stop:08d}_w{len(pool)}',
        ['control-space','train','--start',str(start),'--stop',str(stop)],gpu_count=len(pool),
        estimate_seconds=max(90,(stop-start)*12/len(pool)),required_free_MiB=32000)
    batch.execute(job,pool if len(pool)>1 else pool[0])
    result=read_json(args.root/'training'/f'segment_{start:08d}_{stop:08d}.json')
    print('EMBER_TRAIN_SEGMENT '+json.dumps({k:result[k] for k in ('start','stop','world_size','job_wall_seconds')}),flush=True)
    return result


def feature_jobs(args,batch,devices,state,executor):
    futures=[]
    if (args.root/'features/completion.json').is_file() and all(Path(e['cache_path']).is_file() for e in __import__('ember.control_space.data',fromlist=['feature_entries']).feature_entries(args.root)):
        return futures
    for i,device in enumerate(devices):
        cpu_admit(args.root,device[0],sum(d[0]==device[0] for d in devices))
        job=Job(f'features_{i:02d}', ['control-space','features','--feature-shard',str(i),'--feature-shards',str(len(devices))],
            estimate_seconds=2400,required_free_MiB=18000)
        future=executor.submit(batch.execute,job,device);future.add_done_callback(state.completed);futures.append(future)
    return futures


def learn_and_readout(args,batch,train_pool,readout_pool,features,state,executor):
    from .readout import owner as evaluate
    from .checkpoint import path,latest
    start=latest(args.root)
    segments=[read_json(p) for p in sorted((args.root/'training').glob('segment_*.json')) if read_json(p).get('complete')]
    while start<900 and (start==0 or not all(f.done() for f in features)):
        stop=9 if start==0 else min(900,((start//90)+1)*90);state.require(required_features(args.root,start,stop))
        segments.append(training_segment(args,batch,train_pool[-1:],start,stop));start=stop
    for future in features:future.result()
    write_json_atomic(args.root/'features/completion.json',dict(complete=True,UTC=utc(),videos=len(state.ready)))
    # Entire36-task cycles provide useful world-size comparisons without extra labels.
    for size in (2,4,6):
        if size>len(train_pool) or start>=900 or any(r['world_size']==size for r in segments):continue
        result=training_segment(args,batch,train_pool[:size],start,start+9);segments.append(result);start+=9
    observed={r['world_size']:statistics.median(m['seconds'] for m in r['measurements']) for r in segments}
    size=min(observed,key=observed.get) if observed else 1
    choices=[w for w in observed if w<=min(4,len(train_pool))]
    parallel_size=min(choices,key=lambda w:observed[w]) if choices else 1
    write_json_atomic(args.root/'training/topology_choice.json',dict(observed_seconds_per_update=observed,selected_parallel_training_ranks=parallel_size,selected_before900_ranks=size,
        pure_training_fastest=min(observed,key=observed.get) if observed else size,
        reason='fastest useful ranks before900; after900 allocate fastest<=4 alongside two ready readout workers',
        comparison_is_useful_work_not_paired_profile=True,UTC=utc()))
    if start<900:segments.append(training_segment(args,batch,train_pool[:size],start,900))
    ready=threading.Event()
    eval900=executor.submit(evaluate,replace_args(args,update=900,devices=[f'{n}:{g}' for n,g in readout_pool]))
    eval900.add_done_callback(lambda future:ready.set())
    if not path(args.root,1800).is_file():segments.append(training_segment(args,batch,train_pool[:parallel_size],900,1800))
    immediate=train_pool[:min(4,len(train_pool))]
    if eval900.done():
        eval900.result();delayed=[];immediate+=readout_pool
    else:delayed=[(d,ready) for d in readout_pool]
    eval1800=executor.submit(evaluate,replace_args(args,update=1800,devices=[f'{n}:{g}' for n,g in immediate]),delayed_devices=delayed)
    eval900.result();eval1800.result()
    return segments


def replace_args(args,**values):return argparse.Namespace(**{**vars(args),**values})


def owner(args):
    from .data import prepare,feature_entries
    from .training import label_codes
    from .analysis import analyze
    args.root.mkdir(parents=True,exist_ok=True)
    lock=(args.root/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (args.root/'completion.json').exists():raise ValueError('completed video batch must not restart')
    prepare(args.root);label_codes(args.root);entries=feature_entries(args.root)
    train_pool,readout_pool=device_pool(args,args.root)
    # Five useful feature replicas plus one ready train replica respect global6.
    feature_devices=train_pool[:min(4,max(0,len(train_pool)-1))]+readout_pool[:1]
    state=FeatureState(args.root,entries)
    batch=Batch(Path.cwd(),train_pool+readout_pool,root=args.root,required_free_MiB=18000,work_event=state.event)
    with ThreadPoolExecutor(max_workers=10) as executor:
        features=feature_jobs(args,batch,feature_devices,state,executor)
        learn_and_readout(args,batch,train_pool,readout_pool,features,state,executor)
    result=analyze(args.root)
    write_json_atomic(args.root/'owner_result.json',dict(complete=True,UTC=utc(),schema=SCHEMA))
    lock.close();return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['owner','features','train','readout-worker'])
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--devices',nargs='+')
    parser.add_argument('--worker-id');parser.add_argument('--slots',type=int,default=16)
    parser.add_argument('--physical-gpu',type=int,default=0);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int,default=1800)
    parser.add_argument('--update',type=int,choices=CHECKPOINTS,default=900)
    parser.add_argument('--frame-chunk',type=int,default=0);parser.add_argument('--readout-required-free-MiB',type=int,default=26000)
    parser.add_argument('--feature-shard',type=int,default=0);parser.add_argument('--feature-shards',type=int,default=1)
    args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip() or subprocess.check_output(['git','branch','--show-current'],text=True).strip():
        raise ValueError('formal features/train/eval requires clean detached pushed source')
    subprocess.run(['git','merge-base','--is-ancestor','HEAD','origin/main'],check=True)
    if args.command=='readout-worker':
        from .readout import worker
        worker(args)
    elif args.command=='features':
        from .features import worker
        worker(args)
    elif args.command=='train':
        from .training import train
        train(args)
    else:owner(args)


if __name__=='__main__':main()
