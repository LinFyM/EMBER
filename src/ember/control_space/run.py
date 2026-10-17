"""One read-only1090 compiler owner with bounded M and fixed976 finals."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from itertools import combinations
from dataclasses import replace
import argparse,datetime,fcntl,json,os,shlex,subprocess,time
import torch
from ember.pi05_source_checkpoint import write_json_atomic,read_json
from ember.proposal_writer.batch import Batch,Job,DeviceWorkloadFailure,live_gpu_snapshot,quota_headroom
from ember.reference.coordinator import Scheduler
from .contract import ROOT,REFERENCE_ROOT,LIMITS,SOURCE_INPUTS,OPERATOR_FILE,PREVIOUS_ROOT,TRAIN_ROOT,FINAL,checkpoint_path

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def queue_path(root):return Path(root)/'evaluation.sqlite'


def storage_admission(root):
    quota=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','strg01',"xfs_quota -c 'quota -u -b ymdai' /data1"],text=True)
    used,limit=quota_headroom(quota,peak_GiB=LIMITS['added_peak_GiB'])
    shared=os.statvfs('/data1');free=shared.f_bavail*shared.f_frsize
    if free<LIMITS['added_peak_GiB']*2**30:raise RuntimeError('shared data1 free capacity cannot cover the registered peak')
    relevant=subprocess.check_output(['du','-sk',str(PREVIOUS_ROOT),str(TRAIN_ROOT),*([str(root)] if root.exists() else [])],text=True)
    root.mkdir(parents=True,exist_ok=True)
    (root/'admission').mkdir(exist_ok=True)
    write_json_atomic(root/'admission'/f'storage_{time.time_ns()}.json',dict(UTC=utc(),quota=quota,used_KiB=used,quota_KiB=limit,
        shared_free_bytes=free,relevant_usage=relevant,additional_peak_GiB=LIMITS['added_peak_GiB'],new_outputs_filesystem='data1'))


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



def execute_worker(batch,job,device,scheduler):
    """Only an unlaunched device can fall back from16 to8 once."""
    def launched():
        attempts=sorted((scheduler.root/'jobs'/job.identity).glob('attempt_*'))
        return bool(attempts and (attempts[-1]/'launch.json').exists())
    try:return batch.execute(job,device)
    except DeviceWorkloadFailure:
        if launched():raise
    if scheduler.capacities[job.identity]>8:
        with scheduler.lock:scheduler.capacities[job.identity]=8
        arguments=list(job.arguments);arguments[arguments.index('--slots')+1]='8'
        job=replace(job,arguments=arguments,required_free_MiB=18000)
        try:return batch.execute(job,device)
        except DeviceWorkloadFailure:
            if launched():raise
    with scheduler.lock:
        if job.identity in scheduler.running:raise RuntimeError('unlaunched worker cannot own a case')
        scheduler.worker_node.pop(job.identity,None);scheduler.waiting.pop(job.identity,None)
        write_json_atomic(scheduler.root/'admission'/f'{job.identity}_unavailable_{time.time_ns()}.json',
            dict(no_child_launched=True,no_environment_submission=True,UTC=utc(),device=device))
        scheduler.dispatch()
    return job.identity


def owner(args):
    from .data import prepare
    from .readout import owner as readout_owner
    from .comparison import analyze
    storage_admission(args.root);torch.set_num_threads(8)
    with (args.root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.root/'completion.json').exists():raise ValueError('completed finite source-weight batch cannot restart')
        prepare(args.root)
        if host_available_GiB()<8:raise RuntimeError('source preparation needs8GiB actual CPU headroom')
        identity=f'owner_{time.time_ns()}';record=dict(received_observed_UTC='2026-10-17T01:39:14.985237082Z',start_UTC=utc(),PID=os.getpid(),
            code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),workspace=str(Path.cwd()),command=os.sys.argv,
            estimate_minutes=[35,65],family_first_reference_UTC='2026-10-15T09:05:48.038Z',new_root_is_not_zero_dependency_cost=True)
        write_json_atomic(args.root/'owners'/f'{identity}.json',record)
        try:
            write_json_atomic(args.root/'admission/implementation_peak.json',dict(generator_parameters=4838160,
                source_checkpoint=str(checkpoint_path(args.root,FINAL)),checkpoint_copy=False,optimizer_restored=False,
                final_LoRA_bytes=976*643584*4,M_global_limit=2**30,forecast_peak_GiB=6,
                rationale='976 full LoRAs, metadata/logs and atomic headroom; old CP/W/M are read-only'))
            from .cache import ReferencedMemoryCache
            shared=ReferencedMemoryCache(args.root/'cache/M',PREVIOUS_ROOT/'cache/M')
            write_json_atomic(args.root/'memory/references.json',dict(shared.usage(),
                originals=1800,held=400,original_sampled_frames=59114,held_sampled_frames=14528,
                new_payload_copy=False,old_SQLite_modified=False,raw_inputs_are_paid_dependencies=True))
            readout_owner(args)
            result=analyze(args.root)
            record.update(exit_code=0,end_UTC=utc());write_json_atomic(args.root/'owners'/f'{identity}.json',record)
            return result
        except BaseException as error:
            record.update(exit_code=1,end_UTC=utc(),error=repr(error));write_json_atomic(args.root/'owners'/f'{identity}.json',record);raise


def host_available_GiB():
    return next(int(l.split()[1])/2**20 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['owner','worker'])
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--devices',nargs='+')
    parser.add_argument('--worker-id');parser.add_argument('--slots',type=int,default=16)
    parser.add_argument('--physical-gpu',type=int,default=0);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--row-chunk',type=int,default=4096)
    parser.add_argument('--frame-chunk',type=int,default=32);parser.add_argument('--materialize-pack',type=int,default=4)
    args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip() or subprocess.check_output(['git','branch','--show-current'],text=True).strip():raise ValueError('formal compiler requires clean pushed detached source')
    subprocess.run(['git','merge-base','--is-ancestor','HEAD','origin/main'],check=True)
    if args.command=='worker':
        from .readout import worker
        worker(args)
    else:
        if not args.devices:args.devices=[f'{node}:{index}' for node in ('gpu01','gpu02') for index in range(8)]
        owner(args)


if __name__=='__main__':main()
