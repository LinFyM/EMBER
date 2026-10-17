"""One fresh native-operator learning owner and the registered144 final queries."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from itertools import combinations
from dataclasses import replace
import argparse,datetime,fcntl,json,os,shlex,subprocess,time
import torch
from ember.pi05_source_checkpoint import write_json_atomic,read_json
from ember.proposal_writer.batch import Batch,Job,DeviceWorkloadFailure,live_gpu_snapshot,quota_headroom
from ember.reference.coordinator import Scheduler
from .contract import ROOT,REFERENCE_ROOT,LIMITS,PREVIOUS_ROOT,H_ROOT,FINAL,checkpoint_path

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def queue_path(root):return Path(root)/'evaluation.sqlite'


def storage_admission(root):
    quota=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','strg01',"xfs_quota -c 'quota -u -b ymdai' /data1"],text=True)
    used,limit=quota_headroom(quota,peak_GiB=LIMITS['added_peak_GiB'])
    shared=os.statvfs('/data1');free=shared.f_bavail*shared.f_frsize
    if free<LIMITS['added_peak_GiB']*2**30:raise RuntimeError('shared data1 free capacity cannot cover the registered peak')
    relevant=subprocess.check_output(['du','-sk',str(PREVIOUS_ROOT),str(H_ROOT),*([str(root)] if root.exists() else [])],text=True)
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


def learning_devices(args,world,*,minimum=12000,label='learning'):
    folder=args.root/'admission'/f'{label}_{time.time_ns()}';folder.mkdir(parents=True)
    snapshot=live_gpu_snapshot(folder);groups,cap,outside=gpu_candidates(args,snapshot,minimum)
    resources=cpu_snapshot(args.root,label);available=max(0,cap-outside)
    plans=[]
    for node,gpus in groups.items():
        cpu_count=max(0,int((resources[node]['memory_KiB']['MemAvailable']/2**20-48)//12))
        count=min(world,6,available,cpu_count,len(gpus))
        if count:plans.append((count,sum(g['mem_free_mib'] for g in gpus[:count]),node,gpus[:count]))
    if not plans:raise RuntimeError('no live single-node device/CPU allocation for useful compiler work')
    count,_,node,gpus=max(plans)
    devices=[(node,g['index']) for g in gpus]
    write_json_atomic(folder/'allocation.json',dict(UTC=utc(),devices=devices,requested_world=world,actual_world=count,
        minimum_free_MiB=minimum,cap=cap,outside_project_devices=outside,CPU=resources[node],
        CPU_forecast_GiB_per_rank=12,CPU_reserve_GiB=48,reason='useful single-node ranks; physical profile decides formal throughput'))
    return devices


def launch(args,identity,command,world,*,estimate=600,minimum=12000,exact=False):
    devices=learning_devices(args,world,minimum=minimum,label=identity)
    if exact and len(devices)!=world:raise RuntimeError('profile topology changed before launch; no repeated campaign permitted')
    def event(job,value,stream):
        receipt=dict(UTC=utc(),job=job,**value)
        with (args.root/'work_events.jsonl').open('a') as handle:handle.write(json.dumps(receipt)+'\n')
        print('EMBER_OWNER_EVENT '+json.dumps(receipt),flush=True)
    batch=Batch(Path.cwd(),devices,root=args.root,required_free_MiB=minimum,work_event=event)
    job=Job(identity,['control-space',*command],gpu_count=len(devices),estimate_seconds=estimate,required_free_MiB=minimum)
    batch.execute(job,devices);return len(devices)


def select_profile(args):
    path=args.root/'profile/training/updates.jsonl'
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    if [r['update'] for r in rows]!=[1,2,3,4] or sum(r['queries'] for r in rows)!=448:
        raise ValueError('single disposable campaign did not preserve4 updates/448 actual FM queries')
    if any(not r['disposable'] for r in rows):raise ValueError('profile trajectory cannot become formal')
    # The first update includes source-H vision and actual nonlinear consumers.
    # The last three are complete useful updates with frozen source F0 retained.
    candidates=rows[1:]
    chosen=min(candidates,key=lambda r:r['seconds'])
    result={k:chosen[k] for k in ('world_size','support_microbatch','FM_microbatch','seconds',
        'compilers_per_second','queries_per_second','peak_allocated_MiB','peak_reserved_MiB')}
    result.update(disposable_updates=4,disposable_FM_queries=448,temporary_parameters_adopted=False,
        selection_basis='whole useful compiler/native update throughput; first cold/nonlinear consumer reported separately',
        all_measurements=rows)
    write_json_atomic(args.root/'physical_selection.json',result)
    return result


def profile_campaign(args):
    if (args.root/'physical_selection.json').exists():return read_json(args.root/'physical_selection.json')
    profile=args.root/'profile'
    if (args.root/'profiles/campaign.json').exists():
        raise RuntimeError('partial consumer campaign requires exact receipt reconciliation, never a repeated profile')
    write_json_atomic(args.root/'profiles/campaign.json',dict(start_UTC=utc(),max_updates=4,max_FM_queries=448,
        tasks=[0,13,20,32],temporary_parameters_discarded_from_formal=True,new_environment_episodes=0))
    launch(args,'profile_first3',['profile','--root',str(profile),'--start','0','--stop','3'],4,estimate=600,minimum=22000)
    available=learning_devices(args,6,minimum=22000,label='profile_last_available')
    launch(args,'profile_last1',['profile','--root',str(profile),'--start','3','--stop','4'],len(available),
        estimate=300,minimum=22000,exact=True)
    return select_profile(args)


def owner(args):
    from .data import prepare
    from .model import NativeOperatorContext
    from .checkpoint import latest
    from .readout import owner as readout_owner
    from .comparison import analyze
    storage_admission(args.root);torch.set_num_threads(8)
    with (args.root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.root/'completion.json').exists():raise ValueError('completed finite native first-write batch cannot restart')
        prepare(args.root)
        record=dict(first_dated_local_observation_UTC='2026-10-17T02:39:51.303327242Z',start_UTC=utc(),PID=os.getpid(),
            code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),workspace=str(Path.cwd()),command=os.sys.argv,
            estimate_hours=[2.5,5.5],family_first_reference_UTC='2026-10-15T09:05:48.038Z',new_root_is_not_zero_dependency_cost=True)
        identity=f'owner_{time.time_ns()}';write_json_atomic(args.root/'owners'/f'{identity}.json',record)
        try:
            model=NativeOperatorContext();parameters=sum(p.numel() for p in model.parameters());del model
            write_json_atomic(args.root/'admission/implementation_peak.json',dict(trainable_parameters=parameters,
                model_FP32_bytes=parameters*4,Adam_moments_bytes=parameters*8,final_LoRA_bytes=36*643584*4,
                H_global_limit=4*2**30,M_global_limit=2**30,forecast_peak_GiB=12,
                rationale='shared frozen H features,90/180 plus atomic rolling, disposable CP,36 complete LoRA, logs/headroom; no source/H/M copies'))
            from .cache import ReferencedMemoryCache
            shared=ReferencedMemoryCache(args.root/'cache/M',PREVIOUS_ROOT/'cache/M')
            write_json_atomic(args.root/'memory/references.json',dict(shared.usage(),originals=1800,
                original_sampled_frames=59114,held_read=False,new_payload_copy=False,old_SQLite_modified=False,
                historical_source_practice_episodes=144,historical_source_practice_steps=40377,zero_interaction=False))
            choice=profile_campaign(args);current=latest(args.root)
            if current<FINAL:
                launch(args,f'train_{current:08d}_{FINAL:08d}',
                    ['train','--start',str(current),'--stop',str(FINAL),'--support-microbatch',str(choice['support_microbatch']),
                     '--microbatch',str(choice['FM_microbatch'])],choice['world_size'],
                    estimate=max(60,(FINAL-current)*choice['seconds']),minimum=max(18000,int(choice['peak_allocated_MiB'])+1024))
            readout_owner(args);result=analyze(args.root)
            record.update(exit_code=0,end_UTC=utc());write_json_atomic(args.root/'owners'/f'{identity}.json',record)
            return result
        except BaseException as error:
            record.update(exit_code=1,end_UTC=utc(),error=repr(error));write_json_atomic(args.root/'owners'/f'{identity}.json',record);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['owner','profile','train','worker'])
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--devices',nargs='+')
    parser.add_argument('--worker-id');parser.add_argument('--slots',type=int,default=16)
    parser.add_argument('--physical-gpu',type=int,default=0);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int,default=180)
    parser.add_argument('--support-microbatch',type=int,default=4);parser.add_argument('--microbatch',type=int,default=14)
    args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip() or subprocess.check_output(['git','branch','--show-current'],text=True).strip():
        raise ValueError('all native-operator GPU consumers require clean pushed detached source')
    subprocess.run(['git','merge-base','--is-ancestor','HEAD','origin/main'],check=True)
    if args.command=='worker':
        from .readout import worker
        worker(args)
    elif args.command in ('train','profile'):
        from .training import train
        train(args)
    else:
        if not args.devices:args.devices=[f'{node}:{index}' for node in ('gpu01','gpu02') for index in range(8)]
        owner(args)


if __name__=='__main__':main()
