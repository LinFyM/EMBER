"""One source-weight compiler owner: bounded M, fresh1180 learning and fixed976."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from itertools import combinations
from dataclasses import replace
import argparse,datetime,fcntl,json,os,shlex,subprocess,time
import torch
from ember.pi05_source_checkpoint import write_json_atomic,read_json
from ember.proposal_writer.batch import Batch,Job,DeviceWorkloadFailure,live_gpu_snapshot,quota_headroom
from ember.reference.coordinator import Scheduler
from .contract import ROOT,REFERENCE_ROOT,LIMITS

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def queue_path(root):return Path(root)/'evaluation.sqlite'


def storage_admission(root):
    quota=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','strg01',"xfs_quota -c 'quota -u -b ymdai' /data1"],text=True)
    used,limit=quota_headroom(quota,peak_GiB=LIMITS['added_peak_GiB'])
    shared=os.statvfs('/data1');free=shared.f_bavail*shared.f_frsize
    if free<LIMITS['added_peak_GiB']*2**30:raise RuntimeError('shared data1 free capacity cannot cover16GiB peak')
    relevant=subprocess.check_output(['du','-sk',str(REFERENCE_ROOT.parent)],text=True)
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
    batch=Batch(Path.cwd(),devices,root=args.root,required_free_MiB=minimum)
    job=Job(identity,['control-space',*command],gpu_count=len(devices),estimate_seconds=estimate,required_free_MiB=minimum)
    batch.execute(job,devices);return len(devices)


def memory_worker(args):
    from ember.pi05_source_setup import initialize_distributed
    from .video import MemoryStore
    context=initialize_distributed(require_numa=True,defer_process_group=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    store=MemoryStore(args.root,context.device);start=time.monotonic();measurements=[]
    try:
        for entry in store.training_entries[context.rank::context.world_size]:
            store.training(entry['task'],entry['demo']);measurements.extend(store.drain_measurements())
        torch.cuda.synchronize(context.device)
        write_json_atomic(args.root/'memory'/f'rank_{context.rank}.json',dict(rank=context.rank,world=context.world_size,
            entries=len(measurements),wall_seconds=time.monotonic()-start,measurements=measurements,
            peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
            cache=store.cache.usage(),NCCL=False,closed_UTC=utc()))
    finally:store.close()


def finish_memory(root,world):
    records=[read_json(root/'memory'/f'rank_{r}.json') for r in range(world)]
    rows=[row for record in records for row in record['measurements']]
    if (len(rows)!=1800 or len({r['key'] for r in rows})!=1800 or sum(r['frames'] for r in rows)!=59114
            or any(r['identity']['held'] for r in rows)):raise ValueError('bounded M preparation changed original-only1800/59114')
    value=dict(complete=True,videos=1800,frames=59114,world=world,wall_seconds=max(r['wall_seconds'] for r in records),
        M_builds=sum(r['M_builds'] for r in rows),M_hits=sum(r['M_hits'] for r in rows),
        prefix_frames_reread=sum(r['prefix_frames_reread'] for r in rows),producer_videos_read=0,high_dimensional_prefix_copied=False)
    write_json_atomic(root/'memory/completion.json',value);print('EMBER_MEMORY_OBSERVATION '+json.dumps(value),flush=True)


def select_profile(args):
    records=[read_json(p) for p in sorted((args.root/'profiles').glob('world_*.json'))]
    if not records or any(not r.get('complete') for r in records):raise RuntimeError('incomplete disposable profile requires reconciliation, not repeated equivalents')
    totals={key:sum(r['actual_'+key] for r in records) for key in ('BA_equivalents','joint_equivalents','FM_queries')}
    if totals['BA_equivalents']>8 or totals['joint_equivalents']>4 or totals['FM_queries']>448:raise ValueError('disposable profile budget exceeded')
    result={}
    for stage in ('BA','joint'):
        candidates=[row for record in records for row in record['results'] if row['stage']==stage and not row['OOM']]
        if not candidates:raise RuntimeError('no finite useful physical execution for '+stage)
        chosen=min(candidates,key=lambda r:r['seconds'])
        result[stage]={k:chosen[k] for k in ('world_size','task_pack','row_chunk','microbatch','checkpointed','seconds','peak_allocated_MiB','peak_reserved_MiB','LoRAs_per_second','queries_per_second')}
    result['profile_budget_used']=totals
    result['selection_basis']='complete useful update wall including all full factors/native queries, not loss or lowest VRAM'
    write_json_atomic(args.root/'physical_selection.json',result);return result


def run_segment(args,start,stop,choice):
    identity=f'train_{start:08d}_{stop:08d}'
    command=['train','--start',str(start),'--stop',str(stop),'--task-pack',str(choice['task_pack']),
        '--row-chunk',str(choice['row_chunk']),'--microbatch',str(choice['microbatch'])]
    if not choice['checkpointed']:command+=['--no-checkpoint']
    launch(args,identity,command,choice['world_size'],estimate=max(60,(stop-start)*choice['seconds']),minimum=max(12000,int(choice['peak_allocated_MiB'])+1024))


def owner(args):
    from .data import prepare
    from .weights import prepare_source_inputs
    from .model import SourceWeightCompiler
    from .learning import prepare_operator
    from .checkpoint import latest
    from .readout import owner as readout_owner
    from .comparison import analyze
    storage_admission(args.root)
    with (args.root/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (args.root/'completion.json').exists():raise ValueError('completed finite source-weight batch cannot restart')
        prepare(args.root)
        if host_available_GiB()<8:raise RuntimeError('source preparation needs8GiB actual CPU headroom')
        identity=f'owner_{time.time_ns()}';record=dict(received_observed_UTC='2026-10-16T21:56:50.026590261Z',start_UTC=utc(),PID=os.getpid(),
            code=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),workspace=str(Path.cwd()),command=os.sys.argv,
            estimate_hours=[4,7],family_first_reference_UTC='2026-10-15T09:05:48.038Z',new_root_is_not_zero_dependency_cost=True)
        write_json_atomic(args.root/'owners'/f'{identity}.json',record)
        try:
            inputs=prepare_source_inputs(args.root);model=SourceWeightCompiler(inputs)
            prepare_operator(args.root,inputs,model.lora)
            parameters=sum(p.numel() for p in model.parameters());source_bytes=sum(t['W'].numel()*4 for t in inputs['targets'])
            write_json_atomic(args.root/'admission/implementation_peak.json',dict(trainable_parameters=parameters,
                source_W_bytes=source_bytes,generator_FP32_bytes=parameters*4,Adam_moments_bytes=parameters*8,
                final_LoRA_bytes=976*643584*4,M_global_limit=2**30,forecast_peak_GiB=16,
                rationale='three full permanent CP plus rolling/atomic, global M, actual W, 976 full LoRA and metadata/headroom'))
            del model,inputs
            if not (args.root/'memory/completion.json').exists():
                world=launch(args,'memory_original1800',['memory'],6,estimate=900,minimum=4000);finish_memory(args.root,world)
            if not (args.root/'physical_selection.json').exists():
                existing=list((args.root/'profiles').glob('world_*.json'))
                if existing or (args.root/'profiles/campaign.json').exists():raise RuntimeError('partial profile campaign must reconcile before any new disposable query')
                write_json_atomic(args.root/'profiles/campaign.json',dict(start_UTC=utc(),single_disposable_campaign=True,max_BA_equivalents=8,max_joint_equivalents=4,max_FM_queries=448))
                world=launch(args,'profile_world1',['profile'],1,estimate=900,exact=True)
                available=learning_devices(args,6,label='profile_maximum')
                if len(available)>1:launch(args,f'profile_world{len(available)}',['profile'],len(available),estimate=900,exact=True)
                select_profile(args)
            selection=read_json(args.root/'physical_selection.json');current=latest(args.root)
            if current<1000:run_segment(args,current,1000,selection['BA']);current=1000
            if current<1180:run_segment(args,current,1180,selection['joint'])
            readout_owner(args)
            result=analyze(args.root)
            record.update(exit_code=0,end_UTC=utc());write_json_atomic(args.root/'owners'/f'{identity}.json',record)
            return result
        except BaseException as error:
            record.update(exit_code=1,end_UTC=utc(),error=repr(error));write_json_atomic(args.root/'owners'/f'{identity}.json',record);raise


def host_available_GiB():
    return next(int(l.split()[1])/2**20 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['owner','memory','profile','train','worker'])
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--devices',nargs='+')
    parser.add_argument('--worker-id');parser.add_argument('--slots',type=int,default=16)
    parser.add_argument('--physical-gpu',type=int,default=0);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int,default=1180)
    parser.add_argument('--task-pack',type=int,default=1);parser.add_argument('--row-chunk',type=int,default=512)
    parser.add_argument('--microbatch',type=int,default=7);parser.add_argument('--no-checkpoint',action='store_true')
    parser.add_argument('--frame-chunk',type=int,default=32);parser.add_argument('--materialize-pack',type=int,default=4)
    args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip() or subprocess.check_output(['git','branch','--show-current'],text=True).strip():raise ValueError('formal compiler requires clean pushed detached source')
    subprocess.run(['git','merge-base','--is-ancestor','HEAD','origin/main'],check=True)
    if args.command=='worker':
        from .readout import worker
        worker(args)
    elif args.command=='memory':memory_worker(args)
    elif args.command=='profile':
        from .profile import run
        run(args)
    elif args.command=='train':
        from .training import train
        train(args)
    else:
        if not args.devices:args.devices=[f'{node}:{index}' for node in ('gpu01','gpu02') for index in range(8)]
        owner(args)


if __name__=='__main__':main()
