"""Event-driven bounded jobs with live shared-device admission and exact receipts."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
import datetime
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import threading
import time

from .contract import ASSET_ROOT, RUN_ROOT, TASKS
from .analysis import billing, teacher_summary, report_summary, freeze_bank_scale

PREFLIGHT = '/data0/user/ymdai/.codex/skills/gpu-preflight/scripts/gpu_preflight.py'
UNMEASURED_TEACHER_SECONDS = 22.5  # Shared-card demo16.84 + measured rec4/keep4 + optimizer margin.


@dataclass
class Job:
    identity: str
    arguments: list
    dependencies: tuple = ()
    estimate_seconds: float = 0
    eligible_devices: tuple = ()
    device_estimates: dict = field(default_factory=dict)
    gpu_count: int = 1
    required_free_MiB: int = 0

    def estimated_seconds(self, device):
        return self.device_estimates.get(device, self.estimate_seconds)


def initial_jobs(microbatch,slots,teacher_devices,*,root=RUN_ROOT,teacher_update_seconds=None):
    """Seed160 immediately enables independent H160, while seed continues480."""
    jobs=[]
    def teacher(task,event,stop,dependency,updates):
        estimates={device:60+updates*seconds for device,seconds in (teacher_update_seconds or {}).items()}
        return Job(f'teacher_{task:04d}_{event}_{stop}',
            ['teacher','--task',str(task),'--event',str(event),'--microbatch',str(microbatch),'--stop',str(stop)],
            (() if dependency is None else (dependency,)),60+updates*UNMEASURED_TEACHER_SECONDS,tuple(teacher_devices),estimates,
            required_free_MiB=18000)
    for task in TASKS:
        seed160=f'teacher_{task:04d}_0_160'
        seed480=f'teacher_{task:04d}_0_480'
        jobs.append(teacher(task,0,160,None,160))
        jobs.append(teacher(task,0,480,seed160,320))
        for event in (1,2,3):
            collect=f'collect_{task:04d}_{event}'
            dependency=seed160 if event==1 else seed480 if event==2 else f'collect_{task:04d}_1'
            jobs.append(Job(collect,['collect','--task',str(task),'--event',str(event),'--slots',str(min(slots,4))],
                (dependency,),600 if task in (32,38) else 360))
            jobs.append(teacher(task,event,480,collect,480))
        for event in range(4):
            jobs.append(Job(f'audit_{task:04d}_{event}',['audit','--task',str(task),'--event',str(event),
                '--slots',str(slots)],(f'teacher_{task:04d}_{event}_480',),600 if task in (32,38) else 360))
    return jobs


def dependency_completion(root, identities):
    """Wait on coordinator receipt writes, without polling training or shared caches."""
    import ctypes
    native=ctypes.CDLL(None,use_errno=True)
    descriptor=native.inotify_init1(os.O_CLOEXEC)
    if descriptor<0:raise OSError(ctypes.get_errno(),'dependency event descriptor')
    try:
        for identity in identities:
            directory=Path(root)/'jobs'/identity;directory.mkdir(parents=True,exist_ok=True)
            if native.inotify_add_watch(descriptor,os.fsencode(directory),0x8|0x80)<0:
                raise OSError(ctypes.get_errno(),'dependency receipt watch')
        while True:
            complete=set()
            for identity in identities:
                path=Path(root)/'jobs'/identity/'latest.json'
                if path.exists():
                    if read_exit(path):raise RuntimeError(f'external dependency failed: {identity}')
                    complete.add(identity)
            if complete:return complete
            os.read(descriptor,65536)
    finally:os.close(descriptor)


def quota_headroom(report):
    numbers=[line.split() for line in report.splitlines() if line.startswith('/dev/')]
    if len(numbers)!=1:
        raise RuntimeError('authoritative data1 quota could not be parsed')
    used,limit=int(numbers[0][1]),int(numbers[0][2])
    if used+64*1024**2>=limit:
        raise RuntimeError('data1 independent quota cannot cover retained peak')
    return used,limit


class Batch:
    """One coordinator owns resource decisions; completion futures release slots."""
    def __init__(self, workspace, devices, *, root=RUN_ROOT, maximum_GPUh=None, required_free_MiB=12000):
        self.workspace, self.devices, self.root = Path(workspace), list(devices), Path(root)
        registered = json.loads((self.root/'run_contract.json').read_text())['limits']['GPU_hours']
        maximum_GPUh = registered if maximum_GPUh is None else maximum_GPUh
        if not math.isfinite(maximum_GPUh) or not 0 < maximum_GPUh <= registered:
            raise ValueError('requested GPUh exceeds the registered cumulative hard budget')
        self.maximum_GPUh, self.lock = maximum_GPUh, threading.Lock()
        self.running = {}
        self.required_free_MiB=required_free_MiB
        self.code = subprocess.check_output(['git','rev-parse','HEAD'],cwd=workspace,text=True).strip()
        if subprocess.check_output(['git','status','--porcelain'],cwd=workspace,text=True).strip() or subprocess.check_output(
                ['git','branch','--show-current'],cwd=workspace,text=True).strip():
            raise ValueError('batch requires the frozen clean detached source')

    def admit(self, job, node, gpu, destination):
        # Serial admission excludes overlapping launch snapshots within this batch.
        with self.lock:
            result = subprocess.run(['python3',PREFLIGHT,'--nodes','gpu01,gpu02','--json'],
                                    check=True,capture_output=True,text=True)
            snapshot = json.loads(result.stdout)
            if snapshot['bad_nodes']:
                raise RuntimeError('live two-node allocation evidence unavailable')
            quota = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','strg01',
                "xfs_quota -c 'quota -u -b ymdai' /data1"],check=True,capture_output=True,text=True)
            (destination/'gpu_preflight.json').write_text(result.stdout)
            (destination/'quota.txt').write_text(quota.stdout)
            devices = [(n,g) for n,r in snapshot['node_results'].items() for g in r.get('gpus',[])]
            idle = sum(not g['processes'] and g['mem_used_mib'] < 1000 for _,g in devices)
            cap = 6 if idle <= 10 else 8
            allocated = set(self.running)
            selected = next(g for n,g in devices if n==node and g['index']==gpu)
            if selected['compute_mode']=='Prohibited' or selected['mem_free_mib'] < (job.required_free_MiB or self.required_free_MiB):
                raise RuntimeError(f'live device headroom below measured workload requirement: {node}:{gpu}')
            if len(allocated) >= cap or sum(n==node for n,_ in allocated) >= 6:
                raise RuntimeError('current physical GPU project cap prevents this launch')
            # Live allocation may include a task-owned process outside this driver's futures.
            outside = {(n,g['index']) for n,g in devices if any(p['owner']=='ymdai' for p in g['processes'])}
            combined = allocated | outside | {(node,gpu)}
            if len(combined) > cap or sum(n==node for n,_ in combined)>6:
                raise RuntimeError('outside project allocation would exceed the shared cap')
            used,limit=quota_headroom(quota.stdout)
            self.reserve_budget(job,node,gpu)
            return dict(node=node,physical_gpu=gpu,uuid=selected['uuid'],free_MiB=selected['mem_free_mib'],
                        utilization=selected['util_gpu_pct'],concurrency_cap=cap,quota_used_KiB=used,quota_KiB=limit)

    def reserve_budget(self,job,node,gpu):
        spent = billing(self.root)['GPU_hours']
        inflight=sum(max(time.time()-r['start'],r['estimate_seconds'])/3600 for r in self.running.values())
        # Other stage coordinators retain their original launch receipts; count their active jobs too.
        owned = {r['job'] for r in self.running.values()}
        for path in (self.root/'jobs').glob('*/attempt_*/launch.json'):
            if (path.parent/'exit.json').exists():continue
            launch = json.loads(path.read_text())
            if launch['job'] in owned:continue
            elapsed = time.time()-datetime.datetime.fromisoformat(launch['start_utc']).timestamp()
            estimate = launch['estimate_seconds']
            command = launch['command']
            if 'teacher' in command and 'budget_GPUh' not in launch:
                stop = int(command[command.index('--stop')+1])
                # Legacy launches predate the complete-update measurement wiring.
                estimate=max(estimate,60+UNMEASURED_TEACHER_SECONDS*stop)
            inflight+=launch.get('physical_GPU_count',1)*max(elapsed,estimate)/3600
        estimate = job.estimated_seconds((node,gpu))
        if not math.isfinite(estimate) or estimate <= 0:
            raise ValueError('GPU job requires a finite positive full-cost estimate')
        if spent + inflight + estimate/3600 > self.maximum_GPUh:
            raise RuntimeError('registered total GPUh exhausted/projected boundary; scientific budget decision required')
        self.running[(node,gpu)] = dict(start=time.time(),job=job.identity,estimate_seconds=estimate)

    def execute(self, job, device):
        devices = device if isinstance(device,list) else [device]
        node,gpu=devices[0]
        if any(n!=node for n,_ in devices):raise ValueError('shared training cannot span GPU nodes')
        job_root=self.root/'jobs'/job.identity;job_root.mkdir(parents=True,exist_ok=True)
        ordinal=len(list(job_root.glob('attempt_*')))+1
        destination=job_root/f'attempt_{ordinal:03d}';destination.mkdir(exist_ok=False)
        try:
            admissions=[]
            for n,g in devices:
                admission_root=destination if len(devices)==1 else destination/f'gpu_{g}'
                admission_root.mkdir(exist_ok=True)
                admissions.append(self.admit(job,n,g,admission_root))
            env=dict(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(self.workspace/'src'),
                CUDA_VISIBLE_DEVICES=','.join(str(g) for _,g in devices),NCCL_P2P_DISABLE='1',
                TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MUJOCO_GL='egl')
            command=[str(ASSET_ROOT/'.venv/bin/python')]
            if len(devices)>1:
                command+=['-m','torch.distributed.run','--standalone',f'--nproc-per-node={len(devices)}']
            arguments=list(job.arguments)
            if arguments[0]=='teacher' and admissions[0]['free_MiB']>=27000:
                arguments[arguments.index('--microbatch')+1]='56'
            if arguments[0] in ('audit','report') and admissions[0]['free_MiB']<25000:
                arguments[arguments.index('--slots')+1]=str(min(16,int(arguments[arguments.index('--slots')+1])))
            command+=[str(self.workspace/'scripts/video_guided_proposal_writer.py'),*arguments,
                      '--root',str(self.root),'--physical-gpu',str(gpu)]
            return self.launch(job,devices,destination,command,env,admissions)
        finally:
            with self.lock:
                for d in devices:self.running.pop(d,None)

    def launch(self,job,devices,destination,command,env,admissions):
        node=devices[0][0]
        start=time.time();receipt=dict(job=job.identity,command=command,env=env,workspace=str(self.workspace),
            code_commit=self.code,admission=admissions,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            physical_GPU_count=len(devices),estimate_seconds=job.estimated_seconds(devices[0]),budget_GPUh=self.maximum_GPUh)
        (destination/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
        local=socket.gethostname().split('.')[0].lower() in {node,'bci-'+node}
        if local:
            invocation=command;child_env={**os.environ,**env}
        else:
            remote='cd '+shlex.quote(str(self.workspace))+' && exec '+shlex.join(['env',*[f'{k}={v}' for k,v in env.items()],*command])
            invocation=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',node,remote];child_env=os.environ.copy()
        with (destination/'process.log').open('w') as log:
            code=subprocess.call(invocation,cwd=self.workspace,env=child_env,stdout=log,stderr=subprocess.STDOUT)
        elapsed=time.time()-start
        receipt.update(exit_code=code,elapsed_seconds=elapsed,GPU_hours=len(devices)*elapsed/3600,
            end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (destination/'exit.json').write_text(json.dumps(receipt,indent=2)+'\n')
        (destination.parent/'latest.json').write_text(json.dumps(dict(exit_code=code,receipt=str(destination/'exit.json')))+'\n')
        if code:raise RuntimeError(f'{job.identity} exited {code}; original log {destination}/process.log')
        return job.identity

    @staticmethod
    def allocation(job,free):
        eligible=[d for d in free if not job.eligible_devices or d in job.eligible_devices]
        if job.gpu_count==1:return eligible[0] if eligible else None
        for node in dict.fromkeys(n for n,_ in eligible):
            group=[d for d in eligible if d[0]==node]
            if len(group)>=job.gpu_count:return group[:job.gpu_count]
        return None

    def _consume_completions(self, completed, futures, executor, done, external, free, failures):
        for future in completed:
            device=futures.pop(future)
            if device is not None:free.extend(device if isinstance(device,list) else [device])
            try:
                result=future.result()
                if device is not None:done.add(result)
                else:
                    done.update(result)
                    if external-done and not failures:
                        futures[executor.submit(dependency_completion,self.root,external-done)]=None
            except Exception as error:failures.append(str(error))

    def _release_deferred_devices(self, deferred, done, free):
        for device,dependencies in list(deferred.items()):
            if set(dependencies)<=done:
                free.append(device);del deferred[device]

    def run(self, jobs, *, deferred_devices=None):
        pending={job.identity:job for job in jobs};done=set();failures=[]
        external=set().union(*(job.dependencies for job in jobs))-pending.keys()
        deferred=dict(deferred_devices or {})
        if not set(deferred)<=set(self.devices):raise ValueError('deferred devices must belong to this batch')
        for identity in list(pending):
            file=self.root/'jobs'/identity/'latest.json'
            if file.exists() and read_exit(file)==0:
                done.add(identity);del pending[identity]
        with ThreadPoolExecutor(max_workers=len(self.devices)+1) as executor:
            futures={};free=[d for d in self.devices if d not in deferred]
            if external:futures[executor.submit(dependency_completion,self.root,external)]=None
            while pending or futures:
                self._release_deferred_devices(deferred,done,free)
                ready=sorted([j for j in pending.values() if set(j.dependencies)<=done],
                             key=lambda j:-j.estimate_seconds)
                for job in ready:
                    eligible=self.allocation(job,free)
                    if eligible is None or failures:continue
                    for d in (eligible if isinstance(eligible,list) else [eligible]):free.remove(d)
                    del pending[job.identity]
                    futures[executor.submit(self.execute,job,eligible)]=eligible
                if not futures:
                    if failures:break
                    raise RuntimeError('pending jobs have missing dependencies')
                completed,_=wait(futures,return_when=FIRST_COMPLETED)
                self._consume_completions(completed,futures,executor,done,external,free,failures)
            if failures:
                raise RuntimeError('batch engineering/budget failure: '+ '; '.join(failures))
        return sorted(done)


def read_exit(path):
    return json.loads(Path(path).read_text())['exit_code']


def stage_jobs(args):
    root=args.root;G=root/'checkpoints/G/update_00000480'
    local=root/'checkpoints/local/update_00000128';RL=root/'checkpoints/RL/update_00000016'
    def shared(identity,arguments,dependencies,seconds):
        return Job(identity,arguments+['--frame-chunk',str(args.shared_frame_chunk)],dependencies,seconds,
            gpu_count=args.shared_gpus,required_free_MiB=28000 if args.shared_frame_chunk>=16 else 20000)
    if args.stage=='G':
        estimate=60+240*4*args.CFM_condition_seconds/args.shared_gpus
        return [shared('G240',['G','--stop','240'],(),estimate),
                shared('G480',['G','--stop','480','--resume',str(root/'checkpoints/G/update_00000240')],('G240',),estimate)]
    jobs=[Job(f'local_data_{task:04d}',['local-data','--task',str(task),'--checkpoint',str(G),
        '--slots',str(args.slots),'--frame-chunk',str(args.shared_frame_chunk)],estimate_seconds=1800) for task in TASKS]
    jobs.append(shared('local128',['local','--checkpoint',str(G)],tuple(j.identity for j in jobs),
        128*4*6/args.shared_gpus+60))
    jobs.append(shared('RL16',['RL','--checkpoint',str(local)],('local128',),16*4*180/args.shared_gpus+60))
    for version,checkpoint,dependency in [('local128',local,'local128'),('RL16',RL,'RL16')]:
        for task in TASKS:
            identity=f'report_{version}_{task:04d}'
            jobs.append(Job(identity,['report','--task',str(task),'--checkpoint',str(checkpoint),'--slots',str(args.slots),
                '--frame-chunk',str(args.shared_frame_chunk)],(dependency,),4200 if version=='RL16' and task==32 else
                2500 if version=='RL16' and task==12 else 900))
    for task in (12,32):
        for arm in ('other','wrong'):
            jobs.append(Job(f'control_{arm}_{task:04d}',['report','--task',str(task),'--video-arm',arm,
                '--checkpoint',str(RL),'--slots',str(args.slots),'--frame-chunk',str(args.shared_frame_chunk)],
                (f'report_RL16_{task:04d}',),600))
    return jobs


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['teacher-bank','G','decision-report'])
    parser.add_argument('--root',type=Path,default=RUN_ROOT)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--devices',required=True,help='node:index comma-separated live eligible devices')
    parser.add_argument('--teacher-devices',default='',help='measured throughput/budget selection, not utilization filter')
    parser.add_argument('--microbatch',type=int,required=True)
    parser.add_argument('--slots',type=int,default=8)
    parser.add_argument('--required-free-MiB',type=int,required=True)
    parser.add_argument('--maximum-GPUh',type=float,default=None,help='defaults to the registered cumulative root limit')
    parser.add_argument('--shared-gpus',type=int,default=2)
    parser.add_argument('--shared-frame-chunk',type=int,default=16)
    parser.add_argument('--CFM-condition-seconds',type=float,default=28.)
    parser.add_argument('--teacher-update-seconds',action='append',default=[],help='node:index=seconds for a measured complete update')
    args=parser.parse_args()
    devices=[(s.split(':')[0],int(s.split(':')[1])) for s in args.devices.split(',')]
    batch=Batch(args.workspace,devices,root=args.root,maximum_GPUh=args.maximum_GPUh,required_free_MiB=args.required_free_MiB)
    teacher_devices=[(v.split(':')[0],int(v.split(':')[1])) for v in args.teacher_devices.split(',') if v]
    if not set(teacher_devices)<=set(devices):raise ValueError('teacher devices must be in the admitted batch device set')
    estimates={}
    for item in args.teacher_update_seconds:
        location,value=item.split('=');node,gpu=location.split(':');seconds=float(value)
        if (node,int(gpu)) not in teacher_devices or not math.isfinite(seconds) or seconds<=0:
            raise ValueError('teacher estimate must be positive and identify an eligible device')
        estimates[(node,int(gpu))]=seconds
    if args.stage=='teacher-bank':
        from .run import seed_events
        from .contract import materialize_panel
        seed_events(args.root,materialize_panel(args.root))
        done=batch.run(initial_jobs(args.microbatch,args.slots,teacher_devices,root=args.root,teacher_update_seconds=estimates))
        summary=teacher_summary(args.root)
        print(json.dumps(dict(stage=args.stage,complete=True,jobs=done,billing=summary['billing'],
            no_beneficial_teacher_supply=summary['no_beneficial_teacher_supply'])))
    else:
        if args.stage=='G':freeze_bank_scale(args.root)
        done=batch.run(stage_jobs(args))
        result=report_summary(args.root) if args.stage=='decision-report' else billing(args.root)
        print(json.dumps(dict(stage=args.stage,complete=True,jobs=done,summary=result if args.stage=='G' else dict(billing=result['billing'],adjacent=result['adjacent']))))


if __name__=='__main__':main()
