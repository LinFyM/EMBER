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
from .analysis import billing, teacher_summary, report_summary

PREFLIGHT = '/data0/user/ymdai/.codex/skills/gpu-preflight/scripts/gpu_preflight.py'


@dataclass
class Job:
    identity: str
    arguments: list
    dependencies: tuple = ()
    estimate_seconds: float = 0
    eligible_devices: tuple = ()
    device_estimates: dict = field(default_factory=dict)

    def estimated_seconds(self, device):
        return self.device_estimates.get(device, self.estimate_seconds)


def initial_jobs(microbatch,slots,teacher_devices,*,root=RUN_ROOT,teacher_update_seconds=None):
    # Complete query/function/optimizer measurements, with a conservative unmeasured fallback.
    jobs = []
    for task in TASKS:
        collect = f'collect_{task:04d}'
        jobs.append(Job(collect, ['collect','--task',str(task),'--slots',str(slots)],estimate_seconds=600 if task in (32,38) else 360))
        for event in (0,1):
            teacher = f'teacher_{task:04d}_{event}'
            checkpoint=Path(root)/'events'/f'task_{task:04d}_event_{event:02d}'/'teacher/checkpoints/update_00000160/manifest.json'
            updates=320 if checkpoint.exists() and json.loads(checkpoint.read_text())['complete'] else 480
            estimates = {device: 60+updates*seconds for device,seconds in (teacher_update_seconds or {}).items()}
            jobs.append(Job(teacher,['teacher','--task',str(task),'--event',str(event),
                '--microbatch',str(microbatch),'--stop','480'],(collect,),
                60+updates*20.8,tuple(teacher_devices),estimates))
            jobs.append(Job(f'audit_{task:04d}_{event}',['audit','--task',str(task),'--event',str(event),
                '--slots',str(slots)],(teacher,),600 if task in (32,38) else 360))
    return jobs


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
            quota = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','strg01',
                "xfs_quota -c 'quota -u -b ymdai' /data1"],check=True,capture_output=True,text=True)
            (destination/'gpu_preflight.json').write_text(result.stdout)
            (destination/'quota.txt').write_text(quota.stdout)
            devices = [(n,g) for n,r in snapshot['node_results'].items() for g in r.get('gpus',[])]
            idle = sum(not g['processes'] and g['mem_used_mib'] < 1000 for _,g in devices)
            cap = 6 if idle <= 10 else 8
            allocated = set(self.running)
            selected = next(g for n,g in devices if n==node and g['index']==gpu)
            if selected['compute_mode']=='Prohibited' or selected['mem_free_mib'] < self.required_free_MiB:
                raise RuntimeError(f'live device headroom below measured workload requirement: {node}:{gpu}')
            if len(allocated) >= cap or sum(n==node for n,_ in allocated) >= 6:
                raise RuntimeError('current physical GPU project cap prevents this launch')
            # Live allocation may include a task-owned process outside this driver's futures.
            outside = {(n,g['index']) for n,g in devices if any(p['owner']=='ymdai' and
                ('proposal_writer' in p['process_name'] or 'EMBER' in p['process_name']) for p in g['processes'])}
            if len(allocated | outside | {(node,gpu)}) > cap:
                raise RuntimeError('outside project allocation would exceed the shared cap')
            numbers = [line.split() for line in quota.stdout.splitlines() if line.startswith('/dev/')]
            if len(numbers)!=1:
                raise RuntimeError('authoritative data1 quota could not be parsed')
            used, limit = int(numbers[0][1]), int(numbers[0][2])
            if used + 64*1024**2 >= limit:
                raise RuntimeError('data1 independent quota cannot cover retained peak')
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
                estimate=max(estimate,60+20.8*stop)
            inflight+=max(elapsed,estimate)/3600
        estimate = job.estimated_seconds((node,gpu))
        if not math.isfinite(estimate) or estimate <= 0:
            raise ValueError('GPU job requires a finite positive full-cost estimate')
        if spent + inflight + estimate/3600 > self.maximum_GPUh:
            raise RuntimeError('registered total GPUh exhausted/projected boundary; scientific budget decision required')
        self.running[(node,gpu)] = dict(start=time.time(),job=job.identity,estimate_seconds=estimate)

    def execute(self, job, device):
        node, gpu = device
        job_root=self.root/'jobs'/job.identity;job_root.mkdir(parents=True,exist_ok=True)
        ordinal=len(list(job_root.glob('attempt_*')))+1
        destination=job_root/f'attempt_{ordinal:03d}';destination.mkdir(exist_ok=False)
        admission=self.admit(job,node,gpu,destination)
        env=dict(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(self.workspace/'src'),CUDA_VISIBLE_DEVICES=str(gpu),
                 NCCL_P2P_DISABLE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MUJOCO_GL='egl')
        command=[str(ASSET_ROOT/'.venv/bin/python'),str(self.workspace/'scripts/video_guided_proposal_writer.py'),
                 *job.arguments,'--root',str(self.root),'--physical-gpu',str(gpu)]
        start=time.time();receipt=dict(job=job.identity,command=command,env=env,workspace=str(self.workspace),
            code_commit=self.code,admission=admission,start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            physical_GPU_count=1,estimate_seconds=job.estimated_seconds(device),budget_GPUh=self.maximum_GPUh)
        (destination/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
        local=socket.gethostname().split('.')[0].lower() in {node,'bci-'+node}
        if local:
            invocation=command;child_env={**os.environ,**env}
        else:
            remote='cd '+shlex.quote(str(self.workspace))+' && exec '+shlex.join(['env',*[f'{k}={v}' for k,v in env.items()],*command])
            invocation=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',node,remote];child_env=os.environ.copy()
        try:
            with (destination/'process.log').open('w') as log:
                code=subprocess.call(invocation,cwd=self.workspace,env=child_env,stdout=log,stderr=subprocess.STDOUT)
            elapsed=time.time()-start
            receipt.update(exit_code=code,elapsed_seconds=elapsed,GPU_hours=elapsed/3600,
                           end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            (destination/'exit.json').write_text(json.dumps(receipt,indent=2)+'\n')
            (job_root/'latest.json').write_text(json.dumps(dict(exit_code=code,receipt=str(destination/'exit.json')))+'\n')
            if code:
                raise RuntimeError(f'{job.identity} exited {code}; original log {destination}/process.log')
            return job.identity
        finally:
            with self.lock:self.running.pop((node,gpu),None)

    def run(self, jobs):
        pending={job.identity:job for job in jobs};done=set();failures=[]
        for identity in list(pending):
            file=self.root/'jobs'/identity/'latest.json'
            if file.exists() and read_exit(file)==0:
                done.add(identity);del pending[identity]
        with ThreadPoolExecutor(max_workers=len(self.devices)) as executor:
            futures={};free=list(self.devices)
            while pending or futures:
                ready=sorted([j for j in pending.values() if set(j.dependencies)<=done],
                             key=lambda j:-j.estimate_seconds)
                for job in ready:
                    eligible=next((d for d in free if not job.eligible_devices or d in job.eligible_devices),None)
                    if eligible is None or failures:continue
                    free.remove(eligible);del pending[job.identity]
                    futures[executor.submit(self.execute,job,eligible)]=eligible
                if not futures:
                    if failures:break
                    raise RuntimeError('pending jobs have missing dependencies')
                completed,_=wait(futures,return_when=FIRST_COMPLETED)
                for future in completed:
                    free.append(futures.pop(future))
                    try:done.add(future.result())
                    except Exception as error:failures.append(str(error))
            if failures:
                raise RuntimeError('batch engineering/budget failure: '+ '; '.join(failures))
        return sorted(done)


def read_exit(path):
    return json.loads(Path(path).read_text())['exit_code']


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['initial-teachers'])
    parser.add_argument('--root',type=Path,default=RUN_ROOT)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--devices',required=True,help='node:index comma-separated live eligible devices')
    parser.add_argument('--teacher-devices',required=True,help='measured throughput/budget selection, not utilization filter')
    parser.add_argument('--microbatch',type=int,required=True)
    parser.add_argument('--slots',type=int,default=8)
    parser.add_argument('--required-free-MiB',type=int,required=True)
    parser.add_argument('--maximum-GPUh',type=float,default=None,help='defaults to the registered cumulative root limit')
    parser.add_argument('--teacher-update-seconds',action='append',default=[],help='node:index=seconds for a measured complete update')
    args=parser.parse_args()
    devices=[(s.split(':')[0],int(s.split(':')[1])) for s in args.devices.split(',')]
    batch=Batch(args.workspace,devices,root=args.root,maximum_GPUh=args.maximum_GPUh,required_free_MiB=args.required_free_MiB)
    teacher_devices=[(v.split(':')[0],int(v.split(':')[1])) for v in args.teacher_devices.split(',')]
    if not set(teacher_devices)<=set(devices):raise ValueError('teacher devices must be in the admitted batch device set')
    estimates={}
    for item in args.teacher_update_seconds:
        location,value=item.split('=');node,gpu=location.split(':');seconds=float(value)
        if (node,int(gpu)) not in teacher_devices or not math.isfinite(seconds) or seconds<=0:
            raise ValueError('teacher estimate must be positive and identify an eligible device')
        estimates[(node,int(gpu))]=seconds
    done=batch.run(initial_jobs(args.microbatch,args.slots,teacher_devices,root=args.root,teacher_update_seconds=estimates))
    summary=teacher_summary(args.root)
    write=json.dumps(dict(stage=args.stage,complete=True,jobs=done,billing=summary['billing'],
                         no_beneficial_teacher_supply=summary['no_beneficial_teacher_supply']))
    print(write)


if __name__=='__main__':main()
