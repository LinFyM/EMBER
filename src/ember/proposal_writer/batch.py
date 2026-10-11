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
from .analysis import billing, teacher_summary, freeze_bank_scale

PREFLIGHT = '/data0/user/ymdai/.codex/skills/gpu-preflight/scripts/gpu_preflight.py'
UNMEASURED_TEACHER_SECONDS = 4.4  # Actual shared-card query28 native demoFM complete update.


class DeviceWorkloadFailure(RuntimeError):
    """Located renderer/headroom/OOM failure; quarantine devices, never retry."""


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
    bank_priority: int = 0

    def estimated_seconds(self, device):
        return self.device_estimates.get(device, self.estimate_seconds)


def critical_path_costs(jobs):
    children = {identity: [] for identity in jobs}
    for identity, job in jobs.items():
        for dependency in job.dependencies:
            if dependency in children:children[dependency].append(identity)
    costs = {}
    def cost(identity, ancestors=frozenset()):
        if identity in ancestors:raise ValueError('job dependencies contain a cycle')
        if identity not in costs:
            job=jobs[identity]
            duration=min(job.device_estimates.values(),default=job.estimate_seconds)
            costs[identity]=duration+max((cost(child,ancestors|{identity}) for child in children[identity]),default=0)
        return costs[identity]
    for identity in jobs:cost(identity)
    return costs


def initial_jobs(microbatch,slots,teacher_devices,*,root=RUN_ROOT,teacher_update_seconds=None):
    """One continuous teacher/task, while released checkpoint readouts run independently."""
    contract=json.loads((Path(root)/'run_contract.json').read_text())
    nodes=contract['teacher']['readout_updates'];specs={r['global_task_id']:r for r in contract['environment']['tasks']}
    jobs=[]
    for task in TASKS:
        predecessor=None;readouts=[]
        for node in nodes:
            if node:
                identity=f'teacher_{task:04d}_{node:08d}';updates=node-nodes[nodes.index(node)-1]
                estimates={device:15+updates*seconds for device,seconds in (teacher_update_seconds or {}).items()}
                jobs.append(Job(identity,['teacher','--task',str(task),'--microbatch',str(microbatch),'--stop',str(node)],
                    () if predecessor is None else (predecessor,),15+updates*UNMEASURED_TEACHER_SECONDS,
                    tuple(teacher_devices),estimates,required_free_MiB=18000))
                predecessor=identity
            readout=f'readout_{task:04d}_{node:08d}';readouts.append(readout)
            seconds=15+contract['teacher']['selection_H_per_node']*31.335*specs[task]['horizon']/520
            jobs.append(Job(readout,['readout','--task',str(task),'--stop',str(node),'--slots',str(slots)],
                () if node==0 else (predecessor,),seconds,required_free_MiB=18000))
        seconds=15+2*contract['teacher']['report_per_arm']*31.335*specs[task]['horizon']/520
        jobs.append(Job(f'teacher_report_{task:04d}',['teacher-report','--task',str(task),'--slots',str(slots)],
            tuple(readouts),seconds,required_free_MiB=18000,bank_priority=1))
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


def quota_headroom(report, *, peak_GiB=128):
    numbers=[line.split() for line in report.splitlines() if line.startswith('/dev/')]
    if len(numbers)!=1:
        raise RuntimeError('authoritative data1 quota could not be parsed')
    used,limit=int(numbers[0][1]),int(numbers[0][2])
    if used+peak_GiB*1024**2>=limit:
        raise RuntimeError('data1 independent quota cannot cover retained peak')
    return used,limit


class Batch:
    """One coordinator owns resource decisions; completion futures release slots."""
    def __init__(self, workspace, devices, *, root=RUN_ROOT, required_free_MiB=12000):
        self.workspace, self.devices, self.root = Path(workspace), list(devices), Path(root)
        self.lock = threading.Lock()
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
                raise DeviceWorkloadFailure(f'live device headroom below measured workload requirement: {node}:{gpu}')
            if len(allocated) >= cap or sum(n==node for n,_ in allocated) >= 6:
                raise RuntimeError('current physical GPU project cap prevents this launch')
            # Live allocation may include a task-owned process outside this driver's futures.
            outside = {(n,g['index']) for n,g in devices if any(p['owner']=='ymdai' for p in g['processes'])}
            combined = allocated | outside | {(node,gpu)}
            if len(combined) > cap or sum(n==node for n,_ in combined)>6:
                raise RuntimeError('outside project allocation would exceed the shared cap')
            peak=json.loads((self.root/'run_contract.json').read_text())['limits']['added_peak_GiB']
            used,limit=quota_headroom(quota.stdout,peak_GiB=peak)
            forecast=self.track_cost_estimate(job,node,gpu)
            return dict(node=node,physical_gpu=gpu,uuid=selected['uuid'],free_MiB=selected['mem_free_mib'],
                        utilization=selected['util_gpu_pct'],concurrency_cap=cap,quota_used_KiB=used,quota_KiB=limit,
                        cost_forecast=forecast)

    def track_cost_estimate(self,job,node,gpu):
        # Owner policy: cost forecasts inform scheduling, never prohibit a launch.
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
        self.running[(node,gpu)] = dict(start=time.time(),job=job.identity,estimate_seconds=estimate)
        return dict(settled_GPU_hours=spent,inflight_GPU_hours=inflight,
                    added_device_GPU_hours=estimate/3600,projected_GPU_hours=spent+inflight+estimate/3600,
                    GPU_hours_hard_cap_active=False)

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
            if arguments[0] in ('readout','teacher-report','report') and admissions[0]['free_MiB']<25000:
                arguments[arguments.index('--slots')+1]=str(min(16,int(arguments[arguments.index('--slots')+1])))
            if len(devices)>1 and '--frame-chunk' in arguments:
                env['EMBER_META_FRAME_CHUNKS']=','.join(str(16 if a['free_MiB']>=28000 else
                    8 if a['free_MiB']>=21000 else 4) for a in admissions)
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
            physical_GPU_count=len(devices),estimate_seconds=job.estimated_seconds(devices[0]),
            budget_GPUh=None,GPU_hours_hard_cap_active=False)
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
        if code:
            failure=f'{job.identity} exited {code}; original log {destination}/process.log'
            log=(destination/'process.log').read_text(errors='replace')
            scoped=any(marker in log for marker in ('Offscreen framebuffer is not complete',
                'torch.OutOfMemoryError', 'torch.cuda.OutOfMemoryError', 'CUDA out of memory'))
            raise (DeviceWorkloadFailure if scoped else RuntimeError)(failure)
        return job.identity

    @staticmethod
    def allocation(job,free):
        eligible=[d for d in free if not job.eligible_devices or d in job.eligible_devices]
        if job.gpu_count==1:return min(eligible,key=job.estimated_seconds) if eligible else None
        for node in dict.fromkeys(n for n,_ in eligible):
            group=[d for d in eligible if d[0]==node]
            if len(group)>=job.gpu_count:return group[:job.gpu_count]
        return None

    def _consume_completions(self, completed, futures, executor, done, external, free, failures):
        for future in completed:
            device=futures.pop(future)
            try:
                result=future.result()
                if device is not None:
                    done.add(result);free.extend(device if isinstance(device,list) else [device])
                else:
                    done.update(result)
                    if external-done and not any(not isinstance(e,DeviceWorkloadFailure) for e in failures):
                        futures[executor.submit(dependency_completion,self.root,external-done)]=None
            except Exception as error:failures.append(error)

    def _release_deferred_devices(self, deferred, done, free):
        for device,dependencies in list(deferred.items()):
            if set(dependencies)<=done:
                free.append(device);del deferred[device]

    def run(self, jobs, *, deferred_devices=None, on_completion=None):
        pending={job.identity:job for job in jobs};done=set();failures=[]
        external=set().union(*(job.dependencies for job in jobs))-pending.keys()
        deferred=dict(deferred_devices or {})
        if not set(deferred)<=set(self.devices):raise ValueError('deferred devices must belong to this batch')
        for identity in list(pending):
            file=self.root/'jobs'/identity/'latest.json'
            if file.exists() and read_exit(file)==0:
                done.add(identity);del pending[identity]
        if on_completion:on_completion(done)
        with ThreadPoolExecutor(max_workers=len(self.devices)+1) as executor:
            futures={};free=[d for d in self.devices if d not in deferred]
            if external:futures[executor.submit(dependency_completion,self.root,external)]=None
            while pending or futures:
                self._release_deferred_devices(deferred,done,free)
                costs=critical_path_costs(pending)
                ready=sorted([j for j in pending.values() if set(j.dependencies)<=done],
                             key=lambda j:(j.bank_priority,-costs[j.identity]))
                for job in ready:
                    eligible=self.allocation(job,free)
                    if eligible is None or any(not isinstance(e,DeviceWorkloadFailure) for e in failures):continue
                    for d in (eligible if isinstance(eligible,list) else [eligible]):free.remove(d)
                    del pending[job.identity]
                    futures[executor.submit(self.execute,job,eligible)]=eligible
                if not futures:
                    if failures:break
                    raise RuntimeError('pending jobs have missing dependencies')
                completed,_=wait(futures,return_when=FIRST_COMPLETED)
                self._consume_completions(completed,futures,executor,done,external,free,failures)
                if on_completion:on_completion(done)
            if failures:
                raise RuntimeError('batch engineering/budget failure: '+ '; '.join(map(str,failures)))
        return sorted(done)


def teacher_bank_ready(root, done):
    """Publish E/H readiness at completion events, independently of teacher reports."""
    from .run import select_teacher
    from ember.pi05_source_checkpoint import read_json, write_json_atomic
    root=Path(root)
    if (root/'bank_ready.json').exists():return
    contract=read_json(root/'run_contract.json');nodes=contract['teacher']['readout_updates']
    for task in TASKS:
        directory=root/'teachers'/f'task_{task:04d}'
        if not (directory/'E.json').exists() and all(f'readout_{task:04d}_{node:08d}' in done for node in nodes):
            select_teacher(root,task)
    if not all((root/'teachers'/f'task_{task:04d}'/'E.json').exists() for task in TASKS):return
    scale=freeze_bank_scale(root)
    pending=[task for task in TASKS if f'teacher_report_{task:04d}' not in done]
    write_json_atomic(root/'bank_ready.json',dict(complete=True,task_count=len(TASKS),
        parent_count=len(TASKS)*len(nodes),history_episodes=len(TASKS)*len(nodes)*contract['teacher']['selection_H_per_node'],
        ready_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),bank_scale=str(root/'bank_scale.json'),
        scale_source_count=len(scale['sources']),independent_teacher_report_pending_tasks=pending,
        reports_not_used_for_E_or_scale=True))


def read_exit(path):
    return json.loads(Path(path).read_text())['exit_code']


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['teacher-data'])
    parser.add_argument('--root',type=Path,default=RUN_ROOT)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--devices',required=True,help='node:index comma-separated live eligible devices')
    parser.add_argument('--teacher-devices',default='',help='measured throughput/budget selection, not utilization filter')
    parser.add_argument('--microbatch',type=int,required=True)
    parser.add_argument('--slots',type=int,default=8)
    parser.add_argument('--required-free-MiB',type=int,required=True)
    parser.add_argument('--teacher-update-seconds',action='append',default=[],help='node:index=seconds for a measured complete update')
    args=parser.parse_args()
    devices=[(s.split(':')[0],int(s.split(':')[1])) for s in args.devices.split(',')]
    batch=Batch(args.workspace,devices,root=args.root,required_free_MiB=args.required_free_MiB)
    teacher_devices=[(v.split(':')[0],int(v.split(':')[1])) for v in args.teacher_devices.split(',') if v]
    if not set(teacher_devices)<=set(devices):raise ValueError('teacher devices must be in the admitted batch device set')
    estimates={}
    for item in args.teacher_update_seconds:
        location,value=item.split('=');node,gpu=location.split(':');seconds=float(value)
        if (node,int(gpu)) not in teacher_devices or not math.isfinite(seconds) or seconds<=0:
            raise ValueError('teacher estimate must be positive and identify an eligible device')
        estimates[(node,int(gpu))]=seconds
    done=batch.run(initial_jobs(args.microbatch,args.slots,teacher_devices,root=args.root,teacher_update_seconds=estimates),
                   on_completion=lambda done:teacher_bank_ready(args.root,done))
    summary=teacher_summary(args.root)
    print(json.dumps(dict(stage=args.stage,complete=True,jobs=done,billing=summary['billing'],teacher_report=summary['independent_report'])))


if __name__=='__main__':main()
