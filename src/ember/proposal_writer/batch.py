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
    """Read each complete selection panel before the next teacher segment."""
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
            readout=f'readout_{task:04d}_{node:08d}';readouts.append(readout)
            seconds=15+contract['teacher']['selection_per_node']*31.335*specs[task]['horizon']/520
            jobs.append(Job(readout,['readout','--task',str(task),'--stop',str(node),'--slots',str(slots)],
                () if node==0 else (identity,predecessor),seconds,required_free_MiB=18000))
            predecessor=readout
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

    def bounded_analysis_devices(self,devices):
        """Only an explicitly registered, still-live analysis child gets the exception."""
        active=set()
        for file in self.root.glob('profile*/launch.json'):
            if (file.parent/'exit.json').exists():continue
            launch=json.loads(file.read_text())
            if not launch.get('analysis_exception_only') or not launch.get('PID'):continue
            address=launch['node'],launch['physical_gpu']
            if any((node,gpu['index'])==address and any(p['owner']=='ymdai' and int(p['pid'])==launch['PID']
                for p in gpu['processes']) for node,gpu in devices):active.add(address)
        return active

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
            analyses=self.bounded_analysis_devices(devices)
            ordinary=allocated | (outside-analyses) | {(node,gpu)}
            if len(ordinary) > cap or sum(n==node for n,_ in combined)>6:
                raise RuntimeError('outside project allocation would exceed the shared cap')
            peak=json.loads((self.root/'run_contract.json').read_text())['limits']['added_peak_GiB']
            used,limit=quota_headroom(quota.stdout,peak_GiB=peak)
            forecast=self.track_cost_estimate(job,node,gpu)
            return dict(node=node,physical_gpu=gpu,uuid=selected['uuid'],free_MiB=selected['mem_free_mib'],
                        utilization=selected['util_gpu_pct'],concurrency_cap=cap,quota_used_KiB=used,quota_KiB=limit,
                        ordinary_physical_GPU_count=len(ordinary),physical_GPU_count_including_analysis=len(combined),
                        active_bounded_analysis_devices=[f'{n}:{g}' for n,g in sorted(analyses)],
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
                # The shared gpu01:5 MT101 consumer exhausted its EGL framebuffers at16.
                # Smaller physical waves still evaluate every registered selection/report row.
                arguments[arguments.index('--slots')+1]=str(min(8,int(arguments[arguments.index('--slots')+1])))
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

    def _saturated_omissions(self,pending,done,omitted,checked):
        from ember.pi05_source_checkpoint import read_json,write_json_atomic
        contract=read_json(self.root/'run_contract.json')
        if contract.get('teacher',{}).get('saturation_stop')!=16:return
        nodes=contract['teacher']['readout_updates']
        for identity in sorted(done-checked):
            if not identity.startswith('readout_'):continue
            _,task_text,node_text=identity.split('_');task,node=int(task_text),int(node_text)
            directory=self.root/'teachers'/f'task_{task:04d}'/'selection'
            paths=[directory/f'update_{u:08d}'/'complete.json' for u in nodes if u<=node]
            if not all(p.exists() for p in paths):continue
            records=[read_json(p) for p in paths]
            if any(not r['complete'] or len(r['rows'])!=16 for r in records):continue
            checked.add(identity)
            saturated=[u for u,r in zip((u for u in nodes if u<=node),records,strict=True)
                       if sum(bool(row['success']) for row in r['rows'])==16]
            if not saturated:continue
            winner=min(saturated);file=self.root/'scientific_omissions'/f'task_{task:04d}.json'
            old=read_json(file) if file.exists() else dict(omitted_jobs=[])
            skipped=[key for key in pending if key.startswith((f'teacher_{task:04d}_',f'readout_{task:04d}_'))
                     and int(key.split('_')[-1])>winner]
            for key in skipped:del pending[key]
            omitted.update(skipped);omitted.update(old['omitted_jobs'])
            write_json_atomic(file,dict(task=task,reason='selection_saturated',selection_successes=16,
                selection_rows=16,incumbent_update=winner,all_earlier_candidates_complete=True,
                complete_selection_source=str(directory/f'update_{winner:08d}'/'complete.json'),
                omitted_jobs=sorted(set(old['omitted_jobs'])|set(skipped)),new_work_not_started=True,
                prior_attempts_and_actual_scores_preserved=True,
                in_flight_jobs_not_interrupted=True,existing_late_checkpoint_weights_retained=True))

    def _dispatch_ready(self,executor,pending,free,settled,failures,futures):
        costs=critical_path_costs(pending)
        ready=sorted((j for j in pending.values() if set(j.dependencies)<=settled),
                     key=lambda j:(j.bank_priority,-costs[j.identity]))
        for job in ready:
            eligible=self.allocation(job,free)
            if eligible is None or any(not isinstance(e,DeviceWorkloadFailure) for e in failures):continue
            for d in (eligible if isinstance(eligible,list) else [eligible]):free.remove(d)
            del pending[job.identity]
            futures[executor.submit(self.execute,job,eligible)]=eligible

    def run(self, jobs, *, deferred_devices=None, on_completion=None):
        pending={job.identity:job for job in jobs};done=set();omitted=set();checked=set();failures=[]
        external=set().union(*(job.dependencies for job in jobs))-pending.keys()
        deferred=dict(deferred_devices or {})
        if not set(deferred)<=set(self.devices):raise ValueError('deferred devices must belong to this batch')
        for identity in list(pending):
            file=self.root/'jobs'/identity/'latest.json'
            if file.exists() and read_exit(file)==0:
                done.add(identity);del pending[identity]
        self._saturated_omissions(pending,done,omitted,checked)
        if on_completion:on_completion(done)
        with ThreadPoolExecutor(max_workers=len(self.devices)+1) as executor:
            futures={};free=[d for d in self.devices if d not in deferred]
            if external:futures[executor.submit(dependency_completion,self.root,external)]=None
            while pending or futures:
                self._release_deferred_devices(deferred,done,free)
                self._dispatch_ready(executor,pending,free,done|omitted,failures,futures)
                if not futures:
                    if failures:break
                    raise RuntimeError('pending jobs have missing dependencies')
                completed,_=wait(futures,return_when=FIRST_COMPLETED)
                self._consume_completions(completed,futures,executor,done,external,free,failures)
                self._saturated_omissions(pending,done,omitted,checked)
                if on_completion:on_completion(done)
            if failures:
                raise RuntimeError('batch engineering/budget failure: '+ '; '.join(map(str,failures)))
        return sorted(done)


def teacher_bank_ready(root, done):
    """Publish all36 E and MT-bootstrap readiness, independently of reports."""
    from .run import select_teacher
    from ember.pi05_source_checkpoint import read_json, write_json_atomic
    root=Path(root)
    if (root/'bank_ready.json').exists():return
    contract=read_json(root/'run_contract.json');nodes=contract['teacher']['readout_updates']
    for task in TASKS:
        directory=root/'teachers'/f'task_{task:04d}'
        omission=root/'scientific_omissions'/f'task_{task:04d}.json'
        skipped=set(read_json(omission)['omitted_jobs']) if omission.exists() else set()
        if not (directory/'E.json').exists() and all(f'readout_{task:04d}_{node:08d}' in done|skipped for node in nodes):
            select_teacher(root,task)
    if not all((root/'teachers'/f'task_{task:04d}'/'E.json').exists() for task in TASKS):return
    if not all((root/'bootstrap'/f'task_{task:04d}'/'complete.json').exists() for task in TASKS):return
    scale=freeze_bank_scale(root)
    pending=[task for task in TASKS if f'teacher_report_{task:04d}' not in done]
    entries=[read_json(root/'teachers'/f'task_{t:04d}'/'E.json') for t in TASKS]
    write_json_atomic(root/'bank_ready.json',dict(complete=True,task_count=len(TASKS),
        initial_parent_count=len(TASKS),history_episodes=len(TASKS),initial_parent='MT300 only',
        evaluated_teacher_candidate_count=sum(len(e['evaluated_candidates']) for e in entries),
        ready_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),bank_scale=str(root/'bank_scale.json'),
        scale_source_count=len(scale['sources']),independent_teacher_report_pending_tasks=pending,
        reports_not_used_for_E_or_scale=True))


def read_exit(path):
    return json.loads(Path(path).read_text())['exit_code']


def shared_learning_job(root,contract,stage,shared_gpus,stop,slots):
    """One shared current-theta batch owner, with an explicit recorded physical topology."""
    if shared_gpus is None or not 1<=shared_gpus<=4:raise ValueError('shared physical world size needs actual throughput selection')
    if stop not in contract['updates'].get(stage,[]):raise ValueError('formal shared update not preregistered')
    arguments=[stage,'--stop',str(stop),'--slots',str(slots),'--frame-chunk','16']
    if stage=='local':arguments+=['--checkpoint',str(root/'checkpoints/G'/f"update_{contract['fixed_G_update']:08d}")]
    if stage=='RL':arguments+=['--checkpoint',str(root/'checkpoints/local'/f"update_{contract['fixed_local_update']:08d}")]
    previous=[u for u in contract['updates'][stage] if u<stop and (root/'checkpoints'/stage/f'update_{u:08d}/manifest.json').exists()]
    if previous:arguments+=['--resume',str(root/'checkpoints'/stage/f'update_{max(previous):08d}')]
    seconds=contract.get('stage_estimate_seconds',{}).get(f'{stage}_{stop}',0.)
    return [Job(f'{stage}_{stop:08d}',arguments,estimate_seconds=seconds,gpu_count=shared_gpus)]


def stage_jobs(root,stage,*,microbatch,slots,teacher_devices,estimates,shared_gpus,stop,
               physical_workers=None,condition_batch=None,video_arm='correct'):
    """Independent condition jobs reuse this queue; shared actor batches retain one owner."""
    root=Path(root);contract=json.loads((root/'run_contract.json').read_text())
    if stage=='teacher-data':return initial_jobs(microbatch,slots,teacher_devices,root=root,teacher_update_seconds=estimates)
    if stage=='teacher-reports':
        return [j for j in initial_jobs(microbatch,slots,teacher_devices,root=root,teacher_update_seconds=estimates)
                if j.identity.startswith('teacher_report_')]
    if stage in ('G','local','RL'):return shared_learning_job(root,contract,stage,shared_gpus,stop,slots)
    jobs=[];seconds=contract.get('condition_estimate_seconds',{}).get(stage,0.)
    if stage=='student-data':
        checkpoint=root/'checkpoints/G'/f"update_{contract['student_refresh']['G_update']:08d}"
        for task in TASKS:jobs.append(Job(f"student_{contract['student_refresh']['version']}_{task:04d}",
            ['student-data','--task',str(task),'--checkpoint',str(checkpoint),'--slots',str(slots)],estimate_seconds=seconds))
    elif stage=='training-report':
        for name,update in contract['training_report']['versions']:
            checkpoint=root/'checkpoints'/name/f'update_{update:08d}'
            for task in TASKS:
                for ordinal in range(contract['training_report']['conditions_per_task']):
                    jobs.append(Job(f'train_report_{name}_{update:08d}_{task:04d}_{ordinal:03d}',
                        ['report','--task',str(task),'--condition-ordinal',str(ordinal),'--checkpoint',str(checkpoint),
                         '--slots',str(slots)],estimate_seconds=seconds))
    elif stage=='formal':
        from .evaluation import formal_queue
        panel=json.loads((root/'formal_panel.json').read_text())
        if physical_workers is None or physical_workers<1 or condition_batch is None or condition_batch<1:
            raise ValueError('formal persistent workers need measured positive physical condition batch and devices')
        formal_queue(root,workers=physical_workers,condition_batch=condition_batch,arm=video_arm)
        for ordinal in range(physical_workers):
            identity=f"formal_{Path(panel['checkpoint']).parent.name}_{Path(panel['checkpoint']).name}_{video_arm}_worker{ordinal:03d}"
            jobs.append(Job(identity,['formal','--worker-id',identity,'--video-arm',video_arm,
                '--checkpoint',panel['checkpoint'],'--slots',str(slots),'--condition-batch',str(condition_batch)],
                estimate_seconds=seconds*len(panel['rows'])/physical_workers))
    if not jobs:raise ValueError('stage has no registered independent work')
    return jobs


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['teacher-data','teacher-reports','G','student-data','local','RL','training-report','formal'])
    parser.add_argument('--shared-gpus',type=int)
    parser.add_argument('--stop',type=int)
    parser.add_argument('--bank-only',action='store_true',help='prepare E/H first; reports use their independently frozen consumer')
    parser.add_argument('--root',type=Path,default=RUN_ROOT)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--devices',required=True,help='node:index comma-separated live eligible devices')
    parser.add_argument('--teacher-devices',default='',help='measured throughput/budget selection, not utilization filter')
    parser.add_argument('--microbatch',type=int,required=True)
    parser.add_argument('--slots',type=int,default=8)
    parser.add_argument('--condition-batch',type=int,help='measured independent formal paths per persistent queue shard')
    parser.add_argument('--video-arm',choices=['correct','other','wrong'],default='correct')
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
    jobs=stage_jobs(args.root,args.stage,microbatch=args.microbatch,slots=args.slots,teacher_devices=teacher_devices,
        estimates=estimates,shared_gpus=args.shared_gpus,stop=args.stop,physical_workers=len(devices),
        condition_batch=args.condition_batch,video_arm=args.video_arm)
    if args.bank_only:jobs=[j for j in jobs if not j.identity.startswith('teacher_report_')]
    done=batch.run(jobs,on_completion=(lambda done:teacher_bank_ready(args.root,done)) if args.stage=='teacher-data' else None)
    if args.bank_only:
        print(json.dumps(dict(stage='teacher-bank',complete=True,jobs=done,billing=billing(args.root),independent_report_pending=True)))
        return
    summary=teacher_summary(args.root) if args.stage in ('teacher-data','teacher-reports') else None
    if args.stage=='formal':
        from .evaluation import formal_summary
        formal_summary(args.root,args.video_arm)
    print(json.dumps(dict(stage=args.stage,complete=True,jobs=done,billing=billing(args.root),
        teacher_report=summary['independent_report'] if summary else None)))


if __name__=='__main__':main()
