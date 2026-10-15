"""Event-driven bounded jobs with live shared-device admission and exact receipts."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
import datetime
import json
import math
import os
from pathlib import Path
import select
import shlex
import signal
import socket
import subprocess
import threading
import time

from .contract import ASSET_ROOT, RUN_ROOT, TASKS
from .analysis import billing

PREFLIGHT = '/data0/user/ymdai/.codex/skills/gpu-preflight/scripts/gpu_preflight.py'
UNMEASURED_TEACHER_SECONDS = 4.4  # Actual shared-card query28 native demoFM complete update.


class DeviceWorkloadFailure(RuntimeError):
    """Located renderer/headroom/OOM failure; quarantine devices, never retry."""


def process_identity(pid):
    """Linux process identity and retained child status for an explicit CPU handoff."""
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(PID=int(pid), state=fields[0], parent_PID=int(fields[1]),
                start_ticks=int(fields[19]), wait_status=int(fields[49]))


def live_gpu_snapshot(destination):
    """Retain failed probes and retry the query once; never launch from missing state."""
    for attempt, timeout in enumerate((8, 15), 1):
        result = subprocess.run(['python3', PREFLIGHT, '--nodes', 'gpu01,gpu02', '--json',
                                 '--timeout-s', str(timeout)], capture_output=True, text=True)
        prefix = Path(destination)/f'gpu_preflight_attempt_{attempt:02d}'
        prefix.with_suffix('.stdout.txt').write_text(result.stdout)
        prefix.with_suffix('.stderr.txt').write_text(result.stderr)
        prefix.with_suffix('.exit.json').write_text(json.dumps(dict(exit_code=result.returncode,
            query_timeout_seconds=timeout, UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
        try:
            snapshot = json.loads(result.stdout)
        except json.JSONDecodeError:
            snapshot = {}
        results = snapshot.get('node_results', {})
        if (result.returncode == 0 and not snapshot.get('bad_nodes')
                and set(results) == {'gpu01', 'gpu02'}
                and all(r.get('gpus') and not r.get('process_check_error') for r in results.values())):
            (Path(destination)/'gpu_preflight.json').write_text(result.stdout)
            return snapshot
    raise RuntimeError(f'live two-node allocation query remains incomplete; original probes: {destination}')


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
    def __init__(self, workspace, devices, *, root=RUN_ROOT, required_free_MiB=12000, work_event=None):
        self.workspace, self.devices, self.root = Path(workspace), list(devices), Path(root)
        self.lock = threading.Lock()
        self.running = {}
        self.required_free_MiB=required_free_MiB
        self.work_event=work_event
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
            snapshot = live_gpu_snapshot(destination)
            quota = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','strg01',
                "xfs_quota -c 'quota -u -b ymdai' /data1"],check=True,capture_output=True,text=True)
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
            if (path.parent/'exit.json').exists() or (path.parent/'exit_interval.json').exists():continue
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
                TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8',MUJOCO_GL='egl',
                CUDA_CACHE_PATH=str(self.root/'cache/cuda'),TRITON_CACHE_DIR=str(self.root/'cache/triton'),
                HF_HOME=str(self.root/'cache/huggingface'))
            command=[str(ASSET_ROOT/'.venv/bin/python')]
            if len(devices)>1:
                command+=['-m','torch.distributed.run','--standalone',f'--nproc-per-node={len(devices)}']
            arguments=list(job.arguments)
            if arguments[0] in ('readout','teacher-report','report') and admissions[0]['free_MiB']<25000:
                # The shared gpu01:5 MT101 consumer exhausted its EGL framebuffers at16.
                # Smaller physical waves still evaluate every registered selection/report row.
                arguments[arguments.index('--slots')+1]=str(min(8,int(arguments[arguments.index('--slots')+1])))
            if '--frame-chunk' in arguments:
                # Joint learning retains both optimizers after real R5 collection;
                # its resident EGL workers also need headroom beyond the G-only peak.
                chunk16_headroom=40000 if arguments[0]=='RL' else 28000
                ceiling=int(arguments[arguments.index('--frame-chunk')+1])
                env['EMBER_META_FRAME_CHUNKS']=','.join(str(min(ceiling,16 if a['free_MiB']>=chunk16_headroom else
                    8 if a['free_MiB']>=21000 else 4)) for a in admissions)
            if arguments[0] in ('video-effect','source-reference','control-space'):
                module={'video-effect':'ember.video_effect.run','source-reference':'ember.reference.run','control-space':'ember.control_space.run'}[arguments[0]]
                command+=['-m',module,*arguments[1:]]
                env['EMBER_EPISODE_EVENT_LOG']=str(destination/'episode_events.jsonl')
                env['TMPDIR']='/data1/user/ymdai/ember_ipc'
                if arguments[0] in ('source-reference','control-space'):
                    env.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='8',NUMEXPR_NUM_THREADS='1')
            else:
                command+=[str(self.workspace/'scripts/video_guided_proposal_writer.py'),*arguments]
            # One batch owner can dispatch independent fixed readout panels.
            # Their output roots stay explicit; admission and job billing remain
            # under this owner's master root, rather than silently overriding it.
            if '--root' not in arguments:command+=['--root',str(self.root)]
            command+=['--physical-gpu',str(gpu)]
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
            if self.work_event is None:
                code=subprocess.call(invocation,cwd=self.workspace,env=child_env,stdout=log,stderr=subprocess.STDOUT)
            else:
                child=subprocess.Popen(invocation,cwd=self.workspace,env=child_env,stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                callback_error=None
                try:
                    for line in child.stdout:
                        log.write(line);log.flush()
                        if line.startswith('EMBER_WORK_EVENT '):
                            self.work_event(job.identity,json.loads(line[len('EMBER_WORK_EVENT '):]),child.stdin)
                except BaseException as error:
                    callback_error=repr(error)
                finally:
                    child.stdin.close()
                    for line in child.stdout:log.write(line);log.flush()
                    code=child.wait();child.stdout.close()
                receipt['child_exit_code']=code
                if callback_error:
                    receipt['coordinator_callback_error']=callback_error
                    code=code or 1
        elapsed=time.time()-start
        return self._settle(job, devices, destination, receipt, code, elapsed)

    def _settle(self, job, devices, destination, receipt, code, elapsed):
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

    def adopt(self, job, device, handoff):
        """Keep a running child untouched while replacing its stopped CPU coordinator."""
        destination = Path(handoff['destination'])
        receipt = json.loads((destination/'launch.json').read_text())
        parent, launcher = handoff['supervisor'], handoff['launcher']
        descriptor = os.pidfd_open(launcher['PID'])
        try:
            old = process_identity(parent['PID'])
            child = process_identity(launcher['PID'])
            if (old['start_ticks'] != parent['start_ticks'] or old['state'] != 'T'
                    or child['start_ticks'] != launcher['start_ticks']
                    or child['parent_PID'] != old['PID'] or receipt['job'] != job.identity
                    or (receipt['admission'][0]['node'], receipt['admission'][0]['physical_gpu']) != device
                    or (destination/'exit.json').exists()):
                raise ValueError('adoption requires the identified stopped owner and its original live job')
            # The stopped parent cannot reap this child or dispatch its old pending jobs.
            select.select([descriptor], [], [])
            child = process_identity(launcher['PID'])
            old = process_identity(parent['PID'])
            if (child['state'] != 'Z' or child['start_ticks'] != launcher['start_ticks']
                    or old['state'] != 'T' or old['start_ticks'] != parent['start_ticks']):
                raise ValueError('original child exit status was not preserved during handoff')
            code = os.waitstatus_to_exitcode(child['wait_status'])
            os.kill(parent['PID'], signal.SIGKILL)  # Only the retired CPU owner; never its process group.
            receipt['coordinator_adoption'] = dict(supervisor=parent, launcher=launcher,
                replacement_commit=self.code, replacement_workspace=str(self.workspace),
                original_wait_status=child['wait_status'], original_child_uninterrupted=True,
                retired_CPU_owner_only=True)
            start = datetime.datetime.fromisoformat(receipt['start_utc']).timestamp()
            return self._settle(job, [device], destination, receipt, code, time.time()-start)
        finally:
            os.close(descriptor)
            with self.lock:
                self.running.pop(device, None)

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

    def run(self, jobs, *, deferred_devices=None, on_completion=None, adopted_jobs=None):
        pending={job.identity:job for job in jobs};done=set();omitted=set();checked=set();failures=[]
        external=set().union(*(job.dependencies for job in jobs))-pending.keys()
        deferred=dict(deferred_devices or {})
        adopted=dict(adopted_jobs or {})
        if not set(adopted)<=pending.keys():raise ValueError('adopted jobs must belong to this finite batch')
        if not set(deferred)<=set(self.devices):raise ValueError('deferred devices must belong to this batch')
        for identity in list(pending):
            file=self.root/'jobs'/identity/'latest.json'
            if file.exists() and read_exit(file)==0:
                done.add(identity);del pending[identity]
        if on_completion:on_completion(done)
        with ThreadPoolExecutor(max_workers=len(self.devices)+1) as executor:
            futures={};free=[d for d in self.devices if d not in deferred]
            for identity, handoff in adopted.items():
                job=pending.pop(identity)
                device=tuple(handoff['device'])
                free.remove(device)
                launch=json.loads((Path(handoff['destination'])/'launch.json').read_text())
                self.running[device]=dict(job=identity,estimate_seconds=job.estimate_seconds,
                    start=datetime.datetime.fromisoformat(launch['start_utc']).timestamp())
                futures[executor.submit(self.adopt,job,device,handoff)]=device
            if external:futures[executor.submit(dependency_completion,self.root,external)]=None
            while pending or futures:
                self._release_deferred_devices(deferred,done,free)
                self._dispatch_ready(executor,pending,free,done|omitted,failures,futures)
                if not futures:
                    if failures:break
                    raise RuntimeError('pending jobs have missing dependencies')
                completed,_=wait(futures,return_when=FIRST_COMPLETED)
                self._consume_completions(completed,futures,executor,done,external,free,failures)
                if on_completion:on_completion(done)
            if failures:
                raise RuntimeError('batch engineering/budget failure: '+ '; '.join(map(str,failures)))
        return sorted(done)


def read_exit(path):
    return json.loads(Path(path).read_text())['exit_code']


def shared_learning_job(root,contract,stage,shared_gpus,stop,slots):
    """One shared current-theta batch owner, with an explicit recorded physical topology."""
    if shared_gpus is None or not 1<=shared_gpus<=4:raise ValueError('shared physical world size needs actual throughput selection')
    if stop not in contract['updates'].get(stage,[]):raise ValueError('formal shared update not preregistered')
    chunk=contract.get('physical_execution',{}).get(f'{stage}_frame_chunk_ceiling',16)
    arguments=[stage,'--stop',str(stop),'--slots',str(slots),'--frame-chunk',str(chunk)]
    if stage=='local':arguments+=['--checkpoint',contract['source_checkpoint']]
    if stage=='RL':arguments+=['--checkpoint',contract['initial_writer_checkpoint']]
    nodes=contract['updates'][stage]
    previous=[u for u in nodes if u<=stop and (root/'checkpoints'/stage/f'update_{u:08d}/manifest.json').exists()
        and json.loads((root/'checkpoints'/stage/f'update_{u:08d}/manifest.json').read_text()).get('complete')]
    if stage=='RL':
        previous=[saved['next_update']
            for p in (root/'checkpoints/RL').glob('update_*/manifest.json')
            if (saved:=json.loads(p.read_text())).get('complete') and saved.get('stage')=='RL'
            and saved['next_update']<=stop]
    if previous:arguments+=['--resume',str(root/'checkpoints'/stage/f'update_{max(previous):08d}')]
    seconds=contract.get('stage_estimate_seconds',{}).get(f'{stage}_{stop}',0.)
    return [Job(f'{stage}_{stop:08d}',arguments,estimate_seconds=seconds,gpu_count=shared_gpus)]


def stage_jobs(root,stage,*,microbatch=28,slots=16,teacher_devices=(),estimates=None,shared_gpus=None,stop=None,
               physical_workers=None,condition_batch=None,video_arm='correct'):
    """Registered selector stages reuse one dynamic queue and one shared update owner."""
    root=Path(root);contract=json.loads((root/'run_contract.json').read_text())
    if stage in ('local','RL'):return shared_learning_job(root,contract,stage,shared_gpus,stop,slots)
    if stage not in ('student-data','formal') or not physical_workers or not condition_batch:
        raise ValueError('selector data/formal requires actual measured workers/group size')
    from .evaluation import student_queue,formal_queue
    if stage=='student-data':
        student_queue(root);checkpoint=contract['readout_checkpoint']
    else:
        checkpoint=contract['formal_report']['checkpoint'];formal_queue(root,workers=physical_workers,condition_batch=condition_batch)
    return [Job(f'{stage}_worker_{i:02d}',[stage,'--checkpoint',checkpoint,'--slots',str(slots),
        '--worker-id',f'{stage}_worker_{i:02d}','--condition-batch',str(condition_batch),'--video-arm',video_arm],
        estimate_seconds=3600,required_free_MiB=24000) for i in range(physical_workers)]
