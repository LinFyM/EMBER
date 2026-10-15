"""Two fixed video checkpoints share one physical interaction budget."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import datetime
import json
import os
import sys
import time

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.pi05_eval_contract import policy_noise_seed
from ember.pi05_eval_queue import EvaluationShard, initialize_queue, complete_job, fail_job
from ember.proposal_writer.batch import Batch, Job
from ember.reference.coordinator import Scheduler
from ember.reference.accounting import reserve, finish, summary, host_resources
from ember.reference.analysis import pairing_valid
from ember.reference.run import close_runner, event
from .contract import TASKS

UPDATES = (900, 1800)
VALIDATION = (3, 6, 11, 16, 23, 26, 31, 39)


def phase_root(root, update):
    if update not in UPDATES:raise ValueError('only the two registered video checkpoints may execute')
    return Path(root)/'evaluation'/f'update_{update:08d}'


def queue_path(root):return Path(root)/'readout.sqlite'


def inputs(root):
    root=Path(root);run=read_json(root/'run_contract.json')
    if run.get('phase')!='video_initial' or run['schema']!='ember_source_video_initial_r8_seed7_v1':
        raise ValueError('only the fresh video-initial phase is active')
    bank={int(r['task']):r for r in read_json(root/'bank.json')['tasks']}
    tasks={int(r['global_task_id']):r for r in run['environment']['tasks']}
    if set(bank)!=set(TASKS) or set(tasks)!=set(TASKS)|set(VALIDATION):
        raise ValueError('readout requires exactly frozen36 bank plus validation8 metadata')
    return run,bank,tasks,Path(run['reference_root'])


def panels(root):
    result=read_json(Path(root)/'panels.json')
    from ember.expert_manifold.video_schedule import condition_demo_index
    _,_,tasks,_=inputs(root)
    for role,count,scope,n in (('seen',576,set(TASKS),16),('validation',400,set(VALIDATION),50)):
        rows=result[role];keys={(r['task'],r['init_state_id']) for r in rows}
        if len(rows)!=count or len(keys)!=count or {r['task'] for r in rows}!=scope:
            raise ValueError('registered video panel coverage changed')
        for task in scope:
            selected=[r for r in rows if r['task']==task];spec=tasks[task]
            if {r['init_state_id'] for r in selected}!=set(range(n)) or len({r['teacher_demo'] for r in selected})!=n:
                raise ValueError('teaching must be unique within the fixed per-task panel')
            for row in selected:
                demo=condition_demo_index(7,spec['suite'],spec['task_id'],row['init_state_id'],
                    condition='correct',demo_count=50,sampling_mode='without_replacement')
                if row['teacher_demo']!=demo or row['noise_root']!=(7 if role=='validation' else 202610151):
                    raise ValueError('official teacher/native RNG schedule changed')
                if role=='validation' and not row.get('registered_scene'):
                    raise ValueError('held readout requires the frozen MT153 scene')
    return result


def generated_path(root, update, row):
    return Path(root)/'generated'/f'update_{update:08d}'/row['role']/f'task_{row["task"]:04d}'/f'state_{row["init_state_id"]:03d}.safetensors'


def row_path(root, update, row):
    return phase_root(root,update)/'readout'/row['role']/f'task_{row["task"]:04d}'/f'state_{row["init_state_id"]:03d}.json'


def parameter_ref(update, row):
    return f'video_initial_u{update:08d}_{row["role"]}_t{row["task"]:04d}_s{row["init_state_id"]:03d}'


def validate_case(row, task, condition):
    init=condition['init_state_id'];noise=condition['noise_root']
    expected=dict(suite=task['suite'],task_id=task['task_id'],global_task_id=task['global_task_id'],
        language=task['language'],init_state_id=init,env_seed=noise,policy_seed_root=noise,
        noise_root=noise,settling_steps=10,restored_control_steps=0,full_native_flow_steps=10,
        executed_chunk=5,source_frozen=True,MT_loaded=False)
    if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('fixed video case execution contract changed')
    steps=row.get('steps',0)
    if (type(row.get('success')) is not bool or not 0<steps<=task['horizon']
            or (not row['success'] and steps!=task['horizon']) or row.get('model_failure')
            or row.get('budget_truncated') or row.get('failure_type') or row.get('environment_steps')!=steps+10):
        raise ValueError('incomplete or engineering-failed episode is not a valid scientific row')
    noises=[policy_noise_seed(noise,task['suite'],task['task_id'],init,i) for i in range((steps+4)//5)]
    if row.get('policy_noise_seeds')!=noises or row.get('first_noise_seed')!=noises[0]:
        raise ValueError('native absolute noise stream changed')
    if condition.get('registered_scene'):
        expected_name=f'{task["suite"]}_task_{task["task_id"]:02d}_state_{init:03d}.npz'
        scene=row.get('scene_reference',{})
        if Path(scene.get('path','')).parent!=Path(condition['registered_scene']['root']) or Path(scene.get('path','')).name!=expected_name:
            raise ValueError('held scene does not match MT153 registered scene')


def saved_case(root, update, condition, task, run=None):
    run=inputs(root)[0] if run is None else run
    row=read_json(row_path(root,update,condition));validate_case(row,task,condition)
    expected=dict(phase='video_initial',reader_update=update,role=condition['role'],
        teacher_demo=condition['teacher_demo'],LoRA=str(generated_path(root,update,condition)),
        basis_checkpoint=run['frozen_basis'],actual_parameter_ref=parameter_ref(update,condition))
    if any(row.get(k)!=v for k,v in expected.items()) or row.get('physical_reuse'):
        raise ValueError('video readout must use its unique condition LoRA and a real new episode')
    return row


def cases(args, runtime, work, task, runner):
    from safetensors.torch import load_file
    from ember.lora import validate_lora_state
    run=inputs(args.root)[0];conditions=work['rows']
    missing=[r for r in conditions if not row_path(args.root,args.update,r).exists()]
    states={r['init_state_id']:load_file(str(generated_path(args.root,args.update,r)),device=str(runtime.device)) for r in missing}
    for state in states.values():validate_lora_state(state,runtime.lora)
    by_init={r['init_state_id']:r for r in conditions}
    def requests():
        for row in missing:
            attempt=reserve(args.root,parameter_ref(args.update,row),task['horizon'],limits=run['limits'])
            yield dict(task=task,state=states[row['init_state_id']],parameter_ref=parameter_ref(args.update,row),
                state_id=row['init_state_id'],noise_root=row['noise_root'],episode_id=attempt,
                registered_scene=row.get('registered_scene'),capture_history=False,capture_hidden=False,
                capture_phi=False,capture_rgb=False,uses_teaching=False)
    before=deepcopy(runner.components);tick=time.monotonic();peak=None
    for output in runner.run(requests()) if missing else ():
        if peak is None:
            peak=host_resources();peak['environment_PIDs']=[p.pid for p in runner.environments.processes if p is not None]
        row=output['row'];condition=by_init[output['request']['state_id']];first=output['history'].records[0]
        row.update(global_task_id=task['global_task_id'],logical_arm='video_initial',phase='video_initial',
            reader_update=args.update,role=condition['role'],teacher_demo=condition['teacher_demo'],
            LoRA=str(generated_path(args.root,args.update,condition)),basis_checkpoint=run['frozen_basis'],
            noise_root=condition['noise_root'],first_noise_seed=first['noise_seed'],
            first_native_normalized_actions=first['normalized_actions'].tolist(),full_native_flow_steps=10,
            executed_chunk=5,source_frozen=True,MT_loaded=False)
        validate_case(row,task,condition)
        destination=row_path(args.root,args.update,condition);write_json_atomic(destination,row)
        finish(args.root,output['request']['episode_id'],row,destination)
    elapsed=time.monotonic()-tick
    components={k:v-before[k] for k,v in runner.components.items() if k!='physical_batch_histogram'}
    components['physical_batch_histogram']={k:v-before['physical_batch_histogram'].get(k,0) for k,v in runner.components['physical_batch_histogram'].items()}
    return [saved_case(args.root,args.update,r,task,run) for r in conditions],dict(new_episodes=len(missing),
        seconds=elapsed,episodes_per_second=len(missing)/max(elapsed,1e-6),runner=components,
        peak_environment_resources=peak,after_shard_resources=host_resources())


class ReadoutScheduler(Scheduler):
    def evaluate_selection(self, statuses):pass
    def eligible(self, statuses):return [i for i in self.panel if statuses[i]=='pending']
    def choose_work(self, worker, ready):
        self.pending_ready_count=len(ready)
        return super().choose_work(worker,ready)
    def slot_candidates(self, worker, work, budget):
        occupied=budget['occupied'];others=sum(v for k,v in occupied.items() if k!=worker)
        peers=min(sum(k!=worker and not v for k,v in occupied.items()),max(0,getattr(self,'pending_ready_count',1)-1))
        floor=8 if budget['slot_cap']>=8*budget['workers'] else 4
        peers=min(peers,max(0,(budget['slot_cap']-others)//floor-1))
        maximum=getattr(self,'worker_max_slots',{}).get(worker,8)
        return tuple(s for s in (16,8,4) if s<=maximum and others+min(s,len(work['states']))+peers*floor<=budget['slot_cap'])
    def event(self,worker,value,stream):
        if 'max_slots' in value:
            if not hasattr(self,'worker_max_slots'):self.worker_max_slots={}
            self.worker_max_slots[worker]=value['max_slots']
        return super().event(worker,value,stream)

    def terminal(self, statuses):return all(statuses[i]=='complete' for i in self.panel)
    def stop_idle(self, statuses):return self.failed or all(statuses[i]!='pending' for i in self.panel)


def publish(root, identity, token, worker, work, output, *, seconds=0., peak_allocated_MiB=0., peak_reserved_MiB=0.):
    rows=output['rows'];path=root/'shards'/f'{identity}.json'
    write_json_atomic(path,dict(identity=identity,work=work,output=output,complete=True,seconds=seconds,
        peak_allocated_MiB=peak_allocated_MiB,peak_reserved_MiB=peak_reserved_MiB))
    complete_job(queue_path(root),job_id=identity,worker_id=worker,claim_token=token,
        rows_path=str(path.relative_to(root)),rows_bytes=path.stat().st_size,row_count=len(rows),successes=sum(r['success'] for r in rows))


def prepare_queue(args):
    run,_,tasks,reference_root=inputs(args.root);folder=phase_root(args.root,args.update);folder.mkdir(parents=True,exist_ok=True)
    panel={};ordinal=0;shards=[]
    for role,rows in panels(args.root).items():
        for task in sorted({r['task'] for r in rows}):
            values=sorted((dict(r,role=role) for r in rows if r['task']==task),key=lambda r:r['init_state_id'])
            for first in range(0,len(values),16):
                part=values[first:first+16];identity=f'{role}_t{task:04d}_s{first:02d}'
                work=dict(kind='readout',role=role,task=task,states=[r['init_state_id'] for r in part],rows=part,cost=tasks[task]['horizon']*len(part))
                panel[identity]=work
                shards.append(EvaluationShard(identity,ordinal,tasks[task]['suite'],task,tasks[task]['horizon'],tuple(work['states']),work['cost']));ordinal+=1
    write_json_atomic(folder/'queue_plan.json',dict(reader_update=args.update,panel=panel,all_episodes_new=True,master_budget=str(args.root)))
    initialize_queue(queue_path(folder),shards,contract_reference=str(args.root/'run_contract.json'),
        recover_claims=args.resume,retry_failed=args.resume)
    return run,tasks,reference_root,panel


def feature_chunk(root,hostname,available_MiB,requested=0):
    if requested>0:return requested
    rows=[]
    for path in (Path(root)/'features').glob('worker_*.json'):
        value=read_json(path)
        if value['resources']['hostname']!=hostname:continue
        rows.extend(m for m in value['measurements'] if not m['OOM_attempts']
            and m['peak_reserved_MiB']+512<=available_MiB)
    if not rows:raise ValueError('no completed useful source-feature chunk fits current live headroom')
    return max(rows,key=lambda m:m['frames_per_second'])['requested_frame_chunk']


def worker(args):
    import torch
    from ember.pi05_source_setup import initialize_distributed
    from ember.pi05_assets import prepare_libero_config, configure_libero_runtime_assets
    from ember.writer.practice import EpisodeRunner
    from .data import make_runtime
    from .analysis import materialize_shard
    context=initialize_distributed(require_numa=True,defer_process_group=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    run,_,tasks,_=inputs(args.root);environment=run['environment'];folder=phase_root(args.root,args.update)
    os.environ.update(EMBER_LIBERO_ASSETS_ROOT=environment['libero_paths']['assets'],MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    prepare_libero_config(folder/'libero_config');configure_libero_runtime_assets(Path(environment['libero_paths']['assets']))
    runtime=make_runtime(args.root,context.device,include_validation=True)
    runtime.episode_event_log=os.environ.get('EMBER_EPISODE_EVENT_LOG');runner=None
    available=(torch.cuda.mem_get_info()[0]+torch.cuda.memory_reserved())/2**20
    args.frame_chunk=feature_chunk(args.root,os.uname().nodename,available,args.frame_chunk)
    try:
        while True:
            # Reuse measured16-slot peak plus renderer margin; no permanent node cap.
            available=(torch.cuda.mem_get_info()[0]+torch.cuda.memory_reserved())/2**20
            event(kind='ready',worker=args.worker_id,resources=host_resources(),
                max_slots=16 if available>=34000 else 8,available_workload_MiB=available,frame_chunk=args.frame_chunk)
            while True:
                line=sys.stdin.readline();message=json.loads(line) if line else dict(command='stop')
                if message['command']!='idle':break
                runner=close_runner(runner,args.worker_id)
            if message['command']=='stop':break
            work=message['work'];identity=message['identity'];slots=message['slots'];tick=time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            try:
                # Previous runner.run has drained every request. Keep its idle slots
                # across immediately ready shards; explicit idle/stop closes them.
                generation=materialize_shard(args,runtime,work['rows'])
                if runner is not None and slots<runner.slots:runner=close_runner(runner,args.worker_id)
                if runner is None:runner=EpisodeRunner(runtime,environment,args.physical_gpu,slots=slots)
                runner.slots=slots
                rows,resources=cases(args,runtime,work,tasks[work['task']],runner);resources['materialization']=generation
                publish(folder,identity,message['claim_token'],args.worker_id,work,dict(rows=rows,resources=resources),
                    seconds=time.monotonic()-tick,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
                    peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2)
            except BaseException as error:
                fail_job(queue_path(folder),job_id=identity,worker_id=args.worker_id,claim_token=message['claim_token'],error=repr(error))
                event(kind='failed',worker=args.worker_id,identity=identity,error=repr(error));raise
    finally:close_runner(runner,args.worker_id);runtime.videos.close();runtime.execution.close()


def owner(args,delayed_devices=None):
    _,tasks,reference_root,panel=prepare_queue(args);folder=phase_root(args.root,args.update)
    immediate=[(v.split(':')[0],int(v.split(':')[1])) for v in args.devices]
    delayed=list(delayed_devices or [])
    devices=immediate+[device for device,_ in delayed]
    release={device:ready for device,ready in delayed}
    if len(set(devices))!=len(devices):raise ValueError('readout device may only have one actual worker')
    worker_nodes={f'video_u{args.update:08d}_{i:02d}':node for i,(node,_) in enumerate(devices)}
    scheduler=ReadoutScheduler(folder,panel,len(devices),worker_nodes=worker_nodes,queue_path_fn=queue_path,evaluation_reference_root=reference_root)
    write_json_atomic(folder/'readout_execution.json',scheduler.evaluation_reference)
    tick=time.monotonic();failure=None
    if not scheduler.terminal(scheduler.statuses()):
        if not devices:raise ValueError('pending readout requires explicit live-admitted devices')
        required=getattr(args,'readout_required_free_MiB',26000)
        batch=Batch(Path.cwd(),devices,root=args.root,required_free_MiB=required,work_event=scheduler.event)
        jobs=[Job(worker,['control-space','readout-worker','--root',str(args.root),'--update',str(args.update),
            '--worker-id',worker,'--slots','16','--frame-chunk',str(getattr(args,'frame_chunk',8))],
            eligible_devices=(device,),estimate_seconds=3600,required_free_MiB=required) for worker,device in zip(worker_nodes,devices)]
        def execute(job,device):
            # A dependency's actual owner exit releases the GPU, with no polling
            # or resident GPU process while the event is pending.
            if device in release:
                release[device].wait()
                if scheduler.failed or scheduler.terminal(scheduler.statuses()):return job.identity
            return batch.execute(job,device)
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures=[executor.submit(execute,job,device) for job,device in zip(jobs,devices)]
            for future in as_completed(futures):
                try:future.result()
                except BaseException as error:
                    failure=repr(error) if failure is None else failure+'; '+repr(error)
                    with scheduler.lock:scheduler.failed=True;scheduler.dispatch()
    all_rows=[dict(row,role=role) for role,rows in panels(args.root).items() for row in rows]
    missing=[str(row_path(args.root,args.update,r)) for r in all_rows if not row_path(args.root,args.update,r).is_file()]
    result=dict(complete=scheduler.terminal(scheduler.statuses()) and not missing and failure is None,
        queue=scheduler.statuses(),missing_rows=missing,logical_rows=976-len(missing),identity_reused_rows=0,
        reader_update=args.update,interactions=summary(args.root),owner_failure=failure,
        owner_elapsed_seconds=time.monotonic()-tick,end_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json_atomic(folder/'readout_completion.json',result)
    if failure or not result['complete']:raise RuntimeError(failure or 'incomplete video readout')
    return result
