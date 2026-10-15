"""One fixed-LoRA readout queue, with immutable reference rows and full costs."""
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
from ember.pi05_eval_queue import EvaluationShard, initialize_queue, claim_next, complete_job, fail_job
from ember.proposal_writer.batch import Batch, Job
from ember.reference.coordinator import Scheduler
from ember.reference.accounting import reserve, finish, summary, host_resources
from ember.reference.analysis import pairing_valid
from ember.reference.run import close_runner, event

from .contract import TASKS, LIMITS, REFERENCE_ROOT, queue_path

NOISE_ROOT = 202610151
STATES = tuple(range(16))


def inputs(root):
    run=read_json(root/'run_contract.json');bank=read_json(root/'bank.json')
    references={int(row['task']):row for row in bank['tasks']}
    tasks={int(row['global_task_id']):row for row in run['environment']['tasks']}
    if (len(bank['tasks'])!=len(TASKS) or set(references)!=set(TASKS) or set(tasks)!=set(TASKS)
            or sum(row['selected_update']==0 for row in references.values())!=5):
        raise ValueError('readout requires the complete frozen 36-task bank and five identity references')
    return run,references,tasks,Path(run.get('reference_root',REFERENCE_ROOT))


def validate_case(row, task, init):
    expected=dict(suite=task['suite'],task_id=task['task_id'],global_task_id=task['global_task_id'],
        language=task['language'],init_state_id=init,env_seed=NOISE_ROOT,policy_seed_root=NOISE_ROOT,
        noise_root=NOISE_ROOT,settling_steps=10,restored_control_steps=0,
        full_native_flow_steps=10,executed_chunk=5,source_frozen=True,MT_loaded=False)
    if any(row.get(key)!=value for key,value in expected.items()):
        raise ValueError('readout/reference case changed its task, official execution or paired RNG')
    steps=row.get('steps',0)
    if (type(row.get('success')) is not bool or not 0<steps<=task['horizon']
            or (not row['success'] and steps!=task['horizon']) or row.get('model_failure')
            or row.get('budget_truncated') or row.get('failure_type')
            or row.get('environment_steps')!=steps+10):
        raise ValueError('incomplete or engineering-failed episode cannot become a valid readout row')
    noises=[policy_noise_seed(NOISE_ROOT,task['suite'],task['task_id'],init,i) for i in range((steps+4)//5)]
    if row.get('policy_noise_seeds')!=noises or row.get('first_noise_seed')!=noises[0]:
        raise ValueError('readout/reference lost the absolute native noise schedule')


def reference_case(reference_root, task, bank, init):
    base=reference_root/'selection'/f'task_{task["global_task_id"]:04d}'
    source=base/'update_00000000'/f'state_{init:03d}.json'
    selected=base/f'update_{bank["selected_update"]:08d}'/f'state_{init:03d}.json'
    rows=[read_json(path) for path in (source,selected)]
    for row,update in zip(rows,(0,bank['selected_update']),strict=True):
        validate_case(row,task,init)
        if row['update']!=update:raise ValueError('reference row does not belong to its frozen parameter point')
    if not pairing_valid(rows[:1],rows[1:]):raise ValueError('frozen source/selected reference cases are not paired')
    return rows[0],dict(source_row=str(source),reference_row=str(selected))


def saved_case(path, task, bank, reference_root, generated):
    row=read_json(path);init=int(path.stem.split('_')[-1]);validate_case(row,task,init)
    source,provenance=reference_case(reference_root,task,bank,init)
    expected=dict(LoRA=str(generated),reconstruction_update=144,reference_update=bank['selected_update'],**provenance)
    if any(row.get(key)!=value for key,value in expected.items()) or not pairing_valid([source],[row]):
        raise ValueError('saved readout row has a different frozen LoRA or paired initial observation')
    if bank['selected_update']==0 and row.get('physical_reuse')!=provenance['source_row']:
        raise ValueError('identity readout must identify its original source execution')
    if bank['selected_update']!=0 and row.get('actual_parameter_ref')!=f'control144_t{task["global_task_id"]:04d}':
        raise ValueError('physical readout changed its actual complete parameter identity')
    return row


def identity_cases(root, work, task, bank, reference_root):
    folder=root/'readout'/f'task_{work["task"]:04d}';folder.mkdir(parents=True,exist_ok=True)
    generated=root/'generated'/f'task_{work["task"]:04d}.safetensors';rows=[]
    for init in work['states']:
        path=folder/f'state_{init:03d}.json'
        if not path.exists():
            source,provenance=reference_case(reference_root,task,bank,init)
            row=dict(source,logical_arm='reconstruction',LoRA=str(generated),reconstruction_update=144,
                reference_update=0,physical_reuse=provenance['source_row'],reused_execution_LoRA=source['LoRA'],**provenance)
            write_json_atomic(path,row)
        rows.append(saved_case(path,task,bank,reference_root,generated))
    return rows,dict(new_episodes=0,identity_source_reuse=True)


def cases(args, runtime, work, task, bank, reference_root, runner):
    from safetensors.torch import load_file
    from ember.lora import validate_lora_state
    if bank['selected_update']==0:return identity_cases(args.root,work,task,bank,reference_root)
    folder=args.root/'readout'/f'task_{work["task"]:04d}';folder.mkdir(parents=True,exist_ok=True)
    generated=args.root/'generated'/f'task_{work["task"]:04d}.safetensors'
    missing=[init for init in work['states'] if not (folder/f'state_{init:03d}.json').exists()]
    references={init:reference_case(reference_root,task,bank,init) for init in work['states']}
    state=load_file(str(generated),device=str(runtime.device)) if missing else None
    if state is not None:validate_lora_state(state,runtime.lora)
    def requests():
        for init in missing:
            attempt=reserve(args.root,f'reconstruction_t{work["task"]:04d}_s{init:03d}',task['horizon'],limits=LIMITS)
            yield dict(task=task,state=state,parameter_ref=f'control144_t{work["task"]:04d}',state_id=init,
                noise_root=NOISE_ROOT,episode_id=attempt,capture_history=False,capture_hidden=False,
                capture_phi=False,capture_rgb=False,uses_teaching=False)
    before=deepcopy(runner.components);started=time.monotonic();peak=None
    for result in runner.run(requests()) if missing else ():
        if peak is None:
            peak=host_resources();peak['environment_PIDs']=[p.pid for p in runner.environments.processes if p is not None]
        row=result['row'];init=result['request']['state_id'];first=result['history'].records[0]
        row.update(global_task_id=work['task'],logical_arm='reconstruction',LoRA=str(generated),
            reconstruction_update=144,reference_update=bank['selected_update'],noise_root=NOISE_ROOT,
            first_noise_seed=first['noise_seed'],first_native_normalized_actions=first['normalized_actions'].tolist(),
            full_native_flow_steps=10,executed_chunk=5,source_frozen=True,MT_loaded=False,
            producer_parameter_ref=result['request']['parameter_ref'],**references[init][1])
        validate_case(row,task,init)
        if not pairing_valid([references[init][0]],[row]):raise ValueError('readout changed the paired initial observation')
        path=folder/f'state_{init:03d}.json';write_json_atomic(path,row)
        finish(args.root,result['request']['episode_id'],row,path)
    elapsed=time.monotonic()-started
    components={key:value-before[key] for key,value in runner.components.items() if key!='physical_batch_histogram'}
    components['physical_batch_histogram']={key:value-before['physical_batch_histogram'].get(key,0)
        for key,value in runner.components['physical_batch_histogram'].items()}
    return [saved_case(folder/f'state_{init:03d}.json',task,bank,reference_root,generated) for init in work['states']],dict(
        new_episodes=len(missing),seconds=elapsed,episodes_per_second=len(missing)/max(elapsed,1e-6),runner=components,
        peak_environment_resources=peak,after_shard_resources=host_resources())


class ReadoutScheduler(Scheduler):
    def evaluate_selection(self, statuses):pass

    def eligible(self, statuses):
        return [identity for identity in self.panel if statuses[identity]=='pending']

    def terminal(self, statuses):
        return all(statuses[identity]=='complete' for identity in self.panel)


def publish(root, identity, token, worker, work, output, *, seconds=0., peak_allocated_MiB=0., peak_reserved_MiB=0.):
    rows=output['rows'];path=root/'shards'/f'{identity}.json'
    write_json_atomic(path,dict(identity=identity,work=work,output=output,complete=True,seconds=seconds,
        peak_allocated_MiB=peak_allocated_MiB,peak_reserved_MiB=peak_reserved_MiB))
    complete_job(queue_path(root),job_id=identity,worker_id=worker,claim_token=token,
        rows_path=str(path.relative_to(root)),rows_bytes=path.stat().st_size,row_count=len(rows),
        successes=sum(row['success'] for row in rows))


def prepare_queue(args):
    run,bank,tasks,reference_root=inputs(args.root)
    panel={f'control_t{task:04d}_s00':dict(kind='readout',task=task,states=list(STATES),cost=tasks[task]['horizon']*16)
        for task in TASKS}
    shards=[EvaluationShard(identity,index,tasks[work['task']]['suite'],work['task'],tasks[work['task']]['horizon'],
        tuple(STATES),work['cost']) for index,(identity,work) in enumerate(panel.items())]
    for task in TASKS:
        if not (args.root/'generated'/f'task_{task:04d}.safetensors').is_file():
            raise ValueError('readout cannot precede all 36 frozen generated LoRAs')
    initialize_queue(queue_path(args.root),shards,contract_reference=str(args.root/'run_contract.json'),
        recover_claims=args.resume,retry_failed=args.resume)
    identity_ids=[identity for identity,work in panel.items() if bank[work['task']]['selected_update']==0]
    while (claim:=claim_next(queue_path(args.root),worker_id='identity_row_reuse',eligible_job_ids=identity_ids)) is not None:
        work=panel[claim.shard.job_id]
        rows,resources=identity_cases(args.root,work,tasks[work['task']],bank[work['task']],reference_root)
        publish(args.root,claim.shard.job_id,claim.claim_token,'identity_row_reuse',work,dict(rows=rows,resources=resources))
    return run,bank,tasks,reference_root,panel


def worker(args):
    import torch
    from ember.pi05_source_setup import initialize_distributed
    from ember.pi05_assets import prepare_libero_config, configure_libero_runtime_assets
    from ember.reference.training import Runtime
    from ember.writer.practice import EpisodeRunner
    context=initialize_distributed(require_numa=True,defer_process_group=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    run,bank,tasks,reference_root=inputs(args.root);environment=run['environment']
    os.environ.update(EMBER_LIBERO_ASSETS_ROOT=environment['libero_paths']['assets'],MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    prepare_libero_config(args.root/'libero_config');configure_libero_runtime_assets(Path(environment['libero_paths']['assets']))
    runtime=Runtime(args.root,context.device);runtime.episode_event_log=os.environ.get('EMBER_EPISODE_EVENT_LOG');runner=None
    try:
        while True:
            event(kind='ready',worker=args.worker_id,resources=host_resources())
            while True:
                line=sys.stdin.readline();message=json.loads(line) if line else dict(command='stop')
                if message['command']!='idle':break
                runner=close_runner(runner,args.worker_id)
            if message['command']=='stop':break
            work=message['work'];identity=message['identity'];slots=message['slots'];tick=time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            try:
                if runner is not None and slots<runner.slots:runner=close_runner(runner,args.worker_id)
                if runner is None:runner=EpisodeRunner(runtime,environment,args.physical_gpu,slots=slots)
                runner.slots=slots
                rows,resources=cases(args,runtime,work,tasks[work['task']],bank[work['task']],reference_root,runner)
                publish(args.root,identity,message['claim_token'],args.worker_id,work,dict(rows=rows,resources=resources),seconds=time.monotonic()-tick,
                    peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
                    peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2)
            except BaseException as error:
                fail_job(queue_path(args.root),job_id=identity,worker_id=args.worker_id,claim_token=message['claim_token'],error=repr(error))
                event(kind='failed',worker=args.worker_id,identity=identity,error=repr(error));raise
    finally:
        close_runner(runner,args.worker_id);runtime.execution.close()


def owner(args):
    _,_,tasks,reference_root,panel=prepare_queue(args)
    devices=[(value.split(':')[0],int(value.split(':')[1])) for value in args.devices]
    scheduler=ReadoutScheduler(args.root,panel,len(devices),
        worker_nodes={f'control_readout_{i:02d}':node for i,(node,_) in enumerate(devices)},
        queue_path_fn=queue_path,evaluation_reference_root=reference_root)
    write_json_atomic(args.root/'readout_execution.json',scheduler.evaluation_reference)
    started=time.monotonic();failure=None
    if not scheduler.terminal(scheduler.statuses()):
        if not devices:raise ValueError('pending readout requires explicit live-admitted devices')
        batch=Batch(Path.cwd(),devices,root=args.root,required_free_MiB=34000,work_event=scheduler.event)
        jobs=[Job(f'control_readout_{i:02d}',['control-space','readout-worker','--worker-id',f'control_readout_{i:02d}','--slots','16'],
            eligible_devices=(device,),estimate_seconds=3600,required_free_MiB=34000) for i,device in enumerate(devices)]
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures=[executor.submit(batch.execute,job,device) for job,device in zip(jobs,devices)]
            for future in as_completed(futures):
                try:future.result()
                except BaseException as error:
                    failure=repr(error) if failure is None else failure+'; '+repr(error)
                    with scheduler.lock:scheduler.failed=True;scheduler.dispatch()
    complete=scheduler.terminal(scheduler.statuses())
    missing=[str(args.root/'readout'/f'task_{task:04d}'/f'state_{init:03d}.json') for task in TASKS for init in STATES
        if not (args.root/'readout'/f'task_{task:04d}'/f'state_{init:03d}.json').is_file()]
    result=dict(complete=complete and not missing and failure is None,queue=scheduler.statuses(),missing_rows=missing,
        logical_rows=576-len(missing),identity_reused_rows=80,interactions=summary(args.root),
        owner_failure=failure,owner_elapsed_seconds=time.monotonic()-started,
        end_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json_atomic(args.root/'readout_completion.json',result)
    if failure or not result['complete']:raise RuntimeError(failure or 'readout exited with incomplete rows')
    return result
