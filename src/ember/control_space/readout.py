"""Only the eight frozen complete LoRAs execute; no basis/Reader/H at rollout."""
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,os,sys,time,sqlite3,subprocess
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.pi05_eval_contract import policy_noise_seed
from ember.pi05_eval_queue import EvaluationShard,initialize_queue,complete_job,fail_job
from ember.reference.accounting import reserve,finish,connect,host_resources,summary
from ember.reference.analysis import pairing_valid
from ember.reference.coordinator import Scheduler
from ember.reference.run import close_runner,event
from ember.proposal_writer.batch import Batch,Job
from .contract import SCHEMA,PHASE,TASKS,STATES,NOISE_ROOT,queue_path,CENTER_REFERENCE


def inputs(root):
    root=Path(root);run=read_json(root/'run_contract.json');manifest=read_json(root/'evaluation_manifest.json')
    if run['schema']!=SCHEMA or run['phase']!=PHASE or manifest['basis_checkpoint']!=run['basis_checkpoint']:
        raise ValueError('readout must bind the sole task-excluded fitted basis')
    tasks={t['global_task_id']:t for t in run['environment']['tasks']};entries={e['task']:e for e in manifest['entries']}
    if set(tasks)!=set(TASKS) or set(entries)!=set(TASKS):raise ValueError('only eight registered tasks may execute')
    return run,tasks,entries


def validate_case(row,task,init):
    expected=dict(suite=task['suite'],task_id=task['task_id'],global_task_id=task['global_task_id'],language=task['language'],
        init_state_id=init,env_seed=NOISE_ROOT,policy_seed_root=NOISE_ROOT,noise_root=NOISE_ROOT,settling_steps=10,
        restored_control_steps=0,full_native_flow_steps=10,executed_chunk=5,source_frozen=True,MT_loaded=False)
    if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('official case task/native/RNG contract changed')
    steps=row.get('steps',0)
    if (type(row.get('success')) is not bool or not 0<steps<=task['horizon'] or (not row['success'] and steps!=task['horizon'])
            or row.get('model_failure') or row.get('budget_truncated') or row.get('failure_type')
            or row.get('environment_steps')!=steps+10):raise ValueError('engineering/incomplete row cannot complete a query')
    stream=[policy_noise_seed(NOISE_ROOT,task['suite'],task['task_id'],init,i) for i in range((steps+4)//5)]
    if row.get('policy_noise_seeds')!=stream or row.get('first_noise_seed')!=stream[0]:raise ValueError('absolute native policy noise changed')


def references(entry,task,init):
    rows={a:read_json(entry[a+'_rows'][init]) for a in ('source','reference','U36')}
    for row in rows.values():validate_case(row,task,init)
    if rows['source']['update']!=0 or rows['reference']['update']!=entry['selected_update']:
        raise ValueError('reference/source parameter point changed')
    if rows['reference']['LoRA']!=entry['selected_reference'] or rows['U36']['basis_checkpoint']!=str(CENTER_REFERENCE):
        raise ValueError('original saved reference/U36 identity changed')
    if (rows['U36']['LoRA']!=entry['U36_LoRA'] or rows['U36'].get('phase')!='linear64'
            or rows['U36']['reference_update']!=entry['selected_update']
            or rows['U36']['source_row']!=entry['source_rows'][init]
            or rows['U36']['reference_row']!=entry['reference_rows'][init]):raise ValueError('old U36 row must identify its actual complete controller')
    if not all(pairing_valid([rows['source']],[r]) for r in rows.values()):raise ValueError('original source/reference/U36 initial observations differ')
    return rows


def saved_case(root,entry,task,init):
    path=Path(root)/'readout'/f'task_{task["global_task_id"]:04d}'/f'state_{init:03d}.json'
    row=read_json(path);validate_case(row,task,init)
    if row.get('LoRA')!=entry['path'] or row.get('basis_checkpoint')!=str(Path(root)/'basis/state.pt') or row.get('phase')!=PHASE:
        raise ValueError('new case lost the projected complete controller identity')
    if row.get('actual_parameter_ref')!=entry['path'] or not pairing_valid([references(entry,task,init)['source']],[row]):
        raise ValueError('new projected controller lost the old paired initial observation')
    return row


def cases(args,runtime,work,task,entry,runner):
    from safetensors.torch import load_file
    from ember.lora import validate_lora_state
    root=args.root;folder=root/'readout'/f'task_{work["task"]:04d}';folder.mkdir(parents=True,exist_ok=True);rows={};missing=[]
    for init in work['states']:
        destination=folder/f'state_{init:03d}.json'
        if destination.exists():
            row=saved_case(root,entry,task,init)
            with connect(root) as db:status=db.execute('SELECT status FROM attempts WHERE identity=?',(row['attempt_id'],)).fetchone()
            if status==('reserved_unknown',):finish(root,row['attempt_id'],row,destination)
            elif status!=('complete',):raise ValueError('saved valid row lost its physical ledger')
            rows[init]=row
        else:missing.append(init)
    state=load_file(entry['path'],device=str(runtime.device)) if missing else None
    if state is not None:
        import torch
        validate_lora_state(state,runtime.lora)
        if any(not bool(v.isfinite().all()) for v in state.values()):raise ValueError('nonfinite projected controller')
    prior={i:references(entry,task,i) for i in missing}
    def requests():
        for init in missing:
            attempt=reserve(root,f'projected_t{work["task"]:04d}_s{init:03d}',task['horizon'],limits=read_json(root/'run_contract.json')['limits'])
            yield dict(task=task,state=state,parameter_ref=entry['path'],state_id=init,noise_root=NOISE_ROOT,episode_id=attempt,
                registered_scene=prior[init]['source'].get('registered_scene'),
                capture_history=False,capture_hidden=False,capture_phi=False,capture_rgb=False,uses_teaching=False)
    before=deepcopy(runner.components);start=time.monotonic();peak=None
    for output in runner.run(requests()) if missing else ():
        if peak is None:
            peak=host_resources();peak['environment_PIDs']=[p.pid for p in runner.environments.processes if p is not None]
        req=output['request'];row=output['row'];init=req['state_id'];first=output['history'].records[0]
        row.update(global_task_id=task['global_task_id'],LoRA=entry['path'],phase=PHASE,basis_checkpoint=str(root/'basis/state.pt'),
            attempt_id=req['episode_id'],noise_root=NOISE_ROOT,first_noise_seed=first['noise_seed'],
            first_native_normalized_actions=first['normalized_actions'].tolist(),full_native_flow_steps=10,executed_chunk=5,
            source_frozen=True,MT_loaded=False,producer_parameter_ref=entry['path'])
        validate_case(row,task,init)
        if not pairing_valid([prior[init]['source']],[row]):raise ValueError('projected case changed actual paired initial observation')
        destination=folder/f'state_{init:03d}.json';write_json_atomic(destination,row);finish(root,req['episode_id'],row,destination);rows[init]=row
    elapsed=time.monotonic()-start
    components={k:v-before[k] for k,v in runner.components.items() if k!='physical_batch_histogram'}
    components['physical_batch_histogram']={k:v-before['physical_batch_histogram'].get(k,0) for k,v in runner.components['physical_batch_histogram'].items()}
    return [rows[i] for i in work['states']],dict(new_episodes=len(missing),seconds=elapsed,episodes_per_second=len(missing)/max(elapsed,1e-6),
        peak_environment_resources=peak,after_shard_resources=host_resources(),runner=components)


class ReadoutScheduler(Scheduler):
    def __init__(self,*a,capacities,**kw):super().__init__(*a,**kw);self.capacities=capacities
    def evaluate_selection(self,statuses):pass
    def eligible(self,statuses):return [i for i in self.panel if statuses[i]=='pending']
    def terminal(self,statuses):return all(statuses[i]=='complete' for i in self.panel)
    def stop_idle(self,statuses):return self.failed or all(statuses[i]!='pending' for i in self.panel)
    def slot_candidates(self,worker,work,budget):return tuple(s for s in (16,8,4) if s<=self.capacities[worker])
    def event(self,worker,ev,stream):
        if ev['kind']=='physical_pack':
            with self.lock:self.running_slots[worker]=ev['slots'];self.capacities[worker]=ev['native_slot_capacity']
        super().event(worker,ev,stream)
def physical_slots(logical_rows,project_available_MiB):
    """Only measured native capacities; a logical four-case panel stays whole."""
    for slots,required in ((16,28000),(8,18000),(4,14000)):
        if project_available_MiB>=required:return min(logical_rows,slots)
    raise RuntimeError('live local memory cannot cover the measured four-slot native workload before environment submission')


def panel_runner(runner,runtime,environment,args,logical_rows,limit=None):
    import torch
    from ember.writer.practice import EpisodeRunner
    def available():
        torch.cuda.empty_cache()
        free,_=torch.cuda.mem_get_info(runtime.device)
        return (free+torch.cuda.memory_allocated(runtime.device))/2**20
    headroom=available();slots=min(limit or logical_rows,physical_slots(logical_rows,headroom))
    if runner is not None and slots<runner.slots:
        # Existing own EGL allocations are outside PyTorch. Close them before
        # recalculating the available project share; never credit foreign memory.
        runner=close_runner(runner,args.worker_id)
        headroom=available();slots=min(limit or logical_rows,physical_slots(logical_rows,headroom))
    if runner is None:runner=EpisodeRunner(runtime,environment,args.physical_gpu,slots=slots)
    runner.slots=slots
    return runner,dict(slots=slots,logical_rows=logical_rows,project_available_MiB=headroom,
        native_slot_capacity=physical_slots(16,headroom))



def worker(args):
    import torch
    from ember.pi05_source_setup import initialize_distributed
    from ember.pi05_assets import prepare_libero_config,configure_libero_runtime_assets
    from ember.reference.training import Runtime
    context=initialize_distributed(require_numa=True,defer_process_group=True);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    run,tasks,entries=inputs(args.root);environment=run['environment']
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
            work=message['work'];identity=message['identity'];tick=time.monotonic()
            try:
                runner,admission=panel_runner(runner,runtime,environment,args,len(work['states']),message['slots'])
                event(kind='physical_pack',worker=args.worker_id,identity=identity,**admission,resources=host_resources())
                torch.cuda.reset_peak_memory_stats();rows,resources=cases(args,runtime,work,tasks[work['task']],entries[work['task']],runner)
                receipt=dict(complete=True,work=work,output=dict(rows=rows,resources=resources),seconds=time.monotonic()-tick,
                    slots=runner.slots,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
                path=args.root/'shards'/f'{identity}.json';write_json_atomic(path,receipt)
                complete_job(queue_path(args.root),job_id=identity,worker_id=args.worker_id,claim_token=message['claim_token'],rows_path=str(path.relative_to(args.root)),
                    rows_bytes=path.stat().st_size,row_count=len(rows),successes=sum(r['success'] for r in rows))
                event(kind='useful_readout',worker=args.worker_id,identity=identity,slots=runner.slots,seconds=receipt['seconds'],new_episodes=resources['new_episodes'])
            except BaseException as error:
                fail_job(queue_path(args.root),job_id=identity,worker_id=args.worker_id,claim_token=message['claim_token'],error=repr(error))
                event(kind='failed',worker=args.worker_id,identity=identity,error=repr(error));raise
    finally:close_runner(runner,args.worker_id);runtime.execution.close()


def owner(args):
    from .run import execute_worker,device_pool
    run,tasks,entries=inputs(args.root);panel={f'projected_t{t:04d}':dict(kind='readout',task=t,states=list(STATES),cost=16*(tasks[t]['horizon']+10)) for t in TASKS}
    shards=[EvaluationShard(i,n,tasks[w['task']]['suite'],w['task'],tasks[w['task']]['horizon'],tuple(STATES),w['cost']) for n,(i,w) in enumerate(panel.items())]
    initialize_queue(queue_path(args.root),shards,contract_reference=str(args.root/'run_contract.json'),recover_claims=args.resume,retry_failed=args.resume)
    with sqlite3.connect(queue_path(args.root)) as db:statuses=dict(db.execute('SELECT job_id,status FROM jobs'))
    if all(statuses[i]=='complete' for i in panel):
        write_json_atomic(args.root/'readout_completion.json',dict(complete=True,rows=128,interactions=summary(args.root)));return
    choices,cap=device_pool(args)
    write_json_atomic(args.root/'execution.json',dict(devices=choices,total_cap=cap,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    devices=[(n,g) for n,g,_ in choices];capacities={f'worker_{i:02d}':16 if free>=28000 else 8 for i,(_,_,free) in enumerate(choices)}
    scheduler=ReadoutScheduler(args.root,panel,len(devices),worker_nodes={f'worker_{i:02d}':n for i,(n,_) in enumerate(devices)},
        queue_path_fn=queue_path,evaluation_reference_root=Path(run['reference_root']),capacities=capacities)
    batch=Batch(Path.cwd(),devices,root=args.root,required_free_MiB=18000,work_event=scheduler.event);failure=None
    jobs=[Job(f'worker_{i:02d}',['control-space','worker','--worker-id',f'worker_{i:02d}','--slots',str(capacities[f'worker_{i:02d}'])],
        eligible_devices=(d,),estimate_seconds=1200,required_free_MiB=28000 if capacities[f'worker_{i:02d}']==16 else 18000) for i,d in enumerate(devices)]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures=[pool.submit(execute_worker,batch,j,d,scheduler) for j,d in zip(jobs,devices)]
        for future in as_completed(futures):
            try:future.result()
            except BaseException as error:
                failure=repr(error)
                with scheduler.lock:scheduler.failed=True;scheduler.dispatch()
    if failure or not scheduler.terminal(scheduler.statuses()):raise RuntimeError(failure or '128 fixed readout not complete')
    write_json_atomic(args.root/'readout_completion.json',dict(complete=True,rows=128,interactions=summary(args.root)))
