"""Every registered video gets one complete1180 LoRA and one real final episode."""
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,os,sys,time,sqlite3,subprocess
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.pi05_eval_contract import policy_noise_seed
from ember.pi05_eval_queue import EvaluationShard,initialize_queue,complete_job,fail_job
from ember.reference.accounting import reserve,finish,connect,host_resources,summary
from ember.reference.coordinator import Scheduler
from ember.reference.run import close_runner,event
from ember.proposal_writer.batch import Batch,Job
from .contract import SCHEMA,PHASE,TASKS,VALIDATION_TASKS,FINAL,REFERENCE_ROOT,checkpoint_path,queue_path


def inputs(root):
    from .data import registered_panels
    root=Path(root);run=read_json(root/'run_contract.json');panels=read_json(root/'panels.json')
    if run['schema']!=SCHEMA or run['phase']!=PHASE or run['final_update']!=FINAL or panels!=registered_panels():
        raise ValueError('readout must bind the single full1180/seen576/validation400 contract')
    tasks={}
    # The original video owner retained one44-task environment union. Consume
    # only this role's registered panel; retain the immutable old contract file.
    for role,required in (('seen',TASKS),('validation',VALIDATION_TASKS)):
        environment=run['environments'][role];all_tasks={t['global_task_id']:t for t in environment['tasks']}
        if len(all_tasks)!=len(environment['tasks']) or not set(required)<=set(all_tasks):
            raise ValueError('role environment is missing an explicitly registered task')
        tasks[role]={t:all_tasks[t] for t in required}
        run['environments'][role]=dict(environment,tasks=list(tasks[role].values()))
    return run,tasks,panels


def validate_case(row,task,panel,role):
    noise=panel['noise_root'];init=panel['init_state_id']
    expected=dict(suite=task['suite'],task_id=task['task_id'],global_task_id=task['global_task_id'],language=task['language'],
        init_state_id=init,env_seed=noise,policy_seed_root=noise,noise_root=noise,settling_steps=10,
        restored_control_steps=0,full_native_flow_steps=10,executed_chunk=5,source_frozen=True,MT_loaded=False,
        role=role,teacher_demo=panel['teacher_demo'],generator_update=FINAL,phase=PHASE)
    if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('official per-video task/native/RNG contract changed')
    steps=row.get('steps',0)
    if (type(row.get('success')) is not bool or not 0<steps<=task['horizon'] or (not row['success'] and steps!=task['horizon'])
            or row.get('model_failure') or row.get('budget_truncated') or row.get('failure_type')
            or row.get('environment_steps')!=steps+10):raise ValueError('engineering/incomplete row cannot complete a final query')
    stream=[policy_noise_seed(noise,task['suite'],task['task_id'],init,i) for i in range((steps+4)//5)]
    if row.get('policy_noise_seeds')!=stream or row.get('first_noise_seed')!=stream[0]:raise ValueError('absolute native policy-noise stream changed')
    from ember.pi05_eval.scene import validate_scene_row
    validate_scene_row(row,task,dict(operator_read_write_scene=panel.get('registered_scene')))


def row_path(root,role,panel):
    return Path(root)/'readout'/role/f'task_{panel["task"]:04d}'/f'state_{panel["init_state_id"]:03d}.json'


def saved_case(root,role,task,panel):
    from .comparison import validate_generated
    row=read_json(row_path(root,role,panel));validate_case(row,task,panel,role)
    meta=validate_generated(root,role,panel)
    expected=dict(LoRA=meta['path'],actual_parameter_ref=meta['path'],producer_parameter_ref=meta['path'],
        generator_checkpoint=str(checkpoint_path(root,FINAL)),interaction_adaptation=False)
    if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('final row lost its own frozen per-video complete controller')
    return row


def cases(args,runtime,work,task,panels,runner):
    import torch
    from safetensors.torch import load_file
    from ember.lora import validate_lora_state
    from .comparison import validate_generated
    root=args.root;role=work['role'];rows={};missing=[]
    for panel in panels:
        init=panel['init_state_id'];destination=row_path(root,role,panel)
        if destination.exists():
            row=saved_case(root,role,task,panel)
            with connect(root) as db:status=db.execute('SELECT status FROM attempts WHERE identity=?',(row['attempt_id'],)).fetchone()
            if status==('reserved_unknown',):finish(root,row['attempt_id'],row,destination)
            elif status!=('complete',):raise ValueError('saved valid row lost its original physical ledger')
            rows[init]=row
        else:missing.append(panel)
    states={};metadata={p['init_state_id']:validate_generated(root,role,p) for p in missing}
    for init,meta in metadata.items():
        state=load_file(meta['path'],device=str(runtime.device));validate_lora_state(state,runtime.lora)
        if any(not bool(v.isfinite().all()) for v in state.values()):raise ValueError('nonfinite saved controller cannot execute')
        states[init]=state
    def requests():
        for panel in missing:
            init=panel['init_state_id'];path=metadata[init]['path']
            attempt=reserve(root,f'{role}_t{work["task"]:04d}_s{init:03d}',task['horizon'],limits=read_json(root/'run_contract.json')['limits'])
            yield dict(task=task,state=states[init],parameter_ref=path,state_id=init,noise_root=panel['noise_root'],episode_id=attempt,
                registered_scene=panel.get('registered_scene'),capture_history=False,capture_hidden=False,capture_phi=False,
                capture_rgb=False,uses_teaching=False,panel=panel)
    before=deepcopy(runner.components);start=time.monotonic();peak=None
    for output in runner.run(requests()) if missing else ():
        if peak is None:
            peak=host_resources();peak['environment_PIDs']=[p.pid for p in runner.environments.processes if p is not None]
        req=output['request'];row=output['row'];panel=req['panel'];init=panel['init_state_id'];first=output['history'].records[0]
        path=metadata[init]['path']
        row.update(global_task_id=task['global_task_id'],LoRA=path,phase=PHASE,generator_checkpoint=str(checkpoint_path(root,FINAL)),
            generator_update=FINAL,role=role,teacher_demo=panel['teacher_demo'],attempt_id=req['episode_id'],noise_root=panel['noise_root'],
            first_noise_seed=first['noise_seed'],first_native_normalized_actions=first['normalized_actions'].tolist(),full_native_flow_steps=10,
            executed_chunk=5,source_frozen=True,MT_loaded=False,producer_parameter_ref=path,interaction_adaptation=False)
        validate_case(row,task,panel,role)
        destination=row_path(root,role,panel);write_json_atomic(destination,row);finish(root,req['episode_id'],row,destination);rows[init]=row
    elapsed=time.monotonic()-start
    components={k:v-before[k] for k,v in runner.components.items() if k!='physical_batch_histogram'}
    components['physical_batch_histogram']={k:v-before['physical_batch_histogram'].get(k,0) for k,v in runner.components['physical_batch_histogram'].items()}
    return [rows[p['init_state_id']] for p in panels],dict(new_episodes=len(missing),seconds=elapsed,episodes_per_second=len(missing)/max(elapsed,1e-6),
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
    for slots,required in ((16,28000),(8,18000),(4,14000)):
        if project_available_MiB>=required:return min(logical_rows,slots)
    raise RuntimeError('live local memory cannot cover the measured four-slot native workload before submission')


def panel_runner(runner,runtime,environment,args,logical_rows,limit=None):
    import torch
    from ember.writer.practice import EpisodeRunner
    def available():
        torch.cuda.empty_cache();free,_=torch.cuda.mem_get_info(runtime.device)
        return (free+torch.cuda.memory_allocated(runtime.device))/2**20
    headroom=available();slots=min(limit or logical_rows,physical_slots(logical_rows,headroom))
    if runner is not None and slots<runner.slots:
        runner=close_runner(runner,args.worker_id);headroom=available();slots=min(limit or logical_rows,physical_slots(logical_rows,headroom))
    if runner is None:runner=EpisodeRunner(runtime,environment,args.physical_gpu,slots=slots)
    runner.slots=slots
    return runner,dict(slots=slots,logical_rows=logical_rows,project_available_MiB=headroom,native_slot_capacity=physical_slots(16,headroom))


def worker(args):
    import torch
    from ember.pi05_source_setup import initialize_distributed
    from ember.pi05_assets import prepare_libero_config,configure_libero_runtime_assets
    from ember.reference.training import Runtime
    from .comparison import Materializer,generated
    context=initialize_distributed(require_numa=True,defer_process_group=True);torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    run,tasks,panels=inputs(args.root);lookup={role:{(p['task'],p['init_state_id']):p for p in values} for role,values in panels.items()}
    environment=run['environments']['seen']
    os.environ.update(EMBER_LIBERO_ASSETS_ROOT=environment['libero_paths']['assets'],MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    prepare_libero_config(args.root/'libero_config');configure_libero_runtime_assets(Path(environment['libero_paths']['assets']))
    runtime=Runtime(args.root,context.device);runtime.episode_event_log=os.environ.get('EMBER_EPISODE_EVENT_LOG');runner=None;active_role=None
    compiler=Materializer(args.root,context.device,runtime,row_chunk=args.row_chunk,pack=args.materialize_pack,frame_chunk=args.frame_chunk)
    try:
        while True:
            event(kind='ready',worker=args.worker_id,resources=host_resources())
            while True:
                line=sys.stdin.readline();message=json.loads(line) if line else dict(command='stop')
                if message['command']!='idle':break
                runner=close_runner(runner,args.worker_id)
            if message['command']=='stop':break
            work=message['work'];identity=message['identity'];role=work['role'];tick=time.monotonic()
            selected=[lookup[role][work['task'],i] for i in work['states']]
            try:
                if active_role!=role or (role=='validation' and any(not generated(args.root,role,p).exists() for p in selected)):
                    runner=close_runner(runner,args.worker_id)
                active_role=role;environment=run['environments'][role]
                torch.cuda.reset_peak_memory_stats();controllers,materialization=compiler.compile(role,selected)
                runner,admission=panel_runner(runner,runtime,environment,args,len(selected),message['slots'])
                event(kind='physical_pack',worker=args.worker_id,identity=identity,**admission,resources=host_resources())
                rows,resources=cases(args,runtime,work,tasks[role][work['task']],selected,runner)
                resources['materialization']=materialization
                receipt=dict(complete=True,work=work,output=dict(rows=rows,resources=resources),seconds=time.monotonic()-tick,
                    slots=runner.slots,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
                path=args.root/'shards'/f'{identity}.json';write_json_atomic(path,receipt)
                complete_job(queue_path(args.root),job_id=identity,worker_id=args.worker_id,claim_token=message['claim_token'],result_path=path)
                event(kind='useful_readout',worker=args.worker_id,identity=identity,seconds=receipt['seconds'],episodes=len(rows),slots=runner.slots)
            except BaseException as error:
                fail_job(queue_path(args.root),job_id=identity,worker_id=args.worker_id,claim_token=message['claim_token'],reason=repr(error));raise
    finally:
        runner=close_runner(runner,args.worker_id);compiler.close();runtime.execution.close()


def work_panel(tasks,panels):
    result={}
    for role in ('seen','validation'):
        width=16 if role=='seen' else 8
        for task in tasks[role]:
            states=sorted(p['init_state_id'] for p in panels[role] if p['task']==task)
            for offset in range(0,len(states),width):
                selected=states[offset:offset+width];identity=f'{role}_t{task:04d}_s{selected[0]:03d}'
                result[identity]=dict(kind='readout',role=role,task=task,states=selected,cost=len(selected)*(tasks[role][task]['horizon']+10))
    return result


def owner(args):
    from .run import execute_worker,device_pool
    run,tasks,panels=inputs(args.root);panel=work_panel(tasks,panels)
    manifest=read_json(checkpoint_path(args.root,FINAL).parent/'manifest.json')
    if not manifest['complete'] or manifest['schema']!=SCHEMA or manifest['next_update']!=FINAL:raise ValueError('endpoint1180 must be complete before any materialization/evaluation')
    shards=[EvaluationShard(i,n,tasks[w['role']][w['task']]['suite'],w['task'],tasks[w['role']][w['task']]['horizon'],tuple(w['states']),w['cost']) for n,(i,w) in enumerate(panel.items())]
    initialize_queue(queue_path(args.root),shards,contract_reference=str(args.root/'run_contract.json'),recover_claims=args.resume,retry_failed=args.resume)
    with sqlite3.connect(queue_path(args.root)) as db:statuses=dict(db.execute('SELECT job_id,status FROM jobs'))
    if all(statuses[i]=='complete' for i in panel):
        write_json_atomic(args.root/'readout_completion.json',dict(complete=True,rows=976,interactions=summary(args.root)));return
    choices,cap=device_pool(args)
    write_json_atomic(args.root/'execution_readout.json',dict(devices=choices,total_cap=cap,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    devices=[(n,g) for n,g,_ in choices];capacities={f'readout_worker_{i:02d}':16 if free>=28000 else 8 for i,(_,_,free) in enumerate(choices)}
    scheduler=ReadoutScheduler(args.root,panel,len(devices),worker_nodes={f'readout_worker_{i:02d}':n for i,(n,_) in enumerate(devices)},
        queue_path_fn=queue_path,evaluation_reference_root=REFERENCE_ROOT,capacities=capacities)
    batch=Batch(Path.cwd(),devices,root=args.root,required_free_MiB=18000,work_event=scheduler.event);failure=None
    jobs=[]
    for i,device in enumerate(devices):
        name=f'readout_worker_{i:02d}';slots=capacities[name]
        arguments=['control-space','worker','--worker-id',name,'--slots',str(slots),'--row-chunk',str(args.row_chunk),
            '--materialize-pack',str(args.materialize_pack),'--frame-chunk',str(args.frame_chunk)]
        jobs.append(Job(name,arguments,eligible_devices=(device,),estimate_seconds=3600,required_free_MiB=28000 if slots==16 else 18000))
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures=[pool.submit(execute_worker,batch,j,d,scheduler) for j,d in zip(jobs,devices)]
        for future in as_completed(futures):
            try:future.result()
            except BaseException as error:
                failure=repr(error)
                with scheduler.lock:scheduler.failed=True;scheduler.dispatch()
    if failure or not scheduler.terminal(scheduler.statuses()):raise RuntimeError(failure or '976 registered finals not complete')
    write_json_atomic(args.root/'readout_completion.json',dict(complete=True,rows=976,interactions=summary(args.root)))
