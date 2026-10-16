"""Persistent raw-H practice/query worker; frozen source with one complete LoRA."""
from pathlib import Path
import json,os,sys,time
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.reference.accounting import host_resources
from ember.reference.run import close_runner,event
from .practice import panel,specifications,case_id


def propose(root,command,conditions,model,features,physical,device):
    import torch
    from ember.writer.runtime import autocast
    receipts=[]
    for request in command['requests']:
        condition=conditions[request['condition_id']]
        saved=read_json(Path(root)/'conditions'/condition['id']/'path.json')
        if (saved.get('pending_proposal')!=request or request['task']!=condition['task']
                or request['arm']!=condition['arm'] or request['original']['demo']!=condition['teacher_demo']
                or request['original']['task']!=condition['task']):raise ValueError('revision request differs from the saved causal condition')
        destination=Path(root)/'proposals'/f"{request['identity']}.json"
        if destination.is_file():
            receipt=read_json(destination)
            if receipt['request']!=request:raise ValueError('saved actual proposal lost its original input')
        else:
            tick=time.monotonic();features.measurements=[]
            memory=features.memory(request['original'],frame_chunk=32).to(device)
            events=features.events(request['history_paths'],image_chunk=physical['image_chunk'])
            with torch.no_grad(),autocast(device):
                candidate=model.propose(memory,events,torch.tensor(request['m'],device=device,dtype=torch.float32),
                    request['best_score'],request['remaining_steps'],event_chunk=physical['event_chunk'],checkpointed=False)
            if candidate.shape!=(64,) or not bool(candidate.isfinite().all()):raise FloatingPointError('nonfinite revision may not execute')
            torch.cuda.synchronize(device)
            receipt=dict(request=request,candidate=candidate.float().cpu().tolist(),seconds=time.monotonic()-tick,
                H_events=len(events),teaching_tokens=memory.shape[1],feature_cache=features.measurements,
                revision_update=180,checkpoint=features.revision_checkpoint)
            write_json_atomic(destination,receipt)
        receipts.append(receipt)
    return receipts


def frozen_reader(device):
    import torch
    from .contract import BASIS,INITIAL_READER
    from .reader import ImageReader
    saved=torch.load(INITIAL_READER,map_location='cpu',weights_only=False,mmap=True)
    if saved['phase']!='video_initial' or saved['next_update']!=900 or saved['frozen_basis']!=str(BASIS):
        raise ValueError('only the original frozen Reader900 may condition revision')
    reader=ImageReader();reader.load_state_dict(saved['reader'],strict=True)
    return reader.to(device).requires_grad_(False).eval()


def revision_runtime(root,runtime,device):
    import torch
    from .contract import CHECKPOINT_SCHEMA,BASIS,INITIAL_READER,REVISION_CHECKPOINT,active_run
    from .feedback import RevisionReader
    from .features import FrozenFeatures
    run=active_run(root)
    saved=torch.load(REVISION_CHECKPOINT,map_location='cpu',weights_only=False,mmap=True)
    if (saved['schema']!=CHECKPOINT_SCHEMA or saved['next_update']!=180
            or saved['frozen_basis']!=str(BASIS) or saved['frozen_reader']!=str(INITIAL_READER)):
        raise ValueError('terminal revision/frozen identity changed')
    from ember.reference.contract import ASSETS
    from ember.writer.learning_data import load_learning_tasks
    from ember.writer.data import RawTeacherVideoStore
    held=load_learning_tasks(ASSETS,[3,6,11,16,23,26,31,39],role='validation',protocol_path='configs/libero_24_8_8_coverage_v1/protocol.json')
    runtime.videos=RawTeacherVideoStore([t.authority for t in held.values()],frame_stride=5,camera_view='dual')
    model=RevisionReader();model.load_state_dict(saved['revision'],strict=True)
    model=model.to(device).requires_grad_(False).eval()
    features=FrozenFeatures(root,runtime,frozen_reader(device));features.revision_checkpoint=str(REVISION_CHECKPOINT)
    # Reuse the useful long-H chunk evidence; full source/prefix tensors remain
    # transient, and a physical chunk never changes event/frame order.
    physical=dict(saved['physical'])
    return model,features,physical


def authorize(root,specs,conditions):
    grouped={}
    for spec in specs:grouped.setdefault(spec['condition_id'],[]).append(spec)
    for identity,rows in grouped.items():
        condition=conditions[identity]
        state=read_json(Path(root)/'conditions'/identity/'path.json')
        pending=state.get('pending')
        if pending is None or state['query_complete']:raise ValueError('worker received an already resolved condition')
        authorized={case_id(s):s for s in specifications(pending)}
        if set(authorized)!={case_id(s) for s in rows}:raise ValueError('only an entire registered practice panel or single final may execute')
        for spec in rows:
            if (spec['task'],spec['arm'])!=(condition['task'],condition['arm']):raise ValueError('condition crossed arm/task')
            if any(spec.get(k)!=v for k,v in authorized[case_id(spec)].items()):raise ValueError('actual sample differs from the atomic pending decision')

def physical_slots(logical_rows,project_available_MiB):
    """Only measured native capacities; a logical four-case panel stays whole."""
    for slots,required in ((16,28000),(8,18000),(4,14000)):
        if project_available_MiB>=required:return min(logical_rows,slots)
    raise RuntimeError('live local memory cannot cover the measured four-slot native workload before environment submission')


def panel_runner(runner,runtime,environment,args,logical_rows):
    import torch
    from ember.writer.practice import EpisodeRunner
    def available():
        torch.cuda.empty_cache()
        free,_=torch.cuda.mem_get_info(runtime.device)
        return (free+torch.cuda.memory_allocated(runtime.device))/2**20
    headroom=available();slots=physical_slots(logical_rows,headroom)
    if runner is not None and slots<runner.slots:
        # Existing own EGL allocations are outside PyTorch. Close them before
        # recalculating the available project share; never credit foreign memory.
        runner=close_runner(runner,args.worker_id)
        headroom=available();slots=physical_slots(logical_rows,headroom)
    if runner is None:runner=EpisodeRunner(runtime,environment,args.physical_gpu,slots=slots)
    runner.slots=slots
    return runner,dict(slots=slots,logical_rows=logical_rows,project_available_MiB=headroom,
        native_slot_capacity=physical_slots(16,headroom))


def worker(args):
    import torch
    from ember.pi05_source_setup import initialize_distributed
    from ember.pi05_assets import prepare_libero_config,configure_libero_runtime_assets
    from ember.reference.training import Runtime
    from ember.writer.practice import EpisodeRunner
    context=initialize_distributed(require_numa=True,defer_process_group=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    from .contract import active_run
    run=active_run(args.root);environment=run['environment']
    if len(environment['tasks'])!=8 or len(run['conditions'])!=400:
        raise ValueError('only validation8/400 fixed180 conditions are active')
    conditions={c['id']:c for c in run['conditions']}
    os.environ.update(EMBER_LIBERO_ASSETS_ROOT=environment['libero_paths']['assets'],MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    prepare_libero_config(args.root/'libero_config');configure_libero_runtime_assets(Path(environment['libero_paths']['assets']))
    runtime=Runtime(args.root,context.device);runtime.episode_event_log=os.environ.get('EMBER_EPISODE_EVENT_LOG');runner=None
    model,features,physical=revision_runtime(args.root,runtime,context.device)
    try:
        while True:
            event(kind='ready',worker=args.worker_id,resources=host_resources())
            while True:
                line=sys.stdin.readline();command=json.loads(line) if line else dict(command='stop')
                if command['command']!='idle':break
                runner=close_runner(runner,args.worker_id)
            if command['command']=='stop':break
            if command['command']=='retire':
                from .lifecycle import retire_condition
                runner=close_runner(runner,args.worker_id)
                receipts=[retire_condition(args.root,conditions[i],features,environments_closed=True) for i in command['conditions']]
                event(kind='retired',worker=args.worker_id,receipts=[dict(condition_id=r['condition_id'],
                    deleted_raw_bytes=r['deleted_raw_bytes'],permanent_raw_witness=r['permanent_raw_witness']) for r in receipts],resources=host_resources())
                continue
            if command['command']=='propose':
                # Long-H inference needs no resident renderer children.
                runner=close_runner(runner,args.worker_id)
                receipts=propose(args.root,command,conditions,model,features,physical,context.device)
                event(kind='proposed',worker=args.worker_id,identity=command['identity'],receipts=receipts,resources=host_resources())
                continue
            if command['command']!='panel':raise ValueError('only current complete panels may execute')
            specs=specifications(command);authorize(args.root,specs,conditions);logical_rows=len(specs)
            if not 1<=logical_rows<=16:raise ValueError('registered useful packs contain at most16 real cases')
            runner,admission=panel_runner(runner,runtime,environment,args,logical_rows);slots=runner.slots
            event(kind='physical_pack',worker=args.worker_id,identity=command['identity'],**admission,resources=host_resources())
            tick=time.monotonic();torch.cuda.reset_peak_memory_stats()
            try:
                result=panel(args.root,runtime,runner,command,None,None)
                event(kind='complete',worker=args.worker_id,identity=command['identity'],**result,seconds=time.monotonic()-tick,slots=slots,
                    peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
            except BaseException as error:
                event(kind='failed',worker=args.worker_id,identity=command['identity'],error=repr(error));raise
    finally:
        close_runner(runner,args.worker_id);runtime.videos.close();runtime.execution.close()
