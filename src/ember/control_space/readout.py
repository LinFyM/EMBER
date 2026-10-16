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
            memory=features.memory(request['original']).to(device)
            events=features.events(request['history_paths'],image_chunk=physical['image_chunk'])
            with torch.no_grad(),autocast(device):
                candidate=model.propose(memory,events,torch.tensor(request['m'],device=device,dtype=torch.float32),
                    request['best_score'],request['remaining_steps'],event_chunk=physical['event_chunk'],checkpointed=False)
            if candidate.shape!=(64,) or not bool(candidate.isfinite().all()):raise FloatingPointError('nonfinite revision may not execute')
            torch.cuda.synchronize(device)
            receipt=dict(request=request,candidate=candidate.float().cpu().tolist(),seconds=time.monotonic()-tick,
                H_events=len(events),teaching_tokens=memory.shape[1],feature_cache=features.measurements,
                revision_update=180,checkpoint=str(Path(root)/'checkpoints/update_00000180/state.pt'))
            write_json_atomic(destination,receipt)
        receipts.append(receipt)
    return receipts


def revision_runtime(root,runtime,device):
    import torch
    from .contract import SCHEMA,BASIS,INITIAL_READER
    from .training import frozen_reader
    from .feedback import RevisionReader
    from .features import FrozenFeatures
    if read_json(Path(root)/'training_completion.json')['next_update']!=180:raise ValueError('only terminal180 revision may compile')
    saved=torch.load(Path(root)/'checkpoints/update_00000180/state.pt',map_location='cpu',weights_only=False,mmap=True)
    if saved['schema']!=SCHEMA or saved['next_update']!=180 or saved['frozen_basis']!=str(BASIS) or saved['frozen_reader']!=str(INITIAL_READER):
        raise ValueError('terminal revision/frozen identity changed')
    model=RevisionReader();model.load_state_dict(saved['revision'],strict=True)
    model=model.to(device).requires_grad_(False).eval();reader,_=frozen_reader(device)
    return model,FrozenFeatures(root,runtime,reader),dict(saved['physical'])


def authorize(root,specs,conditions):
    grouped={}
    for spec in specs:grouped.setdefault(spec['condition_id'],[]).append(spec)
    for identity,rows in grouped.items():
        condition=conditions[identity]
        state=read_json(Path(root)/'conditions'/identity/'path.json')
        pending=state.get('pending')
        if pending is None or state['query_complete']:raise ValueError('worker received an already resolved condition')
        authorized={case_id(s):s for s in specifications(pending)}
        if set(authorized)!={case_id(s) for s in rows}:raise ValueError('only an entire four-case registered panel may execute')
        for spec in rows:
            if (spec['task'],spec['arm'])!=(condition['task'],condition['arm']):raise ValueError('condition crossed arm/task')
            if any(spec.get(k)!=v for k,v in authorized[case_id(spec)].items()):raise ValueError('actual sample differs from the atomic pending decision')


def worker(args):
    import torch
    from ember.pi05_source_setup import initialize_distributed
    from ember.pi05_assets import prepare_libero_config,configure_libero_runtime_assets
    from ember.reference.training import Runtime
    from ember.writer.practice import EpisodeRunner
    context=initialize_distributed(require_numa=True,defer_process_group=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    run=read_json(args.root/'run_contract.json');environment=run['environment']
    if run['phase']!='functional_feedback' or len(environment['tasks'])!=36 or len(run['conditions'])!=36:
        raise ValueError('only the fixed36 terminal revision conditions are active')
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
            if command['command']=='propose':
                # Long-H inference needs no resident renderer children.
                runner=close_runner(runner,args.worker_id)
                receipts=propose(args.root,command,conditions,model,features,physical,context.device)
                event(kind='proposed',worker=args.worker_id,identity=command['identity'],receipts=receipts,resources=host_resources())
                continue
            if command['command']!='panel':raise ValueError('only current complete panels may execute')
            specs=specifications(command);authorize(args.root,specs,conditions);slots=len(specs)
            if not 4<=slots<=24 or slots%4:raise ValueError('useful worker packs contain1..6 complete four-case panels')
            # EGL allocates outside PyTorch; H temporaries must not pin its room.
            torch.cuda.empty_cache()
            if runner is not None and slots<runner.slots:runner=close_runner(runner,args.worker_id)
            if runner is None:runner=EpisodeRunner(runtime,environment,args.physical_gpu,slots=slots)
            runner.slots=slots;tick=time.monotonic();torch.cuda.reset_peak_memory_stats()
            try:
                result=panel(args.root,runtime,runner,command,None,None)
                event(kind='complete',worker=args.worker_id,identity=command['identity'],**result,seconds=time.monotonic()-tick,slots=slots,
                    peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
            except BaseException as error:
                event(kind='failed',worker=args.worker_id,identity=command['identity'],error=repr(error));raise
    finally:
        close_runner(runner,args.worker_id);runtime.execution.close()
