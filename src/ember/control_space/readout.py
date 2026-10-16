"""Persistent raw-H practice/query worker; frozen source with one complete LoRA."""
from pathlib import Path
import json,os,sys,time
from ember.pi05_source_checkpoint import read_json
from ember.reference.accounting import host_resources
from ember.reference.run import close_runner,event
from .practice import panel,specifications,case_id


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
    if run['phase']!='incumbent_compilation' or len(environment['tasks'])!=36:raise ValueError('only the72 nonheld incumbent conditions are active')
    conditions={c['id']:c for c in run['conditions']}
    os.environ.update(EMBER_LIBERO_ASSETS_ROOT=environment['libero_paths']['assets'],MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    prepare_libero_config(args.root/'libero_config');configure_libero_runtime_assets(Path(environment['libero_paths']['assets']))
    runtime=Runtime(args.root,context.device);runtime.episode_event_log=os.environ.get('EMBER_EPISODE_EVENT_LOG');runner=None
    try:
        while True:
            event(kind='ready',worker=args.worker_id,resources=host_resources())
            while True:
                line=sys.stdin.readline();command=json.loads(line) if line else dict(command='stop')
                if command['command']!='idle':break
                runner=close_runner(runner,args.worker_id)
            if command['command']=='stop':break
            if command['command']!='panel':raise ValueError('only current complete panels may execute')
            specs=specifications(command);authorize(args.root,specs,conditions);slots=len(specs)
            if not 4<=slots<=24 or slots%4:raise ValueError('useful worker packs contain1..6 complete four-case panels')
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
