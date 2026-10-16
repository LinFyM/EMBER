"""Persistent source-only worker for registered saved-controller query packs."""
from pathlib import Path
import json
import os
import sys
import time

from ember.pi05_source_checkpoint import read_json
from ember.reference.accounting import host_resources
from ember.reference.run import close_runner,event
from .practice import panel,specifications,case_id


def worker(args):
    import torch
    from ember.pi05_source_setup import initialize_distributed
    from ember.pi05_assets import prepare_libero_config,configure_libero_runtime_assets
    from ember.reference.training import Runtime
    from ember.writer.practice import EpisodeRunner
    context=initialize_distributed(require_numa=True,defer_process_group=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    from .comparison import execution_contract
    run=execution_contract(args.root);environment=run['environment']
    if len(environment['tasks'])!=36:raise ValueError('this window has only36 non-held practice/query tasks')
    os.environ.update(EMBER_LIBERO_ASSETS_ROOT=environment['libero_paths']['assets'],MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    prepare_libero_config(args.root/'libero_config');configure_libero_runtime_assets(Path(environment['libero_paths']['assets']))
    runtime=Runtime(args.root,context.device);runtime.episode_event_log=os.environ.get('EMBER_EPISODE_EVENT_LOG')
    if run['phase']!='feedback_readout':raise ValueError('only the current saved-controller comparison is active')
    authorized={case_id(s):s for s in run['requests']}
    runner=None
    try:
        while True:
            event(kind='ready',worker=args.worker_id,resources=host_resources())
            while True:
                line=sys.stdin.readline();command=json.loads(line) if line else dict(command='stop')
                if command['command']!='idle':break
                runner=close_runner(runner,args.worker_id)
            if command['command']=='stop':break
            if command['command']!='panel':raise ValueError('worker executes registered panels only')
            specs=specifications(command)
            for spec in specs:
                registered=authorized[case_id(spec)]
                if any(spec.get(k)!=v for k,v in registered.items()):raise ValueError('query differs from the fixed manifest')
            count=len(specs);slots=int(command.get('slots',min(getattr(args,'slots',16),count)))
            if not 1<=slots<=24:raise ValueError('actual saved-controller renderer slots must be1..24')
            if runner is not None and slots<runner.slots:runner=close_runner(runner,args.worker_id)
            if runner is None:runner=EpisodeRunner(runtime,environment,args.physical_gpu,slots=slots)
            runner.slots=slots;tick=time.monotonic();torch.cuda.reset_peak_memory_stats()
            try:
                result=panel(args.root,runtime,runner,command,None,None)
                event(kind='complete',worker=args.worker_id,identity=command['identity'],request_id=command.get('request_id'),
                    **result,seconds=time.monotonic()-tick,slots=slots,
                    peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
                    peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2)
            except BaseException as error:
                event(kind='failed',worker=args.worker_id,identity=command['identity'],request_id=command.get('request_id'),error=repr(error));raise
    finally:
        close_runner(runner,args.worker_id);runtime.execution.close()
