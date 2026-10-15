"""Thin formal worker CLI; imports stay inside main for spawned environments."""
from pathlib import Path
import argparse
import json
import os
import sys
import time


def event(**fields):
    print('EMBER_WORK_EVENT '+json.dumps(fields),flush=True)


def save_rgb(folder, init, history, row, weights):
    import numpy as np
    capture=history.rgb
    indices=np.asarray([r['control_step'] for r in capture],dtype=np.int32)
    frames=np.stack([r['images'] for r in capture])
    control=row['environment_steps']-row['settling_steps']
    if frames.dtype!=np.uint8 or frames.shape[1:]!=(2,256,256,3) or indices[0]!=0 or indices[-1]!=control:
        raise ValueError('producer RGB lost true raw cameras/start/terminal')
    video=folder/f'state_{init:03d}.npz';temporary=video.with_suffix('.partial')
    with temporary.open('wb') as handle:np.savez_compressed(handle,frames=frames,frame_indices=indices)
    temporary.rename(video)
    row['RGB']=dict(path=str(video),frame_indices=indices.tolist(),already_sampled_stride=5,
        terminal_included=True,camera_order=['agentview','eye_in_hand'],stored_rotation='none',
        policy_rotation_degrees=180,model_resolution=224,render_resolution=256,
        correct_auxiliary_teaching_eligible=bool(row['success']),producer_LoRA=weights)


def cases(args, runtime, work, task, runner):
    from copy import deepcopy
    import torch
    from safetensors.torch import load_file
    from ember.pi05_source_checkpoint import read_json, write_json_atomic
    from .contract import checkpoint, parameter_ref
    from .accounting import reserve, finish, host_resources
    root=args.root;kind=work['kind'];t=work['task'];arm=work.get('arm')
    update=work.get('update',0)
    if kind=='report' and arm=='selected':update=read_json(root/'selection'/f'task_{t:04d}'/'selected.json')['selected_update']
    folder=(root/'selection'/f'task_{t:04d}'/f'update_{update:08d}' if kind=='selection'
            else root/'report'/f'task_{t:04d}'/arm)
    folder.mkdir(parents=True,exist_ok=True)
    state=runtime.identity if update==0 else load_file(str(checkpoint(root,t,update)/'lora.safetensors'),device=str(runtime.device))
    weights='raw_source_identity' if update==0 else str(checkpoint(root,t,update)/'lora.safetensors')
    ref=parameter_ref(t,update);noise_root=202610151 if kind=='selection' else 202610152
    # The selected identity shares the actual source report execution and RGB.
    if kind=='report' and arm=='selected' and update==0:
        for init in work['states']:
            original=root/'report'/f'task_{t:04d}'/'source'/f'state_{init:03d}.json'
            row=read_json(original);row.update(logical_arm='selected',physical_reuse=str(original))
            write_json_atomic(folder/f'state_{init:03d}.json',row)
        return [read_json(folder/f'state_{i:03d}.json') for i in work['states']],dict(source_identity_reuse=True)
    missing=[i for i in work['states'] if not (folder/f'state_{i:03d}.json').exists()]
    def requests():
        for init in missing:
            case=f'{kind}_t{t:04d}_{arm or update}_s{init:03d}'
            attempt=reserve(root,case,task['horizon'])
            yield dict(task=task,state=state,parameter_ref=ref,state_id=init,noise_root=noise_root,
                episode_id=attempt,capture_history=False,capture_hidden=False,capture_phi=False,
                capture_rgb=kind=='report',uses_teaching=False)
    started=time.monotonic();peak_resource=None;components_before=deepcopy(runner.components)
    with torch.no_grad():
        for result in runner.run(requests()) if missing else ():
            if peak_resource is None:
                peak_resource=host_resources()
                peak_resource['environment_PIDs']=[p.pid for p in runner.environments.processes if p is not None]
            row=result['row'];init=result['request']['state_id'];history=result['history']
            if row.get('model_failure') or row.get('budget_truncated'):
                raise RuntimeError('engineering/incomplete result cannot become a scientific row')
            first=history.records[0]
            row.update(global_task_id=t,update=update,logical_arm=arm or 'selection',LoRA=weights,
                first_noise_seed=first['noise_seed'],first_native_normalized_actions=first['normalized_actions'].tolist(),
                full_native_flow_steps=10,executed_chunk=5,source_frozen=True,MT_loaded=False,
                producer_parameter_ref=ref,noise_root=noise_root)
            if kind=='report':
                save_rgb(folder,init,history,row,weights)
            path=folder/f'state_{init:03d}.json'
            write_json_atomic(path,row);finish(root,result['request']['episode_id'],row,path)
    elapsed=time.monotonic()-started
    components={k:v-components_before[k] for k,v in runner.components.items() if k!='physical_batch_histogram'}
    components['physical_batch_histogram']={k:v-components_before['physical_batch_histogram'].get(k,0)
        for k,v in runner.components['physical_batch_histogram'].items()}
    return [read_json(folder/f'state_{i:03d}.json') for i in work['states']],dict(
        seconds=elapsed,new_episodes=len(missing),episodes_per_second=len(missing)/elapsed,
        peak_environment_resources=peak_resource,after_shard_resources=host_resources(),runner=components)


def close_runner(runner,worker):
    if runner is None:return None
    from .accounting import host_resources
    children=[] if runner.environments is None else [p.pid for p in runner.environments.processes if p is not None]
    runner.close()
    remaining=[pid for pid in children if Path(f'/proc/{pid}').exists()]
    event(kind='environments_closed',worker=worker,released_PIDs=children,
          remaining_environment_PIDs=remaining,resources=host_resources())
    if remaining:raise RuntimeError('task-owned environments survived complete close')
    return None


def worker(args):
    import torch
    from ember.pi05_source_checkpoint import read_json, write_json_atomic
    from ember.pi05_eval_queue import complete_job, fail_job
    from ember.pi05_source_setup import initialize_distributed
    from ember.writer.practice import EpisodeRunner
    from .contract import SCHEMA
    from .training import Runtime, train
    from .accounting import host_resources
    context=initialize_distributed(require_numa=True,defer_process_group=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    run=read_json(args.root/'run_contract.json');args.environment=run['environment']
    os.environ.update(EMBER_LIBERO_ASSETS_ROOT=args.environment['libero_paths']['assets'],MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl')
    from ember.pi05_assets import prepare_libero_config, configure_libero_runtime_assets
    prepare_libero_config(args.root/'libero_config');configure_libero_runtime_assets(Path(args.environment['libero_paths']['assets']))
    runtime=Runtime(args.root,context.device);runtime.episode_event_log=os.environ.get('EMBER_EPISODE_EVENT_LOG')
    tasks={t['global_task_id']:t for t in args.environment['tasks']}
    queue=args.root/'work.sqlite';runner=None
    try:
        while True:
            event(kind='ready',worker=args.worker_id,resources=host_resources())
            while True:
                line=sys.stdin.readline();message=json.loads(line) if line else dict(command='stop')
                if message['command']!='idle':break
                runner=close_runner(runner,args.worker_id)
            if message['command']=='stop':break
            work=message['work'];identity=message['identity'];token=message['claim_token']
            args.slots=message.get('slots',args.slots)
            tick=time.monotonic();torch.cuda.reset_peak_memory_stats()
            try:
                if work['kind']=='train':
                    runner=close_runner(runner,args.worker_id)
                    output=train(runtime,work['task'],work['start'],work['stop']);successes=0
                else:
                    if runner is not None and args.slots<runner.slots:runner=close_runner(runner,args.worker_id)
                    if runner is None:runner=EpisodeRunner(runtime,args.environment,args.physical_gpu,slots=args.slots)
                    runner.slots=args.slots
                    rows,resources=cases(args,runtime,work,tasks[work['task']],runner)
                    output=dict(rows=rows,resources=resources);successes=sum(r['success'] for r in rows)
                receipt=dict(schema=SCHEMA,identity=identity,work=work,output=output,complete=True,
                    seconds=time.monotonic()-tick,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
                    peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2,resources=host_resources())
                path=args.root/'shards'/f'{identity}.json';write_json_atomic(path,receipt)
                complete_job(queue,job_id=identity,worker_id=args.worker_id,claim_token=token,
                    rows_path=str(path.relative_to(args.root)),rows_bytes=path.stat().st_size,
                    row_count=len(work['states']),successes=successes)
            except BaseException as error:
                fail_job(queue,job_id=identity,worker_id=args.worker_id,claim_token=token,error=repr(error))
                event(kind='failed',worker=args.worker_id,identity=identity,error=repr(error))
                raise
    finally:
        close_runner(runner,args.worker_id)
        runtime.execution.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['worker','owner'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--worker-id')
    p.add_argument('--physical-gpu',type=int,default=0);p.add_argument('--slots',type=int,default=8)
    p.add_argument('--devices',nargs='+');p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    import subprocess
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip() or subprocess.check_output(['git','branch','--show-current'],text=True).strip():
        raise ValueError('formal reference consumers require clean detached source')
    subprocess.run(['git','merge-base','--is-ancestor','HEAD','origin/main'],check=True)
    if args.command=='worker':worker(args)
    else:
        from .coordinator import owner
        owner(args)


if __name__=='__main__':main()
