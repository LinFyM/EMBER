"""Thin single control-space formal CLI and dependency-ordered batch owner."""
from pathlib import Path
import argparse,datetime,fcntl,json,os,socket,subprocess
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from .contract import ROOT,REFERENCE_ROOT,SCHEMA


def owner(args):
    from .data import prepare
    from .analysis import materialize,analyze
    from .readout import owner as readout_owner
    from ember.proposal_writer.batch import Batch,Job
    lock=(args.root/'owner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (args.root/'completion.json').exists():raise ValueError('completed window must not restart')
    prepare(args.root)
    devices=[(a.split(':')[0],int(a.split(':')[1])) for a in args.devices]
    if len({n for n,_ in devices})!=1:raise ValueError('E/D training must stay within one node')
    def observation(worker,event,stream):
        # These are actual useful-update events, never status polls or self Queue.
        print('EMBER_TRAINING_OBSERVATION '+json.dumps(event),flush=True)
    batch=Batch(Path.cwd(),devices,root=args.root,required_free_MiB=40000,work_event=observation)
    phases=[(0,36)] if not (args.root/'checkpoints/update_00000036/state.pt').exists() else []
    for start,stop in phases:
        job=Job(f'train_{start:08d}_{stop:08d}',['control-space','train','--start',str(start),'--stop',str(stop)],
            gpu_count=len(devices),estimate_seconds=3600,required_free_MiB=40000)
        batch.execute(job,devices)
    physical=read_json(args.root/'physical_execution.json')
    chosen=devices[:physical['recommended_world_size']]
    if not (args.root/'checkpoints/update_00000144/state.pt').exists():
        job=Job('train_00000036_00000144',['control-space','train','--start','36','--stop','144'],
            gpu_count=len(chosen),estimate_seconds=7200,required_free_MiB=40000)
        batch.execute(job,chosen)
    materialize(args.root);readout_owner(args);result=analyze(args.root)
    write_json_atomic(args.root/'owner_result.json',dict(complete=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),schema=SCHEMA))
    lock.close();return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['owner','train','readout-worker'])
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--devices',nargs='+')
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int,default=144)
    parser.add_argument('--worker-id');parser.add_argument('--slots',type=int,default=16)
    parser.add_argument('--physical-gpu',type=int,default=0);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip() or subprocess.check_output(['git','branch','--show-current'],text=True).strip():
        raise ValueError('formal control-space consumers require clean detached pushed source')
    subprocess.run(['git','merge-base','--is-ancestor','HEAD','origin/main'],check=True)
    if args.command=='train':
        if args.start not in (0,36,72,108) or args.stop not in (36,72,108,144) or args.stop<=args.start:
            raise ValueError('unregistered complete-boundary segment')
        from .training import train
        train(args)
    elif args.command=='readout-worker':
        from .readout import worker
        worker(args)
    else:owner(args)


if __name__=='__main__':main()
