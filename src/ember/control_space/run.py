"""Thin formal owner for the single active operator-learning and readout phase."""
from pathlib import Path
import argparse,datetime,fcntl,json,subprocess
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from .contract import ROOT,SCHEMA


def owner(args):
    from .data import prepare
    from .analysis import materialize,analyze
    from .readout import owner as readout_owner
    from .checkpoint import latest
    from ember.proposal_writer.batch import Batch,Job
    args.root.mkdir(parents=True,exist_ok=True)
    lock=(args.root/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (args.root/'completion.json').exists():raise ValueError('completed scientific phase must not restart')
    prepare(args.root)
    devices=[(a.split(':')[0],int(a.split(':')[1])) for a in args.devices]
    if not 1<=len(devices)<=6 or len({n for n,_ in devices})!=1:raise ValueError('operator training requires 1–6 same-node GPUs')
    def observation(worker,event,stream):print('EMBER_TRAINING_OBSERVATION '+json.dumps(event),flush=True)
    batch=Batch(Path.cwd(),devices,root=args.root,required_free_MiB=12000,work_event=observation)
    if not (args.root/'training_completion.json').exists():
        start=latest(args.root)['update']
        if start==144:
            job=Job('operator_profile_from144',['control-space','train','--profile-only'],gpu_count=len(devices),
                estimate_seconds=600,required_free_MiB=12000)
            batch.execute(job,devices)
        if not (args.root/'training_completion.json').exists():
            physical=read_json(args.root/'physical_execution.json');chosen=devices[:physical['recommended_world_size']]
            job=Job(f'operator_from_{latest(args.root)["update"]:08d}',['control-space','train'],gpu_count=len(chosen),
                estimate_seconds=1800,required_free_MiB=12000)
            batch.execute(job,chosen)
    materialize(args.root);readout_owner(args);result=analyze(args.root)
    write_json_atomic(args.root/'owner_result.json',dict(complete=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),schema=SCHEMA))
    lock.close();return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['owner','train','readout-worker'])
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--devices',nargs='+')
    parser.add_argument('--profile-only',action='store_true');parser.add_argument('--worker-id')
    parser.add_argument('--slots',type=int,default=16);parser.add_argument('--physical-gpu',type=int,default=0)
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip() or subprocess.check_output(['git','branch','--show-current'],text=True).strip():
        raise ValueError('formal control-space consumers require clean detached pushed source')
    subprocess.run(['git','merge-base','--is-ancestor','HEAD','origin/main'],check=True)
    if args.command=='train':
        from .training import train
        train(args)
    elif args.command=='readout-worker':
        from .readout import worker
        worker(args)
    else:owner(args)


if __name__=='__main__':main()
