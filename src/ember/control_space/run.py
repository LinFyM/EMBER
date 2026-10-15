"""Single formal CPU-fit then fixed-LoRA readout owner; no training fallback."""
from pathlib import Path
import argparse,datetime,fcntl,subprocess
from ember.pi05_source_checkpoint import write_json_atomic
from .contract import ROOT,SCHEMA


def owner(args):
    from .data import prepare
    from .fit import fit
    from .analysis import materialize,analyze
    from .readout import owner as readout_owner
    args.root.mkdir(parents=True,exist_ok=True)
    lock=(args.root/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (args.root/'completion.json').exists():raise ValueError('completed linear64 window must not restart')
    prepare(args.root)
    result=fit(args.root)
    print('EMBER_FIT_OBSERVATION '+__import__('json').dumps({k:result[k] for k in
        ['method','wall_seconds','code_dim','effective_rank','numerical_rank','retained_spectral_mass','peak_RSS_GiB']}),flush=True)
    materialize(args.root);readout_owner(args);result=analyze(args.root)
    write_json_atomic(args.root/'owner_result.json',dict(complete=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),schema=SCHEMA))
    lock.close();return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['owner','readout-worker'])
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--devices',nargs='+')
    parser.add_argument('--worker-id');parser.add_argument('--slots',type=int,default=16)
    parser.add_argument('--physical-gpu',type=int,default=0);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip() or subprocess.check_output(['git','branch','--show-current'],text=True).strip():
        raise ValueError('formal fit/readout requires clean detached pushed source')
    subprocess.run(['git','merge-base','--is-ancestor','HEAD','origin/main'],check=True)
    if args.command=='readout-worker':
        from .readout import worker
        worker(args)
    else:owner(args)


if __name__=='__main__':main()
