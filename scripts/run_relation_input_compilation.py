"""Finite entry point for the matched train-only relation input diagnostic."""
from pathlib import Path
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('train', 'materialize', 'eval'))
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--arm', choices=('P', 'Q'), required=True)
    parser.add_argument('--u', type=int, choices=(0, 180), default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--asset-root', type=Path, default=Path('/data1/user/ymdai/projects/EMBER'))
    parser.add_argument('--node', default='gpu02')
    parser.add_argument('--gpus', default='')
    parser.add_argument('--replicas', type=int, default=1)
    parser.add_argument('--microbatch', type=int, default=28)
    parser.add_argument('--frame-chunk', type=int, default=64)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--profile', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--allow-topology-change', action='store_true')
    parser.add_argument('--deadline-epoch', type=float)
    args = parser.parse_args()
    from ember.pi05_source_checkpoint import read_json
    from ember.pi05_assets import prepare_libero_config
    spec = read_json(args.spec)
    prepare_libero_config(Path(spec['run_root']) / 'runtime' / f'{args.phase}_{args.arm}_{os.getpid()}')
    if args.phase == 'train':
        from ember.relation_input_training import train
        train(spec, args)
    else:
        from ember.relation_input_readout import materialize, evaluate
        action = materialize if args.phase == 'materialize' else evaluate
        action(spec, args)


if __name__ == '__main__':
    main()
