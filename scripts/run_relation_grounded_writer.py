"""Thin owner for the one registered relational-feedback batch; retire when closed."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('labels', 'train', 'materialize'))
    parser.add_argument('--spec', type=Path, default=Path('configs/relation_grounded_writer_v1/spec.json'))
    parser.add_argument('--asset-root', type=Path, default=Path('/data1/user/ymdai/projects/EMBER'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--cpu-threads', type=int, default=6)
    parser.add_argument('--microbatch', type=int, default=28)
    parser.add_argument('--frame-chunk', type=int, default=16)
    parser.add_argument('--profile', type=int, default=0, choices=range(4))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--allow-topology-change', action='store_true')
    parser.add_argument('--node', default=os.uname().nodename)
    parser.add_argument('--deadline-epoch', type=float)
    parser.add_argument('--model', choices=('G', 'F'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--panel', choices=('correct400', 'seen144'))
    args = parser.parse_args()
    from ember.pi05_source_checkpoint import read_json
    from ember.pi05_assets import prepare_libero_config
    spec = read_json(args.spec)
    if spec.get('task') != 'relation_grounded_writer_20261006' or spec['execution']['updates_per_mode'] != 450:
        raise ValueError('entry only owns the registered fixed450 batch')
    root = Path(spec['run_root'])
    prepare_libero_config(root / 'runtime' / f'libero_{args.phase}_{os.getpid()}')
    if args.phase == 'labels':
        from ember.relation_writer.label_preparation import prepare_labels
        print(prepare_labels(spec, args))
    elif args.phase == 'train':
        if args.output is None:
            raise ValueError('train requires a distinct attempt output')
        from ember.relation_writer.training import train
        train(spec, args)
    else:
        if not args.model or not args.checkpoint or not args.panel:
            raise ValueError('materialize requires the fixed model/checkpoint/panel')
        import torch
        from ember.relation_writer.readout import materialize
        devices = [torch.device('cuda', index) for index in range(torch.cuda.device_count())]
        print(materialize(args.model, args.checkpoint, args.asset_root, args.panel,
              devices=devices, native_frame_chunk=args.frame_chunk, cpu_threads=args.cpu_threads))


if __name__ == '__main__':
    main()
