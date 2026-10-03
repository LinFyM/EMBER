"""Temporary entrypoint for the fixed 48-case same-input attention diagnostic."""
import argparse
import json
from pathlib import Path

from ember.pi05_eval.image_attention_contract import ROOT, launch_linked, prepare, run_linked_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--gpu-indices', required=True)
    prep.add_argument('--replicas-per-gpu', type=int, required=True)
    prep.add_argument('--env-batch', type=int, default=4)
    commands.add_parser('start')
    worker = commands.add_parser('worker')
    worker.add_argument('--output-dir', type=Path, required=True)
    worker.add_argument('--worker-id', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(tuple(int(v) for v in args.gpu_indices.split(',')),
                                 args.replicas_per_gpu, env_batch=args.env_batch)))
    elif args.command == 'start':
        launch_linked(Path(__file__).resolve())
    else:
        run_linked_worker(args.output_dir.resolve(), args.worker_id)


if __name__ == '__main__':
    main()
