#!/usr/bin/env python3
"""Prepare fixed transition-study banks/panels; rollout uses evaluate_pi05.py."""
from pathlib import Path
import argparse
import json
import sys

from ember.operator_writer import transition_read_eval as study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("geometry", "materialize", "register-reader", "prepare"))
    parser.add_argument("--arm", choices=("L", "R", "parent"))
    parser.add_argument("--panel", choices=("validation", "seen12"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--frame-chunk", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--gpu-indices", default="0")
    parser.add_argument("--replicas-per-gpu", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--envs-per-replica", type=int, help="Pure physical environment batch; default official capacity8")
    args = parser.parse_args()
    if args.phase == "geometry":
        print(json.dumps({"study": study.STUDY, "new_rows": 836, "full": 25, "compact": 811,
            "panels": {panel: [len(study.source_geometry(panel)[1]), len(study.source_geometry(panel)[2])]
                       for panel in ("validation", "seen12")}}, sort_keys=True))
        return
    if args.arm is None or args.panel is None:
        parser.error("this phase requires its registered arm and panel")
    study.paths(args.arm, args.panel)
    if args.phase == "prepare":
        from ember.pi05_eval.preparation import prepare_evaluation_run
        result = prepare_evaluation_run(study.prepare_arguments(args.arm, args.panel,
            gpu_indices=args.gpu_indices, replicas=args.replicas_per_gpu,
            envs_per_replica=args.envs_per_replica), repo_root=study.REPO, command=sys.argv)
    else:
        checkpoint = study.PARENT if args.arm == "parent" else args.checkpoint
        if checkpoint is None:
            parser.error("L/R consumers require their complete checkpoint270")
        if args.phase == "register-reader":
            if args.arm != "R":
                parser.error("only R has a streaming reader manifest")
            result = study.register_bank(args.arm, args.panel, checkpoint, frame_chunk=args.frame_chunk)
        else:
            if args.asset_root is None:
                parser.error("materialization requires the canonical asset root")
            result = study.materialize(args.arm, args.panel, checkpoint, asset_root=args.asset_root,
                devices=tuple(args.devices.split(",")), frame_chunk=args.frame_chunk, cpu_threads=args.cpu_threads)
    print(json.dumps(str(result) if isinstance(result, Path) else result, sort_keys=True))


if __name__ == "__main__":
    main()
