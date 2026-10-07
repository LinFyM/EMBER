"""Thin, finite entry for this registered study; removed at its final retirement."""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("train", "profile", "encode"))
    parser.add_argument("--arm", choices=("M", "V"), required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--attempt", default="formal128")
    parser.add_argument("--kind", choices=("formal", "smoke", "profile"), default="formal")
    parser.add_argument("--microbatch", type=int, default=14)
    parser.add_argument("--frame-chunk", type=int, default=16)
    parser.add_argument("--frame-chunks-per-rank", type=lambda x: tuple(map(int, x.split(","))))
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--stop-after", type=int, default=128)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--condition", choices=("correct", "same_task_other"), default="correct")
    parser.add_argument("--profile-configs", default="7x16,14x32,28x32")
    args = parser.parse_args()
    if args.phase == "profile":
        args.kind = "profile"
        from .profile import run_profile
        run_profile(args)
    elif args.phase == "train":
        from .training import run_training
        run_training(args)
    else:
        if args.checkpoint is None:
            parser.error("encoding needs the registered terminal checkpoint")
        from .encoding import encode
        print(encode(args), flush=True)


if __name__ == "__main__":
    main()
