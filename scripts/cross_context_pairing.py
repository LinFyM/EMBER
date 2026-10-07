"""Finite matched-pairing study; scientific scope is fixed in its active design."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ember.pi05_source_checkpoint import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "bank"))
    parser.add_argument("--spec", type=Path, default=Path(__file__).resolve().parents[1] / "configs/cross_context_pairing_v1/learning_spec.json")
    parser.add_argument("--asset-root", type=Path, default=Path("/data1/user/ymdai/projects/EMBER"))
    parser.add_argument("--arm", choices=("Within", "Product", "parent"), required=True)
    parser.add_argument("--phase", choices=("formal", "smoke", "profile"), default="formal")
    parser.add_argument("--attempt", default="formal")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--microbatch", type=int, default=28)
    parser.add_argument("--frame-chunk", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--slot", choices=("official", "train"), default="official")
    args = parser.parse_args()
    spec = read_json(args.spec)
    if spec["task"] != "cross_context_pairing_20261008":
        raise ValueError("entry belongs only to the registered pairing study")
    if args.command == "train":
        if args.arm == "parent" or args.resume is None or not 1 <= args.microbatch <= 28 or args.frame_chunk < 1:
            raise ValueError("training requires its paired arm and full ECP source")
        from ember.cross_context_pairing.training import execute
        execute(spec, args)
    else:
        from ember.cross_context_pairing.readout import materialize
        materialize(spec, args)


if __name__ == "__main__":
    main()
