"""Prepare the registered frozen-bank subset; execution remains evaluate_pi05.py."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ember.pi05_eval.object_position_transport import REFERENCES, prepare

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--model', choices=REFERENCES, required=True)
parser.add_argument('--layout', choices=('original', 'swapped'), required=True)
parser.add_argument('--gpu', type=int, required=True)
args = parser.parse_args()
print(json.dumps(prepare(args.model, args.layout, args.gpu)))
