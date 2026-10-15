"""The single bounded source/r8 control-function learning window."""
from pathlib import Path
from ember.reference.contract import TASKS, ROOT as REFERENCE_ROOT, OPTIMIZER

SCHEMA = 'ember_source_control_space_r8_seed7_v1'
ROOT = REFERENCE_ROOT.parent/'control_space_seed7'
LIMITS = dict(optimizer_updates=144, observation_queries=4032, response_pairs=8064,
    physical_submissions=592,control_steps=211840,dummy_steps=5920,total_steps=217760,
    engineering_extra_attempts=16,added_peak_GiB=12)
SAVES = (36,72,108,144)


def queue_path(root):
    return Path(root)/'readout.sqlite'
