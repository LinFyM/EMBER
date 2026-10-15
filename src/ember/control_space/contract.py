"""One operator-objective continuation and its fixed readout budget."""
from pathlib import Path
from ember.reference.contract import TASKS, ROOT as REFERENCE_ROOT, OPTIMIZER
SCHEMA='ember_source_control_operator_from144_r8_seed7_v1'
PARENT_ROOT=REFERENCE_ROOT.parent/'control_space_seed7'
ROOT=REFERENCE_ROOT.parent/'control_operator_from144_seed7'
PARENT_UPDATE=144
LIMITS=dict(optimizer_updates=10800,parameter_pairs=86400,physical_submissions=512,
    control_steps=179840,dummy_steps=5120,total_steps=184960,engineering_extra_attempts=16,added_peak_GiB=12)
MAX_UPDATE=PARENT_UPDATE+LIMITS['optimizer_updates']

def queue_path(root):return Path(root)/'readout.sqlite'
