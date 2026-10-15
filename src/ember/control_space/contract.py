"""One fresh closed-form64 basis, with an independent fixed readout ledger."""
from pathlib import Path
from ember.reference.contract import TASKS, ROOT as REFERENCE_ROOT
SCHEMA='ember_source_control_linear64_r8_seed7_v1'
PARENT_ROOT=REFERENCE_ROOT.parent/'control_space_seed7'
OPERATOR_ROOT=REFERENCE_ROOT.parent/'control_operator_from144_seed7'
ROOT=REFERENCE_ROOT.parent/'control_linear64_seed7'
LIMITS=dict(closed_form_fits=1,optimizer_updates=0,physical_submissions=512,control_steps=179840,
    dummy_steps=5120,total_steps=184960,engineering_extra_attempts=16,added_peak_GiB=4)

def queue_path(root):return Path(root)/'readout.sqlite'
