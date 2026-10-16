"""One internal task-excluded linear64 fit and its fixed128 query readout."""
from pathlib import Path
from ember.reference.contract import ROOT as REFERENCE_ROOT,FIRST_TASKS,NEW_TASKS
BASIS_SCHEMA='ember_source_control_linear64_r8_seed7_v1'
SCHEMA='ember_source_control_task_exclusion8_r8_seed7_v1'
PHASE='task_exclusion8_linear64'
BANK_ROOT=REFERENCE_ROOT.parent/'control_space_seed7'
LINEAR_ROOT=REFERENCE_ROOT.parent/'control_linear64_seed7'
OPERATOR_ROOT=REFERENCE_ROOT.parent/'control_operator_from144_seed7'
CENTER_REFERENCE=LINEAR_ROOT/'basis/state.pt'
ROOT=REFERENCE_ROOT.parent/'control_task_exclusion8_seed7'
TASKS=FIRST_TASKS
FIT_TASKS=NEW_TASKS
LIMITS=dict(closed_form_fits=1,optimizer_updates=0,physical_submissions=144,control_steps=50560,
    dummy_steps=1440,total_steps=52000,engineering_extra_attempts=16,added_peak_GiB=4)
NOISE_ROOT=202610151
STATES=tuple(range(16))
def queue_path(root):return Path(root)/'evaluation.sqlite'
