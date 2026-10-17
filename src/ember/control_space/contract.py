"""The single source-weight-conditioned complete-LoRA learning window."""
from pathlib import Path
from ember.reference.contract import ASSETS,SOURCE,NORMALIZATION,TASKS,ROOT as REFERENCE_ROOT,OPTIMIZER

BASE=REFERENCE_ROOT.parent
PREVIOUS_ROOT=BASE/'source_weight_compiler_seed7'
ROOT=BASE/'source_weight_factor_credit_seed7'
SOURCE_INPUTS=PREVIOUS_ROOT/'source_inputs/state.pt'
OPERATOR_FILE=PREVIOUS_ROOT/'operator_constants.json'
OBJECTIVE='teacher_detached_pair1000_then_native_FM_plus_0.1_complete_BA180_v1'
ARM='factor_credit1180'
VIDEO_ROOT=BASE/'video_initial_seed7'
INITIAL_READER=VIDEO_ROOT/'checkpoints/update_00000900/state.pt'
BANK_ROOT=BASE/'control_space_seed7'
CENTER_REFERENCE=BASE/'control_linear64_seed7/basis/state.pt'
UNITS_FILE=BANK_ROOT/'units.json'
SCHEMA='ember_source_weight_factor_credit_r8_seed7_v1'
PHASE='source_weight_factor_credit'
PERMANENT=(1000,1090,1180)
FINAL=1180
VALIDATION_TASKS=(3,6,11,16,23,26,31,39)
LIMITS=dict(optimizer_updates=1180,BA_conditions=36000,joint_conditions=720,FM_queries=20160,
 physical_submissions=992,control_steps=343840,dummy_steps=9920,total_steps=353760,
 engineering_extra_attempts=16,added_peak_GiB=16,profile_BA_equivalents=4,
 profile_joint_equivalents=2,profile_FM_queries=224,M_cache_bytes=2**30)

def queue_path(root):return Path(root)/'evaluation.sqlite'

def checkpoint_path(root,update):return Path(root)/'checkpoints'/f'update_{update:08d}'/'state.pt'
