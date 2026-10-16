"""Only the first complete non-held feedback compilation cycle is active."""
from pathlib import Path
from ember.reference.contract import TASKS,ROOT as REFERENCE_ROOT,OPTIMIZER
SCHEMA='ember_source_feedback_proposals_r8_seed7_v1'
BASIS_SCHEMA='ember_source_control_linear64_r8_seed7_v1'
PARENT_ROOT=REFERENCE_ROOT.parent/'control_space_seed7'
OPERATOR_ROOT=REFERENCE_ROOT.parent/'control_operator_from144_seed7'
LINEAR_ROOT=REFERENCE_ROOT.parent/'control_linear64_seed7'
BASIS=LINEAR_ROOT/'basis/state.pt'
VIDEO_ROOT=REFERENCE_ROOT.parent/'video_initial_seed7'
INITIAL_READER=VIDEO_ROOT/'checkpoints/update_00000900/state.pt'
ROOT=REFERENCE_ROOT.parent/'feedback_proposals_seed7'
META_UPDATES=9
PRACTICE_ROOT=202610161
QUERY_ROOT=202610151
PRACTICE_STEPS=8192
SIGMA=.25
LIMITS=dict(optimizer_updates=9,FM_queries=1008,query_episodes=144,conditions=36,
    physical_submissions=2848,control_steps=352672,dummy_steps=28480,total_steps=355712,
    engineering_extra_attempts=16,added_peak_GiB=48,dense_H_GiB=16)

def condition_id(task):return f'task_{task:04d}_v0'
def condition_root(root,task):return Path(root)/'conditions'/condition_id(task)
def checkpoint(root,update):return Path(root)/'checkpoints'/f'meta_update_{update:08d}'/'state.pt'
