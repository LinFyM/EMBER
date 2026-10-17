"""The bounded native-operator first-write window, with read-only shared assets."""
from pathlib import Path
from ember.reference.contract import ASSETS, SOURCE, NORMALIZATION, TASKS, ROOT as REFERENCE_ROOT

BASE = REFERENCE_ROOT.parent
PREVIOUS_ROOT = BASE/'source_weight_compiler_seed7'
VIDEO_ROOT = BASE/'video_initial_seed7'
INITIAL_READER = VIDEO_ROOT/'checkpoints/update_00000900/state.pt'
H_ROOT = BASE/'feedback_proposals_seed7'
ROOT = Path('/data1/user/ymdai/ember_runs/source_native_operator_compiler_20261017/first_write_seed7')
SCHEMA = 'ember_source_native_operator_first_write_r8_seed7_v1'
PHASE = 'source_native_operator_first_write'
OBJECTIVE = 'real_source_full10flow_operator_pullback_rank8_QR2_functional_alpha_native_FM_v1'
ARM = 'native_operator180'
FINAL = 180
PERMANENT = (90, 180)
STATES = (0, 4, 8, 12)
IDENTITY_TASKS = (42, 55, 64, 96, 101)
OPTIMIZER = dict(lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=1e-4)
LIMITS = dict(optimizer_updates=180, FM_queries=20160, logical_conditions=720,
    physical_submissions=160, control_steps=59200, dummy_steps=1600, total_steps=60800,
    engineering_extra_attempts=16, added_peak_GiB=12, profile_joint_equivalents=4,
    profile_FM_queries=448, M_cache_bytes=2**30, frozen_H_cache_bytes=4*2**30)


def queue_path(root):
    return Path(root)/'evaluation.sqlite'


def checkpoint_path(root, update):
    if update not in PERMANENT:
        raise ValueError('only recovery90 and the unique terminal180 are permanent checkpoints')
    return Path(root)/'checkpoints'/f'update_{update:08d}'/'state.pt'
