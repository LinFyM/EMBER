"""Fixed source/r8 assets and the sole frozen-feedback validation phase."""
from pathlib import Path
from ember.reference.contract import ROOT as REFERENCE_ROOT
from ember.pi05_source_checkpoint import read_json
BASIS_SCHEMA='ember_source_control_linear64_r8_seed7_v1'
BASIS=REFERENCE_ROOT.parent/'control_linear64_seed7/basis/state.pt'
INITIAL_READER=REFERENCE_ROOT.parent/'video_initial_seed7/checkpoints/update_00000900/state.pt'
REVISION_CHECKPOINT=REFERENCE_ROOT.parent/'functional_feedback_seed7/checkpoints/update_00000180/state.pt'
PRACTICE_STEPS=8192
CHECKPOINT_SCHEMA='ember_source_functional_feedback_r8_seed7_v1'
PHASE='functional_feedback_validation180'


def active_run(root,original_run=None):
    run=original_run or read_json(Path(root)/'run_contract.json')
    if run['phase']!=PHASE or run['revision_checkpoint']!=str(REVISION_CHECKPOINT):
        raise ValueError('only the fixed180 validation400 contract is active')
    return run


def request_identity(condition,role,ordinal=0):
    return condition['id']+(f'_practice_{ordinal:03d}' if role=='practice' else '_query')
