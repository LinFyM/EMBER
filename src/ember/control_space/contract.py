"""Frozen source-control assets and unchanged task-local proposal units."""
from ember.reference.contract import ROOT as REFERENCE_ROOT
BASIS_SCHEMA='ember_source_control_linear64_r8_seed7_v1'
BASIS=REFERENCE_ROOT.parent/'control_linear64_seed7/basis/state.pt'
INITIAL_READER=REFERENCE_ROOT.parent/'video_initial_seed7/checkpoints/update_00000900/state.pt'
PRACTICE_STEPS=8192
SCHEMA='ember_source_functional_feedback_r8_seed7_v1'
from ember.reference.contract import OPTIMIZER


def active_run(root,original_run=None):
    """Apply the named incident authority without rewriting the original run."""
    from copy import deepcopy
    from pathlib import Path
    from ember.pi05_source_checkpoint import read_json
    root=Path(root);run=deepcopy(original_run or read_json(root/'run_contract.json'))
    file=root/'recovery_contract.json'
    if not file.exists():return run
    recovery=read_json(file);authority=read_json(Path(recovery['authority_path']))
    if (recovery['schema']!='ember_section17_cache_recovery_leg_v1' or not recovery['prepared']
            or recovery['leg']!='cache_recovery_001' or recovery['original_contract']!=str(root/'run_contract.json')
            or recovery['training_checkpoint']!=str(root/'checkpoints/update_00000180/state.pt')
            or recovery['repair_tasks']!=authority['repair_tasks']):
        raise ValueError('named cache recovery lost its immutable scientific authority')
    bounds=authority['authorized_cumulative_bounds']
    expected={k:bounds[v] for k,v in dict(physical_submissions='submissions',control_steps='control',dummy_steps='dummy',
        total_steps='total',engineering_extra_attempts='ordinary_engineering_extra_attempts').items()}
    if recovery['limits']!=expected:raise ValueError('cache incident budget differs from the finite main ruling')
    run['limits'].update(expected);run['cache_recovery']=recovery
    for condition in run['conditions']:
        if condition['task'] in recovery['repair_tasks']:condition['execution_leg']=recovery['leg']
    return run


def request_identity(condition,role,ordinal=0):
    stem=condition['id']+('_'+condition['execution_leg'] if condition.get('execution_leg') else '')
    return stem+(f'_practice_{ordinal:03d}' if role=='practice' else '_query')
