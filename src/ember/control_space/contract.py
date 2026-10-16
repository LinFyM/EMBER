"""Fixed source/r8 assets and the sole frozen-feedback validation phase."""
from pathlib import Path
from functools import lru_cache
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


@lru_cache(maxsize=2)
def pending_recovery(root):
    """One immutable, additive authority; it never changes scientific remaining."""
    file=Path(root)/'pending_recovery_contract.json'
    if not file.exists():return None
    saved=read_json(file);ruling=saved['ruling']
    if (saved['schema']!='ember_validation180_named_pending_recovery_v1'
            or ruling['schema']!='ember_main_named_pending_recovery_ruling_v1'
            or saved['root']!=str(Path(root).resolve()) or ruling['revision_update']!=180
            or ruling['recipient']!='01a13dbf-dbc3-7621-82be-87be44a5bca6'):
        raise ValueError('named pending recovery lost its authority/ROOT')
    run=read_json(Path(root)/'run_contract.json');caps=ruling['global_original_caps_unchanged']
    for key,other in (('physical_submissions','submissions'),('control_steps','control'),
            ('dummy_steps','dummy'),('total_steps','joint'),('engineering_extra_attempts','engineering_episode_extras_total')):
        if run['limits'][key]!=caps[other]:raise ValueError('pending amendment must not enlarge global limits')
    if len(saved['pending_requests'])!=4 or sum(len(c['unknown_attempts']) for c in ruling['conditions'])!=16:
        raise ValueError('only four saved panels and sixteen named unknowns may recover')
    return saved


def recovery_panel(root,request):
    saved=pending_recovery(str(Path(root).resolve()))
    if not saved:return None
    named=next((c for c in saved['ruling']['conditions'] if c['pending_identity']==request['identity']),None)
    if not named:return None
    expected=saved['pending_requests'][named['condition']]
    fields={k:v for k,v in expected.items() if k not in ('command','states')}
    if request.get('role')!='practice' or any(request.get(k)!=v for k,v in fields.items()):
        raise ValueError('recovery must keep the actual saved candidate/input identity')
    if 'states' in request:
        if request['states']!=named['states']:raise ValueError('saved recovery state order changed')
    elif request.get('state_id') not in named['states']:raise ValueError('unnamed recovery case')
    return named


def practice_cap(root,request):
    named=recovery_panel(root,request)
    return named['pending_only_physical_cap'] if named else PRACTICE_STEPS


def reported_practice_cap(root,identity):
    saved=pending_recovery(str(Path(root).resolve()))
    named=next((c for c in saved['ruling']['conditions'] if c['condition']==identity),None) if saved else None
    return named['pending_only_physical_cap'] if named else PRACTICE_STEPS


def validate_named_retry(root,spec,db):
    named=recovery_panel(root,spec)
    if not named:return
    case=f"practice_{spec['identity']}_state_{spec['state_id']:03d}"
    original=case+'_attempt_001'
    old=db.execute('SELECT identity,status FROM attempts WHERE case_id=?',(case,)).fetchall()
    if original not in named['unknown_attempts'] or old!=[(original,'reserved_unknown')]:
        raise RuntimeError('named unknown permits one recovery only; adopt valid rows or report the new boundary')
