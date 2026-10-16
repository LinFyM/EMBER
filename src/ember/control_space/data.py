"""Freeze a task-disjoint weighted fit manifest before reading any factor tensor."""
from pathlib import Path
from collections import Counter
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.reference.contract import ASSETS
from .contract import SCHEMA,PHASE,LIMITS,TASKS,FIT_TASKS,BANK_ROOT,CENTER_REFERENCE,REFERENCE_ROOT


def exclusion_bank(bank):
    if len(bank['tasks'])!=36 or {t['task'] for t in bank['tasks']}!=set(TASKS+FIT_TASKS):
        raise ValueError('original audited36 bank changed')
    tasks=[t for t in bank['tasks'] if t['task'] in FIT_TASKS];points=[]
    for item in tasks:
        others=[p for p in item['checkpoints'] if p['update']!=item['selected_update']]
        points.append(dict(task=item['task'],point=0,path=item['selected_path'],update=item['selected_update'],weight=(.5 if others else 1.)/28))
        points.extend(dict(task=item['task'],point=1,path=p['path'],update=p['update'],weight=.5/28/len(others)) for p in others)
    if (len(points)!=208 or sum(p['path'] is None for p in points)!=5
            or len({(p['task'],p['update']) for p in points})!=208):raise ValueError('the explicit28 bank must have203 nonzero plus five identities')
    mass=Counter()
    for p in points:mass[p['task']]+=p['weight']
    if set(mass)!=set(FIT_TASKS) or any(abs(w-1/28)>1e-12 for w in mass.values()):raise ValueError('task weights changed')
    return dict(bank, tasks=tasks),points


def audit_exclusion():
    audit=read_json(ASSETS/'configs/pi05_source_corpus_v1/overlap_audit.json')
    suites=('libero_spatial','libero_object','libero_goal','libero_10')
    specs={10*suites.index(t['suite'])+t['task_id']:t['specification'] for t in audit['target_tasks']}
    specs.update({40+t['task_id']:t['specification'] for t in audit['source_tasks']})
    exact=[(a,b) for a in TASKS for b in FIT_TASKS if specs[a]==specs[b]]
    declared=[]
    for group in audit['equivalences']:
        target=group['target'];a=10*suites.index(target['suite'])+target['task_id']
        declared.extend((a,40+b) for b in group['source_task_ids'] if a in TASKS and 40+b in FIT_TASKS)
    if exact or declared:raise ValueError(f'excluded-task equivalent in fit allowlist: {exact+declared}')
    return dict(authority=str(ASSETS/'configs/pi05_source_corpus_v1/overlap_audit.json'),
        compared_fields='existing full specification and declared equivalences',cross_set_exact_duplicates=exact,declared_equivalences=declared)


def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    bank,points=exclusion_bank(read_json(BANK_ROOT/'bank.json'))
    environment=read_json(BANK_ROOT/'run_contract.json')['environment']
    environment=dict(environment,tasks=[t for t in environment['tasks'] if t['global_task_id'] in TASKS])
    if {t['global_task_id'] for t in environment['tasks']}!=set(TASKS):raise ValueError('query environment must be the original FIRST_TASKS')
    manifest=dict(schema='ember_control_task_exclusion8_fit_manifest_v1',fit_tasks=list(FIT_TASKS),excluded_tasks=list(TASKS),
        excluded_points=63,points=points,source_center_checkpoint=str(CENTER_REFERENCE),audit=audit_exclusion(),
        units='76 tensor RMS from fit28 only; mean of valid scalar squares; sqrt;floor1e-6',mean_subtracted=False)
    contract=dict(schema=SCHEMA,phase=PHASE,limits=LIMITS,environment=environment,fit_manifest=str(root/'fit_manifest.json'),
        basis_checkpoint=str(root/'basis/state.pt'),fit_tasks=list(FIT_TASKS),query_tasks=list(TASKS),
        reference_root=str(REFERENCE_ROOT),seed=7,shared_updates=0,held=False,Test=False)
    for name,value in (('fit_bank.json',bank),('fit_manifest.json',manifest),('run_contract.json',contract)):
        path=root/name
        if path.exists() and read_json(path)!=value:raise ValueError('immutable internal exclusion contract changed')
        if not path.exists():write_json_atomic(path,value)
    return contract
