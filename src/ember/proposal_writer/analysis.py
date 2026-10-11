"""CPU consumption of paired originals, endpoint changes and actual billing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from .contract import TASKS, SCHEMA


def paired(parent, candidate):
    key = lambda r: (r.get('global_task_id', r['task_id']), r['init_state_id'])
    p, c = {key(r): r for r in parent}, {key(r): r for r in candidate}
    if len(p) != len(parent) or len(c) != len(candidate) or set(p) != set(c):
        raise ValueError('raw rows are not a strict unique paired pool')
    for k in p:
        if (p[k]['env_seed'], p[k]['policy_seed_root']) != (c[k]['env_seed'], c[k]['policy_seed_root']):
            raise ValueError('paired policy/environment RNG changed')
    ps = {k for k, r in p.items() if r['success']}; cs = {k for k, r in c.items() if r['success']}
    changes = [np.mean((np.asarray(c[k]['first_native_normalized_actions']) -
                       np.asarray(p[k]['first_native_normalized_actions'])) ** 2) for k in p
               if 'first_native_normalized_actions' in p[k] and 'first_native_normalized_actions' in c[k]]
    return dict(rows=len(p), parent_success=len(ps), candidate_success=len(cs), net=len(cs)-len(ps),
        retained=len(ps & cs), gained=len(cs-ps), lost=len(ps-cs), churn=len(ps ^ cs),
        success_set_overlap=len(ps & cs), gained_states=sorted([list(k) for k in cs-ps]),
        lost_states=sorted([list(k) for k in ps-cs]), retained_states=sorted([list(k) for k in ps & cs]),
        first_native_first5_RMSE=float(np.sqrt(np.mean(changes))) if changes else None)


def parameter_changes(parent, candidate, scales=None):
    squared = {side: 0. for side in ('A','B')}; scaled = 0.; count = 0
    for name, value in candidate.items():
        delta = value.float() - parent[name].float()
        side = 'A' if '.lora_A.' in name else 'B'
        squared[side] += float(delta.square().sum())
        if scales is not None:
            scale = float(scales[name])
            scaled += float((delta / scale).square().sum())
        count += delta.numel()
    return dict(factor_change_L2={k:v**.5 for k,v in squared.items()},
                fixed_S_prox_coordinate_RMS=(scaled/count)**.5 if scales is not None else None, valid_coordinates=count)


def billing(root):
    paths = sorted(Path(root).glob('profile_*/exit.json')) + sorted((Path(root)/'jobs').glob('*/attempt_*/exit.json'))
    rows = [dict(path=str(p), **read_json(p)) for p in paths]
    contract_path = Path(root)/'run_contract.json'
    carry = read_json(contract_path)['limits'].get('carry_in_GPU_hours',0.) if contract_path.exists() else 0.
    current = sum(r['GPU_hours'] for r in rows)
    return dict(GPU_hours=carry+current, current_root_GPU_hours=current, carry_in_GPU_hours=carry, jobs=len(rows),
                GPU_hours_limit=None, GPU_hours_hard_cap_active=False,
                failed_jobs=sum(r['exit_code'] != 0 for r in rows),
                attempts=sum(r.get('attempts',1) for r in rows),
                failed_attempts=sum(r.get('failed_attempts',int(r['exit_code'] != 0)) for r in rows),raw_receipts=rows)


def teacher_summary(root):
    """Consume actual evaluated nodes, scientific omissions and paired all36 reports."""
    root=Path(root);contract=read_json(root/'run_contract.json')
    selection=[];reports=[];tasks=[]
    initial=load_file(str(root/'initial.safetensors'));prox=read_json(root/'coordinates.json')['S_prox']
    for task in TASKS:
        directory=root/'teachers'/f'task_{task:04d}'
        chosen=read_json(directory/'E.json')
        candidates=chosen['evaluated_candidates']
        if (not chosen['complete'] or sorted([p['update'] for p in candidates]+chosen['scientifically_omitted_nodes'])
                !=contract['teacher']['readout_updates']):
            raise ValueError('all teacher nodes need actual readout or registered scientific omission')
        if any(len(p['rows'])!=contract['teacher']['selection_per_node'] for p in candidates):
            raise ValueError('teacher choice requires complete selection panels')
        selection.extend(row for candidate in candidates for row in candidate['rows'])
        rows=read_json(directory/'report/complete.json')['rows'];reports.extend(rows)
        report=paired([r for r in rows if r['arm']=='MT'],[r for r in rows if r['arm']=='E'])
        weights=load_file(chosen['weights'])
        tasks.append(dict(task=task,selected_update=chosen['selected_update'],
            node_successes={str(p['update']):p['selection_successes'] for p in candidates},
            independent_report=report,parameters=parameter_changes(initial,weights,prox),
            evaluated_candidate_references=[p['reference'] for p in candidates],
            scientifically_omitted_nodes=chosen['scientifically_omitted_nodes'],saturation_proof=chosen['saturation_proof'],
            selection_uses='teacher choice only; fixed first MT rollout also supplies bootstrap',
            bootstrap=chosen['bootstrap']))
    expected_selection=sum(len(t['node_successes']) for t in tasks)*contract['teacher']['selection_per_node']
    expected_report=36*2*contract['teacher']['report_per_arm']
    if len(selection)!=expected_selection or len(reports)!=expected_report:
        raise ValueError('full36 teacher raw row coverage incomplete')
    aggregate=paired([r for r in reports if r['arm']=='MT'],[r for r in reports if r['arm']=='E'])
    result=dict(complete=True,schema_version=SCHEMA,tasks=tasks,selection_raw_rows=selection,report_raw_rows=reports,
        independent_report=aggregate,initial_G_parent='MT only',initial_G_history_episodes=len(TASKS),
        teacher_supports_video_writer=False,interpretation='teacher control and label supply only, not G or video capability',
        no_nonMT_selected=all(t['selected_update']==0 for t in tasks),
        selected_teacher_report_net=aggregate['net'],billing=billing(root))
    write_json_atomic(root/'teacher_summary.json',result)
    import json
    for name,rows in [('teacher_selection_raw_rows',selection),('teacher_report_raw_rows',reports)]:
        with (root/f'{name}.jsonl').open('w') as handle:
            for row in rows:handle.write(json.dumps(row)+'\n')
    return result


def freeze_bank_scale(root):
    """Task-equal E−MT statistic fixed before initialization; reports never set S_G."""
    import math
    root=Path(root);path=root/'bank_scale.json'
    if path.exists():return read_json(path)
    initial=load_file(str(root/'initial.safetensors'));prox=read_json(root/'coordinates.json')['S_prox']
    mean={name:0. for name in initial};sources=[]
    for task in TASKS:
        record=read_json(root/'teachers'/f'task_{task:04d}'/'E.json')
        bootstrap=read_json(root/'bootstrap'/f'task_{task:04d}'/'complete.json')
        if not bootstrap['complete'] or not Path(bootstrap['history']).is_file():
            raise ValueError('all36 initial MT histories required before fixed G scale')
        target=load_file(record['weights']);weight=1/len(TASKS)
        for name in mean:mean[name]+=weight*float((target[name].float()-initial[name].float()).square().mean())
        sources.append(dict(task=task,parent='MT300',E=record['reference'],weight=weight))
    scales={name:max(math.sqrt(value),1e-3*prox[name]) for name,value in mean.items()}
    record=dict(schema_version=SCHEMA,S_G=scales,edit_mean_square=mean,S_prox=prox,sources=sources,
        task_weight=1/len(TASKS),initial_parent='MT300 only',history_count_does_not_weight_parent=True,
        floor_active=[name for name in mean if math.sqrt(mean[name])<=1e-3*prox[name]],
        training_selection_only=True,independent_teacher_report_not_used=True,fixed_before_first_G_update=True,
        student_refresh_reestimates_scale=False)
    write_json_atomic(path,record)
    return record



def report_summary(root):
    """All registered finite train conditions, with logical rows and paired success sets."""
    root=Path(root);contract=read_json(root/'run_contract.json');spec=contract['training_report']
    versions={};raw={}
    for stage,update in spec['versions']:
        version=f'{stage}_{update:08d}';conditions=[];rows=[]
        for task in TASKS:
            for ordinal in range(spec['conditions_per_task']):
                destination=root/'reports'/version/'correct'/f'task_{task:04d}_condition_{ordinal:03d}'
                read_json(destination/'complete.json');data=read_json(destination/'query.json')['rows']
                parent=[r for r in data if r['arm']=='MT'];selected=[r for r in data if r['arm']=='selected']
                result=paired(parent,selected);meta=read_json(destination/'path.json')
                conditions.append(dict(task=task,ordinal=ordinal,paired=result,selected=meta['selected'],
                    actual_environment_steps=meta['environment_steps'],completed_rounds=meta['completed_rounds']))
                rows.extend(dict(**r,report_ordinal=ordinal) for r in data)
        versions[version]=dict(conditions=conditions,rows=sum(c['paired']['rows'] for c in conditions),
            retained=sum(c['paired']['retained'] for c in conditions),gained=sum(c['paired']['gained'] for c in conditions),
            lost=sum(c['paired']['lost'] for c in conditions),churn=sum(c['paired']['churn'] for c in conditions),
            selected_success=sum(c['paired']['candidate_success'] for c in conditions),
            MT_success=sum(c['paired']['parent_success'] for c in conditions))
        raw[version]=rows
    result=dict(complete=True,versions=versions,raw_rows=raw,all36=True,
        finite_train_diagnostic_not_formal_qualification=True,billing=billing(root))
    write_json_atomic(root/'report_summary.json',result)
    return result
