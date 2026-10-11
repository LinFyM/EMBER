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
    """Consume every node and the independent paired report across all36 tasks."""
    root=Path(root);contract=read_json(root/'run_contract.json')
    selection=[];reports=[];tasks=[]
    initial=load_file(str(root/'initial.safetensors'));prox=read_json(root/'coordinates.json')['S_prox']
    for task in TASKS:
        directory=root/'teachers'/f'task_{task:04d}'
        chosen=read_json(directory/'E.json')
        if not chosen['complete'] or [p['update'] for p in chosen['parents']] != contract['teacher']['readout_updates']:
            raise ValueError('complete registered common-E parent bank required')
        selection.extend(row for parent in chosen['parents'] for row in parent['rows'])
        rows=read_json(directory/'report/complete.json')['rows'];reports.extend(rows)
        report=paired([r for r in rows if r['arm']=='MT'],[r for r in rows if r['arm']=='E'])
        weights=load_file(chosen['weights'])
        tasks.append(dict(task=task,selected_update=chosen['selected_update'],
            node_successes={str(p['update']):p['selection_successes'] for p in chosen['parents']},
            independent_report=report,parameters=parameter_changes(initial,weights,prox),
            parent_references=[p['reference'] for p in chosen['parents']],
            checkpoint_selection_is_training_data=True,selection_uses=f"{contract['teacher']['selection_H_per_node']} selection/H rows, equal two folds"))
    expected_selection=36*len(contract['teacher']['readout_updates'])*contract['teacher']['selection_H_per_node']
    expected_report=36*2*contract['teacher']['report_per_arm']
    if len(selection)!=expected_selection or len(reports)!=expected_report:
        raise ValueError('full36 teacher raw row coverage incomplete')
    aggregate=paired([r for r in reports if r['arm']=='MT'],[r for r in reports if r['arm']=='E'])
    result=dict(complete=True,schema_version=SCHEMA,tasks=tasks,selection_raw_rows=selection,report_raw_rows=reports,
        independent_report=aggregate,bank_task_weight=1/36,parent_weight='equal actual registered parent within task',
        H_weight='uniform within parent; H count does not weight parent',
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
    """Exactly task→parent equal E−P statistic; independent report never sets S_G."""
    import math
    root=Path(root);path=root/'bank_scale.json'
    if path.exists():return read_json(path)
    initial=load_file(str(root/'initial.safetensors'));prox=read_json(root/'coordinates.json')['S_prox']
    mean={name:0. for name in initial};sources=[]
    for task in TASKS:
        record=read_json(root/'teachers'/f'task_{task:04d}'/'E.json')
        target=load_file(record['weights']);parents=record['parents']
        if not parents or any(not p['histories'] for p in parents):raise ValueError('G requires nonempty actual H for every parent')
        weight=1/(len(TASKS)*len(parents))
        for parent in parents:
            state=load_file(parent['weights'])
            for name in mean:mean[name]+=weight*float((target[name].float()-state[name].float()).square().mean())
            sources.append(dict(task=task,parent=parent['reference'],E=record['reference'],weight=weight))
    scales={name:max(math.sqrt(value),1e-3*prox[name]) for name,value in mean.items()}
    record=dict(schema_version=SCHEMA,S_G=scales,edit_mean_square=mean,S_prox=prox,sources=sources,
        task_weight=1/len(TASKS),parent_weight='equal all actual registered parents',history_count_does_not_weight_parent=True,
        floor_active=[name for name in mean if math.sqrt(mean[name])<=1e-3*prox[name]],
        training_selection_only=True,independent_teacher_report_not_used=True,fixed_before_first_G_update=True,
        student_refresh_reestimates_scale=False)
    write_json_atomic(path,record)
    return record


def report_rows(root,version):
    all_rows, conditions = [], []
    for task in TASKS:
        for ordinal in (0,1):
            destination=root/'reports'/version/'correct'/f'task_{task:04d}_condition_{ordinal:02d}'
            read_json(destination/'complete.json');rows=read_json(destination/'query.json')['rows']
            selected=[r for r in rows if r['arm']=='selected']; parent=[r for r in rows if r['arm']=='MT']
            stats=paired(parent,selected)
            record=read_json(destination/'path.json')
            conditions.append(dict(task_id=task,ordinal=ordinal,paired=stats,
                final_source=record['selected'],environment_steps=record['environment_steps'],
                practiced_P=record['practiced_P'],valid_U_count=len(record['valid_U']),suite=rows[0]['suite']))
            all_rows.extend([{**r,'report_ordinal':ordinal} for r in rows])
    return all_rows,conditions


def version_summary(conditions):
    result=dict(conditions=conditions,selected_success=sum(c['paired']['candidate_success'] for c in conditions),
        MT_success=sum(c['paired']['parent_success'] for c in conditions),rows=48,
        gained=sum(c['paired']['gained'] for c in conditions),lost=sum(c['paired']['lost'] for c in conditions),
        retained=sum(c['paired']['retained'] for c in conditions),churn=sum(c['paired']['churn'] for c in conditions))
    for group in ('task_id','suite'):
        result['per_'+group] = {str(value):dict(
            rows=sum(c['paired']['rows'] for c in conditions if c[group]==value),
            selected_success=sum(c['paired']['candidate_success'] for c in conditions if c[group]==value),
            MT_success=sum(c['paired']['parent_success'] for c in conditions if c[group]==value),
            gained=sum(c['paired']['gained'] for c in conditions if c[group]==value),
            lost=sum(c['paired']['lost'] for c in conditions if c[group]==value)) for value in {c[group] for c in conditions}}
    return result


def complete_U_diagnostics(root,version):
    diagnostics=[]
    for task in (12,32):
        destination=root/'reports'/version/'correct'/f'task_{task:04d}_condition_00'
        data=read_json(destination/'U_diagnostic.json')['rows']; path=read_json(destination/'path.json')
        rates={ref:sum(int(r['success']) for r in data if r['reference']==ref)/6 for ref in path['valid_U']}
        U=max(rates.values()); P=max(rates[ref] for ref in path['practiced_P']); selected=rates[path['selected']]
        diagnostics.append(dict(task_id=task,complete_U=True,fixed_policy_success_rates=rates,
            MT=rates['MT300'],selected=selected,generation_gain=U-rates['MT300'],
            practice_retention_loss=U-P,final_selection_loss=P-selected,
            empirical_maximum_sampling_bias=True,not_per_state_success_union=True))
    return diagnostics


def video_controls(root):
    controls=[]
    for task in (12,32):
        original=root/'reports'/'RL16'/'correct'/f'task_{task:04d}_condition_00'
        correct=[r for r in read_json(original/'query.json')['rows'] if r['arm']=='selected']
        for arm in ('other','wrong'):
            destination=root/'reports'/'RL16'/arm/f'task_{task:04d}_condition_00'
            rows=read_json(destination/'query.json')['rows']
            selected=[r for r in rows if r['arm']=='selected'];mt=[r for r in rows if r['arm']=='MT']
            controls.append(dict(task_id=task,video_arm=arm,paired_MT=paired(mt,selected),
                                 paired_correct=paired(correct,selected),raw_rows=rows))
    return controls


def success_index(rows):
    return {(r['global_task_id'],r['report_ordinal'],r['init_state_id']) for r in rows
            if r['arm']=='selected' and r['success']}


def report_summary(root):
    root=Path(root);summaries={};raw_by_version={}
    for version in ('local128','RL16'):
        rows,conditions=report_rows(root,version)
        summaries[version]=version_summary(conditions)
        summaries[version]['complete_U_diagnostics']=complete_U_diagnostics(root,version) if version=='RL16' else []
        raw_by_version[version]=rows
    a,b=map(success_index,(raw_by_version['local128'],raw_by_version['RL16']))
    result=dict(complete=True,versions=summaries,raw_rows=raw_by_version,video_controls=video_controls(root),
        adjacent=dict(retained=len(a&b),gained=len(b-a),lost=len(a-b),churn=len(a^b),success_set_overlap=len(a&b)),
        fixed_checkpoints_only=True,finite_train_diagnostic_not_paired400=True,
        absent_suite_coverage=['libero_spatial'],billing=billing(root))
    write_json_atomic(root/'report_summary.json',result)
    return result
