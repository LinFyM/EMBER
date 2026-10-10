"""CPU consumption of paired originals, endpoint changes and actual billing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from .contract import TASKS, MT_PATH


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


def parameter_changes(parent, candidate, mt):
    squared = {side: 0. for side in ('A','B')}; scaled = 0.; count = 0
    for name, value in candidate.items():
        delta = value.float() - parent[name].float()
        side = 'A' if '.lora_A.' in name else 'B'
        squared[side] += float(delta.square().sum())
        scale = max(float(mt[name].float().square().mean().sqrt()),1e-6)
        scaled += float((delta / scale).square().sum()); count += delta.numel()
    return dict(factor_change_L2={k:v**.5 for k,v in squared.items()},
                fixed_MT_scaled_coordinate_RMS=(scaled/count)**.5, valid_coordinates=count)


def billing(root):
    paths = sorted(Path(root).glob('profile_*/exit.json')) + sorted((Path(root)/'jobs').glob('*/attempt_*/exit.json'))
    rows = [dict(path=str(p), **read_json(p)) for p in paths]
    return dict(GPU_hours=sum(r['GPU_hours'] for r in rows), jobs=len(rows),
                failed_jobs=sum(r['exit_code'] != 0 for r in rows), raw_receipts=rows)


def teacher_event(event_root,task,ordinal,mt):
    event = read_json(event_root/'event.json')
    data = read_json(event_root/'readout/aggregate.json')
    rows = data['rows']; raw=[{**r,'global_task_id':task} for r in rows]
    parent = mt if event['parent_ref']=='MT300' else load_file(event['parent'])
    effects = {}
    for node in (160,480):
        effects[str(node)] = {pool: paired(
            [r for r in rows if r['pool']==pool and r['endpoint']=='parent'],
            [r for r in rows if r['pool']==pool and r['endpoint']==str(node)])
            for pool in ('selection','audit')}
        weights = load_file(str(event_root/'teacher/checkpoints'/f'update_{node:08d}'/'lora.safetensors'))
        effects[str(node)]['parameters'] = parameter_changes(parent, weights, mt)
    adjacent = paired([r for r in rows if r['pool']=='audit' and r['endpoint']=='160'],
              [r for r in rows if r['pool']=='audit' and r['endpoint']=='480'])
    result=dict(task_id=task,event_ordinal=ordinal,teaching_demo=event['teacher_demo'],
        rec_labels=event['rec_labels'],keep_labels=event['keep_labels'],q_T=data['q_T'],
        effects=effects,adjacent160_480=adjacent,missing_label=False)
    return result,raw


def no_supply(events):
    return all(r['effects'][str(node)][pool]['net']<=0 for r in events
               if not r.get('missing_label') for node in (160,480) for pool in ('selection','audit'))


def teacher_summary(root, *, refreshed=False):
    root=Path(root);mt=load_file(str(MT_PATH));events=[];raw=[]
    for task in TASKS:
        for ordinal in range(3 if refreshed else 2):
            event_root=root/'events'/f'task_{task:04d}_event_{ordinal:02d}'
            if not (event_root/'event.json').exists():
                if ordinal!=2:raise ValueError('teacher supply needs the full eight-event batch')
                events.append(dict(task_id=task,event_ordinal=ordinal,missing_label=True));continue
            event,rows=teacher_event(event_root,task,ordinal,mt)
            events.append(event);raw.extend(rows)
    absent=no_supply(events)
    result=dict(complete=True,events=events,all_raw_rows=raw,no_beneficial_teacher_supply=absent,
        interpretation='finite train paired selection4/audit16; neither formal400 nor evidence of video learning',
        teaching_split_not_all_action_label_holdout=True,learning_tasks=list(TASKS),
        absent_suite_coverage=['libero_spatial'],billing=billing(root),
        scientific_stop_reason='consume absent beneficial teacher supply before G/pi investment' if absent else None)
    write_json_atomic(root/'teacher_summary.json',result)
    return result


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
        summaries[version]['complete_U_diagnostics']=complete_U_diagnostics(root,version)
        raw_by_version[version]=rows
    a,b=map(success_index,(raw_by_version['local128'],raw_by_version['RL16']))
    result=dict(complete=True,versions=summaries,raw_rows=raw_by_version,video_controls=video_controls(root),
        adjacent=dict(retained=len(a&b),gained=len(b-a),lost=len(a-b),churn=len(a^b),success_set_overlap=len(a&b)),
        fixed_checkpoints_only=True,finite_train_diagnostic_not_paired400=True,
        absent_suite_coverage=['libero_spatial'],billing=billing(root))
    write_json_atomic(root/'report_summary.json',result)
    return result
