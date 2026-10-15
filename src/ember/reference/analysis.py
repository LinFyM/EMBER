"""Finite-window evidence, including negative tasks and independent report."""
from pathlib import Path
import json
import numpy as np

from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.proposal_writer.analysis import billing
from .contract import TASKS, LIMITS, nodes, selection_decision
from .accounting import summary


def paired(before, after):
    a={r['init_state_id'] for r in before if r['success']}
    b={r['init_state_id'] for r in after if r['success']}
    return dict(before=len(a),after=len(b),retained=sorted(a&b),gained=sorted(b-a),lost=sorted(a-b),
                churn=len(a^b),success_set_overlap=len(a&b),before_successes=sorted(a),after_successes=sorted(b))


def pairing_valid(before, after):
    fields=('suite','task_id','init_state_id','env_seed','policy_seed_root','first_noise_seed')
    a=sorted(before,key=lambda r:r['init_state_id']);b=sorted(after,key=lambda r:r['init_state_id'])
    if len(a)!=len(b):return False
    for x,y in zip(a,b):
        if any(x[k]!=y[k] for k in fields):return False
        if not np.allclose(x['initial_proprio'],y['initial_proprio'],rtol=0,atol=1e-6):return False
    return True


def read_cases(directory,states):
    paths=[directory/f'state_{i:03d}.json' for i in states]
    return [read_json(p) for p in paths] if all(p.exists() for p in paths) else None


def task_evidence(root,task,mt):
    base=root/'selection'/f'task_{task:04d}';candidates={};report={}
    for update in nodes(task):
        rows=read_cases(base/f'update_{update:08d}',range(32))
        if rows is not None:candidates[update]=rows
    scores={u:sum(r['success'] for r in rows) for u,rows in candidates.items()}
    decision=selection_decision(scores,task);selected=decision['selected_update'] if decision else None
    ordered_nodes=sorted(candidates)
    adjacent=[dict(before_update=a,after_update=b,**paired(candidates[a],candidates[b])) for a,b in zip(ordered_nodes,ordered_nodes[1:])]
    if 0 in candidates:
        for rows in candidates.values():
            if not pairing_valid(candidates[0],rows):raise ValueError('candidate selection case pairing changed')
    for arm in ('source','selected'):
        rows=read_cases(root/'report'/f'task_{task:04d}'/arm,range(32,50))
        if rows is not None:report[arm]=rows
    if len(report)==2 and not pairing_valid(report['source'],report['selected']):
        raise ValueError('independent report source/selected pairing changed')
    reference=[r for r in mt if r['global_task_id']==task]
    candidate=[] if selected is None else [r for r in candidates[selected] if r['init_state_id']<16]
    matching=len(reference)==len(candidate)==16 and pairing_valid(reference,candidate)
    result=dict(task_id=task,candidates=scores,decision=decision,adjacent=adjacent,
        selection_vs_source=paired(candidates[0],candidates[selected]) if selected is not None else None,
        MT_pairing_valid=matching,MT_comparison=paired(reference,candidate) if matching else None,
        independent_report=paired(report['source'],report['selected']) if len(report)==2 else None)
    return result,candidates,report


def aggregate_tasks(tasks):
    rows=dict(tasks=[t['task_id'] for t in tasks],
        settled=sum(bool(t['decision'] and t['decision']['settled']) for t in tasks),
        unsettled=[t['task_id'] for t in tasks if not (t['decision'] and t['decision']['settled'])])
    for key,field,size in [('selection','selection_vs_source',32),('report','independent_report',18)]:
        available=[t[field] for t in tasks if t[field]]
        rows[key+'_count']=len(available)*size
        rows['source_'+key]=sum(t['before'] for t in available)
        rows['selected_'+key]=sum(t['after'] for t in available)
        rows[key+'_RGL']={name:sum(len(t[name]) for t in available) for name in ('retained','gained','lost')}
    return rows


def suite_evidence(tasks):
    return {suite:aggregate_tasks([t for t in tasks if t['suite']==suite]) for suite in sorted({t['suite'] for t in tasks})}


def write_report(destination,result):
    lines=['# Source/rank8参考库补齐窗口','', '|task|suite|selection候选(update:success/32)|selected|收束|独立report source→selected /18|', '|---|---|---|---|---|---|']
    for t in result['tasks']:
        decision=t['decision'] or {};report=t['independent_report']
        score=str(report['before'])+'→'+str(report['after']) if report else '未打开/未完成'
        lines.append(f"|{t['task_id']}|{t['suite']}|{t['candidates']}|{decision.get('selected_update')}|{decision.get('settled',False)}|{score}|")
    lines.extend(['','完整候选、相邻成功集合、source/MT保持得失、独立report及真实producer视频见同目录JSON。',
                  '36任务参考读出不证明控制空间或held教学迁移。未收束任务不打开report。',
                  '',f"交互账本（未知按完整上界）：`{json.dumps(result['interactions'])}`",f"GPU费用：`{json.dumps(result['billing'])}`"])
    (destination/'report.md').write_text('\n'.join(lines)+'\n')


def analyze(root):
    root=Path(root);tasks=[];videos=[];all_selection=[];all_report=[]
    mt=read_json(root/'inputs/MT128_reference.json')['rows']
    environment=read_json(root/'run_contract.json')['environment']
    for task in TASKS:
        result,candidates,report=task_evidence(root,task,mt)
        result['group']='target24' if task<40 else 'source_seen_support12'
        result['suite']=next(t['suite'] for t in environment['tasks'] if t['global_task_id']==task)
        tasks.append(result)
        for update,rows in candidates.items():all_selection.extend(dict(r,candidate_update=update) for r in rows)
        for arm,rows in report.items():
            all_report.extend(dict(r,report_arm=arm) for r in rows)
            if arm=='selected':videos.extend(dict(task_id=task,init_state_id=r['init_state_id'],success=r['success'],**r['RGB']) for r in rows)
    result=dict(window_complete=all(t['decision'] and (t['decision']['settled'] or t['decision'].get('window_exhausted')) for t in tasks),
        reference_bank_ready=all(t['decision'] and t['decision']['settled'] and t['independent_report'] for t in tasks),
        tasks=tasks,per_suite=suite_evidence(tasks),
        per_group={g:aggregate_tasks([t for t in tasks if t['group']==g]) for g in ('target24','source_seen_support12')},
        interactions=summary(root),billing=billing(root),limits=LIMITS,
        scientific_scope='target24 + source-seen support12 references only; not D/compiler/held400 evidence')
    destination=root/'analysis';destination.mkdir(exist_ok=True)
    for name,value in [('results',result),('selection_rows',all_selection),('report_rows',all_report),('producer_videos',videos)]:
        write_json_atomic(destination/f'{name}.json',value)
    write_report(destination,result)
    return result
