"""First36 training paths and independent144 query; no checkpoint/held claim."""
from pathlib import Path
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.reference.analysis import paired,pairing_valid,read_cases
from ember.reference.accounting import summary
from ember.proposal_writer.analysis import billing
from .contract import TASKS,REFERENCE_ROOT,PARENT_ROOT,VIDEO_ROOT,condition_root,LIMITS

def summarize_task(task,spec,arms,*,identity=False):
    anchor=next(iter(arms.values()))
    if not anchor or any(not values or not pairing_valid(anchor,values) for values in arms.values()):
        raise ValueError(f'paired initial state/scene/native noise mismatch for task{task}')
    result=dict(task=task,suite=spec['suite'],group='target24' if task<40 else 'source_seen_support12',
        identity=identity,cases=len(anchor),success_sets={a:sorted(r['init_state_id'] for r in rows if r['success']) for a,rows in arms.items()})
    comparisons=[(a,b) for i,a in enumerate(arms) for b in list(arms)[i+1:]]
    result['comparisons']={a+'_to_'+b:paired(arms[a],arms[b]) for a,b in comparisons}
    for value in result['comparisons'].values():
        union=sorted(set(value['before_successes'])|set(value['after_successes']))
        value.update(success_set_union=union,jaccard=len(value['retained'])/len(union) if union else 1.)
    if 'reference' in arms:
        gains=set(result['success_sets']['reference'])-set(result['success_sets']['source'])
        result['reference_new_gains']=sorted(gains)
        result['reference_gains_preserved']={a:sorted(gains&set(values)) for a,values in result['success_sets'].items()}
    return result


def aggregate(rows):
    if not rows:return dict(tasks=[],cases=0,successes={},breadth={},comparisons={})
    arms=rows[0]['success_sets'];comparisons=rows[0]['comparisons']
    result=dict(tasks=[r['task'] for r in rows],cases=sum(r['cases'] for r in rows),
        successes={a:sum(len(r['success_sets'][a]) for r in rows) for a in arms},
        breadth={a:sum(bool(r['success_sets'][a]) for r in rows) for a in arms},comparisons={})
    for name in comparisons:
        values=[r['comparisons'][name] for r in rows]
        result['comparisons'][name]={k:sum(len(v[k]) for v in values) for k in ('retained','gained','lost')}
        result['comparisons'][name].update(churn=sum(v['churn'] for v in values),success_set_overlap=sum(v['success_set_overlap'] for v in values),
            before=sum(v['before'] for v in values),after=sum(v['after'] for v in values),
            breadth_improved=sum(v['after']>v['before'] for v in values),
            success_set_union=sum(len(v['success_set_union']) for v in values))
        union=result['comparisons'][name]['success_set_union']
        result['comparisons'][name]['jaccard']=sum(len(v['retained']) for v in values)/union if union else 1.
    if 'reference' in arms:
        result['reference_new_gains']=sum(len(r['reference_new_gains']) for r in rows)
        result['reference_gains_preserved']={a:sum(len(r['reference_gains_preserved'][a]) for r in rows) for a in arms}
    return result


def grouped(rows):
    return dict(total=aggregate(rows),tasks=rows,
        suites={s:aggregate([r for r in rows if r['suite']==s]) for s in sorted({r['suite'] for r in rows})},
        groups={g:aggregate([r for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})},
        identity=aggregate([r for r in rows if r['identity']]),nonidentity=aggregate([r for r in rows if not r['identity']]))



def analyze(root):
    root=Path(root);run=read_json(root/'run_contract.json');spec={t['global_task_id']:t for t in run['environment']['tasks']}
    bank={r['task']:r for r in read_json(PARENT_ROOT/'bank.json')['tasks']}
    old=read_json(VIDEO_ROOT/'analysis/results.json')['checkpoints']['900']['raw_rows']['seen']
    tasks=[];anchors=[];paths=[]
    for task in TASKS:
        path=read_json(condition_root(root,task)/'path.json')
        if not path.get('complete') or not path['query_complete']:raise ValueError('all36 complete paths/144 independent query required')
        current=path['query_rows'];states=[0,4,8,12];base=REFERENCE_ROOT/'selection'/f'task_{task:04d}'
        arms=dict(source=read_cases(base/'update_00000000',states),reference=read_cases(base/f"update_{bank[task]['selected_update']:08d}",states),generated=current)
        tasks.append(summarize_task(task,spec[task],arms,identity=bank[task]['selected_update']==0))
        anchor=[r for r in current if r['init_state_id']==0]
        previous=[r for r in old if r['global_task_id']==task and r['init_state_id']==0]
        anchors.append(summarize_task(task,spec[task],dict(Reader900=previous,generated=anchor),identity=bank[task]['selected_update']==0))
        paths.append(dict(task=task,meta_update=path['meta_update'],teacher_demo=path['teacher_demo'],
            selected_panel=path['best_panel'],selected_source=path['best_panel'].endswith('_000'),practice_successes=path['best_successes'],
            practice_panels=len(path['panels']),actual_proposals=len(path['proposals']),teaching_reads=path['teaching_reads'],
            frame_equivalent=path['frame_equivalent'],practice_steps=path['used_steps'],query_return=path['R'],
            query_successes=sum(r['success'] for r in current),stop_reason=path['stop_reason'],history_paths=path['history_paths'],
            candidate_successes=[p['successes'] for p in path['panels']]))
    training=[read_json(root/'training'/f'meta_{i:08d}.json') for i in range(1,10)]
    result=dict(complete=True,schema=run['schema'],phase='feedback_proposals',meta_updates=9,conditions=36,query_episodes=144,FM_queries=1008,
        seen=grouped(tasks),anchor_same_video36=grouped(anchors),paths=paths,training=training,interactions=summary(root),billing=billing(root),limits=LIMITS,
        no_single_checkpoint_qualification=True,held=False,Test=False,H_video_causality_controls=False,
        source_early_stops=sum(p['selected_source'] and p['actual_proposals']==0 for p in paths),
        total_proposals=sum(p['actual_proposals'] for p in paths),feedback_gradient_updates=sum(t['PG_norm']>0 for t in training))
    folder=root/'analysis';folder.mkdir(exist_ok=True);write_json_atomic(folder/'results.json',result);write_json_atomic(root/'completion.json',result)
    lines=['# First36 actual feedback compilation cycle','',
        'Fixed source/full38r8/D; selected Reader900 initialization with fresh H and Adam. These144 query rows come from changing training theta, not a single-checkpoint or held qualification.',
        '',f"Independent query144: {result['seen']['total']['successes']}; same-video anchor36: {result['anchor_same_video36']['total']['successes']}.",
        f"Actual proposals {result['total_proposals']}; real PG updates {result['feedback_gradient_updates']}/9; all-source early stops {result['source_early_stops']}.",'',
        '|task|meta update|practice best/4|panels|proposals|steps|query/4|selected source|','|---|---|---|---|---|---|---|---|']
    lines += [f"|{p['task']}|{p['meta_update']}|{p['practice_successes']}|{p['practice_panels']}|{p['actual_proposals']}|{p['practice_steps']}|{p['query_successes']}|{p['selected_source']}|" for p in paths]
    lines+=['',f"Interaction accounting: {result['interactions']}",f"New root GPUh: {result['billing']['GPU_hours']}",
        'Full suite/task/target/support/identity RGL, churn and overlap are retained in results.json. No automatic18/36 updates, held, controls, Test or new references.']
    (folder/'report.md').write_text('\n'.join(lines)+'\n');return result
