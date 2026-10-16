"""First36 training paths and independent144 query; no checkpoint/held claim."""
from ember.reference.analysis import paired,pairing_valid

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