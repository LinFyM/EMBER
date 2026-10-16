"""Shared per-task and paired success-set aggregation for the active readout."""

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