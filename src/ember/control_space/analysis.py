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

def operator_decomposition(root):
    import json
    import time
    import resource
    from pathlib import Path
    import numpy as np
    from safetensors.numpy import load_file

    ROOT=Path(root)
    from .contract import OPERATOR_FILE
    OUT=ROOT/'analysis/operator_readout.json'
    if OUT.exists():return json.loads(OUT.read_text())
    tick=time.monotonic()
    contract=json.loads((ROOT/'run_contract.json').read_text())
    refs=contract['references']
    tasks=[r['task'] for r in refs]
    constants=json.loads(OPERATOR_FILE.read_text())['denominators']
    selected=[None if r['path'] is None else load_file(r['path']) for r in refs]
    generated=[[load_file(ROOT/'generated/seen'/f'task_{task:04d}'/f'state_{init:03d}'/'lora.safetensors') for init in range(16)] for task in tasks]
    print(json.dumps({'event':'loaded_existing_factors','tasks':len(tasks),'generated':len(tasks)*16,'seconds':time.monotonic()-tick}),flush=True)

    def energy(A,B):
        aa=A@A.transpose(0,2,1)
        bb=B.transpose(0,2,1)@B
        return np.einsum('nrs,nsr->n',aa,bb,optimize=True)

    def dot(x,y):return float(np.sum(x*y,dtype=np.float64))

    layers=[]
    task_stats={t:dict(error=0.,reference_energy=0.,generated_energy=0.,within_video_variance=0.,mean_error=0.) for t in tasks}
    for index,(name,denominator) in enumerate(constants.items()):
        ka=name+'.lora_A.default.weight';kb=name+'.lora_B.default.weight'
        di=generated[0][0][ka].shape[1];do=generated[0][0][kb].shape[0]
        sumG=np.zeros((do,di),dtype=np.float64);sumE=np.zeros_like(sumG)
        raw=dict(error=0.,reference_energy=0.,generated_energy=0.,task_mean_energy=0.,within_video_variance=0.,mean_error=0.,task_mean_dot_reference=0.)
        for task,reference,conditions in zip(tasks,selected,generated):
            A=np.stack([v[ka] for v in conditions]).astype(np.float64)
            B=np.stack([v[kb] for v in conditions]).astype(np.float64)
            Gmean=B.transpose(1,0,2).reshape(do,16*8)@A.reshape(16*8,di)/16
            E=np.zeros((do,di),dtype=np.float64) if reference is None else reference[kb].astype(np.float64)@reference[ka].astype(np.float64)
            eg=float(energy(A,B).mean());eG=dot(Gmean,Gmean);eE=dot(E,E);cross=dot(Gmean,E)
            values=dict(error=eg+eE-2*cross,reference_energy=eE,generated_energy=eg,task_mean_energy=eG,
                within_video_variance=eg-eG,mean_error=eG+eE-2*cross,task_mean_dot_reference=cross)
            sumG+=Gmean;sumE+=E
            for key,value in values.items():raw[key]+=value/36
            for key in task_stats[task]:task_stats[task][key]+=values[key]/denominator/38
        globalG=sumG/36;globalE=sumE/36
        public_error=dot(globalG-globalE,globalG-globalE)
        public_ref=dot(globalE,globalE);public_gen=dot(globalG,globalG)
        raw.update(public_error=public_error,public_reference_energy=public_ref,public_generated_energy=public_gen,
            conditional_error=raw['mean_error']-public_error,
            conditional_reference_energy=raw['reference_energy']-public_ref,
            conditional_generated_energy=raw['task_mean_energy']-public_gen,
            conditional_alignment=raw['task_mean_dot_reference']-dot(globalG,globalE),
            public_alignment=dot(globalG,globalE))
        normalized={key:value/denominator for key,value in raw.items()}
        layers.append(dict(name=name,denominator=denominator,normalized=normalized))
        print(json.dumps({'event':'layer_done','index':index+1,'total':38,'name':name,'error':normalized['error'],'seconds':time.monotonic()-tick}),flush=True)

    totals={key:float(np.mean([l['normalized'][key] for l in layers])) for key in layers[0]['normalized']}
    totals['decomposition_residual']=totals['error']-totals['public_error']-totals['conditional_error']-totals['within_video_variance']
    totals['public_relative_error']=totals['public_error']/totals['public_reference_energy']
    totals['conditional_relative_error']=totals['conditional_error']/totals['conditional_reference_energy']
    totals['conditional_least_squares_alignment_coefficient']=totals['conditional_alignment']/totals['conditional_reference_energy']
    totals['conditional_cosine']=totals['conditional_alignment']/np.sqrt(totals['conditional_generated_energy']*totals['conditional_reference_energy'])
    totals['public_least_squares_alignment_coefficient']=totals['public_alignment']/totals['public_reference_energy']
    for t,v in task_stats.items():
        v['relative_error']=v['error']/v['reference_energy'] if v['reference_energy']>0 else None
    report=dict(purpose='decompose existing endpoint BA approximation into common operator, task differences and within-task video variation',
        inputs=dict(run=str(ROOT),generated='all576 existing seen controllers',references=refs,constants=str(OPERATOR_FILE)),
        definitions=dict(metric='mean38 squared Frobenius divided by original fixed D_l; alpha/rank=1',
            means='means of BA operators for analysis only; no factor averaging and no deployed mean controller',
            decomposition='mean_t,v ||G_tv-E_t||^2 = ||Gbar-Ebar||^2 + mean_t ||(G_tbar-Gbar)-(E_t-Ebar)||^2 + mean_t,v ||G_tv-G_tbar||^2',
            conditional_alignment='mean_t inner(G_tbar-Gbar,E_t-Ebar), weighted by D_l',
            limitations='descriptive operator approximation, not action-weighted metric, intervention, lossless representation or causal failure attribution'),
        totals=totals,tasks=[dict(task=t,**task_stats[t]) for t in tasks],layers=layers,
        elapsed_seconds=time.monotonic()-tick,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        new_model_forwards=0,new_training_updates=0,new_environment_episodes=0,GPU_used=False)
    from ember.pi05_source_checkpoint import write_json_atomic
    write_json_atomic(OUT,report)
    print(json.dumps({'output':str(OUT),'totals':totals,'elapsed_seconds':report['elapsed_seconds'],'peak_RSS_KiB':report['peak_RSS_KiB']}),flush=True)
    return report
