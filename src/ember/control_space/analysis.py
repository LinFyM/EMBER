"""The unique linear-basis materialization and paired training-domain readout."""
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from ember.lora import validate_lora_state
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.reference.analysis import paired,pairing_valid,read_cases
from ember.reference.accounting import summary
from ember.proposal_writer.analysis import billing
from .model import load_basis
from .readout import configuration,inputs,saved_case,basis_metadata


def materialize(root):
    root=Path(root);config=configuration(root);folder=root/'generated';folder.mkdir(exist_ok=True)
    entries=[dict(task=item['task'],path=str(folder/f"task_{item['task']:04d}.safetensors"),
        selected_update=item['selected_update'],identity_reuse=item['selected_update']==0)
        for item in read_json(root/'bank.json')['tasks']]
    manifest=dict(schema=config['schema'],name=config['name'],**basis_metadata(config),rank=8,targets=38,generated=entries)
    manifest_path=folder/'manifest.json'
    if manifest_path.exists() and any(read_json(manifest_path).get(k)!=manifest[k] for k in ('schema','phase','basis_checkpoint')):
        raise ValueError('existing generated factors belong to another basis')
    # Publish the immutable basis identity before partial output, so resume skips valid files.
    write_json_atomic(manifest_path,manifest)
    if all(Path(item['path']).is_file() for item in entries):return manifest
    torch.set_num_threads(8);model=load_basis(config['checkpoint'])
    # All source factors come from the basis checkpoint's authoritative center.
    identity={k:v.detach() for k,v in model.layout.unpack(model.layout.center).items()}
    for item in read_json(root/'bank.json')['tasks']:
        destination=folder/f"task_{item['task']:04d}.safetensors"
        if destination.exists():continue
        state=identity if item['selected_path'] is None else load_file(item['selected_path'])
        with torch.no_grad():generated=identity if item['selected_update']==0 else model.forward(state)
        validate_lora_state(generated,model.lora)
        if any(v.dtype!=torch.float32 or not bool(v.isfinite().all()) for v in generated.values()):
            raise ValueError('generated factors must be finite FP32')
        if item['selected_update']==0 and any(v.count_nonzero() for k,v in generated.items() if 'lora_B' in k):
            raise ValueError('saved source identity must have zero B')
        temporary=destination.with_suffix('.partial')
        save_file({k:v.contiguous() for k,v in generated.items()},str(temporary));temporary.rename(destination)
    return manifest


ARMS=('source','reference','prior144','operator1170','generated')
COMPARISONS=tuple(a+'_to_'+b for index,a in enumerate(ARMS) for b in ARMS[index+1:])


def aggregate(rows):
    result=dict(tasks=[r['task'] for r in rows],cases=len(rows)*16)
    for arm in ('source','reference','prior144','operator1170','new'):
        result[arm]=sum(len(r['success_sets'][arm]) for r in rows)
    result['generated']=result['new']
    for arm in COMPARISONS:
        result[arm]={key:sum(len(r[arm][key]) for r in rows) for key in ('retained','gained','lost')}
        result[arm].update(churn=sum(r[arm]['churn'] for r in rows),
            success_set_overlap=sum(r[arm]['success_set_overlap'] for r in rows))
    for key in ('reference_new_gains_preserved','reference_new_gains_preserved_prior144','reference_new_gains_preserved_operator1170','reference_new_gains'):
        result[key]=sum(len(r[key]) for r in rows)
    result['breadth']={arm:sum(bool(r['success_sets'][arm]) for r in rows)
        for arm in ('source','reference','prior144','operator1170','new')}
    result['breadth_improved_over_source']=sum(r['source_to_generated']['after']>r['source_to_generated']['before'] for r in rows)
    result['breadth_improved_over_prior144']=sum(r['prior144_to_generated']['after']>r['prior144_to_generated']['before'] for r in rows)
    result['breadth_improved_over_operator1170']=sum(r['operator1170_to_generated']['after']>r['operator1170_to_generated']['before'] for r in rows)
    return result


def task_readout(root,item,environment,config=None,reference_root=None):
    config=configuration(root) if config is None else config
    reference_root=inputs(root)[3] if reference_root is None else reference_root
    task=item['task'];base=reference_root/'selection'/f'task_{task:04d}'
    task_config=next(t for t in environment['tasks'] if t['global_task_id']==task)
    source=read_cases(base/'update_00000000',range(16))
    reference=read_cases(base/f"update_{item['selected_update']:08d}",range(16))
    generated=[saved_case(root/'readout'/f'task_{task:04d}'/f'state_{init:03d}.json',task_config,item,
        reference_root,root/'generated'/f'task_{task:04d}.safetensors',config) for init in range(16)]
    prior=read_cases(config['parent_root']/'readout'/f'task_{task:04d}',range(16))
    operator=read_cases(config['operator_root']/'readout'/f'task_{task:04d}',range(16))
    if any(not arm or not pairing_valid(source,arm) for arm in (reference,prior,operator,generated)):
        raise ValueError(f'source/reference/prior144/operator1170/new pairing incomplete: {task}')
    arms=dict(source=source,reference=reference,prior144=prior,operator1170=operator,generated=generated)
    successes={key:{r['init_state_id'] for r in arm if r['success']} for key,arm in arms.items()}
    result=dict(task=task,suite=task_config['suite'],
        group='target24' if task<40 else 'source_seen_support12',identity=item['selected_update']==0,
        selected_update=item['selected_update'],success_sets={
            ('new' if k=='generated' else k):sorted(v) for k,v in successes.items()},
        reference_new_gains=sorted(successes['reference']-successes['source']),
        reference_new_gains_preserved=sorted((successes['reference']-successes['source'])&successes['generated']),
        reference_new_gains_preserved_prior144=sorted((successes['reference']-successes['source'])&successes['prior144']),
        reference_new_gains_preserved_operator1170=sorted((successes['reference']-successes['source'])&successes['operator1170']),
        source_lost=sorted(successes['source']-successes['generated']))
    result.update({name:paired(arms[a],arms[b]) for name in COMPARISONS for a,b in [name.split('_to_')]})
    return result


def pool_summary(rows):
    mass=sum(r['weight'] for r in rows)
    def mean(key):return sum(r['weight']*r[key] for r in rows)/mass
    numerator=mean('factor_squared_error');denominator=mean('factor_source_squared_norm')
    return dict(points=len(rows),weight=mass,operator_error=mean('error'),source_baseline=mean('source'),
        factor_squared_error=numerator,factor_source_squared_norm=denominator,
        factor_relative_error=numerator/denominator if denominator else None,
        error_by_layer=[sum(r['weight']*r['error_by_layer'][i] for r in rows)/mass for i in range(38)],
        source_by_layer=[sum(r['weight']*r['source_by_layer'][i] for r in rows)/mass for i in range(38)])


def fit_evidence(root):
    config=configuration(root);completion=read_json(root/'fit_completion.json')
    if (not completion.get('complete') or completion['basis_path']!=str(config['checkpoint'])
            or completion['code_dim']!=64 or completion['optimizer_updates']!=0):
        raise ValueError('linear fit requires its complete unique64-dimensional basis, without optimizer updates')
    path=root/'diagnostics/pool.json';pool=read_json(path);rows=pool['rows']
    if len(rows)!=271:raise ValueError('linear diagnostic pool must cover all271 parameter points')
    tasks={t['global_task_id']:t for t in inputs(root)[0]['environment']['tasks']}
    completion=dict(completion,pool=dict(path=str(path),**{k:v for k,v in pool.items() if k!='rows'}),
        by_task={str(t):pool_summary([r for r in rows if r['task']==t]) for t in tasks},
        by_suite={s:pool_summary([r for r in rows if tasks[r['task']]['suite']==s])
            for s in sorted({t['suite'] for t in tasks.values()})},by_layer=pool_summary(rows),
        native_observation_queries=0,native_response_pairs=0)
    return completion


def analyze(root):
    root=Path(root);config=configuration(root);run,bank,_,reference_root=inputs(root)
    rows=[task_readout(root,item,run['environment'],config,reference_root) for item in bank.values()]
    result=dict(complete=True,name=config['name'],schema=config['schema'],**basis_metadata(config),tasks=rows,total=aggregate(rows),
        reference_root=str(reference_root),parent_root=str(config['parent_root']),operator_root=str(config['operator_root']),
        groups={g:aggregate([r for r in rows if r['group']==g]) for g in ('target24','source_seen_support12')},
        suites={s:aggregate([r for r in rows if r['suite']==s]) for s in sorted({r['suite'] for r in rows})},
        nonidentity=aggregate([r for r in rows if not r['identity']]),identity=aggregate([r for r in rows if r['identity']]),
        identity_tasks=[r['task'] for r in rows if r['identity']],interactions=summary(root),billing=billing(root),
        limits=config['limits'],fit=fit_evidence(root),
        scope='reference-parameter input, non-held training-domain control preservation; selection init panel; no video/held/adjacent-learning claim')
    folder=root/'analysis';folder.mkdir(exist_ok=True)
    write_json_atomic(folder/'results.json',result)
    lines=['# Source/r8线性64维控制空间唯一basis读出','',f"整体：`{result['total']}`",'',
        '|task|selected|source|reference|prior144|operator1170|new|相对source得/失|相对1170得/失|参考新增保存|',
        '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        s=r['source_to_generated'];p=r['operator1170_to_generated'];sets=r['success_sets']
        counts='|'.join(str(len(sets[a])) for a in ('source','reference','prior144','operator1170','new'))
        lines.append(f"|{r['task']}|{r['selected_update']}|{counts}|{len(s['gained'])}/{len(s['lost'])}|{len(p['gained'])}/{len(p['lost'])}|{len(r['reference_new_gains_preserved'])}/{len(r['reference_new_gains'])}|")
    lines+=['','全部五方success sets、R/G/L/churn/overlap、suite/group/breadth与真实费用见results.json。',
        '五identity共80行复用source，不称学习；其余31任务496行是真实新执行。',
        '此轮无视频、held或相邻同目标学习稳定性资格；不能由训练算子误差直接推出迁移或闭环保存。']
    (folder/'report.md').write_text('\n'.join(lines)+'\n');write_json_atomic(root/'completion.json',result)
    return result
