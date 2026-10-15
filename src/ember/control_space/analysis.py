"""The unique terminal materialization and paired training-domain readout."""
from pathlib import Path
import json
import statistics
import torch
from safetensors.torch import load_file,save_file
from ember.lora import identity_lora_state,validate_lora_state
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.reference.analysis import paired,pairing_valid,read_cases
from ember.reference.accounting import summary
from ember.proposal_writer.analysis import billing
from .model import ControlSpace
from .data import topology
from .readout import configuration,inputs,saved_case


def materialize(root):
    root=Path(root);config=configuration(root);folder=root/'generated';folder.mkdir(exist_ok=True)
    entries=[dict(task=item['task'],path=str(folder/f"task_{item['task']:04d}.safetensors"),
        selected_update=item['selected_update'],identity_reuse=item['selected_update']==0)
        for item in read_json(root/'bank.json')['tasks']]
    manifest=dict(checkpoint=config['update'],phase=config['phase'],rank=8,targets=38,generated=entries)
    manifest_path=folder/'manifest.json'
    if manifest_path.exists() and read_json(manifest_path)['checkpoint']!=config['update']:
        raise ValueError('existing generated factors belong to another endpoint')
    # Publish the immutable endpoint before partial output, so resume skips valid files.
    write_json_atomic(manifest_path,manifest)
    if all(Path(item['path']).is_file() for item in entries):return manifest
    torch.set_num_threads(8);lora=topology();identity=identity_lora_state(lora)
    saved=torch.load(config['checkpoint'],map_location='cpu',weights_only=False)
    if saved['next_update']!=config['update']:raise ValueError('checkpoint is not the unique terminal endpoint')
    model=ControlSpace(lora,identity,read_json(root/'units.json'));model.load_state_dict(saved['model']);model.eval()
    # The physical source has mixed dtypes; its saved center is authoritative.
    identity={k:v.detach() for k,v in model.layout.unpack(model.layout.center).items()}
    for item in read_json(root/'bank.json')['tasks']:
        destination=folder/f"task_{item['task']:04d}.safetensors"
        if destination.exists():continue
        state=identity if item['selected_path'] is None else load_file(item['selected_path'])
        with torch.no_grad():generated=identity if item['selected_update']==0 else model(state)
        validate_lora_state(generated,lora)
        if any(not bool(v.isfinite().all()) for v in generated.values()):raise ValueError('nonfinite generated factors')
        if item['selected_update']==0 and any(v.count_nonzero() for k,v in generated.items() if 'lora_B' in k):
            raise ValueError('saved source identity must have zero B')
        temporary=destination.with_suffix('.partial')
        save_file({k:v.contiguous() for k,v in generated.items()},str(temporary));temporary.rename(destination)
    return manifest


COMPARISONS=('source_to_reference','source_to_prior144','source_to_generated',
    'reference_to_prior144','reference_to_generated','prior144_to_generated')


def aggregate(rows):
    result=dict(tasks=[r['task'] for r in rows],cases=len(rows)*16)
    for arm in ('source','reference','prior144','new'):
        result[arm]=sum(len(r['success_sets'][arm]) for r in rows)
    result['generated']=result['new']
    for arm in COMPARISONS:
        result[arm]={key:sum(len(r[arm][key]) for r in rows) for key in ('retained','gained','lost')}
        result[arm].update(churn=sum(r[arm]['churn'] for r in rows),
            success_set_overlap=sum(r[arm]['success_set_overlap'] for r in rows))
    for key in ('reference_new_gains_preserved','reference_new_gains_preserved_prior144','reference_new_gains'):
        result[key]=sum(len(r[key]) for r in rows)
    result['breadth']={arm:sum(bool(r['success_sets'][arm]) for r in rows)
        for arm in ('source','reference','prior144','new')}
    result['breadth_improved_over_source']=sum(r['source_to_generated']['after']>r['source_to_generated']['before'] for r in rows)
    result['breadth_improved_over_prior144']=sum(r['prior144_to_generated']['after']>r['prior144_to_generated']['before'] for r in rows)
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
    prior=generated if config['parent_root'] is None else read_cases(config['parent_root']/'readout'/f'task_{task:04d}',range(16))
    if any(not arm or not pairing_valid(source,arm) for arm in (reference,prior,generated)):
        raise ValueError(f'source/reference/prior144/new pairing incomplete: {task}')
    arms=dict(source=source,reference=reference,prior144=prior,generated=generated)
    successes={key:{r['init_state_id'] for r in arm if r['success']} for key,arm in arms.items()}
    result=dict(task=task,suite=task_config['suite'],
        group='target24' if task<40 else 'source_seen_support12',identity=item['selected_update']==0,
        selected_update=item['selected_update'],success_sets={
            ('new' if k=='generated' else k):sorted(v) for k,v in successes.items()},
        reference_new_gains=sorted(successes['reference']-successes['source']),
        reference_new_gains_preserved=sorted((successes['reference']-successes['source'])&successes['generated']),
        reference_new_gains_preserved_prior144=sorted((successes['reference']-successes['source'])&successes['prior144']),
        source_lost=sorted(successes['source']-successes['generated']))
    result.update({name:paired(arms[a],arms[b]) for name in COMPARISONS for a,b in [name.split('_to_')]})
    return result


def training_evidence(root):
    config=configuration(root)
    metrics=[json.loads(line) for line in (root/'training_metrics.jsonl').read_text().splitlines() if line.strip()]
    if config['parent_root'] is not None:
        updates=[m.get('global_update',m.get('update')) for m in metrics]
        if updates!=list(range(145,config['update']+1)):
            raise ValueError('operator metrics are not the complete post144 trajectory')
        completion=read_json(root/'training_completion.json')
        if not completion.get('complete') or completion['endpoint_update']!=config['update']:
            raise ValueError('operator training has no valid unique completion')
        pools={}
        for name in ('start144','first_check','endpoint'):
            path=root/'diagnostics'/f'{name}.json'
            if path.exists():
                pool=read_json(path)
                pools[name]=dict(path=str(path),rows=len(pool['rows']),**{key:pool[key] for key in
                    ('weighted_error','source_baseline','selected','other','global_update','read_wall_seconds')})
        return dict(phase=config['phase'],parent_update=144,endpoint_update=config['update'],
            updates=len(metrics),median_update_seconds=statistics.median(m['seconds'] for m in metrics),
            first=metrics[0],last=metrics[-1],completion=completion,pool_readouts=pools,
            logical_parameter_pairs=sum(m['logical_parameter_pairs'] for m in metrics),
            native_observation_queries=0,native_response_pairs=0)
    if len(metrics)!=144 or [m['update'] for m in metrics]!=list(range(1,145)):
        raise ValueError('native metrics are not the complete144 trajectory')
    return dict(updates=144,observation_queries=4032,response_pairs=8064,
        first_MSE=metrics[0]['MSE'],last_MSE=metrics[-1]['MSE'],median_update_seconds=statistics.median(m['seconds'] for m in metrics),
        counters={k:sum(r['counters'].get(k,0) for m in metrics for r in m['rank_measures'])
            for k in ('velocity_calls','velocity_rows','prefix_queries','identity_student_reuse')})


def analyze(root):
    root=Path(root);config=configuration(root);run,bank,_,reference_root=inputs(root)
    rows=[task_readout(root,item,run['environment'],config,reference_root) for item in bank.values()]
    result=dict(complete=True,checkpoint=config['update'],phase=config['phase'],tasks=rows,total=aggregate(rows),
        reference_root=str(reference_root),parent_root=str(config['parent_root']) if config['parent_root'] else None,
        groups={g:aggregate([r for r in rows if r['group']==g]) for g in ('target24','source_seen_support12')},
        suites={s:aggregate([r for r in rows if r['suite']==s]) for s in sorted({r['suite'] for r in rows})},
        nonidentity=aggregate([r for r in rows if not r['identity']]),identity=aggregate([r for r in rows if r['identity']]),
        identity_tasks=[r['task'] for r in rows if r['identity']],interactions=summary(root),billing=billing(root),
        limits=config['limits'],training=training_evidence(root),
        scope='reference-parameter input, non-held training-domain control preservation; selection init panel; no video/held/adjacent-learning claim')
    folder=root/'analysis';folder.mkdir(exist_ok=True)
    write_json_atomic(folder/'results.json',result)
    lines=[f"# Source/r8控制空间唯一终点{config['update']}读出",'',f"整体：`{result['total']}`",'',
        '|task|selected|source|reference|prior144|new|相对source得/失|相对144得/失|参考新增保存|',
        '|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        s=r['source_to_generated'];p=r['prior144_to_generated'];sets=r['success_sets']
        counts='|'.join(str(len(sets[a])) for a in ('source','reference','prior144','new'))
        lines.append(f"|{r['task']}|{r['selected_update']}|{counts}|{len(s['gained'])}/{len(s['lost'])}|{len(p['gained'])}/{len(p['lost'])}|{len(r['reference_new_gains_preserved'])}/{len(r['reference_new_gains'])}|")
    lines+=['','全部四方success sets、R/G/L/churn/overlap、suite/group/breadth与真实费用见results.json。',
        '五identity共80行复用source，不称学习；其余31任务496行是真实新执行。',
        '此轮无视频、held或相邻同目标学习稳定性资格；不能由训练算子误差直接推出迁移或闭环保存。']
    (folder/'report.md').write_text('\n'.join(lines)+'\n');write_json_atomic(root/'completion.json',result)
    return result
