"""The unique terminal144 materialization and paired training-domain readout."""
from pathlib import Path
import json,statistics
import torch
from safetensors.torch import load_file,save_file
from ember.lora import identity_lora_state,validate_lora_state
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.reference.analysis import paired,pairing_valid,read_cases
from ember.reference.accounting import summary
from ember.proposal_writer.analysis import billing
from .model import ControlSpace
from .data import topology
from .contract import TASKS,REFERENCE_ROOT,LIMITS


def materialize(root):
    torch.set_num_threads(8);lora=topology();identity=identity_lora_state(lora)
    saved=torch.load(root/'checkpoints/update_00000144/state.pt',map_location='cpu',weights_only=False)
    if saved['next_update']!=144:raise ValueError('only terminal144 may be read out')
    model=ControlSpace(lora,identity,read_json(root/'units.json'));model.load_state_dict(saved['model']);model.eval()
    # The native source uses mixed physical dtypes; resume its saved identity.
    identity={k:v.detach() for k,v in model.layout.unpack(model.layout.center).items()}
    folder=root/'generated';folder.mkdir(exist_ok=True);manifest=[]
    for item in read_json(root/'bank.json')['tasks']:
        destination=folder/f"task_{item['task']:04d}.safetensors"
        manifest.append(dict(task=item['task'],path=str(destination),selected_update=item['selected_update'],identity_reuse=item['selected_update']==0))
        if destination.exists():continue
        state=identity if item['selected_path'] is None else load_file(item['selected_path'])
        with torch.no_grad():generated=model(state)
        validate_lora_state(generated,lora)
        if any(not bool(v.isfinite().all()) for v in generated.values()):raise ValueError('nonfinite generated factors')
        exact_identity=all(torch.equal(generated[k],identity[k]) for k in identity) if item['selected_update']==0 else False
        if item['selected_update']==0 and not exact_identity:raise ValueError('the anchored source must remain exactly identity')
        temporary=destination.with_suffix('.partial')
        save_file({k:v.contiguous() for k,v in generated.items()},str(temporary));temporary.rename(destination)
    write_json_atomic(folder/'manifest.json',dict(checkpoint=144,rank=8,targets=38,generated=manifest))


def aggregate(rows):
    result=dict(tasks=[r['task'] for r in rows],cases=len(rows)*16,
        source=sum(r['source_to_generated']['before'] for r in rows),
        reference=sum(r['reference_to_generated']['before'] for r in rows),
        generated=sum(r['source_to_generated']['after'] for r in rows))
    for arm in ('source_to_generated','reference_to_generated'):
        result[arm]={key:sum(len(r[arm][key]) for r in rows) for key in ('retained','gained','lost')}
        result[arm]['churn']=sum(r[arm]['churn'] for r in rows)
    result['reference_new_gains_preserved']=sum(len(r['reference_new_gains_preserved']) for r in rows)
    result['reference_new_gains']=sum(len(r['reference_new_gains']) for r in rows)
    result['breadth_improved_over_source']=sum(r['source_to_generated']['after']>r['source_to_generated']['before'] for r in rows)
    return result


def task_readout(root,item,environment):
    task=item['task'];base=REFERENCE_ROOT/'selection'/f'task_{task:04d}'
    source=read_cases(base/'update_00000000',range(16))
    reference=read_cases(base/f"update_{item['selected_update']:08d}",range(16))
    generated=read_cases(root/'readout'/f'task_{task:04d}',range(16))
    if not generated or not pairing_valid(source,generated) or not pairing_valid(reference,generated):
        raise ValueError(f'generated/source/reference pairing incomplete: {task}')
    src={r['init_state_id'] for r in source if r['success']};ref={r['init_state_id'] for r in reference if r['success']}
    out={r['init_state_id'] for r in generated if r['success']}
    return dict(task=task,suite=next(t['suite'] for t in environment['tasks'] if t['global_task_id']==task),
        group='target24' if task<40 else 'source_seen_support12',identity=item['selected_update']==0,
        selected_update=item['selected_update'],source_to_generated=paired(source,generated),
        reference_to_generated=paired(reference,generated),reference_new_gains=sorted(ref-src),
        reference_new_gains_preserved=sorted((ref-src)&out),source_lost=sorted(src-out))

def training_evidence(root):
    metrics=[json.loads(line) for line in (root/'training_metrics.jsonl').read_text().splitlines()]
    if len(metrics)!=144 or [m['update'] for m in metrics]!=list(range(1,145)):raise ValueError('training is not one complete144 trajectory')
    return dict(updates=144,observation_queries=4032,response_pairs=8064,
            first_MSE=metrics[0]['MSE'],last_MSE=metrics[-1]['MSE'],median_update_seconds=statistics.median(m['seconds'] for m in metrics),
            counters={k:sum(r['counters'].get(k,0) for m in metrics for r in m['rank_measures'])
                for k in ('velocity_calls','velocity_rows','prefix_queries','identity_student_reuse')})


def analyze(root):
    bank=read_json(root/'bank.json');environment=read_json(root/'run_contract.json')['environment'];rows=[]
    rows=[task_readout(root,item,environment) for item in bank['tasks']]
    result=dict(complete=True,checkpoint=144,tasks=rows,total=aggregate(rows),
        groups={g:aggregate([r for r in rows if r['group']==g]) for g in ('target24','source_seen_support12')},
        suites={s:aggregate([r for r in rows if r['suite']==s]) for s in {r['suite'] for r in rows}},
        nonidentity=aggregate([r for r in rows if not r['identity']]),identity_tasks=[r['task'] for r in rows if r['identity']],
        interactions=summary(root),billing=billing(root),limits=LIMITS,
        training=training_evidence(root),
        scope='reference-parameter input, non-held training-domain function preservation; selection init panel; no video/held/adjacent-learning claim')
    folder=root/'analysis';folder.mkdir(exist_ok=True)
    write_json_atomic(folder/'results.json',result)
    lines=['# Source/r8控制空间144唯一读出','',f"整体：`{result['total']}`",'',
        '|task|selected|source|reference|generated|相对source得/失|参考新得保存|','|---|---|---|---|---|---|---|']
    for r in rows:
        s=r['source_to_generated'];t=r['reference_to_generated']
        lines.append(f"|{r['task']}|{r['selected_update']}|{s['before']}|{t['before']}|{s['after']}|{len(s['gained'])}/{len(s['lost'])}|{len(r['reference_new_gains_preserved'])}/{len(r['reference_new_gains'])}|")
    lines+=['','全部success sets、suite/group、真实费用见results.json；五identity复用明示，不称学习。',
        '此轮无视频输入、无held、无学习相邻点读出；不能由训练函数误差直接推出迁移或闭环保存。']
    (folder/'report.md').write_text('\n'.join(lines)+'\n');write_json_atomic(root/'completion.json',result)
    return result
