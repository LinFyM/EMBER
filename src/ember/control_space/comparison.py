"""Post-freeze privileged projection only, then four-arm512 paired analysis."""
from pathlib import Path
from itertools import combinations
import datetime,time
import torch
from safetensors.torch import load_file,save_file
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.reference.analysis import paired,pairing_valid
from ember.reference.accounting import summary
from ember.proposal_writer.analysis import billing
from .contract import SCHEMA,PHASE,TASKS,BANK_ROOT,LINEAR_ROOT,OPERATOR_ROOT,REFERENCE_ROOT,STATES
from .model import load_basis
from .analysis import grouped
from .readout import inputs,references,saved_case


def materialize(root):
    """The fit completion is a hard dependency before opening excluded references."""
    root=Path(root);done=root/'materialization_completion.json'
    if not read_json(root/'fit_completion.json')['complete']:raise ValueError('selected excluded factors may only open after basis freeze')
    if done.exists():
        receipt=read_json(done)
        if receipt['basis_checkpoint']!=str(root/'basis/state.pt') or not all(Path(p).is_file() for p in receipt['paths']):raise ValueError('completed projected factors missing or wrong basis')
        return receipt
    torch.set_num_threads(8);model=load_basis(root/'basis/state.pt');model.requires_grad_(False)
    old_bank=read_json(BANK_ROOT/'bank.json');selected={t['task']:t for t in old_bank['tasks'] if t['task'] in TASKS}
    if set(selected)!=set(TASKS) or any(t['selected_path'] is None for t in selected.values()):raise ValueError('eight excluded selected references changed')
    folder=root/'generated';folder.mkdir(exist_ok=True);entries=[]
    for task in TASKS:
        b=selected[task];base=REFERENCE_ROOT/'selection'/f'task_{task:04d}'
        entries.append(dict(task=task,path=str(folder/f'task_{task:04d}.safetensors'),selected_reference=b['selected_path'],
            selected_update=b['selected_update'],source_rows=[str(base/'update_00000000'/f'state_{i:03d}.json') for i in STATES],
            reference_rows=[str(base/f'update_{b["selected_update"]:08d}'/f'state_{i:03d}.json') for i in STATES],
            U36_rows=[str(LINEAR_ROOT/'readout'/f'task_{task:04d}'/f'state_{i:03d}.json') for i in STATES],
            U36_LoRA=str(LINEAR_ROOT/'generated'/f'task_{task:04d}.safetensors')))
    evaluation=dict(schema=SCHEMA,phase=PHASE,basis_checkpoint=str(root/'basis/state.pt'),entries=entries,
        rank=8,targets=38,teacher_input=False,privileged_label_projection=True,
        pairing='old official init files/bytes, actual initial proprio, absolute env/policy noise; reuse stored scene only if present')
    write_json_atomic(root/'evaluation_manifest.json',evaluation)
    # Old S36/C_l are read for diagnostics only after the new basis is immutable.
    from .learning import OperatorLoss
    from ember.proposal_writer.model import FactorLayout
    old_units=read_json(LINEAR_ROOT/'units.json');constants=read_json(OPERATOR_ROOT/'operator_constants.json')
    raw_identity=model.layout.unpack(model.layout.center);metric=FactorLayout(model.lora,raw_identity,old_units)
    objective=OperatorLoss(model.lora,constants);prior_pool=read_json(LINEAR_ROOT/'diagnostics/pool.json')
    old_errors={p['task']:p for p in prior_pool['rows'] if p['point']==0 and p['task'] in TASKS}
    diagnostics=[];tick=time.monotonic();opened=datetime.datetime.now(datetime.timezone.utc).isoformat()
    for entry in entries:
        reference=load_file(entry['selected_reference'],device='cpu');vector=model.normalized(reference);code=vector@model.basis
        projected=model.decode(code);destination=Path(entry['path'])
        if not destination.exists():
            temporary=destination.with_suffix('.partial');save_file({k:v.detach().float().contiguous() for k,v in projected.items()},str(temporary));temporary.replace(destination)
        else:projected=load_file(str(destination),device='cpu')
        if set(projected)!=set(model.layout.names) or any(not bool(v.isfinite().all()) for v in projected.values()):raise ValueError('projected complete finite38/r8 factors changed')
        error,energy=objective([projected],[reference]);delta=(model.layout.pack(projected)-model.layout.pack(reference))/metric.scale
        baseline=(model.layout.pack(reference)-model.layout.center)/metric.scale
        F=float(delta[metric.valid].double().square().sum());B=float(baseline[metric.valid].double().square().sum())
        relative=float(error.sum()/energy.sum()) if float(energy.sum()) else None
        old=old_errors[entry['task']]
        diagnostics.append(dict(task=entry['task'],reference=entry['selected_reference'],output=entry['path'],z=code.tolist(),
            factor_squared_error_S36=F,factor_source_squared_norm_S36=B,factor_relative_error_S36=F/B if B else None,
            factor_squared_error_S28=float((model.normalized(projected)-vector).double().square().sum()),
            BA_error_by_layer_C36=error[0].tolist(),BA_source_by_layer_C36=energy[0].tolist(),BA_relative_error_C36=relative,
            U36_existing_factor_relative_error=old['factor_relative_error'],U36_existing_BA_error=old['error'],U36_existing_BA_source=old['source']))
    write_json_atomic(root/'diagnostics/excluded_selected.json',dict(complete=True,fit_frozen_before_open=True,reference_open_UTC=opened,
        records=diagnostics,seconds=time.monotonic()-tick,old_S_C_for_reporting_only=True,additional_policy_forwards=0))
    run,tasks,verified=inputs(root)
    # Source/reference/U36 raw row identities are read, never copied from main's score summary.
    for task in TASKS:
        for init in STATES:references(verified[task],tasks[task],init)
    result=dict(complete=True,basis_checkpoint=str(root/'basis/state.pt'),paths=[e['path'] for e in entries],outputs=8,
        no_reference_bypass=True,rank=8,targets=38,optimizer_updates=0)
    write_json_atomic(done,result);return result


def analyze(root):
    root=Path(root);run,tasks,entries=inputs(root);raw={a:[] for a in ('source','reference','U36','U28')};rows=[]
    for task in TASKS:
        values={a:[] for a in raw};provenance=[]
        for init in STATES:
            old=references(entries[task],tasks[task],init);new=saved_case(root,entries[task],tasks[task],init)
            for a in old:values[a].append(old[a])
            values['U28'].append(new);provenance.append(dict(init=init,**{a:entries[task][a+'_rows'][init] for a in old},U28=str(root/'readout'/f'task_{task:04d}'/f'state_{init:03d}.json')))
        sets={a:{r['init_state_id'] for r in v if r['success']} for a,v in values.items()};comparisons={}
        for a,b in combinations(values,2):
            item=paired(values[a],values[b]);union=sets[a]|sets[b]
            item.update(success_set_overlap=len(sets[a]&sets[b]),success_set_union=sorted(union),jaccard=len(sets[a]&sets[b])/len(union) if union else 1.)
            comparisons[a+'_to_'+b]=item
        gains=sets['reference']-sets['source']
        rows.append(dict(task=task,suite=tasks[task]['suite'],group='internal_excluded_FIRST8',identity=False,cases=16,
            success_sets={a:sorted(v) for a,v in sets.items()},comparisons=comparisons,reference_new_gains=sorted(gains),
            reference_gains_preserved={a:sorted(v&gains) for a,v in sets.items()},provenance=provenance))
        for a,v in values.items():raw[a].extend(v)
    if len(raw['U28'])!=128 or any(len(v)!=128 for v in raw.values()):raise ValueError('only the fixed four-arm512 panel can complete')
    result=dict(schema=SCHEMA,phase=PHASE,complete=True,arms=grouped(rows),raw_rows=raw,limits=run['limits'],
        interactions=summary(root),billing=billing(root),fit=read_json(root/'fit_completion.json'),
        diagnostic=str(root/'diagnostics/excluded_selected.json'),new_optimizer_updates=0,held=False,Test=False,
        privileged_factor_projection=True,video_compiler=False,checkpoint_selection=False,automatic_successor=False)
    write_json_atomic(root/'analysis/results.json',result);write_json_atomic(root/'completion.json',dict(complete=True,result=str(root/'analysis/results.json')))
    lines=['# 内部任务排除控制空间诊断','',str(result['arms']['total']['successes']),'',
        '|task|source|reference|U36|U28|','|---|---|---|---|---|']
    for r in rows:lines.append('|'+str(r['task'])+'|'+'|'.join(str(len(r['success_sets'][a])) for a in raw)+'|')
    lines+=['','全部512原始行/配对路径、每suite/breadth/RGL/churn/Jaccard、reference新增83保持及source损失见results.json。',
        '欧氏因子投影不是子空间内闭环最优；通过不证明视频可求代码或正式held迁移，失败不证明所有64维无解。',
        'S/U共同重算，是完整覆盖构造干预，不唯一归因某列或任务数。0神经训练/适应/新视频/held/Test/自动后继。']
    (root/'analysis/report.md').write_text('\n'.join(lines)+'\n');return result
