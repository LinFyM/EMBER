"""Read-only training provenance and the exact registered976 panel."""
from pathlib import Path
from ember.pi05_source_checkpoint import read_json,write_json_atomic
from ember.expert_manifold.video_schedule import condition_demo_index
from .contract import ASSETS,TASKS,VALIDATION_TASKS,VIDEO_ROOT,SCHEMA,PHASE,LIMITS,SOURCE,NORMALIZATION,INITIAL_READER,PREVIOUS_ROOT,SOURCE_INPUTS,OPERATOR_FILE,OBJECTIVE,TRAIN_ROOT,TRAINING_PRODUCER,MODEL_SCHEMA,MODEL_PHASE,FINAL,checkpoint_path


def registered_panels():
    panels=read_json(VIDEO_ROOT/'panels.json')
    if set(panels)!= {'seen','validation'}:raise ValueError('only registered seen576/strict400')
    for role,tasks,count,noise in (('seen',TASKS,16,202610151),('validation',VALIDATION_TASKS,50,7)):
        rows=panels[role]
        if len(rows)!=len(tasks)*count or {(r['task'],r['init_state_id']) for r in rows}!={(t,i) for t in tasks for i in range(count)}:
            raise ValueError('whole fixed task/state panel changed')
        for row in rows:
            task=row['task'];suite=('libero_spatial','libero_object','libero_goal','libero_10')[task//10] if task<40 else 'libero_90'
            local=task%10 if task<40 else task-40
            demo=condition_demo_index(7,suite,local,row['init_state_id'],condition='correct',demo_count=50,sampling_mode='without_replacement')
            if row['noise_root']!=noise or row['teacher_demo']!=demo:raise ValueError('official state-video/noise mapping changed')
        if count==50 and any(len({r['teacher_demo'] for r in rows if r['task']==t})!=50 for t in tasks):raise ValueError('held teaching must use all50 videos once')
    return panels


def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    origin=read_json(TRAIN_ROOT/'run_contract.json');refs=origin['references']
    if (origin['schema']!=MODEL_SCHEMA or origin['phase']!=MODEL_PHASE or origin['tasks']!=list(TASKS)
            or len(refs)!=36 or sum(r['path'] is None for r in refs)!=5):
        raise ValueError('read-only1090 must inherit the original36 source identities')
    old=read_json(VIDEO_ROOT/'run_contract.json')
    from ember.proposal_writer.contract import environment_contract as environment
    environments=dict(seen=old['environment'],validation=environment(ASSETS,root=root,role='validation'))
    contract=dict(schema=SCHEMA,phase=PHASE,objective=OBJECTIVE,static_inputs=dict(source_inputs=str(SOURCE_INPUTS),source_manifest=str(SOURCE_INPUTS.parent/'manifest.json'),operator_constants=str(OPERATOR_FILE),producer=read_json(PREVIOUS_ROOT/'owner_launch.json')['code_commit'],read_only=True),shared_M=dict(root=str(PREVIOUS_ROOT/'cache/M'),producer_phase='source_weight_compiler',read_only=True,global_capacity=2**30),limits=LIMITS,tasks=list(TASKS),references=refs,
        environments=environments,panels=str(root/'panels.json'),source_checkpoint=str(ASSETS/SOURCE),normalization=str(ASSETS/NORMALIZATION),
        frozen_reader=str(INITIAL_READER),rank=8,alpha=8,dropout=0.,targets=38,seed=7,
        stages={},shared_updates=0,optimizer_restored=False,
        training_source=dict(checkpoint=str(checkpoint_path(root,FINAL)),manifest=str(checkpoint_path(root,FINAL).with_name('manifest.json')),producer=TRAINING_PRODUCER,schema=MODEL_SCHEMA,phase=MODEL_PHASE,stage='joint',next_update=FINAL,read_only=True),
        final_update=FINAL,eval_rows=976,interaction_adaptation=False,Test=False)
    for name,value in (('run_contract.json',contract),('panels.json',registered_panels())):
        path=root/name
        if path.exists() and read_json(path)!=value:raise ValueError('immutable source-weight contract changed')
        if not path.exists():write_json_atomic(path,value)
    return contract
