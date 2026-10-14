"""Run-scoped non-held authority and fixed-G selector window registration."""
from pathlib import Path
import numpy as np
from ember.pi05_source_checkpoint import read_json,write_json_atomic,git_state
from ember.writer.data import WriterTaskAuthority
from ember.writer.learning_data import LearningTask,load_learning_tasks
from .contract import (TASKS,SOURCE,MT_PATH,OPTIMIZER,SCHEMA,PANEL_ROLE_SCHEMA,
                       SELECTOR_SCHEMA,CHECKPOINT_SCHEMA,OPERATOR_OBJECTIVE,seed)

ROOT_SCHEMA='ember_selector_transfer72_fixedG38349_v1'
SOURCE_ROOT=Path('/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261012/full_writer_after24849_20261013')
SOURCE_CP=SOURCE_ROOT/'checkpoints/RL/update_00000009'
SEEN_ROOT=Path('/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261013/endpoint_native_FM_after38349/parent_seen38349_query16')


def link(destination,source):
    source=Path(source).resolve(strict=True);destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve(strict=True)!=source:raise ValueError('registered readonly source changed')
    else:destination.symlink_to(source,target_is_directory=source.is_dir())


def learning_tasks(asset_root,root,task_ids,role):
    if role!='train':return load_learning_tasks(asset_root,task_ids,role=role,protocol_path=SOURCE['data_protocol'])
    run=read_json(Path(root)/'run_contract.json');path=Path(run['run_scoped_task_authority'])
    authority=read_json(path);requested=tuple(task_ids)
    function_run=run['schema_version']=='ember_full_g_native_function_identification_v1'
    value_run=run['schema_version']=='ember_independent_value_fixed_paths_v1'
    image_run=run['schema_version']=='ember_image_only_gemma_compiler_v1'
    schema=('ember_image_compiler_task_authorities_v1' if image_run else 'ember_independent_value_task_authorities_v1' if value_run else
            'ember_full_g_function_task_authorities_v1' if function_run else 'ember_selector_transfer72_task_authorities_v1')
    roles=['compiler'] if image_run else ['phi'] if value_run else ['G'] if function_run else ['theta','baseline']
    if (authority['schema_version']!=schema
            or Path(authority['run_root']).resolve()!=Path(root).resolve()
            or authority['training_tasks']!=run['training_tasks']
            or (not image_run and (authority['seen36']!=run['G_seen_tasks'] or authority['novel36']!=run['G_novel_tasks']))
            or not requested or len(set(requested))!=len(requested)
            or set(requested)-set(run['training_tasks'])):
        raise ValueError('runtime needs the audited exact run-scoped non-held task authority')
    rows={q['authority']['task_id']:q for q in authority['tasks']};output={}
    for task in requested:
        row=rows[task];a=row['authority'];p=Path(a['path'])
        if (p.stat().st_size!=a['expected_bytes'] or row['gradient_roles']!=roles
                or len(row['episode_lengths'])!=50):raise ValueError('registered task dataset or label consumer changed')
        if task>=40 and task-40 not in authority['source71_active_local_ids']:
            raise ValueError('source90 task falls outside specification-audited source71')
        output[task]=LearningTask(WriterTaskAuthority(task,a['language'],p,a['expected_bytes']),
            row['suite'],row['suite_task_id'],tuple(row['episode_lengths']))
    return output


def new_panel(task):
    video=np.random.default_rng(seed(0,task)).permutation(50).tolist()
    init=np.random.default_rng(seed(1,task)).permutation(50).tolist()
    states=init[36:46]+init[14:16]+init[32:36]
    return dict(task_id=task,video_order=video,init_order=init,policy_videos=[q for q in video if q>=46],
        bootstrap_state_id=init[0],bootstrap_noise_root=seed(4,task,init[0]),practice_states=init[:14],
        query_count=16,query_states=states,local_query_states=states,rl_query_states=states,
        query_teacher_selection_overlap=[],query_independent_of_teacher_selection=True,
        offline_teacher_labels_read=False)


def prepare(args):
    root=Path(args.root);selection=read_json(root/'analysis/main_contract/selector_G_novel72_task_selection_20261013.json')
    authority_path=root/'analysis/task_audit/task_authorities.json';authority=read_json(authority_path)
    seen=selection['seen36'];novel=selection['G_novel36'];tasks=selection['theta_tasks72']
    if (seen!=list(TASKS) or len(novel)!=36 or len(tasks)!=72 or set(seen)&set(novel)
            or sorted(seen+novel)!=tasks or authority['training_tasks']!=tasks):
        raise ValueError('frozen selected seen36/novel36 authority required')
    manifest=read_json(SOURCE_CP/'manifest.json')
    if (manifest['schema_version']!='ember_full_writer_frozen_frontend_endpoint_v10'
            or not manifest['complete'] or not manifest['full_G_theta_baseline']
            or manifest['next_update']!=9 or manifest['G_next_update']!=38349
            or manifest['supervision_version']!=9):raise ValueError('source must be full v10 RL9/F38349/bank9')
    old=read_json(SOURCE_ROOT/'run_contract.json');original=read_json(SOURCE_ROOT/'panel.json')
    panel_by_task={q['task_id']:q for q in original['tasks']}
    rows=[panel_by_task[t] if t in panel_by_task else new_panel(t) for t in tasks]
    sources={};parent_panel=read_json(SEEN_ROOT/'fixed_g_panel.json')
    reference_by_task={q['task']:q for q in parent_panel['rows']}
    for task in seen:
        row=reference_by_task[task];generation=SEEN_ROOT/'endpoints'/f'task_{task:04d}'/'generation.json'
        actual=read_json(generation)['input_identity']
        if (actual['checkpoint']!=str(SOURCE_CP) or actual['F_cursor']!=38349
                or actual['prefix_episodes']!=1 or actual['teacher_demo']!=panel_by_task[task]['policy_videos'][0]
                or row['init_state_id'] not in panel_by_task[task]['query_states']):
            raise ValueError('seen fixed endpoint, designated teacher or actual MT input identity conflict')
        bootstrap=Path(actual['history_source']).parent/'complete.json'
        sources[str(task)]=dict(root=str(SEEN_ROOT),panel=str(SEEN_ROOT/'fixed_g_panel.json'),
            results=str(SEEN_ROOT/'results.json'),weights=str(generation.with_name('G_000.safetensors')),
            generation=str(generation),bootstrap_complete=str(bootstrap))
        link(root/'bootstrap'/f'task_{task:04d}',bootstrap.parent)
    for name in ['initial.safetensors','task_lora_contract.json','bank_scale.json','operator_scale.json','coordinates.json','base','G_student_bank.json']:
        link(root/name,SOURCE_ROOT/name)
    for name in ['cuda','triton','huggingface','tmp']:(root/'cache'/name).mkdir(parents=True,exist_ok=True)
    mapping={str(q['task_id']):dict(teacher_demo=q['policy_videos'][0],path_ordinal=0,
        noise_root=seed(71,q['task_id'],0)) for q in rows}
    practice={str(q['task_id']):dict(init_state_id=q['init_order'][1],
        env_seed=seed(32,mapping[str(q['task_id'])]['noise_root'],1),
        policy_seed_root=seed(32,mapping[str(q['task_id'])]['noise_root'],1),
        history_episode_ordinal=1,parameter_ref='G_000') for q in rows}
    panel=dict(schema_version=SCHEMA,data_role_schema=PANEL_ROLE_SCHEMA,tasks=rows,
        learning_task_allowlist=tasks,G_seen_tasks=seen,G_novel_tasks=novel,seed=20261011,
        teaching_pool=list(range(46,50)),query_actions='own online query only; no offline labels')
    write_json_atomic(root/'panel.json',panel)
    selector={k:v for k,v in old['selector_training'].items() if k not in ['registered_theta_updates']}
    contract=dict(schema_version=ROOT_SCHEMA,checkpoint_schema=CHECKPOINT_SCHEMA,
        design='docs/designs/video_guided_proposal_writer_20261010.md#23',source=SOURCE,mt_checkpoint=str(MT_PATH),
        environment=authority['environment'],training_tasks=tasks,G_seen_tasks=seen,G_novel_tasks=novel,
        run_scoped_task_authority=str(authority_path),fixed_pair_sources=sources,
        learning_migration=dict(source_checkpoint=str(SOURCE_CP),source_schema=manifest['schema_version'],
            destination_schema=CHECKPOINT_SCHEMA,fixed_G_cursor=38349,fixed_bank_version=9,
            theta_Adam_baseline_samplers_RNG_inherited=True,original_contract_exact_resume=False,
            source_theta_stage='RL',source_theta_update=9,source_local_updates=90,
            new_training_task_distribution='equal seen/novel, uniform within each36'),
        G_objective=OPERATOR_OBJECTIVE,G_gradient_owner='frozen',optimizer=OPTIMIZER,
        source_checkpoint=str(SOURCE_CP),readout_checkpoint=str(SOURCE_CP),initial_writer_checkpoint=str(root/'checkpoints/local/update_00000180'),
        initial_writer_stage='local',fixed_local_update=180,fixed_G_update=38349,
        student_refresh=dict(version='fixed_G38349_pair72',G_update=38349,readout_mode='selector_fixed_pair',
            readonly_E_query=False,task_count=72,query_states_per_parameter=16,condition_mapping=mapping,practice_mapping=practice),
        updates=dict(local=[90,180],RL=[18]),logical_batch=dict(actor_conditions=4),
        selector_schema=SELECTOR_SCHEMA,selector_training=dict(selector,rounds=5,continue_cost=.001,
            registered_theta_updates=18,query_states_per_parameter=16),compilation=old['compilation'],
        final_warm=dict(updates=180,conditions=720,task_exposure=10,final_only=True,baseline_frozen=True,
            no_extra_GPU_readback=True),on_policy=dict(updates=18,conditions=72,rounds=5,teaching_pool_ordinal=1,
            query_visit=0,common_noise=True,G_updates=0,G_bank_append=False),
        formal_report=dict(version=['RL',18],checkpoint=str(root/'checkpoints/RL/update_00000018'),
            arms=['correct'],video_seed=7,max_rounds=5,environment_multiplier=6,states_per_task=50,conditions=400,
            bootstrap_reference_root=str(SOURCE_ROOT/'validation/RL_00000009/correct'),
            test_enabled=False,held_labels_read=False,checkpoint_selection='sole complete B18'),
        expected_initial_pair_episodes=1260,expected_initial_new_generations=36,
        expected_initial_reused_query_rows=1152,expected_B_practice_max=360,expected_B_G_query_max=5760,
        expected_B_additional_MT_query_max=1152,expected_formal_additional_MT_bootstrap_max=400,
        physical_execution=dict(response_microbatch=16),limits=dict(added_peak_GiB=160,GPU_hours=None,GPU_hours_hard_cap_active=False),
        task_batch_seed=[20261013,23],source_record=str(SOURCE_ROOT/'run_contract.json'),git=git_state(),
        information_wall='query labels only theta/baseline, never H/G; no offline actions or held gradients/Test',
        Owner_latest='11:50 autonomous execution; tonight23:00 goal subsequently cancelled; no deadline guard',
        bounded_stop='fixed pair72,A180,B18,one correct400; no automatic successor')
    target=root/'run_contract.json'
    if target.exists():raise ValueError('selector transfer registration already exists; resume its immutable contract')
    write_json_atomic(target,contract)
    return contract
