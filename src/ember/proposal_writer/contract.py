"""Full36 authorities, explicit role panels and reproducible independent streams."""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path

import numpy as np

from ember.pi05_source_checkpoint import read_json, write_json_atomic

ASSET_ROOT = Path('/data1/user/ymdai/projects/EMBER')
RUN_ROOT = Path('/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261013/'
                'endpoint_native_FM_after38349')
PRIOR_ROOT = Path('/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261011/rank8_multistart')
SCHEMA = 'ember_proposal_full36_deterministic_endpoint_query16_v1'
PANEL_ROLE_SCHEMA = 'ember_full36_practice14_query16_report16_v3'
ALLOWLIST = (0, 1, 2, 4, 5, 7, 12, 13, 14, 15, 17, 19, 20, 21, 22, 25, 28, 29,
             32, 34, 35, 36, 37, 38, 42, 43, 51, 55, 56, 62, 64, 73, 95, 96, 97, 101)
TASKS = ALLOWLIST
SOURCE = dict(checkpoint='runs/outputs/pi05_source_aligned_seed7_1k_20260915/checkpoints/step_00001000',
              evaluation_config='configs/libero_24_8_8_coverage_v1/evaluation.json',
              data_protocol='configs/libero_24_8_8_coverage_v1/protocol.json',
              lora_contract='configs/pi05_lora_rank128_aligned.json',
              tokenizer='models/tokenizers/openpi/paligemma_tokenizer.model',
              normalization='configs/pi05_source_corpus_v1/source_normalization.json')
MT_PATH = Path('/data0/user/ymdai/ember_runs/coverage_retraining_20260920/training/mtbc'
               '/checkpoints/step_00000300/lora.safetensors')
OPTIMIZER = dict(lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=1e-4)
SELECTOR_SCHEMA = 'ember_full36_student_joint_conditional_continue_selector_v5'
CHECKPOINT_SCHEMA = 'ember_full_writer_endpoint_native_FM_v11'
G_OBJECTIVE = 'task_equal_cross_episode_native_FM50x7_v1'
OPERATOR_OBJECTIVE = 'task_equal_layer_mean_full_BA_Frobenius_v1'
OPERATOR_CHECKPOINT_SCHEMA = 'ember_full_writer_frozen_frontend_endpoint_v10'


def seed(domain, *coordinates):
    return int(np.random.SeedSequence([20261011, int(domain), *map(int, coordinates)]).generate_state(1)[0])


def parameter_ref(task, update):
    return 'MT300' if int(update) == 0 else f'task_{int(task):04d}_MT300merged_r8_u{int(update):08d}'


def materialize_panel(root=RUN_ROOT, *, selection_count=None, report_count=None):
    path = Path(root) / 'panel.json'
    if selection_count is None:
        panel = read_json(path)
        if (panel['schema_version'] != SCHEMA or panel['data_role_schema'] != PANEL_ROLE_SCHEMA
                or tuple(r['task_id'] for r in panel['tasks']) != TASKS):
            raise ValueError('registered full36 panel scope changed')
        return panel
    if (selection_count, report_count) != (16, 16):
        raise ValueError('reused teacher selection/report panels must remain16/16')
    rows = []
    for task in TASKS:
        video = np.random.default_rng(seed(0, task)).permutation(50).tolist()
        init = np.random.default_rng(seed(1, task)).permutation(50).tolist()
        selection = init[:selection_count]
        rows.append(dict(task_id=task, video_order=video, init_order=init,
            teacher_videos=[i for i in video if i >= 46], policy_videos=[i for i in video if i >= 46],
            teacher_query_demos=list(range(46)), teacher_selection_states=selection,
            selection_folds=[selection[:selection_count//2], selection[selection_count//2:]],
            bootstrap_state_id=selection[0],bootstrap_noise_root=seed(4,task,selection[0]),
            teacher_report_states=init[selection_count:selection_count+report_count],
            practice_states=init[:14],query_count=16,
            query_states=init[36:46]+init[14:16]+init[32:36],
            local_query_states=init[36:46]+init[14:16]+init[32:36],
            rl_query_states=init[36:46]+init[14:16]+init[32:36],report_states=init[46:50],
            query_teacher_selection_overlap=init[14:16],query_independent_of_teacher_selection=False,
            practice_reuses_teacher_selection_training=True,local_RL_share_query_pool=True,
            rng_roots={role: seed(domain, task) for role, domain in
                       [('teacher', 2), ('practice', 3), ('selection', 4), ('teacher_report', 5),
                        ('local', 6), ('rl', 7), ('report', 8), ('refresh', 9)]}))
    panel = dict(schema_version=SCHEMA,data_role_schema=PANEL_ROLE_SCHEMA,seed=20261011,tasks=rows,learning_task_allowlist=list(TASKS),
        action_training_demos=list(range(46)), task_internal_unfitted_demos=list(range(46,50)),
        historical_exposure='MT saw50/previousQ49; this holdout applies to the new teacher only',
        information_wall='task/file/role metadata only; RGB and exact language are teaching',
        selection_uses=['teacher_selection'],selection_folds_are_historical_metadata=True,independent_report=False,
        bootstrap_uses=['MT_self_practice','G_initialization','student_bootstrap'],
        teacher_report_uses=['independent_teacher_report'], teacher_report_excluded_from_H_and_gradient=True,
        query_uses=['train_credit','train_mechanism_readout'],query_count_per_parameter=16,
        query_independent_of_condition_practice=True,query_independent_of_teacher_selection=False)
    if path.exists() and read_json(path) != panel:
        raise ValueError('pre-registered full36 panel changed')
    if not path.exists():
        write_json_atomic(path, panel)
    return panel


def environment_contract(asset_root=ASSET_ROOT, *, root=RUN_ROOT, role='train'):
    """Reuse installed official task/asset authority, including support horizon400."""
    from ember.pi05_eval_contract import load_evaluation_authorities, inspect_installed_target_tasks
    asset_root, root = Path(asset_root), Path(root)
    os.environ['EMBER_LIBERO_ASSETS_ROOT'] = str(asset_root / 'data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6')
    authorities = load_evaluation_authorities(asset_root / SOURCE['evaluation_config'], asset_root)
    installed, paths = inspect_installed_target_tasks(authorities,
        role='operator_seen_training36' if role == 'train' else role,
        state_count=50, libero_config_dir=root / 'libero_config')
    suites = ('libero_spatial','libero_object','libero_goal','libero_10')
    tasks = []
    for task in installed:
        row = asdict(task)
        row['global_task_id'] = (40 + task.task_id if task.suite == 'libero_90'
                                 else 10*suites.index(task.suite)+task.task_id)
        row['init_state_ids'] = list(row['init_state_ids'])
        tasks.append(row)
    if role == 'train' and tuple(r['global_task_id'] for r in tasks) != TASKS:
        raise ValueError('full36 environment crosses the audited train24+support12 scope')
    return dict(role=role, tasks=tasks, environment=authorities.config['environment'],
        libero_paths=paths, policy=dict(num_inference_steps=10,replan_steps=5),
        rng=dict(policy_seed=7,env_seed=7), parallel=dict(envs_per_replica=1))
