"""Pilot authority, fixed role panels and reproducible independent streams."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from ember.pi05_source_checkpoint import read_json, write_json_atomic

ASSET_ROOT = Path('/data1/user/ymdai/projects/EMBER')
RUN_ROOT = Path('/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261011/rank8_multistart')
PRIOR_ROOT = Path('/data1/user/ymdai/ember_runs/video_guided_proposal_writer_20261010/pilot')
CARRY_IN_GPUH = 18.640412103864882
SCHEMA = 'ember_proposal_rank8_multistart_v2'
EVENT_KINDS = ('seed', 'mid', 'late', 'return')
TASKS = (12, 29, 32, 38)
ALLOWLIST = (0, 1, 2, 4, 5, 7, 12, 13, 14, 15, 17, 19, 20, 21, 22, 25, 28, 29,
             32, 34, 35, 36, 37, 38, 42, 43, 51, 55, 56, 62, 64, 73, 95, 96, 97, 101)
SOURCE = dict(checkpoint='runs/outputs/pi05_source_aligned_seed7_1k_20260915/checkpoints/step_00001000',
              evaluation_config='configs/libero_24_8_8_coverage_v1/evaluation.json',
              data_protocol='configs/libero_24_8_8_coverage_v1/protocol.json',
              lora_contract='configs/pi05_lora_rank128_aligned.json',
              tokenizer='models/tokenizers/openpi/paligemma_tokenizer.model',
              normalization='configs/pi05_source_corpus_v1/source_normalization.json')
MT_PATH = Path('/data0/user/ymdai/ember_runs/coverage_retraining_20260920/training/mtbc'
               '/checkpoints/step_00000300/lora.safetensors')
EXPERT_ROOT = Path('/data1/user/ymdai/ember_runs/aligned_teacher_recovery_20261008/readouts/weights')
ENVIRONMENT_REFERENCE = Path('/data1/user/ymdai/ember_runs/denoising_return_writer_20261006'
                             '/readouts/parent/seen/SDE/evaluation/run_contract.json')
OPTIMIZER = dict(lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=1e-4)


def seed(domain, *coordinates):
    # LIBERO's wrapper ultimately seeds NumPy's uint32 RandomState.
    return int(np.random.SeedSequence([20261010, int(domain), *map(int, coordinates)]).generate_state(1)[0])


def pilot_panel():
    rows = []
    for task in TASKS:
        video = np.random.default_rng(np.random.SeedSequence([20261010, 0, task])).permutation(50).tolist()
        init = np.random.default_rng(np.random.SeedSequence([20261010, 1, task])).permutation(50).tolist()
        rows.append(dict(task_id=task, video_order=video, init_order=init,
            teacher_videos=video[:2], policy_videos=video[2:4], report_videos=video[4:6], other_video=video[6],
            selection_states=init[:4], audit_states=init[4:20], practice_states=init[20:36],
            local_query_states=init[36:40], rl_query_states=init[40:44], report_states=init[44:50],
            rng_roots={role: seed(domain, task) for role, domain in
                       [('teacher', 2), ('practice', 3), ('selection', 4), ('audit', 5),
                        ('local', 6), ('rl', 7), ('report', 8), ('refresh', 9)]}))
    return dict(schema_version=SCHEMA, seed=20261010, tasks=rows, learning_task_allowlist=list(ALLOWLIST),
        information_wall='task/file/role metadata schedule only; RGB and exact L are teaching',
        teaching_scope='finite train condition split, not all-episode action-label holdout',
        practice_reuse='fixed sixteen-state cycle, finite pool reuse recorded')


def materialize_panel(root=RUN_ROOT):
    path = Path(root) / 'panel.json'
    panel = pilot_panel()
    if path.exists():
        if read_json(path) != panel:
            raise ValueError('pre-registered pilot panel changed')
    else:
        write_json_atomic(path, panel)
    return panel


def environment_contract(asset_root=ASSET_ROOT):
    """Reuse official environment/assets only; no historical T or SDE consumer."""
    original = read_json(ENVIRONMENT_REFERENCE)
    result = {key: deepcopy(original[key]) for key in ('environment', 'libero_paths', 'policy', 'rng')}
    result['policy'].update(num_inference_steps=10, replan_steps=5)
    suites = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
    result['tasks'] = [deepcopy(row) for row in original['tasks']
                       if row['suite'] in suites and 10 * suites.index(row['suite']) + row['task_id'] in TASKS]
    for row in result['tasks']:
        row['global_task_id'] = 10 * ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10').index(row['suite']) + row['task_id']
        row['init_state_ids'] = list(range(50))
        if row['split_role'] != 'train' or row['installed_init_state_count'] != 50:
            raise ValueError('pilot crossed fixed train/full50 authority')
    if sorted(row['global_task_id'] for row in result['tasks']) != list(TASKS):
        raise ValueError('official pilot environment metadata missing')
    result.update(role='train', parallel=dict(envs_per_replica=1))
    return result
