"""The bounded eight-task reference window and immutable case identities."""
from dataclasses import asdict
from pathlib import Path
import os

from ember.pi05_source_checkpoint import read_json, write_json_atomic

ASSETS = Path('/data1/user/ymdai/projects/EMBER')
ROOT = Path('/data1/user/ymdai/ember_runs/source_policy_space_compiler_20261015/reference_seed7')
TASKS = (0, 1, 12, 13, 20, 21, 32, 34)
BASE = (80, 160, 320, 480, 720, 960)
EXTENSIONS = (1200, 1440, 1680, 1920)
NODES = (0, *BASE, *EXTENSIONS)
SHARD = 16
SCHEMA = 'ember_source_reference_r8_seed7_v1'
SOURCE = 'runs/outputs/pi05_source_aligned_seed7_1k_20260915/checkpoints/step_00001000'
EVALUATION = 'configs/libero_24_8_8_coverage_v1/evaluation.json'
NORMALIZATION = 'configs/pi05_source_corpus_v1/source_normalization.json'
OPTIMIZER = dict(lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=1e-4)
LIMITS = dict(optimizer_updates=15360, FM_queries=430080, physical_submissions=3120,
              control_steps=1032640, dummy_steps=31200, total_steps=1063840,
              engineering_extra_attempts=16, added_peak_GiB=24)


def checkpoint(root, task, update):
    return Path(root) / 'training' / f'task_{task:04d}' / 'checkpoints' / f'update_{update:08d}'


def parameter_ref(task, update):
    return 'raw_source_identity' if update == 0 else f'source_r8_t{task:04d}_u{update:08d}'


def environment_contract(root=ROOT):
    from ember.pi05_eval_contract import load_evaluation_authorities, inspect_installed_target_tasks
    os.environ['EMBER_LIBERO_ASSETS_ROOT'] = str(ASSETS / 'data/simulation/ember_assets/datasets/libero-assets/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6')
    authorities = load_evaluation_authorities(ASSETS / EVALUATION, ASSETS)
    installed, paths = inspect_installed_target_tasks(authorities, role='development_train',
        state_count=50, libero_config_dir=Path(root) / 'libero_config')
    suites = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
    tasks = []
    for task in installed:
        global_id = 10 * suites.index(task.suite) + task.task_id
        if global_id in TASKS:
            tasks.append(dict(asdict(task), global_task_id=global_id))
    if sorted(t['global_task_id'] for t in tasks) != list(TASKS):
        raise ValueError('reference environment escaped the registered eight train tasks')
    return dict(tasks=tasks, environment=authorities.config['environment'], libero_paths=paths,
                policy=dict(num_inference_steps=10, replan_steps=5),
                rng=dict(policy_seed=7, env_seed=7), parallel=dict(envs_per_replica=1))


def plan(root, contract):
    """All potential work is registered; conditional nodes remain ineligible."""
    panel = {}
    for task in contract['tasks']:
        t, horizon = task['global_task_id'], task['horizon']
        for before, after in zip(NODES, NODES[1:]):
            identity = f'train_t{t:04d}_u{after:08d}'
            panel[identity] = dict(kind='train', task=t, start=before, stop=after,
                                   states=[0], cost=(after-before)*5)
        for update in NODES:
            for first in range(0, 32, SHARD):
                identity = f'selection_t{t:04d}_u{update:08d}_s{first:02d}'
                panel[identity] = dict(kind='selection', task=t, update=update,
                                       states=list(range(first, first+SHARD)), cost=horizon*SHARD)
        for arm in ('source', 'selected'):
            for first in range(32, 50, SHARD):
                identity = f'report_t{t:04d}_{arm}_s{first:02d}'
                panel[identity] = dict(kind='report', task=t, arm=arm,
                    states=list(range(first, min(first+SHARD, 50))), cost=horizon*min(SHARD,50-first))
    value = dict(schema=SCHEMA, panel=panel)
    path = Path(root) / 'queue_plan.json'
    if path.exists() and read_json(path) != value:
        raise ValueError('immutable reference work plan changed')
    write_json_atomic(path, value)
    return panel


def selected_candidate(scores):
    """Scores contain complete selection panels only; ties favor earliest."""
    return min(scores, key=lambda u: (-scores[u], u))


def selection_decision(scores):
    if not all(u in scores for u in (0, *BASE)):
        return None
    best = selected_candidate(scores)
    later = sum(u > best for u in scores)
    if later >= 2:
        return dict(selected_update=best, settled=True, scores=scores, later_nodes=later)
    following = next((u for u in EXTENSIONS if u not in scores), None)
    return dict(selected_update=best, settled=False, scores=scores,
                later_nodes=later, next_update=following, window_exhausted=following is None)
