"""The audited reference pool, bounded task trajectories and immutable cases."""
from pathlib import Path

from ember.pi05_source_checkpoint import read_json, write_json_atomic

ASSETS = Path('/data1/user/ymdai/projects/EMBER')
ROOT = Path('/data1/user/ymdai/ember_runs/source_policy_space_compiler_20261015/reference_seed7')
FIRST_TASKS = (0, 1, 12, 13, 20, 21, 32, 34)
NEW_TASKS = (2, 4, 5, 7, 14, 15, 17, 19, 22, 25, 28, 29, 35, 36, 37, 38,
             42, 43, 51, 55, 56, 62, 64, 73, 95, 96, 97, 101)
# Keep the original eight first so their queue ordinals/payloads are unchanged.
TASKS = FIRST_TASKS + NEW_TASKS
WINDOW = 'final_four_20261016'
REVISION = 3
ACTIVE_TASKS = (5,25,38,43)
FINAL_EXTENSIONS = (2160,2400,2640,2880)
BASE = (80, 160, 320, 480, 720, 960)
EXTENSIONS = (1200, 1440, 1680, 1920)
NODES = (0, *BASE, *EXTENSIONS)
SHARD = 16
SCHEMA = 'ember_source_reference_r8_seed7_v1'
SOURCE = 'runs/outputs/pi05_source_aligned_seed7_1k_20260915/checkpoints/step_00001000'
EVALUATION = 'configs/libero_24_8_8_coverage_v1/evaluation.json'
NORMALIZATION = 'configs/pi05_source_corpus_v1/source_normalization.json'
OPTIMIZER = dict(lr=1e-4, betas=(.9, .95), eps=1e-8, weight_decay=1e-4)
LIMITS = dict(optimizer_updates=49200, FM_queries=1377600, physical_submissions=11238,
              control_steps=2527141, dummy_steps=112380, total_steps=2639521,
              engineering_extra_attempts=16, added_peak_GiB=6)


def nodes(task):
    if task in ACTIVE_TASKS:return (*NODES,*FINAL_EXTENSIONS)
    return (*NODES, 2160) if task == 0 else NODES


def queue_path(root):
    return Path(root)/'windows'/WINDOW/'work.sqlite'


def checkpoint(root, task, update):
    return Path(root) / 'training' / f'task_{task:04d}' / 'checkpoints' / f'update_{update:08d}'


def parameter_ref(task, update):
    return 'raw_source_identity' if update == 0 else f'source_r8_t{task:04d}_u{update:08d}'


def environment_contract(root=ROOT):
    # This metadata owner already audits operator_seen_training36, support IDs
    # 40+local, source71 membership and horizon400. It loads no MT/model weights.
    from ember.proposal_writer.contract import environment_contract as audited_environment
    contract = audited_environment(ASSETS, root=root, role='train')
    tasks = {t['global_task_id']: t for t in contract['tasks']}
    if set(tasks) != set(TASKS):
        raise ValueError('reference environment escaped audited train24+support12')
    return dict(contract, tasks=[tasks[t] for t in TASKS])


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
    # Append the sole extra task0 segment after all historical identities.
    panel['train_t0000_u00002160'] = dict(kind='train',task=0,start=1920,stop=2160,states=[0],cost=240*5)
    horizon = next(t['horizon'] for t in contract['tasks'] if t['global_task_id']==0)
    for first in range(0,32,SHARD):
        panel[f'selection_t0000_u00002160_s{first:02d}'] = dict(kind='selection',task=0,update=2160,
            states=list(range(first,first+SHARD)),cost=horizon*SHARD)
    # Original1299 identities/ordinals stay intact. Only these four trajectories
    # receive new work; historical untriggered pending work remains ineligible.
    for t in ACTIVE_TASKS:
        horizon=next(task['horizon'] for task in contract['tasks'] if task['global_task_id']==t)
        for before,after in zip((1920,*FINAL_EXTENSIONS),FINAL_EXTENSIONS):
            panel[f'train_t{t:04d}_u{after:08d}']=dict(kind='train',task=t,start=before,stop=after,states=[0],cost=(after-before)*5)
            for first in range(0,32,SHARD):
                panel[f'selection_t{t:04d}_u{after:08d}_s{first:02d}']=dict(kind='selection',task=t,update=after,states=list(range(first,first+SHARD)),cost=horizon*SHARD)
    value = dict(schema=SCHEMA, revision=REVISION, window=WINDOW, panel=panel)
    path = Path(root)/'windows'/WINDOW/'queue_plan.json'
    if path.exists() and read_json(path) != value:
        raise ValueError('immutable reference work plan changed')
    write_json_atomic(path, value)
    return panel


def selected_candidate(scores):
    """Scores contain complete selection panels only; ties favor earliest."""
    return min(scores, key=lambda u: (-scores[u], u))


def selection_decision(scores, task=None):
    if not all(u in scores for u in (0, *BASE)):
        return None
    best = selected_candidate(scores)
    later = sum(u > best for u in scores)
    if later >= 2:
        return dict(selected_update=best, settled=True, scores=scores, later_nodes=later)
    extensions = tuple(u for u in nodes(task) if u>960)
    following = next((u for u in extensions if u not in scores), None)
    return dict(selected_update=best, settled=False, scores=scores,
                later_nodes=later, next_update=following, window_exhausted=following is None)
