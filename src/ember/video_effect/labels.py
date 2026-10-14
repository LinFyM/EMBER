"""Read the registered 224 shared-training label caches; never build labels.

Registry, public object focus and physical masks are supervision metadata only.
The Phi call consumes none of them. Probe teacher labels remain sealed.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from collections.abc import Sequence
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import torch

LABEL_ROOT = Path('/data1/user/ymdai/ember_runs/relation_grounded_writer_20261006/labels')
TRAIN_TASKS = (2, 4, 5, 7, 14, 15, 17, 19, 22, 25, 28, 29, 35, 36, 37, 38,
               42, 43, 51, 55, 56, 62, 64, 73, 95, 96, 97, 101)
TRAIN_DEMOS = tuple(range(16, 24))
PROBE_TASKS = (0, 1, 12, 13, 20, 21, 32, 34)
LABEL_SCHEMA = 'ember_relation_physical_labels_v1'
ALIGNMENT = 'states[i+1]; mj_integratePos(-timestep); mj_forward'
FIELD_SHAPES = {
    'p': (33, 3), 'R': (33, 3, 3), 'semantic': (33, 64),
    'presence': (33,), 'joint_type': (33, 3), 'joint_q': (33, 2),
    'hand_q': (2,), 'valid': (), 'frame_indices': (),
    'p_mask': (33,), 'R_mask': (33,), 'semantic_mask': (33,),
    'joint_type_mask': (33,), 'joint_q_mask': (33, 2), 'hand_q_mask': (2,),
}


def _condition(task: int, demo: int) -> tuple[int, int]:
    if task not in TRAIN_TASKS or demo not in TRAIN_DEMOS:
        raise ValueError(f'Phi supervision excludes task{task}/demo{demo}; probe/held labels are sealed')
    return int(task), int(demo)


@lru_cache(maxsize=1)
def _semantic_table() -> dict[str, np.ndarray]:
    record = json.loads((LABEL_ROOT / 'semantic_vectors.json').read_text())
    if (record.get('normalization') != 'unit64' or record.get('projection') !=
            'NumPy PCG64 seed20261006 QR2048x64 sign_fixed FP64'):
        raise ValueError('the existing seed20261006 semantic projection contract changed')
    vectors = {name: np.asarray(value, dtype=np.float32)
               for name, value in record['vectors'].items()}
    for name, vector in vectors.items():
        if vector.shape != (64,) or not np.isfinite(vector).all():
            raise ValueError(f'invalid retained semantic vector: {name}')
    return vectors


def semantic_vector(name: str) -> np.ndarray:
    """Return a retained generic public-name vector, without creating a dictionary."""
    if name not in _semantic_table():
        raise ValueError(f'public semantic name absent from the retained vocabulary: {name}')
    return _semantic_table()[name].copy()


def semantic_vectors() -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in _semantic_table().items()}


@lru_cache(maxsize=224)
def _read(task: int, demo: int, root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    _condition(task, demo)
    stem = root / f'task{task}_demo{demo}'
    metadata = json.loads(stem.with_suffix('.json').read_text())
    with np.load(stem.with_suffix('.npz'), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in FIELD_SHAPES}
    _check(metadata, arrays, task, demo)
    return metadata, arrays


def _check_registry(registry: dict, task: int, demo: int) -> None:
    entities = registry['entities']
    if (registry.get('entity_capacity') != 32 or registry.get('hand_slot') != 0
            or registry['entity_count'] != len(entities) or len(entities) > 32
            or [entity['slot'] for entity in entities] != list(range(1, len(entities) + 1))):
        raise ValueError(f'physical registry capacity/order differs at task{task}/demo{demo}')


def _check_frames(metadata: dict, arrays: dict, task: int, demo: int) -> None:
    indices = arrays['frame_indices']
    if (indices.dtype.kind not in 'iu' or not len(indices) or indices[0] != 0
            or np.any(np.diff(indices) <= 0) or indices.tolist() != metadata['frames']):
        raise ValueError(f'invalid real frame indices at task{task}/demo{demo}')
    valid = arrays['valid'].astype(bool)
    if not valid[:-1].all() or valid[-1] or int(valid.sum()) != metadata['valid_frames']:
        raise ValueError(f'expected only the original missing terminal state at task{task}/demo{demo}')


def _check(metadata: dict, arrays: dict, task: int, demo: int) -> None:
    if (metadata.get('schema') != LABEL_SCHEMA or metadata.get('task') != task
            or metadata.get('demo') != demo or metadata.get('state_alignment') != ALIGNMENT
            or metadata.get('semantic_projection_seed') != 20261006):
        raise ValueError(f'physical label provenance differs at task{task}/demo{demo}')
    registry = metadata['registry']
    entities, indices = registry['entities'], arrays['frame_indices']
    _check_registry(registry, task, demo)
    for name, suffix in FIELD_SHAPES.items():
        if arrays[name].shape != (len(indices), *suffix):
            raise ValueError(f'physical label shape differs: task{task}/demo{demo}/{name}')
    _check_frames(metadata, arrays, task, demo)
    valid = arrays['valid'].astype(bool)
    present = arrays['presence'].astype(bool)
    if not present[valid, :len(entities) + 1].all() or present[valid, len(entities) + 1:].any():
        raise ValueError(f'physical existence differs from registry at task{task}/demo{demo}')
    for name in ('p', 'R', 'semantic', 'joint_type', 'joint_q', 'hand_q'):
        mask = arrays[f'{name}_mask'].astype(bool) & valid.reshape(
            len(valid), *([1] * (arrays[f'{name}_mask'].ndim - 1)))
        if not np.isfinite(arrays[name][mask]).all():
            raise ValueError(f'nonfinite valid physical label: task{task}/demo{demo}/{name}')
    for entity in entities:
        if entity['semantic'] not in _semantic_table():
            raise ValueError(f'unknown retained semantic name: {entity["semantic"]}')


@lru_cache(maxsize=28)
def _focus(bddl: str) -> tuple[tuple[str, ...], dict[str, str], dict[str, str]]:
    """Read public OOI and region owners, never goal/init predicates or ranges."""
    text = Path(bddl).read_text()
    block = re.search(r'\(:obj_of_interest\s+([^()]*)\)', text)
    if block is None or not block.group(1).split():
        raise ValueError(f'missing public object-focus list: {bddl}')
    regions = {f'{owner}_{name}': owner for name, owner in re.findall(
        r'\(([^\s()]+)\s+\(:target\s+([^\s()]+)\)', text)}
    fixtures = re.search(r'\(:fixtures\s+([^()]*)\)', text)
    if fixtures is None:
        raise ValueError(f'missing public fixtures: {bddl}')
    tokens, pending, index, excluded = fixtures.group(1).split(), [], 0, {}
    while index < len(tokens):
        if tokens[index] != '-':
            pending.append(tokens[index])
            index += 1
            continue
        if not pending or index + 1 >= len(tokens):
            raise ValueError(f'invalid public fixture declaration: {bddl}')
        category = tokens[index + 1]
        if {'table', 'floor'}.intersection(category.split('_')):
            excluded.update((name, category) for name in pending)
        pending, index = [], index + 2
    if pending:
        raise ValueError(f'untyped public fixture declaration: {bddl}')
    return tuple(block.group(1).split()), regions, excluded


@lru_cache(maxsize=224)
def _arena_exists(source_hdf5: str, demo: int, owner: str, category: str) -> bool:
    # Only XML metadata is read. No action/state/pose/outcome value is opened.
    with h5py.File(source_hdf5, 'r') as file:
        xml = file[f'data/demo_{demo}'].attrs['model_file']
    tree = ET.fromstring(xml)
    arena_names = {owner, category} | {'table', 'floor'}.intersection(category.split('_'))
    return any(element.get('name') in arena_names for element in tree.iter()
               if element.tag in ('body', 'geom'))


def _relevance(metadata: dict) -> tuple[np.ndarray, list[dict]]:
    registry = metadata['registry']
    names, regions, excluded = _focus(registry['bddl'])
    owners = {entity['source_object'] for entity in registry['entities']}
    wanted, missing = set(), []
    for name in names:
        owner = name if name in owners else regions.get(name)
        if owner not in owners:
            category = excluded.get(owner)
            known_arena = bool(category and _arena_exists(metadata['source_hdf5'], metadata['demo'],
                                                         owner, category))
            missing.append({'focus': name, 'region_owner': owner,
                            'public_category': category, 'owner_exists': known_arena,
                            'kind': 'retained_registry_arena_exclusion' if known_arena else 'unknown_owner',
                            'exclusion_reason': ('original registry excludes table/floor fixtures'
                                                 if known_arena else 'unresolved public focus owner'),
                            'relevance_group_mask': False})
        else:
            wanted.add(owner)
    target = np.zeros(32, dtype=np.float32)
    for entity in registry['entities']:
        target[entity['slot'] - 1] = entity['source_object'] in wanted
    return target, missing


def label_metadata(task: int, demo: int, *, root: Path = LABEL_ROOT) -> dict:
    """Supervision provenance for the run record; never a perception input."""
    task, demo = _condition(task, demo)
    metadata, _ = _read(task, demo, Path(root))
    _, gaps = _relevance(metadata)
    return {'task': task, 'demo': demo, 'label_schema': metadata['schema'],
            'state_alignment': metadata['state_alignment'],
            'public_focus': list(_focus(metadata['registry']['bddl'])[0]),
            'focus_exclusions': gaps, 'relevance_group_mask': not gaps}


def load_geometry_labels(task: int, demo: int, indices: Sequence[int] | torch.Tensor,
                         *, root: Path = LABEL_ROOT) -> dict[str, torch.Tensor]:
    """Select true cached frames. Missing labels are masked, never imputed.

    Returned CPU tensors use hand_* and T×32 entity fields. Task, registry and
    public focus lists are not returned and cannot become Phi conditions.
    """
    task, demo = _condition(task, demo)
    metadata, arrays = _read(task, demo, Path(root))
    requested = np.asarray(indices.detach().cpu() if isinstance(indices, torch.Tensor)
                           else indices)
    if (requested.ndim != 1 or requested.dtype.kind not in 'iu' or not len(requested)
            or np.any(np.diff(requested) <= 0)):
        raise ValueError('label selection needs ordered actual integer frame indices')
    rows = np.searchsorted(arrays['frame_indices'], requested)
    if ((rows >= len(arrays['frame_indices'])).any()
            or not np.array_equal(arrays['frame_indices'][rows], requested)):
        raise ValueError(f'uncached requested geometry at task{task}/demo{demo}: {requested.tolist()}')
    relevance, missing = _relevance(metadata)
    if any(gap['kind'] != 'retained_registry_arena_exclusion' for gap in missing):
        raise ValueError(f'unresolved public OOI label at task{task}/demo{demo}: {missing}')
    selected = {key: value[rows] for key, value in arrays.items()}
    result = {'hand_p': selected['p'][:, 0], 'hand_R': selected['R'][:, 0],
              'hand_q': selected['hand_q'], 'entity_p': selected['p'][:, 1:],
              'entity_R': selected['R'][:, 1:]}
    for name in ('semantic', 'presence', 'joint_type', 'joint_q', 'semantic_mask',
                 'joint_type_mask', 'joint_q_mask'):
        result[name] = selected[name][:, 1:]
    for name in ('p', 'R'):
        result[f'hand_{name}_mask'] = selected[f'{name}_mask'][:, 0]
        result[f'entity_{name}_mask'] = selected[f'{name}_mask'][:, 1:]
    for name in ('valid', 'frame_indices', 'hand_q_mask'):
        result[name] = selected[name]
    result['relevance'] = np.broadcast_to(relevance, (len(rows), 32)).copy()
    result['relevance_mask'] = np.broadcast_to(
        (selected['valid'] & (not missing))[:, None], (len(rows), 32)).copy()
    result['relevance_group_mask'] = np.asarray(not missing, dtype=bool)
    return {key: torch.from_numpy(value.copy() if value.ndim == 0 else np.ascontiguousarray(value))
            for key, value in result.items()}


def audit_labels(*, root: Path = LABEL_ROOT) -> dict:
    """Audit exactly the registered 28×8 pool; report concrete scientific gaps."""
    report = {'schema': LABEL_SCHEMA, 'expected_episodes': 224, 'episodes': 0,
              'max_entities': 0, 'terminal_missing_frames': 0,
              'missing_assets': [], 'geometry_errors': [], 'unrepresented_focus': [],
              'masked_relevance': [], 'relevance_supervised_episodes': 0,
              'tasks': list(TRAIN_TASKS), 'demos': list(TRAIN_DEMOS)}
    for task in TRAIN_TASKS:
        for demo in TRAIN_DEMOS:
            try:
                metadata, arrays = _read(task, demo, Path(root))
                canonical = list(range(0, int(arrays['frame_indices'][-1]) + 1, 5))
                canonical.append(int(arrays['frame_indices'][-1]))
                absent = sorted(set(canonical) - set(arrays['frame_indices']))
                if absent:
                    raise ValueError(f'canonical stride5 frames absent: {absent}')
                _, gaps = _relevance(metadata)
                for gap in gaps:
                    key = ('masked_relevance' if gap['kind'] == 'retained_registry_arena_exclusion'
                           else 'unrepresented_focus')
                    report[key].append(dict(task=task, demo=demo, **gap))
                report['relevance_supervised_episodes'] += int(not gaps)
                report['episodes'] += 1
                report['max_entities'] = max(report['max_entities'], metadata['registry']['entity_count'])
                report['terminal_missing_frames'] += int((~arrays['valid'].astype(bool)).sum())
            except FileNotFoundError as error:
                report['missing_assets'].append({'task': task, 'demo': demo, 'path': error.filename})
            except (ValueError, KeyError) as error:
                report['geometry_errors'].append({'task': task, 'demo': demo, 'error': str(error)})
    report['complete'] = (report['episodes'] == 224 and not report['missing_assets']
                          and not report['geometry_errors'] and not report['unrepresented_focus'])
    return report
