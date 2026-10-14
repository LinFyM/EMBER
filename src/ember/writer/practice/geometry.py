"""Own live physical observations; no teacher registry, goal or future state."""
from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
from scipy.spatial.transform import Rotation

SEMANTICS = Path('/data1/user/ymdai/ember_runs/relation_grounded_writer_20261006/labels/semantic_vectors.json')


def public_entities(path):
    """Read only public generic object categories, excluding table/floor arenas."""
    text, result = Path(path).read_text(), []
    for section in (':objects', ':fixtures'):
        block = re.search(r'\(' + re.escape(section) + r'\s+([^()]*)\)', text)
        if block is None:
            raise ValueError(f'missing public entity section {section}: {path}')
        words, pending, i = block.group(1).split(), [], 0
        while i < len(words):
            if words[i] != '-':
                pending.append(words[i]); i += 1; continue
            if not pending or i + 1 >= len(words):
                raise ValueError('invalid public category declaration')
            category = words[i + 1]
            if not (section == ':fixtures' and {'table', 'floor'}.intersection(category.split('_'))):
                result.extend((name, category) for name in pending)
            pending, i = [], i + 2
        if pending:
            raise ValueError('entity declaration lacks generic category')
    return result


def generic_name(name):
    if name.strip('_') == 'microdoorroot':
        name = 'door'
    words = re.sub(r'\d+', '', name).strip('_').replace('_', ' ').split()
    if not words or {'scene', 'path'}.intersection(words):
        raise ValueError('own physical semantic name is not a generic category')
    return ' '.join(words)


def _inside(model,body,root):
    while body not in (0,root):body=int(model.body_parentid[body])
    return body==root


def bind_entities(owner,model,bddl):
    entries = []
    for name, category in public_entities(bddl):
        if name not in owner.obj_body_id:
            raise ValueError(f'actual environment has no public object root {name}')
        root = int(owner.obj_body_id[name])
        for body in range(model.nbody):
            if not _inside(model,body,root):
                continue
            joints = np.flatnonzero(model.jnt_bodyid == body)
            internal = [int(j) for j in joints if int(model.jnt_type[j]) in (2, 3)]
            if len(joints) > 1 or any(int(model.jnt_type[j]) == 1 for j in joints):
                raise ValueError('multiple/ball joint has no registered physical representation')
            if body != root and not internal:
                continue
            if internal:
                descendants=[candidate for candidate in range(model.nbody) if _inside(model,candidate,body)]
                if not int(np.asarray(model.body_geomnum)[descendants].sum()):
                    raise ValueError('articulated entity has no visible geometry')
            body_name = model.body(body).name
            semantic = generic_name(category if body == root else body_name[len(name):])
            joint = internal[0] if internal else None
            kind = {2: 1, 3: 2}[int(model.jnt_type[joint])] if joint is not None else 0
            entries.append(dict(body=body_name, body_id=body, semantic=semantic,
                origin='root' if body == root else 'moving', joint_type=kind,
                joint=model.joint(joint).name if joint is not None else None,
                qpos_address=int(model.jnt_qposadr[joint]) if joint is not None else None))
    return entries


class PhysicalEntities:
    """Bind roots and visible articulated parts from this actual environment."""
    def __init__(self, env, task, paths):
        owner = env.env
        model = getattr(owner.sim.model, '_model', owner.sim.model)
        bddl = Path(paths['bddl_files']) / task['problem_folder'] / task['bddl_file']
        entries=bind_entities(owner,model,bddl)
        if not entries or len(entries) > 32 or len({e['body_id'] for e in entries}) != len(entries):
            raise ValueError('actual own entity registry is empty, duplicate or exceeds 32')
        vectors = json.loads(SEMANTICS.read_text())['vectors']
        self.entries = entries
        self.semantic = np.asarray([vectors[e['semantic']] for e in entries], dtype=np.float32)
        self.kind = np.asarray([e['joint_type'] for e in entries], dtype=np.int64)
        self.bodies = [e['body_id'] for e in entries]
        self.description = dict(entities=entries, semantic_source=str(SEMANTICS),
            teacher_registry_read=False, goal_or_OOI_read=False,
            timing='live post-mj_step pose cache and joint qpos; EEF/fingers from actual returned observation')

    def observe(self, env, observation):
        owner = env.env
        data = getattr(owner.sim.data, '_data', owner.sim.data)
        q = np.zeros((len(self.entries), 2), dtype=np.float32)
        for i, entry in enumerate(self.entries):
            if entry['joint_type']:
                q[i, entry['joint_type'] - 1] = data.qpos[entry['qpos_address']]
        result = dict(hand_p=np.asarray(observation['robot0_eef_pos'], dtype=np.float32),
            hand_R=Rotation.from_quat(observation['robot0_eef_quat']).as_matrix().astype(np.float32),
            hand_q=np.asarray(observation['robot0_gripper_qpos'], dtype=np.float32),
            entity_p=data.xpos[self.bodies].astype(np.float32).copy(),
            entity_R=data.xmat[self.bodies].reshape(-1, 3, 3).astype(np.float32).copy(),
            joint_q=q, joint_type=self.kind.copy(), semantic=self.semantic.copy(), registry=self.description)
        if (result['hand_p'].shape != (3,) or result['hand_q'].shape != (2,)
                or not all(np.isfinite(value).all() for key, value in result.items() if key != 'registry')):
            raise ValueError('actual own physical observation is invalid')
        return result
