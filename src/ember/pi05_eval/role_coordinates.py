"""Task-only exact world-XY intervention reused from the completed position batch."""
from pathlib import Path
import json
import numpy as np
from ember.pi05_assets import Pi05EvaluationError
NAMES=('butter_1','orange_juice_1')
def passive_in(env):
    owner = env.env
    goal = owner.parsed_problem['goal_state']
    if len(goal) != 1 or len(goal[0]) != 3 or str(goal[0][0]).lower() != 'in' or goal[0][1] != NAMES[0]:
        raise Pi05EvaluationError('butter basket native In goal changed')
    return np.asarray([owner._eval_predicate([goal[0][0], name, goal[0][2]]) for name in NAMES], dtype=np.bool_)


def _controller_snapshot(owner):
    return {key: np.asarray(value).copy() for key, value in vars(owner.robots[0].controller).items()
            if isinstance(value, (np.ndarray, float, int)) and np.asarray(value).dtype.kind in 'biuf'}


def _contacts(sim):
    return [{'geom1': int(c.geom1), 'geom2': int(c.geom2), 'distance_m': float(c.dist)}
            for c in sim.data.contact[:sim.data.ncon]]


def apply(env, observation, contract, state):
    """Translate registered free-joint roots by the required world-body delta; never step."""
    from ember.pi05_eval.scene import _capture_image

    owner, info = env.env, contract['role_coordinate_layout']
    sim, model, data = owner.sim, owner.sim.model, owner.sim.data
    registry = owner.obj_body_id
    bodies = [int(registry[name]) for name in NAMES]
    joints, addresses, roots = [], [], []
    for name in NAMES:
        obj = owner.objects_dict[name]
        if len(obj.joints) != 1:
            raise Pi05EvaluationError(f'{name}: expected one real free-joint root')
        joint = int(model.joint_name2id(obj.joints[0]))
        root = int(model.jnt_bodyid[joint])
        if int(model.jnt_type[joint]) != 0 or int(model.body_parentid[root]) != 0:
            raise Pi05EvaluationError(f'{name}: free-joint world frame is not established')
        joints.append(joint); roots.append(root); addresses.append(int(model.jnt_qposadr[joint]))
    before = {name: np.asarray(value).copy() for name, value in {
        'qpos': data.qpos, 'qvel': data.qvel, 'act': data.act,
        'body_xpos': data.body_xpos, 'body_xquat': data.body_xquat,
        'model_body_pos': model.body_pos, 'model_body_quat': model.body_quat,
        'sim_state': env.get_sim_state(), 'objects_world': data.body_xpos[bodies],
        'eef_pos': observation['robot0_eef_pos'], 'eef_quat': observation['robot0_eef_quat'],
        'gripper_qpos': observation['robot0_gripper_qpos'],
    }.items()}
    controller = _controller_snapshot(owner)
    initial_rgb = _capture_image(observation)
    sim_time, timestep = float(data.time), int(owner.timestep)
    old_contacts = _contacts(sim)
    allowed = np.zeros(len(data.qpos), dtype=np.bool_)
    if info['layout'] == 'swapped':
        for index, address in enumerate(addresses):
            delta = before['objects_world'][1-index, :2] - before['objects_world'][index, :2]
            data.qpos[address:address+2] += delta
            allowed[address:address+2] = True
    sim.forward()
    observation = owner._get_observations(force_update=True)
    after = {name: np.asarray(value).copy() for name, value in {
        'qpos': data.qpos, 'qvel': data.qvel, 'act': data.act,
        'body_xpos': data.body_xpos, 'body_xquat': data.body_xquat,
        'model_body_pos': model.body_pos, 'model_body_quat': model.body_quat,
        'sim_state': env.get_sim_state(), 'objects_world': data.body_xpos[bodies],
        'eef_pos': observation['robot0_eef_pos'], 'eef_quat': observation['robot0_eef_quat'],
        'gripper_qpos': observation['robot0_gripper_qpos'],
    }.items()}
    target = before['objects_world'].copy()
    if info['layout'] == 'swapped':
        target[:, :2] = target[::-1, :2]
    affected = np.zeros(model.nbody, dtype=np.bool_)
    for body in range(model.nbody):
        ancestor = body
        while ancestor and ancestor not in roots:
            ancestor = int(model.body_parentid[ancestor])
        affected[body] = ancestor in roots
    other_controller = _controller_snapshot(owner)
    invariants = {
        'only_world_XY': bool(np.allclose(after['objects_world'], target, rtol=0, atol=1e-9)),
        'other_qpos': bool(np.array_equal(before['qpos'][~allowed], after['qpos'][~allowed])),
        'qvel': bool(np.array_equal(before['qvel'], after['qvel'])),
        'act': bool(np.array_equal(before['act'], after['act'])),
        'body_quaternions': bool(np.allclose(before['body_xquat'], after['body_xquat'], rtol=0, atol=1e-9)),
        'other_bodies': bool(np.allclose(before['body_xpos'][~affected], after['body_xpos'][~affected], rtol=0, atol=1e-9)),
        'model_pose': bool(all(np.array_equal(before[k], after[k]) for k in ('model_body_pos', 'model_body_quat'))),
        'no_step': bool(float(data.time) == sim_time and int(owner.timestep) == timestep),
        'controller': bool(controller.keys() == other_controller.keys() and
                           all(np.array_equal(value, other_controller[key]) for key, value in controller.items())),
        'own_state8_inputs': bool(all(np.array_equal(before[key], after[key]) for key in
                                     ('eef_pos', 'eef_quat', 'gripper_qpos'))),
    }
    path = Path(contract['output_dir']) / 'initial_states' / f'state_{state:03d}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.savez_compressed(handle, **{f'before_{key}': value for key, value in before.items()},
                            **{f'after_{key}': value for key, value in after.items()},
                            body_names=np.asarray([model.body_id2name(i) or '' for i in range(model.nbody)]),
                            object_names=np.asarray(NAMES), object_body_ids=np.asarray(bodies),
                            joint_names=np.asarray([model.joint_id2name(j) for j in joints]),
                            qpos_addresses=np.asarray(addresses), before_rgb=initial_rgb,
                            after_rgb=_capture_image(observation), sim_time=np.asarray(sim_time),
                            **{f'controller_{key}': value for key, value in controller.items()})
    receipt = {**info, 'state': state, 'physical_state': {'path': str(path), 'bytes': path.stat().st_size},
               'before_world': before['objects_world'].tolist(), 'after_world': after['objects_world'].tolist(),
               'invariants': invariants, 'contacts_before': old_contacts, 'contacts_after': _contacts(sim),
               'native_passive_In': passive_in(env).tolist(), 'environment_steps_added': 0}
    path.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if not all(invariants.values()):
        raise Pi05EvaluationError(f'physical XY intervention invariant failed; preserved {path}')
    return observation, receipt


