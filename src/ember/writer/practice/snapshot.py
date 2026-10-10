"""Label-side live simulator/controller/wrapper/RNG snapshots, never Reader input."""
from __future__ import annotations

from copy import deepcopy
import random

import numpy as np
import torch

from ember.pi05_eval.scene import _scene_snapshot, _restore_scene, _assert_scene_pair


def _simple_attributes(owner):
    """Mutable numerical controller/wrapper state; exclude simulator references."""
    result = {}
    for key, value in vars(owner).items():
        if isinstance(value, (str, int, float, bool, type(None), np.ndarray, np.generic)):
            result[key] = deepcopy(value)
    return result


def capture_snapshot(env, observation, slot):
    owner = env.env
    names = sorted(owner.obj_body_id)
    goals = [list(row) for row in owner.parsed_problem['goal_state']]
    controllers = []
    for robot in owner.robots:
        controller = robot.controller
        interpolators = {key: _simple_attributes(value) for key, value in vars(controller).items()
                         if 'interpolator' in key and value is not None}
        controllers.append(dict(robot=_simple_attributes(robot), controller=_simple_attributes(controller),
            interpolators=interpolators, gripper=_simple_attributes(robot.gripper),
            buffers={key:_simple_attributes(value) for key,value in vars(robot).items()
                     if key.startswith('recent_') and hasattr(value,'__dict__')}))
    return dict(schema_version='ember_actual_practice_snapshot_v3',
        scene=_scene_snapshot(env, observation, names, goals, image=True), names=names, goals=goals,
        model_arrays={key: np.asarray(getattr(owner.sim.model, key)).copy()
                      for key in ('geom_rgba', 'site_rgba', 'eq_active') if hasattr(owner.sim.model, key)},
        simulator_arrays={key: np.asarray(getattr(owner.sim.data, key)).copy() for key in
            ('ctrl','qacc_warmstart','mocap_pos','mocap_quat','qfrc_applied','xfrc_applied','userdata')
            if hasattr(owner.sim.data,key)},
        observables={key:_simple_attributes(value) for key,value in owner._observables.items()},
        observation_cache=deepcopy(owner._obs_cache),
        wrapper=_simple_attributes(env), environment=_simple_attributes(owner), controllers=controllers,
        rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.random.get_rng_state()),
        episode_control_steps=slot['steps'], episode_replans=slot['replan_index'])


def restore_snapshot(env, snapshot):
    if snapshot.get('schema_version') != 'ember_actual_practice_snapshot_v3':
        raise ValueError('recovery needs a real live snapshot, never RGB/proprio reconstruction')
    owner = env.env
    observation = _restore_scene(env, snapshot['scene'])
    for key, value in snapshot['model_arrays'].items():
        getattr(owner.sim.model, key)[:] = value
    for object_, values in [(env, snapshot['wrapper']), (owner, snapshot['environment'])]:
        for key, value in values.items():
            setattr(object_, key, deepcopy(value))
    for robot, saved in zip(owner.robots, snapshot['controllers'], strict=True):
        for object_, values in [(robot, saved['robot']), (robot.controller, saved['controller'])]:
            for key, value in values.items():
                setattr(object_, key, deepcopy(value))
        for key, values in saved['interpolators'].items():
            object_ = getattr(robot.controller, key)
            for field, value in values.items():
                setattr(object_, field, deepcopy(value))
        for key,value in saved['gripper'].items():
            setattr(robot.gripper,key,deepcopy(value))
        for key,values in saved['buffers'].items():
            for name,value in values.items():
                setattr(getattr(robot,key),name,deepcopy(value))
    owner.sim.forward()
    for key,value in snapshot.get('simulator_arrays',{}).items():
        getattr(owner.sim.data,key)[:] = value
    if 'observables' not in snapshot:
        raise ValueError('old snapshot lacks live observable timing/cache; it cannot create recovery labels')
    owner._obs_cache=deepcopy(snapshot['observation_cache'])
    for key,values in snapshot['observables'].items():
        for name,value in values.items():
            setattr(owner._observables[key],name,deepcopy(value))
    observation = owner._get_observations(force_update=False)
    _assert_scene_pair(env, observation, snapshot['names'], snapshot['goals'], snapshot['scene'], image=True)
    random.setstate(snapshot['rng']['python'])
    np.random.set_state(snapshot['rng']['numpy'])
    torch.random.set_rng_state(snapshot['rng']['torch'])
    return observation
