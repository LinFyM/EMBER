"""Bounded persistent CPU environment slots with independent process RNG."""
from __future__ import annotations

from copy import deepcopy
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import time
import traceback

import numpy as np
import torch

from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_eval.episode import start_fixed_episode, finish_episode_row
from ember.pi05_processing import libero_policy_input


def raw_observation(obs, language):
    value = libero_policy_input(obs, language)
    images = torch.stack((value['observation.images.base_0_rgb'],
                          value['observation.images.left_wrist_0_rgb'])).mul(255).round().to(torch.uint8)
    return dict(images=images, proprio=value['observation.state'])


def _wire(raw):
    return {k: v.numpy() for k, v in raw.items()}


def _serve(pipe, contract, physical_gpu, affinity):
    """One active environment per process; reset cannot seed another slot."""
    if affinity:
        os.sched_setaffinity(0, affinity)
    torch.set_num_threads(1)
    pool, slot = None, None
    operation_steps = 0
    started = time.monotonic()
    try:
        pool = PersistentTaskEnvironmentPool(contract, physical_gpu_id=physical_gpu)
        while True:
            command = pipe.recv()
            if command['kind'] == 'close':
                break
            tick = time.monotonic()
            operation_steps = 0
            if command['kind'] == 'start':
                task, episode_contract = command['task'], deepcopy(command['contract'])
                noise_root = command['noise_root']
                episode_contract['parallel'] = dict(envs_per_replica=1)
                pool.contract = episode_contract
                envs, states = pool.switch(task)
                env = envs[0]
                settling = int(episode_contract['environment']['dummy_settling_steps'])
                remaining = command.get('remaining')
                if remaining is not None:
                    settling = min(settling, int(remaining))
                    episode_contract['environment']['dummy_settling_steps'] = settling
                slot = start_fixed_episode(env=env, init_state_id=command['state_id'], init_states=states,
                    task=task, contract=episode_contract, root_seed=command['noise_root'],
                    dummy=np.asarray(episode_contract['environment']['dummy_action']),
                    task_adapter=None, capture_level=None)
                controls = int(episode_contract['environment']['horizons'][task['suite']])
                if remaining is not None:
                    controls = min(controls, remaining - settling)
                initial = raw_observation(slot['obs'], task['language'])
                operation_steps = settling
                initial_proprio = initial['proprio'].tolist()
                reward, done = 0., False
                result = dict(kind='start', raw=_wire(initial), settling_steps=settling,
                    state_id=command['state_id'], controls=controls, initial_proprio=initial_proprio)
            elif command['kind'] == 'step':
                if slot is None:
                    raise ValueError('environment step preceded a registered start')
                executed = []
                for action in command['actions'][:min(5, controls - slot['steps'])]:
                    slot['obs'], reward, done, _ = env.step(action.tolist())
                    slot['steps'] += 1
                    operation_steps += 1
                    executed.append(action)
                    if done:
                        break
                slot['replan_index'] += 1
                slot['policy_noise_seeds'].append(command['noise_seed'])
                slot['episode_done'] = bool(done)
                result = dict(kind='step', raw=_wire(raw_observation(slot['obs'], task['language'])),
                    executed=np.asarray(executed, dtype=np.float32).reshape(-1, 7),
                    done=bool(done), reward=float(reward), steps=slot['steps'],
                    replans=slot['replan_index'])
            else:
                raise ValueError('unknown environment slot operation')
            ended = bool(done or slot['steps'] >= controls)
            result.update(episode_ended=ended, operation_seconds=time.monotonic() - tick)
            if ended:
                slot['episode_done'] = bool(done)
                row = finish_episode_row(slot=slot, task=task,
                    contract={**episode_contract, 'rng': dict(inference_seed=noise_root)},
                    task_adapter=None, worker_started=started)
                truncated = bool(remaining is not None and settling + slot['steps'] == remaining and not done)
                row.update(settling_steps=settling, environment_steps=settling + slot['steps'],
                    budget_truncated=truncated, replans=slot['replan_index'], initial_proprio=initial_proprio)
                result['row'] = row
            pipe.send(result)
            operation_steps = 0
    except EOFError:
        pass
    except BaseException:
        try:
            pipe.send(dict(error=traceback.format_exc(), confirmed_environment_steps=operation_steps))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if pool is not None:
            pool.close()
        pipe.close()


class EnvironmentSlots:
    """CPU reset/step messages overlap GPU work; bound resident EGL contexts."""
    def __init__(self, contract, physical_gpu, count):
        if count < 1:
            raise ValueError('slot count must be positive')
        contract = deepcopy(contract)
        contract['parallel'] = dict(envs_per_replica=1)
        context = mp.get_context('spawn')
        self.pipes, self.processes, self.pending = [], [], {}
        affinity = list(os.sched_getaffinity(0))
        for _ in range(count):
            parent, child = context.Pipe()
            process = context.Process(target=_serve, args=(child, contract, physical_gpu, affinity), daemon=True)
            process.start()
            child.close()
            self.pipes.append(parent)
            self.processes.append(process)

    def submit(self, index, kind, **fields):
        if index in self.pending:
            raise ValueError('one environment slot already has an outstanding operation')
        self.pipes[index].send(dict(kind=kind, **fields))
        self.pending[index] = kind

    def receive(self, *, block=False):
        connections = [self.pipes[index] for index in self.pending]
        if not connections:
            return []
        ready = wait(connections, timeout=None if block else 0)
        output = []
        for pipe in ready:
            index = self.pipes.index(pipe)
            try:
                result = pipe.recv()
            except EOFError:
                result = dict(error='environment process exited without an operation receipt',
                              confirmed_environment_steps=0)
            self.pending.pop(index)
            if 'raw' in result:
                result['raw'] = {k: torch.from_numpy(v) for k, v in result['raw'].items()}
            output.append((index, result))
        return output

    def close(self):
        # Only task-owned child processes are stopped; other GPU users remain.
        while self.pending:
            self.receive(block=True)
        for pipe, process in zip(self.pipes, self.processes, strict=True):
            try:
                pipe.send(dict(kind='close'))
            except (BrokenPipeError, EOFError):
                pass
            process.join(timeout=20)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            pipe.close()
