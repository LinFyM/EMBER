"""Budget/success driven practice, cumulative facts and independent query episodes."""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import time

import numpy as np
import torch

from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_eval.episode import start_fixed_episode, finish_episode_row
from ember.pi05_eval_contract import policy_noise_seed
from ember.pi05_processing import libero_policy_input
from ember.writer.runtime import autocast

from .execution import NativeVelocity, action_chunk


def cpu_state(state):
    return {k: v.detach().cpu() for k, v in state.items()}


def raw_observation(obs, language):
    value = libero_policy_input(obs, language)
    images = torch.stack((value['observation.images.base_0_rgb'],
                          value['observation.images.left_wrist_0_rgb'])).mul(255).round().to(torch.uint8)
    return {'images': images, 'proprio': value['observation.state']}


def processed(runtime, raw, language):
    return runtime.processor({'observation.images.base_0_rgb': raw['images'][0].float().div(255),
        'observation.images.left_wrist_0_rgb': raw['images'][1].float().div(255),
        'observation.state': raw['proprio'], 'task': language})


@dataclass
class Chain:
    states: list = field(default_factory=list)
    records: list = field(default_factory=list)
    endpoints: list = field(default_factory=list)
    episodes: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def experience(self, endpoint):
        rows = self.records[:endpoint]
        if not rows:
            return {}
        result = {key: torch.stack([row[key] for row in rows]) for key in (
            'hidden', 'proprio', 'actions', 'executed', 'feedback')}
        result['images'] = torch.stack([torch.stack((row['_phi_pre'], row['_phi_post'])) for row in rows])
        result.update({key: torch.tensor([row[key] for row in rows], dtype=torch.long)
                       for key in ('episode', 'step')})
        return result

    @property
    def experiences(self):
        return self.to_record()['records']

    def to_record(self):
        return dict(records=[{k: v for k, v in row.items() if not k.startswith('_phi')}
                             for row in self.records], endpoints=self.endpoints,
                    episodes=self.episodes, metrics=self.metrics)


class Runner:
    def __init__(self, runtime, contract, physical_gpu):
        self.runtime, self.contract = runtime, copy.deepcopy(contract)
        self.contract.setdefault('parallel', {'envs_per_replica': 1})
        self.pool = PersistentTaskEnvironmentPool(self.contract, physical_gpu_id=int(physical_gpu))
        self.started = time.monotonic()
        self.total_environment_steps = 0

    def _start(self, task, state_id, noise_root, *, remaining=None):
        envs, states = self.pool.switch(task)
        settling = int(self.contract['environment']['dummy_settling_steps'])
        if remaining is not None:
            settling = min(settling, remaining)
        contract = copy.deepcopy(self.contract)
        contract['environment']['dummy_settling_steps'] = settling
        slot = start_fixed_episode(env=envs[0], init_state_id=state_id, init_states=states,
            task=task, contract=contract, root_seed=noise_root,
            dummy=np.asarray(contract['environment']['dummy_action']), task_adapter=None, capture_level=None)
        self.total_environment_steps += settling
        return envs[0], slot, settling

    @torch.no_grad()
    def episode(self, task, state_id, state, *, noise_root=7, remaining=None,
                collect=False, episode_index=0, sde=False):
        env, slot, settling = self._start(task, state_id, noise_root, remaining=remaining)
        horizon = int(self.contract['environment']['horizons'][task['suite']])
        controls = horizon if remaining is None else min(horizon, remaining - settling)
        rows, reservoir, replans = [], [], 0
        selection = np.random.default_rng(np.random.SeedSequence([noise_root, state_id, 0x51DE]))
        done, reward = False, 0.
        raw = raw_observation(slot['obs'], task['language'])
        initial_proprio = raw['proprio'].tolist()
        with self.runtime.execution.activate([state]), autocast(self.runtime.device):
            while slot['steps'] < controls and not done:
                velocity = NativeVelocity(self.runtime.policy, processed(self.runtime, raw, task['language']))
                if rows:
                    rows[-1]['_phi_post'] = velocity.phi[0]
                seed = policy_noise_seed(noise_root, task['suite'], int(task['task_id']), state_id, replans)
                noise = torch.randn((1, 50, 32), generator=torch.Generator().manual_seed(seed)).to(self.runtime.device)
                slot['policy_noise_seeds'].append(seed)
                sde_seed = policy_noise_seed(noise_root, 'sde:' + task['suite'], int(task['task_id']), state_id, replans)
                chunk, hidden, path = action_chunk(velocity, noise, sde_seed=sde_seed if sde else None)
                actions = self.runtime.processor.unnormalize_action(chunk)[0].cpu().numpy()
                executed = []
                before, before_step = raw, slot['steps']
                for action in actions[:min(5, controls - slot['steps'])]:
                    slot['obs'], reward, done, _ = env.step(action.tolist())
                    slot['steps'] += 1
                    self.total_environment_steps += 1
                    executed.append(torch.as_tensor(action).clone())
                    if done:
                        break
                raw = raw_observation(slot['obs'], task['language'])
                truncated = remaining is not None and settling + slot['steps'] == remaining and not done
                if collect:
                    padded = torch.zeros(5, 7)
                    padded[:len(executed)] = torch.stack(executed)
                    rows.append(dict(images=torch.stack((before['images'], raw['images'])), hidden=hidden[0],
                        proprio=torch.stack((before['proprio'], raw['proprio'])), actions=padded,
                        executed=torch.arange(5) < len(executed),
                        feedback=torch.tensor([float(reward), float(done), float(done), float(truncated)]),
                        episode=episode_index, step=before_step, _phi_pre=velocity.phi[0]))
                replans += 1
                if sde:
                    chosen = selection.choice(10, 2, replace=False)
                    record = dict(raw=before, transitions=[path[int(i)].cpu() for i in chosen],
                                  replan=replans - 1, noise_seed=seed, sde_seed=sde_seed)
                    replace = replans - 1 if replans <= 16 else int(selection.integers(replans))
                    if replace < 16:
                        if replans <= 16:
                            reservoir.append(record)
                        else:
                            reservoir[replace] = record
                slot['replan_index'] = replans
                del velocity, path
        if rows:
            rows[-1]['_phi_post'] = self.runtime.frozen_images(raw['images'][None])[0]
        slot['episode_done'] = bool(done)
        row_contract = {**self.contract, 'rng': {'inference_seed': noise_root}}
        row = finish_episode_row(slot=slot, task=task, contract=row_contract,
                                 task_adapter=None, worker_started=self.started)
        row.update(settling_steps=settling, environment_steps=settling + slot['steps'],
                   budget_truncated=bool(remaining is not None and settling + slot['steps'] == remaining and not done),
                   replans=replans, initial_proprio=initial_proprio)
        return row, rows, reservoir

    @torch.no_grad()
    def adapt(self, task, teacher, condition_seed, excluded_states, masked=False):
        """The only adaptation stopping rule is own success or 1024 real steps."""
        from .contract import state_stream

        chain, budget = Chain(), 1024
        started, native_before = time.monotonic(), self.runtime.feature_seconds
        with autocast(self.runtime.device):
            q = self.runtime.compiler.initial(teacher)
            current = self.runtime.compiler.decode(q)
        chain.states.append(cpu_state(current))
        stream = state_stream({'seed': condition_seed}, excluded_states)
        used_states, repeats, success = set(), 0, False
        while budget > 0:
            state_id = next(stream)
            repeats += int(state_id in used_states)
            used_states.add(state_id)
            noise_root = int(np.random.SeedSequence([condition_seed, len(chain.episodes), 0xADA]).generate_state(1)[0])
            row, facts, _ = self.episode(task, state_id, current, noise_root=noise_root, remaining=budget,
                                         collect=True, episode_index=len(chain.episodes))
            budget -= row['environment_steps']
            chain.records.extend(facts)
            chain.episodes.append(row)
            success = row['success']
            if success:
                break
            # A partial settling-only reset has no new decision evidence.
            if facts:
                chain.endpoints.append(len(chain.records))
                with autocast(self.runtime.device):
                    q = self.runtime.compiler.revise(q, teacher, {} if masked else chain.experience(len(chain.records)))
                    current = self.runtime.compiler.decode(q)
                chain.states.append(cpu_state(current))
        reads = 1 + len(chain.endpoints)
        chain.metrics = dict(initial_reads=1, rereads=len(chain.endpoints), full_video_equivalent_reads=float(reads),
            teacher_frames=len(teacher['indices']), read_frames=len(teacher['indices']) * reads,
            environment_steps=1024 - budget, resets=len(chain.episodes), state_reuses=repeats,
            stop_reason='own_success' if success else 'step_budget', practice_success=success,
            masked_experience=bool(masked), wall_seconds=time.monotonic() - started,
            native_feature_seconds=self.runtime.feature_seconds - native_before,
            excluded_states=list(map(int, excluded_states)), condition_seed=int(condition_seed))
        chain.metrics['teacher_feature_cost'] = dict(self.runtime.last_teacher_cost)
        chain.q_context = q.detach().float().mean((0, 1)).cpu()
        return chain

    def query(self, task, state_id, state, seed):
        row, _, reservoir = self.episode(task, state_id, state, noise_root=seed, sde=True)
        return dict(row=row, reservoir=reservoir)

    def final(self, task, state_id, lora, noise_root=7):
        return self.episode(task, state_id, lora, noise_root=noise_root)[0]

    @torch.no_grad()
    def final_many(self, task, state_ids, lora, noise_root=7):
        """Batch the actual independent final episodes, with per-slot RNG."""
        envs, initial = self.pool.switch(task)
        if len(state_ids) > len(envs):
            raise ValueError('final environment batch exceeds the registered persistent pool')
        slots = []
        settling = int(self.contract['environment']['dummy_settling_steps'])
        horizon = int(self.contract['environment']['horizons'][task['suite']])
        for env, state_id in zip(envs, state_ids):
            slot = start_fixed_episode(env=env, init_state_id=state_id, init_states=initial,
                task=task, contract=self.contract, root_seed=noise_root,
                dummy=np.asarray(self.contract['environment']['dummy_action']),
                task_adapter=None, capture_level=None)
            self.total_environment_steps += settling
            slots.append(slot)
        active = list(range(len(slots)))
        with autocast(self.runtime.device):
            while active:
                inputs = [self.runtime.processor(libero_policy_input(slots[i]['obs'], task['language']))
                          for i in active]
                batch = {k: torch.cat([row[k] for row in inputs]) for k in inputs[0]}
                noise = []
                for i in active:
                    slot = slots[i]
                    seed = policy_noise_seed(noise_root, task['suite'], int(task['task_id']),
                                             slot['init_state_id'], slot['replan_index'])
                    slot['policy_noise_seeds'].append(seed)
                    noise.append(torch.randn((50, 32), generator=torch.Generator().manual_seed(seed)))
                with self.runtime.execution.activate([lora] * len(active)):
                    velocity = NativeVelocity(self.runtime.policy, batch)
                    chunks, _, _ = action_chunk(velocity, torch.stack(noise).to(self.runtime.device))
                actions = self.runtime.processor.unnormalize_action(chunks).cpu().numpy()
                continuing = []
                for i, plan in zip(active, actions):
                    slot, done = slots[i], False
                    for action in plan[:min(5, horizon - slot['steps'])]:
                        slot['obs'], _, done, _ = envs[i].step(action.tolist())
                        slot['steps'] += 1
                        self.total_environment_steps += 1
                        if done:
                            break
                    slot['replan_index'] += 1
                    slot['episode_done'] = bool(done)
                    if not done and slot['steps'] < horizon:
                        continuing.append(i)
                active = continuing
        row_contract = {**self.contract, 'rng': {'inference_seed': noise_root}}
        rows = [finish_episode_row(slot=s, task=task, contract=row_contract,
                                   task_adapter=None, worker_started=self.started) for s in slots]
        for row in rows:
            row.update(settling_steps=settling, environment_steps=settling + row['steps'],
                       replans=len(row['policy_noise_seeds']), physical_final_batch=len(state_ids))
        return rows

    def close(self):
        self.pool.close()
