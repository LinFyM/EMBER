"""Same-version rollout, joint LoRA cotangent and single Writer/Meta replay."""

from __future__ import annotations

from dataclasses import dataclass
import random
import time

import torch

from ember.writer.flow import flow_actions, flow_mean_lora_gradient
from ember.writer.functional import (
    INDEPENDENT_BETA_TIME_SAMPLING_SCHEME, INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    functional_lora_loss_gradient, writer_chain_rule_surrogate,
)
from ember.writer.native import NativeCondition, autocast
from ember.writer.rl_math import exploration_covariance, loo_advantages, rl_mean_cotangent, task_trust_kl
from ember.writer.rollout import WriterRollouts, decision_batch


@dataclass
class TrustEvidence:
    task: int
    condition: NativeCondition
    records: list[dict]
    episode_ids: list[int]


class JointUpdateEngine:
    def __init__(self, runtime, data, cache, context, config, asset_root, output) -> None:
        self.runtime, self.data, self.cache, self.config = runtime, data, cache, config
        self.device = context.device
        self.rollouts = WriterRollouts(runtime, data, asset_root, output, context, config)
        self.precision = torch.linalg.inv(exploration_covariance(device=self.device))

    def _time(self, timings, name, start):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timings[name] = time.perf_counter() - start
        return time.perf_counter()

    def backward(self, draw) -> tuple[dict, TrustEvidence]:
        runtime = self.runtime
        timings = {}
        start = time.perf_counter()
        task, demos = draw["task"], draw["video_demos"]
        condition = self.cache.condition(task, demos)
        inputs = runtime.observer.writer_arguments(condition)
        start = self._time(timings, "prefix_seconds", start)
        responses = runtime.observer.responses(condition)
        start = self._time(timings, "observer_forward_seconds", start)
        with torch.no_grad(), autocast(self.device):
            state = runtime.state.writer(responses, *inputs)
        start = self._time(timings, "writer_forward_seconds", start)
        episodes = self.rollouts.collect(task, state, draw["episodes"])
        start = self._time(timings, "rollout_seconds", start)
        raw_batch, query_trace = self.data.action_batch(
            task, draw["occurrence"], demos, query_seed=draw["query_seed"],
        )
        batch = runtime.processor.training_batch(raw_batch)
        with autocast(self.device):
            loss, _, gradients = functional_lora_loss_gradient(
                runtime.policy, state, runtime.lora, batch=batch,
                policy_rng_seed=query_trace["policy_rng_seed"], policy_rng_device=self.device,
                flow_time_sampling_scheme=INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
                flow_noise_sampling_scheme=INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
                policy_microbatch_size=int(self.config["runtime"]["policy_microbatch"]),
                collect_policy_details=False,
            )
        gradients = {name: value.float().mul_(0.25) for name, value in gradients.items()}
        fm_norm = float(torch.stack([value.norm() for value in gradients.values()]).norm())
        del batch, raw_batch
        start = self._time(timings, "fm_vjp_seconds", start)
        advantages = loo_advantages(torch.tensor([episode.success for episode in episodes]))
        records, cotangents = [], []
        evidence = TrustEvidence(task, condition, [], [])
        for index, (episode, advantage) in enumerate(zip(episodes, advantages, strict=True)):
            saved = episode.reservoir.items
            selected = episode.reservoir.sample(rng=random.Random(episode.seed["trust"]))
            evidence.records.extend(selected)
            evidence.episode_ids.extend([index] * len(selected))
            if float(advantage) != 0:
                records.extend(saved)
                cotangents.append(rl_mean_cotangent(
                    torch.stack([record["z"] for record in saved]).to(self.device),
                    torch.stack([record["old_mean"] for record in saved]).to(self.device),
                    advantage, episode.reservoir.total_seen, len(saved), precision=self.precision,
                ))
        rl_gradients = {name: torch.zeros_like(value) for name, value in gradients.items()}
        if records:
            all_cotangents = torch.cat(cotangents)
            microbatch = int(self.config["runtime"]["rl_microbatch"])
            for begin in range(0, len(records), microbatch):
                stop = begin + microbatch
                batch, noise = decision_batch(records[begin:stop], self.device)
                with autocast(self.device):
                    part = flow_mean_lora_gradient(
                        runtime.policy, state, runtime.lora, batch, noise, all_cotangents[begin:stop],
                    )
                for name, value in part.items():
                    rl_gradients[name].add_(value)
            del batch, noise, part, all_cotangents
        rl_norm = float(torch.stack([value.norm() for value in rl_gradients.values()]).norm())
        for name, value in rl_gradients.items():
            gradients[name].add_(value)
        del rl_gradients, records, cotangents, state
        start = self._time(timings, "rl_vjp_seconds", start)
        leaves = tuple(value.detach().requires_grad_(True) for value in responses)
        with autocast(self.device):
            replay = runtime.state.writer(leaves, *inputs)
            surrogate = writer_chain_rule_surrogate(replay, gradients)
        surrogate.backward()
        cotangents = tuple(value.grad for value in leaves)
        if any(value is None for value in cotangents):
            raise RuntimeError("joint Writer detached an R leaf")
        del replay, surrogate, gradients, leaves, responses, inputs
        start = self._time(timings, "writer_vjp_seconds", start)
        runtime.observer.backward(condition, cotangents)
        self._time(timings, "observer_vjp_seconds", start)
        return {
            "flow_loss": float(loss), "task_weight": 0.25, "normalizer": 1.0,
            "fm_lora_gradient_norm": fm_norm, "rl_lora_gradient_norm": rl_norm,
            "rl_successes": sum(episode.success for episode in episodes), "rl_episodes": 4,
            "rl_mixed_group": bool((advantages != 0).any()),
            "rl_decisions": [episode.reservoir.total_seen for episode in episodes],
            "rl_saved_decisions": [len(episode.reservoir.items) for episode in episodes],
            "rl_steps": [episode.steps for episode in episodes],
            "rl_rewards": [int(episode.success) for episode in episodes],
            "rl_advantages": advantages.tolist(), "rl_seeds": [episode.seed for episode in episodes],
            "queries": len(query_trace["action_demos"]), **query_trace, **timings,
        }, evidence

    @torch.no_grad()
    def trust_score(self, evidence: TrustEvidence) -> float:
        runtime = self.runtime
        responses = runtime.observer.responses(evidence.condition)
        with autocast(self.device):
            state = runtime.state.writer(responses, *runtime.observer.writer_arguments(evidence.condition))
        del responses
        means = []
        microbatch = int(self.config["runtime"]["trust_microbatch"])
        for start in range(0, len(evidence.records), microbatch):
            batch, noise = decision_batch(evidence.records[start:start + microbatch], self.device)
            with autocast(self.device):
                output = flow_actions(runtime.policy, state, runtime.lora, batch, noise)
            means.append(output[:, :5, :7].flatten(1).float())
        means = torch.cat(means)
        old_means = torch.stack([record["old_mean"] for record in evidence.records]).to(self.device)
        return task_trust_kl(
            means, old_means, [evidence.task] * len(means), evidence.episode_ids, precision=self.precision,
        )[evidence.task]

    def close(self) -> None:
        self.rollouts.close()
