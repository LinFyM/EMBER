"""Cross-episode FM cotangent followed by one complete Writer replay."""

from __future__ import annotations

import time

import torch

from ember.writer.functional import (
    INDEPENDENT_BETA_TIME_SAMPLING_SCHEME, INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    functional_lora_loss_gradient, functional_lora_loss_value, writer_chain_rule_surrogate,
)
from ember.writer.native import autocast


class SupervisedEngine:
    def __init__(self, runtime, data, cache, context, config) -> None:
        self.runtime, self.data, self.cache, self.config = runtime, data, cache, config
        self.device = context.device

    def _time(self, timings, name, start):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timings[name] = time.perf_counter() - start
        return time.perf_counter()

    def backward(self, draw) -> dict:
        runtime, timings = self.runtime, {}
        start = time.perf_counter()
        task, demos = draw["task"], draw["video_demos"]
        hits, misses = self.cache.hits, self.cache.misses
        condition = self.cache.condition(task, demos)
        inputs = runtime.observer.writer_arguments(condition)
        start = self._time(timings, "prefix_seconds", start)
        responses = runtime.observer.responses(condition)
        start = self._time(timings, "observer_forward_seconds", start)
        with torch.no_grad(), autocast(self.device):
            state = runtime.state.writer(responses, *inputs)
        start = self._time(timings, "writer_forward_seconds", start)
        raw_batch, query_trace = self.data.action_batch(
            task, draw["occurrence"], demos, query_seed=draw["query_seed"],
        )
        batch = runtime.processor.training_batch(raw_batch)
        start = self._time(timings, "query_preparation_seconds", start)
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
        del batch, raw_batch, state
        start = self._time(timings, "fm_vjp_seconds", start)
        leaves = tuple(value.detach().requires_grad_(True) for value in responses)
        with autocast(self.device):
            replay = runtime.state.writer(leaves, *inputs)
            surrogate = writer_chain_rule_surrogate(replay, gradients)
        surrogate.backward()
        cotangents = tuple(value.grad for value in leaves)
        if any(value is None for value in cotangents):
            raise RuntimeError("supervised Writer detached an R leaf")
        del replay, surrogate, gradients, leaves, responses, inputs
        start = self._time(timings, "writer_vjp_seconds", start)
        runtime.observer.backward(condition, cotangents)
        self._time(timings, "observer_vjp_seconds", start)
        return {
            "flow_loss": float(loss), "task_weight": 0.25, "normalizer": 1.0,
            "fm_lora_gradient_norm": fm_norm,
            "queries": len(query_trace["action_demos"]), **query_trace, **timings,
            "prefix_cache_hits": self.cache.hits - hits,
            "prefix_cache_misses": self.cache.misses - misses,
            "prefix_cache_bytes": self.cache.bytes,
            "policy_microbatch": int(self.config["runtime"]["policy_microbatch"]),
        }

    @torch.no_grad()
    def validate(self, task: int, demo: int, *, seed: int, queries: int) -> dict:
        condition = self.cache.condition(task, (demo,))
        responses = self.runtime.observer.responses(condition)
        raw, trace = self.data.diagnostic_batch(task, seed=seed, count=queries)
        batch = self.runtime.processor.training_batch(raw)
        with autocast(self.device):
            state = self.runtime.state.writer(responses, *self.runtime.observer.writer_arguments(condition))
            loss, _ = functional_lora_loss_value(
                self.runtime.policy, state, self.runtime.lora, batch=batch,
                policy_rng_seed=trace["policy_rng_seed"], policy_rng_device=self.device,
                flow_time_sampling_scheme=INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
                flow_noise_sampling_scheme=INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
                policy_microbatch_size=int(self.config["runtime"]["policy_microbatch"]),
                collect_policy_details=False,
            )
        return {"task": task, "suite": self.data.tasks[task].suite, "video_demos": [demo],
                "flow_loss": float(loss), "queries": queries, **trace, "gradients": False}
