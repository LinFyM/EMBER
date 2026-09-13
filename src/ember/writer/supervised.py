"""Complete-LoRA main FM with joint learned native and video-prior consumption."""
from __future__ import annotations

import time

import torch

from ember.writer.function_credit import paired_functional_credit
from ember.writer.functional import writer_chain_rule_surrogate
from ember.writer.native import autocast
from ember.writer.spatial_supervision import SpatialLabelStore, spatial_objective
from ember.writer.correction_supervision import CorrectionLabelStore, native_update_objective


def _encode_leaves(writer, responses, inputs, *, backward, return_grounding=False):
    leaves = tuple(value.detach().requires_grad_(backward) for value in responses)
    visuals = tuple(value.detach().requires_grad_(backward) for value in inputs[3])
    videos = writer.encode(leaves, *inputs[:3], visuals, *inputs[4:], return_grounding=return_grounding)
    return videos, leaves, visuals


def _native_cotangents(leaves, visuals):
    if any(value.grad is None for value in (*leaves, *visuals)):
        raise RuntimeError("functional Writer detached a native R/Z leaf")
    return tuple(value.grad for value in leaves), tuple(value.grad for value in visuals)


def replay_functional_credit(writer, responses, inputs, native_inputs, compiled, *,
                             spatial_labels=None, spatial_weight=0., correction_target=None, correction_weight=0.):
    """Replay all registered training credits through the same complete generator."""
    videos, leaves, visuals = _encode_leaves(writer, responses, inputs, backward=True,
                                           return_grounding=spatial_labels is not None)
    metrics, auxiliary = {}, 0.
    if spatial_labels is not None:
        videos, distributions = videos
        auxiliary, metrics = spatial_objective(distributions, spatial_labels)
    state = writer.decode(videos, inputs[0], native_inputs)
    objective = writer_chain_rule_surrogate(state, compiled) + spatial_weight * auxiliary
    if correction_target is not None:
        correction, correction_metrics = native_update_objective(state, correction_target, writer.contract)
        objective = objective + correction_weight * correction
        metrics.update(correction_metrics)
    objective.backward()
    return _native_cotangents(leaves, visuals), metrics



class SupervisedEngine:
    def __init__(self, runtime, data, cache, context, config) -> None:
        self.runtime, self.data, self.cache, self.config = runtime, data, cache, config
        self.device, self.step = context.device, 0
        self.spatial = SpatialLabelStore(data.asset_root, config["spatial_supervision"],
                                        data.tasks, data.video_pool)
        self.correction = CorrectionLabelStore(data.asset_root, config["correction_supervision"],
                                               data.tasks, data.video_pool, runtime.lora)

    def _time(self, timings, name, start):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timings[name] = time.perf_counter() - start
        return time.perf_counter()

    def _credit(self, state, batch, trace, offset, *, backward):
        with autocast(self.device):
            return paired_functional_credit(
                self.runtime.policy, state, self.runtime.lora, batch,
                seed=trace["policy_rng_seed"], device=self.device,
                random_batch=trace.get("policy_random_batch_size", len(trace["action_demos"])), offset=offset,
                microbatch=min(int(self.config["runtime"]["policy_microbatch"]), len(trace["action_demos"])),
                condition_weight=1.0 / (4 * self.config["data"]["conditions_per_task"]), backward=backward,
            )

    def backward(self, draw) -> dict:
        runtime, timings = self.runtime, {}
        start = time.perf_counter()
        task, demos = draw["task"], draw["video_demos"]
        hits, misses = self.cache.hits, self.cache.misses
        condition = self.cache.condition(task, demos)
        start = self._time(timings, "prefix_seconds", start)
        responses, inputs = runtime.observer.read(condition)
        labels = self.spatial.load(task, demos, inputs[0])
        target = self.correction.load(task, demos, self.device)
        start = self._time(timings, "observer_forward_seconds", start)
        with torch.no_grad(), autocast(self.device):
            videos = runtime.state.writer.encode(responses, *inputs)
            state = runtime.state.writer.decode(videos, inputs[0], condition.native_inputs)
        start = self._time(timings, "writer_forward_seconds", start)
        raw, trace = self.data.action_batch(
            task, draw["occurrence"], demos, query_seed=draw["query_seed"],
            query_offset=draw["query_offset"], query_count=draw["query_count"],
        )
        batch = runtime.processor.training_batch(raw)
        start = self._time(timings, "query_preparation_seconds", start)
        credit = self._credit(state, batch, trace, draw["query_offset"], backward=True)
        del batch, raw, state, videos
        start = self._time(timings, "fm_vjp_seconds", start)
        compiled = credit.pop("lora_cotangent")
        fm_norm = float(torch.stack([value.norm() for value in compiled.values()]).norm())
        with autocast(self.device):
            condition_weight = 1. / (4 * self.config["data"]["conditions_per_task"])
            cotangents, auxiliary_metrics = replay_functional_credit(
                runtime.state.writer, responses, inputs, condition.native_inputs, compiled, spatial_labels=labels,
                spatial_weight=self.config["spatial_supervision"]["weight"] * condition_weight,
                correction_target=target,
                correction_weight=self.config["correction_supervision"]["weight"] * condition_weight)
        del compiled, responses, inputs
        start = self._time(timings, "writer_vjp_seconds", start)
        runtime.observer.backward(condition, *cotangents)
        self._time(timings, "observer_vjp_seconds", start)
        del condition, cotangents
        return {**credit, **auxiliary_metrics, "task_weight": .25,
                "condition_weight": 1. / (4 * self.config["data"]["conditions_per_task"]),
                "normalizer": 1., "fm_lora_gradient_norm": fm_norm, "queries": len(trace["action_demos"]),
                **trace, **timings, "prefix_cache_hits": self.cache.hits - hits,
                "prefix_cache_misses": self.cache.misses - misses, "prefix_cache_bytes": self.cache.bytes,
                "policy_microbatch": int(self.config["runtime"]["policy_microbatch"])}

    @torch.no_grad()
    def validate(self, task: int, demo: int, *, seed: int, queries: int) -> dict:
        condition = self.cache.condition(task, (demo,))
        responses, inputs = self.runtime.observer.read(condition)
        raw, trace = self.data.diagnostic_batch(task, seed=seed, count=queries)
        batch = self.runtime.processor.training_batch(raw)
        with autocast(self.device):
            videos = self.runtime.state.writer.encode(responses, *inputs)
            state = self.runtime.state.writer.decode(videos, inputs[0], condition.native_inputs)
            credit = self._credit(state, batch, trace, 0, backward=False)
        credit.pop("lora_cotangent")
        return {"task": task, "suite": self.data.tasks[task].suite, "video_demos": [demo],
                **credit, "queries": queries, **trace, "gradients": False}
