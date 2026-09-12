"""Main complete-LoRA FM replay plus true local-action credit to the shared encoder."""
from __future__ import annotations

import math
import time

import torch

from ember.writer.attention import position_encoding
from ember.writer.function_credit import local_flow_sample, mean_velocity_loss, paired_functional_credit
from ember.writer.functional import writer_chain_rule_surrogate
from ember.writer.native import autocast


def _encode_leaves(writer, responses, inputs, *, backward):
    leaves = tuple(value.detach().requires_grad_(backward) for value in responses)
    visuals = tuple(value.detach().requires_grad_(backward) for value in inputs[3])
    videos = writer.encode(leaves, *inputs[:3], visuals, *inputs[4:])
    return videos, leaves, visuals


def _native_cotangents(leaves, visuals):
    if any(value.grad is None for value in (*leaves, *visuals)):
        raise RuntimeError("functional Writer detached a native R/Z leaf")
    return tuple(value.grad for value in leaves), tuple(value.grad for value in visuals)


def replay_functional_credit(writer, responses, inputs, compiled):
    """Replay the main FM cotangent through the single complete generation graph."""
    videos, leaves, visuals = _encode_leaves(writer, responses, inputs, backward=True)
    state = writer.decode(videos, inputs[0])
    writer_chain_rule_surrogate(state, compiled).backward()
    return _native_cotangents(leaves, visuals)


def local_action_credit(writer, reader, responses, inputs, sample, *, weight, backward):
    """All four frames supply local Values; no Compiler parameter enters this graph."""
    videos, leaves, visuals = _encode_leaves(writer, responses, inputs, backward=backward)
    if len(videos) != 1 or videos[0].shape[0] != 4:
        raise ValueError("local supervision requires one four-frame clip")
    video = videos[0]
    memory = video.flatten(0, 1)
    routing = torch.zeros_like(memory)
    if writer.config.process_mode == "ordered":
        relative = torch.arange(4, device=video.device)
        routing = position_encoding(relative, video.shape[-1], video.dtype)
        routing = routing[:, None].expand_as(video).flatten(0, 1)
    prior = memory.new_full((1, len(memory)), -math.log(len(memory)))
    noisy, flow_time, target = sample
    prediction = reader(noisy, flow_time, memory, routing, prior)
    loss = mean_velocity_loss(prediction, target, 7)
    if backward:
        (loss * weight).backward()
    return float(loss.detach()), _native_cotangents(leaves, visuals) if backward else None


class SupervisedEngine:
    def __init__(self, runtime, data, cache, context, config) -> None:
        self.runtime, self.data, self.cache, self.config = runtime, data, cache, config
        self.device, self.step = context.device, 0

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

    def local_action(self, task, occurrence, *, diagnostic=False):
        runtime, timings = self.runtime, {}
        start = time.perf_counter()
        frames, indices, raw, trace = self.data.local_action_clip(task, occurrence, diagnostic=diagnostic)
        actions = runtime.processor.normalize_action(raw)
        sample = local_flow_sample(actions, seed=trace["local_flow_seed"], draws=self.config["local_action"]["noise_draws"])
        condition = runtime.observer.prepare(frames, indices, self.data.tasks[task].authority.language)
        start = self._time(timings, "local_prefix_seconds", start)
        responses, inputs = runtime.observer.read(condition)
        start = self._time(timings, "local_observer_forward_seconds", start)
        with torch.set_grad_enabled(not diagnostic), autocast(self.device):
            loss, cotangents = local_action_credit(
                runtime.state.writer, runtime.state.reader, responses, inputs, sample,
                weight=.25 * self.config["local_action"]["weight"], backward=not diagnostic,
            )
        start = self._time(timings, "local_writer_seconds", start)
        if not diagnostic:
            runtime.observer.backward(condition, *cotangents)
        self._time(timings, "local_observer_vjp_seconds", start)
        return {"local_flow_loss": loss, "local_noise_draws": self.config["local_action"]["noise_draws"],
                **trace, **timings}

    def backward(self, draw) -> dict:
        runtime, timings = self.runtime, {}
        start = time.perf_counter()
        task, demos = draw["task"], draw["video_demos"]
        hits, misses = self.cache.hits, self.cache.misses
        condition = self.cache.condition(task, demos)
        start = self._time(timings, "prefix_seconds", start)
        responses, inputs = runtime.observer.read(condition)
        start = self._time(timings, "observer_forward_seconds", start)
        with torch.no_grad(), autocast(self.device):
            videos = runtime.state.writer.encode(responses, *inputs)
            state = runtime.state.writer.decode(videos, inputs[0])
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
            cotangents = replay_functional_credit(runtime.state.writer, responses, inputs, compiled)
        del compiled, responses, inputs
        start = self._time(timings, "writer_vjp_seconds", start)
        runtime.observer.backward(condition, *cotangents)
        self._time(timings, "observer_vjp_seconds", start)
        del condition, cotangents
        local = {"local_flow_loss": 0., "local_noise_draws": 0, "local_weight": 0.}
        if runtime.state.reader is not None and draw["condition_index"] == 0:
            local = {**self.local_action(task, draw["occurrence"]), "local_weight": .25}
        return {**credit, **local, "task_weight": .25,
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
            state = self.runtime.state.writer.decode(videos, inputs[0])
            credit = self._credit(state, batch, trace, 0, backward=False)
        credit.pop("lora_cotangent")
        return {"task": task, "suite": self.data.tasks[task].suite, "video_demos": [demo],
                **credit, "queries": queries, **trace, "gradients": False}
