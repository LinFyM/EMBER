"""One-condition direct functional credit, complete LoRA VJP and native replay."""
from __future__ import annotations

import time

import torch

from ember.writer.function_credit import paired_functional_credit
from ember.writer.function_reader import distillation_weight
from ember.writer.functional import writer_chain_rule_surrogate
from ember.writer.native import autocast


def replay_functional_credit(writer, responses, inputs, compiled, distilled, auxiliary, rho):
    """Route L_C to E and normalized L_C/L_D to G using one decode graph.

    Cotangents already include condition/query weights. Auxiliary is the L_R
    cotangent of flatten(E), independent of the Compiler's time-routing weights.
    """
    leaves = tuple(value.detach().requires_grad_(True) for value in responses)
    visual_leaves = tuple(value.detach().requires_grad_(True) for value in inputs[3])
    learned_inputs = (*inputs[:3], visual_leaves, *inputs[4:])
    videos = writer.encode(leaves, *learned_inputs)
    memory_leaves = tuple(video.detach().requires_grad_(True) for video in videos)
    state = writer.decode(memory_leaves, inputs[0])
    correct = writer_chain_rule_surrogate(state, compiled)
    memory_correct = torch.autograd.grad(correct, memory_leaves, retain_graph=True)
    mixed = ({name: (1 - rho) * value + rho * distilled[name] for name, value in compiled.items()}
             if rho else compiled)
    generator = correct if not rho else writer_chain_rule_surrogate(state, mixed)
    generator.backward(inputs=tuple(writer.compiler_parameters()))
    memory_gradients, cursor = [], 0
    for video, gradient in zip(videos, memory_correct, strict=True):
        count = video.shape[0] * video.shape[1]
        if auxiliary is not None:
            gradient = gradient + auxiliary[cursor:cursor + count].reshape_as(video).to(gradient)
        memory_gradients.append(gradient)
        cursor += count
    if auxiliary is not None and cursor != len(auxiliary):
        raise ValueError("auxiliary memory cotangent lost video/token alignment")
    torch.autograd.backward(videos, memory_gradients)
    cotangents = tuple(value.grad for value in (*leaves, *visual_leaves))
    if any(value is None for value in cotangents):
        raise RuntimeError("functional Writer detached a native R/Z leaf")
    return cotangents[:len(leaves)], cotangents[len(leaves):]


class SupervisedEngine:
    def __init__(self, runtime, data, cache, context, config) -> None:
        self.runtime, self.data, self.cache, self.config = runtime, data, cache, config
        self.device, self.step = context.device, 0

    def _time(self, timings, name, start):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timings[name] = time.perf_counter() - start
        return time.perf_counter()

    def _credit(self, state, videos, indices, batch, trace, offset, *, backward):
        writer = self.runtime.state.writer
        with torch.no_grad(), autocast(self.device):
            memory, _, prior = writer.memory(videos, indices)
        aux = self.config["auxiliary"]
        rho = distillation_weight(aux, self.step)
        with autocast(self.device):
            return paired_functional_credit(
                self.runtime.policy, state, self.runtime.lora, batch,
                reader=self.runtime.state.reader, memory=memory, prior=prior,
                seed=trace["policy_rng_seed"], device=self.device,
                random_batch=trace.get("policy_random_batch_size", len(trace["action_demos"])), offset=offset,
                microbatch=min(int(self.config["runtime"]["policy_microbatch"]), len(trace["action_demos"])),
                condition_weight=1.0 / (4 * self.config["data"]["conditions_per_task"]),
                auxiliary_weight=float(aux["weight"]), rho=rho, backward=backward,
            )

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
        credit = self._credit(state, videos, inputs[0], batch, trace, draw["query_offset"], backward=True)
        del batch, raw, state, videos
        start = self._time(timings, "fm_vjp_seconds", start)
        compiled, distilled = credit.pop("lora_cotangent"), credit.pop("distill_cotangent")
        auxiliary = credit.pop("memory_cotangent")
        fm_norm = float(torch.stack([value.norm() for value in compiled.values()]).norm())
        with autocast(self.device):
            cotangents = replay_functional_credit(runtime.state.writer, responses, inputs,
                                                 compiled, distilled, auxiliary, credit["rho"])
        del compiled, distilled, auxiliary, responses, inputs
        start = self._time(timings, "writer_vjp_seconds", start)
        runtime.observer.backward(condition, *cotangents)
        self._time(timings, "observer_vjp_seconds", start)
        return {**credit, "task_weight": .25, "condition_weight": 1. / (4 * self.config["data"]["conditions_per_task"]),
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
            credit = self._credit(state, videos, inputs[0], batch, trace, 0, backward=False)
        for key in ("lora_cotangent", "distill_cotangent", "memory_cotangent"):
            credit.pop(key)
        return {"task": task, "suite": self.data.tasks[task].suite, "video_demos": [demo],
                **credit, "queries": queries, **trace, "gradients": False}
