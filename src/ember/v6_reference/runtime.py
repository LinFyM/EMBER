"""Fresh historical v6 runtime and complete-graph FM cotangent replay.

No Text/VL/Action Meta activation or prefix KV is cached across updates. The
historical encoder owns its original checkpointed frame chunks and Meta hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Sequence
import time

import torch

from ember.pi05_eval_contract import inspect_source_checkpoint, load_evaluation_authorities
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json
from ember.pi05_source_setup import load_policy
from ember.v6_reference.contract import validate_config
from ember.v6_reference.model import CompleteLoRAWriter, build_lora_tensor_specs
from ember.writer.functional import (
    INDEPENDENT_BETA_TIME_SAMPLING_SCHEME, INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
    functional_lora_loss_gradient, functional_lora_loss_value, prepare_frozen_writer_policy,
    writer_chain_rule_surrogate,
)
from ember.writer.native import autocast


class V6State(torch.nn.Module):
    """One checkpoint owner; Meta modules occur exactly once in its state dict."""

    def __init__(self, writer: CompleteLoRAWriter) -> None:
        super().__init__()
        self.writer = writer

    def meta_parameters(self):
        encoder = self.writer.semantic_encoder
        return chain.from_iterable(module.parameters() for module in (
            encoder.text_meta_lora, encoder.vl_meta_lora, encoder.action_meta_lora,
        ))

    @property
    def probe(self):
        return self.writer.semantic_encoder.fixed_suffix_noise


@dataclass
class V6Runtime:
    policy: torch.nn.Module
    state: V6State
    processor: Pi05LiberoProcessor
    tokenizer: Pi05TeacherPrefixTokenizer
    lora: Any
    source: dict[str, Any]
    device: torch.device

    def pack(self, frames: Sequence[torch.Tensor], indices: Sequence[torch.Tensor], language: str):
        if len(frames) != 1 or len(indices) != 1:
            raise ValueError("the original v6 reference accepts exactly one ordered teaching video")
        video, positions = frames[0], indices[0]
        if not len(video) or positions.shape != (len(video),) or not bool((positions[1:] > positions[:-1]).all()):
            raise ValueError("v6 teaching frames must preserve their real stride5 order")
        tokens, mask, span = self.tokenizer([language])
        return (video.to(self.device), positions.to(self.device),
                torch.tensor([0, len(video)], dtype=torch.long, device=self.device), tokens, mask, span)

    def generate(self, frames, indices, language):
        packed = self.pack(frames, indices, language)
        with autocast(self.device):
            return self.state.writer(*packed, policy=self.policy)


def build_runtime(asset_root: Path, config, device: torch.device) -> V6Runtime:
    validate_config(config)
    authorities = load_evaluation_authorities(asset_root / "configs/pi05_target_evaluation_v1.json", asset_root)
    reuse = read_json(asset_root / "configs/pi05_writer_data_v1.json")["authorities"]
    checkpoint = asset_root / reuse["source_checkpoint"]
    source = inspect_source_checkpoint(authorities, checkpoint.parent.parent, checkpoint, evaluation_mode="formal")
    policy = load_policy(Path(source["model_path"]), read_json(asset_root / reuse["source_base_config"]), device)
    lora = load_pi05_lora_contract(asset_root / reuse["lora_contract"])
    if lora.rank != 16 or lora.alpha != 16 or len(lora.targets) != 38:
        raise ValueError("historical v6 requires the same complete rank16 execution adapter")
    template = prepare_frozen_writer_policy(policy, lora)
    policy.model.gradient_checkpointing_disable()
    policy.config.gradient_checkpointing = False
    bridge = policy.model.paligemma_with_expert
    writer = CompleteLoRAWriter(
        build_lora_tensor_specs(template), template_state=template,
        paligemma_model=bridge.paligemma.model.language_model,
        expert_model=bridge.gemma_expert.model, **config["model"],
    )
    state = V6State(writer).to(device)
    tokenizer_path = asset_root / reuse["tokenizer"]
    tokenizer = Pi05TeacherPrefixTokenizer(tokenizer_path, 200, str(device))
    stats = read_json(asset_root / reuse["source_normalization"])["stats"]
    processor = Pi05LiberoProcessor(stats, tokenizer_path, 200, str(device))
    return V6Runtime(policy, state, processor, tokenizer, lora, source, device)


class V6SupervisedEngine:
    """Current logical FM protocol with an uncached, unmodified v6 model graph."""

    def __init__(self, runtime, data, context, config) -> None:
        self.runtime, self.data, self.config = runtime, data, config
        self.device = context.device

    def _time(self, timings, name, start):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        timings[name] = time.perf_counter() - start
        return time.perf_counter()

    def _flow_arguments(self, trace):
        return dict(
            policy_rng_seed=trace["policy_rng_seed"], policy_rng_device=self.device,
            flow_time_sampling_scheme=INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
            flow_noise_sampling_scheme=INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
            policy_microbatch_size=int(self.config["runtime"]["policy_microbatch"]),
            collect_policy_details=False,
        )

    def backward(self, draw) -> dict:
        runtime, timings = self.runtime, {}
        task, demos = draw["task"], draw["video_demos"]
        start = time.perf_counter()
        frames, indices = self.data.load_videos(task, demos)
        packed = runtime.pack(frames, indices, self.data.tasks[task].authority.language)
        start = self._time(timings, "condition_seconds", start)
        with torch.no_grad(), autocast(self.device):
            state = runtime.state.writer(*packed, policy=runtime.policy)
        start = self._time(timings, "writer_forward_seconds", start)
        raw, trace = self.data.action_batch(task, draw["occurrence"], demos, query_seed=draw["query_seed"])
        batch = runtime.processor.training_batch(raw)
        with autocast(self.device):
            loss, _, gradients = functional_lora_loss_gradient(
                runtime.policy, state, runtime.lora, batch=batch, **self._flow_arguments(trace),
            )
        gradients = {name: value.float().mul_(0.25) for name, value in gradients.items()}
        norm = float(torch.stack([value.norm() for value in gradients.values()]).norm())
        del state, batch, raw
        start = self._time(timings, "fm_vjp_seconds", start)
        with autocast(self.device):
            replay = runtime.state.writer(*packed, policy=runtime.policy)
            surrogate = writer_chain_rule_surrogate(replay, gradients)
        surrogate.backward()
        self._time(timings, "complete_writer_vjp_seconds", start)
        return {"flow_loss": float(loss), "task_weight": 0.25, "normalizer": 1.0,
                "fm_lora_gradient_norm": norm, "queries": len(trace["action_demos"]), **trace, **timings,
                "prefix_cache_bytes": 0, "policy_microbatch": int(self.config["runtime"]["policy_microbatch"])}

    @torch.no_grad()
    def validate(self, task: int, demo: int, *, seed: int, queries: int) -> dict:
        frames, indices = self.data.load_videos(task, (demo,))
        state = self.runtime.generate(frames, indices, self.data.tasks[task].authority.language)
        raw, trace = self.data.diagnostic_batch(task, seed=seed, count=queries)
        batch = self.runtime.processor.training_batch(raw)
        with autocast(self.device):
            loss, _ = functional_lora_loss_value(
                self.runtime.policy, state, self.runtime.lora, batch=batch, **self._flow_arguments(trace),
            )
        return {"task": task, "suite": self.data.tasks[task].suite, "video_demos": [demo],
                "flow_loss": float(loss), "queries": queries, **trace, "gradients": False}
