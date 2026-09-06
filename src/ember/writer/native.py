"""State-free native video reading with trainable Action Meta and frozen KV.

Only frozen prefix tensors may outlive an optimizer step. Responses belong to
one condition at the current parameter version; their cotangents are replayed
through independently chunked native forwards before the optimizer advances.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import time
from typing import Any, Sequence

import torch

from ember.ecp.observer import ActionLayerStateCapture
from ember.ecp.policy_effects import (
    ExecutionPolicyPrefix,
    prepare_execution_policy_prefix,
    prepare_prefix_kv_cache,
)
from ember.pi05_processing import Pi05TeacherPrefixTokenizer
from ember.writer.meta_lora import MetaLoRAStack


def autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@dataclass(frozen=True)
class FrozenPrefixChunk:
    """CPU-resident native prefix KV, independent of all learned Writer state."""

    padding: torch.Tensor
    layers: tuple[tuple[torch.Tensor, torch.Tensor, Any], ...]

    @property
    def tensor_bytes(self) -> int:
        return self.padding.numel() * self.padding.element_size() + sum(
            value.numel() * value.element_size()
            for keys, values, _ in self.layers for value in (keys, values)
        )

    def on_device(self, device: torch.device):
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache(tuple(
            (keys.to(device, non_blocking=True), values.to(device, non_blocking=True), window)
            for keys, values, window in self.layers
        ))
        return self.padding.to(device, non_blocking=True), cache


@dataclass(frozen=True)
class NativeCondition:
    """One ephemeral condition; no task identity or privileged teacher fields."""

    videos: tuple[tuple[FrozenPrefixChunk, ...], ...]
    frame_indices: tuple[torch.Tensor, ...]
    language_embeddings: torch.Tensor
    language_mask: torch.Tensor


class NativeVideoObserver:
    """Own execution scope, not the frozen policy's parameter registration."""

    def __init__(
        self, policy: torch.nn.Module, meta: MetaLoRAStack,
        tokenizer: Pi05TeacherPrefixTokenizer, probe: torch.Tensor,
        *, frame_chunk: int = 4,
    ) -> None:
        if probe.shape != (50, 32) or frame_chunk <= 0:
            raise ValueError("native observer requires one public 50x32 probe")
        if any(parameter.requires_grad for parameter in policy.parameters()):
            raise ValueError("native observer base policy must be frozen")
        self.policy, self.meta, self.tokenizer = policy, meta, tokenizer
        self.probe, self.frame_chunk = probe, int(frame_chunk)
        self.device = probe.device
        self.expert = policy.model.paligemma_with_expert.gemma_expert.model
        if len(self.expert.layers) != 18:
            raise ValueError("native observer requires all 18 Action Expert layers")

    @torch.no_grad()
    def prefix(self, frames: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor) -> FrozenPrefixChunk:
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        if frames.ndim != 4 or frames.shape[1] != 3 or frames.shape[0] <= 0:
            raise ValueError("native observer requires real RGB frame batches")
        images = frames.to(self.device, non_blocking=True)
        images = images.float().div(255) if images.dtype == torch.uint8 else images.float()
        batch = {
            "observation.images.base_0_rgb": images,
            OBS_LANGUAGE_TOKENS: tokens.expand(len(frames), -1),
            OBS_LANGUAGE_ATTENTION_MASK: mask.expand(len(frames), -1),
        }
        prefix = prepare_execution_policy_prefix(self.policy, batch)
        # Remove only columns masked for every frame. Valid-token positions and
        # native causal semantics stay unchanged, without missing-camera work
        # in the language/Action Expert attention stacks.
        keep = prefix.padding.any(dim=0)
        prefix = ExecutionPolicyPrefix(prefix.embeddings[:, keep], prefix.padding[:, keep])
        cache = prepare_prefix_kv_cache(self.policy, prefix)
        return FrozenPrefixChunk(
            prefix.padding.cpu(),
            tuple((keys.detach().cpu(), values.detach().cpu(), window) for keys, values, window in cache),
        )

    @torch.no_grad()
    def prepare(
        self, frames: Sequence[torch.Tensor], frame_indices: Sequence[torch.Tensor], language: str,
    ) -> NativeCondition:
        if len(frames) not in (1, 2, 4) or len(frames) != len(frame_indices):
            raise ValueError("native condition requires K1/2/4 separately ordered videos")
        tokens, mask, task_span = self.tokenizer([language])
        bridge = self.policy.model.paligemma_with_expert
        embeddings = bridge.embed_language_tokens(tokens).detach()[0]
        videos = []
        positions = []
        for video, indices in zip(frames, frame_indices, strict=True):
            if indices.shape != (len(video),) or len(video) == 0 or not bool((indices[1:] > indices[:-1]).all()):
                raise ValueError("native video positions must preserve real frame order")
            videos.append(tuple(
                self.prefix(video[start:start + self.frame_chunk], tokens, mask)
                for start in range(0, len(video), self.frame_chunk)
            ))
            positions.append(indices.to(self.device))
        return NativeCondition(tuple(videos), tuple(positions), embeddings, task_span[0])

    def capture(self, chunk: FrozenPrefixChunk) -> torch.Tensor:
        padding, cache = chunk.on_device(self.device)
        noise = self.probe.expand(len(padding), -1, -1)
        time = torch.ones(len(padding), device=self.device)
        with self.meta.installed(self.expert), ActionLayerStateCapture(self.expert, detach=False) as capture:
            with autocast(self.device):
                self.policy.model.denoise_step(padding, cache, noise, time)
                responses = capture.stacked()[:, 1:]
        if responses.shape[1:] != (18, 50, 1024):
            raise ValueError("native observer lost post-layer or complete horizon states")
        return responses

    @torch.no_grad()
    def responses(self, condition: NativeCondition) -> tuple[torch.Tensor, ...]:
        return tuple(torch.cat([self.capture(chunk) for chunk in video]) for video in condition.videos)

    def backward(self, condition: NativeCondition, cotangents: Sequence[torch.Tensor]) -> None:
        if len(condition.videos) != len(cotangents):
            raise ValueError("observer VJP lost a video")
        for video, gradient in zip(condition.videos, cotangents, strict=True):
            cursor = 0
            for chunk in video:
                response = self.capture(chunk)
                stop = cursor + len(response)
                if gradient[cursor:stop].shape != response.shape:
                    raise ValueError("observer VJP frame chunk changed")
                torch.autograd.backward(response, gradient[cursor:stop].to(response.dtype))
                cursor = stop
            if cursor != len(gradient):
                raise ValueError("observer VJP omitted frames")


def joint_functional_backward(
    writer: torch.nn.Module, observer: NativeVideoObserver, condition: NativeCondition,
    *, policy: torch.nn.Module, contract: Any, batch: dict[str, Any],
    task_weight: float, policy_rng_seed: int, policy_microbatch_size: int,
) -> dict[str, float]:
    """One task's positive flow VJP, Writer/R-leaf VJP, and native Meta VJP."""

    from ember.writer.functional import (
        INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
        INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
        functional_lora_loss_gradient,
        writer_chain_rule_surrogate,
    )

    if not 0 < task_weight <= 1:
        raise ValueError("joint functional objective requires explicit global task mass")
    timing = {}
    def boundary(label, started):
        if observer.device.type == "cuda":
            torch.cuda.synchronize(observer.device)
        timing[label] = time.perf_counter() - started
        return time.perf_counter()
    tick = time.perf_counter()
    responses = observer.responses(condition)
    tick = boundary("observer_forward_seconds", tick)
    inputs = (condition.frame_indices, condition.language_embeddings, condition.language_mask)
    with torch.no_grad(), autocast(observer.device):
        state = writer(responses, *inputs)
    tick = boundary("writer_forward_seconds", tick)
    with autocast(observer.device):
        loss, _, gradients = functional_lora_loss_gradient(
            policy, state, contract, batch=batch, policy_rng_seed=policy_rng_seed,
            policy_rng_device=observer.device,
            flow_time_sampling_scheme=INDEPENDENT_BETA_TIME_SAMPLING_SCHEME,
            flow_noise_sampling_scheme=INDEPENDENT_GAUSSIAN_NOISE_SAMPLING_SCHEME,
            policy_microbatch_size=policy_microbatch_size, collect_policy_details=False,
        )
    tick = boundary("policy_vjp_seconds", tick)
    del state
    leaves = tuple(value.detach().requires_grad_(True) for value in responses)
    with autocast(observer.device):
        replay = writer(leaves, *inputs)
        surrogate = writer_chain_rule_surrogate(replay, gradients) * task_weight
    surrogate.backward()
    tick = boundary("writer_vjp_seconds", tick)
    cotangents = tuple(value.grad for value in leaves)
    if any(value is None for value in cotangents):
        raise RuntimeError("Writer detached a native R leaf")
    del replay, surrogate, gradients, leaves, responses
    observer.backward(condition, cotangents)
    boundary("observer_vjp_seconds", tick)
    return {"flow_loss": float(loss), "task_weight": task_weight, "normalizer": 1.0, **timing}
