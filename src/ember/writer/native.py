"""State-free teacher reading with trainable VL/Action Meta and joint Z/R VJP.

Only pre-Gemma embeddings may outlive an optimizer step. Z/KV/R belong to one
condition at the current parameter version and are replayed together before
the optimizer advances. Teacher adapters never enter execution forwards.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Sequence

import torch

from ember.ecp.policy_effects import (
    ExecutionPolicyPrefix,
    prepare_execution_policy_prefix,
    prepare_prefix_features_and_cache,
)
from ember.pi05_processing import Pi05TeacherPrefixTokenizer
from ember.writer.data import teacher_camera_names
from ember.writer.meta_lora import MetaLoRAStack


def autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@dataclass(frozen=True)
class FrozenInputChunk:
    """CPU-resident vision/token embeddings, preceding every learned Meta."""

    embeddings: torch.Tensor
    padding: torch.Tensor
    evidence_mask: torch.Tensor
    task_mask: torch.Tensor

    @property
    def tensor_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in (
            self.embeddings, self.padding, self.evidence_mask, self.task_mask,
        ))

    def on_device(self, device: torch.device) -> ExecutionPolicyPrefix:
        return ExecutionPolicyPrefix(self.embeddings.to(device, non_blocking=True),
                                     self.padding.to(device, non_blocking=True))


@dataclass(frozen=True)
class NativeCondition:
    """One ephemeral condition; no task identity or privileged teacher fields."""

    videos: tuple[tuple[FrozenInputChunk, ...], ...]
    frame_indices: tuple[torch.Tensor, ...]
    language_embeddings: torch.Tensor
    language_mask: torch.Tensor


class NativeVideoObserver:
    """Own execution scope, not the frozen policy's parameter registration."""

    def __init__(
        self, policy: torch.nn.Module, meta: MetaLoRAStack, vl_meta: MetaLoRAStack,
        tokenizer: Pi05TeacherPrefixTokenizer, probe: torch.Tensor,
        *, frame_chunk: int = 4, camera_view: str = "agentview",
    ) -> None:
        if probe.shape != (50, 32) or frame_chunk <= 0:
            raise ValueError("native observer requires one public 50x32 probe")
        if any(parameter.requires_grad for parameter in policy.parameters()):
            raise ValueError("native observer base policy must be frozen")
        self.policy, self.meta, self.vl_meta, self.tokenizer = policy, meta, vl_meta, tokenizer
        self.probe, self.frame_chunk = probe, int(frame_chunk)
        self.device = probe.device
        self.camera_view = camera_view
        self.camera_names = teacher_camera_names(camera_view)
        self.expert = policy.model.paligemma_with_expert.gemma_expert.model
        self.gemma = policy.model.paligemma_with_expert.paligemma.model.language_model
        if len(self.expert.layers) != 18 or len(self.gemma.layers) != 18:
            raise ValueError("native observer requires all 18 VL and Action Expert layers")

    @torch.no_grad()
    def prefix(self, frames: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor,
               task_span: torch.Tensor) -> FrozenInputChunk:
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        dual = len(self.camera_names) == 2
        expected_axes = (2, 3) if dual else (3,)
        if frames.ndim != (5 if dual else 4) or tuple(frames.shape[1:-2]) != expected_axes or frames.shape[0] <= 0:
            raise ValueError("native observer RGB shape must match its declared teacher camera views")
        images = frames.to(self.device, non_blocking=True)
        images = images.float().div(255) if images.dtype == torch.uint8 else images.float()
        batch = {
            OBS_LANGUAGE_TOKENS: tokens.expand(len(frames), -1),
            OBS_LANGUAGE_ATTENTION_MASK: mask.expand(len(frames), -1),
        }
        native_keys = {"agentview": "base_0_rgb", "eye_in_hand": "left_wrist_0_rgb"}
        for camera, pixels in zip(self.camera_names, images.unbind(1) if dual else (images,), strict=True):
            batch[f"observation.images.{native_keys[camera]}"] = pixels
        prefix = prepare_execution_policy_prefix(self.policy, batch)
        evidence_mask = prefix.padding.clone()
        evidence_mask[:, -tokens.shape[1]:] = task_span.expand(len(frames), -1)
        task_mask = torch.zeros_like(prefix.padding)
        task_mask[:, -tokens.shape[1]:] = task_span.expand(len(frames), -1)
        # Remove only columns masked for every frame. Valid-token positions and
        # native causal semantics stay unchanged, without missing-camera work
        # in the language/Action Expert attention stacks.
        keep = prefix.padding.any(dim=0)
        prefix = ExecutionPolicyPrefix(prefix.embeddings[:, keep], prefix.padding[:, keep])
        evidence_mask = evidence_mask[:, keep]
        task_mask = task_mask[:, keep]
        return FrozenInputChunk(prefix.embeddings.cpu(), prefix.padding.cpu(),
                                evidence_mask.cpu(), task_mask.cpu())

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
                self.prefix(video[start:start + self.frame_chunk], tokens, mask, task_span)
                for start in range(0, len(video), self.frame_chunk)
            ))
            positions.append(indices.to(self.device))
        return NativeCondition(tuple(videos), tuple(positions), embeddings, task_span[0])

    def capture(self, chunk: FrozenInputChunk) -> tuple[torch.Tensor, torch.Tensor]:
        """One shared prefix graph supplies both direct Z and R-through-KV paths."""
        prefix = chunk.on_device(self.device)
        with self.vl_meta.installed(self.gemma):
            features, cache = prepare_prefix_features_and_cache(
                self.policy, prefix, track_grad=torch.is_grad_enabled())
        noise = self.probe.expand(len(prefix.padding), -1, -1)
        time = torch.ones(len(prefix.padding), device=self.device)
        captured = []
        def capture_input(module, args):
            captured.append(args[0])
        handle = self.policy.model.action_out_proj.register_forward_pre_hook(capture_input)
        try:
            with self.meta.installed(self.expert), autocast(self.device):
                self.policy.model.denoise_step(prefix.padding, cache, noise, time)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise ValueError("native observer requires exactly one real action output projection")
        responses = captured[0]
        if responses.shape[1:] != (50, 1024) or responses.dtype != torch.float32:
            raise ValueError("native observer lost final normalized FP32 complete horizon states")
        return responses, features[:, chunk.evidence_mask.any(dim=0).to(self.device)]

    @torch.no_grad()
    def read(self, condition: NativeCondition) -> tuple[tuple[torch.Tensor, ...], tuple]:
        """Materialize same-version R/Z once; no learned values enter the cache."""
        responses, visuals, masks, task_masks = [], [], [], []
        for video in condition.videos:
            captured = [self.capture(chunk) for chunk in video]
            responses.append(torch.cat([pair[0] for pair in captured]))
            visuals.append(torch.cat([pair[1] for pair in captured]))
            masks.append(torch.cat([chunk.evidence_mask[:, chunk.evidence_mask.any(0)] for chunk in video]).to(self.device))
            task_masks.append(torch.cat([chunk.task_mask[:, chunk.evidence_mask.any(0)] for chunk in video]).to(self.device))
        inputs = (condition.frame_indices, condition.language_embeddings, condition.language_mask,
                  tuple(visuals), tuple(masks), tuple(task_masks))
        return tuple(responses), inputs

    def backward(self, condition: NativeCondition, response_cotangents: Sequence[torch.Tensor],
                 visual_cotangents: Sequence[torch.Tensor]) -> None:
        if not len(condition.videos) == len(response_cotangents) == len(visual_cotangents):
            raise ValueError("observer VJP lost a video")
        for video, response_gradient, visual_gradient in zip(
                condition.videos, response_cotangents, visual_cotangents, strict=True):
            cursor = 0
            for chunk in video:
                response, visual = self.capture(chunk)
                stop = cursor + len(response)
                gradients = (response_gradient[cursor:stop], visual_gradient[cursor:stop])
                if gradients[0].shape != response.shape or gradients[1].shape != visual.shape:
                    raise ValueError("observer VJP frame chunk changed")
                torch.autograd.backward((response, visual),
                                        (gradients[0].to(response.dtype), gradients[1].to(visual.dtype)))
                cursor = stop
            if cursor != len(response_gradient) or cursor != len(visual_gradient):
                raise ValueError("observer VJP omitted frames")
