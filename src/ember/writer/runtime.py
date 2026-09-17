"""Unified native video Writer and raw input preparation."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ember.pi05_eval_contract import inspect_source_checkpoint, load_evaluation_authorities
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json
from ember.pi05_source_setup import load_policy
from ember.writer.functional import prepare_frozen_writer_policy


MODEL_SCHEMA = "ember_unified_native_writer_v1"
ARCHITECTURE = "joint_native_video_continuous_parameter_lora"
MODEL_DEFAULTS = {
    "schema": MODEL_SCHEMA, "architecture": ARCHITECTURE,
    "image_width": 2048, "expert_width": 1024, "program_width": 256,
    "text_meta_lora_rank": 4, "vl_meta_lora_rank": 4, "action_meta_lora_rank": 4,
    "patch_grounding_heads": 8, "max_frames_per_encoder_call": 4,
    "action_horizon": 50, "padded_action_dim": 32,
    "native_split_layer": 9, "joint_heads": 8, "joint_blocks": 2,
    "decoder_heads": 8, "decoder_blocks": 2,
    "factor_hidden_width": 216, "initialization_seed": 7, "activation_checkpointing": True,
    "camera_view": "dual",
}


def require_architecture_identity(model: Mapping[str, Any]) -> None:
    variable = {"max_frames_per_encoder_call", "activation_checkpointing"}
    if (set(model) != set(MODEL_DEFAULTS)
            or any(model[key] != value for key, value in MODEL_DEFAULTS.items() if key not in variable)
            or type(model["max_frames_per_encoder_call"]) is not int
            or model["max_frames_per_encoder_call"] <= 0
            or type(model["activation_checkpointing"]) is not bool):
        raise ValueError("canonical unified native Writer architecture changed")


def autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


class WriterState(torch.nn.Module):
    """One state owner: complete Writer including its three reading Meta stacks."""

    def __init__(self, writer: torch.nn.Module) -> None:
        super().__init__()
        self.writer = writer

    @property
    def meta(self):
        return self.writer.semantic_encoder.action_meta_lora

    @property
    def vl_meta(self):
        return self.writer.semantic_encoder.vl_meta_lora

    @property
    def text_meta(self):
        return self.writer.semantic_encoder.text_meta_lora

    @property
    def probe(self):
        return self.writer.semantic_encoder.fixed_suffix_noise


@dataclass
class WriterRuntime:
    policy: torch.nn.Module
    state: WriterState
    tokenizer: Pi05TeacherPrefixTokenizer
    processor: Pi05LiberoProcessor
    lora: Any
    source: dict[str, Any]
    device: torch.device

    def prepare(self, frames: Sequence[torch.Tensor], indices: Sequence[torch.Tensor], language: str) -> tuple:
        if len(frames) != 1 or len(indices) != 1 or len(frames[0]) != len(indices[0]):
            raise ValueError("the canonical Writer requires one complete video in its declared camera mode")
        tokens, mask, span = self.tokenizer([language])
        pixels = frames[0].to(self.device, non_blocking=True)
        positions = indices[0].to(self.device, non_blocking=True)
        offsets = torch.tensor([0, len(pixels)], dtype=torch.long, device=self.device)
        return pixels, positions, offsets, tokens, mask, span

    def compile(self, condition: tuple, *, frame_parallel_group=None) -> dict[str, torch.Tensor]:
        with autocast(self.device):
            return self.state.writer(*condition, policy=self.policy, frame_parallel_group=frame_parallel_group)


def build_runtime(asset_root: Path, config: Mapping[str, Any], device: torch.device) -> WriterRuntime:
    from ember.writer.model import CompleteLoRAWriter, WRITER_CONSTRUCTOR_KEYS, build_lora_tensor_specs

    require_architecture_identity(config["model"])
    if config["observer"]["camera_view"] != config["model"]["camera_view"]:
        raise ValueError("observer camera mode differs from the Writer architecture")
    source_config = config["source"]
    authorities = load_evaluation_authorities(asset_root / source_config["evaluation_config"], asset_root)
    reuse = read_json(asset_root / "configs/pi05_writer_data_v1.json")["authorities"]
    checkpoint = asset_root / source_config["checkpoint"]
    source = inspect_source_checkpoint(authorities, checkpoint.parent.parent, checkpoint, evaluation_mode="formal")
    policy = load_policy(Path(source["model_path"]), authorities.source_base_config, device)
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(asset_root / reuse["lora_contract"]), rank=16)
    template = prepare_frozen_writer_policy(policy, lora)
    # Each encoder checkpoint reinstalls the correct Meta hooks during replay.
    # Upstream layer checkpoint closures outlive those scopes and must be disabled.
    policy.model.gradient_checkpointing_disable()
    bridge = policy.model.paligemma_with_expert
    model = {key: config["model"][key] for key in WRITER_CONSTRUCTOR_KEYS}
    model["max_frames_per_encoder_call"] = int(config["observer"]["frame_chunk"])
    writer = CompleteLoRAWriter(
        build_lora_tensor_specs(template), template_state=template,
        paligemma_model=bridge.paligemma.model.language_model,
        expert_model=bridge.gemma_expert.model, **model,
    )
    state = WriterState(writer).to(device)
    tokenizer = asset_root / reuse["tokenizer"]
    stats = read_json(asset_root / reuse["source_normalization"])["stats"]
    return WriterRuntime(policy, state, Pi05TeacherPrefixTokenizer(tokenizer, 200, str(device)),
                         Pi05LiberoProcessor(stats, tokenizer, 200, str(device)), lora, source, device)


class VideoConditionCache:
    """Bounded CPU cache of raw RGB and frame positions; no adapted activations."""

    def __init__(self, runtime: WriterRuntime, data, byte_limit: int) -> None:
        if data.videos.camera_view != runtime.state.writer.camera_view or byte_limit <= 0:
            raise ValueError("raw input cache requires the Writer camera mode and a positive byte cap")
        self.runtime, self.data, self.byte_limit = runtime, data, int(byte_limit)
        self.bytes = self.hits = self.misses = 0
        self.entries = OrderedDict()

    def condition(self, task: int, demos: Sequence[int]) -> tuple:
        if len(demos) != 1:
            raise ValueError("the trained Writer supports K1")
        key = (task, int(demos[0]))
        if key in self.entries:
            self.hits += 1
            frames, positions, size = self.entries.pop(key)
            self.entries[key] = frames, positions, size
        else:
            self.misses += 1
            frames, positions = self.data.load_videos(task, demos)
            size = sum(value.numel() * value.element_size() for value in (*frames, *positions))
            if size <= self.byte_limit:
                while self.entries and self.bytes + size > self.byte_limit:
                    self.bytes -= self.entries.popitem(last=False)[1][2]
                self.entries[key] = frames, positions, size
                self.bytes += size
        return self.runtime.prepare(frames, positions, self.data.tasks[task].authority.language)
