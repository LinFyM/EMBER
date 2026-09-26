"""Video-teaching Writer and action-hidden raw input preparation."""
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
from ember.writer.video_program import VIDEO_READ_MODES


MODEL_SCHEMA = "ember_video_teaching_writer_v1"
ARCHITECTURE = "video_core_recurrent_procedure_complete_lora"
MODEL_DEFAULTS = {
    "schema": MODEL_SCHEMA, "architecture": ARCHITECTURE,
    "image_width": 2048, "expert_width": 1024, "program_width": 256,
    "text_meta_lora_rank": 4, "vl_meta_lora_rank": 4, "action_meta_lora_rank": 4,
    "patch_grounding_heads": 8, "max_frames_per_encoder_call": 4,
    "action_horizon": 50, "padded_action_dim": 32,
    "semantic_core_heads": 8, "semantic_core_blocks": 2, "frame_attention_initial_lambda": .05,
    "procedure_heads": 8, "procedure_blocks": 2, "fusion_heads": 8,
    "factor_hidden_width": 216, "initialization_seed": 7, "activation_checkpointing": True,
    "camera_view": "agentview", "horizon_read": "repeated_full",
}


def require_architecture_identity(model: Mapping[str, Any]) -> None:
    variable = {"max_frames_per_encoder_call", "activation_checkpointing", "camera_view"}
    if (set(model) != set(MODEL_DEFAULTS)
            or any(model[key] != value for key, value in MODEL_DEFAULTS.items() if key not in variable)
            or (model["camera_view"], model["horizon_read"]) not in VIDEO_READ_MODES
            or type(model["max_frames_per_encoder_call"]) is not int
            or model["max_frames_per_encoder_call"] <= 0
            or type(model["activation_checkpointing"]) is not bool):
        raise ValueError("canonical video-teaching Writer architecture changed")


def autocast(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


class WriterState(torch.nn.Module):
    """One checkpoint state owner for a Writer or direct shared LoRA parameters."""

    def __init__(self, writer: torch.nn.Module) -> None:
        super().__init__()
        self.writer = writer
        self._empty_module = torch.nn.Module()

    def _encoder(self):
        return getattr(self.writer, "semantic_encoder", None)

    @property
    def meta(self):
        return getattr(self._encoder(), "action_meta_lora", self._empty_module)

    @property
    def vl_meta(self):
        return getattr(self._encoder(), "vl_meta_lora", self._empty_module)

    @property
    def text_meta(self):
        return getattr(self._encoder(), "text_meta_lora", self._empty_module)

    @property
    def probe(self):
        return getattr(self._encoder(), "fixed_suffix_noise", None)


@dataclass
class WriterRuntime:
    policy: torch.nn.Module
    state: WriterState
    tokenizer: Pi05TeacherPrefixTokenizer
    processor: Pi05LiberoProcessor
    lora: Any
    source: dict[str, Any]
    device: torch.device
    parameterization: str = "video_writer"
    camera_view: str = "agentview"

    @property
    def uses_video(self) -> bool:
        return self.parameterization == "video_writer"

    def prepare(self, frames: Sequence[torch.Tensor], indices: Sequence[torch.Tensor], language: str) -> tuple:
        if not self.uses_video:
            raise ValueError("non-video parameterizations cannot prepare RGB conditions")
        if len(frames) != 1 or len(indices) != 1 or len(frames[0]) != len(indices[0]):
            raise ValueError("the canonical Writer requires one complete video in its declared camera mode")
        tokens, mask, span = self.tokenizer([language])
        pixels = frames[0].to(self.device, non_blocking=True)
        positions = indices[0].to(self.device, non_blocking=True)
        offsets = torch.tensor([0, len(pixels)], dtype=torch.long, device=self.device)
        return pixels, positions, offsets, tokens, mask, span

    def prepare_language(self, language: str) -> tuple:
        if self.parameterization != "language_writer":
            raise ValueError("only the language Writer accepts a text condition")
        tokens, mask, span = self.tokenizer([language])
        return tokens, mask, span

    def compile(self, condition: tuple, *, frame_parallel_group=None) -> dict[str, torch.Tensor]:
        with autocast(self.device):
            if self.parameterization == "direct_lora":
                return self.state.writer()
            if self.parameterization == "language_writer":
                return self.state.writer.compile_language_task(self.policy, *condition)[0]
            if self.parameterization != "video_writer":
                raise ValueError("unknown Writer parameterization")
            return self.state.writer(*condition, policy=self.policy, frame_parallel_group=frame_parallel_group)


def build_runtime(asset_root: Path, config: Mapping[str, Any], device: torch.device) -> WriterRuntime:
    from ember.writer.model import (CompleteLoRAWriter, DirectLoRAParameters,
                                    LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS, build_lora_tensor_specs)

    parameterization = config.get("experiment", {}).get("parameterization", "video_writer")
    if parameterization not in {"direct_lora", "language_writer", "video_writer"}:
        raise ValueError("unknown conditional compilation parameterization")
    if parameterization != "direct_lora":
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
    if parameterization == "direct_lora":
        writer = DirectLoRAParameters(template)
    else:
        bridge = policy.model.paligemma_with_expert
        model = {key: config["model"][key] for key in LANGUAGE_AXIAL_WRITER_CONSTRUCTOR_KEYS}
        model["max_frames_per_encoder_call"] = int(config["observer"]["frame_chunk"])
        from ember.writer.language_content_contract import EXPERIMENT as LANGUAGE_CONTENT_EXPERIMENT

        model["language_content_path"] = (
            config.get("experiment", {}).get("kind") == LANGUAGE_CONTENT_EXPERIMENT
            and config["experiment"]["language_content_path"] is True)
        from ember.writer.learned_initial_content_contract import STUDY as INITIAL_CONTENT_STUDY

        model["initial_content_only"] = (
            config.get("experiment", {}).get("kind") == INITIAL_CONTENT_STUDY
            and config["experiment"].get("initial_content_only") is True)
        writer = CompleteLoRAWriter(
            build_lora_tensor_specs(template), template_state=template,
            paligemma_model=bridge.paligemma.model.language_model,
            expert_model=bridge.gemma_expert.model, **model,
        )
        writer.semantic_core.capture_forward_summary = (
            config.get("experiment", {}).get("kind") == LANGUAGE_CONTENT_EXPERIMENT)
        if parameterization == "language_writer":
            writer.set_language_only_trainable()
    state = WriterState(writer).to(device)
    tokenizer = asset_root / reuse["tokenizer"]
    stats = read_json(asset_root / reuse["source_normalization"])["stats"]
    return WriterRuntime(policy, state, Pi05TeacherPrefixTokenizer(tokenizer, 200, str(device)),
                         Pi05LiberoProcessor(stats, tokenizer, 200, str(device)), lora, source, device,
                         parameterization, config["model"].get("camera_view", "agentview"))


class VideoConditionCache:
    """Bounded CPU cache of raw RGB and frame positions; no adapted activations."""

    def __init__(self, runtime: WriterRuntime, data, byte_limit: int) -> None:
        if (runtime.uses_video and data.videos.camera_view != runtime.camera_view) or byte_limit <= 0:
            raise ValueError("raw input cache requires the Writer camera mode and a positive byte cap")
        self.runtime, self.data, self.byte_limit = runtime, data, int(byte_limit)
        self.bytes = self.hits = self.misses = 0
        self.entries = OrderedDict()

    def condition(self, task: int, demos: Sequence[int]) -> tuple:
        if len(demos) != 1:
            raise ValueError("the trained Writer supports K1")
        if self.runtime.parameterization == "direct_lora":
            self.misses += 1
            return None
        if self.runtime.parameterization == "language_writer":
            self.misses += 1
            return self.runtime.prepare_language(self.data.tasks[task].authority.language)
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
