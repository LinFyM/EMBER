"""Canonical Writer with two teacher Meta stacks and frozen input-embedding cache."""

from __future__ import annotations

from collections import OrderedDict
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
from ember.writer.learning_data import WriterTrainingData
from ember.writer.meta_lora import MetaLoRAStack
from ember.writer.native import NativeCondition, NativeVideoObserver


class WriterState(torch.nn.Module):
    """Checkpoint owner for the complete Writer, its reading module and public probe."""

    def __init__(self, writer: torch.nn.Module, meta: MetaLoRAStack, probe_seed: int, reader: torch.nn.Module | None = None) -> None:
        super().__init__()
        self.writer, self.meta, self.reader = writer, meta, reader
        generator = torch.Generator(device="cpu").manual_seed(probe_seed)
        self.register_buffer("probe", torch.randn(50, 32, generator=generator))


@dataclass
class WriterRuntime:
    policy: torch.nn.Module
    state: WriterState
    observer: NativeVideoObserver
    processor: Pi05LiberoProcessor
    lora: Any
    source: dict[str, Any]


def build_runtime(asset_root: Path, config: Mapping[str, Any], device: torch.device) -> WriterRuntime:
    from ember.writer.video import VideoConditionedWriter, VideoWriterConfig, require_architecture_identity
    from ember.writer.function_reader import LocalActionReader

    require_architecture_identity(config["model"])
    model_config = VideoWriterConfig(**config["model"])

    authorities = load_evaluation_authorities(asset_root / "configs/pi05_target_evaluation_v1.json", asset_root)
    reuse = read_json(asset_root / "configs/pi05_writer_data_v1.json")["authorities"]
    checkpoint = asset_root / reuse["source_checkpoint"]
    source = inspect_source_checkpoint(authorities, checkpoint.parent.parent, checkpoint, evaluation_mode="formal")
    source_config = read_json(asset_root / reuse["source_base_config"])
    policy = load_policy(Path(source["model_path"]), source_config, device)
    lora = derive_pi05_lora_rank(load_pi05_lora_contract(asset_root / reuse["lora_contract"]), rank=16)
    prepare_frozen_writer_policy(policy, lora)
    # Source eval state disables upstream checkpoint closures whose hook scope
    # would otherwise differ during observer replay. Chunking bounds that graph.
    policy.model.gradient_checkpointing_disable()
    expert = policy.model.paligemma_with_expert.gemma_expert.model
    state = WriterState(
        VideoConditionedWriter(lora, model_config),
        MetaLoRAStack(expert.layers, rank=int(config["observer"]["meta_rank"])),
        int(config["observer"]["probe_seed"]),
    )
    # Finish every common module before optional local-head initialization.
    gemma = policy.model.paligemma_with_expert.paligemma.model.language_model
    state.vl_meta = MetaLoRAStack(gemma.layers, rank=int(config["observer"]["vl_meta_rank"]))
    if config["local_action"]["enabled"]:
        state.reader = LocalActionReader(model_config.width, model_config.heads)
    state.to(device)
    tokenizer = asset_root / reuse["tokenizer"]
    observer = NativeVideoObserver(
        policy, state.meta, state.vl_meta, Pi05TeacherPrefixTokenizer(tokenizer, 200, str(device)), state.probe,
        frame_chunk=int(config["observer"]["frame_chunk"]),
        camera_view=config["observer"].get("camera_view", "agentview"),
    )
    stats = read_json(asset_root / reuse["source_normalization"])["stats"]
    processor = Pi05LiberoProcessor(stats, tokenizer, 200, str(device))
    return WriterRuntime(policy, state, observer, processor, lora, source)


class FrozenVideoPrefixCache:
    """Loader cache scoped to one frozen policy/preprocessing runtime.

    Identity keys never enter a learned module. Each entry holds only frozen
    pre-Gemma vision/token embeddings and masks; no learned Z/KV/R/E is cached.
    """

    def __init__(self, observer: NativeVideoObserver, data: WriterTrainingData, byte_limit: int) -> None:
        if observer.camera_view != data.videos.camera_view:
            raise ValueError("prefix cache teacher camera views differ from the observer contract")
        self.observer, self.data = observer, data
        self.byte_limit, self.bytes = int(byte_limit), 0
        self.hits = self.misses = 0
        self.entries: OrderedDict[tuple[int, int], tuple[NativeCondition, int]] = OrderedDict()

    def condition(self, task: int, demos: Sequence[int]) -> NativeCondition:
        if len(demos) not in (1, 2, 4) or len(set(demos)) != len(demos):
            raise ValueError("a condition needs K1/2/4 distinct videos")
        values = []
        for demo in demos:
            key = (task, int(demo))
            if key in self.entries:
                self.hits += 1
                value, size = self.entries.pop(key)
                self.entries[key] = (value, size)
            else:
                self.misses += 1
                frames, indices = self.data.load_videos(task, (demo,))
                value = self.observer.prepare(frames, indices, self.data.tasks[task].authority.language)
                size = sum(chunk.tensor_bytes for chunk in value.videos[0])
                if size <= self.byte_limit:
                    while self.entries and self.bytes + size > self.byte_limit:
                        _, (_, evicted) = self.entries.popitem(last=False)
                        self.bytes -= evicted
                    self.entries[key] = (value, size)
                    self.bytes += size
            values.append(value)
        return NativeCondition(
            tuple(value.videos[0] for value in values), tuple(value.frame_indices[0] for value in values),
            values[0].language_embeddings, values[0].language_mask,
        )
