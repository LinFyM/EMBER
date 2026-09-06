"""Canonical model and bounded frozen-prefix lifetime for the layered Writer."""

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
from ember.writer.learning_data import JointTrainingData
from ember.writer.meta_lora import MetaLoRAStack
from ember.writer.native import NativeCondition, NativeVideoObserver


class JointWriterState(torch.nn.Module):
    """Checkpoint owner for Writer, reading-only Meta, and the public probe."""

    def __init__(self, writer: torch.nn.Module, meta: MetaLoRAStack, probe_seed: int) -> None:
        super().__init__()
        self.writer, self.meta = writer, meta
        generator = torch.Generator(device="cpu").manual_seed(probe_seed)
        self.register_buffer("probe", torch.randn(50, 32, generator=generator))


@dataclass
class JointRuntime:
    policy: torch.nn.Module
    state: JointWriterState
    observer: NativeVideoObserver
    processor: Pi05LiberoProcessor
    lora: Any
    source: dict[str, Any]


def build_joint_runtime(asset_root: Path, config: Mapping[str, Any], device: torch.device) -> JointRuntime:
    from ember.writer.layered import LayeredRelationWriter, LayeredWriterConfig

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
    state = JointWriterState(
        LayeredRelationWriter(lora, LayeredWriterConfig(**config["model"])),
        MetaLoRAStack(expert.layers, rank=int(config["observer"]["meta_rank"])),
        int(config["observer"]["probe_seed"]),
    ).to(device)
    tokenizer = asset_root / reuse["tokenizer"]
    observer = NativeVideoObserver(
        policy, state.meta, Pi05TeacherPrefixTokenizer(tokenizer, 200, str(device)), state.probe,
        frame_chunk=int(config["observer"]["frame_chunk"]),
    )
    stats = read_json(asset_root / reuse["source_normalization"])["stats"]
    processor = Pi05LiberoProcessor(stats, tokenizer, 200, str(device))
    return JointRuntime(policy, state, observer, processor, lora, source)


class FrozenVideoPrefixCache:
    """Loader cache scoped to one frozen policy/preprocessing runtime.

    Identity keys never enter a learned module. Each entry holds only frozen
    prefix KV and exact-language embeddings; no R/U/E/generated LoRA is cached.
    """

    def __init__(self, observer: NativeVideoObserver, data: JointTrainingData, byte_limit: int) -> None:
        self.observer, self.data = observer, data
        self.byte_limit, self.bytes = int(byte_limit), 0
        self.entries: OrderedDict[tuple[int, int], tuple[NativeCondition, int]] = OrderedDict()

    def condition(self, task: int, demos: Sequence[int]) -> NativeCondition:
        if len(demos) not in (1, 2, 4) or len(set(demos)) != len(demos):
            raise ValueError("a condition needs K1/2/4 distinct videos")
        values = []
        for demo in demos:
            key = (task, int(demo))
            if key in self.entries:
                value, size = self.entries.pop(key)
                self.entries[key] = (value, size)
            else:
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
