"""Reuse the frozen source, legal native teaching read and official FM owners."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from ember.lora import copy_task_lora_state_, validate_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json
from ember.pi05_source_setup import load_policy
from ember.writer.functional import prepare_frozen_writer_policy
from .specification import MT_WEIGHTS
from .model import NativeVideoControl


@dataclass
class Runtime:
    policy: torch.nn.Module
    controller: NativeVideoControl
    tokenizer: Pi05TeacherPrefixTokenizer
    processor: Pi05LiberoProcessor
    lora: object
    source: dict
    device: torch.device


def frozen_git() -> dict:
    from ember.pi05_eval_contract import git_state, git_state_is_clean_pushed_or_frozen_authority

    repo = Path(__file__).resolve().parents[3]
    state = git_state(repo)
    if state["branch"] or not git_state_is_clean_pushed_or_frozen_authority(state):
        raise ValueError("native-video GPU work requires clean pushed detached source")
    return {**state, "pushed_ref": "origin/main"}


def build_runtime(asset_root: Path, spec: dict, device: torch.device, arm: str) -> Runtime:
    from ember.pi05_eval_contract import load_evaluation_authorities, inspect_source_checkpoint

    source_config = spec["source"]
    authorities = load_evaluation_authorities(asset_root / source_config["evaluation_config"], asset_root)
    checkpoint = asset_root / source_config["checkpoint"]
    source = inspect_source_checkpoint(authorities, checkpoint.parent.parent, checkpoint,
                                       evaluation_mode="formal")
    policy = load_policy(Path(source["model_path"]), authorities.source_base_config, device).to(device)
    lora = load_pi05_lora_contract(asset_root / source_config["lora_contract"])
    prepare_frozen_writer_policy(policy, lora)
    policy.model.gradient_checkpointing_disable()
    template = {name: value.float() for name, value in load_file(str(MT_WEIGHTS), device="cpu").items()}
    validate_lora_state(template, lora)
    copy_task_lora_state_(policy, template, lora)
    controller = NativeVideoControl(lora, template, arm=arm).to(device).train()
    if any(p.requires_grad for p in policy.parameters()) or any(p.dtype != torch.float32 for p in controller.parameters()):
        raise ValueError("source freeze or FP32 learning parameter contract changed")
    tokenizer = asset_root / source_config["tokenizer"]
    stats = read_json(asset_root / source_config["normalization"])["stats"]
    return Runtime(policy, controller, Pi05TeacherPrefixTokenizer(tokenizer, 200, str(device)),
                   Pi05LiberoProcessor(stats, tokenizer, 200, str(device)), lora, source, device)
