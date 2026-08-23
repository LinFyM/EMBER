"""Load the frozen native plus Action-Meta Stage 0 observer authority."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from ember.ecp.checkpoint import checkpoint_macro
from ember.ecp.contracts import TargetOwner
from ember.ecp.stage0 import ECPStage0Model
from ember.ecp.stage0_training import build_stage0_model
from ember.pi05_source_checkpoint import read_json
from ember.writer.meta_lora import MetaLoRAStack


@dataclass(frozen=True)
class FrozenObserverAuthority:
    model: ECPStage0Model
    action_meta: MetaLoRAStack
    native_checkpoint: Path
    action_meta_checkpoint: Path


def load_frozen_native_observer(
    *,
    stage0_config: dict,
    owners: tuple[TargetOwner, ...],
    native_checkpoint: Path,
    device: torch.device,
    max_frames_per_call: int | None = None,
) -> ECPStage0Model:
    """Load the frozen native observer without installing the Action-Meta control."""

    native_macro = int(stage0_config["action_meta_lora"]["native_checkpoint_macro"])
    model = build_stage0_model(
        stage0_config,
        owners,
        max_frames_per_call=max_frames_per_call,
    ).to(device)
    model.load_state_dict(
        load_file(
            str(
                _checkpoint_weights(
                    native_checkpoint,
                    stage="stage0_native",
                    required_macro=native_macro,
                )
            ),
            device=str(device),
        ),
        strict=True,
    )
    return model.requires_grad_(False).eval()


def _checkpoint_weights(checkpoint: Path, *, stage: str, required_macro: int) -> Path:
    manifest = read_json(checkpoint / "checkpoint_manifest.json")
    weights = checkpoint / "ecp.safetensors"
    if (
        checkpoint_macro(checkpoint) != required_macro
        or manifest.get("stage") != stage
        or int(manifest.get("next_macro", -1)) != required_macro
        or not weights.is_file()
        or weights.stat().st_size
        != int(manifest.get("files", {}).get(weights.name, {}).get("bytes", -1))
    ):
        raise ValueError(f"frozen ECP {stage} authority changed")
    return weights


def load_frozen_observer_authority(
    *,
    stage0_config: dict,
    owners: tuple[TargetOwner, ...],
    policy: torch.nn.Module,
    native_checkpoint: Path,
    action_meta_checkpoint: Path,
    device: torch.device,
    max_frames_per_call: int | None = None,
) -> FrozenObserverAuthority:
    model = load_frozen_native_observer(
        stage0_config=stage0_config,
        owners=owners,
        native_checkpoint=native_checkpoint,
        device=device,
        max_frames_per_call=max_frames_per_call,
    )

    expert = policy.model.paligemma_with_expert.gemma_expert.model
    action_meta = MetaLoRAStack(
        expert.layers, int(stage0_config["action_meta_lora"]["rank"])
    ).to(device)
    action_meta.load_state_dict(
        load_file(
            str(
                _checkpoint_weights(
                    action_meta_checkpoint,
                    stage="stage0_action_meta",
                    required_macro=int(
                        stage0_config["action_meta_lora"]["formal_run"]["total_macros"]
                    ),
                )
            ),
            device=str(device),
        ),
        strict=True,
    )
    action_meta.requires_grad_(False).eval()
    return FrozenObserverAuthority(
        model=model,
        action_meta=action_meta,
        native_checkpoint=native_checkpoint.resolve(),
        action_meta_checkpoint=action_meta_checkpoint.resolve(),
    )
