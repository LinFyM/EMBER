"""Deterministic policy-functional query panels for fixed-decoder fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import default_collate

from ember.functional_adaptation.functional_response import (
    FunctionalResponseTarget,
    build_functional_response_target,
    functional_response_distillation_loss,
)
from ember.lora import LoRAContract


@dataclass(frozen=True)
class FunctionalProbePanel:
    batch: Mapping[str, Any]
    target: FunctionalResponseTarget
    policy_seed: int


def panel_for_visit(
    panels: Sequence[FunctionalProbePanel], visit: int
) -> tuple[FunctionalProbePanel, ...]:
    """Rotate fixed panels for one task without outcome-based resampling."""

    if not panels or visit < 0:
        raise ValueError("invalid functional probe visit")
    return (panels[visit % len(panels)],)


def selected_probe_rows(
    task_episode_rows: Mapping[int, Sequence[int]],
    *,
    demo_indices: Sequence[int],
    panel_count: int,
    batch_size: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Select fixed query rows without reading losses, rewards, or outcomes."""

    pool = tuple(
        row
        for demo_index in demo_indices
        for row in task_episode_rows[int(demo_index)]
    )
    needed = panel_count * batch_size
    if needed <= 0 or len(pool) < needed:
        raise ValueError("functional probe panel exceeds its episode pool")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    selected = torch.randperm(len(pool), generator=generator)[:needed].tolist()
    flat = tuple(pool[index] for index in selected)
    return tuple(
        flat[start : start + batch_size]
        for start in range(0, needed, batch_size)
    )


def build_probe_panels(
    *,
    policy: torch.nn.Module,
    processor: Any,
    dataset: Any,
    rows: Sequence[Sequence[int]],
    identity_state: Mapping[str, torch.Tensor],
    expert_state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    policy_seed: int,
) -> tuple[FunctionalProbePanel, ...]:
    panels = []
    for ordinal, selected in enumerate(rows):
        batch = processor.training_batch(
            default_collate([dataset[index] for index in selected])
        )
        seed = int(policy_seed) + ordinal
        target = build_functional_response_target(
            policy,
            identity_state,
            expert_state,
            contract,
            batch,
            policy_seed=seed,
        )
        panels.append(FunctionalProbePanel(batch=batch, target=target, policy_seed=seed))
    return tuple(panels)


def mean_functional_probe_loss(
    policy: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    contract: LoRAContract,
    panels: Sequence[FunctionalProbePanel],
) -> torch.Tensor:
    losses = [
        functional_response_distillation_loss(
            policy,
            state,
            contract,
            panel.batch,
            panel.target,
            policy_seed=panel.policy_seed,
        )
        for panel in panels
    ]
    if not losses:
        raise ValueError("functional probe panel is empty")
    return torch.stack(losses).mean()
