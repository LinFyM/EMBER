"""Frozen historical-v6 and counterfactual video contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file

from ember.expert_manifold.contract import ExpertManifoldError, ExpertTask
from ember.expert_manifold.legacy_v6_architecture import V6_WRITER_PARAMETER_COUNT
from ember.expert_manifold.legacy_v6_model import CompleteLoRAWriter


V6_WRITER_STATE_TENSOR_COUNT = 600
V6_WRITER_PARAMETER_TENSOR_COUNT = 523
COUNTERFACTUAL_KINDS = ("reversed", "shuffled", "wrong")


@dataclass(frozen=True)
class V6PriorOwnership:
    """Exact all-frozen ownership of the historical v6 Writer."""

    frozen_parameter_count: int
    frozen_parameter_tensor_count: int
    state_tensor_count: int
    trainable_parameter_count: int = 0


@dataclass(frozen=True)
class V6PriorWarmStart:
    checkpoint: Path
    state_tensor_count: int
    state_value_count: int


def load_v6_prior_warm_start_(
    writer: CompleteLoRAWriter,
    checkpoint: Path,
) -> V6PriorWarmStart:
    """Strictly load all 600 historical v6 tensors without optimizer state."""

    selected = checkpoint / "writer.safetensors" if checkpoint.is_dir() else checkpoint
    if not selected.is_file():
        raise ExpertManifoldError("historical v6 Writer checkpoint is missing")
    state = load_file(str(selected), device="cpu")
    if len(state) != V6_WRITER_STATE_TENSOR_COUNT:
        raise ExpertManifoldError("historical v6 Writer state tensor count changed")
    try:
        incompatible = writer.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ExpertManifoldError(
            "historical v6 Writer checkpoint is structurally incompatible"
        ) from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ExpertManifoldError("historical v6 Writer strict load was incomplete")
    return V6PriorWarmStart(
        checkpoint=selected.resolve(),
        state_tensor_count=len(state),
        state_value_count=sum(value.numel() for value in state.values()),
    )


def freeze_v6_prior_writer(writer: CompleteLoRAWriter) -> V6PriorOwnership:
    """Freeze and seal every historical v6 parameter and state tensor."""

    writer.requires_grad_(False).eval()
    parameters = tuple(writer.parameters())
    state = writer.state_dict()
    ownership = V6PriorOwnership(
        frozen_parameter_count=sum(value.numel() for value in parameters),
        frozen_parameter_tensor_count=len(parameters),
        state_tensor_count=len(state),
    )
    if (
        ownership.frozen_parameter_count != V6_WRITER_PARAMETER_COUNT
        or ownership.frozen_parameter_tensor_count != V6_WRITER_PARAMETER_TENSOR_COUNT
        or ownership.state_tensor_count != V6_WRITER_STATE_TENSOR_COUNT
        or any(parameter.requires_grad for parameter in parameters)
        or writer.training
    ):
        raise ExpertManifoldError("historical v6 Writer freeze seal changed")
    return ownership


def counterfactual_kind(task_ordinal: int, task_visit: int) -> str:
    """Balance reversed, shuffled, and wrong rows within every train24 macro."""

    if task_ordinal < 0 or task_visit < 0:
        raise ExpertManifoldError("counterfactual schedule cursor is negative")
    return COUNTERFACTUAL_KINDS[(task_ordinal + task_visit) % len(COUNTERFACTUAL_KINDS)]


def _nontrivial_shuffle(length: int, seed: int) -> torch.Tensor:
    if length < 3 or seed < 0:
        raise ExpertManifoldError("shuffled video needs at least three frames")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(length, generator=generator)
    identity = torch.arange(length)
    reverse = identity.flip(0)
    if torch.equal(order, identity) or torch.equal(order, reverse):
        order = identity.roll(1)
    return order


def counterfactual_frame_order(
    kind: str,
    offsets: Sequence[int],
    *,
    seed: int,
    task_ordinal: int,
    task_visit: int,
    teacher_demo: int,
    device: torch.device | str,
) -> torch.Tensor | None:
    """Reorder real content while retaining the original display ordinals."""

    values = tuple(int(value) for value in offsets)
    if (
        kind not in COUNTERFACTUAL_KINDS
        or len(values) < 2
        or values[0] != 0
        or any(right <= left for left, right in zip(values, values[1:]))
        or min(seed, task_ordinal, task_visit, teacher_demo) < 0
    ):
        raise ExpertManifoldError("invalid counterfactual frame-order request")
    if kind == "wrong":
        return None
    rows = []
    for condition, (left, right) in enumerate(zip(values, values[1:])):
        length = right - left
        if kind == "reversed":
            local = torch.arange(length - 1, -1, -1)
        else:
            local_seed = (
                seed
                + 1_000_003 * task_ordinal
                + 9_176 * task_visit
                + 131 * teacher_demo
                + 17 * condition
            ) % (2**63 - 1)
            local = _nontrivial_shuffle(length, local_seed)
        rows.append(local + left)
    return torch.cat(rows).to(device=device)


def cross_suite_wrong_task(
    tasks: Sequence[ExpertTask],
    *,
    task_ordinal: int,
    task_visit: int,
) -> ExpertTask:
    """Cycle each target language through another suite's teacher videos."""

    ordered = tuple(sorted(tasks, key=lambda item: item.ordinal))
    if (
        len(ordered) < 2
        or tuple(item.ordinal for item in ordered) != tuple(range(len(ordered)))
        or not 0 <= task_ordinal < len(ordered)
        or task_visit < 0
    ):
        raise ExpertManifoldError("invalid cross-suite wrong-video task bank")
    source = ordered[task_ordinal]
    suites = tuple(dict.fromkeys(item.suite for item in ordered))
    if len(suites) < 2:
        raise ExpertManifoldError("wrong-video bank has only one suite")
    source_suite = suites.index(source.suite)
    suite_offset = 1 + (task_visit % (len(suites) - 1))
    target_suite = suites[(source_suite + suite_offset) % len(suites)]
    candidates = tuple(item for item in ordered if item.suite == target_suite)
    if not candidates:
        raise ExpertManifoldError("wrong-video target suite is empty")
    target = candidates[
        (task_ordinal + task_visit // (len(suites) - 1)) % len(candidates)
    ]
    if target.suite == source.suite:
        raise ExpertManifoldError("wrong-video schedule did not cross suites")
    return target
