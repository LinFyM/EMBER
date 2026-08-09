"""Warm-start and counterfactual contracts for the v6-prior Writer."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file

from ember.expert_manifold.contract import ExpertManifoldError, ExpertTask
from ember.writer.architecture import V6_WRITER_PARAMETER_COUNT
from ember.writer.model import CompleteLoRAWriter


V6_WRITER_STATE_TENSOR_COUNT = 600
V6_PRIOR_FROZEN_PARAMETER_COUNT = 7_060_992
V6_PRIOR_TRAINABLE_PARAMETER_COUNT = 3_714_304
V6_PRIOR_FROZEN_ROOTS = (
    "semantic_encoder",
    "semantic_core",
    "visual_transition",
    "procedure",
)
V6_PRIOR_TRAINABLE_ROOTS = ("compiler", "factor_heads")
COUNTERFACTUAL_KINDS = ("reversed", "shuffled", "wrong")


@dataclass(frozen=True)
class V6PriorOwnership:
    frozen_parameter_count: int
    trainable_parameter_count: int
    frozen_tensor_count: int
    trainable_tensor_count: int


@dataclass(frozen=True)
class V6PriorWarmStart:
    checkpoint: Path
    state_tensor_count: int
    state_value_count: int


@dataclass(frozen=True)
class V6PriorDynamicAnchor:
    """Training-only frozen historical decoder, outside student ownership."""

    compiler: torch.nn.Module
    factor_heads: torch.nn.ModuleDict
    parameter_count: int
    tensor_count: int


def load_v6_prior_warm_start_(
    writer: CompleteLoRAWriter,
    checkpoint: Path,
) -> V6PriorWarmStart:
    """Strictly load the historical v6-fast Writer without optimizer state."""

    selected = checkpoint / "writer.safetensors" if checkpoint.is_dir() else checkpoint
    if not selected.is_file():
        raise ExpertManifoldError("v6-prior Writer checkpoint is missing")
    state = load_file(str(selected), device="cpu")
    if len(state) != V6_WRITER_STATE_TENSOR_COUNT:
        raise ExpertManifoldError("v6-prior Writer state tensor count changed")
    try:
        incompatible = writer.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ExpertManifoldError(
            "v6-prior Writer checkpoint is not structurally compatible"
        ) from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ExpertManifoldError("v6-prior Writer strict load was incomplete")
    return V6PriorWarmStart(
        checkpoint=selected.resolve(),
        state_tensor_count=len(state),
        state_value_count=sum(value.numel() for value in state.values()),
    )


def configure_v6_prior_trainability(
    writer: CompleteLoRAWriter,
) -> V6PriorOwnership:
    """Freeze learned video representations and train only compiler/output heads."""

    roots = {name.split(".", 1)[0] for name, _ in writer.named_parameters()}
    expected_roots = set(V6_PRIOR_FROZEN_ROOTS) | set(V6_PRIOR_TRAINABLE_ROOTS)
    if roots != expected_roots:
        raise ExpertManifoldError("v6-prior Writer parameter ownership changed")
    writer.train()
    for name in V6_PRIOR_FROZEN_ROOTS:
        module = getattr(writer, name)
        module.requires_grad_(False)
        module.eval()
    for name in V6_PRIOR_TRAINABLE_ROOTS:
        module = getattr(writer, name)
        module.requires_grad_(True)
        module.train()
    frozen = [
        parameter
        for name, parameter in writer.named_parameters()
        if name.split(".", 1)[0] in V6_PRIOR_FROZEN_ROOTS
    ]
    trainable = [
        parameter
        for name, parameter in writer.named_parameters()
        if name.split(".", 1)[0] in V6_PRIOR_TRAINABLE_ROOTS
    ]
    observed = V6PriorOwnership(
        frozen_parameter_count=sum(value.numel() for value in frozen),
        trainable_parameter_count=sum(value.numel() for value in trainable),
        frozen_tensor_count=len(frozen),
        trainable_tensor_count=len(trainable),
    )
    if (
        observed.frozen_parameter_count != V6_PRIOR_FROZEN_PARAMETER_COUNT
        or observed.trainable_parameter_count
        != V6_PRIOR_TRAINABLE_PARAMETER_COUNT
        or observed.frozen_parameter_count + observed.trainable_parameter_count
        != V6_WRITER_PARAMETER_COUNT
        or any(value.requires_grad for value in frozen)
        or any(not value.requires_grad for value in trainable)
    ):
        raise ExpertManifoldError("v6-prior Writer trainability seal changed")
    return observed


def build_v6_prior_dynamic_anchor(
    writer: CompleteLoRAWriter,
) -> V6PriorDynamicAnchor:
    """Clone only the synchronized macro0 decoder before any resume load."""

    compiler = copy.deepcopy(writer.compiler)
    factor_heads = copy.deepcopy(writer.factor_heads)
    compiler.requires_grad_(False).eval()
    factor_heads.requires_grad_(False).eval()
    anchor_rows = tuple(
        (f"compiler.{name}", parameter)
        for name, parameter in compiler.named_parameters()
    ) + tuple(
        (f"factor_heads.{name}", parameter)
        for name, parameter in factor_heads.named_parameters()
    )
    student_rows = tuple(
        (name, parameter)
        for name, parameter in writer.named_parameters()
        if name.split(".", 1)[0] in V6_PRIOR_TRAINABLE_ROOTS
    )
    valid = (
        len(anchor_rows) == len(student_rows) == 41
        and tuple(name for name, _ in anchor_rows)
        == tuple(name for name, _ in student_rows)
        and sum(parameter.numel() for _, parameter in anchor_rows)
        == V6_PRIOR_TRAINABLE_PARAMETER_COUNT
        and all(not parameter.requires_grad for _, parameter in anchor_rows)
        and all(
            anchor.shape == student.shape
            and anchor.dtype == student.dtype
            and anchor.device == student.device
            and anchor.data_ptr() != student.data_ptr()
            for (_, anchor), (_, student) in zip(
                anchor_rows,
                student_rows,
                strict=True,
            )
        )
    )
    if not valid:
        raise ExpertManifoldError("v6 dynamic anchor ownership changed")
    return V6PriorDynamicAnchor(
        compiler=compiler,
        factor_heads=factor_heads,
        parameter_count=V6_PRIOR_TRAINABLE_PARAMETER_COUNT,
        tensor_count=len(anchor_rows),
    )


def load_v6_prior_comparison_decoder(
    writer: CompleteLoRAWriter,
    checkpoint: Path,
) -> V6PriorDynamicAnchor:
    """Load only a frozen compiler/head decoder for a no-update comparison."""

    selected = checkpoint / "writer.safetensors"
    if not selected.is_file():
        raise ExpertManifoldError("v6 comparison Writer checkpoint is missing")
    state = load_file(str(selected), device="cpu")
    if len(state) != V6_WRITER_STATE_TENSOR_COUNT:
        raise ExpertManifoldError("v6 comparison Writer state tensor count changed")
    decoder = build_v6_prior_dynamic_anchor(writer)
    groups = {
        "compiler": decoder.compiler,
        "factor_heads": decoder.factor_heads,
    }
    selected_names = []
    for root, module in groups.items():
        prefix = f"{root}."
        values = {
            name[len(prefix) :]: value
            for name, value in state.items()
            if name.startswith(prefix)
        }
        try:
            incompatible = module.load_state_dict(values, strict=True)
        except RuntimeError as error:
            raise ExpertManifoldError(
                "v6 comparison decoder is structurally incompatible"
            ) from error
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ExpertManifoldError("v6 comparison decoder strict load was incomplete")
        selected_names.extend(f"{root}.{name}" for name in values)
    if (
        len(selected_names) != decoder.tensor_count
        or any(
            parameter.requires_grad
            for module in groups.values()
            for parameter in module.parameters()
        )
    ):
        raise ExpertManifoldError("v6 comparison decoder ownership changed")
    return decoder


def v6_prior_trainable_parameters(
    writer: CompleteLoRAWriter,
) -> tuple[torch.nn.Parameter, ...]:
    """Return only the sealed compiler/head parameters in name order."""

    parameters = tuple(
        parameter for _, parameter in writer.named_parameters() if parameter.requires_grad
    )
    if sum(value.numel() for value in parameters) != V6_PRIOR_TRAINABLE_PARAMETER_COUNT:
        raise ExpertManifoldError("v6-prior trainable parameter selection changed")
    return parameters


def counterfactual_kind(task_ordinal: int, task_visit: int) -> str:
    """Balance all three negative arms within every 24-task macro update."""

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
    """Return display-order indices; wrong-video negatives use another evidence set."""

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
    """Cycle each task through all other suites and their videos."""

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
    target = candidates[(task_ordinal + task_visit // (len(suites) - 1)) % len(candidates)]
    if target.suite == source.suite:
        raise ExpertManifoldError("wrong-video schedule did not cross suites")
    return target
