from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from ember.expert_manifold.effective_objective import (
    effective_auxiliary_output_gradients,
)
from ember.expert_manifold.v6_prior import (
    configure_v6_prior_trainability,
    v6_prior_trainable_parameters,
)
from ember.expert_manifold.v6_prior_step import (
    generate_counterfactual_pair,
    merged_output_gradients,
    parameter_gradient_components,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.writer.data import RawTeacherVideo


ROOT = Path(__file__).resolve().parents[1]


def _writer_and_encoder():
    path = Path(__file__).with_name("test_writer_model.py")
    spec = importlib.util.spec_from_file_location("v6_writer_step_helper", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    writer, _ = module._model()
    configure_v6_prior_trainability(writer)
    writer.semantic_encoder = module._FakeSemanticEncoder()
    torch.nn.init.normal_(writer.compiler.modulation.weight, std=0.01)
    for head in writer.factor_heads.values():
        torch.nn.init.normal_(head.network[-1].weight, std=0.01)
    return writer


def _video(offset: int) -> RawTeacherVideo:
    values = (
        np.arange(5 * 3 * 4 * 4, dtype=np.uint8).reshape(5, 3, 4, 4)
        + offset
    )
    return RawTeacherVideo(
        frames=values,
        frame_indices=np.arange(5, dtype=np.int64) * 5,
        raw_frame_count=21,
    )


def _language() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = torch.tensor([[1, 10, 11, 12, 0]], dtype=torch.long)
    mask = tokens.ne(0)
    span = torch.tensor([[False, True, True, True, False]])
    return tokens, mask, span


def test_v6_prior_step_merges_all_output_gradients_into_trainable_blocks() -> None:
    writer = _writer_and_encoder()
    pair = generate_counterfactual_pair(
        writer=writer,
        policy=torch.nn.Identity(),
        correct_video=_video(0),
        counterfactual_video=None,
        language_tokens=_language(),
        kind="reversed",
        counterfactual_seed=17,
        task_ordinal=3,
        task_visit=2,
        teacher_demo=7,
        device=torch.device("cpu"),
    )
    contract = load_pi05_lora_contract(ROOT / "configs/pi05_lora_v1.json")
    target = {
        name: value.detach().clone().add_(0.01)
        for name, value in pair.correct.items()
    }
    auxiliary = effective_auxiliary_output_gradients(
        pair.correct,
        pair.counterfactual,
        target,
        contract,
        smooth_l1_beta=0.5,
        required_margin=0.1,
        temperature=0.05,
    )
    functional = {
        name: torch.full_like(value, 1e-4) for name, value in pair.correct.items()
    }
    outputs, gradients = merged_output_gradients(
        pair=pair,
        functional=functional,
        auxiliary=auxiliary,
        projection_weight=0.2,
        ranking_weight=0.1,
        task_scale=0.25,
    )
    torch.autograd.backward(outputs, gradients)
    trainable = v6_prior_trainable_parameters(writer)
    assert all(value.grad is not None for value in trainable)
    assert all(
        torch.isfinite(value.grad).all()
        for value in trainable
        if value.grad is not None
    )
    assert all(
        value.grad is None for name in (
            "semantic_encoder",
            "semantic_core",
            "visual_transition",
            "procedure",
        ) for value in getattr(writer, name).parameters()
    )


def test_v6_prior_gradient_profile_returns_three_complete_parameter_vectors() -> None:
    writer = _writer_and_encoder()
    pair = generate_counterfactual_pair(
        writer=writer,
        policy=torch.nn.Identity(),
        correct_video=_video(0),
        counterfactual_video=_video(13),
        language_tokens=_language(),
        kind="wrong",
        counterfactual_seed=17,
        task_ordinal=3,
        task_visit=2,
        teacher_demo=7,
        device=torch.device("cpu"),
    )
    contract = load_pi05_lora_contract(ROOT / "configs/pi05_lora_v1.json")
    target = {
        name: value.detach().clone().add_(0.01)
        for name, value in pair.correct.items()
    }
    auxiliary = effective_auxiliary_output_gradients(
        pair.correct,
        pair.counterfactual,
        target,
        contract,
        smooth_l1_beta=0.5,
        required_margin=0.1,
        temperature=0.05,
    )
    functional = {
        name: torch.full_like(value, 1e-4) for name, value in pair.correct.items()
    }
    parameters = v6_prior_trainable_parameters(writer)
    components = parameter_gradient_components(
        pair=pair,
        functional=functional,
        auxiliary=auxiliary,
        parameters=parameters,
    )
    assert len(components.positive) == len(parameters) == 41
    assert len(components.projection) == len(parameters)
    assert len(components.ranking) == len(parameters)
    assert all(
        sum(int(torch.count_nonzero(value)) for value in component) > 0
        for component in (
            components.positive,
            components.projection,
            components.ranking,
        )
    )
