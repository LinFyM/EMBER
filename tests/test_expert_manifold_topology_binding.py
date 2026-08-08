import copy
import json
from pathlib import Path

import pytest
import torch

from ember.expert_manifold.contract import (
    ExpertManifoldError,
    load_barycentric_writer_config,
)
from ember.expert_manifold.model import HardRoutedPolicyEffectiveWriter
from ember.lora import (
    LoRAContract,
    LoRATarget,
    SmolVLALoRAContract,
    identity_lora_state,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT
    / "configs/pi05_video_expert_manifold_hard_routed_policy_effective_v2.json"
)


def _writer() -> tuple[
    HardRoutedPolicyEffectiveWriter,
    LoRAContract,
    dict[str, torch.Tensor],
    tuple[dict[str, torch.Tensor], ...],
]:
    contract = SmolVLALoRAContract(
        targets=(LoRATarget("layer", 3, 4),),
        rank=2,
        alpha=2,
        dropout=0.0,
        identity_seed=7,
    )
    template = identity_lora_state(contract)
    experts = []
    for ordinal in range(3):
        state = {name: value.clone() for name, value in template.items()}
        state["layer.lora_A.default.weight"].add_(
            torch.tensor(
                [
                    [0.1 + ordinal, 0.2, -0.3],
                    [0.4, -0.5 - ordinal, 0.6],
                ]
            )
        )
        state["layer.lora_B.default.weight"].copy_(
            torch.tensor(
                [
                    [0.2 + ordinal, -0.3],
                    [0.4, 0.5 + ordinal],
                    [-0.6, 0.7],
                    [0.8 + ordinal, -0.9],
                ]
            )
        )
        experts.append(state)
    centroids = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    writer = HardRoutedPolicyEffectiveWriter(
        contract=contract,
        template_state=template,
        expert_states=experts,
        task_centroids=centroids,
        phase_slots=4,
        feature_width=4,
        ridge=0.3,
        effective_basis_rank=4,
    )
    return writer, contract, template, tuple(experts)


def _effective(
    state: dict[str, torch.Tensor], index: int | None = None
) -> torch.Tensor:
    a = state["layer.lora_A.default.weight"]
    b = state["layer.lora_B.default.weight"]
    if index is not None:
        a, b = a[index], b[index]
    return b @ a


def test_hard_routed_config_has_no_learned_or_language_only_value_path() -> None:
    config = load_barycentric_writer_config(CONFIG)
    assert config["method"]["learned_writer_parameter_count"] == 0
    assert config["method"]["language_only_lora_path"] is False
    assert config["video_features"]["shots"] == 1
    assert config["expert_basis"]["expert_step"] == 2000
    assert config["barycentric_writer"]["ridge"] == 0.3
    assert config["barycentric_writer"]["effective_basis_rank"] == 96
    assert config["barycentric_writer"]["deployed_coefficient_support"] == 1
    assert config["evaluation"]["formal_status"] == (
        "blocked_until_cpu_hard_route_evidence"
    )
    assert "online_smoke_evidence" not in config["evaluation"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("ridge", 0.4),
        ("language_only_lora_path", True),
        ("effective_basis_rank", 64),
    ),
)
def test_barycentric_config_fails_closed_on_method_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    changed = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    changed["barycentric_writer"][field] = value
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="scientific boundary changed"):
        load_barycentric_writer_config(path)


def test_barycentric_config_fails_closed_on_information_wall_drift(
    tmp_path: Path,
) -> None:
    changed = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    changed["information_wall"]["writer_forbidden_inputs"].remove("action")
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ExpertManifoldError, match="scientific boundary changed"):
        load_barycentric_writer_config(path)


def test_zero_and_phase_constant_video_are_exact_source_identity() -> None:
    writer, _, template, _ = _writer()
    for video in (
        torch.zeros(2, 4, 4),
        torch.randn(2, 1, 4).expand(2, 4, 4).clone(),
    ):
        coefficients = writer.coefficients(video)
        assert torch.count_nonzero(coefficients) == 0
        generated = writer(video)
        assert all(
            torch.equal(generated[name], value.expand(2, *value.shape))
            for name, value in template.items()
        )


def test_one_hot_coefficients_reconstruct_each_expert_effective_update() -> None:
    writer, _, _, experts = _writer()
    generated = writer.states_from_coefficients(torch.eye(3))
    for expert_index, expert in enumerate(experts):
        assert torch.allclose(
            _effective(generated, expert_index),
            _effective(expert),
            atol=2e-5,
            rtol=2e-5,
        )


def test_nonzero_video_routes_are_deterministic_one_hot_and_ordered() -> None:
    writer, _, _, _ = _writer()
    video = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )[None]
    affine = writer.affine_coefficients(video)
    reversed_affine = writer.affine_coefficients(video.flip(1))
    first = writer.coefficients(video)
    second = writer.coefficients(video.clone())
    reversed_value = writer.coefficients(video.flip(1))
    assert torch.equal(first, second)
    assert torch.allclose(affine.sum(dim=1), torch.ones(1), atol=1e-6)
    assert torch.allclose(reversed_affine.sum(dim=1), torch.ones(1), atol=1e-6)
    assert torch.allclose(first.sum(dim=1), torch.ones(1), atol=1e-6)
    assert torch.allclose(reversed_value.sum(dim=1), torch.ones(1), atol=1e-6)
    assert torch.count_nonzero(first) == 1
    assert torch.count_nonzero(reversed_value) == 1
    assert first.argmax(dim=1).item() == affine.argmax(dim=1).item()
    assert reversed_value.argmax(dim=1).item() == reversed_affine.argmax(dim=1).item()
    assert not torch.allclose(first, reversed_value)
    assert not torch.allclose(
        _effective(writer(video), 0), _effective(writer(video.flip(1)), 0)
    )


def test_effective_direction_log_norm_matches_best_public_rank_projection() -> None:
    writer, contract, _, experts = _writer()
    coefficients = torch.tensor([[1.4, -0.6, 0.2], [-0.5, 0.75, 0.75]])
    generated = writer.states_from_coefficients(coefficients)
    expert_updates = torch.stack([_effective(expert) for expert in experts])
    expert_norms = expert_updates.flatten(1).norm(dim=1)
    for batch, coefficient in enumerate(coefficients):
        direction = torch.einsum(
            "k,koi->oi", coefficient / expert_norms, expert_updates
        )
        scale = (
            (coefficient @ expert_norms.log())
            .clamp(expert_norms.log().min(), expert_norms.log().max())
            .exp()
        )
        target = direction / direction.norm() * scale
        u, singular, vh = torch.linalg.svd(target, full_matrices=False)
        expected = (u[:, : contract.rank] * singular[: contract.rank]) @ vh[
            : contract.rank
        ]
        assert torch.allclose(
            _effective(generated, batch), expected, atol=3e-5, rtol=3e-5
        )


def test_factor_gauge_and_shapes_are_finite_and_expert_scaled() -> None:
    writer, contract, _, experts = _writer()
    coefficients = torch.tensor([[1.4, -0.6, 0.2], [-0.5, 0.75, 0.75]])
    generated = writer.states_from_coefficients(coefficients)
    a = generated["layer.lora_A.default.weight"]
    b = generated["layer.lora_B.default.weight"]
    expected_a_rms = (
        torch.stack(
            [
                expert["layer.lora_A.default.weight"].square().mean().sqrt()
                for expert in experts
            ]
        )
        .log()
        .mean()
        .exp()
    )
    assert a.shape == (2, contract.rank, 3)
    assert b.shape == (2, 4, contract.rank)
    assert bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    assert torch.allclose(
        a.square().mean(dim=(-2, -1)).sqrt(), expected_a_rms.expand(2), atol=2e-6
    )
    assert tuple(writer.parameters()) == ()
