import copy
import json
from pathlib import Path

import pytest
import torch

from ember.expert_manifold.contract import (
    ExpertManifoldError,
    load_barycentric_writer_config,
)
from ember.expert_manifold.model import CausalBarycentricTopologicalWriter
from ember.lora import LoRAContract, LoRATarget, SmolVLALoRAContract, identity_lora_state


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT
    / "configs/pi05_video_expert_manifold_causal_barycentric_v1.json"
)


def _writer() -> tuple[
    CausalBarycentricTopologicalWriter,
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
    writer = CausalBarycentricTopologicalWriter(
        contract=contract,
        template_state=template,
        expert_states=experts,
        task_centroids=centroids,
        phase_slots=4,
        feature_width=4,
        chunk_width=2,
        ridge=0.3,
    )
    return writer, contract, template, tuple(experts)


def test_barycentric_config_has_no_learned_or_language_only_value_path() -> None:
    config = load_barycentric_writer_config(CONFIG)
    assert config["method"]["learned_writer_parameter_count"] == 0
    assert config["method"]["language_only_lora_path"] is False
    assert config["video_features"]["shots"] == 1
    assert config["expert_basis"]["expert_step"] == 2000
    assert config["barycentric_writer"]["ridge"] == 0.3
    assert config["evaluation"]["formal_status"] == "sealed"
    assert config["evaluation"]["online_smoke_evidence"][
        "writer_modules_released"
    ] is True


@pytest.mark.parametrize(
    ("field", "value"),
    (("ridge", 0.4), ("language_only_lora_path", True)),
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


def test_one_hot_coefficients_reconstruct_each_complete_expert() -> None:
    writer, _, _, experts = _writer()
    generated = writer.layout.detokenize(
        writer.values_from_coefficients(torch.eye(3)), writer.template_state()
    )
    for expert_index, expert in enumerate(experts):
        assert set(generated) == set(expert)
        assert all(
            torch.allclose(
                generated[name][expert_index], value, atol=2e-6, rtol=2e-6
            )
            for name, value in expert.items()
        )


def test_nonzero_video_coefficients_are_deterministic_affine_and_ordered() -> None:
    writer, _, _, _ = _writer()
    video = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )[None]
    first = writer.coefficients(video)
    second = writer.coefficients(video.clone())
    reversed_value = writer.coefficients(video.flip(1))
    assert torch.equal(first, second)
    assert torch.allclose(first.sum(dim=1), torch.ones(1), atol=1e-6)
    assert torch.allclose(reversed_value.sum(dim=1), torch.ones(1), atol=1e-6)
    assert not torch.allclose(first, reversed_value)
    assert not torch.allclose(
        writer.forward_values(video), writer.forward_values(video.flip(1))
    )


def test_chunk_scales_stay_inside_expert_envelope_and_shapes_are_finite() -> None:
    writer, contract, _, _ = _writer()
    coefficients = torch.tensor([[1.4, -0.6, 0.2], [-0.5, 0.75, 0.75]])
    values = writer.values_from_coefficients(coefficients)
    mask = writer.valid_value_mask[None, :, None, :].to(values.dtype)
    count = writer.valid_value_mask.sum(dim=1)[None].to(values.dtype) * contract.rank
    scale = torch.sqrt(
        (values.square() * mask).sum(dim=(-2, -1)) / count.clamp_min(1.0)
    )
    assert values.shape == (2, writer.layout.chunk_count, contract.rank, 2)
    assert bool(torch.isfinite(values).all())
    assert bool(
        (
            scale.log()
            >= writer.chunk_log_scale_min[None] - 2e-6
        ).all()
    )
    assert bool(
        (
            scale.log()
            <= writer.chunk_log_scale_max[None] + 2e-6
        ).all()
    )
    assert tuple(writer.parameters()) == ()
