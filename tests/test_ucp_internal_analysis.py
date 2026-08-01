from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.writer.ucp_analysis import (
    build_initial_program,
    effective_variance,
    effective_metrics,
    lora_geometry,
    pack_flat,
    reader_attention,
    reader_attention_summary,
    relative_metrics,
    resample_intervals,
    type_ablation,
    validate_analysis_provenance,
)
from ember.writer.model import LoraTensorSpec, WriterModelError
from ember.writer.program_compiler import TargetRankProgramReader


def test_initial_program_uses_outgoing_grounded_change() -> None:
    absolute = torch.arange(1 * 4 * 2 * 4, dtype=torch.float32).reshape(1, 4, 2, 4)
    grounded = absolute.square()
    action = torch.arange(1 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4)
    positions = torch.tensor([[0, 5, 10, 15]])
    valid_frames = torch.ones(1, 4, dtype=torch.bool)
    valid_tokens = torch.tensor([[True, False]])

    program, endpoints, valid, semantics = build_initial_program(
        absolute, grounded, action, positions, valid_frames, valid_tokens,
    )

    assert program.shape == (1, 3, 5, 4)
    torch.testing.assert_close(program[:, :, :2], absolute[:, :-1].masked_fill(
        ~valid_tokens[:, None, :, None], 0,
    ))
    torch.testing.assert_close(program[:, :, 2], action[:, :-1])
    torch.testing.assert_close(
        program[:, :, 3], grounded[:, 1:, 0] - grounded[:, :-1, 0]
    )
    assert torch.count_nonzero(program[:, :, 4]) == 0
    assert torch.equal(endpoints, positions[:, 1:])
    assert bool(valid.all())
    assert semantics.tolist() == [[True, False, True, True, False]]


def test_reader_attention_is_normalized_and_type_mass_is_complete() -> None:
    torch.manual_seed(9)
    compiler = TargetRankProgramReader(
        width=32, heads=4, target_count=38, rank=16, initialization_seed=7,
    )
    writer = SimpleNamespace(compiler=compiler)
    program = torch.randn(1, 3, 5, 32)
    endpoints = torch.tensor([[5, 10, 15]])
    valid_intervals = torch.tensor([[True, True, False]])
    valid_semantics = torch.tensor([[True, True, True, True, False]])

    weights = reader_attention(
        writer, program, endpoints, valid_intervals, valid_semantics,
    )
    summary = reader_attention_summary(weights, valid_intervals, valid_semantics)

    assert weights.shape == (1, 4, 38 * 16, 15)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights[..., 0]))
    assert summary["x_mass"] + summary["a_mass"] + summary["d_mass"] == pytest.approx(1.0)
    assert 0 <= summary["normalized_entropy_mean"] <= 1.000001
    assert summary["top_mass_mean"] > 0


def _fake_writer_and_states() -> tuple[SimpleNamespace, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    specs = []
    left, right = {}, {}
    modules = (
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.1.self_attn.v_proj",
        "model.action_in_proj",
        "model.action_out_proj",
    )
    for index, module in enumerate(modules):
        name_a = module + ".lora_A.default.weight"
        name_b = module + ".lora_B.default.weight"
        specs.extend((
            LoraTensorSpec(name_a, module, index, 0, 16, 3, False),
            LoraTensorSpec(name_b, module, index, 1, 16, 2, True),
        ))
        base_a = torch.arange(48, dtype=torch.float32).reshape(16, 3) / 50 + index
        base_b = torch.arange(32, dtype=torch.float32).reshape(2, 16) / 40 + index
        left[name_a], left[name_b] = base_a, base_b
        right[name_a], right[name_b] = base_a * 1.1, base_b * .9
    return SimpleNamespace(tensor_specs=tuple(specs), PUBLIC_LORA_RANK=16), left, right


def test_effective_geometry_uses_functional_ba_not_raw_factor_sign() -> None:
    writer, left, right = _fake_writer_and_states()
    sign_flipped = {
        name: (-value if ".lora_" in name else value)
        for name, value in left.items()
    }
    metrics = effective_metrics(writer, left, sign_flipped)
    geometry = lora_geometry(writer, right)

    assert metrics["relative_l2"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert geometry["effective_lora_norm"] > 0
    assert 1 <= geometry["stable_rank"] <= 16
    assert sum(geometry["coordinate_energy_participation"]) == pytest.approx(1.0)
    assert sum(geometry["q_v_action_energy_ratio"].values()) == pytest.approx(1.0)
    assert all(value > 0 for value in geometry["q_v_action_energy_ratio"].values())
    variance = effective_variance(writer, [left, right])
    assert variance["centered_variance_over_sample_energy"] > 0
    assert (
        variance["scale_like_video_variance_fraction"]
        + variance["orthogonal_direction_video_variance_fraction"]
    ) == pytest.approx(1.0, abs=1e-5)


def test_metric_and_interval_helpers_are_shape_strict() -> None:
    value = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
    resized = resample_intervals(value, 5)
    assert resized.shape == (5, 2, 4)
    assert relative_metrics(value, value)["relative_l2"] == 0
    with pytest.raises(WriterModelError):
        relative_metrics(torch.ones(2), torch.ones(3))

    packed, valid = pack_flat(torch.arange(20).reshape(5, 4), (0, 2, 5))
    assert packed.shape == (2, 3, 4)
    assert valid.tolist() == [[True, True, False], [True, True, True]]
    program = torch.ones(1, 2, 5, 4)
    x_only = type_ablation(program, "x_only")
    dynamic_only = type_ablation(program, "dynamic_only")
    assert torch.count_nonzero(x_only[:, :, 2:]) == 0
    assert torch.count_nonzero(dynamic_only[:, :, :2]) == 0


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True,
    ).stdout.strip()


def test_provenance_accepts_unprotected_descendant_and_rejects_model_change(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    (repo / "src/ember/writer").mkdir(parents=True)
    (repo / "configs").mkdir()
    (repo / "scripts").mkdir()
    (repo / "src/ember/writer/model.py").write_text("trained\n")
    (repo / "src/ember/pi05_lora.py").write_text("contract\n")
    (repo / "configs/pi05_as_writer_unified_causal_program_full24_decay400_v1.json").write_text("{}\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "training")
    training = _git(repo, "rev-parse", "HEAD")
    (repo / "scripts/analyze.py").write_text("analysis\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "analysis")
    head = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", head)
    state = {"commit": head, "origin_main": head, "dirty_paths": []}

    record = validate_analysis_provenance(
        repo=repo, state=state, training={"git": {"commit": training}},
    )
    assert record["training_is_ancestor"] is True

    (repo / "src/ember/writer/model.py").write_text("changed\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "model change")
    changed = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", changed)
    with pytest.raises(WriterModelError, match="model/config changed"):
        validate_analysis_provenance(
            repo=repo,
            state={"commit": changed, "origin_main": changed, "dirty_paths": []},
            training={"git": {"commit": training}},
        )
