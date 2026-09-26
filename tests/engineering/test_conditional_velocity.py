"""Exact factor algebra, teaching readout credit and valid resume prefix."""

import json
from pathlib import Path

import pytest
import torch

from ember.lora import identity_lora_state, validate_lora_state
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.writer.conditional_velocity import ActionRowReadout, compile_velocity_state
from ember.writer.conditional_velocity_engineering import _resume_prefix


def test_full_rank135_identity_and_exact_action_out_operator():
    base = load_pi05_lora_contract(
        Path(__file__).resolve().parents[2] / "configs/pi05_lora_v1.json"
    )
    common = identity_lora_state(derive_pi05_lora_rank(base, rank=128))
    coefficient = torch.randn(7, 256, requires_grad=True)
    projection = torch.randn(256, 1024, requires_grad=True)
    compiled = compile_velocity_state(common, coefficient, projection)
    validate_lora_state(compiled, derive_pi05_lora_rank(base, rank=135))
    prefix = "model.action_out_proj"
    a, b = compiled[prefix + ".lora_A.default.weight"], compiled[prefix + ".lora_B.default.weight"]
    expected = torch.zeros(32, 1024)
    expected[:7] = coefficient @ projection
    assert torch.allclose(b @ a, expected, rtol=1e-5, atol=1e-5)
    other = "model.action_in_proj"
    assert torch.count_nonzero(compiled[other + ".lora_B.default.weight"] @
                               compiled[other + ".lora_A.default.weight"]) == 0
    loss = (b @ a).square().mean()
    loss.backward()
    assert coefficient.grad is not None and coefficient.grad.norm() > 0
    assert projection.grad is not None and projection.grad.norm() > 0


def test_zero_readout_head_then_real_core_procedure_credit():
    torch.manual_seed(7)
    readout = ActionRowReadout()
    core = torch.randn(1, 4, 256, requires_grad=True)
    procedure = torch.randn(1, 5, 256, requires_grad=True)
    core_mask = torch.tensor([[True, True, True, False]])
    frame_mask = torch.ones(1, 5, dtype=torch.bool)
    positions = torch.tensor([[0, 5, 10, 15, 17]])
    first = readout(core, core_mask, procedure, positions, frame_mask)
    assert first.shape == (7, 256) and torch.count_nonzero(first) == 0
    first.sum().backward()
    assert core.grad is None or torch.count_nonzero(core.grad) == 0
    assert readout.output.weight.grad is not None and readout.output.weight.grad.norm() > 0
    readout.zero_grad(set_to_none=True)
    core.grad = None
    procedure.grad = None
    with torch.no_grad():
        readout.output.weight.normal_(std=0.01)
    second = readout(core, core_mask, procedure, positions, frame_mask)
    second.square().mean().backward()
    assert core.grad is not None and core.grad.norm() > 0
    assert procedure.grad is not None and procedure.grad.norm() > 0
    assert readout.queries.grad is not None and readout.queries.grad.norm() > 0


def test_resume_uses_checkpoint_prefix_and_preserves_parent(tmp_path):
    path = tmp_path / "metrics.jsonl"
    complete = "".join(json.dumps({"update": step}) + "\n" for step in range(1, 5))
    path.write_text(complete)
    assert [json.loads(row)["update"] for row in _resume_prefix(tmp_path, 2)] == [1, 2]
    assert path.read_text() == complete
    path.write_text(json.dumps({"update": 1}) + "\n")
    with pytest.raises(ValueError, match="history prefix"):
        _resume_prefix(tmp_path, 2)
