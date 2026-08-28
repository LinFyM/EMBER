import json

import torch
from safetensors.torch import save_file

from ember.ecp.behavior.codes import (
    BEHAVIOR_CODE_SCHEMA,
    BehaviorCodeDecoder,
    behavior_alignment_loss,
    load_behavior_code_authority,
    load_program_model_initialization,
)
from ember.ecp.natural_program import NaturalProgram


def _program(*, process: torch.Tensor, language: torch.Tensor) -> NaturalProgram:
    batch, events, owners, width = process.shape
    return NaturalProgram(
        p_lang=language,
        p_scene=torch.randn(batch, owners, width),
        p_process=process,
        rho=torch.rand(batch, events).clamp_min(0.1),
        tau=torch.rand(batch, events, 2),
        sigma=torch.rand(batch, events, owners, width).clamp_min(0.1),
    )


def test_behavior_decoder_uses_only_event_bearing_program_fields():
    torch.manual_seed(7)
    process = torch.randn(2, 4, 38, 8, requires_grad=True)
    language = torch.randn(2, 38, 8)
    program = _program(process=process, language=language)
    decoder = BehaviorCodeDecoder(
        program_width=8,
        hidden_width=12,
        event_slots=4,
        selected_targets=(0, 16, 34, 1, 17, 35, 36, 37),
        family_ids=(0, 0, 0, 1, 1, 1, 2, 3),
        family_count=4,
        dimension=5,
    )
    prediction = decoder(program)
    changed_static = NaturalProgram(
        p_lang=program.p_lang + 100,
        p_scene=program.p_scene - 100,
        p_process=program.p_process,
        rho=program.rho,
        tau=program.tau,
        sigma=program.sigma,
    )
    assert prediction.shape == (2, 8, 5)
    assert torch.equal(prediction, decoder(changed_static))
    target = torch.randn(8, 5)
    loss = behavior_alignment_loss(prediction[:1], prediction[1:2], target)
    loss.backward()
    assert process.grad is not None
    assert torch.isfinite(process.grad).all()
    assert process.grad.abs().sum() > 0


def test_behavior_authority_loads_and_decodes(tmp_path):
    first = tmp_path / "factors_a"
    second = tmp_path / "factors_b"
    first.mkdir()
    second.mkdir()
    fit = tuple(range(75))
    held = tuple(range(75, 95))
    coordinates = torch.randn(95, 8, 16)
    mean = torch.randn(8, 16)
    scale = torch.rand(8, 16).clamp_min(0.1)
    tensors = {
        "task_ids": torch.arange(95),
        "fit_task_ids": torch.tensor(fit),
        "held_task_ids": torch.tensor(held),
        "selected_targets": torch.tensor([0, 16, 34, 1, 17, 35, 36, 37]),
        "coordinates": coordinates,
        "mean": mean,
        "scale": scale,
        "eigenvectors": torch.randn(8, 75, 16),
        "eigenvalues": torch.rand(8, 16).clamp_min(0.1),
        "norms": torch.rand(8, 95).clamp_min(0.1),
        "train_sqrt_weights": torch.full((75,), 75**-0.5),
    }
    tensor_path = tmp_path / "codes.safetensors"
    save_file(tensors, tensor_path)
    manifest = {
        "schema_version": BEHAVIOR_CODE_SCHEMA,
        "status": "complete",
        "tensor_file": tensor_path.name,
        "tensor_bytes": tensor_path.stat().st_size,
        "dimension": 16,
        "factor_roots": [first.name, second.name],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    authority = load_behavior_code_authority(
        manifest_path, asset_root=tmp_path, device=torch.device("cpu")
    )
    standardized = authority.target(90, standardized=True)
    assert torch.allclose(authority.decode(standardized), coordinates[90])
    assert authority.fit_task_ids == fit
    assert authority.held_task_ids == held


def test_model_only_initialization_allows_only_fresh_behavior_prefix(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(3, 4)
            self.behavior_decoder = torch.nn.Linear(4, 2)

    source = Model()
    target = Model()
    checkpoint = tmp_path / "checkpoints" / "macro_00000020"
    checkpoint.mkdir(parents=True)
    source_state = {
        name: value
        for name, value in source.state_dict().items()
        if not name.startswith("behavior_decoder.")
    }
    weights = checkpoint / "ecp.safetensors"
    save_file(source_state, weights)
    manifest = {
        "schema_version": "ember_ecp_checkpoint_v1",
        "stage": "g2_natural_program",
        "next_macro": 20,
        "run_contract_schema": "ember_ecp_natural_program_g2_run_v2",
        "world_size": 4,
        "files": {"ecp.safetensors": {"bytes": weights.stat().st_size}},
    }
    (checkpoint / "checkpoint_manifest.json").write_text(json.dumps(manifest))
    fresh = target.behavior_decoder.weight.detach().clone()
    report = load_program_model_initialization(
        target,
        checkpoint,
        device=torch.device("cpu"),
        allowed_new_prefix="behavior_decoder.",
        expected_macro=20,
    )
    assert torch.equal(target.base.weight, source.base.weight)
    assert torch.equal(target.behavior_decoder.weight, fresh)
    assert report["optimizer_loaded"] is False
