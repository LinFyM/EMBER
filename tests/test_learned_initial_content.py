"""CPU contract for the fresh first-content arm and its staged bank scope."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.lora import identity_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json
from ember.pi05_eval.learned_initial_content import cases, full_cases
from ember.writer import model as writer_model
from ember.writer.learned_initial_content_contract import (
    bank_panel, evaluation_panel, selection_for, spec, validate_config,
)
from ember.writer.native_feature_change import intervene_packed
from ember.writer.training import _checkpoint_nodes, _segment_limit


ROOT = Path(__file__).resolve().parents[1]


class SyntheticNative(torch.nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.73))
        self.interaction_projection = torch.nn.Linear(1024, 256, bias=False)

    def forward(self, _policy, frames, condition_ids, language_tokens, *_args,
                frame_parallel_group=None):
        del frame_parallel_group
        content = frames.float().mean(dim=(1, 2, 3)) * self.scale
        batch = language_tokens.shape[0]
        q = torch.ones(batch, 2, 256) * self.scale
        evidence = content[:, None, None].expand(-1, 2, 256)
        horizon = content[:, None, None].expand(-1, 50, 1024)
        interaction = self.interaction_projection(horizon.mean(dim=1))
        return q, evidence, interaction, horizon, torch.ones(batch, 2, dtype=torch.bool)


def _writer(monkeypatch):
    monkeypatch.setattr(writer_model, "Pi05LanguageAxialEncoder", SyntheticNative)
    lora = load_pi05_lora_contract(ROOT / "configs/pi05_lora_v1.json")
    template = identity_lora_state(lora)
    backbone = SimpleNamespace(layers=range(18))
    writer = writer_model.CompleteLoRAWriter(
        writer_model.build_lora_tensor_specs(template), template_state=template,
        paligemma_model=backbone, expert_model=backbone,
        image_width=2048, expert_width=1024, program_width=256,
        text_meta_lora_rank=4, vl_meta_lora_rank=4, action_meta_lora_rank=4,
        patch_grounding_heads=8, max_frames_per_encoder_call=4,
        action_horizon=50, padded_action_dim=32, semantic_core_heads=8,
        semantic_core_blocks=2, frame_attention_initial_lambda=.05,
        procedure_heads=8, procedure_blocks=2, fusion_heads=8,
        factor_hidden_width=216, initialization_seed=7,
        activation_checkpointing=False, initial_content_only=True,
    ).eval()
    with torch.no_grad():
        for head in writer.factor_heads.values():
            head.network[-1].weight.normal_(std=.03)
    return writer


def test_first_native_read_matches_full_read_projection_and_summed_gradient(monkeypatch):
    torch.manual_seed(20260926)
    torch.set_num_threads(1)
    writer = _writer(monkeypatch)
    frames = torch.randn(5, 3, 4, 4, requires_grad=True)
    positions = torch.tensor([0, 5, 0, 5, 7])
    offsets = torch.tensor([0, 2, 5])
    tokens = torch.ones(2, 3, dtype=torch.long)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    span = mask.clone()
    args = (frames, positions, offsets, tokens, mask, span)

    optimized, native = writer.encode_task(None, *args, return_trace=True)
    result = writer.compile_encoded_task(*optimized[:5])[0]
    name = next(key for key in result if key.endswith("action_out_proj.lora_B.default.weight"))
    result[name].float().square().sum().backward()
    fast_gradient = writer.semantic_encoder.scale.grad.detach().clone()
    first_gradient = frames.grad.detach().clone()
    assert first_gradient[1].abs().max() == 0 and first_gradient[3:].abs().max() == 0
    assert first_gradient[[0, 2]].abs().sum() > 0 and fast_gradient.abs() > 0
    assert native["valid_frames"].tolist() == [[True, True, False], [True, True, True]]
    assert native["positions"].tolist() == [[0, 5, 0], [0, 5, 7]]

    writer.zero_grad(set_to_none=True)
    frames.grad.zero_()
    writer.initial_content_only = False
    _, full = writer.encode_task(None, *args, return_trace=True)
    e, i, h, clock, valid = writer._pack_video_program(
        full["frame_evidence"], full["interactions"], full["horizon"],
        positions, (0, 2, 5))
    e, i, h = intervene_packed(e, i, h, valid,
                              writer.semantic_encoder.interaction_projection, e=0, h=0)
    q = full["text_queries"]
    valid_tokens = torch.ones(2, 2, dtype=torch.bool)
    core, _ = writer.semantic_core(q, e, valid, valid_tokens)
    procedure = writer.procedure(i, h, e, clock, valid, valid_tokens)
    reference = writer.compile_encoded_task(core, valid_tokens, procedure, clock, valid)[0]
    torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-6)
    reference[name].float().square().sum().backward()
    torch.testing.assert_close(writer.semantic_encoder.scale.grad, fast_gradient, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(frames.grad, first_gradient, rtol=1e-5, atol=1e-6)


def test_training_bank_and_exact_stage_partition():
    study = spec()
    config = read_json(ROOT / "configs/learned_initial_content_causality_v1/train_S0.json")
    assert validate_config(config) == config
    args = SimpleNamespace(checkpoint_updates=None, mode="formal", stop_after_step=630,
                           support_slot_arm=None)
    assert _checkpoint_nodes(args, config) == (105, 210, 315, 420, 525, 630)
    assert _segment_limit(args, config) == 630
    changed = deepcopy(config)
    changed["optimization"]["lr"] *= 2
    with pytest.raises(ValueError, match="optimizer"):
        validate_config(changed)
    run = Path(study["outputs"]["planned_run_root"])
    selections = []
    full_count = 0
    for declared in study["evaluation"]["panels"]:
        selection = selection_for(declared["id"])
        selections.append(selection)
        if declared["model"] == "S0":
            assert bank_panel(config, selection,
                              checkpoint=run / "training/S0/checkpoints/macro_00000630",
                              output=run / "materialization" / declared["id"])["id"] == declared["id"]
        stages = ("pilot", "remaining") if declared["kind"] == "held_correct" else ("complete",)
        staged = [evaluation_panel(run / "evaluation" / stage / declared["id"]) for stage in stages]
        assert sum(len(cases(row)) for row in staged) == declared["rows"]
        assert len(set().union(*(cases(row) for row in staged))) == declared["rows"]
        full_count += sum(len(full_cases(row)) for row in staged)
    assert full_count == 24
    assert sum(x["rows"] for x in study["evaluation"]["panels"]) == 448
    assert all(len(x["video_pool"]) == 50 for x in selections if x["task_ids"] == study["evaluation"]["held_task_ids"])
    invalid = deepcopy(selection_for("S0_630_held_correct"))
    invalid["init_state_ids"] = list(range(50))
    with pytest.raises(ValueError, match="video map"):
        bank_panel(config, invalid, checkpoint=run / "training/S0/checkpoints/macro_00000630",
                   output=run / "materialization/S0_630_held_correct")
