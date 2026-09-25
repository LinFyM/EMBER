"""CPU-only construction checks with synthetic Core/Procedure intermediates."""

from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from ember.lora import identity_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.writer import model as writer_model
from ember.writer.temporal import LanguageSemanticCore


class SyntheticIntermediateEncoder(torch.nn.Module):
    """Pass a synthetic text memory through the real language-only owner."""

    def __init__(self, **_kwargs):
        super().__init__()

    def encode_text_only(self, _policy, text, _language_mask, task_span_mask):
        return text, task_span_mask


@pytest.fixture(autouse=True)
def cpu_only():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(20260926)
    yield
    torch.set_num_threads(previous)


def _writer(monkeypatch):
    monkeypatch.setattr(writer_model, "Pi05LanguageAxialEncoder", SyntheticIntermediateEncoder)
    contract = load_pi05_lora_contract(
        Path(__file__).resolve().parents[1] / "configs/pi05_lora_v1.json"
    )
    template = identity_lora_state(contract)
    backbone = SimpleNamespace(layers=range(18))
    writer = writer_model.CompleteLoRAWriter(
        writer_model.build_lora_tensor_specs(template),
        template_state=template,
        paligemma_model=backbone,
        expert_model=backbone,
        image_width=2048,
        expert_width=1024,
        program_width=256,
        text_meta_lora_rank=4,
        vl_meta_lora_rank=4,
        action_meta_lora_rank=4,
        patch_grounding_heads=8,
        max_frames_per_encoder_call=4,
        action_horizon=50,
        padded_action_dim=32,
        semantic_core_heads=8,
        semantic_core_blocks=2,
        frame_attention_initial_lambda=.05,
        procedure_heads=8,
        procedure_blocks=2,
        fusion_heads=8,
        factor_hidden_width=216,
        initialization_seed=7,
        activation_checkpointing=False,
        language_content_path=True,
    ).eval()
    with torch.no_grad():
        for head in writer.factor_heads.values():
            head.network[-1].weight.normal_(std=.03)
        writer.compiler.modulation.weight.normal_(std=.02)
    return writer


def _synthetic_intermediates():
    # No policy, frames, actions, or teacher data: only explicit CPU memories.
    text = torch.randn(2, 3, 256)
    evidence = torch.randn(2, 4, 3, 256)
    procedure = torch.randn(2, 4, 256)
    tokens = torch.tensor([[True, True, False], [True, True, True]])
    frames = torch.tensor([[True, True, False, False], [True] * 4])
    positions = torch.tensor([[0, 5, 0, 0], [0, 5, 10, 15]])
    return text, evidence, procedure, tokens, frames, positions


def _compiled(writer, text, evidence, procedure, tokens, frames, positions):
    core, _ = writer.semantic_core(text, evidence, frames, tokens)
    return writer.compile_encoded_task(core, tokens, procedure, positions, frames)[0]


def _maximum_difference(left, right):
    return max((left[key] - right[key]).abs().max().item() for key in left)


def test_candidate_zero_preserves_canonical_parameters_rng_and_full_output(monkeypatch):
    writer = _writer(monkeypatch)
    text, evidence, procedure, tokens, frames, positions = _synthetic_intermediates()
    candidate = writer.semantic_core
    canonical = LanguageSemanticCore(
        width=256, heads=8, blocks=2, frame_attention_initial_lambda=.05,
    )
    state = {key: value for key, value in candidate.state_dict().items()
             if key != "language_content_scale"}
    canonical.load_state_dict(state, strict=True)
    assert candidate.language_content_scale.item() == 0
    assert "language_content_scale" not in canonical.state_dict()
    initial_rng = torch.get_rng_state()
    LanguageSemanticCore(width=256, heads=8, blocks=2,
                         frame_attention_initial_lambda=.05)
    after_canonical = torch.get_rng_state()
    torch.set_rng_state(initial_rng)
    LanguageSemanticCore(width=256, heads=8, blocks=2,
                         frame_attention_initial_lambda=.05,
                         language_content_path=True)
    assert torch.equal(torch.get_rng_state(), after_canonical)
    with torch.no_grad():
        original_core, _ = canonical(text, evidence, frames, tokens)
        original = writer.compile_encoded_task(
            original_core, tokens, procedure, positions, frames,
        )[0]
        observed = _compiled(writer, text, evidence, procedure, tokens, frames, positions)
    assert len(observed) == 76
    torch.testing.assert_close(observed, original, rtol=1e-5, atol=1e-6)
    assert any(value.count_nonzero() for key, value in observed.items()
               if key.endswith(".lora_B.default.weight"))
    print(f"a0_full76_maxabs={_maximum_difference(observed, original):.9g}")


def test_b_containment_video_path_padding_and_scale_gradient(monkeypatch):
    writer = _writer(monkeypatch)
    text, evidence, procedure, tokens, frames, positions = _synthetic_intermediates()
    with torch.no_grad():
        original_frame_output = writer.semantic_core.frame_attention.output.weight.clone()
        writer.semantic_core.language_content_scale.fill_(1)
        writer.semantic_core.frame_attention.output.weight.zero_()
        writer.compiler.modulation.weight.zero_()
        b_output = writer.compile_language_task(None, text, tokens, tokens)[0]
        c_as_b = _compiled(writer, text, evidence, procedure, tokens, frames, positions)
        torch.testing.assert_close(c_as_b, b_output, rtol=1e-5, atol=1e-6)
        b_head_maxabs = max(value.abs().max().item() for key, value in b_output.items()
                            if key.endswith(".lora_B.default.weight"))
        assert b_head_maxabs > 1e-6
        b_error = _maximum_difference(c_as_b, b_output)

        writer.semantic_core.frame_attention.output.weight.copy_(original_frame_output)
        writer.semantic_core.language_content_scale.fill_(.3)
        writer.compiler.modulation.weight.normal_(std=.02)
        video_output = _compiled(writer, text, evidence, procedure, tokens, frames, positions)
        changed_evidence = evidence.clone()
        changed_evidence[0, 0, 0] += 2
        changed_output = _compiled(writer, text, changed_evidence, procedure,
                                   tokens, frames, positions)
        video_difference = _maximum_difference(video_output, changed_output)
        assert video_difference > 1e-6
        single = _compiled(writer, text[:1, :2], evidence[:1, :2, :2],
                           procedure[:1, :2], tokens[:1, :2], frames[:1, :2],
                           positions[:1, :2])
        padding_difference = max((video_output[key][0] - single[key]).abs().max().item()
                                 for key in single)
        assert padding_difference < 1e-5

    writer.zero_grad(set_to_none=True)
    writer.semantic_core.language_content_scale.data.zero_()
    captured = []

    def retain_content(_module, inputs):
        inputs[0].retain_grad()
        captured.append(inputs[0])

    hook = writer.semantic_core.blocks[0].register_forward_pre_hook(retain_content)
    output = _compiled(writer, text, evidence, procedure, tokens, frames, positions)
    hook.remove()
    name = next(key for key in output if key.endswith("action_out_proj.lora_B.default.weight"))
    target = torch.linspace(-1, 1, output[name].numel()).reshape_as(output[name])
    loss = (output[name] - target).square().mean()
    loss.backward()
    observed_gradient = writer.semantic_core.language_content_scale.grad.item()
    expected_gradient = (captured[0].grad * text).sum().item()
    assert abs(observed_gradient) > 1e-8
    assert observed_gradient == pytest.approx(expected_gradient, rel=1e-5, abs=1e-7)
    print(f"b_full76_maxabs={b_error:.9g} b_nonzero_head_maxabs={b_head_maxabs:.9g} "
          f"video_full76_maxabs={video_difference:.9g} "
          f"padding_full76_maxabs={padding_difference:.9g} "
          f"a_grad={observed_gradient:.9g} dot_grad_q={expected_gradient:.9g}")
