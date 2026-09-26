"""CPU-only construction checks with synthetic Core/Procedure intermediates."""

from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from ember.lora import identity_lora_state
from ember.pi05_lora import load_pi05_lora_contract
from ember.writer import model as writer_model
from ember.writer.temporal import LanguageSemanticCore
from ember.writer.native_feature_change import (
    CELLS, compile_grid, intervene_packed, panel as native_panel, registered_selection,
    spec as native_spec,
)


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


def test_native_feature_broadcast_and_complete_compilation(monkeypatch):
    writer = _writer(monkeypatch)
    q, evidence, _, tokens, frames, positions = _synthetic_intermediates()
    horizon = torch.randn(2, 4, 50, 1024)
    projection = torch.nn.Linear(1024, 256)
    original_i = projection(horizon.float().mean(dim=-2))
    outputs = {}
    with torch.inference_mode():
        for cell in CELLS:
            e, h = int(cell[1]), int(cell[-1])
            cell_e, cell_i, cell_h = intervene_packed(
                evidence, original_i, horizon, frames, projection, e=e, h=h,
            )
            assert cell_h.shape == (2, 4, 50, 1024)
            if e:
                assert cell_e is evidence
            else:
                assert torch.equal(cell_e[0, 1], evidence[0, 0])
                assert torch.equal(cell_e[1, 3], evidence[1, 0])
                assert not cell_e[0, 2].count_nonzero()
            if h:
                assert cell_h is horizon and cell_i is original_i
            else:
                assert torch.equal(cell_h[1, 3], horizon[1, 0])
                assert not cell_h[0, 2].count_nonzero()
                torch.testing.assert_close(
                    cell_i[1, 3], projection(horizon[1, 0].mean(dim=0)),
                    rtol=1e-5, atol=1e-6,
                )
                assert not cell_i[0, 2].count_nonzero()
            core, attention = writer.semantic_core(q, cell_e, frames, tokens)
            procedure = writer.procedure(cell_i, cell_h, cell_e, positions, frames, tokens)
            outputs[cell], trace = writer.compile_encoded_task(
                core, tokens, procedure, positions, frames, return_trace=True,
            )
            assert len(outputs[cell]) == 76
            assert all(torch.isfinite(tensor).all() for tensor in outputs[cell].values())
            assert trace["output_slots"].shape[1] == writer.compiler.QUERY_COUNT
            assert trace["core_slots"].shape == trace["output_slots"].shape
            assert attention.shape[-2] == frames.shape[-1]
        canonical = writer.compile_encoded_task(
            *(
                writer.semantic_core(q, evidence, frames, tokens)[0],
                tokens,
                writer.procedure(original_i, horizon, evidence, positions, frames, tokens),
                positions,
                frames,
            ),
        )[0]
    torch.testing.assert_close(outputs["E1_H1"], canonical, rtol=0, atol=0)
    assert _maximum_difference(outputs["E0_H0"], outputs["E1_H1"]) > 1e-8


def test_native_feature_scope_is_exact_and_has_sixteen_full_cases():
    from ember.pi05_eval.native_feature_change import cases, full_cases
    from ember.writer.language_content_contract import evaluation_scope
    from ember.writer.materialization import planned_episodes

    study = native_spec()
    selection = registered_selection(study)
    assert selection["task_ids"] == study["evaluation"]["task_ids"]
    assert selection["init_state_ids"] == list(range(10))
    for task in selection["task_ids"]:
        demos = [row["teacher_demo_indices"][0] for row in planned_episodes(selection, task)]
        assert len(set(demos)) == 10
    assert cases("pilot").isdisjoint(cases("remaining"))
    assert len(cases("pilot") | cases("remaining")) == 80
    assert len(full_cases("pilot")) == 1 and len(full_cases("remaining")) == 3
    for cell in CELLS:
        for stage, rows in (("pilot", 2), ("remaining", 78)):
            path = Path(study["outputs"]["planned_run_root"]) / "evaluation" / stage / cell
            registered, _, panel = evaluation_scope(path)
            assert registered == study and panel == native_panel(cell, stage)
            assert panel["rows"] == rows
    with pytest.raises(ValueError):
        native_panel("E1_H2", "pilot")


def test_native_feature_grid_reads_one_synthetic_condition_and_preserves_canonical_11(monkeypatch):
    class SyntheticNative(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.interaction_projection = torch.nn.Linear(1024, 256)
            self.calls = 0

        def forward(self, _policy, frames, condition_ids, language_tokens,
                    _language_mask, task_span_mask, *, frame_parallel_group=None):
            self.calls += 1
            assert frame_parallel_group is None
            q = language_tokens.float()[..., None].expand(-1, -1, 256) / 20
            source = frames.float().mean(dim=(1, 2, 3))
            e = q[condition_ids] + source[:, None, None]
            h = source[:, None, None].expand(-1, 50, 1024)
            h = h + torch.arange(50).float()[None, :, None] / 100
            i = self.interaction_projection(h.mean(dim=1))
            return q, e, i, h, task_span_mask

    writer = _writer(monkeypatch)
    encoder = SyntheticNative()
    writer.semantic_encoder = encoder
    frames = torch.arange(4 * 3 * 4 * 4).reshape(4, 3, 4, 4).float() / 1000
    condition = (
        frames, torch.tensor([0, 5, 10, 13]), torch.tensor([0, 4]),
        torch.tensor([[1, 2, 0]]), torch.tensor([[True, True, False]]),
        torch.tensor([[True, True, False]]),
    )
    with torch.inference_mode():
        grid = compile_grid(writer, None, condition)
        assert encoder.calls == 1
        canonical = writer(*condition, policy=None)
        assert encoder.calls == 2
    torch.testing.assert_close(grid["E1_H1"][0], canonical, atol=0, rtol=0)
    assert _maximum_difference(grid["E0_H0"][0], grid["E1_H1"][0]) > 1e-8
    assert torch.equal(grid["E1_H0"][1]["positions"], condition[1][None])
    assert len(grid) == 4


def test_native_feature_evaluator_stages_select_only_registered_cases():
    from ember.pi05_eval.native_feature_change import cases, select_tasks
    from ember.pi05_eval_queue import EvaluationTask
    from ember.pi05_target_data import SUITE_ORDER

    study = native_spec()
    root = Path(study["outputs"]["planned_run_root"])
    repo = Path(__file__).resolve().parents[1]
    installed = tuple(EvaluationTask(SUITE_ORDER[task // 10], task % 10, 300,
                                     tuple(range(10))) for task in study["evaluation"]["task_ids"])
    for stage in ("pilot", "remaining"):
        args = SimpleNamespace(
            output_dir=root / "evaluation" / stage / "E0_H0",
            role="development_train", mode="screen", state_count=10,
            init_state_ids=None, config=repo / "configs/libero_24_8_8_coverage_v1/evaluation.json",
            static_task_lora_manifest=root / "materialization" / "E0_H0" / "manifest.json",
            task_subset_selection=None, trajectory_capture_selection=None,
            occupancy_capture_selection=None, frozen_replay_registration=None,
            frozen_prefix_panel=None, approach_channel_panel=None,
            readout_realization_panel=None, native_reader_transfer_cell=None,
            support_slot_model=None, source_sft_checkpoint=None, task_expert_config=None,
            capture_stage_predicates=False, exploration_sigma=False,
        )
        chosen, capture, predicates = select_tasks(args, installed, repo)
        actual = {(task.suite, task.task_id, state) for task in chosen for state in task.init_state_ids}
        assert actual == cases(stage)
        assert len(capture["full_conditions"]) == (1 if stage == "pilot" else 3)
        assert predicates["full_conditions_only"] is False
        args.static_task_lora_manifest = root / "materialization" / "E1_H1" / "manifest.json"
        with pytest.raises(Exception, match="native-feature evaluator args"):
            select_tasks(args, installed, repo)


def test_native_feature_bank_rejects_wrong_cell_commit_and_map():
    from copy import deepcopy
    from ember.pi05_source_checkpoint import read_json
    from ember.writer.language_content_contract import validate_evaluation_bank

    study = native_spec()
    root = Path(study["outputs"]["planned_run_root"])
    source = study["frozen_input"]
    run = read_json(Path(source["checkpoint"]).parent.parent / "run_contract.json")
    panel = native_panel("E1_H0", "pilot")
    path = root / "materialization" / panel["id"] / "manifest.json"
    manifest = {
        "selection": registered_selection(study),
        "native_feature_change": {
            "study_id": study["study_id"], "cell": panel["id"],
            "spec_path": "configs/native_feature_change_causality_v1/experiment_spec.json",
            "native_read_conditions": 80, "feature_index": str(root / "features" / "index.json"),
            "source_training_commit": source["training_commit"],
        },
        "materialization_git": {"commit": "future_E"},
        "writer_checkpoint": {"path": source["checkpoint"], "macro": 630},
        "arm": "correct", "conditions": [{} for _ in range(80)],
    }
    assert validate_evaluation_bank(panel, path, manifest, run, "future_E") is False
    for altered in (
        {"native_feature_change": {**manifest["native_feature_change"], "cell": "E0_H0"}},
        {"materialization_git": {"commit": "old_E"}},
        {"selection": {**manifest["selection"], "init_state_ids": list(range(1, 11))}},
        {"conditions": manifest["conditions"][:-1]},
    ):
        with pytest.raises(ValueError, match="native-feature bank"):
            validate_evaluation_bank(panel, path, {**manifest, **deepcopy(altered)}, run, "future_E")
