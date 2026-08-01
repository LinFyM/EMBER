from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

import ember.writer.ucp_analysis_runtime as analysis_runtime
from ember.writer.ucp_analysis import (
    CONDITIONS,
    STAGES,
    build_initial_program,
    compile_with_target_identity_permutation,
    effective_ba_error,
    effective_variance,
    effective_metrics,
    fixed_stream_counterfactual,
    lora_geometry,
    pack_flat,
    rank_gauge_permute,
    reader_attention,
    reader_attention_summary,
    relative_metrics,
    resample_intervals,
    type_ablation,
    validate_analysis_provenance,
    validate_canonical_program_parity,
    variance_metrics,
)
from ember.writer.ucp_analysis_runtime import fixed_policy_query
from ember.writer.ucp_analysis_summary import (
    aggregate_numeric_records,
    summarize_records,
    validate_finite_tree,
    validate_rank_payloads,
)
from ember.writer.data import WriterTaskAuthority
from ember.writer.model import LoraTensorSpec, WriterModelError
from ember.writer.program_compiler import FactorHead, TargetRankProgramReader
from ember.writer.semantic_program import UnifiedCausalProgram
from ember.writer.ucp_geometry import (
    component_coordinate_geometry,
    effective_ba_spectrum,
)


def test_initial_program_uses_outgoing_grounded_change() -> None:
    assert CONDITIONS == (
        "correct", "same_task_other", "cross_suite_wrong", "shuffled", "reversed",
    )
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


def test_target_identity_permutation_keeps_real_decode_slots() -> None:
    torch.manual_seed(12)
    compiler = TargetRankProgramReader(
        width=32, heads=4, target_count=38, rank=16, initialization_seed=7,
    )
    writer = SimpleNamespace(compiler=compiler)
    program = torch.randn(1, 3, 5, 32)
    endpoints = torch.tensor([[5, 10, 15]])
    intervals = torch.ones(1, 3, dtype=torch.bool)
    semantics = torch.ones(1, 5, dtype=torch.bool)
    canonical = compiler(program, endpoints, intervals, semantics)
    identity_before = compiler.target_identity.detach().clone()
    permutation = torch.roll(torch.arange(38), -1)
    permuted = compile_with_target_identity_permutation(
        writer, program, endpoints, intervals, semantics, permutation,
    )

    assert canonical.shape == permuted.shape == (1, 38, 16, 32)
    assert not torch.equal(canonical, permuted)
    assert torch.equal(identity_before, compiler.target_identity)
    with pytest.raises(WriterModelError, match="bijective"):
        compile_with_target_identity_permutation(
            writer, program, endpoints, intervals, semantics,
            torch.zeros(38, dtype=torch.long),
        )


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
    assert geometry["effective_lora_norm"] == pytest.approx(
        effective_metrics(writer, right, right)["reference_l2"], rel=1e-6,
    )
    assert 1 <= geometry["stable_rank_mean"] <= 16
    assert sum(geometry["rank_coordinate_component_gram"][
        "coordinate_energy_participation"
    ]) == pytest.approx(1.0)
    assert sum(geometry["q_v_action_energy_ratio"].values()) == pytest.approx(1.0)
    assert all(value > 0 for value in geometry["q_v_action_energy_ratio"].values())
    for kind in ("q", "v", "action"):
        spectrum = geometry["per_kind_effective_ba_spectrum"][kind]
        component = geometry["per_kind_rank_coordinate_component_gram"][kind]
        assert 1 <= spectrum["stable_rank_mean"] <= 16
        assert spectrum["rank90_mean"] <= spectrum["rank99_mean"] <= 16
        assert sum(component["coordinate_energy_participation"]) == pytest.approx(1.0)
        assert component["layer_energy_cv"] >= 0
        assert kind in geometry["per_layer_energy_cv_by_kind"]
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


def test_effective_ba_spectrum_is_not_component_coordinate_gram_rank() -> None:
    a = torch.eye(2)
    b_rank_one = torch.tensor([[1., 1.], [0., 0.]])
    component_gram = (b_rank_one.T @ b_rank_one) * (a @ a.T)
    component = component_coordinate_geometry(
        component_gram, b_rank_one.T @ b_rank_one, [2.0],
    )
    rank_one = effective_ba_spectrum(a, b_rank_one)
    rank_two = effective_ba_spectrum(a, torch.eye(2))

    assert torch.linalg.matrix_rank(component_gram) == 2
    assert rank_one["stable_rank"] == pytest.approx(1.0)
    assert rank_one["effective_ba_energy"] == pytest.approx(
        float((b_rank_one @ a).square().sum())
    )
    assert rank_one["rank99"] == 1
    assert rank_two["stable_rank"] == pytest.approx(2.0)
    assert "stable_rank" not in component


def test_canonical_semantic_program_matches_manual_reconstruction() -> None:
    torch.manual_seed(21)
    owner = UnifiedCausalProgram(
        width=32, heads=4, blocks=2, initialization_seed=7,
    ).eval()
    compiler = TargetRankProgramReader(
        width=32, heads=4, target_count=38, rank=16, initialization_seed=9,
    ).eval()
    absolute = torch.randn(2, 5, 3, 32)
    grounded = torch.randn_like(absolute)
    action = torch.randn(2, 5, 32)
    positions = torch.tensor([[0, 5, 10, 15, 20], [0, 5, 10, 15, 0]])
    valid_frames = torch.tensor([
        [True, True, True, True, True], [True, True, True, True, False],
    ])
    valid_tokens = torch.tensor([[True, True, True], [True, False, True]])
    with torch.inference_mode():
        canonical = owner(
            absolute, grounded, action, positions, valid_frames, valid_tokens,
        )
        canonical_coordinates = compiler(*canonical)
        value, endpoints, intervals, semantics = build_initial_program(
            absolute, grounded, action, positions, valid_frames, valid_tokens,
        )
        for block in owner.blocks:
            value = block(value, endpoints, intervals, semantics)
        coordinates = compiler(value, endpoints, intervals, semantics)
    proof = analysis_runtime.canonical_program_parity(
        SimpleNamespace(semantic_program=owner, compiler=compiler),
        absolute, grounded, action, positions, valid_frames, valid_tokens,
        (value, endpoints, intervals, semantics, coordinates),
    )
    assert proof["final_program"]["relative_l2"] == 0
    assert proof["endpoint_mismatch_count"] == 0
    with pytest.raises(WriterModelError, match="semantic_program"):
        validate_canonical_program_parity(
            (*canonical, canonical_coordinates),
            (value + .1, endpoints, intervals, semantics, coordinates),
        )


def test_fixed_stream_counterfactual_preserves_dynamic_change_contract() -> None:
    initial = torch.zeros(3, 5, 5, 2)
    valid = torch.tensor([
        [True, True, True, True, False],
        [True, True, False, False, False],
        [True, True, True, True, False],
    ])
    initial[0, :4, :2] = 1
    initial[0, :4, 2:] = 10
    initial[1, :2, :2] = 2
    initial[1, :2, 2] = torch.tensor([[3., 4.], [5., 6.]])
    initial[1, :2, 3:] = torch.tensor([
        [[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]],
    ])
    initial[2, :4] = torch.arange(4 * 5 * 2).reshape(4, 5, 2)

    fixed_x = fixed_stream_counterfactual(initial, valid, 1, fixed="x")
    fixed_dynamic = fixed_stream_counterfactual(initial, valid, 1, fixed="a_d")
    torch.testing.assert_close(fixed_x[0, :4, :2], initial[0, :4, :2])
    torch.testing.assert_close(
        fixed_x[0, :4, 3:].sum(0), initial[1, :2, 3:].sum(0),
    )
    torch.testing.assert_close(fixed_dynamic[0, :4, 2:], initial[0, :4, 2:])
    same_length = fixed_stream_counterfactual(initial, valid, 2, fixed="x")
    assert torch.equal(same_length[0, :4, 2:], initial[2, :4, 2:])


def test_rank_gauge_changes_raw_factors_but_preserves_complete_function() -> None:
    writer, state, _ = _fake_writer_and_states()
    permutation = torch.roll(torch.arange(16), -1)
    permuted, per_target = rank_gauge_permute(writer, state, permutation)

    assert mapping_metrics_for_test(state, permuted, "a") > 0
    assert mapping_metrics_for_test(state, permuted, "b") > 0
    assert len(per_target) == 6
    assert effective_ba_error(writer, state, permuted)["relative_l2"] < 1e-6
    query = torch.arange(3, dtype=torch.float32)
    for names in _pairs_for_test(writer).values():
        left = state[names["b"]] @ (state[names["a"]] @ query)
        right = permuted[names["b"]] @ (permuted[names["a"]] @ query)
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)


def _pairs_for_test(writer: SimpleNamespace) -> dict[str, dict[str, str]]:
    pairs: dict[str, dict[str, str]] = {}
    for spec in writer.tensor_specs:
        pairs.setdefault(spec.module, {})[
            "a" if spec.factor_index == 0 else "b"
        ] = spec.name
    return pairs


def mapping_metrics_for_test(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], factor: str,
) -> float:
    names = [name for name in left if f"lora_{factor.upper()}" in name]
    return float(sum((left[name] - right[name]).square().sum() for name in names))


def _summary_row(reference: int, scale: float = 1.0) -> dict[str, object]:
    metric = {
        "relative_l2": .2 * scale, "cosine": .8,
        "reference_rms": .3, "candidate_rms": .4,
    }
    stages = (*STAGES, "factor_output", "public_a", "public_b", "effective_ba", "policy_action")
    geometry = {
        "stable_rank_mean": 1.2,
        "rank_coordinate_component_gram": {
            "coordinate_energy_participation": [.25] * 4,
        },
        "q_v_action_energy_ratio": {"q": .5, "v": .3, "action": .2},
        "cross_layer_effective_ba_cosine": {"q": .9, "v": .8},
        "per_kind_effective_ba_spectrum": {
            kind: {"stable_rank_mean": 1.1, "rank90_mean": 1.0}
            for kind in ("q", "v", "action")
        },
        "per_kind_rank_coordinate_component_gram": {
            kind: {"layer_energy_cv": .2,
                   "coordinate_energy_participation": [.25] * 4}
            for kind in ("q", "v", "action")
        },
    }
    variant = {"policy_action": metric, "effective_ba": metric, "coordinates": metric}
    return {
        "global_task_id": 40, "suite": "libero_spatial", "task_id": 0,
        "reference_ordinal": reference,
        "comparisons_to_correct": {
            condition: {stage: dict(metric) for stage in stages}
            for condition in CONDITIONS
        },
        "reader_attention": {condition: {"x_mass": .4} for condition in CONDITIONS},
        "coordinate_routing": {
            condition: {"target_centered_energy_ratio": .1}
            for condition in CONDITIONS
        },
        "lora_geometry": {condition: geometry for condition in CONDITIONS},
        "canonical_program_parity": {
            "final_program": metric, "coordinates": metric,
            "endpoint_mismatch_count": 0,
            "valid_interval_mismatch_count": 0,
            "valid_semantic_mismatch_count": 0,
        },
        "same_task_video_variance": {
            "effective_ba": {"estimable": True, "centered_variance": .2},
            "fixed_policy_action": {"estimable": True, "centered_variance": .3},
        },
        "type_ablations": {"x_only": {"relative_to_full": variant}},
        "fixed_x_vary_a_d": {"correct": variant},
        "fixed_a_d_vary_x": {"correct": variant},
        "dynamic_scale": {"0.5": {"relative_to_scale1": variant}},
        "variant_recompute": variant,
        "target_identity_permutation": {"relative_to_canonical": variant},
        "rank_gauge_permutation": {
            "public_a": metric, "effective_ba_numerical_error": metric,
        },
    }


def test_recursive_summary_retains_nested_geometry_and_all_counterfactuals() -> None:
    direct = aggregate_numeric_records([
        {"nested": {"value": 1.0}, "vector": [1.0, 3.0]},
        {"nested": {"value": 3.0}, "vector": [3.0, 5.0]},
    ])
    assert direct["nested"]["value"]["mean"] == pytest.approx(2.0)
    assert direct["vector"]["mean"] == pytest.approx([2.0, 4.0])
    with pytest.raises(WriterModelError, match="key set"):
        aggregate_numeric_records([{"value": 1.0}, {"value": 2.0, "lost": 3.0}])
    with pytest.raises(WriterModelError, match="vector changed length"):
        aggregate_numeric_records([[1.0], [1.0, 2.0]])

    summary = summarize_records([_summary_row(0), _summary_row(1, 2.0)])
    action = summary["conditions"]["same_task_other"]["policy_action"]
    assert action["reference_rms"]["mean"] == pytest.approx(.3)
    assert summary["coordinate_routing"]["correct"][
        "target_centered_energy_ratio"
    ]["mean"] == pytest.approx(.1)
    q_spectrum = summary["lora_geometry"]["correct"][
        "per_kind_effective_ba_spectrum"
    ]["q"]
    q_component = summary["lora_geometry"]["correct"][
        "per_kind_rank_coordinate_component_gram"
    ]["q"]
    assert q_spectrum["stable_rank_mean"]["mean"] == pytest.approx(1.1)
    assert q_component["coordinate_energy_participation"]["mean"] == pytest.approx([.25] * 4)
    assert set(summary["counterfactuals"]) == {
        "type_ablations", "fixed_x_vary_a_d", "fixed_a_d_vary_x",
        "dynamic_scale", "variant_recompute", "target_identity_permutation",
        "rank_gauge_permutation",
    }
    task = summary["per_task"]["libero_spatial:task_00"]
    assert summary["canonical_program_parity"]["endpoint_mismatch_count"]["max"] == 0
    assert summary["canonical_program_parity"]["coordinates"][
        "relative_l2"
    ]["mean"] == pytest.approx(.3)
    assert task["canonical_program_parity"]["final_program"][
        "relative_l2"
    ]["mean"] == pytest.approx(.3)
    assert task["same_task_video_variance"]["effective_ba"]["estimable"] is True
    assert task["counterfactuals"]["fixed_x_vary_a_d"]["correct"][
        "policy_action"
    ]["candidate_rms"]["mean"] == pytest.approx(.4)


def test_variance_marks_single_reference_unestimable() -> None:
    writer, state, _ = _fake_writer_and_states()
    raw = variance_metrics([torch.ones(4)])
    effective = effective_variance(writer, [state])
    assert raw["estimable"] is False and raw["centered_variance"] is None
    assert effective["estimable"] is False
    assert effective["centered_variance_over_sample_energy"] is None
    assert variance_metrics([torch.ones(4), torch.zeros(4)])["estimable"] is True
    row = _summary_row(0)
    for value in row["same_task_video_variance"].values():
        value["estimable"] = False
        value["centered_variance"] = None
    task = summarize_records([row])["per_task"]["libero_spatial:task_00"]
    assert task["same_task_video_variance"]["effective_ba"]["estimable"] is False


def test_rank_payload_cartesian_and_recursive_finite_validation() -> None:
    payloads = [{"rank": rank, "rows": []} for rank in range(4)]
    for task in range(8):
        for reference in range(2):
            payloads[task % 4]["rows"].append({
                "global_task_id": 40 + task, "suite": f"suite_{task // 2}",
                "task_id": task, "reference_ordinal": reference, "metric": 1.0,
            })
    rows = validate_rank_payloads(payloads, 2)
    assert len(rows) == 16
    bad_rank = [dict(payload) for payload in payloads]
    bad_rank[0] = {**bad_rank[0], "rank": 3}
    with pytest.raises(WriterModelError, match="rank payload"):
        validate_rank_payloads(bad_rank, 2)
    missing = [{**payload, "rows": list(payload["rows"])} for payload in payloads]
    missing[0]["rows"].pop()
    with pytest.raises(WriterModelError, match="8-task x references"):
        validate_rank_payloads(missing, 2)
    with pytest.raises(WriterModelError, match="non-finite"):
        validate_finite_tree({"nested": [1.0, float("nan")]})


def _load_analysis_script() -> object:
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_as_writer_ucp.py"
    spec = importlib.util.spec_from_file_location("test_analyze_as_writer_ucp", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _variant_writer() -> SimpleNamespace:
    writer = SimpleNamespace(
        semantic_program=UnifiedCausalProgram(
            width=32, heads=4, blocks=2, initialization_seed=4,
        ),
        compiler=TargetRankProgramReader(
            width=32, heads=4, target_count=38, rank=16,
            initialization_seed=7,
        ),
        PUBLIC_LORA_RANK=16,
    )
    modules = (
        "model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj", "model.layers.1.self_attn.v_proj",
        "model.action_in_proj", "model.action_out_proj",
    )
    specs, decoding, templates, heads = [], {}, {}, {}
    for target, module in enumerate(modules):
        kind = "q" if module.endswith("q_proj") else "v" if module.endswith("v_proj") else "action"
        for factor, width, transpose in (("a", 3, False), ("b", 2, True)):
            name = f"{module}.lora_{factor.upper()}.default.weight"
            key = f"{kind}_{factor}"
            specs.append(LoraTensorSpec(
                name, module, target, 0 if factor == "a" else 1,
                16, width, transpose,
            ))
            decoding[name] = (key, target)
            templates[name] = f"template_{len(templates)}"
            setattr(writer, templates[name],
                    torch.randn(16, width) if factor == "a" else torch.zeros(width, 16))
            if key not in heads:
                heads[key] = FactorHead(32, 32, width)
    writer.tensor_specs = tuple(specs)
    writer._decoding = decoding
    writer._template_buffers = templates
    writer.factor_heads = torch.nn.ModuleDict(heads)
    for head in writer.factor_heads.values():
        torch.nn.init.normal_(head.network[-1].weight, std=.02)
    return writer


def test_real_variant_program_compiler_factor_forward_uses_inference_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_analysis_script()
    writer = _variant_writer()
    observed = []
    hook = writer.compiler.reader.key.register_forward_pre_hook(
        lambda _module, _values: observed.append(torch.is_inference_mode_enabled())
    )
    monkeypatch.setattr(script, "validate_lora_state", lambda *_args: None)
    monkeypatch.setattr(
        script, "policy_action",
        lambda **_kwargs: torch.arange(14, dtype=torch.float32).reshape(2, 7),
    )
    initial = torch.randn(1, 3, 5, 32, dtype=torch.bfloat16)
    shared = {
        "writer": writer, "policy": SimpleNamespace(), "processor": SimpleNamespace(),
        "prepared": {}, "identity": {}, "lora": None, "seed": 7,
        "initial": initial, "endpoints": torch.tensor([[5, 10, 15]]),
        "valid_intervals": torch.ones(1, 3, dtype=torch.bool),
        "valid_semantics": torch.ones(1, 5, dtype=torch.bool),
        "device": torch.device("cpu"),
    }
    first = script._variant_result(**shared)
    second = script._variant_result(**shared)
    hook.remove()

    assert observed and all(observed)
    assert first["coordinates"].shape == (38, 16, 32)
    for name in first["public"]:
        torch.testing.assert_close(first["public"][name], second["public"][name])


def test_variant_result_keeps_carrier_batch_and_selects_one_reference_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_analysis_script()
    writer = _variant_writer()
    batch, intervals, columns, width = 5, 3, 5, 32
    blocks = [
        torch.stack([
            torch.full((intervals, columns, width), float(row + 1))
            for row in range(batch)
        ])
    ]
    final = blocks[-1].clone()
    coordinates = torch.stack([
        torch.full((38, 16, width), float(row + 1))
        for row in range(batch)
    ])
    attention = torch.full(
        (batch, 4, 38 * 16, intervals * columns),
        1.0 / (intervals * columns),
    )
    monkeypatch.setattr(
        script, "_run_program",
        lambda *_args, **_kwargs: (blocks, final, coordinates, attention),
    )
    monkeypatch.setattr(script, "validate_lora_state", lambda *_args: None)
    monkeypatch.setattr(
        script, "policy_action",
        lambda **_kwargs: torch.arange(14, dtype=torch.float32).reshape(2, 7),
    )
    valid_intervals = torch.ones(batch, intervals, dtype=torch.bool)
    valid_semantics = torch.ones(batch, columns, dtype=torch.bool)
    result = script._variant_result(
        writer=writer, policy=SimpleNamespace(), processor=SimpleNamespace(),
        prepared={}, identity={}, lora=None, seed=7,
        initial=torch.zeros(batch, intervals, columns, width),
        endpoints=torch.arange(intervals).repeat(batch, 1),
        valid_intervals=valid_intervals, valid_semantics=valid_semantics,
        device=torch.device("cpu"), selected_row=3,
    )

    assert result["coordinates"].shape == (38, 16, width)
    assert float(result["coordinates"].mean()) == pytest.approx(4.0)
    assert result["block_rms"] == pytest.approx([4.0])
    assert (
        result["reader"]["x_mass"]
        + result["reader"]["a_mass"]
        + result["reader"]["d_mass"]
    ) == pytest.approx(1.0)


def test_counterfactuals_preserve_five_condition_carrier_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_analysis_script()
    batch, intervals, columns, width = 5, 3, 5, 4
    initial = torch.arange(
        batch * intervals * columns * width, dtype=torch.float32,
    ).reshape(batch, intervals, columns, width)
    endpoints = torch.arange(intervals).repeat(batch, 1)
    valid_intervals = torch.ones(batch, intervals, dtype=torch.bool)
    valid_semantics = torch.ones(batch, columns, dtype=torch.bool)
    calls = []

    def fake_variant_result(**kwargs: object) -> dict[str, object]:
        calls.append({
            key: kwargs[key].clone()
            for key in (
                "initial", "endpoints", "valid_intervals", "valid_semantics",
            )
        })
        return {
            "program": torch.ones(2), "coordinates": torch.ones(2),
            "factor": {"value": torch.ones(2)},
            "public": {"value": torch.ones(2)},
            "action": torch.ones(2), "reader": {},
            "coordinate_summary": {}, "geometry": {},
        }

    zero_metrics = {
        key: {"relative_l2": 0.0}
        for key in (
            "program", "coordinates", "factor", "public_a", "public_b",
            "effective_ba", "policy_action",
        )
    }
    monkeypatch.setattr(script, "_variant_result", fake_variant_result)
    monkeypatch.setattr(
        script, "_variant_comparison", lambda *_args, **_kwargs: zero_metrics,
    )
    monkeypatch.setattr(
        script, "_routing_diagnostics", lambda **_kwargs: {},
    )
    encoded = {
        "initial": initial, "endpoints": endpoints,
        "valid_intervals": valid_intervals,
        "valid_semantics": valid_semantics,
        "prepared": {}, "action_seed": 7,
        "states": [{"value": torch.ones(2)}],
        "factor_states": [{"value": torch.ones(2)}],
        "coordinates": torch.ones(batch, 2),
        "actions": [torch.ones(2)],
    }
    script._counterfactual_diagnostics(
        writer=SimpleNamespace(), policy=SimpleNamespace(),
        processor=SimpleNamespace(), identity={}, lora=None,
        device=torch.device("cpu"), encoded=encoded,
    )

    assert len(calls) == 17
    assert any(not torch.equal(call["initial"][0], initial[0]) for call in calls[1:])
    for call in calls:
        assert call["initial"].shape[0] == batch
        assert torch.equal(call["initial"][1:], initial[1:])
        assert torch.equal(call["endpoints"], endpoints)
        assert torch.equal(call["valid_intervals"], valid_intervals)
        assert torch.equal(call["valid_semantics"], valid_semantics)


def test_finalize_broadcasts_main_failure_without_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_analysis_script()
    args = SimpleNamespace(output_dir=tmp_path, references_per_task=1)
    context = SimpleNamespace(is_main=True, world_size=1, device=torch.device("cpu"))
    monkeypatch.setattr(
        "ember.writer.ucp_analysis_run.dist.barrier",
        lambda *_args, **_kwargs: pytest.fail("finalize used barrier"),
    )
    with pytest.raises(WriterModelError, match="rows_rank_00"):
        script.finalize_results(
            output_dir=args.output_dir,
            context=context,
            references_per_task=args.references_per_task,
            conditions=script.CONDITIONS,
            result_schema=script.RESULT_SCHEMA,
            started=0.0,
            control_group=object(),
        )


def test_fixed_policy_query_never_opens_hdf5_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "task.hdf5"
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.create_dataset("actions", data=np.ones((1, 7), dtype=np.float32))
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", data=np.zeros((1, 4, 5, 3), dtype=np.uint8))
        obs.create_dataset("eye_in_hand_rgb", data=np.ones((1, 4, 5, 3), dtype=np.uint8))
        obs.create_dataset("ee_states", data=np.ones((1, 6), dtype=np.float32))
        obs.create_dataset("gripper_states", data=np.ones((1, 1), dtype=np.float32))
    real_file = h5py.File

    class GuardedFile:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.handle = real_file(*args, **kwargs)

        def __enter__(self) -> "GuardedFile":
            return self

        def __exit__(self, *args: object) -> None:
            self.handle.close()

        def __getitem__(self, key: str) -> object:
            if "actions" in key:
                raise AssertionError("actions dataset was opened")
            return self.handle[key]

    monkeypatch.setattr(analysis_runtime.h5py, "File", GuardedFile)
    processor = SimpleNamespace(_tokenize_prompts=lambda states, languages: (
        torch.ones(1, 3, dtype=torch.long), torch.ones(1, 3, dtype=torch.bool),
    ))
    prepared, identity = fixed_policy_query(
        WriterTaskAuthority(
            task_id=40, language="test task", path=path,
            expected_bytes=path.stat().st_size,
        ),
        processor, torch.device("cpu"),
    )
    assert identity["actions_dataset_opened"] is False
    assert identity["observation_only"] is True
    assert prepared["observation.images.base_0_rgb"].shape == (1, 3, 4, 5)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True,
    ).stdout.strip()


def test_provenance_accepts_unprotected_descendant_and_rejects_runtime_owner_change(
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
    (repo / "src/ember/writer/validation.py").write_text("runtime\n")
    (repo / "src/ember/writer/checkpoint.py").write_text("checkpoint v1\n")
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
    assert "src/ember/writer/validation.py" in record["protected_paths_unchanged"]
    assert "src/ember/pi05_processing.py" in record["protected_paths_unchanged"]
    assert record[
        "runtime_compatibility_paths_validated_by_training_contract"
    ] == [
        "src/ember/writer/as_config.py",
        "src/ember/writer/as_contract.py",
        "src/ember/writer/checkpoint.py",
    ]

    (repo / "src/ember/writer/checkpoint.py").write_text("checkpoint v2\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "backward-compatible checkpoint reader")
    compatibility_head = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", compatibility_head)
    validate_analysis_provenance(
        repo=repo,
        state={
            "commit": compatibility_head,
            "origin_main": compatibility_head,
            "dirty_paths": [],
        },
        training={"git": {"commit": training}},
    )

    (repo / "src/ember/writer/validation.py").write_text("changed\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "runtime owner change")
    changed = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", changed)
    with pytest.raises(WriterModelError, match="model/config changed"):
        validate_analysis_provenance(
            repo=repo,
            state={"commit": changed, "origin_main": changed, "dirty_paths": []},
            training={"git": {"commit": training}},
        )
