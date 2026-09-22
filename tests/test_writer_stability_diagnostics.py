import copy
from types import SimpleNamespace

import torch


def test_parameter_groups_are_exclusive_and_grad_metrics_match() -> None:
    from ember.writer.stability_diagnostics import grouped_gradient_metrics, parameter_group

    names = (
        "writer.body.weight",
        "writer.semantic_encoder.text_meta_lora.a",
        "writer.semantic_encoder.vl_meta_lora.b",
        "writer.semantic_encoder.action_meta_lora.c",
    )
    assert [parameter_group(name) for name in names] == [
        "writer_main", "text_meta", "vl_meta", "action_meta"
    ]
    metrics = grouped_gradient_metrics(names, tuple(torch.tensor([value]) for value in (3.0, 4.0, 5.0, 12.0)))
    assert metrics == {
        "writer_main_grad_norm": 3.0,
        "text_meta_grad_norm": 4.0,
        "vl_meta_grad_norm": 5.0,
        "action_meta_grad_norm": 12.0,
    }


def test_gradient_dot_and_combination_preserve_registered_weights() -> None:
    from ember.writer.stability_diagnostics import combine_gradients, gradient_dot

    left = (torch.tensor([1.0, 2.0]), torch.tensor([3.0]))
    right = (torch.tensor([4.0, 5.0]), torch.tensor([6.0]))
    assert gradient_dot(left, right) == 32.0
    parameters = (torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(1)))
    combine_gradients(parameters, (left, right), (0.25, 0.75))
    torch.testing.assert_close(parameters[0].grad, torch.tensor([3.25, 4.25]))
    torch.testing.assert_close(parameters[1].grad, torch.tensor([5.25]))


def test_output_column_space_uses_all_numerically_independent_columns() -> None:
    from ember.writer.output_space_diagnostics import column_space, space_overlap

    matrix = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=torch.float64)
    basis, singular, tolerance = column_space(matrix)
    assert basis.shape == (3, 2)
    assert len(singular) == 2 and tolerance > 0
    projected = basis @ (basis.T @ matrix)
    torch.testing.assert_close(projected, matrix, rtol=1e-12, atol=1e-12)
    overlap = space_overlap("left", basis, "right", basis, "q_b")
    assert overlap["chordal_distance"] < 1e-6


def test_captured_factor_codes_restore_numeric_layer_order() -> None:
    from ember.writer.output_space_diagnostics import codes_by_numeric_layer

    lexical_order = [0, 1, *range(10, 18), *range(2, 10)]
    rows = [torch.full((1, 16, 216), float(layer)) for layer in lexical_order]
    codes = codes_by_numeric_layer(rows, lexical_order, "q_b")
    assert tuple(codes.shape) == (18, 16, 216)
    torch.testing.assert_close(codes[:, 0, 0], torch.arange(18, dtype=torch.float32))


def test_output_projection_arms_preserve_self_and_match_shrink_norm() -> None:
    from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX
    from ember.writer.output_space_diagnostics import (
        FAMILIES, effective_frobenius, factors, target_name, transform_state,
    )

    generator = torch.Generator().manual_seed(9)
    state = {"unrelated": torch.tensor([3.0])}
    old_bases, new_bases = {}, {}
    for family in FAMILIES:
        rows = 2048 if family == "q_b" else 256
        old = torch.eye(rows, dtype=torch.float64)[:, :16]
        old_bases[family], new_bases[family] = old, old[:, :8]
        for layer in range(18):
            owner = target_name(layer, family)
            state[owner + LORA_A_SUFFIX] = torch.randn((16, 1024), generator=generator)
            coordinates = torch.randn((16, 16), generator=generator)
            state[owner + LORA_B_SUFFIX] = (old @ coordinates.to(torch.float64)).float()
    self_state, self_rows = transform_state(
        state, arm="SELF", old_bases=old_bases, new_bases=new_bases, scale=1.0,
    )
    new_state, new_rows = transform_state(
        state, arm="NEWSPACE", old_bases=old_bases, new_bases=new_bases, scale=1.0,
    )
    shrink_state, shrink_rows = transform_state(
        state, arm="SHRINK", old_bases=old_bases, new_bases=new_bases, scale=1.0,
    )
    assert len(self_rows) == len(new_rows) == len(shrink_rows) == 36
    assert self_state["unrelated"].item() == new_state["unrelated"].item() == 3.0
    for family in FAMILIES:
        for layer in range(18):
            owner = target_name(layer, family)
            torch.testing.assert_close(self_state[owner + LORA_B_SUFFIX],
                                       state[owner + LORA_B_SUFFIX], rtol=0, atol=1e-6)
            a, projected_b = factors(new_state, layer, family)
            _, shrink_b = factors(shrink_state, layer, family)
            assert abs(effective_frobenius(projected_b, a)
                       - effective_frobenius(shrink_b, a)) < 2e-4


def _causal_encoded(value: float):
    from ember.writer.causal_diagnostics import EncodedPath

    tensor = torch.tensor([value])
    return EncodedPath(tensor, tensor, tensor, tensor.long(), tensor.bool(),
                       {"frame_evidence": tensor, "horizon": tensor})


def test_causal_path_selection_keeps_masks_positions_with_memory() -> None:
    from ember.writer.causal_diagnostics import select_path_sources

    correct, wrong, other = _causal_encoded(1.0), _causal_encoded(2.0), _causal_encoded(3.0)
    core, procedure = select_path_sources({"correct": correct, "wrong": wrong, "other": other}, "CW")
    assert core is correct and core.valid_core is correct.valid_core
    assert procedure is wrong and procedure.positions is wrong.positions and procedure.valid_frames is wrong.valid_frames
    core, procedure = select_path_sources({"correct": correct, "wrong": wrong, "other": other}, "CO")
    assert core is correct and procedure is other


class _CausalAllGroups(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.semantic_encoder = torch.nn.Module()
        self.semantic_encoder.text_meta_lora = torch.nn.Linear(1, 1, bias=False)
        self.semantic_encoder.vl_meta_lora = torch.nn.Linear(1, 1, bias=False)
        self.semantic_encoder.action_meta_lora = torch.nn.Linear(1, 1, bias=False)
        self.semantic_encoder.projection = torch.nn.Linear(1, 1, bias=False)
        self.semantic_core = torch.nn.Linear(1, 1, bias=False)
        self.procedure = torch.nn.Linear(1, 1, bias=False)
        self.compiler = torch.nn.Linear(1, 1, bias=False)
        self.factor_heads = torch.nn.Linear(1, 1, bias=False)


def test_causal_parameter_groups_are_exhaustive_and_identity_unique() -> None:
    from ember.writer.causal_diagnostics import validate_causal_parameter_groups

    state = _CausalAllGroups()
    optimizer = torch.optim.AdamW(state.parameters(), lr=0.01)
    loaded = SimpleNamespace(runtime=SimpleNamespace(state=state), optimizer=optimizer,
                             parameter_names=tuple(name for name, _ in state.named_parameters()))
    assert validate_causal_parameter_groups(loaded) == {
        "text_meta": 1, "vl_meta": 1, "action_meta": 1, "semantic_core": 2,
        "procedure": 1, "compiler": 1, "factor_heads": 1,
    }


def _causal_tiny_loaded():
    state = torch.nn.Module()
    state.semantic_core = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(state.parameters(), lr=0.1, betas=(0.9, 0.95), weight_decay=0.01)
    for parameter in state.parameters():
        parameter.grad = torch.full_like(parameter, 0.2)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return SimpleNamespace(runtime=SimpleNamespace(state=state), optimizer=optimizer,
                           parameter_names=tuple(name for name, _ in state.named_parameters()))


def _causal_reference(parent, gradients):
    reference = torch.nn.Module()
    reference.semantic_core = torch.nn.Linear(1, 1, bias=False)
    reference.load_state_dict(parent[0])
    optimizer = torch.optim.AdamW(reference.parameters(), lr=0.1, betas=(0.9, 0.95), weight_decay=0.01)
    optimizer.load_state_dict(copy.deepcopy(parent[1]))
    parameters = tuple(reference.parameters())
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient.to(parameter)
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    before = tuple(parameter.detach().clone() for parameter in parameters)
    optimizer.step()
    return reference, optimizer, before


def _causal_values(module):
    return tuple(parameter.detach().clone() for parameter in module.parameters())


def test_causal_virtual_candidates_match_adamw_and_m_scaling() -> None:
    from ember.writer.causal_diagnostics import apply_virtual_candidate
    from ember.writer.stability_diagnostics import restore_parent, snapshot_parent

    loaded = _causal_tiny_loaded()
    parent = snapshot_parent(loaded)
    q = tuple(torch.full_like(parameter, 0.12, device="cpu") for parameter in loaded.runtime.state.parameters())
    a = tuple(torch.full_like(parameter, -0.03, device="cpu") for parameter in loaded.runtime.state.parameters())
    for label, gradients in (("J", tuple(left + right for left, right in zip(q, a, strict=True))),
                             ("Q", q), ("Z", tuple(torch.zeros_like(value) for value in q))):
        restore_parent(loaded, parent)
        apply_virtual_candidate(loaded, q=q, a=a, candidate=label, lr=0.1)
        reference, _optimizer, _before = _causal_reference(parent, gradients)
        for actual, expected in zip(_causal_values(loaded.runtime.state), _causal_values(reference), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
    restore_parent(loaded, parent)
    result = apply_virtual_candidate(loaded, q=q, a=a, candidate="M", lr=0.1)
    q_reference, _q_optimizer, q_before = _causal_reference(parent, q)
    joint_reference, joint_optimizer, joint_before = _causal_reference(
        parent, tuple(left + right for left, right in zip(q, a, strict=True)),
    )
    q_norm = torch.sqrt(sum((after - before).square().sum()
                            for after, before in zip(_causal_values(q_reference), q_before, strict=True)))
    joint_norm = torch.sqrt(sum((after - before).square().sum()
                                for after, before in zip(_causal_values(joint_reference), joint_before, strict=True)))
    alpha = q_norm / joint_norm
    assert abs(result["scale_alpha"] - float(alpha)) < 1e-7
    for actual, parent_value, joint_value in zip(_causal_values(loaded.runtime.state), joint_before,
                                                   _causal_values(joint_reference), strict=True):
        torch.testing.assert_close(actual, parent_value + alpha * (joint_value - parent_value), rtol=0, atol=1e-7)
    for parameter, reference_parameter in zip(loaded.runtime.state.parameters(), joint_reference.parameters(), strict=True):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(loaded.optimizer.state[parameter][key], joint_optimizer.state[reference_parameter][key],
                                       rtol=0, atol=1e-7)
