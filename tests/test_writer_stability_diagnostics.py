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
