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
