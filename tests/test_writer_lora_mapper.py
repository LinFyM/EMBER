from __future__ import annotations

import torch

from ember.writer.lora_mapper import CompleteLoRAMapper, build_lora_tensor_specs


def _template(rank: int = 8) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    state: dict[str, torch.Tensor] = {}
    for layer in range(18):
        prefix = (
            "model.paligemma_with_expert.gemma_expert.model.layers."
            f"{layer}.self_attn."
        )
        for projection, output in (("q_proj", 2048), ("v_proj", 256)):
            state[prefix + projection + ".lora_A.default.weight"] = torch.randn(
                rank, 1024, generator=generator
            )
            state[prefix + projection + ".lora_B.default.weight"] = torch.zeros(
                output, rank
            )
    for module, input_width, output_width in (
        ("model.action_in_proj", 32, 1024),
        ("model.action_out_proj", 1024, 32),
    ):
        state[module + ".lora_A.default.weight"] = torch.randn(
            rank, input_width, generator=generator
        )
        state[module + ".lora_B.default.weight"] = torch.zeros(output_width, rank)
    return state


def test_complete_mapper_starts_at_functional_identity_and_opens_b_only() -> None:
    template = _template()
    mapper = CompleteLoRAMapper(
        build_lora_tensor_specs(template),
        template_state=template,
        program_width=256,
        mapper_width=128,
    )
    program = torch.randn(2, 20, 8, 256, requires_grad=True)
    output = mapper(program)
    for name, value in output.items():
        expected = template[name][None].expand_as(value)
        assert torch.equal(value, expected)
    loss = sum(value.square().mean() for name, value in output.items() if ".lora_B." in name)
    loss.backward()
    assert all(
        readout.weight.grad is not None
        for readout in mapper.family_b_readouts.values()
    )
    assert program.grad is not None
    assert torch.count_nonzero(program.grad) == 0
    assert mapper.project.weight.grad is not None
    assert torch.count_nonzero(mapper.project.weight.grad) == 0
    assert all(readout.bias is None for readout in mapper.family_b_readouts.values())
    assert not any(
        ".hidden." in name or ".a." in name or "dynamic_a" in name
        for name in mapper.state_dict()
    )


def test_complete_mapper_generates_all_rank8_shapes_after_b_opens() -> None:
    template = _template()
    mapper = CompleteLoRAMapper(
        build_lora_tensor_specs(template),
        template_state=template,
        program_width=256,
        mapper_width=128,
    )
    for readout in mapper.family_b_readouts.values():
        torch.nn.init.normal_(readout.weight, std=0.01)
    program = torch.randn(3, 20, 8, 256)
    output = mapper(program)
    assert set(output) == set(template)
    assert all(value.shape == (3, *template[name].shape) for name, value in output.items())
    assert any(
        not torch.equal(value[0], value[1])
        for name, value in output.items()
        if ".lora_B." in name
    )
    projected = mapper.project(program)
    q_name = next(
        name
        for name in sorted(template)
        if ".layers.7.self_attn.q_proj.lora_B." in name
    )
    expected_q = mapper.family_b_readouts["q"](projected[:, 8]).transpose(-1, -2)
    torch.testing.assert_close(output[q_name], expected_q)
    action_in_name = next(
        name for name in template if "action_in_proj.lora_B." in name
    )
    expected_action_in = mapper.family_b_readouts["action_in"](
        projected[:, 0]
    ).transpose(-1, -2)
    torch.testing.assert_close(output[action_in_name], expected_action_in)
