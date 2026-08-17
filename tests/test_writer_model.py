from __future__ import annotations

import torch
from lerobot.policies.pi05.modeling_pi05 import compute_layer_complete
from lerobot.policies.pi_gemma import layernorm_forward

from ember.writer.backbone_memory import (
    LayerMatchedBackboneMemoryEncoder,
    Pi05LayerMatchedBackboneMemory,
    make_backbone_memory_mask,
)
from ember.writer.parameter_grid import (
    AddressPreservingVideoSet,
    LayerMatchedMemoryProgramCompiler,
)
from ember.writer.video_program import MetaLoRAStack
from fixtures.writer_model import _inputs, _model, _open_factor_heads


def _sum(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(value.float().sum() for value in state.values())


def _reverse_inputs() -> tuple[torch.Tensor, ...]:
    values = list(_inputs())
    order = torch.tensor([2, 1, 0, 4, 3, 7, 6, 5])
    values[0] = values[0].index_select(0, order)
    return tuple(values)


def _shuffle_inputs() -> tuple[torch.Tensor, ...]:
    values = list(_inputs())
    order = torch.tensor([0, 2, 1, 4, 3, 5, 7, 6])
    values[0] = values[0].index_select(0, order)
    return tuple(values)


def test_fresh_lmmpc_step0_is_exact_source_identity() -> None:
    model, template = _model()
    generated = model(*_inputs(), policy=torch.nn.Identity())
    assert set(generated) == set(template)
    assert all(
        torch.equal(generated[name], value[None].expand(2, *value.shape))
        for name, value in template.items()
    )


def test_training_and_deployment_share_one_positive_order_program() -> None:
    model, _ = _model()
    _open_factor_heads(model)
    calls = {"procedure": 0, "memory": 0}

    def count_procedure(*_args: object) -> None:
        calls["procedure"] += 1

    def count_memory(*_args: object) -> None:
        calls["memory"] += 1

    procedure_handle = model.procedure.register_forward_hook(count_procedure)
    memory_handle = model.memory_reader.register_forward_hook(count_memory)
    try:
        deployment = model(*_inputs(), policy=torch.nn.Identity())
        assert calls == {"procedure": 1, "memory": 1}
        calls.update(procedure=0, memory=0)
        training = model.forward_training(
            *_inputs(), policy=torch.nn.Identity()
        )
        assert calls == {"procedure": 1, "memory": 1}
    finally:
        procedure_handle.remove()
        memory_handle.remove()
    assert all(torch.equal(deployment[name], training[name]) for name in deployment)


def test_program_is_the_parameter_grid_without_a_second_slot_bank() -> None:
    model, _ = _model()
    encoded = model.encode_program(*_inputs(), policy=torch.nn.Identity())
    assert encoded.program.shape == (2, 20, 16, 256)
    assert encoded.diagnostics.per_video_parameter_memory.shape == (3, 18, 16, 256)
    assert encoded.diagnostics.shared_parameter_memory.shape == (2, 18, 16, 256)
    assert encoded.diagnostics.core_fused_grid.shape == encoded.program.shape
    names = tuple(name for name, _ in model.named_parameters())
    assert not any("slot" in name or "routing" in name for name in names)


def test_reverse_recomputes_a_nontrivial_non_antisymmetric_program() -> None:
    model, _ = _model()
    natural = model.encode_program(*_inputs(), policy=torch.nn.Identity())
    reversed_video = model.encode_program(
        *_reverse_inputs(), policy=torch.nn.Identity()
    )
    assert not torch.allclose(
        reversed_video.diagnostics.per_video_parameter_memory,
        natural.diagnostics.per_video_parameter_memory,
    )
    assert not torch.allclose(
        reversed_video.diagnostics.per_video_parameter_memory,
        -natural.diagnostics.per_video_parameter_memory,
    )
    assert not torch.allclose(reversed_video.program, natural.program)


def test_shuffle_recomputes_procedure_and_parameter_memory() -> None:
    model, _ = _model()
    encoded = model.encode_program(*_inputs(), policy=torch.nn.Identity())
    shuffled = model.encode_program(*_shuffle_inputs(), policy=torch.nn.Identity())
    assert not torch.allclose(
        encoded.diagnostics.per_video_procedure,
        shuffled.diagnostics.per_video_procedure,
    )
    assert not torch.allclose(
        encoded.diagnostics.per_video_parameter_memory,
        shuffled.diagnostics.per_video_parameter_memory,
    )


def test_k_video_permutation_preserves_every_shared_stage() -> None:
    model, _ = _model()
    inputs = _inputs()
    natural = model.encode_program(*inputs, policy=torch.nn.Identity())
    frames, _, _, condition_offsets, tokens, masks, spans = inputs
    order = torch.tensor([3, 4, 0, 1, 2, 5, 6, 7])
    permuted = model.encode_program(
        frames.index_select(0, order),
        torch.tensor([0, 5, 0, 5, 10, 0, 5, 10]),
        torch.tensor([0, 2, 5, 8]),
        condition_offsets,
        tokens,
        masks,
        spans,
        policy=torch.nn.Identity(),
    )
    torch.testing.assert_close(
        natural.diagnostics.shared_parameter_memory,
        permuted.diagnostics.shared_parameter_memory,
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(natural.program, permuted.program, rtol=1e-5, atol=1e-5)


def test_k1_video_set_is_exact_identity_at_each_address() -> None:
    model, _ = _model()
    encoded = model.encode_program(*_inputs(), policy=torch.nn.Identity())
    torch.testing.assert_close(
        encoded.diagnostics.shared_parameter_memory[1],
        encoded.diagnostics.per_video_parameter_memory[2],
        rtol=0,
        atol=0,
    )


def test_constant_video_is_identity_even_after_factor_heads_open() -> None:
    model, template = _model()
    _open_factor_heads(model)
    values = list(_inputs())
    values[0][:] = values[0][0]
    encoded = model.encode_program(*values, policy=torch.nn.Identity())
    assert not encoded.diagnostics.per_video_parameter_memory.count_nonzero()
    assert not encoded.diagnostics.shared_parameter_memory.count_nonzero()
    assert not encoded.diagnostics.core_fused_grid.count_nonzero()
    assert not encoded.program.count_nonzero()
    generated = model.decode_output(encoded)
    assert all(
        torch.equal(generated[name], value[None].expand(2, *value.shape))
        for name, value in template.items()
    )


def test_axial_m2p_is_strictly_zero_preserving() -> None:
    compiler = LayerMatchedMemoryProgramCompiler(
        heads=8,
        blocks=2,
        max_relative_correction=0.5,
        initialization_seed=7,
    )
    memory = torch.zeros(2, 18, 16, 256)
    core = torch.randn(2, 3, 256)
    valid = torch.ones(2, 3, dtype=torch.bool)
    language = torch.randn(2, 256)
    fused, compiled = compiler(memory, core, valid, language)
    assert not fused.count_nonzero()
    assert not compiled.count_nonzero()


def test_axial_m2p_commitment_is_cellwise_bounded_and_live() -> None:
    compiler = LayerMatchedMemoryProgramCompiler(
        heads=8,
        blocks=2,
        max_relative_correction=0.5,
        initialization_seed=7,
    )
    anchor = torch.randn(2, 20, 16, 256, requires_grad=True)
    proposal = anchor + 100.0 * torch.randn_like(anchor)
    committed = compiler.bounded_commitment(anchor, proposal)
    correction_rms = (committed - anchor).float().square().mean(-1).sqrt()
    anchor_rms = anchor.float().square().mean(-1).sqrt()
    torch.testing.assert_close(
        correction_rms,
        0.25 * anchor_rms,
        rtol=2e-5,
        atol=2e-6,
    )
    committed.square().mean().backward()
    assert compiler.commitment_logit.grad is not None
    assert compiler.commitment_logit.grad.count_nonzero()


def test_video_set_does_not_mix_layer_or_rank_addresses() -> None:
    fusion = AddressPreservingVideoSet()
    value = torch.zeros(1, 18, 16, 256)
    value[0, 7, 11] = 1.0
    output = fusion(
        value,
        torch.randn(1, 256),
        torch.randn(1, 256),
        (0, 1),
    )
    assert torch.equal(output[0], value[0])
    assert output[0, 7, 11].count_nonzero() == 256
    assert output.count_nonzero() == 256


def test_factor_family_and_layer_ownership_is_native_rank16() -> None:
    model, template = _model()
    _open_factor_heads(model)
    program = torch.zeros(1, 20, 16, 256)
    program[:, 1] = 0.25
    generated = model.decode_program(program)
    for name, original in template.items():
        changed = not torch.equal(generated[name], original)
        in_layer0 = ".layers.0.self_attn." in name
        assert changed == in_layer0
    assert all(value.shape == template[name].shape for name, value in generated.items())


def test_all_factor_families_and_dynamic_path_receive_gradients() -> None:
    model, _ = _model()
    _open_factor_heads(model)
    generated = model.forward_training(
        *_inputs(), policy=torch.nn.Identity()
    )
    _sum(generated).backward()
    for head in model.factor_heads.values():
        gradient = head.network[-1].weight.grad
        assert gradient is not None and gradient.count_nonzero()
    assert model.memory_tokens.grad is not None
    assert model.memory_tokens.grad.count_nonzero()
    assert model.memory_reader.query.weight.grad is not None
    assert model.memory_reader.query.weight.grad.count_nonzero()
    assert model.memory_reader.key.weight.grad is not None
    assert model.memory_reader.key.weight.grad.count_nonzero()
    assert model.memory_reader.address_query.weight.grad is not None
    assert model.memory_reader.address_query.weight.grad.count_nonzero()


def test_memory_reader_uses_internal_procedure_stages_not_only_the_last() -> None:
    model, _ = _model()
    encoded = model.encode_program(*_inputs(), policy=torch.nn.Identity())
    context = model._encode_context(*_inputs(), policy=torch.nn.Identity())
    procedure = encoded.diagnostics.per_video_procedure
    repeated_last = procedure.clone()
    for video, (left, right) in enumerate(
        zip(context.video_bounds[:-1], context.video_bounds[1:], strict=True)
    ):
        repeated_last[video, : right - left] = procedure[video, right - left - 1]
    collapsed = model.memory_reader(
        context.encoding.layer_memory,
        repeated_last,
        torch.arange(repeated_last.shape[1])[None]
        < torch.tensor(
            [right - left for left, right in zip(
                context.video_bounds[:-1], context.video_bounds[1:], strict=True
            )]
        )[:, None],
        context.video_bounds,
    )
    assert not torch.allclose(
        encoded.diagnostics.per_video_parameter_memory,
        collapsed,
    )


def test_writer_state_is_fresh_strictly_reloadable() -> None:
    source, _ = _model()
    target, _ = _model()
    target.load_state_dict(source.state_dict(), strict=True)
    assert all(
        torch.equal(target.state_dict()[name], value)
        for name, value in source.state_dict().items()
    )


class _FixedShapeSemantic:
    max_frames_per_encoder_call = 4
    expert_width = 4

    @staticmethod
    def _validate_forward_batch(
        _policy,
        _frames,
        _frame_condition_ids,
        language_tokens,
        _language_mask,
        _task_span_mask,
    ):
        valid = torch.ones(language_tokens.shape[0], 2, dtype=torch.bool)
        return torch.nn.Identity(), valid, torch.full((language_tokens.shape[0],), 2)

    @staticmethod
    def _encode_text(
        _core, language_tokens, _task_span_mask, maximum_task_tokens
    ) -> torch.Tensor:
        return torch.zeros(language_tokens.shape[0], maximum_task_tokens, 256)


class _FixedShapeRecordingEncoder(LayerMatchedBackboneMemoryEncoder):
    def __init__(self) -> None:
        super().__init__(
            image_width=4, expert_width=4, activation_checkpointing=False
        )
        self.frame_calls: list[torch.Tensor] = []

    def _encode_microbatch(self, _semantic, _core, frames, *_args):
        self.frame_calls.append(frames.clone())
        batch = frames.shape[0]
        evidence = frames.float().mean(dim=(1, 2, 3))[:, None, None]
        evidence = evidence.expand(batch, 2, 256)
        interaction = evidence[:, 0]
        memory = torch.zeros(batch, 18, 16, 4)
        return evidence, evidence, interaction, memory


def test_final_native_encoder_microbatch_uses_discarded_zero_padding() -> None:
    encoder = _FixedShapeRecordingEncoder()
    frames = torch.arange(5 * 3 * 2 * 2, dtype=torch.uint8).reshape(5, 3, 2, 2)
    output = encoder(
        _FixedShapeSemantic(),
        torch.nn.Identity(),
        frames,
        torch.zeros(5, dtype=torch.long),
        torch.ones(1, 3, dtype=torch.long),
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([[False, True, True]]),
        torch.zeros(16, 4),
    )
    assert [value.shape[0] for value in encoder.frame_calls] == [4, 4]
    assert torch.equal(encoder.frame_calls[1][0], frames[4])
    assert not encoder.frame_calls[1][1:].count_nonzero()
    assert output.layer_memory.shape[0] == frames.shape[0]


class _MemoryNorm(torch.nn.Module):
    def forward(
        self, value: torch.Tensor, cond: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, None]:
        del cond
        return value, None


class _MemoryMlp(torch.nn.Module):
    def __init__(self, width: int, scale: float) -> None:
        super().__init__()
        self.up_proj = torch.nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.up_proj.weight.copy_(torch.eye(width))
        self.scale = float(scale)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up_proj(value) * self.scale


class _MemoryAttention(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.head_dim = 2
        self.num_key_value_groups = 8
        self.scaling = self.head_dim**-0.5
        self.q_proj = torch.nn.Linear(width, 16, bias=False)
        self.k_proj = torch.nn.Linear(width, 2, bias=False)
        self.v_proj = torch.nn.Linear(width, 2, bias=False)
        self.o_proj = torch.nn.Linear(16, width, bias=False)


class _MemoryLayer(torch.nn.Module):
    def __init__(self, width: int, index: int) -> None:
        super().__init__()
        self.input_layernorm = _MemoryNorm()
        self.self_attn = _MemoryAttention(width)
        self.post_attention_layernorm = _MemoryNorm()
        self.mlp = _MemoryMlp(width, 0.01 * (index + 1))


class _MemoryRotary(torch.nn.Module):
    def forward(
        self, value: torch.Tensor, _positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones_like(value), torch.zeros_like(value)


class _MemoryBackbone(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            _MemoryLayer(width, index)
            for index in range(Pi05LayerMatchedBackboneMemory.LAYERS)
        )
        self.norm = _MemoryNorm()
        self.rotary_emb = _MemoryRotary()


class _MemoryBridge(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.paligemma = torch.nn.Module()
        self.paligemma.model = torch.nn.Module()
        self.paligemma.model.language_model = _MemoryBackbone(width)
        self.gemma_expert = torch.nn.Module()
        self.gemma_expert.model = _MemoryBackbone(width)

    def forward(
        self,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: None,
        inputs_embeds: list[torch.Tensor],
        use_cache: bool,
        adarms_cond: list[torch.Tensor | None],
    ) -> tuple[list[torch.Tensor], None]:
        assert past_key_values is None and not use_cache
        language = self.paligemma.model.language_model
        expert = self.gemma_expert.model
        streams = inputs_embeds
        for layers in zip(language.layers, expert.layers, strict=True):
            streams = compute_layer_complete(
                streams,
                attention_mask,
                position_ids,
                adarms_cond,
                layers,
                language.rotary_emb,
            )
        outputs = []
        for hidden, norm, condition in zip(
            streams,
            (language.norm, expert.norm),
            adarms_cond,
            strict=True,
        ):
            output, _ = layernorm_forward(norm, hidden, condition)
            outputs.append(output)
        return outputs, None


class _MemoryCore(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.paligemma_with_expert = _MemoryBridge(width)

    @staticmethod
    def _prepare_attention_masks_4d(mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask[:, None], 0.0, -2.0e9)


def _memory_inputs(width: int = 4) -> tuple[torch.Tensor, ...]:
    prefix = torch.randn(2, 5, width)
    prefix_padding = torch.tensor(
        [[True, True, True, True, False], [True, True, True, True, True]]
    )
    action = torch.randn(2, 50, width)
    action_padding = torch.ones(2, 50, dtype=torch.bool)
    action_markers = torch.zeros_like(action_padding)
    action_markers[:, 0] = True
    adarms = torch.randn(2, width)
    memory = torch.nn.Parameter(torch.randn(16, width))
    return (
        prefix,
        prefix_padding,
        action,
        action_padding,
        action_markers,
        adarms,
        memory,
    )


def _memory_loop(core: _MemoryCore, *, checkpointing: bool = False):
    language = core.paligemma_with_expert.paligemma.model.language_model
    expert = core.paligemma_with_expert.gemma_expert.model
    return (
        Pi05LayerMatchedBackboneMemory(
            image_width=4,
            expert_width=4,
            activation_checkpointing=checkpointing,
        ),
        MetaLoRAStack(language.layers, rank=1),
        MetaLoRAStack(expert.layers, rank=1),
    )


def test_three_block_memory_mask_is_one_way() -> None:
    padding = torch.tensor([[True, True, False]])
    mask = make_backbone_memory_mask(
        padding, action_horizon=50, memory_tokens=16
    )[0]
    action = 3
    memory = 53
    assert mask.shape == (69, 69)
    assert mask[0, :3].tolist() == [True, True, False]
    assert not mask[0, action:].any()
    assert mask[action, :memory].sum().item() == 52
    assert not mask[action, memory:].any()
    assert mask[memory, :].sum().item() == 68
    assert not mask[2].any()
    assert not mask[:, 2].any()


def test_joint_loop_returns_all_layer_rank_memories() -> None:
    torch.manual_seed(5)
    core = _MemoryCore(4)
    loop, vl, action = _memory_loop(core)
    output = loop(core, *_memory_inputs(), vl, action)
    assert output.prefix_hidden.shape == (2, 5, 4)
    assert output.action_hidden.shape == (2, 50, 4)
    assert output.layer_memory.shape == (2, 18, 16, 4)
    assert not torch.equal(output.layer_memory[:, 0], output.layer_memory[:, -1])


def test_memory_wrapper_preserves_exact_native_carrier() -> None:
    torch.manual_seed(6)
    core = _MemoryCore(4)
    loop, vl, action_adapters = _memory_loop(core)
    inputs = _memory_inputs()
    language = core.paligemma_with_expert.paligemma.model.language_model
    expert = core.paligemma_with_expert.gemma_expert.model
    prefix, suffix, mask, positions = loop._prepare_native_context(
        core,
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        language.layers[0].self_attn.q_proj.weight.dtype,
    )
    with (
        torch.no_grad(),
        vl.installed(language),
        action_adapters.installed(expert),
    ):
        expected, _ = core.paligemma_with_expert.forward(
            attention_mask=mask,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix, suffix],
            use_cache=False,
            adarms_cond=[None, inputs[5]],
        )
    actual = loop(core, *inputs, vl, action_adapters)
    torch.testing.assert_close(actual.prefix_hidden, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual.action_hidden, expected[1], rtol=0, atol=0)


def test_memory_values_cannot_change_prefix_or_action() -> None:
    torch.manual_seed(7)
    core = _MemoryCore(4)
    loop, vl, action = _memory_loop(core)
    inputs = list(_memory_inputs())
    with torch.no_grad():
        first = loop(core, *inputs, vl, action)
        inputs[-1].add_(100.0)
        second = loop(core, *inputs, vl, action)
    valid_prefix = inputs[1]
    torch.testing.assert_close(
        first.prefix_hidden[valid_prefix],
        second.prefix_hidden[valid_prefix],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(first.action_hidden, second.action_hidden, rtol=0, atol=0)
    assert not torch.equal(first.layer_memory, second.layer_memory)


def test_checkpointed_memory_replay_reaches_memory_tokens_only() -> None:
    torch.manual_seed(11)
    core = _MemoryCore(4).requires_grad_(False)
    loop, vl, action = _memory_loop(core, checkpointing=True)
    vl.requires_grad_(False)
    action.requires_grad_(False)
    inputs = _memory_inputs()
    output = loop(core, *inputs, vl, action)
    output.layer_memory.float().square().mean().backward()
    memory = inputs[-1]
    assert memory.grad is not None and memory.grad.count_nonzero()
    assert all(parameter.grad is None for parameter in core.parameters())
    assert all(parameter.grad is None for parameter in vl.parameters())
    assert all(parameter.grad is None for parameter in action.parameters())
