from __future__ import annotations

from types import SimpleNamespace

import torch

from ember.writer.backbone_memory import (
    Pi05CapacityMatchedBackboneMemory,
    make_backbone_memory_mask,
)
from fixtures.legacy_v6_writer import _model as _legacy_v6_model
from fixtures.writer_model import _inputs, _model
from ember.writer.slot_set import PolicyProcedureCommonValueFusion
from ember.writer.video_program import MetaLoRAStack


def _sum(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(value.float().sum() for value in state.values())


def test_procedure_common_value_is_raw_mean_under_uniform_attention() -> None:
    fusion = PolicyProcedureCommonValueFusion(width=4)
    with torch.no_grad():
        fusion.query.weight.zero_()
        fusion.key.weight.zero_()
        fusion.output.weight.copy_(torch.eye(4))
    slots = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    shared, diagnostics = fusion(slots, torch.tensor([0, 2]))
    common = slots.mean(dim=0)
    torch.testing.assert_close(diagnostics.shared_corrections[0], common)
    torch.testing.assert_close(shared[0], common + common)
    torch.testing.assert_close(diagnostics.attention[0], torch.full((2, 3), 0.5))

    singleton, singleton_diagnostics = fusion(slots[:1], torch.tensor([0, 1]))
    assert torch.equal(singleton, slots[:1])
    assert not singleton_diagnostics.shared_corrections.count_nonzero()


def test_native_compiler_public_stages_are_exact_old_graph() -> None:
    model, _ = _legacy_v6_model()
    compiler = model.compiler
    generator = torch.Generator(device="cpu").manual_seed(29)
    core = torch.randn(2, 4, 256, generator=generator)
    procedure = torch.randn(2, 5, 256, generator=generator)
    valid_core = torch.tensor([[True, True, True, False], [True] * 4])
    valid_procedure = torch.tensor([[True, True, True, False, False], [True] * 5])
    positions = torch.arange(5).expand(2, -1)

    routing = compiler.routing_norm(compiler._routing())[None].expand(2, -1, -1)
    core_slots = compiler.core_reader(routing, core, valid_core)
    procedure_slots, normalized_core, centered = compiler.procedure_reader(
        routing,
        core_slots,
        procedure,
        positions,
        valid_procedure,
    )
    gamma, beta = compiler.modulation(compiler.procedure_norm(procedure_slots)).chunk(
        2, dim=-1
    )
    fused = (1.0 + gamma) * normalized_core + beta
    expected = compiler.post_fusion(fused, routing)

    actual, diagnostics = compiler.fused_slots(
        core,
        valid_core,
        procedure,
        positions,
        valid_procedure,
    )
    assert torch.equal(actual, expected)
    assert torch.equal(diagnostics["core_slots"], core_slots)
    assert torch.equal(diagnostics["procedure_centered"], centered)
    assert torch.equal(diagnostics["procedure_slots"], procedure_slots)
    assert torch.equal(diagnostics["adaln_gamma"], gamma)
    assert torch.equal(diagnostics["adaln_beta"], beta)
    assert torch.equal(diagnostics["fused_slots"], fused)


def test_native_factor_hidden_residual_keeps_family_and_slot_ownership() -> None:
    model, _ = _legacy_v6_model()
    generator = torch.Generator(device="cpu").manual_seed(31)
    with torch.no_grad():
        for head in model.factor_heads.values():
            head.network[-1].weight.normal_(std=0.01, generator=generator)
    slots = torch.randn(1, 320, 256, generator=generator)
    baseline = model.decode_slots(slots)
    residuals = {
        family: torch.zeros_like(slots) for family in model.factor_heads
    }
    exact = model.decode_slots(slots, factor_hidden_residuals=residuals)
    assert all(torch.equal(exact[name], baseline[name]) for name in baseline)

    residuals["q_a"][:, :16] = 0.1
    changed = model.decode_slots(slots, factor_hidden_residuals=residuals)
    for name in baseline:
        is_first_layer_q_a = (
            ".layers.0.self_attn.q_proj.lora_A." in name
        )
        assert (not torch.equal(changed[name], baseline[name])) == is_first_layer_q_a


def test_v6_layerwise_conditioner_step0_k1_is_exact_native_v6() -> None:
    model, _ = _model()
    frames, indices, _, _, tokens, masks, spans = _inputs()
    inputs = (
        frames[5:],
        indices[5:],
        torch.tensor([0, 3]),
        torch.tensor([0, 1]),
        tokens[1:],
        masks[1:],
        spans[1:],
    )
    encoded = model.encode_program(*inputs, policy=torch.nn.Identity())
    frames1, indices1, video_offsets1, _, tokens1, masks1, spans1 = inputs
    evidence = model.base_writer.encode_video_evidence(
        torch.nn.Identity(),
        frames1,
        video_offsets1,
        tokens1,
        masks1,
        spans1,
    )
    memories = model.base_writer.build_memories(evidence, indices1)
    native = model.base_writer.decode_slots(model.base_writer.compile_slots(memories))
    bridged = model(*inputs, policy=torch.nn.Identity())
    for name, value in native.items():
        if not name.endswith(".lora_A.default.weight"):
            continue
        b_name = name.replace(".lora_A.", ".lora_B.")
        expected_ba = torch.matmul(native[b_name].double(), value.double())
        actual_ba = torch.matmul(
            bridged[b_name].double(), bridged[name].double()
        )
        torch.testing.assert_close(actual_ba, expected_ba, rtol=0, atol=0)

    torch.nn.init.normal_(model.query_delta.weight, std=1.0)
    bridged_after = model(*inputs, policy=torch.nn.Identity())
    assert any(not torch.equal(bridged_after[name], native[name]) for name in native)


def test_v6_procedure_common_value_is_set_invariant_and_order_sensitive() -> None:
    model, _ = _model()
    inputs = _inputs()
    natural = model.encode_program(*inputs, policy=torch.nn.Identity())

    frames, _, _, condition_offsets, tokens, masks, spans = inputs
    video_order = torch.tensor([3, 4, 0, 1, 2, 5, 6, 7])
    permuted = model.encode_program(
        frames[video_order],
        torch.tensor([0, 5, 0, 5, 10, 0, 5, 10]),
        torch.tensor([0, 2, 5, 8]),
        condition_offsets,
        tokens,
        masks,
        spans,
        policy=torch.nn.Identity(),
    )
    torch.testing.assert_close(natural.program, permuted.program)

    reversed_frames = frames.clone()
    reversed_frames[:3] = frames[:3].flip(0)
    reversed_program = model.encode_program(
        reversed_frames,
        inputs[1],
        inputs[2],
        condition_offsets,
        tokens,
        masks,
        spans,
        policy=torch.nn.Identity(),
    )
    assert not torch.allclose(natural.program, reversed_program.program)


def test_v6_layerwise_conditioner_gradient_staging_and_freeze_boundary() -> None:
    model, _ = _model()
    generated, auxiliary = model.forward_training(
        *_inputs(), policy=torch.nn.Identity()
    )
    assert auxiliary.item() == 0.0
    _sum(generated).backward()
    assert model.query_delta.weight.grad is not None
    assert model.query_delta.weight.grad.abs().sum() > 0
    assert all(
        parameter.grad is None or not parameter.grad.count_nonzero()
        for parameter in model.probe_conditioner.parameters()
    )
    assert all(parameter.grad is None for parameter in model.base_writer.parameters())
    assert all(parameter.grad is None for parameter in model.procedure_set.parameters())

    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.query_delta.weight.normal_(std=0.01)
    generated, _ = model.forward_training(*_inputs(), policy=torch.nn.Identity())
    _sum(generated).backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.probe_conditioner.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.layer_probe_reader.parameters()
    )


def test_cached_readouts_recompile_exactly_without_video_backbone() -> None:
    model, _ = _model()
    torch.nn.init.normal_(model.query_delta.weight, std=0.01)
    encoded = model.encode_program(*_inputs(), policy=torch.nn.Identity())
    cached = model.compile_readouts(
        encoded.diagnostics.shared_core_slots,
        encoded.diagnostics.per_video_procedure_slots,
        _inputs()[3],
        per_video_query_conditioners=(
            encoded.diagnostics.per_video_query_conditioners
        ),
        per_video_query_deltas=encoded.diagnostics.per_video_query_deltas,
    )
    torch.testing.assert_close(cached.program, encoded.program, rtol=0, atol=0)
    _sum(model.decode_program(cached.program)).backward()
    assert model.query_delta.weight.grad is not None
    assert model.query_delta.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.base_writer.parameters())
    assert all(parameter.grad is None for parameter in model.procedure_set.parameters())


def test_k1_uses_same_graph_and_has_exact_zero_auxiliary() -> None:
    model, _ = _model()
    frames, indices, _, _, tokens, masks, spans = _inputs()
    generated, auxiliary = model.forward_training(
        frames[5:],
        indices[5:],
        torch.tensor([0, 3]),
        torch.tensor([0, 1]),
        tokens[1:],
        masks[1:],
        spans[1:],
        policy=torch.nn.Identity(),
    )
    assert auxiliary.item() == 0.0
    assert all(
        value.shape
        == (
            (32, model.template_state()[name].shape[1])
            if name.endswith(".lora_A.default.weight")
            else (model.template_state()[name].shape[0], 32)
        )
        for name, value in generated.items()
    )


def test_layerwise_conditioner_step0_is_as139_graph() -> None:
    model, _ = _model()
    inputs = _inputs()
    actual = model.encode_program(*inputs, policy=torch.nn.Identity()).program
    frames, indices, video_offsets, condition_offsets, tokens, masks, spans = inputs
    video_bounds = tuple(int(value) for value in video_offsets.tolist())
    condition_bounds = tuple(int(value) for value in condition_offsets.tolist())
    counts = torch.tensor(
        [right - left for left, right in zip(condition_bounds, condition_bounds[1:])]
    )
    condition_ids = torch.repeat_interleave(torch.arange(len(counts)), counts)
    evidence = model.base_writer.encode_video_evidence(
        torch.nn.Identity(),
        frames,
        video_offsets,
        tokens.index_select(0, condition_ids),
        masks.index_select(0, condition_ids),
        spans.index_select(0, condition_ids),
    )
    memories = model.base_writer.build_memories(evidence, indices)
    compiler = model.base_writer.compiler
    shared_core = []
    for left, right in zip(condition_bounds, condition_bounds[1:]):
        core = memories.core[left:right].reshape(1, -1, 256)
        valid = memories.valid_core[left:right].reshape(1, -1)
        routing = compiler.routing(1)
        slots = compiler.read_core_slots(routing, core, valid)
        shared_core.append(compiler.normalize_core_slots(slots)[0])
    shared_core = torch.stack(shared_core)
    routing = compiler.routing(len(video_bounds) - 1)
    procedure, _ = compiler.read_procedure_slots(
        routing,
        shared_core.index_select(0, condition_ids),
        memories.procedure,
        memories.positions,
        memories.valid_procedure,
    )
    mean_procedure = torch.stack(
        [
            procedure[left:right].mean(dim=0)
            for left, right in zip(condition_bounds, condition_bounds[1:])
        ]
    )
    expected = compiler.fuse_readouts(
        compiler.routing(len(shared_core)), shared_core, mean_procedure
    )[0]
    assert torch.equal(actual, expected)


def test_layerwise_conditioner_constant_video_has_zero_dynamic_value() -> None:
    model, _ = _model()
    frames, indices, offsets, condition_offsets, tokens, masks, spans = _inputs()
    frames[:] = frames[0]
    encoded = model.encode_program(
        frames,
        indices,
        offsets,
        condition_offsets,
        tokens,
        masks,
        spans,
        policy=torch.nn.Identity(),
    )
    assert not encoded.diagnostics.per_video_query_conditioners.count_nonzero()
    assert encoded.parameter_grid is not None
    assert not encoded.parameter_grid.count_nonzero()
    assert encoded.residual_b_rows is not None
    assert all(not value.count_nonzero() for value in encoded.residual_b_rows.values())


def test_capacity_grid_is_exact_lpcp_at_zero_init() -> None:
    model, _ = _model()
    with torch.no_grad():
        model.query_delta.weight.normal_(std=0.01)
        encoded = model.encode_program(*_inputs(), policy=torch.nn.Identity())
        carrier = model.base_writer.decode_slots(encoded.program)
        public_template = model.template_state()
        lpcp = model.decode_program(encoded.program)
        committed = model.decode_output(encoded)
    assert sum(
        parameter.numel()
        for parameter in model.parameter_grid.branch.parameters()
    ) == 2_828_928
    assert all(torch.equal(committed[name], lpcp[name]) for name in lpcp)
    for name, value in model.base_writer.template_state().items():
        if name.endswith(".lora_A.default.weight"):
            assert torch.equal(public_template[name][..., :16, :], value)
            assert torch.equal(public_template[name][..., 16:, :], value)
        else:
            assert torch.equal(public_template[name][..., :16], value)
            assert not public_template[name][..., 16:].count_nonzero()
    for name, value in carrier.items():
        if name.endswith(".lora_A.default.weight"):
            assert torch.equal(lpcp[name][..., :16, :], value)
            assert torch.equal(lpcp[name][..., 16:, :], value)
        else:
            assert torch.equal(lpcp[name][..., :16], value)
            assert not lpcp[name][..., 16:].count_nonzero()
    for name, value in carrier.items():
        if not name.endswith(".lora_A.default.weight"):
            continue
        b_name = name.replace(".lora_A.", ".lora_B.")
        expected_ba = torch.matmul(
            carrier[b_name].double(), value.double()
        )
        actual_ba = torch.matmul(
            lpcp[b_name].double(), lpcp[name].double()
        )
        torch.testing.assert_close(actual_ba, expected_ba, rtol=0, atol=0)
    assert not model.parameter_grid.branch.payload_gate.count_nonzero()


def test_capacity_grid_is_k_set_invariant() -> None:
    model, _ = _model()
    with torch.no_grad():
        model.query_delta.weight.normal_(std=0.01)
        model.parameter_grid.branch.payload_gate.normal_(std=1e-3)
        natural = model.encode_program(*_inputs(), policy=torch.nn.Identity())
        frames, _, _, condition_offsets, tokens, masks, spans = _inputs()
        order = torch.tensor([3, 4, 0, 1, 2, 5, 6, 7])
        permuted = model.encode_program(
            frames[order],
            torch.tensor([0, 5, 0, 5, 10, 0, 5, 10]),
            torch.tensor([0, 2, 5, 8]),
            condition_offsets,
            tokens,
            masks,
            spans,
            policy=torch.nn.Identity(),
        )
    assert natural.parameter_grid is not None
    assert permuted.parameter_grid is not None
    assert natural.parameter_grid.count_nonzero()
    torch.testing.assert_close(natural.parameter_grid, permuted.parameter_grid)


def test_capacity_grid_changes_with_video_under_fixed_language() -> None:
    model, _ = _model()
    with torch.no_grad():
        model.parameter_grid.branch.payload_gate.normal_(std=1e-3)
        first = model.encode_program(*_inputs(), policy=torch.nn.Identity())
        inputs = list(_inputs())
        inputs[0] = inputs[0].roll(1, dims=0)
        second = model.encode_program(*inputs, policy=torch.nn.Identity())
    assert first.parameter_grid is not None
    assert second.parameter_grid is not None
    assert first.parameter_grid.shape == (2, 18, 37, 1024)
    assert not torch.equal(first.parameter_grid, second.parameter_grid)


def test_capacity_grid_is_gradient_open_and_requires_video() -> None:
    model, _ = _model()
    grid = model.parameter_grid.requires_grad_(True)
    state = model.encode_conditioning_state(*_inputs(), policy=torch.nn.Identity())
    rows, value = grid(
        state.layer_memory_states,
        state.frame_indices,
        state.video_bounds,
        (0, 2, 3),
    )
    assert not value.count_nonzero()
    sum(item.float().sum() for item in rows.values()).backward()
    gate = grid.branch.payload_gate
    assert gate.grad is not None and gate.grad.count_nonzero()
    assert grid.branch.memory_tokens.grad is not None
    assert not grid.branch.memory_tokens.grad.count_nonzero()
    with torch.no_grad():
        gate.normal_(std=1e-3)
    grid.zero_grad(set_to_none=True)
    state = model.encode_conditioning_state(*_inputs(), policy=torch.nn.Identity())
    rows, _ = grid(
        state.layer_memory_states,
        state.frame_indices,
        state.video_bounds,
        (0, 2, 3),
    )
    sum(item.float().sum() for item in rows.values()).backward()
    assert all(
        parameter.grad is not None and parameter.grad.count_nonzero()
        for parameter in grid.branch.parameters()
    )

    constant = state.layer_memory_states[:1].expand(4, -1, -1, -1).clone()
    with torch.no_grad():
        zero_rows, zero_grid = grid(
            constant,
            torch.tensor([0, 5, 10, 15]),
            (0, 4),
            (0, 1),
        )
    assert not zero_grid.count_nonzero()
    assert all(not item.count_nonzero() for item in zero_rows.values())


def test_capacity_grid_emits_only_residual_bank_rows() -> None:
    model, _ = _model()
    generator = torch.Generator(device="cpu").manual_seed(47)
    grid = torch.randn(2, 18, 37, 1024, generator=generator)
    with torch.no_grad():
        rows = model.parameter_grid.rows(grid)
        state = model._direct_b_residual_state(rows, batch=2)
    assert rows["q_b"].shape == (2, 18, 16, 2048)
    assert rows["v_b"].shape == (2, 18, 16, 256)
    assert rows["action_in_b"].shape == (2, 16, 1024)
    assert rows["action_out_b"].shape == (2, 16, 32)
    assert len(state) == 38
    assert all(name.endswith(".lora_B.default.weight") for name in state)
    assert all(
        state[name].shape == (2, *value.shape)
        for name, value in model.base_writer.template_state().items()
        if name in state
    )
    assert all(value.count_nonzero() for value in state.values())

    with torch.no_grad():
        model.parameter_grid.branch.payload_gate.normal_(std=1e-3)
        encoded = model.encode_program(*_inputs(), policy=torch.nn.Identity())
        baseline = model.decode_program(encoded.program)
        committed = model.decode_output(encoded)
    assert all(
        torch.equal(committed[name], value)
        for name, value in baseline.items()
        if name.endswith(".lora_A.default.weight")
    )
    assert any(
        not torch.equal(committed[name], value)
        for name, value in baseline.items()
        if name.endswith(".lora_B.default.weight")
    )
    for name, value in baseline.items():
        if name.endswith(".lora_B.default.weight"):
            assert torch.equal(committed[name][..., :16], value[..., :16])


def test_capacity_grid_lpcp_loader_rejects_partial_new_topology() -> None:
    source, _ = _model()
    old = {
        name: value
        for name, value in source.state_dict().items()
        if not name.startswith("parameter_grid.")
    }
    loaded, _ = _model()
    loaded.load_lpcp_state_(old)
    assert all(
        torch.equal(loaded.state_dict()[name], value)
        for name, value in old.items()
    )
    partial = dict(old)
    partial["parameter_grid.branch.payload_gate"] = source.state_dict()[
        "parameter_grid.branch.payload_gate"
    ]
    with torch.no_grad():
        try:
            loaded.load_lpcp_state_(partial)
        except RuntimeError:
            pass
        else:
            raise AssertionError("partial parameter-grid topology was accepted")
    with torch.no_grad():
        try:
            loaded.load_lpcp_state_(source.state_dict())
        except RuntimeError:
            pass
        else:
            raise AssertionError("trained parameter grid was accepted as LPCP cold start")


def test_capacity_grid_records_backbone_memory_in_the_same_forward() -> None:
    model, _ = _model()
    with torch.no_grad():
        state = model.encode_conditioning_state(
            *_inputs(), policy=torch.nn.Identity()
        )
    assert state.layer_memory_states.shape == (8, 18, 37, 1024)
    assert state.layer_memory_states.data_ptr() != (
        state.per_video_query_conditioners.data_ptr()
    )
    assert state.video_bounds == (0, 3, 5, 8)


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
            for index in range(Pi05CapacityMatchedBackboneMemory.LAYERS)
        )
        self.norm = _MemoryNorm()
        self.rotary_emb = _MemoryRotary()


class _MemoryCore(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.paligemma_with_expert = SimpleNamespace(
            paligemma=SimpleNamespace(
                model=SimpleNamespace(language_model=_MemoryBackbone(width))
            ),
            gemma_expert=SimpleNamespace(model=_MemoryBackbone(width)),
        )

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
    memory = torch.nn.Parameter(torch.randn(37, width))
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
        Pi05CapacityMatchedBackboneMemory(
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
        padding, action_horizon=50, memory_tokens=37
    )[0]
    action = 3
    memory = 53
    assert mask.shape == (90, 90)
    assert mask[0, :3].tolist() == [True, True, False]
    assert not mask[0, action:].any()
    assert mask[action, :memory].sum().item() == 52
    assert not mask[action, memory:].any()
    assert mask[memory, :].sum().item() == 89
    assert not mask[2].any()
    assert not mask[:, 2].any()


def test_joint_loop_captures_all_action_and_memory_layers() -> None:
    torch.manual_seed(5)
    core = _MemoryCore(4)
    loop, vl, action = _memory_loop(core)
    output = loop(core, *_memory_inputs(), vl, action)
    assert output.prefix_hidden.shape == (2, 5, 4)
    assert output.action_hidden.shape == (2, 50, 4)
    assert output.action_layer_states.shape == (2, 18, 50, 4)
    assert output.layer_memory.shape == (2, 18, 37, 4)
    assert not torch.equal(output.layer_memory[:, 0], output.layer_memory[:, -1])


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
    torch.testing.assert_close(
        first.action_layer_states, second.action_layer_states, rtol=0, atol=0
    )
    assert not torch.equal(first.layer_memory, second.layer_memory)


def test_checkpointed_joint_loop_reaches_memory_tokens_only() -> None:
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
