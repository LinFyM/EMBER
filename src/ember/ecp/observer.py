"""Policy-native all-layer observation for EMBER-ECP."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import torch

from ember.ecp.contracts import ACTION_HORIZON, TargetFamily, TargetOwner


@dataclass(frozen=True)
class NativeObserverOutput:
    """Compact target-aligned evidence; full 1024-wide states are not retained."""

    owner_lattice: torch.Tensor
    flow_velocity: torch.Tensor


class _LayerStateCapture(AbstractContextManager["_LayerStateCapture"]):
    def __init__(self, expert_model: torch.nn.Module, *, detach: bool) -> None:
        self.expert_model = expert_model
        self.detach = detach
        self.values: list[torch.Tensor | None] = [None] * (
            len(expert_model.layers) + 1
        )
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "_LayerStateCapture":
        modules = [layer.input_layernorm for layer in self.expert_model.layers]
        modules.append(self.expert_model.norm)
        for index, module in enumerate(modules):
            def hook(
                _module: torch.nn.Module,
                inputs: tuple[torch.Tensor, ...],
                *,
                selected: int = index,
            ) -> None:
                value = inputs[0]
                self.values[selected] = value.detach() if self.detach else value

            self.handles.append(module.register_forward_pre_hook(hook))
        return self

    def stacked(self) -> torch.Tensor:
        if any(value is None for value in self.values):
            raise RuntimeError("PI0.5 Action Expert layer capture is incomplete")
        return torch.stack([value for value in self.values if value is not None], dim=1)

    def __exit__(self, *args: object) -> None:
        for handle in self.handles:
            handle.remove()


class TargetOwnerProjector(torch.nn.Module):
    """Project all native layer inputs and residual increments onto 38 owners."""

    def __init__(
        self,
        owners: tuple[TargetOwner, ...],
        *,
        expert_width: int,
        program_width: int,
        padded_action_dim: int,
    ) -> None:
        super().__init__()
        self.owners = owners
        self.state_projection = torch.nn.Linear(expert_width, program_width, bias=False)
        self.delta_projection = torch.nn.Linear(expert_width, program_width, bias=False)
        self.velocity_projection = torch.nn.Linear(
            padded_action_dim, program_width, bias=False
        )
        self.layer_embedding = torch.nn.Embedding(18, program_width)
        self.family_embedding = torch.nn.Embedding(len(TargetFamily), program_width)
        self.family_gate = torch.nn.Parameter(torch.ones(len(TargetFamily), 2))
        self.output_norm = torch.nn.LayerNorm(program_width)
        family_order = tuple(TargetFamily)
        self.register_buffer(
            "family_ids",
            torch.tensor([family_order.index(owner.family) for owner in owners]),
            persistent=False,
        )
        self.register_buffer(
            "owner_layer_ids",
            torch.tensor(
                [
                    owner.layer
                    if owner.layer is not None
                    else (17 if owner.family is TargetFamily.ACTION_OUT else 0)
                    for owner in owners
                ]
            ),
            persistent=False,
        )
        self.register_buffer(
            "layer_owner_mask",
            torch.tensor([owner.layer is not None for owner in owners]),
            persistent=False,
        )
        self.register_buffer(
            "action_out_mask",
            torch.tensor(
                [owner.family is TargetFamily.ACTION_OUT for owner in owners]
            ),
            persistent=False,
        )

    def forward(
        self,
        layer_states: torch.Tensor,
        flow_velocity: torch.Tensor,
    ) -> torch.Tensor:
        layer_inputs = layer_states[:, :-1]
        residuals = layer_states[:, 1:] - layer_inputs
        state = self.state_projection(layer_inputs)
        delta = self.delta_projection(residuals)
        velocity = self.velocity_projection(flow_velocity)
        gates = self.family_gate[self.family_ids]
        owner_state = state[:, self.owner_layer_ids]
        owner_delta = delta[:, self.owner_layer_ids]
        value = (
            gates[None, :, None, 0, None] * owner_state
            + gates[None, :, None, 1, None] * owner_delta
        )
        action_out = (
            gates[None, :, None, 0, None] * state[:, -1, None]
            + gates[None, :, None, 1, None] * velocity[:, None]
        )
        value = torch.where(
            self.action_out_mask[None, :, None, None], action_out, value
        )
        layer_bias = self.layer_embedding(self.owner_layer_ids)
        value = value + (
            layer_bias * self.layer_owner_mask[:, None]
        )[None, :, None]
        family_bias = self.family_embedding(self.family_ids)[None, :, None]
        return self.output_norm(value + family_bias)


class ECPNativeObserver(torch.nn.Module):
    """Run the native PI0.5 prefix/suffix block and compact every Action layer."""

    def __init__(
        self,
        owners: tuple[TargetOwner, ...],
        *,
        expert_width: int = 1024,
        program_width: int = 128,
        padded_action_dim: int = 32,
    ) -> None:
        super().__init__()
        self.projector = TargetOwnerProjector(
            owners,
            expert_width=expert_width,
            program_width=program_width,
            padded_action_dim=padded_action_dim,
        )

    def forward(
        self,
        core: torch.nn.Module,
        prefix_embeddings: torch.Tensor,
        prefix_padding: torch.Tensor,
        suffix_noise: torch.Tensor,
        flow_time: torch.Tensor,
        *,
        track_action_adapter_grad: bool = False,
        action_adapter_context: AbstractContextManager[None] | None = None,
    ) -> NativeObserverOutput:
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        bridge = core.paligemma_with_expert
        expert_model = bridge.gemma_expert.model
        suffix, suffix_padding, suffix_attention, adarms = core.embed_suffix(
            suffix_noise, flow_time
        )
        padding = torch.cat((prefix_padding, suffix_padding), dim=1)
        attention = torch.cat((torch.zeros_like(prefix_padding), suffix_attention), dim=1)
        mask = core._prepare_attention_masks_4d(make_att_2d_masks(padding, attention))
        positions = torch.cumsum(padding, dim=1) - 1
        target_dtype = expert_model.layers[0].self_attn.q_proj.weight.dtype
        grad_context = (
            torch.enable_grad() if track_action_adapter_grad else torch.no_grad()
        )
        adapter_context = action_adapter_context or nullcontext()
        with grad_context, adapter_context, _LayerStateCapture(
            expert_model, detach=not track_action_adapter_grad
        ) as capture:
            (_, suffix_hidden), _ = bridge.forward(
                attention_mask=mask,
                position_ids=positions,
                past_key_values=None,
                inputs_embeds=[
                    prefix_embeddings.to(target_dtype),
                    suffix.to(target_dtype),
                ],
                use_cache=False,
                adarms_cond=[None, adarms],
            )
            layer_states = capture.stacked()
            flow_velocity = core.action_out_proj(suffix_hidden)

        if layer_states.shape[1:3] != (19, ACTION_HORIZON):
            raise RuntimeError("PI0.5 Action Expert observer topology changed")
        lattice = self.projector(layer_states, flow_velocity)
        return NativeObserverOutput(
            owner_lattice=lattice,
            flow_velocity=flow_velocity.detach(),
        )
