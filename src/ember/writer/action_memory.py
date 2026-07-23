"""Frozen PI05 prefix and Action-Expert memory extraction for the Writer."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F


class ActionMemoryError(RuntimeError):
    """Raised when PI05 cannot provide the sealed Action-Memory interface."""


def _orthogonal_action_codes(count: int, width: int) -> torch.Tensor:
    """Build deterministic orthogonal codes from a Sylvester Hadamard matrix."""

    if count <= 0 or width <= 0 or width & (width - 1) or count > width:
        raise ActionMemoryError("Action-Memory code dimensions must fit a power of two")
    value = torch.ones(1, 1, dtype=torch.float32)
    while value.shape[0] < width:
        value = torch.cat(
            (
                torch.cat((value, value), dim=1),
                torch.cat((value, -value), dim=1),
            ),
            dim=0,
        )
    return value[:count].div_(math.sqrt(width))


class MetaLoRAProjection(torch.nn.Module):
    """Writer-owned identity-initialized LoRA for one frozen expert projection."""

    def __init__(self, input_width: int, output_width: int, rank: int) -> None:
        super().__init__()
        if min(input_width, output_width, rank) <= 0:
            raise ActionMemoryError("invalid encoder Meta-LoRA dimensions")
        self.a = torch.nn.Parameter(torch.empty(rank, input_width))
        self.b = torch.nn.Parameter(torch.zeros(output_width, rank))
        torch.nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(value, self.a), self.b)


class Pi05ActionMemoryEncoder(torch.nn.Module):
    """Encode frame-text pairs with frozen PaliGemma and Action Expert states.

    The trainable Meta-LoRA is installed only while producing teacher-video
    memory states.  It is removed before the generated public task LoRA is used
    by the execution policy.
    """

    def __init__(
        self,
        *,
        action_in_projection: torch.nn.Module,
        memory_slots: int,
        expert_layers: int,
        expert_width: int,
        action_code_width: int,
        meta_rank: int,
    ) -> None:
        super().__init__()
        if (
            memory_slots <= 0
            or expert_layers <= 0
            or expert_width <= 0
            or action_code_width <= 0
            or meta_rank <= 0
        ):
            raise ActionMemoryError("invalid PI05 Action-Memory dimensions")
        codes = _orthogonal_action_codes(memory_slots, action_code_width)
        try:
            reference = next(action_in_projection.parameters())
        except StopIteration as error:
            raise ActionMemoryError("PI05 action input projection has no parameters") from error
        with torch.no_grad():
            initial = action_in_projection(
                codes.to(device=reference.device, dtype=reference.dtype)
            )
        if initial.shape != (memory_slots, expert_width):
            raise ActionMemoryError("PI05 action input projection changed shape")
        self.memory_slots = int(memory_slots)
        self.expert_layers = int(expert_layers)
        self.expert_width = int(expert_width)
        self.memory_tokens = torch.nn.Parameter(
            initial.detach().to(device="cpu", dtype=torch.float32).contiguous()
        )
        self.memory_timestep_logit = torch.nn.Parameter(torch.zeros(()))

        self.meta_lora = torch.nn.ModuleDict()
        for layer in range(expert_layers):
            for name, input_width, output_width in (
                ("q_proj", expert_width, 2 * expert_width),
                ("k_proj", expert_width, expert_width // 4),
                ("v_proj", expert_width, expert_width // 4),
                ("o_proj", 2 * expert_width, expert_width),
            ):
                self.meta_lora[f"{layer}_{name}"] = MetaLoRAProjection(
                    input_width, output_width, meta_rank
                )

    @contextmanager
    def _installed_meta_lora(
        self, expert_model: torch.nn.Module
    ) -> Iterator[None]:
        handles = []
        layers = expert_model.layers
        if len(layers) != self.expert_layers:
            raise ActionMemoryError("PI05 Action Expert layer count changed")
        for layer_index, layer in enumerate(layers):
            for projection_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                module = getattr(layer.self_attn, projection_name)
                adapter = self.meta_lora[f"{layer_index}_{projection_name}"]

                def hook(
                    _module: torch.nn.Module,
                    inputs: tuple[torch.Tensor, ...],
                    output: torch.Tensor,
                    *,
                    selected: MetaLoRAProjection = adapter,
                ) -> torch.Tensor:
                    return output + selected(inputs[0]).to(output.dtype)

                handles.append(module.register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @staticmethod
    def _prepare_images(frames: torch.Tensor) -> torch.Tensor:
        from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

        if (
            frames.ndim != 4
            or frames.shape[1] != 3
            or frames.shape[0] < 1
            or frames.dtype != torch.uint8
        ):
            raise ActionMemoryError("teacher video frames changed shape or dtype")
        value = frames.to(torch.float32).div_(255.0).permute(0, 2, 3, 1)
        value = resize_with_pad_torch(value, 224, 224)
        return (value * 2.0 - 1.0).permute(0, 3, 1, 2)

    def _encode_microbatch(
        self,
        core: torch.nn.Module,
        bridge: torch.nn.Module,
        expert: torch.nn.Module,
        frames: torch.Tensor,
        tokens: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        from lerobot.policies.pi05.modeling_pi05 import (
            clone_past_key_values,
            create_sinusoidal_pos_embedding,
            make_att_2d_masks,
        )

        count = frames.shape[0]
        images = self._prepare_images(frames)
        image_masks = [
            torch.ones(count, dtype=torch.bool, device=frames.device)
        ]
        with torch.no_grad():
            prefix, prefix_padding, prefix_attention = core.embed_prefix(
                [images], image_masks, tokens, masks
            )
            prefix = prefix.to(torch.bfloat16)
            prefix_2d = make_att_2d_masks(prefix_padding, prefix_attention)
            prefix_positions = torch.cumsum(prefix_padding, dim=1) - 1
            prefix_4d = core._prepare_attention_masks_4d(prefix_2d)
            bridge.paligemma.model.language_model.config._attn_implementation = (
                "eager"
            )
            (_, _), prefix_cache = bridge.forward(
                attention_mask=prefix_4d,
                position_ids=prefix_positions,
                past_key_values=None,
                inputs_embeds=[prefix, None],
                use_cache=True,
            )

        memory = self.memory_tokens[None].expand(count, -1, -1)
        timestep = torch.sigmoid(self.memory_timestep_logit).expand(count)
        time_embedding = create_sinusoidal_pos_embedding(
            timestep,
            self.expert_width,
            min_period=core.config.min_period,
            max_period=core.config.max_period,
            device=timestep.device,
        ).to(memory.dtype)
        time_embedding = F.silu(core.time_mlp_in(time_embedding))
        time_embedding = F.silu(core.time_mlp_out(time_embedding))

        suffix_padding = torch.ones(
            count,
            self.memory_slots,
            dtype=torch.bool,
            device=frames.device,
        )
        suffix_attention = torch.zeros_like(suffix_padding)
        suffix_attention[:, 0] = True
        suffix_2d = make_att_2d_masks(suffix_padding, suffix_attention)
        prefix_visible = prefix_padding[:, None, :].expand(
            count, self.memory_slots, prefix_padding.shape[1]
        )
        full_2d = torch.cat((prefix_visible, suffix_2d), dim=-1)
        full_4d = core._prepare_attention_masks_4d(full_2d)
        positions = (
            prefix_padding.sum(dim=-1, keepdim=True)
            + torch.cumsum(suffix_padding, dim=1)
            - 1
        )
        expert.config._attn_implementation = "eager"
        output = expert(
            inputs_embeds=memory.to(torch.bfloat16),
            attention_mask=full_4d,
            position_ids=positions,
            past_key_values=clone_past_key_values(prefix_cache),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            adarms_cond=time_embedding,
        )
        hidden = output.hidden_states
        if hidden is None or len(hidden) != self.expert_layers + 1:
            raise ActionMemoryError(
                "PI05 Action Expert did not expose every layer state"
            )
        return torch.stack(hidden[1:], dim=1)

    def forward(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        *,
        frame_microbatch: int,
    ) -> torch.Tensor:
        """Return ``[sum_T, expert_layers, memory_slots, expert_width]``."""

        if (
            frame_condition_ids.ndim != 1
            or frame_condition_ids.shape[0] != frames.shape[0]
            or frame_condition_ids.dtype != torch.long
            or language_tokens.ndim != 2
            or language_mask.shape != language_tokens.shape
            or language_mask.dtype != torch.bool
            or language_tokens.shape[0] < 1
            or frame_microbatch <= 0
            or int(frame_condition_ids.min()) < 0
            or int(frame_condition_ids.max()) >= language_tokens.shape[0]
        ):
            raise ActionMemoryError("invalid frame-language Action-Memory batch")

        core = policy.model
        bridge = core.paligemma_with_expert
        expert = bridge.gemma_expert.model
        previous_training = expert.training
        expert.eval()
        results = []
        try:
            with self._installed_meta_lora(expert):
                for start in range(0, frames.shape[0], frame_microbatch):
                    stop = min(start + frame_microbatch, frames.shape[0])
                    condition_ids = frame_condition_ids[start:stop]
                    results.append(
                        self._encode_microbatch(
                            core,
                            bridge,
                            expert,
                            frames[start:stop],
                            language_tokens.index_select(0, condition_ids),
                            language_mask.index_select(0, condition_ids),
                        )
                    )
        finally:
            expert.train(previous_training)
        return torch.cat(results, dim=0)
