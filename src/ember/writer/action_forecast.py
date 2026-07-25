"""Differentiable pi0.5 action forecasts from one raw teaching video."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ember.pi05_processing import PI05_STATE_ANCHOR_TOKEN_IDS
from ember.writer.visual_state import (
    AnchoredVisualState,
    frozen_digit_embedding_basis,
)


class ActionForecastError(RuntimeError):
    """Raised when the sealed teacher-forecast interface changes."""


class MetaLoRAProjection(torch.nn.Module):
    """Writer-owned identity-initialized LoRA for one frozen projection."""

    def __init__(self, input_width: int, output_width: int, rank: int) -> None:
        super().__init__()
        if min(input_width, output_width, rank) <= 0:
            raise ActionForecastError("invalid Meta-LoRA dimensions")
        self.a = torch.nn.Parameter(torch.empty(rank, input_width))
        self.b = torch.nn.Parameter(torch.zeros(output_width, rank))
        torch.nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        adapted = F.linear(F.linear(value.to(self.a.dtype), self.a), self.b)
        return adapted.to(value.dtype)


class MetaLoRAStack(torch.nn.Module):
    """A fixed q/k/v/o Meta-LoRA stack installed only on the teacher path."""

    PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")

    def __init__(self, layers: Sequence[torch.nn.Module], rank: int) -> None:
        super().__init__()
        if not layers or rank <= 0:
            raise ActionForecastError("invalid Meta-LoRA stack")
        self.layer_count = len(layers)
        adapters: dict[str, torch.nn.Module] = {}
        for layer_index, layer in enumerate(layers):
            for name in self.PROJECTIONS:
                projection = getattr(layer.self_attn, name)
                input_width = int(getattr(projection, "in_features", 0))
                output_width = int(getattr(projection, "out_features", 0))
                if min(input_width, output_width) <= 0:
                    raise ActionForecastError(
                        f"Meta-LoRA projection changed shape: {layer_index}.{name}"
                    )
                adapters[f"{layer_index}_{name}"] = MetaLoRAProjection(
                    input_width,
                    output_width,
                    rank,
                )
        self.adapters = torch.nn.ModuleDict(adapters)

    @contextmanager
    def installed(self, model: torch.nn.Module) -> Iterator[None]:
        layers = model.layers
        if len(layers) != self.layer_count:
            raise ActionForecastError("Meta-LoRA layer count changed")
        handles = []
        for layer_index, layer in enumerate(layers):
            for name in self.PROJECTIONS:
                adapter = self.adapters[f"{layer_index}_{name}"]
                projection = getattr(layer.self_attn, name)

                def hook(
                    _module: torch.nn.Module,
                    inputs: tuple[torch.Tensor, ...],
                    output: torch.Tensor,
                    *,
                    selected: MetaLoRAProjection = adapter,
                ) -> torch.Tensor:
                    return output + selected(inputs[0]).to(output.dtype)

                handles.append(projection.register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()


class Pi05ActionForecastEncoder(torch.nn.Module):
    """Run contextual teacher-view pi0.5 inference for every sampled frame."""

    def __init__(
        self,
        *,
        paligemma_model: torch.nn.Module,
        expert_model: torch.nn.Module,
        image_width: int,
        state_width: int,
        state_slots: int,
        state_coordinates: int,
        state_heads: int,
        vl_meta_lora_rank: int,
        action_meta_lora_rank: int,
        frame_microbatch_size: int,
        num_flow_steps: int,
        action_horizon: int,
        padded_action_dim: int,
        output_action_dim: int,
        initialization_seed: int,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        dimensions = (
            image_width,
            state_width,
            state_slots,
            state_coordinates,
            state_heads,
            vl_meta_lora_rank,
            action_meta_lora_rank,
            frame_microbatch_size,
            num_flow_steps,
            action_horizon,
            padded_action_dim,
            output_action_dim,
        )
        if (
            any(value <= 0 for value in dimensions)
            or output_action_dim > padded_action_dim
            or state_slots != AnchoredVisualState.STATE_SLOTS
            or state_coordinates != AnchoredVisualState.COORDINATES
        ):
            raise ActionForecastError("invalid pi0.5 action-forecast dimensions")
        self.frame_microbatch_size = int(frame_microbatch_size)
        self.num_flow_steps = int(num_flow_steps)
        self.action_horizon = int(action_horizon)
        self.padded_action_dim = int(padded_action_dim)
        self.output_action_dim = int(output_action_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.state_slots = int(state_slots)
        digit_basis = frozen_digit_embedding_basis(
            paligemma_model,
            image_width=image_width,
        )
        self.visual_state = AnchoredVisualState(
            image_width=image_width,
            state_width=state_width,
            heads=state_heads,
            digit_basis=digit_basis,
            initialization_seed=initialization_seed,
        )
        self.register_buffer(
            "state_anchor_token_ids",
            torch.tensor(PI05_STATE_ANCHOR_TOKEN_IDS, dtype=torch.long),
            persistent=True,
        )
        self.vl_meta_lora = MetaLoRAStack(
            paligemma_model.layers,
            vl_meta_lora_rank,
        )
        self.action_meta_lora = MetaLoRAStack(
            expert_model.layers,
            action_meta_lora_rank,
        )

    @staticmethod
    def _prepare_images(frames: torch.Tensor) -> torch.Tensor:
        from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

        if (
            frames.ndim != 4
            or frames.shape[0] <= 0
            or frames.shape[1] != 3
            or frames.dtype != torch.uint8
        ):
            raise ActionForecastError("teacher frames changed shape or dtype")
        value = frames.to(torch.float32).div_(255.0).permute(0, 2, 3, 1)
        value = resize_with_pad_torch(value, 224, 224)
        return (value * 2.0 - 1.0).permute(0, 3, 1, 2)

    @staticmethod
    def _denoise_step(
        core: torch.nn.Module,
        *,
        prefix_padding: torch.Tensor,
        prefix_cache: object,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        from lerobot.policies.pi05.modeling_pi05 import (
            clone_past_key_values,
            make_att_2d_masks,
        )

        suffix, suffix_padding, suffix_attention, adarms = core.embed_suffix(
            x_t,
            timestep,
        )
        suffix_length = suffix_padding.shape[1]
        prefix_length = prefix_padding.shape[1]
        batch = prefix_padding.shape[0]
        prefix_visible = prefix_padding[:, None, :].expand(
            batch,
            suffix_length,
            prefix_length,
        )
        suffix_2d = make_att_2d_masks(suffix_padding, suffix_attention)
        full_2d = torch.cat((prefix_visible, suffix_2d), dim=2)
        positions = (
            prefix_padding.sum(dim=-1, keepdim=True)
            + torch.cumsum(suffix_padding, dim=1)
            - 1
        )
        outputs, _ = core.paligemma_with_expert.forward(
            attention_mask=core._prepare_attention_masks_4d(full_2d),
            position_ids=positions,
            past_key_values=clone_past_key_values(prefix_cache),
            inputs_embeds=[None, suffix],
            use_cache=False,
            adarms_cond=[None, adarms],
        )
        return core.action_out_proj(
            outputs[1][:, -core.config.chunk_size :].to(torch.float32)
        )

    def _forecast_microbatch(
        self,
        core: torch.nn.Module,
        unique_frames: torch.Tensor,
        current_map: torch.Tensor,
        anchor_map: torch.Tensor,
        previous_map: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        state_positions: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        bridge = core.paligemma_with_expert
        language_model = bridge.paligemma.model.language_model
        expert_model = bridge.gemma_expert.model
        images = self._prepare_images(unique_frames)
        with torch.no_grad():
            unique_image_tokens = bridge.embed_image(images)
            text_tokens = bridge.embed_language_tokens(language_tokens)

        image_tokens = unique_image_tokens.index_select(0, current_map)
        anchor_tokens = unique_image_tokens.index_select(0, anchor_map)
        previous_tokens = unique_image_tokens.index_select(0, previous_map)
        state_offsets = self.visual_state(
            image_tokens,
            anchor_tokens,
            previous_tokens,
        )
        observed_anchor = torch.gather(
            language_tokens,
            1,
            state_positions,
        )
        expected_anchor = self.state_anchor_token_ids[None].expand_as(
            observed_anchor
        )
        if not torch.equal(observed_anchor, expected_anchor):
            raise ActionForecastError("native visual-state token anchor changed")
        state_values = torch.zeros_like(text_tokens).scatter(
            1,
            state_positions[..., None].expand(-1, -1, text_tokens.shape[-1]),
            state_offsets.to(text_tokens.dtype),
        )
        text_tokens = text_tokens + state_values
        prefix = torch.cat((image_tokens, text_tokens), dim=1)
        prefix_padding = torch.cat(
            (
                torch.ones(
                    image_tokens.shape[:2],
                    dtype=torch.bool,
                    device=unique_frames.device,
                ),
                language_mask,
            ),
            dim=1,
        )
        prefix_attention = torch.zeros_like(prefix_padding)
        prefix_positions = torch.cumsum(prefix_padding, dim=1) - 1
        prefix_2d = make_att_2d_masks(prefix_padding, prefix_attention)
        target_dtype = language_model.layers[0].self_attn.q_proj.weight.dtype
        with self.vl_meta_lora.installed(language_model):
            (_, _), prefix_cache = bridge.forward(
                attention_mask=core._prepare_attention_masks_4d(prefix_2d),
                position_ids=prefix_positions,
                past_key_values=None,
                inputs_embeds=[prefix.to(target_dtype), None],
                use_cache=True,
            )

        x_t = noise.to(torch.float32)
        delta = -1.0 / self.num_flow_steps
        with self.action_meta_lora.installed(expert_model):
            for step in range(self.num_flow_steps):
                timestep = torch.full(
                    (current_map.shape[0],),
                    1.0 + step * delta,
                    dtype=torch.float32,
                    device=unique_frames.device,
                )
                velocity = self._denoise_step(
                    core,
                    prefix_padding=prefix_padding,
                    prefix_cache=prefix_cache,
                    x_t=x_t,
                    timestep=timestep,
                )
                x_t = x_t + delta * velocity
        return x_t[..., : self.output_action_dim]

    def _validate_forward_batch(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        state_positions: torch.Tensor,
        flow_noise: torch.Tensor,
    ) -> tuple[torch.nn.Module, torch.Tensor]:
        conditions = language_tokens.shape[0]
        if (
            frames.ndim != 4
            or frames.shape[0] <= 0
            or frame_condition_ids.ndim != 1
            or frame_condition_ids.shape[0] != frames.shape[0]
            or frame_condition_ids.dtype != torch.long
            or language_tokens.ndim != 2
            or language_mask.shape != language_tokens.shape
            or language_mask.dtype != torch.bool
            or state_positions.shape != (conditions, self.state_slots)
            or flow_noise.shape
            != (
                conditions,
                self.action_horizon,
                self.padded_action_dim,
            )
            or int(frame_condition_ids.min()) < 0
            or int(frame_condition_ids.max()) >= conditions
        ):
            raise ActionForecastError("invalid frame-language forecast batch")
        counts = torch.bincount(frame_condition_ids, minlength=conditions)
        expected = torch.repeat_interleave(
            torch.arange(conditions, device=frames.device),
            counts,
        )
        if bool((counts <= 0).any()) or not torch.equal(
            frame_condition_ids,
            expected,
        ):
            raise ActionForecastError(
                "forecast frames must be contiguous by video condition"
            )
        core = policy.model
        if (
            int(core.config.chunk_size) != self.action_horizon
            or int(core.config.max_action_dim) != self.padded_action_dim
        ):
            raise ActionForecastError("pi0.5 action forecast topology changed")
        return core, counts

    @staticmethod
    def _context_rows(
        frame_condition_ids: torch.Tensor,
        counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        starts = torch.cumsum(counts, dim=0) - counts
        frame_rows = torch.arange(
            frame_condition_ids.shape[0],
            device=frame_condition_ids.device,
        )
        anchor_rows = starts.index_select(0, frame_condition_ids)
        previous_rows = torch.maximum(frame_rows - 1, anchor_rows)
        return frame_rows, anchor_rows, previous_rows

    def _microbatch_arguments(
        self,
        *,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        frame_rows: torch.Tensor,
        anchor_rows: torch.Tensor,
        previous_rows: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        state_positions: torch.Tensor,
        flow_noise: torch.Tensor,
        start: int,
    ) -> tuple[int, tuple[torch.Tensor, ...]]:
        stop = min(start + self.frame_microbatch_size, frames.shape[0])
        real_count = stop - start
        current_rows = frame_rows[start:stop]
        if real_count < self.frame_microbatch_size:
            current_rows = torch.cat(
                (
                    current_rows,
                    current_rows[-1:].expand(
                        self.frame_microbatch_size - real_count
                    ),
                )
            )
        selected = frame_condition_ids.index_select(0, current_rows)
        selected_anchor = anchor_rows.index_select(0, current_rows)
        selected_previous = previous_rows.index_select(0, current_rows)
        unique_rows, inverse = torch.unique(
            torch.cat((current_rows, selected_anchor, selected_previous)),
            sorted=True,
            return_inverse=True,
        )
        current_map, anchor_map, previous_map = inverse.split(
            self.frame_microbatch_size
        )
        return real_count, (
            frames.index_select(0, unique_rows),
            current_map,
            anchor_map,
            previous_map,
            language_tokens.index_select(0, selected),
            language_mask.index_select(0, selected),
            state_positions.index_select(0, selected),
            flow_noise.index_select(0, selected),
        )

    def forward(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        state_positions: torch.Tensor,
        flow_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Return normalized final plans ``[sum_T,50,7]``."""

        core, counts = self._validate_forward_batch(
            policy,
            frames,
            frame_condition_ids,
            language_tokens,
            language_mask,
            state_positions,
            flow_noise,
        )
        frame_rows, anchor_rows, previous_rows = self._context_rows(
            frame_condition_ids,
            counts,
        )
        results = []
        for start in range(0, frames.shape[0], self.frame_microbatch_size):
            real_count, arguments = self._microbatch_arguments(
                frames=frames,
                frame_condition_ids=frame_condition_ids,
                frame_rows=frame_rows,
                anchor_rows=anchor_rows,
                previous_rows=previous_rows,
                language_tokens=language_tokens,
                language_mask=language_mask,
                state_positions=state_positions,
                flow_noise=flow_noise,
                start=start,
            )

            def invoke(*values: torch.Tensor) -> torch.Tensor:
                return self._forecast_microbatch(core, *values)

            if (
                self.activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                value = checkpoint(
                    invoke,
                    *arguments,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                value = invoke(*arguments)
            results.append(value[:real_count])
        return torch.cat(results, dim=0)
