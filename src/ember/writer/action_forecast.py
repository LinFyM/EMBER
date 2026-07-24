"""Differentiable pi0.5 action forecasts from one raw teaching video."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


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


class QueryReadBlock(torch.nn.Module):
    """Pre-norm queries that read a memory without a memory-side update."""

    def __init__(self, width: int, heads: int, expansion: int) -> None:
        super().__init__()
        if min(width, heads, expansion) <= 0 or width % heads:
            raise ActionForecastError("invalid query-read block")
        self.query_norm = torch.nn.LayerNorm(width)
        self.memory_norm = torch.nn.LayerNorm(width)
        self.attention = torch.nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn_norm = torch.nn.LayerNorm(width)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(width, expansion * width),
            torch.nn.GELU(),
            torch.nn.Linear(expansion * width, width),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized = self.memory_norm(memory)
        attended, _ = self.attention(
            self.query_norm(queries),
            normalized,
            normalized,
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class VisualStateHead(torch.nn.Module):
    """Infer eight continuous agent-centric coordinates from full image tokens."""

    def __init__(
        self,
        *,
        image_width: int,
        state_width: int,
        coordinate_count: int,
        heads: int,
        blocks: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if (
            min(image_width, state_width, coordinate_count, heads, blocks) <= 0
            or state_width % heads
            or coordinate_count > state_width
        ):
            raise ActionForecastError("invalid visual-state dimensions")
        self.image_projection = torch.nn.Linear(image_width, state_width)
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed)
        random = torch.randn(
            state_width,
            coordinate_count,
            generator=generator,
            dtype=torch.float32,
        )
        orthogonal, _ = torch.linalg.qr(random, mode="reduced")
        self.coordinate_queries = torch.nn.Parameter(orthogonal.T.contiguous())
        self.blocks = torch.nn.ModuleList(
            QueryReadBlock(state_width, heads, expansion=2)
            for _ in range(blocks)
        )
        self.scalar_head = torch.nn.Linear(state_width, 1)
        torch.nn.init.normal_(self.scalar_head.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.scalar_head.bias)

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        if image_tokens.ndim != 3 or image_tokens.shape[1] <= 1:
            raise ActionForecastError("full projected image tokens changed shape")
        memory = self.image_projection(image_tokens)
        queries = self.coordinate_queries[None].expand(memory.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, memory)
        return torch.tanh(self.scalar_head(queries).squeeze(-1))


class ContinuousStateEmbedder(torch.nn.Module):
    """Map each imagined scalar to one differentiable PaliGemma-width token."""

    def __init__(
        self,
        *,
        coordinate_count: int,
        fourier_width: int,
        hidden_width: int,
        output_width: int,
    ) -> None:
        super().__init__()
        if (
            coordinate_count <= 0
            or fourier_width <= 0
            or fourier_width % 2
            or min(hidden_width, output_width) <= 0
        ):
            raise ActionForecastError("invalid continuous-state embedder")
        frequencies = torch.arange(
            1,
            fourier_width // 2 + 1,
            dtype=torch.float32,
        )
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(fourier_width, hidden_width),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_width, output_width),
        )
        self.slot_embeddings = torch.nn.Parameter(
            torch.empty(coordinate_count, output_width)
        )
        torch.nn.init.normal_(self.slot_embeddings, mean=0.0, std=0.02)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        angles = math.pi * state[..., None].to(torch.float32) * self.frequencies
        features = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        return self.network(features) + self.slot_embeddings[None]


class Pi05ActionForecastEncoder(torch.nn.Module):
    """Run contextual teacher-view pi0.5 inference for every sampled frame."""

    def __init__(
        self,
        *,
        paligemma_model: torch.nn.Module,
        expert_model: torch.nn.Module,
        image_width: int,
        state_width: int,
        state_coordinates: int,
        state_heads: int,
        state_blocks: int,
        state_fourier_width: int,
        state_embed_hidden: int,
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
            state_coordinates,
            state_heads,
            state_blocks,
            state_fourier_width,
            state_embed_hidden,
            vl_meta_lora_rank,
            action_meta_lora_rank,
            frame_microbatch_size,
            num_flow_steps,
            action_horizon,
            padded_action_dim,
            output_action_dim,
        )
        if any(value <= 0 for value in dimensions) or output_action_dim > padded_action_dim:
            raise ActionForecastError("invalid pi0.5 action-forecast dimensions")
        self.frame_microbatch_size = int(frame_microbatch_size)
        self.num_flow_steps = int(num_flow_steps)
        self.action_horizon = int(action_horizon)
        self.padded_action_dim = int(padded_action_dim)
        self.output_action_dim = int(output_action_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.visual_state = VisualStateHead(
            image_width=image_width,
            state_width=state_width,
            coordinate_count=state_coordinates,
            heads=state_heads,
            blocks=state_blocks,
            initialization_seed=initialization_seed,
        )
        self.state_embedder = ContinuousStateEmbedder(
            coordinate_count=state_coordinates,
            fourier_width=state_fourier_width,
            hidden_width=state_embed_hidden,
            output_width=image_width,
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
        frames: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        state_positions: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        bridge = core.paligemma_with_expert
        language_model = bridge.paligemma.model.language_model
        expert_model = bridge.gemma_expert.model
        images = self._prepare_images(frames)
        with torch.no_grad():
            image_tokens = bridge.embed_image(images)
            text_tokens = bridge.embed_language_tokens(language_tokens)

        imagined_state = self.visual_state(image_tokens)
        virtual_state = self.state_embedder(imagined_state)
        state_values = torch.zeros_like(text_tokens).scatter(
            1,
            state_positions[..., None].expand(-1, -1, text_tokens.shape[-1]),
            virtual_state.to(text_tokens.dtype),
        )
        state_mask = torch.zeros_like(language_mask).scatter(
            1,
            state_positions,
            True,
        )
        text_tokens = torch.where(
            state_mask[..., None],
            state_values,
            text_tokens,
        )
        prefix = torch.cat((image_tokens, text_tokens), dim=1)
        prefix_padding = torch.cat(
            (
                torch.ones(
                    image_tokens.shape[:2],
                    dtype=torch.bool,
                    device=frames.device,
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
                    (frames.shape[0],),
                    1.0 + step * delta,
                    dtype=torch.float32,
                    device=frames.device,
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

        conditions = language_tokens.shape[0]
        if (
            frame_condition_ids.ndim != 1
            or frame_condition_ids.shape[0] != frames.shape[0]
            or frame_condition_ids.dtype != torch.long
            or language_tokens.ndim != 2
            or language_mask.shape != language_tokens.shape
            or language_mask.dtype != torch.bool
            or state_positions.shape != (conditions, 8)
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
        core = policy.model
        if (
            int(core.config.chunk_size) != self.action_horizon
            or int(core.config.max_action_dim) != self.padded_action_dim
        ):
            raise ActionForecastError("pi0.5 action forecast topology changed")
        results = []
        for start in range(0, frames.shape[0], self.frame_microbatch_size):
            stop = min(start + self.frame_microbatch_size, frames.shape[0])
            selected = frame_condition_ids[start:stop]
            arguments = (
                frames[start:stop],
                language_tokens.index_select(0, selected),
                language_mask.index_select(0, selected),
                state_positions.index_select(0, selected),
                flow_noise.index_select(0, selected),
            )

            def invoke(
                batch_frames: torch.Tensor,
                batch_tokens: torch.Tensor,
                batch_mask: torch.Tensor,
                batch_positions: torch.Tensor,
                batch_noise: torch.Tensor,
            ) -> torch.Tensor:
                return self._forecast_microbatch(
                    core,
                    batch_frames,
                    batch_tokens,
                    batch_mask,
                    batch_positions,
                    batch_noise,
                )

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
            results.append(value)
        return torch.cat(results, dim=0)
