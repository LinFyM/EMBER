"""Carrier-exact one-way PI05 memory for layer-matched rank queries."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Iterator

import torch
from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
from lerobot.policies.pi_gemma import _gated_residual, layernorm_forward
from torch.utils.checkpoint import checkpoint
from transformers.models.gemma import modeling_gemma


class BackboneMemoryError(RuntimeError):
    """Raised when the pinned PI05 backbone-memory contract changes."""


@dataclass(frozen=True)
class BackboneMemoryOutput:
    """Joint outputs and the post-layer memory states."""

    prefix_hidden: torch.Tensor
    action_hidden: torch.Tensor
    layer_memory: torch.Tensor


@dataclass(frozen=True)
class LayerMatchedVideoEncoding:
    """V6 evidence and layer/rank memory states from one frame batch."""

    text_queries: torch.Tensor
    frame_evidence: torch.Tensor
    grounded_evidence: torch.Tensor
    interactions: torch.Tensor
    valid_task_tokens: torch.Tensor
    layer_memory: torch.Tensor


def make_backbone_memory_mask(
    prefix_padding: torch.Tensor,
    *,
    action_horizon: int,
    memory_tokens: int,
) -> torch.Tensor:
    """Build prefix, Action, and one-way memory attention blocks."""

    if (
        prefix_padding.ndim != 2
        or prefix_padding.dtype != torch.bool
        or min(prefix_padding.shape) <= 0
        or action_horizon <= 0
        or memory_tokens <= 0
    ):
        raise BackboneMemoryError("invalid backbone-memory mask inputs")
    batch = prefix_padding.shape[0]
    options = {"dtype": torch.bool, "device": prefix_padding.device}
    suffix_padding = torch.ones(
        batch, action_horizon + memory_tokens, **options
    )
    action_markers = torch.zeros(action_horizon, **options)
    memory_markers = torch.zeros(memory_tokens, **options)
    action_markers[0] = True
    memory_markers[0] = True
    markers = torch.cat(
        (
            torch.zeros_like(prefix_padding),
            action_markers[None].expand(batch, -1),
            memory_markers[None].expand(batch, -1),
        ),
        dim=1,
    )
    padding = torch.cat((prefix_padding, suffix_padding), dim=1)
    blocks = torch.cumsum(markers, dim=1)
    block_visible = blocks[:, None, :] <= blocks[:, :, None]
    valid_pairs = padding[:, None, :] & padding[:, :, None]
    return block_visible & valid_pairs


class Pi05LayerMatchedBackboneMemory(torch.nn.Module):
    """Keep the native PI05 carrier exact and update 16 one-way rank memories."""

    LAYERS = 18
    ACTION_HORIZON = 50
    MEMORY_TOKENS = 16

    def __init__(
        self,
        *,
        image_width: int,
        expert_width: int,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        if min(image_width, expert_width) <= 0:
            raise BackboneMemoryError("invalid backbone-memory widths")
        self.image_width = int(image_width)
        self.expert_width = int(expert_width)
        self.activation_checkpointing = bool(activation_checkpointing)

    @staticmethod
    def _project(
        layer: int,
        name: str,
        projection: torch.nn.Module,
        value: torch.Tensor,
        adapters: torch.nn.Module | None,
    ) -> torch.Tensor:
        projected = projection(value)
        if adapters is None:
            return projected
        adapter = adapters.adapters[f"{layer}_{name}"]
        return projected + adapter(value).to(projected.dtype)

    def _stream_kv(
        self,
        layer_index: int,
        hidden: torch.Tensor,
        layer: torch.nn.Module,
        condition: torch.Tensor | None,
        adapters: torch.nn.Module | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed, _ = layernorm_forward(layer.input_layernorm, hidden, condition)
        shape = (*normed.shape[:-1], -1, layer.self_attn.head_dim)
        projected = []
        for name in ("k_proj", "v_proj"):
            value = self._project(
                layer_index,
                name,
                getattr(layer.self_attn, name),
                normed,
                adapters,
            )
            projected.append(value.view(shape).transpose(1, 2))
        return projected[0], projected[1]

    def _memory_qkv(
        self,
        layer_index: int,
        hidden: torch.Tensor,
        layer: torch.nn.Module,
        condition: torch.Tensor,
        adapters: torch.nn.Module,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
        normed, gate = layernorm_forward(
            layer.input_layernorm, hidden, condition
        )
        shape = (*normed.shape[:-1], -1, layer.self_attn.head_dim)
        projected = []
        for name in ("q_proj", "k_proj", "v_proj"):
            value = self._project(
                layer_index,
                name,
                getattr(layer.self_attn, name),
                normed,
                adapters,
            )
            projected.append(value.view(shape).transpose(1, 2))
        return gate, projected[0], projected[1], projected[2]

    def _finish_stream(
        self,
        layer_index: int,
        hidden: torch.Tensor,
        attended: torch.Tensor,
        layer: torch.nn.Module,
        condition: torch.Tensor | None,
        gate: torch.Tensor | None,
        adapters: torch.nn.Module,
    ) -> torch.Tensor:
        projection = layer.self_attn.o_proj
        attended = self._project(
            layer_index,
            "o_proj",
            projection,
            attended.to(projection.weight.dtype),
            adapters,
        )
        after_attention = _gated_residual(hidden, attended, gate)
        if after_attention is None:
            raise BackboneMemoryError("PI05 attention residual returned None")
        normalized, mlp_gate = layernorm_forward(
            layer.post_attention_layernorm, after_attention, condition
        )
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            normalized = normalized.to(torch.bfloat16)
        output = _gated_residual(
            after_attention.clone(), layer.mlp(normalized), mlp_gate
        )
        if output is None:
            raise BackboneMemoryError("PI05 MLP residual returned None")
        return output

    @staticmethod
    def _rotate(
        value: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        cosine = cosine.unsqueeze(1)
        sine = sine.unsqueeze(1)
        return value * cosine + modeling_gemma.rotate_half(value) * sine

    def _memory_layer(
        self,
        layer_index: int,
        memory: torch.Tensor,
        prefix_hidden: torch.Tensor,
        action_hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        adarms_condition: torch.Tensor,
        prefix_layer: torch.nn.Module,
        suffix_layer: torch.nn.Module,
        rotary_embedding: torch.nn.Module,
        vl_adapters: torch.nn.Module | None,
        action_adapters: torch.nn.Module,
    ) -> torch.Tensor:
        prefix_key, prefix_value = self._stream_kv(
            layer_index,
            prefix_hidden,
            prefix_layer,
            None,
            vl_adapters,
        )
        action_key, action_value = self._stream_kv(
            layer_index,
            action_hidden,
            suffix_layer,
            adarms_condition,
            action_adapters,
        )
        gate, query, memory_key, memory_value = self._memory_qkv(
            layer_index,
            memory,
            suffix_layer,
            adarms_condition,
            action_adapters,
        )
        key = torch.cat((prefix_key, action_key, memory_key), dim=2)
        value = torch.cat((prefix_value, action_value, memory_value), dim=2)
        rotary_input = torch.zeros(
            query.shape[0],
            key.shape[2],
            query.shape[-1],
            dtype=query.dtype,
            device=query.device,
        )
        cosine, sine = rotary_embedding(rotary_input, position_ids)
        query = self._rotate(
            query,
            cosine[:, -self.MEMORY_TOKENS :],
            sine[:, -self.MEMORY_TOKENS :],
        )
        key = self._rotate(key, cosine, sine)
        attended, _ = modeling_gemma.eager_attention_forward(
            prefix_layer.self_attn,
            query,
            key,
            value,
            attention_mask,
            prefix_layer.self_attn.scaling,
        )
        attended = attended.reshape(
            query.shape[0],
            self.MEMORY_TOKENS,
            query.shape[1] * query.shape[-1],
        )
        return self._finish_stream(
            layer_index,
            memory,
            attended,
            suffix_layer,
            adarms_condition,
            gate,
            action_adapters,
        )

    def _validate_inputs(
        self,
        core: torch.nn.Module,
        prefix: torch.Tensor,
        prefix_padding: torch.Tensor,
        action_suffix: torch.Tensor,
        action_padding: torch.Tensor,
        action_markers: torch.Tensor,
        adarms_condition: torch.Tensor,
        memory_tokens: torch.Tensor,
    ) -> tuple[torch.nn.Module, torch.nn.Module, int]:
        bridge = core.paligemma_with_expert
        language_model = bridge.paligemma.model.language_model
        expert_model = bridge.gemma_expert.model
        batch = prefix.shape[0]
        expected_markers = torch.zeros_like(action_markers)
        expected_markers[:, 0] = 1
        if (
            prefix.ndim != 3
            or prefix.shape[-1] != self.image_width
            or prefix_padding.shape != prefix.shape[:2]
            or prefix_padding.dtype != torch.bool
            or action_suffix.shape
            != (batch, self.ACTION_HORIZON, self.expert_width)
            or action_padding.shape != action_suffix.shape[:2]
            or action_padding.dtype != torch.bool
            or not bool(action_padding.all())
            or action_markers.shape != action_suffix.shape[:2]
            or not torch.equal(action_markers, expected_markers)
            or adarms_condition.shape != (batch, self.expert_width)
            or memory_tokens.shape != (self.MEMORY_TOKENS, self.expert_width)
            or len(language_model.layers) != self.LAYERS
            or len(expert_model.layers) != self.LAYERS
        ):
            raise BackboneMemoryError("PI05 backbone-memory topology changed")
        return language_model, expert_model, batch

    def _prepare_native_context(
        self,
        core: torch.nn.Module,
        prefix: torch.Tensor,
        prefix_padding: torch.Tensor,
        action_suffix: torch.Tensor,
        action_padding: torch.Tensor,
        action_markers: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        padding = torch.cat((prefix_padding, action_padding), dim=1)
        markers = torch.cat(
            (torch.zeros_like(prefix_padding), action_markers), dim=1
        )
        attention_mask = core._prepare_attention_masks_4d(
            make_att_2d_masks(padding, markers)
        )
        position_ids = torch.cumsum(padding, dim=1) - 1
        return (
            prefix.to(target_dtype),
            action_suffix.to(target_dtype),
            attention_mask,
            position_ids,
        )

    def _prepare_memory_attention(
        self,
        core: torch.nn.Module,
        prefix_padding: torch.Tensor,
        action_padding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = prefix_padding.shape[0]
        boolean_mask = make_backbone_memory_mask(
            prefix_padding,
            action_horizon=self.ACTION_HORIZON,
            memory_tokens=self.MEMORY_TOKENS,
        )
        attention_mask = core._prepare_attention_masks_4d(boolean_mask)
        total_padding = torch.cat(
            (
                prefix_padding,
                action_padding,
                torch.ones(
                    batch,
                    self.MEMORY_TOKENS,
                    dtype=torch.bool,
                    device=prefix_padding.device,
                ),
            ),
            dim=1,
        )
        position_ids = torch.cumsum(total_padding, dim=1) - 1
        return (
            attention_mask[:, :, -self.MEMORY_TOKENS :, :],
            position_ids,
        )

    @contextmanager
    def _capture_native_states(
        self,
        prefix: torch.Tensor,
        action: torch.Tensor,
        language_model: torch.nn.Module,
        expert_model: torch.nn.Module,
    ) -> Iterator[dict[str, list[torch.Tensor | None]]]:
        capture: dict[str, list[torch.Tensor | None]] = {
            "prefix": [prefix.detach(), *([None] * self.LAYERS)],
            "action": [action.detach(), *([None] * self.LAYERS)],
        }
        handles = []

        def install(
            target: torch.nn.Module,
            stream: str,
            output_layer: int,
        ) -> None:
            def hook(
                _module: torch.nn.Module,
                inputs: tuple[torch.Tensor, ...],
                *,
                selected_stream: str = stream,
                selected_layer: int = output_layer,
            ) -> None:
                if (
                    not inputs
                    or capture[selected_stream][selected_layer + 1] is not None
                ):
                    raise BackboneMemoryError("native layer capture changed execution")
                capture[selected_stream][selected_layer + 1] = inputs[0].detach()

            handles.append(target.register_forward_pre_hook(hook))

        for layer_index in range(self.LAYERS - 1):
            install(
                language_model.layers[layer_index + 1].input_layernorm,
                "prefix",
                layer_index,
            )
            install(
                expert_model.layers[layer_index + 1].input_layernorm,
                "action",
                layer_index,
            )
        install(language_model.norm, "prefix", self.LAYERS - 1)
        install(expert_model.norm, "action", self.LAYERS - 1)
        try:
            yield capture
        finally:
            for handle in handles:
                handle.remove()

    def _run_memory_layers(
        self,
        memory: torch.Tensor,
        capture: dict[str, list[torch.Tensor | None]],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        adarms_condition: torch.Tensor,
        language_model: torch.nn.Module,
        expert_model: torch.nn.Module,
        memory_tokens: torch.Tensor,
        vl_adapters: torch.nn.Module | None,
        action_adapters: torch.nn.Module,
    ) -> torch.Tensor:
        if any(value is None for values in capture.values() for value in values):
            raise BackboneMemoryError("native PI05 layer capture is incomplete")
        layer_memories: list[torch.Tensor] = []
        should_checkpoint = (
            self.activation_checkpointing
            and torch.is_grad_enabled()
            and memory_tokens.requires_grad
        )
        for layer_index, (prefix_layer, suffix_layer) in enumerate(
            zip(language_model.layers, expert_model.layers, strict=True)
        ):
            layer_call = partial(
                self._memory_layer,
                layer_index,
                attention_mask=attention_mask,
                position_ids=position_ids,
                adarms_condition=adarms_condition,
                prefix_layer=prefix_layer,
                suffix_layer=suffix_layer,
                rotary_embedding=language_model.rotary_emb,
                vl_adapters=vl_adapters,
                action_adapters=action_adapters,
            )
            prefix_hidden = capture["prefix"][layer_index]
            action_hidden = capture["action"][layer_index]
            if prefix_hidden is None or action_hidden is None:
                raise BackboneMemoryError("native PI05 layer input is missing")
            if should_checkpoint:
                memory = checkpoint(
                    layer_call,
                    memory,
                    prefix_hidden,
                    action_hidden,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                memory = layer_call(memory, prefix_hidden, action_hidden)
            layer_memories.append(memory)
        return torch.stack(layer_memories, dim=1)

    def forward(
        self,
        core: torch.nn.Module,
        prefix: torch.Tensor,
        prefix_padding: torch.Tensor,
        action_suffix: torch.Tensor,
        action_padding: torch.Tensor,
        action_markers: torch.Tensor,
        adarms_condition: torch.Tensor,
        memory_tokens: torch.Tensor,
        vl_adapters: torch.nn.Module | None,
        action_adapters: torch.nn.Module,
    ) -> BackboneMemoryOutput:
        """Run the exact native carrier once, then its one-way memory observer."""

        language_model, expert_model, batch = self._validate_inputs(
            core,
            prefix,
            prefix_padding,
            action_suffix,
            action_padding,
            action_markers,
            adarms_condition,
            memory_tokens,
        )
        target_dtype = language_model.layers[0].self_attn.q_proj.weight.dtype
        prefix, action_suffix, attention_mask, position_ids = (
            self._prepare_native_context(
                core,
                prefix,
                prefix_padding,
                action_suffix,
                action_padding,
                action_markers,
                target_dtype,
            )
        )
        bridge = core.paligemma_with_expert
        vl_context = (
            vl_adapters.installed(language_model)
            if vl_adapters is not None
            else nullcontext()
        )
        with self._capture_native_states(
            prefix,
            action_suffix,
            language_model,
            expert_model,
        ) as capture:
            with (
                torch.no_grad(),
                vl_context,
                action_adapters.installed(expert_model),
            ):
                (prefix_hidden, action_hidden), _ = bridge.forward(
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix, action_suffix],
                    use_cache=False,
                    adarms_cond=[None, adarms_condition],
                )
        memory_attention, memory_positions = self._prepare_memory_attention(
            core,
            prefix_padding,
            action_padding,
        )
        memory = memory_tokens.to(device=prefix.device, dtype=target_dtype)
        memory = memory[None].expand(batch, -1, -1)
        layer_memories = self._run_memory_layers(
            memory,
            capture,
            memory_attention,
            memory_positions,
            adarms_condition,
            language_model,
            expert_model,
            memory_tokens,
            vl_adapters,
            action_adapters,
        )
        output = BackboneMemoryOutput(
            prefix_hidden=prefix_hidden,
            action_hidden=action_hidden,
            layer_memory=layer_memories,
        )
        if (
            output.prefix_hidden.shape != prefix.shape
            or output.action_hidden.shape
            != (batch, self.ACTION_HORIZON, self.expert_width)
            or output.layer_memory.shape
            != (batch, self.LAYERS, self.MEMORY_TOKENS, self.expert_width)
        ):
            raise BackboneMemoryError("backbone-memory output layout changed")
        return output


class LayerMatchedBackboneMemoryEncoder(torch.nn.Module):
    """Embed frames, run the exact carrier, and update layer/rank memories."""

    def __init__(
        self,
        *,
        image_width: int,
        expert_width: int,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.joint = Pi05LayerMatchedBackboneMemory(
            image_width=image_width,
            expert_width=expert_width,
            activation_checkpointing=activation_checkpointing,
        )

    def _encode_microbatch(
        self,
        semantic: torch.nn.Module,
        core: torch.nn.Module,
        frames: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        text_queries: torch.Tensor,
        valid_task_tokens: torch.Tensor,
        memory_tokens: torch.Tensor,
        maximum_task_tokens: int,
    ) -> tuple[torch.Tensor, ...]:
        bridge = core.paligemma_with_expert
        with torch.no_grad():
            image_tokens = bridge.embed_image(semantic._prepare_images(frames))
            language_embeddings = bridge.embed_language_tokens(language_tokens)
        if (
            image_tokens.shape[1:]
            != (semantic.NATIVE_IMAGE_TOKENS, semantic.image_width)
            or language_embeddings.shape[:2] != language_tokens.shape
        ):
            raise BackboneMemoryError("PI05 prefix embedding layout changed")
        prefix = torch.cat((image_tokens, language_embeddings), dim=1)
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
        noise = semantic.fixed_suffix_noise[None].expand(frames.shape[0], -1, -1)
        timestep = torch.ones(
            frames.shape[0], dtype=torch.float32, device=frames.device
        )
        action, padding, markers, adarms = core.embed_suffix(noise, timestep)
        backbone = self.joint(
            core,
            prefix,
            prefix_padding,
            action,
            padding,
            markers,
            adarms,
            memory_tokens,
            semantic.vl_meta_lora,
            semantic.action_meta_lora,
        )
        evidence = semantic._project_joint_evidence(
            backbone.prefix_hidden.detach(),
            backbone.action_hidden.detach(),
            task_span_mask,
            text_queries,
            valid_task_tokens,
            maximum_task_tokens,
        )
        return (*evidence, backbone.layer_memory)

    def forward(
        self,
        semantic: torch.nn.Module,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        memory_tokens: torch.Tensor,
    ) -> LayerMatchedVideoEncoding:
        """Return carrier evidence and live 18x16 layer/rank memory states."""

        core, valid_task_tokens, _ = semantic._validate_forward_batch(
            policy,
            frames,
            frame_condition_ids,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        if memory_tokens.shape != (
            self.joint.MEMORY_TOKENS,
            semantic.expert_width,
        ):
            raise BackboneMemoryError("capacity-matched memory input changed shape")
        maximum_task_tokens = valid_task_tokens.shape[1]
        text_queries = semantic._encode_text(
            core,
            language_tokens,
            task_span_mask,
            maximum_task_tokens,
        )

        columns: list[list[torch.Tensor]] = [[] for _ in range(4)]
        step = semantic.max_frames_per_encoder_call
        for start in range(0, frames.shape[0], step):
            stop = min(start + step, frames.shape[0])
            rows = torch.arange(start, stop, device=frames.device)
            actual = rows.numel()
            frame_values = frames.index_select(0, rows)
            selected = frame_condition_ids.index_select(0, rows)
            if actual < step:
                padding = step - actual
                frame_values = torch.cat(
                    (
                        frame_values,
                        torch.zeros(
                            padding,
                            *frames.shape[1:],
                            dtype=frames.dtype,
                            device=frames.device,
                        ),
                    )
                )
                selected = torch.cat((selected, selected[-1:].expand(padding)))
            values = self._encode_microbatch(
                semantic,
                core,
                frame_values,
                language_tokens.index_select(0, selected),
                language_mask.index_select(0, selected),
                task_span_mask.index_select(0, selected),
                text_queries.index_select(0, selected),
                valid_task_tokens.index_select(0, selected),
                memory_tokens,
                maximum_task_tokens,
            )
            for column, value in zip(columns, values, strict=True):
                column.append(value[:actual])
        evidence, grounded, interactions, memories = (
            torch.cat(column, dim=0) for column in columns
        )
        return LayerMatchedVideoEncoding(
            text_queries=text_queries,
            frame_evidence=evidence,
            grounded_evidence=grounded,
            interactions=interactions,
            valid_task_tokens=valid_task_tokens,
            layer_memory=memories,
        )
