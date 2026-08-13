"""PI05 joint-backbone memory extraction for the Dynamic-K Writer."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
from lerobot.policies.pi_gemma import _gated_residual, layernorm_forward
from torch.utils.checkpoint import checkpoint
from transformers.models.gemma import modeling_gemma

from ember.writer.video_program import MetaLoRAStack, TaskQueriedPatchGrounding


class BackboneMemoryError(RuntimeError):
    """Raised when the pinned PI05 backbone-memory contract changes."""


@dataclass(frozen=True)
class BackboneMemoryOutput:
    """Per-frame Dynamic-K evidence produced by one joint PI05 forward."""

    layer_memory: torch.Tensor
    probe_hidden: torch.Tensor
    task_hidden: torch.Tensor
    visual_evidence: torch.Tensor
    valid_task_tokens: torch.Tensor


def make_backbone_memory_mask(
    prefix_padding: torch.Tensor,
    *,
    action_horizon: int,
    memory_tokens: int,
) -> torch.Tensor:
    """Return the PI05 three-block boolean attention mask.

    Prefix queries see only valid prefix tokens. Action queries see the prefix
    and complete action block, while memory queries additionally see the whole
    memory block.
    """

    if (
        prefix_padding.ndim != 2
        or prefix_padding.dtype != torch.bool
        or prefix_padding.shape[0] <= 0
        or prefix_padding.shape[1] <= 0
        or action_horizon <= 0
        or memory_tokens <= 0
    ):
        raise BackboneMemoryError("invalid backbone-memory mask inputs")
    batch = prefix_padding.shape[0]
    options = {"dtype": torch.bool, "device": prefix_padding.device}
    suffix_padding = torch.ones(batch, action_horizon + memory_tokens, **options)
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
    causal_blocks = blocks[:, None, :] <= blocks[:, :, None]
    valid_pairs = padding[:, None, :] & padding[:, :, None]
    return causal_blocks & valid_pairs


class Pi05BackboneMemoryEncoder(torch.nn.Module):
    """Run one Writer-owned PI05 layer loop and retain Action memory states."""

    NATIVE_IMAGE_TOKENS = 256
    ACTION_HORIZON = 50
    PADDED_ACTION_DIM = 32
    MEMORY_TOKENS = 8
    LAYER_COUNT = 18
    ACTION_META_LORA_RANK = 4

    def __init__(
        self,
        *,
        bridge: torch.nn.Module,
        image_width: int,
        expert_width: int,
        program_width: int,
        evidence_heads: int,
        max_frames_per_encoder_call: int,
        action_horizon: int,
        padded_action_dim: int,
        initialization_seed: int,
        action_meta_lora_rank: int = ACTION_META_LORA_RANK,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        paligemma_model = bridge.paligemma.model.language_model
        expert_model = bridge.gemma_expert.model
        if (
            min(
                image_width,
                expert_width,
                program_width,
                evidence_heads,
                max_frames_per_encoder_call,
                action_horizon,
                padded_action_dim,
                action_meta_lora_rank,
            )
            <= 0
            or action_horizon != self.ACTION_HORIZON
            or padded_action_dim != self.PADDED_ACTION_DIM
            or action_meta_lora_rank != self.ACTION_META_LORA_RANK
            or len(paligemma_model.layers) != self.LAYER_COUNT
            or len(expert_model.layers) != self.LAYER_COUNT
        ):
            raise BackboneMemoryError("invalid PI05 backbone-memory topology")
        self.image_width = int(image_width)
        self.expert_width = int(expert_width)
        self.program_width = int(program_width)
        self.max_frames_per_encoder_call = int(max_frames_per_encoder_call)
        self.action_horizon = int(action_horizon)
        self.padded_action_dim = int(padded_action_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.action_meta_lora = MetaLoRAStack(expert_model.layers, action_meta_lora_rank)
        self.evidence_projection = torch.nn.Linear(
            image_width, program_width, bias=False
        )
        self.patch_grounding = TaskQueriedPatchGrounding(
            width=program_width, heads=evidence_heads
        )
        generator = torch.Generator(device="cpu").manual_seed(
            int(initialization_seed) + 0xD1A6
        )
        fixed_suffix_noise = torch.randn(
            action_horizon,
            padded_action_dim,
            dtype=torch.float32,
            generator=generator,
        )
        initial_memory = torch.empty(self.MEMORY_TOKENS, expert_width)
        torch.nn.init.normal_(
            initial_memory,
            mean=0.0,
            std=expert_width**-0.5,
            generator=generator,
        )
        self.register_buffer("fixed_suffix_noise", fixed_suffix_noise, persistent=True)
        self.memory_tokens = torch.nn.Parameter(initial_memory)

    @staticmethod
    def _prepare_images(frames: torch.Tensor) -> torch.Tensor:
        from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

        if (
            frames.ndim != 4
            or frames.shape[0] <= 0
            or frames.shape[1] != 3
            or frames.dtype != torch.uint8
        ):
            raise BackboneMemoryError("teacher frames changed shape or dtype")
        value = frames.to(torch.float32).div_(255.0).permute(0, 2, 3, 1)
        value = resize_with_pad_torch(value, 224, 224)
        return (value * 2.0 - 1.0).permute(0, 3, 1, 2)

    @staticmethod
    def _pack_task_hidden(
        hidden: torch.Tensor,
        task_span_mask: torch.Tensor,
        maximum_task_tokens: int,
    ) -> torch.Tensor:
        if (
            hidden.ndim != 3
            or task_span_mask.shape != hidden.shape[:2]
            or task_span_mask.dtype != torch.bool
            or maximum_task_tokens <= 0
        ):
            raise BackboneMemoryError("task-token hidden packing changed")
        ordinals = (
            task_span_mask.to(torch.long).cumsum(dim=1) - 1
        ).clamp_min(0)
        packed = hidden.new_zeros(
            hidden.shape[0],
            maximum_task_tokens,
            hidden.shape[-1],
        )
        return packed.scatter_add(
            1,
            ordinals[..., None].expand(-1, -1, hidden.shape[-1]),
            hidden * task_span_mask[..., None],
        )

    def _project(
        self,
        layer_index: int,
        name: str,
        projection: torch.nn.Module,
        value: torch.Tensor,
        *,
        action_stream: bool,
    ) -> torch.Tensor:
        projected = projection(value)
        if action_stream:
            adapter = self.action_meta_lora.adapters[f"{layer_index}_{name}"]
            projected = projected + adapter(value).to(projected.dtype)
        return projected

    def _joint_layer(
        self,
        layer_index: int,
        prefix: torch.Tensor,
        suffix: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        adarms_condition: torch.Tensor,
        prefix_layer: torch.nn.Module,
        suffix_layer: torch.nn.Module,
        rotary_embedding: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        streams = (prefix, suffix)
        layers = (prefix_layer, suffix_layer)
        conditions = (None, adarms_condition)
        gates: list[torch.Tensor | None] = []
        queries: list[torch.Tensor] = []
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for stream_index, (hidden, layer, condition) in enumerate(
            zip(streams, layers, conditions, strict=True)
        ):
            normed, gate = layernorm_forward(
                layer.input_layernorm,
                hidden,
                condition,
            )
            gates.append(gate)
            shape = (*normed.shape[:-1], -1, layer.self_attn.head_dim)
            action_stream = stream_index == 1
            queries.append(
                self._project(
                    layer_index,
                    "q_proj",
                    layer.self_attn.q_proj,
                    normed,
                    action_stream=action_stream,
                ).view(shape).transpose(1, 2)
            )
            keys.append(
                self._project(
                    layer_index,
                    "k_proj",
                    layer.self_attn.k_proj,
                    normed,
                    action_stream=action_stream,
                ).view(shape).transpose(1, 2)
            )
            values.append(
                self._project(
                    layer_index,
                    "v_proj",
                    layer.self_attn.v_proj,
                    normed,
                    action_stream=action_stream,
                ).view(shape).transpose(1, 2)
            )

        query = torch.cat(queries, dim=2)
        key = torch.cat(keys, dim=2)
        value = torch.cat(values, dim=2)
        dummy = torch.zeros(
            query.shape[0],
            query.shape[2],
            query.shape[-1],
            dtype=query.dtype,
            device=query.device,
        )
        cosine, sine = rotary_embedding(dummy, position_ids)
        query, key = modeling_gemma.apply_rotary_pos_emb(
            query, key, cosine, sine, unsqueeze_dim=1
        )
        attention, _ = modeling_gemma.eager_attention_forward(
            prefix_layer.self_attn,
            query,
            key,
            value,
            attention_mask,
            prefix_layer.self_attn.scaling,
        )
        attention = attention.reshape(
            query.shape[0], -1, query.shape[1] * query.shape[-1]
        )
        outputs: list[torch.Tensor] = []
        start = 0
        for stream_index, (hidden, layer, condition, gate) in enumerate(
            zip(streams, layers, conditions, gates, strict=True)
        ):
            stop = start + hidden.shape[1]
            output_projection = layer.self_attn.o_proj
            projection_input = attention[:, start:stop].to(output_projection.weight.dtype)
            attended = self._project(
                layer_index,
                "o_proj",
                output_projection,
                projection_input,
                action_stream=stream_index == 1,
            )
            after_attention = _gated_residual(hidden, attended, gate)
            if after_attention is None:
                raise BackboneMemoryError("PI05 attention residual returned None")
            residual = after_attention.clone()
            normalized_output, mlp_gate = layernorm_forward(
                layer.post_attention_layernorm,
                after_attention,
                condition,
            )
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                normalized_output = normalized_output.to(torch.bfloat16)
            mlp_output = layer.mlp(normalized_output)
            output = _gated_residual(residual, mlp_output, mlp_gate)
            if output is None:
                raise BackboneMemoryError("PI05 MLP residual returned None")
            outputs.append(output)
            start = stop
        return outputs[0], outputs[1]

    def _joint_forward(
        self,
        bridge: torch.nn.Module,
        prefix: torch.Tensor,
        suffix: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        adarms_condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        language_model = bridge.paligemma.model.language_model
        expert_model = bridge.gemma_expert.model
        if (
            len(language_model.layers) != self.LAYER_COUNT
            or len(expert_model.layers) != self.LAYER_COUNT
        ):
            raise BackboneMemoryError("PI05 joint layer count changed")
        memories: list[torch.Tensor] = []
        should_checkpoint = (
            self.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )
        for layer_index, (prefix_layer, suffix_layer) in enumerate(
            zip(language_model.layers, expert_model.layers, strict=True)
        ):
            layer_call = partial(
                self._joint_layer,
                layer_index,
                attention_mask=attention_mask,
                position_ids=position_ids,
                adarms_condition=adarms_condition,
                prefix_layer=prefix_layer,
                suffix_layer=suffix_layer,
                rotary_embedding=language_model.rotary_emb,
            )
            if should_checkpoint:
                prefix, suffix = checkpoint(
                    layer_call,
                    prefix,
                    suffix,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                prefix, suffix = layer_call(prefix, suffix)
            memories.append(suffix[:, -self.MEMORY_TOKENS :])

        prefix, _ = layernorm_forward(language_model.norm, prefix, None)
        suffix, _ = layernorm_forward(
            expert_model.norm,
            suffix,
            adarms_condition,
        )
        return prefix, suffix, torch.stack(memories, dim=1)

    def _encode_microbatch(
        self,
        core: torch.nn.Module,
        frames: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        maximum_task_tokens: int,
    ) -> BackboneMemoryOutput:
        bridge = core.paligemma_with_expert
        language_model = bridge.paligemma.model.language_model
        images = self._prepare_images(frames)
        with torch.no_grad():
            image_embeddings = bridge.embed_image(images)
            language_embeddings = bridge.embed_language_tokens(language_tokens)
        if (
            image_embeddings.shape
            != (frames.shape[0], self.NATIVE_IMAGE_TOKENS, self.image_width)
            or language_embeddings.shape
            != (*language_tokens.shape, self.image_width)
        ):
            raise BackboneMemoryError("PI05 prefix embedding layout changed")
        prefix = torch.cat((image_embeddings, language_embeddings), dim=1)
        prefix_padding = torch.cat(
            (
                torch.ones(
                    image_embeddings.shape[:2],
                    dtype=torch.bool,
                    device=frames.device,
                ),
                language_mask,
            ),
            dim=1,
        )
        attention_mask = core._prepare_attention_masks_4d(
            make_backbone_memory_mask(
            prefix_padding,
            action_horizon=self.action_horizon,
            memory_tokens=self.MEMORY_TOKENS,
            )
        )
        total_padding = torch.cat(
            (
                prefix_padding,
                torch.ones(
                    frames.shape[0],
                    self.action_horizon + self.MEMORY_TOKENS,
                    dtype=torch.bool,
                    device=frames.device,
                ),
            ),
            dim=1,
        )
        position_ids = torch.cumsum(total_padding, dim=1) - 1
        noise = self.fixed_suffix_noise[None].expand(frames.shape[0], -1, -1)
        timestep = torch.ones(frames.shape[0], dtype=torch.float32, device=frames.device)
        action_suffix, action_padding, action_markers, adarms_condition = core.embed_suffix(
            noise, timestep
        )
        expected_markers = action_markers.new_zeros(action_suffix.shape[:2])
        expected_markers[:, 0] = 1
        if (
            action_suffix.shape
            != (frames.shape[0], self.action_horizon, self.expert_width)
            or action_padding.shape != action_suffix.shape[:2]
            or action_padding.dtype != torch.bool
            or not bool(action_padding.all())
            or action_markers.shape != action_suffix.shape[:2]
            or not torch.equal(action_markers, expected_markers)
            or adarms_condition.shape != (frames.shape[0], self.expert_width)
        ):
            raise BackboneMemoryError("PI05 Action probe layout changed")
        target_dtype = language_model.layers[0].self_attn.q_proj.weight.dtype
        learned_memory = self.memory_tokens.to(device=frames.device, dtype=target_dtype)
        learned_memory = learned_memory[None].expand(frames.shape[0], -1, -1)
        suffix = torch.cat((action_suffix.to(target_dtype), learned_memory), dim=1)
        prefix_hidden, suffix_hidden, layer_memory = self._joint_forward(
            bridge,
            prefix.to(target_dtype),
            suffix,
            attention_mask,
            position_ids,
            adarms_condition,
        )
        if (
            prefix_hidden.shape != prefix.shape
            or suffix_hidden.shape
            != (
                frames.shape[0],
                self.action_horizon + self.MEMORY_TOKENS,
                self.expert_width,
            )
            or layer_memory.shape
            != (
                frames.shape[0],
                self.LAYER_COUNT,
                self.MEMORY_TOKENS,
                self.expert_width,
            )
        ):
            raise BackboneMemoryError("PI05 backbone-memory output layout changed")
        task_hidden, visual_evidence, valid_task_tokens = self._extract_task_evidence(
            prefix_hidden,
            task_span_mask,
            maximum_task_tokens,
        )
        return BackboneMemoryOutput(
            layer_memory=layer_memory,
            probe_hidden=suffix_hidden[:, : self.action_horizon],
            task_hidden=task_hidden,
            visual_evidence=visual_evidence,
            valid_task_tokens=valid_task_tokens,
        )

    def _extract_task_evidence(
        self,
        prefix_hidden: torch.Tensor,
        task_span_mask: torch.Tensor,
        maximum_task_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        task_hidden = self._pack_task_hidden(
            prefix_hidden[:, self.NATIVE_IMAGE_TOKENS :],
            task_span_mask,
            maximum_task_tokens,
        )
        valid_task_tokens = (
            torch.arange(maximum_task_tokens, device=prefix_hidden.device)[None]
            < task_span_mask.sum(dim=1)[:, None]
        )
        task_evidence = self.evidence_projection(task_hidden)
        patch_evidence = self.evidence_projection(
            prefix_hidden[:, : self.NATIVE_IMAGE_TOKENS]
        )
        visual_evidence = task_evidence + self.patch_grounding(
            task_evidence,
            patch_evidence,
            valid_task_tokens,
        )
        return task_hidden, visual_evidence, valid_task_tokens

    def _validate_forward_batch(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> tuple[torch.nn.Module, int]:
        conditions = language_tokens.shape[0] if language_tokens.ndim == 2 else 0
        if (
            frames.ndim != 4
            or frames.shape[0] <= 0
            or conditions <= 0
            or frame_condition_ids.shape != (frames.shape[0],)
            or frame_condition_ids.dtype != torch.long
            or language_mask.shape != language_tokens.shape
            or language_mask.dtype != torch.bool
            or task_span_mask.shape != language_tokens.shape
            or task_span_mask.dtype != torch.bool
            or any(
                value.device != frames.device
                for value in (
                    frame_condition_ids,
                    language_tokens,
                    language_mask,
                    task_span_mask,
                )
            )
        ):
            raise BackboneMemoryError("invalid frame-language backbone batch")
        task_counts = task_span_mask.sum(dim=1)
        condition_counts = (
            frame_condition_ids[:, None]
            == torch.arange(conditions, device=frames.device)[None]
        ).sum(dim=0)
        invalid = torch.stack(
            (
                (task_span_mask & ~language_mask).any(),
                ~task_span_mask.any(dim=1).all(),
                task_span_mask[:, 0].any(),
                (
                    (frame_condition_ids < 0)
                    | (frame_condition_ids >= conditions)
                ).any(),
                (condition_counts <= 0).any(),
                (frame_condition_ids[1:] < frame_condition_ids[:-1]).any(),
            )
        )
        validation = torch.cat(
            (invalid.to(torch.long), task_counts.max().reshape(1))
        ).to(device="cpu").tolist()
        if any(validation[:-1]):
            raise BackboneMemoryError("invalid frame-language backbone batch")
        core = policy.model
        if (
            int(core.config.chunk_size) != self.action_horizon
            or int(core.config.max_action_dim) != self.padded_action_dim
        ):
            raise BackboneMemoryError("PI05 Action Expert topology changed")
        return core, int(validation[-1])

    def forward(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> BackboneMemoryOutput:
        """Encode every real frame once with exact-language joint context."""

        core, maximum_task_tokens = self._validate_forward_batch(
            policy,
            frames,
            frame_condition_ids,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        rows: list[BackboneMemoryOutput] = []
        for start in range(0, frames.shape[0], self.max_frames_per_encoder_call):
            stop = min(start + self.max_frames_per_encoder_call, frames.shape[0])
            selected = frame_condition_ids[start:stop]
            rows.append(
                self._encode_microbatch(
                    core,
                    frames[start:stop],
                    language_tokens.index_select(0, selected),
                    language_mask.index_select(0, selected),
                    task_span_mask.index_select(0, selected),
                    maximum_task_tokens,
                )
            )
        return BackboneMemoryOutput(
            layer_memory=torch.cat([row.layer_memory for row in rows]),
            probe_hidden=torch.cat([row.probe_hidden for row in rows]),
            task_hidden=torch.cat([row.task_hidden for row in rows]),
            visual_evidence=torch.cat([row.visual_evidence for row in rows]),
            valid_task_tokens=torch.cat([row.valid_task_tokens for row in rows]),
        )
