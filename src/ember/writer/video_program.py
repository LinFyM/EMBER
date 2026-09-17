"""Unified video memory with one residual writeback inside the native PI05 read."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ember.writer.meta_lora import MetaLoRAStack
from ember.writer.temporal import JointVideoStack, RMSNorm, token_role_addresses


class VideoProgramError(RuntimeError):
    """Raised when the sealed teacher-video semantic interface changes."""


class TaskQueriedPatchGrounding(torch.nn.Module):
    """Read per-frame image-position content with text-only task queries."""

    def __init__(self, *, width: int, heads: int, image_tokens: int = 512) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads or image_tokens != 512:
            raise VideoProgramError("invalid task-queried patch grounding")
        self.image_tokens = image_tokens
        self.heads = int(heads)
        self.head_width = width // heads
        self.query_norm = RMSNorm(width)
        self.patch_norm = RMSNorm(width)
        self.query = torch.nn.Linear(width, width, bias=False)
        self.key = torch.nn.Linear(width, width, bias=False)
        self.output = torch.nn.Linear(width, width, bias=False)

    def forward(
        self,
        task_queries: torch.Tensor,
        patch_content: torch.Tensor,
        valid_task_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if (
            task_queries.ndim != 3
            or patch_content.ndim != 3
            or task_queries.shape[0] != patch_content.shape[0]
            or task_queries.shape[-1] != patch_content.shape[-1]
            or task_queries.shape[-1] != self.heads * self.head_width
            or patch_content.shape[1] != self.image_tokens
            or valid_task_tokens.shape != task_queries.shape[:2]
            or valid_task_tokens.dtype != torch.bool
            or not bool(valid_task_tokens.any(dim=1).all())
        ):
            raise VideoProgramError("invalid task-query patch batch")
        batch, task_tokens, width = task_queries.shape
        patches = patch_content.shape[1]
        query = self.query(self.query_norm(task_queries)).reshape(
            batch,
            task_tokens,
            self.heads,
            self.head_width,
        ).transpose(1, 2)
        key = self.key(self.patch_norm(patch_content)).reshape(
            batch,
            patches,
            self.heads,
            self.head_width,
        ).transpose(1, 2)
        value = patch_content.reshape(
            batch,
            patches,
            self.heads,
            self.head_width,
        ).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        merged = attended.transpose(1, 2).reshape(batch, task_tokens, width)
        return self.output(merged).masked_fill(
            ~valid_task_tokens[..., None],
            0.0,
        )


class _GroundedNativeRead(torch.nn.Module):
    """Use depth-matched text queries and retain every native horizon position."""

    def __init__(self, image_width: int, expert_width: int, width: int, heads: int):
        super().__init__()
        self.language_projection = torch.nn.Linear(image_width, width, bias=False)
        self.action_projection = torch.nn.Linear(expert_width, width, bias=False)
        self.patch_grounding = TaskQueriedPatchGrounding(width=width, heads=heads)

    def forward(self, prefix, suffix, text_queries, task_span_mask, valid_task_tokens):
        semantic = Pi05UnifiedVideoEncoder._pack_hidden(
            prefix[:, 512:], task_span_mask, valid_task_tokens.shape[1]
        )
        evidence = self.language_projection(semantic) + self.patch_grounding(
            self.language_projection(text_queries),
            self.language_projection(prefix[:, :512]),
            valid_task_tokens,
        )
        evidence = evidence.masked_fill(~valid_task_tokens[..., None], 0.0)
        return torch.cat((evidence, self.action_projection(suffix)), dim=1)


class Pi05UnifiedVideoEncoder(torch.nn.Module):
    """Read native layers 1–9, write joint video context, then continue 10–18."""

    PATCHES_PER_CAMERA = 256

    def __init__(
        self,
        *,
        paligemma_model: torch.nn.Module,
        expert_model: torch.nn.Module,
        image_width: int,
        expert_width: int,
        program_width: int,
        text_meta_lora_rank: int,
        vl_meta_lora_rank: int,
        action_meta_lora_rank: int,
        patch_grounding_heads: int,
        max_frames_per_encoder_call: int,
        action_horizon: int,
        padded_action_dim: int,
        initialization_seed: int,
        activation_checkpointing: bool,
        camera_view: str = "dual",
        native_split_layer: int = 9,
        joint_heads: int = 8,
        joint_blocks: int = 2,
    ) -> None:
        super().__init__()
        dimensions = (
            image_width, expert_width, program_width, text_meta_lora_rank,
            vl_meta_lora_rank, action_meta_lora_rank, patch_grounding_heads,
            max_frames_per_encoder_call, joint_heads, joint_blocks,
        )
        if (
            any(value <= 0 for value in dimensions)
            or action_horizon != 50 or padded_action_dim != 32
            or camera_view != "dual" or native_split_layer != 9
            or len(paligemma_model.layers) != 18 or len(expert_model.layers) != 18
        ):
            raise VideoProgramError("invalid PI05 unified encoder dimensions")
        self.image_width = int(image_width)
        self.expert_width = int(expert_width)
        self.program_width = int(program_width)
        self.max_frames_per_encoder_call = int(max_frames_per_encoder_call)
        self.action_horizon = int(action_horizon)
        self.padded_action_dim = int(padded_action_dim)
        self.native_split_layer = int(native_split_layer)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.camera_view = camera_view
        self.camera_count = 2
        self.image_tokens = 2 * self.PATCHES_PER_CAMERA
        self.middle_read = _GroundedNativeRead(
            image_width, expert_width, program_width, patch_grounding_heads
        )
        self.final_read = _GroundedNativeRead(
            image_width, expert_width, program_width, patch_grounding_heads
        )
        self.middle_stack = JointVideoStack(program_width, joint_heads, joint_blocks)
        self.final_stack = JointVideoStack(program_width, joint_heads, joint_blocks)
        self.prefix_writeback = torch.nn.Linear(program_width, image_width, bias=False)
        self.horizon_writeback = torch.nn.Linear(program_width, expert_width, bias=False)
        torch.nn.init.zeros_(self.prefix_writeback.weight)
        torch.nn.init.zeros_(self.horizon_writeback.weight)
        self.type_embeddings = torch.nn.Parameter(torch.empty(2, program_width))
        torch.nn.init.normal_(self.type_embeddings, std=0.02)
        self.text_meta_lora = MetaLoRAStack(paligemma_model.layers, text_meta_lora_rank)
        self.vl_meta_lora = MetaLoRAStack(paligemma_model.layers, vl_meta_lora_rank)
        self.action_meta_lora = MetaLoRAStack(expert_model.layers, action_meta_lora_rank)
        generator = torch.Generator(device="cpu").manual_seed(int(initialization_seed) + 0x5A17)
        self.register_buffer(
            "fixed_suffix_noise",
            torch.randn(action_horizon, padded_action_dim, generator=generator),
            persistent=True,
        )

    def _checkpoint(self, function, *arguments):
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(function, *arguments, use_reentrant=False, preserve_rng_state=False)
        return function(*arguments)

    def _prepare_images(self, frames: torch.Tensor) -> torch.Tensor:
        from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

        if frames.ndim != 5 or frames.shape[1:3] != (2, 3) or frames.dtype != torch.uint8:
            raise VideoProgramError("teacher frames changed shape or dtype")
        pixels = frames.flatten(0, 1).to(torch.float32).div_(255.0).permute(0, 2, 3, 1)
        pixels = resize_with_pad_torch(pixels, 224, 224)
        return (pixels * 2.0 - 1.0).permute(0, 3, 1, 2)

    @staticmethod
    def _pack_hidden(hidden, task_span_mask, maximum_task_tokens):
        if (
            hidden.ndim != 3 or task_span_mask.shape != hidden.shape[:2]
            or task_span_mask.dtype != torch.bool or maximum_task_tokens <= 0
            or int(task_span_mask.sum(dim=1).max()) > maximum_task_tokens
        ):
            raise VideoProgramError("task-token hidden packing changed")
        ordinal = (task_span_mask.long().cumsum(dim=1) - 1).clamp_min(0)
        packed = hidden.new_zeros(hidden.shape[0], maximum_task_tokens, hidden.shape[-1])
        return packed.scatter_add(
            1, ordinal[..., None].expand_as(hidden), hidden * task_span_mask[..., None]
        )

    def _encode_text(self, core, language_tokens, task_span_mask, maximum_task_tokens):
        """One text-only native call exposes pre-norm j9 and final-norm j18."""
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        bridge = core.paligemma_with_expert
        language_model = bridge.paligemma.model.language_model
        text_tokens = language_tokens.new_zeros(language_tokens.shape[0], maximum_task_tokens + 1)
        text_padding = torch.zeros_like(text_tokens, dtype=torch.bool)
        text_tokens[:, 0] = language_tokens[:, 0]
        text_padding[:, 0] = True
        for row in range(language_tokens.shape[0]):
            selected = language_tokens[row, task_span_mask[row]]
            text_tokens[row, 1:selected.numel() + 1] = selected
            text_padding[row, 1:selected.numel() + 1] = True
        with torch.no_grad():
            text_embeds = bridge.embed_language_tokens(text_tokens)
        mask = core._prepare_attention_masks_4d(
            make_att_2d_masks(text_padding, torch.zeros_like(text_padding))
        )
        positions = text_padding.long().cumsum(dim=1) - 1
        middle = []
        handle = language_model.layers[self.native_split_layer - 1].register_forward_hook(
            lambda _module, _inputs, output: middle.append(output)
        )
        try:
            with self.text_meta_lora.installed(language_model):
                (final, suffix), _ = bridge.forward(
                    attention_mask=mask, position_ids=positions, past_key_values=None,
                    inputs_embeds=[text_embeds.to(language_model.layers[0].self_attn.q_proj.weight.dtype), None],
                    use_cache=False, adarms_cond=[None, None],
                )
            expected = (*text_tokens.shape, self.image_width)
            if suffix is not None or final.shape != expected or len(middle) != 1 or middle[0].shape != expected:
                raise VideoProgramError("PI05 depth-matched text-only layout changed")
            return tuple(value[:, 1:].masked_fill(~text_padding[:, 1:, None], 0.0)
                         for value in (middle[0], final))
        finally:
            handle.remove()

    def _native_layout(self, core, language_mask, suffix_padding, suffix_attention):
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        prefix_padding = torch.cat((language_mask.new_ones(language_mask.shape[0], self.image_tokens),
                                    language_mask), dim=1)
        padding = torch.cat((prefix_padding, suffix_padding), dim=1)
        attention = torch.cat((torch.zeros_like(prefix_padding), suffix_attention), dim=1)
        return (core._prepare_attention_masks_4d(make_att_2d_masks(padding, attention)),
                padding.long().cumsum(dim=1) - 1)

    def _native_layers(self, core, prefix, suffix, language_mask, suffix_padding,
                       suffix_attention, adarms, start, stop, *, final_norm):
        """Reuse the actual joint layer operation, including mask, RoPE and AdaRMS."""
        from lerobot.policies.pi05.modeling_pi05 import compute_layer_complete
        from lerobot.policies.pi_gemma import layernorm_forward

        bridge = core.paligemma_with_expert
        language = bridge.paligemma.model.language_model
        expert = bridge.gemma_expert.model
        mask, positions = self._native_layout(core, language_mask, suffix_padding, suffix_attention)
        hidden = [prefix, suffix]
        conditions = [None, adarms]
        # Installation belongs inside every replay closure, never around checkpoint().
        with self.vl_meta_lora.installed(language), self.action_meta_lora.installed(expert):
            for index in range(start, stop):
                hidden = compute_layer_complete(
                    hidden, mask, positions, conditions,
                    layers=[language.layers[index], expert.layers[index]],
                    rotary_emb=language.rotary_emb,
                )
            if final_norm:
                hidden = [layernorm_forward(model.norm, value, condition)[0]
                          for model, value, condition in zip((language, expert), hidden, conditions, strict=True)]
        return tuple(hidden)

    def _embed_prefix(self, core, frames, language_tokens):
        """Keep frozen vision work outside the native activation replay."""
        bridge = core.paligemma_with_expert
        with torch.no_grad():
            patches = bridge.embed_image(self._prepare_images(frames))
            text = bridge.embed_language_tokens(language_tokens)
        if (
            patches.shape != (frames.shape[0] * 2, self.PATCHES_PER_CAMERA, self.image_width)
            or text.shape != (*language_tokens.shape, self.image_width)
        ):
            raise VideoProgramError("PI05 complete real prefix embedding layout changed")
        prefix = torch.cat((patches.reshape(frames.shape[0], self.image_tokens, self.image_width), text), dim=1)
        dtype = bridge.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
        return prefix.to(dtype)

    def _lower_native(self, core, prefix, language_mask, suffix,
                      suffix_padding, suffix_attention, adarms):
        return self._native_layers(
            core, prefix, suffix.to(prefix.dtype), language_mask, suffix_padding,
            suffix_attention, adarms, 0, self.native_split_layer, final_norm=False,
        )

    def _writeback(self, prefix, suffix, middle, task_span_mask, semantic_tokens):
        task_delta = self.prefix_writeback(middle[:, :semantic_tokens]).to(prefix.dtype)
        ordinal = (task_span_mask.long().cumsum(dim=1) - 1).clamp_min(0)
        prompt_delta = task_delta.gather(1, ordinal[..., None].expand(-1, -1, self.image_width))
        prompt_delta = prompt_delta.masked_fill(~task_span_mask[..., None], 0.0)
        prefix = torch.cat((prefix[:, :self.image_tokens],
                            prefix[:, self.image_tokens:] + prompt_delta), dim=1)
        suffix = suffix + self.horizon_writeback(middle[:, semantic_tokens:]).to(suffix.dtype)
        return prefix, suffix

    def _upper_read(self, core, prefix, suffix, middle, language_mask, task_span_mask,
                    text_queries, valid_task_tokens, suffix_padding, suffix_attention, adarms):
        prefix, suffix = self._writeback(prefix, suffix, middle, task_span_mask, valid_task_tokens.shape[1])
        prefix, suffix = self._native_layers(
            core, prefix, suffix, language_mask, suffix_padding, suffix_attention,
            adarms, self.native_split_layer, self.vl_meta_lora.layer_count, final_norm=True,
        )
        return self.final_read(prefix, suffix, text_queries, task_span_mask, valid_task_tokens)

    @staticmethod
    def _validate_input_shapes(frames, frame_indices, video_offsets,
                               language_tokens, language_mask, task_span_mask):
        if (
            frames.ndim != 5 or frames.shape[1:3] != (2, 3) or frames.shape[0] <= 0
            or frames.dtype != torch.uint8
            or frame_indices.shape != (frames.shape[0],) or frame_indices.dtype != torch.long
            or language_tokens.ndim != 2 or language_tokens.dtype != torch.long
            or language_tokens.shape[0] <= 0 or language_tokens.shape[1] <= 1
            or language_mask.shape != language_tokens.shape or language_mask.dtype != torch.bool
            or task_span_mask.shape != language_tokens.shape or task_span_mask.dtype != torch.bool
            or video_offsets.shape != (language_tokens.shape[0] + 1,) or video_offsets.dtype != torch.long
            or any(value.device != frames.device for value in
                   (frame_indices, video_offsets, language_tokens, language_mask, task_span_mask))
        ):
            raise VideoProgramError("invalid unified video-language batch shapes")

    def _validate_forward_batch(self, policy, frames, frame_indices, video_offsets,
                                language_tokens, language_mask, task_span_mask):
        self._validate_input_shapes(frames, frame_indices, video_offsets,
                                    language_tokens, language_mask, task_span_mask)
        if (
            int(video_offsets[0]) != 0 or int(video_offsets[-1]) != frames.shape[0]
            or bool((video_offsets[1:] <= video_offsets[:-1]).any())
            or bool((task_span_mask & ~language_mask).any())
            or not bool(task_span_mask.any(dim=1).all())
            or bool(task_span_mask[:, 0].any()) or not bool(language_mask[:, 0].all())
            or bool((frame_indices < 0).any())
        ):
            raise VideoProgramError("invalid unified video-language batch")
        counts = video_offsets[1:] - video_offsets[:-1]
        ids = torch.repeat_interleave(torch.arange(counts.numel(), device=frames.device), counts)
        same_video = ids[1:] == ids[:-1]
        if bool(((frame_indices[1:] <= frame_indices[:-1]) & same_video).any()):
            raise VideoProgramError("video frame indices must preserve their natural order")
        core = policy.model
        if (int(core.config.chunk_size) != self.action_horizon
                or int(core.config.max_action_dim) != self.padded_action_dim):
            raise VideoProgramError("PI05 Action Expert topology changed")
        task_counts = task_span_mask.sum(dim=1)
        valid_tasks = torch.arange(int(task_counts.max()), device=frames.device)[None] < task_counts[:, None]
        valid_frames = torch.arange(int(counts.max()), device=frames.device)[None] < counts[:, None]
        frame_slots = torch.arange(frames.shape[0], device=frames.device) - video_offsets[:-1].repeat_interleave(counts)
        flat_indices = ids * valid_frames.shape[1] + frame_slots
        return core, ids, valid_tasks, valid_frames, flat_indices

    @staticmethod
    def _video_grid(value, valid_frames, flat_indices):
        padded = value.new_zeros((valid_frames.numel(), *value.shape[1:]))
        return padded.index_copy(0, flat_indices, value).reshape(*valid_frames.shape, *value.shape[1:])

    def forward(self, policy, frames, frame_indices, video_offsets,
                language_tokens, language_mask, task_span_mask):
        """Return M[B,T,m+50,d], validity masks, shared role addresses and m."""
        core, ids, valid_tasks, valid_frames, flat_indices = self._validate_forward_batch(
            policy, frames, frame_indices, video_offsets, language_tokens, language_mask, task_span_mask
        )
        semantic_tokens = valid_tasks.shape[1]
        valid_roles = torch.cat((valid_tasks, valid_tasks.new_ones(valid_tasks.shape[0], self.action_horizon)), dim=1)
        addresses = token_role_addresses(self.type_embeddings, semantic_tokens, self.action_horizon)
        positions = self._video_grid(frame_indices, valid_frames, flat_indices)
        text_middle, text_final = self._checkpoint(
            partial(self._encode_text, core), language_tokens, task_span_mask, semantic_tokens
        )
        with torch.no_grad():
            suffix, suffix_padding, suffix_attention, adarms = core.embed_suffix(
                self.fixed_suffix_noise[None], self.fixed_suffix_noise.new_ones(1)
            )
        if (suffix.shape != (1, self.action_horizon, self.expert_width)
                or suffix_padding.shape != (1, self.action_horizon)
                or suffix_attention.shape != suffix_padding.shape):
            raise VideoProgramError("PI05 fixed-probe suffix layout changed")

        chunks, middle_rows = [], []
        for start in range(0, frames.shape[0], self.max_frames_per_encoder_call):
            stop = min(start + self.max_frames_per_encoder_call, frames.shape[0])
            selected = ids[start:stop]
            count = stop - start
            native_args = (suffix.expand(count, -1, -1), suffix_padding.expand(count, -1),
                           suffix_attention.expand(count, -1), adarms.expand(count, -1))
            prefix = self._embed_prefix(core, frames[start:stop], language_tokens[selected])
            prefix, horizon = self._checkpoint(
                partial(self._lower_native, core), prefix, language_mask[selected], *native_args,
            )
            middle_rows.append(self._checkpoint(
                self.middle_read, prefix, horizon, text_middle[selected], task_span_mask[selected], valid_tasks[selected]
            ))
            # Retain only the necessary native boundary, not a second full concatenation.
            chunks.append((start, stop, prefix, horizon))
        middle = self._video_grid(torch.cat(middle_rows), valid_frames, flat_indices)
        middle = self._checkpoint(self.middle_stack, middle, positions, valid_frames, valid_roles, addresses)
        middle = middle[valid_frames]

        final_rows = []
        for start, stop, prefix, horizon in chunks:
            selected = ids[start:stop]
            count = stop - start
            final_rows.append(self._checkpoint(
                partial(self._upper_read, core), prefix, horizon, middle[start:stop], language_mask[selected],
                task_span_mask[selected], text_final[selected], valid_tasks[selected],
                suffix_padding.expand(count, -1), suffix_attention.expand(count, -1), adarms.expand(count, -1),
            ))
        memory = self._video_grid(torch.cat(final_rows), valid_frames, flat_indices)
        memory = self._checkpoint(self.final_stack, memory, positions, valid_frames, valid_roles, addresses)
        return memory, valid_frames, valid_roles, addresses, semantic_tokens
