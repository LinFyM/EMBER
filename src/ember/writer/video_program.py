"""Trainable PI05 language-aligned evidence for the K4 phase-aligned Writer."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ember.writer.temporal import RMSNorm


class VideoProgramError(RuntimeError):
    """Raised when the sealed teacher-video semantic interface changes."""


class MetaLoRAProjection(torch.nn.Module):
    def __init__(self, input_width: int, output_width: int, rank: int) -> None:
        super().__init__()
        if min(input_width, output_width, rank) <= 0:
            raise VideoProgramError("invalid Meta-LoRA dimensions")
        self.a = torch.nn.Parameter(torch.empty(rank, input_width))
        self.b = torch.nn.Parameter(torch.zeros(output_width, rank))
        torch.nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(value.to(self.a.dtype), self.a), self.b).to(value.dtype)


class MetaLoRAStack(torch.nn.Module):
    PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")

    def __init__(self, layers: Sequence[torch.nn.Module], rank: int) -> None:
        super().__init__()
        if not layers or rank <= 0:
            raise VideoProgramError("invalid Meta-LoRA stack")
        self.layer_count = len(layers)
        adapters = {}
        for layer_index, layer in enumerate(layers):
            for name in self.PROJECTIONS:
                projection = getattr(layer.self_attn, name)
                adapters[f"{layer_index}_{name}"] = MetaLoRAProjection(
                    int(projection.in_features), int(projection.out_features), rank
                )
        self.adapters = torch.nn.ModuleDict(adapters)

    @contextmanager
    def installed(self, model: torch.nn.Module) -> Iterator[None]:
        if len(model.layers) != self.layer_count:
            raise VideoProgramError("Meta-LoRA layer count changed")
        handles = []
        for layer_index, layer in enumerate(model.layers):
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


class TaskQueriedPatchGrounding(torch.nn.Module):
    NATIVE_IMAGE_TOKENS = 256

    def __init__(self, *, width: int, heads: int) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads:
            raise VideoProgramError("invalid task-queried patch grounding")
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
            or patch_content.shape[1] != self.NATIVE_IMAGE_TOKENS
            or valid_task_tokens.shape != task_queries.shape[:2]
        ):
            raise VideoProgramError("invalid task-query patch batch")
        batch, task_tokens, width = task_queries.shape
        query = self.query(self.query_norm(task_queries)).reshape(
            batch, task_tokens, self.heads, self.head_width
        ).transpose(1, 2)
        key = self.key(self.patch_norm(patch_content)).reshape(
            batch, self.NATIVE_IMAGE_TOKENS, self.heads, self.head_width
        ).transpose(1, 2)
        value = patch_content.reshape(
            batch, self.NATIVE_IMAGE_TOKENS, self.heads, self.head_width
        ).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        merged = attended.transpose(1, 2).reshape(batch, task_tokens, width)
        return self.output(merged).masked_fill(~valid_task_tokens[..., None], 0.0)


class Pi05LanguageAxialEncoder(torch.nn.Module):
    """Produce text queries and high-level per-frame video evidence."""

    NATIVE_IMAGE_TOKENS = 256

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
    ) -> None:
        super().__init__()
        dimensions = (
            image_width, expert_width, program_width, text_meta_lora_rank,
            vl_meta_lora_rank, action_meta_lora_rank, patch_grounding_heads,
            max_frames_per_encoder_call, action_horizon, padded_action_dim,
        )
        if any(value <= 0 for value in dimensions) or action_horizon != 50 or padded_action_dim != 32:
            raise VideoProgramError("invalid PI05 language-axial dimensions")
        self.image_width = int(image_width)
        self.expert_width = int(expert_width)
        self.program_width = int(program_width)
        self.max_frames_per_encoder_call = int(max_frames_per_encoder_call)
        self.action_horizon = int(action_horizon)
        self.padded_action_dim = int(padded_action_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.language_projection = torch.nn.Linear(image_width, program_width, bias=False)
        self.interaction_projection = torch.nn.Linear(expert_width, program_width, bias=False)
        self.patch_grounding = TaskQueriedPatchGrounding(
            width=program_width, heads=patch_grounding_heads
        )
        self.text_meta_lora = MetaLoRAStack(paligemma_model.layers, text_meta_lora_rank)
        self.vl_meta_lora = MetaLoRAStack(paligemma_model.layers, vl_meta_lora_rank)
        self.action_meta_lora = MetaLoRAStack(expert_model.layers, action_meta_lora_rank)
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed + 0x5A17)
        self.register_buffer(
            "fixed_suffix_noise",
            torch.randn(action_horizon, padded_action_dim, generator=generator),
            persistent=True,
        )

    @staticmethod
    def _prepare_images(frames: torch.Tensor) -> torch.Tensor:
        from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

        if frames.ndim != 4 or frames.shape[1] != 3 or frames.dtype != torch.uint8:
            raise VideoProgramError("teacher frames changed shape or dtype")
        value = frames.to(torch.float32).div_(255.0).permute(0, 2, 3, 1)
        return (resize_with_pad_torch(value, 224, 224) * 2.0 - 1.0).permute(0, 3, 1, 2)

    @staticmethod
    def _pack_hidden(
        hidden: torch.Tensor, task_span_mask: torch.Tensor, maximum: int
    ) -> torch.Tensor:
        ordinal = (task_span_mask.to(torch.long).cumsum(dim=1) - 1).clamp_min(0)
        packed = hidden.new_zeros(hidden.shape[0], maximum, hidden.shape[-1])
        return packed.scatter_add(
            1,
            ordinal[..., None].expand(-1, -1, hidden.shape[-1]),
            hidden * task_span_mask[..., None],
        )

    def _encode_text(
        self,
        core: torch.nn.Module,
        language_tokens: torch.Tensor,
        task_span_mask: torch.Tensor,
        maximum: int,
    ) -> torch.Tensor:
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        bridge = core.paligemma_with_expert
        batch = language_tokens.shape[0]
        tokens = torch.zeros(batch, maximum + 1, dtype=language_tokens.dtype, device=language_tokens.device)
        padding = torch.zeros_like(tokens, dtype=torch.bool)
        tokens[:, 0], padding[:, 0] = language_tokens[:, 0], True
        for row in range(batch):
            selected = language_tokens[row, task_span_mask[row]]
            tokens[row, 1 : selected.numel() + 1] = selected
            padding[row, 1 : selected.numel() + 1] = True
        with torch.no_grad():
            embeddings = bridge.embed_language_tokens(tokens)
        mask = core._prepare_attention_masks_4d(make_att_2d_masks(padding, torch.zeros_like(padding)))
        positions = torch.cumsum(padding, dim=1) - 1
        target_dtype = bridge.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
        with self.text_meta_lora.installed(bridge.paligemma.model.language_model):
            (hidden, suffix), _ = bridge.forward(
                attention_mask=mask,
                position_ids=positions,
                past_key_values=None,
                inputs_embeds=[embeddings.to(target_dtype), None],
                use_cache=False,
                adarms_cond=[None, None],
            )
        if suffix is not None or hidden.shape != (batch, maximum + 1, self.image_width):
            raise VideoProgramError("PI05 text-only hidden layout changed")
        return self.language_projection(hidden[:, 1:]).masked_fill(~padding[:, 1:, None], 0.0)

    def _encode_microbatch(
        self,
        core: torch.nn.Module,
        frames: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        text_queries: torch.Tensor,
        valid_task_tokens: torch.Tensor,
        maximum: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        bridge = core.paligemma_with_expert
        language_model = bridge.paligemma.model.language_model
        expert_model = bridge.gemma_expert.model
        with torch.no_grad():
            image_tokens = bridge.embed_image(self._prepare_images(frames))
            text_tokens = bridge.embed_language_tokens(language_tokens)
        prefix = torch.cat((image_tokens, text_tokens), dim=1)
        prefix_padding = torch.cat(
            (torch.ones(image_tokens.shape[:2], dtype=torch.bool, device=frames.device), language_mask), dim=1
        )
        suffix_noise = self.fixed_suffix_noise[None].expand(frames.shape[0], -1, -1)
        timestep = torch.ones(frames.shape[0], dtype=torch.float32, device=frames.device)
        suffix, suffix_padding, suffix_attention, adarms = core.embed_suffix(suffix_noise, timestep)
        padding = torch.cat((prefix_padding, suffix_padding), dim=1)
        attention = torch.cat((torch.zeros_like(prefix_padding), suffix_attention), dim=1)
        mask = core._prepare_attention_masks_4d(make_att_2d_masks(padding, attention))
        positions = torch.cumsum(padding, dim=1) - 1
        dtype = language_model.layers[0].self_attn.q_proj.weight.dtype
        with self.vl_meta_lora.installed(language_model), self.action_meta_lora.installed(expert_model):
            (prefix_hidden, suffix_hidden), _ = bridge.forward(
                attention_mask=mask,
                position_ids=positions,
                past_key_values=None,
                inputs_embeds=[prefix.to(dtype), suffix.to(dtype)],
                use_cache=False,
                adarms_cond=[None, adarms],
            )
        language_hidden = prefix_hidden[:, self.NATIVE_IMAGE_TOKENS :]
        packed_language = self._pack_hidden(language_hidden, task_span_mask, maximum)
        multimodal = self.language_projection(packed_language)
        patch_content = self.language_projection(prefix_hidden[:, : self.NATIVE_IMAGE_TOKENS])
        patch = self.patch_grounding(text_queries, patch_content, valid_task_tokens)
        interaction = self.interaction_projection(suffix_hidden.mean(dim=1))
        return multimodal + patch, patch, interaction

    def _validate_batch(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> tuple[torch.nn.Module, torch.Tensor]:
        conditions = language_tokens.shape[0]
        if (
            frames.ndim != 4
            or frame_condition_ids.shape != (frames.shape[0],)
            or language_mask.shape != language_tokens.shape
            or task_span_mask.shape != language_tokens.shape
            or bool((task_span_mask & ~language_mask).any())
            or not bool(task_span_mask.any(dim=1).all())
            or int(frame_condition_ids.min()) < 0
            or int(frame_condition_ids.max()) >= conditions
        ):
            raise VideoProgramError("invalid frame-language semantic batch")
        core = policy.model
        if int(core.config.chunk_size) != self.action_horizon or int(core.config.max_action_dim) != self.padded_action_dim:
            raise VideoProgramError("PI05 Action Expert topology changed")
        counts = task_span_mask.sum(dim=1)
        valid = torch.arange(int(counts.max()), device=frames.device)[None] < counts[:, None]
        return core, valid

    def forward(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        core, valid_task_tokens = self._validate_batch(
            policy, frames, frame_condition_ids, language_tokens, language_mask, task_span_mask
        )
        maximum = valid_task_tokens.shape[1]
        use_checkpoint = self.activation_checkpointing and self.training and torch.is_grad_enabled()

        def text_call(tokens: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
            return self._encode_text(core, tokens, spans, maximum)

        text_queries = (
            checkpoint(text_call, language_tokens, task_span_mask, use_reentrant=False, preserve_rng_state=False)
            if use_checkpoint
            else text_call(language_tokens, task_span_mask)
        )
        evidence_rows, patch_rows, interaction_rows = [], [], []
        for start in range(0, frames.shape[0], self.max_frames_per_encoder_call):
            stop = min(start + self.max_frames_per_encoder_call, frames.shape[0])
            selected = frame_condition_ids[start:stop]
            arguments = (
                frames[start:stop],
                language_tokens.index_select(0, selected),
                language_mask.index_select(0, selected),
                task_span_mask.index_select(0, selected),
                text_queries.index_select(0, selected),
                valid_task_tokens.index_select(0, selected),
            )

            def frame_call(*values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                return self._encode_microbatch(core, *values, maximum)

            output = (
                checkpoint(frame_call, *arguments, use_reentrant=False, preserve_rng_state=False)
                if use_checkpoint
                else frame_call(*arguments)
            )
            evidence_rows.append(output[0])
            patch_rows.append(output[1])
            interaction_rows.append(output[2])
        return (
            text_queries,
            torch.cat(evidence_rows),
            torch.cat(patch_rows),
            torch.cat(interaction_rows),
            valid_task_tokens,
        )
