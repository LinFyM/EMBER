"""Language-aligned per-frame evidence for the canonical EMBER-LMMPC Writer."""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ember.writer.temporal import RMSNorm


class VideoProgramError(RuntimeError):
    """Raised when the sealed teacher-video semantic interface changes."""


@dataclass(frozen=True)
class LanguageAxialProcessFeatures:
    """Full text, visual, and 50-token Action-probe evidence for code inference."""

    text_queries: torch.Tensor
    frame_evidence: torch.Tensor
    patch_evidence: torch.Tensor
    visual_patch_tokens: torch.Tensor
    action_probe_tokens: torch.Tensor
    valid_task_tokens: torch.Tensor


class MetaLoRAProjection(torch.nn.Module):
    """Writer-owned identity-initialized LoRA for one frozen projection."""

    def __init__(self, input_width: int, output_width: int, rank: int) -> None:
        super().__init__()
        if min(input_width, output_width, rank) <= 0:
            raise VideoProgramError("invalid Meta-LoRA dimensions")
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
            raise VideoProgramError("invalid Meta-LoRA stack")
        self.layer_count = len(layers)
        adapters: dict[str, torch.nn.Module] = {}
        for layer_index, layer in enumerate(layers):
            for name in self.PROJECTIONS:
                projection = getattr(layer.self_attn, name)
                input_width = int(getattr(projection, "in_features", 0))
                output_width = int(getattr(projection, "out_features", 0))
                if min(input_width, output_width) <= 0:
                    raise VideoProgramError(
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
            raise VideoProgramError("Meta-LoRA layer count changed")
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


class TaskQueriedPatchGrounding(torch.nn.Module):
    """Read per-frame image-position content with task-token queries."""

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
            or task_queries.shape[-1] != patch_content.shape[-1]
            or task_queries.shape[-1] != self.heads * self.head_width
            or patch_content.shape[1] != self.NATIVE_IMAGE_TOKENS
            or valid_task_tokens.shape != task_queries.shape[:2]
            or valid_task_tokens.dtype != torch.bool
        ):
            raise VideoProgramError("invalid task-query patch batch")
        batch, task_tokens, width = task_queries.shape
        patches = patch_content.shape[1]
        query = (
            self.query(self.query_norm(task_queries))
            .reshape(
                batch,
                task_tokens,
                self.heads,
                self.head_width,
            )
            .transpose(1, 2)
        )
        key = (
            self.key(self.patch_norm(patch_content))
            .reshape(
                batch,
                patches,
                self.heads,
                self.head_width,
            )
            .transpose(1, 2)
        )
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


class Pi05LanguageAxialEncoder(torch.nn.Module):
    """Produce text queries, aligned video evidence, and Action-Expert probes."""

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
        raw_visual_projection: bool = False,
    ) -> None:
        super().__init__()
        dimensions = (
            image_width,
            expert_width,
            program_width,
            action_meta_lora_rank,
            patch_grounding_heads,
            max_frames_per_encoder_call,
            action_horizon,
            padded_action_dim,
        )
        if (
            any(value <= 0 for value in dimensions)
            or text_meta_lora_rank < 0
            or vl_meta_lora_rank < 0
            or action_horizon != 50
            or padded_action_dim != 32
        ):
            raise VideoProgramError("invalid PI05 language-axial dimensions")
        self.image_width = int(image_width)
        self.expert_width = int(expert_width)
        self.program_width = int(program_width)
        self.max_frames_per_encoder_call = int(max_frames_per_encoder_call)
        self.action_horizon = int(action_horizon)
        self.padded_action_dim = int(padded_action_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.language_projection = torch.nn.Linear(
            image_width,
            program_width,
            bias=False,
        )
        self.interaction_projection = torch.nn.Linear(
            expert_width,
            program_width,
            bias=False,
        )
        self.visual_projection = (
            torch.nn.Linear(image_width, program_width, bias=False)
            if raw_visual_projection
            else None
        )
        self.patch_grounding = TaskQueriedPatchGrounding(
            width=program_width,
            heads=patch_grounding_heads,
        )
        self.text_meta_lora = (
            MetaLoRAStack(paligemma_model.layers, text_meta_lora_rank)
            if text_meta_lora_rank
            else None
        )
        self.vl_meta_lora = (
            MetaLoRAStack(paligemma_model.layers, vl_meta_lora_rank)
            if vl_meta_lora_rank
            else None
        )
        self.action_meta_lora = MetaLoRAStack(
            expert_model.layers,
            action_meta_lora_rank,
        )
        generator = torch.Generator(device="cpu").manual_seed(
            int(initialization_seed) + 0x5A17
        )
        self.register_buffer(
            "fixed_suffix_noise",
            torch.randn(
                action_horizon,
                padded_action_dim,
                dtype=torch.float32,
                generator=generator,
            ),
            persistent=True,
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
            raise VideoProgramError("teacher frames changed shape or dtype")
        value = frames.to(torch.float32).div_(255.0).permute(0, 2, 3, 1)
        value = resize_with_pad_torch(value, 224, 224)
        return (value * 2.0 - 1.0).permute(0, 3, 1, 2)

    @staticmethod
    def _pack_hidden(
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
            raise VideoProgramError("task-token hidden packing changed")
        ordinal = (task_span_mask.to(torch.long).cumsum(dim=1) - 1).clamp_min(0)
        packed = hidden.new_zeros(
            hidden.shape[0],
            maximum_task_tokens,
            hidden.shape[-1],
        )
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
        maximum_task_tokens: int,
    ) -> torch.Tensor:
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        bridge = core.paligemma_with_expert
        language_model = bridge.paligemma.model.language_model
        batch = language_tokens.shape[0]
        text_tokens = torch.zeros(
            batch,
            maximum_task_tokens + 1,
            dtype=language_tokens.dtype,
            device=language_tokens.device,
        )
        text_padding = torch.zeros_like(text_tokens, dtype=torch.bool)
        text_tokens[:, 0] = language_tokens[:, 0]
        text_padding[:, 0] = True
        ordinals = (task_span_mask.to(torch.long).cumsum(dim=1) - 1).clamp_min(0)
        text_tokens[:, 1:].scatter_add_(
            1,
            ordinals,
            language_tokens.masked_fill(~task_span_mask, 0),
        )
        packed_padding = torch.zeros(
            batch,
            maximum_task_tokens,
            dtype=torch.long,
            device=language_tokens.device,
        )
        packed_padding.scatter_add_(
            1,
            ordinals,
            task_span_mask.to(torch.long),
        )
        text_padding[:, 1:] = packed_padding.to(torch.bool)
        with torch.no_grad():
            text_embeds = bridge.embed_language_tokens(text_tokens)
        text_attention = torch.zeros_like(text_padding)
        mask = core._prepare_attention_masks_4d(
            make_att_2d_masks(text_padding, text_attention)
        )
        positions = torch.cumsum(text_padding, dim=1) - 1
        target_dtype = language_model.layers[0].self_attn.q_proj.weight.dtype
        if self.text_meta_lora is None:
            with torch.no_grad():
                (text_hidden, suffix_hidden), _ = bridge.forward(
                    attention_mask=mask,
                    position_ids=positions,
                    past_key_values=None,
                    inputs_embeds=[text_embeds.to(target_dtype), None],
                    use_cache=False,
                    adarms_cond=[None, None],
                )
        else:
            with self.text_meta_lora.installed(language_model):
                (text_hidden, suffix_hidden), _ = bridge.forward(
                    attention_mask=mask,
                    position_ids=positions,
                    past_key_values=None,
                    inputs_embeds=[text_embeds.to(target_dtype), None],
                    use_cache=False,
                    adarms_cond=[None, None],
                )
        if suffix_hidden is not None or text_hidden.shape != (
            batch,
            maximum_task_tokens + 1,
            self.image_width,
        ):
            raise VideoProgramError("PI05 text-only hidden layout changed")
        projected = self.language_projection(text_hidden[:, 1:])
        return projected.masked_fill(~text_padding[:, 1:, None], 0.0)

    def _encode_microbatch(
        self,
        core: torch.nn.Module,
        frames: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        text_queries: torch.Tensor,
        valid_task_tokens: torch.Tensor,
        maximum_task_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        bridge = core.paligemma_with_expert
        language_model = bridge.paligemma.model.language_model
        expert_model = bridge.gemma_expert.model
        images = self._prepare_images(frames)
        with torch.no_grad():
            image_tokens = bridge.embed_image(images)
            text_tokens = bridge.embed_language_tokens(language_tokens)
        if (
            image_tokens.shape[1:]
            != (
                self.NATIVE_IMAGE_TOKENS,
                self.image_width,
            )
            or text_tokens.shape[:2] != language_tokens.shape
        ):
            raise VideoProgramError("PI05 prefix embedding layout changed")

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
        suffix_noise = self.fixed_suffix_noise[None].expand(
            frames.shape[0],
            -1,
            -1,
        )
        timestep = torch.ones(
            frames.shape[0],
            dtype=torch.float32,
            device=frames.device,
        )
        suffix, suffix_padding, suffix_attention, adarms = core.embed_suffix(
            suffix_noise,
            timestep,
        )
        padding = torch.cat((prefix_padding, suffix_padding), dim=1)
        attention = torch.cat((prefix_attention, suffix_attention), dim=1)
        mask = core._prepare_attention_masks_4d(make_att_2d_masks(padding, attention))
        positions = torch.cumsum(padding, dim=1) - 1
        target_dtype = language_model.layers[0].self_attn.q_proj.weight.dtype
        vl_context = (
            self.vl_meta_lora.installed(language_model)
            if self.vl_meta_lora is not None
            else nullcontext()
        )
        with vl_context, self.action_meta_lora.installed(expert_model):
            (prefix_hidden, suffix_hidden), _ = bridge.forward(
                attention_mask=mask,
                position_ids=positions,
                past_key_values=None,
                inputs_embeds=[
                    prefix.to(target_dtype),
                    suffix.to(target_dtype),
                ],
                use_cache=False,
                adarms_cond=[None, adarms],
            )
        if (
            prefix_hidden.shape[:2] != prefix.shape[:2]
            or prefix_hidden.shape[-1] != self.image_width
            or suffix_hidden.shape
            != (frames.shape[0], self.action_horizon, self.expert_width)
        ):
            raise VideoProgramError("PI05 semantic hidden layout changed")

        evidence, patch_evidence, action_probe_tokens = self._project_joint_evidence(
            prefix_hidden,
            suffix_hidden,
            task_span_mask,
            text_queries,
            valid_task_tokens,
            maximum_task_tokens,
        )
        visual_patch_tokens = (
            self.visual_projection(image_tokens)
            if self.visual_projection is not None
            else patch_evidence
        )
        return (
            evidence,
            patch_evidence,
            action_probe_tokens,
            visual_patch_tokens,
        )

    def _project_joint_evidence(
        self,
        prefix_hidden: torch.Tensor,
        action_hidden: torch.Tensor,
        task_span_mask: torch.Tensor,
        text_queries: torch.Tensor,
        valid_task_tokens: torch.Tensor,
        maximum_task_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        language_hidden = prefix_hidden[:, self.NATIVE_IMAGE_TOKENS :]
        packed_language = self._pack_hidden(
            language_hidden,
            task_span_mask,
            maximum_task_tokens,
        )
        multimodal_evidence = self.language_projection(packed_language)
        patch_content = self.language_projection(
            prefix_hidden[:, : self.NATIVE_IMAGE_TOKENS]
        )
        patch_evidence = self.patch_grounding(
            text_queries,
            patch_content,
            valid_task_tokens,
        )
        evidence = multimodal_evidence + patch_evidence
        action_probe_tokens = self.interaction_projection(action_hidden)
        return evidence, patch_evidence, action_probe_tokens

    def _validate_language_batch(
        self,
        policy: torch.nn.Module,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
        if (
            language_tokens.ndim != 2
            or language_tokens.shape[0] <= 0
            or language_mask.shape != language_tokens.shape
            or language_mask.dtype != torch.bool
            or task_span_mask.shape != language_tokens.shape
            or task_span_mask.dtype != torch.bool
            or language_mask.device != language_tokens.device
            or task_span_mask.device != language_tokens.device
        ):
            raise VideoProgramError("invalid frame-language semantic batch")
        task_counts = task_span_mask.sum(dim=1)
        invalid_checks = torch.stack(
            (
                (task_span_mask & ~language_mask).any(),
                ~task_span_mask.any(dim=1).all(),
                task_span_mask[:, 0].any(),
            )
        )
        validation = (
            torch.cat(
                (
                    invalid_checks.to(torch.long),
                    task_counts.max().reshape(1),
                )
            )
            .to(device="cpu")
            .tolist()
        )
        if any(validation[:-1]):
            raise VideoProgramError("invalid frame-language semantic batch")
        core = policy.model
        if (
            int(core.config.chunk_size) != self.action_horizon
            or int(core.config.max_action_dim) != self.padded_action_dim
        ):
            raise VideoProgramError("PI05 Action Expert topology changed")
        maximum_task_tokens = int(validation[-1])
        valid_task_tokens = (
            torch.arange(
                maximum_task_tokens,
                device=language_tokens.device,
            )[None]
            < task_counts[:, None]
        )
        return core, valid_task_tokens, task_counts

    def _validate_forward_batch(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
        core, valid_task_tokens, task_counts = self._validate_language_batch(
            policy,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        conditions = language_tokens.shape[0]
        if (
            frames.ndim != 4
            or frames.shape[0] <= 0
            or frame_condition_ids.ndim != 1
            or frame_condition_ids.shape[0] != frames.shape[0]
            or frame_condition_ids.dtype != torch.long
            or frame_condition_ids.device != frames.device
            or language_tokens.device != frames.device
        ):
            raise VideoProgramError("invalid frame-language semantic batch")
        condition_counts = (
            frame_condition_ids[:, None]
            == torch.arange(conditions, device=frames.device)[None]
        ).sum(dim=0)
        if (
            ((frame_condition_ids < 0) | (frame_condition_ids >= conditions)).any()
            or (condition_counts <= 0).any()
            or (frame_condition_ids[1:] < frame_condition_ids[:-1]).any()
        ):
            raise VideoProgramError("invalid frame-language semantic batch")
        return core, valid_task_tokens, task_counts

    def _encode_text_features(
        self,
        core: torch.nn.Module,
        language_tokens: torch.Tensor,
        task_span_mask: torch.Tensor,
        valid_task_tokens: torch.Tensor,
    ) -> torch.Tensor:
        maximum_task_tokens = valid_task_tokens.shape[1]

        def invoke_text(token_values: torch.Tensor, span_values: torch.Tensor) -> torch.Tensor:
            return self._encode_text(core, token_values, span_values, maximum_task_tokens)

        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(
                invoke_text,
                language_tokens,
                task_span_mask,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return invoke_text(language_tokens, task_span_mask)

    def forward_text_features(
        self,
        policy: torch.nn.Module,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode exact language without reading any teacher-video frame."""
        core, valid_task_tokens, _ = self._validate_language_batch(
            policy,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        features = self._encode_text_features(
            core, language_tokens, task_span_mask, valid_task_tokens
        )
        return features, valid_task_tokens

    def forward_visual_features(
        self, policy: torch.nn.Module, frames: torch.Tensor
    ) -> torch.Tensor:
        """Encode raw image patches without language or Action probes."""
        if self.visual_projection is None:
            raise VideoProgramError("raw visual projection is unavailable")
        core = policy.model
        bridge = getattr(core, "paligemma_with_expert", None)
        if bridge is None:
            raise VideoProgramError("PI05 joint backbone changed")
        images = self._prepare_images(frames)
        with torch.no_grad():
            image_tokens = bridge.embed_image(images)
        if image_tokens.shape[1:] != (self.NATIVE_IMAGE_TOKENS, self.image_width):
            raise VideoProgramError("PI05 image embedding layout changed")
        return self.visual_projection(image_tokens)

    def forward_process_features(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> LanguageAxialProcessFeatures:
        """Return full evidence without collapsing the Action horizon."""

        core, valid_task_tokens, _ = self._validate_forward_batch(
            policy,
            frames,
            frame_condition_ids,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        should_checkpoint = (
            self.activation_checkpointing and self.training and torch.is_grad_enabled()
        )
        text_queries = self._encode_text_features(
            core,
            language_tokens,
            task_span_mask,
            valid_task_tokens,
        )
        maximum_task_tokens = valid_task_tokens.shape[1]

        evidence_rows = []
        patch_evidence_rows = []
        interaction_rows = []
        visual_rows = []
        for start in range(0, frames.shape[0], self.max_frames_per_encoder_call):
            stop = min(start + self.max_frames_per_encoder_call, frames.shape[0])
            rows = torch.arange(start, stop, device=frames.device)
            selected = frame_condition_ids.index_select(0, rows)
            arguments = (
                frames.index_select(0, rows),
                language_tokens.index_select(0, selected),
                language_mask.index_select(0, selected),
                task_span_mask.index_select(0, selected),
                text_queries.index_select(0, selected),
                valid_task_tokens.index_select(0, selected),
            )

            def invoke_frames(
                frame_values: torch.Tensor,
                token_values: torch.Tensor,
                mask_values: torch.Tensor,
                span_values: torch.Tensor,
                query_values: torch.Tensor,
                valid_token_values: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                return self._encode_microbatch(
                    core,
                    frame_values,
                    token_values,
                    mask_values,
                    span_values,
                    query_values,
                    valid_token_values,
                    maximum_task_tokens,
                )

            if should_checkpoint:
                evidence, patch_evidence, interaction, visual = checkpoint(
                    invoke_frames,
                    *arguments,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                evidence, patch_evidence, interaction, visual = invoke_frames(
                    *arguments
                )
            evidence_rows.append(evidence)
            patch_evidence_rows.append(patch_evidence)
            interaction_rows.append(interaction)
            visual_rows.append(visual)
        return LanguageAxialProcessFeatures(
            text_queries=text_queries,
            frame_evidence=torch.cat(evidence_rows, dim=0),
            patch_evidence=torch.cat(patch_evidence_rows, dim=0),
            visual_patch_tokens=torch.cat(visual_rows, dim=0),
            action_probe_tokens=torch.cat(interaction_rows, dim=0),
            valid_task_tokens=valid_task_tokens,
        )

    def forward(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Retain the canonical Writer interface while exposing full probes above."""

        features = self.forward_process_features(
            policy,
            frames,
            frame_condition_ids,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        return (
            features.text_queries,
            features.frame_evidence,
            features.patch_evidence,
            features.action_probe_tokens.mean(dim=1),
            features.valid_task_tokens,
        )
