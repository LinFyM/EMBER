"""Language-aligned per-frame evidence for the PI05 v5.1 Writer."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class VideoProgramError(RuntimeError):
    """Raised when the sealed teacher-video semantic interface changes."""


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
        max_frames_per_encoder_call: int,
        action_horizon: int,
        padded_action_dim: int,
        initialization_seed: int,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        dimensions = (
            image_width,
            expert_width,
            program_width,
            text_meta_lora_rank,
            vl_meta_lora_rank,
            action_meta_lora_rank,
            max_frames_per_encoder_call,
            action_horizon,
            padded_action_dim,
        )
        if (
            any(value <= 0 for value in dimensions)
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
        self.text_meta_lora = MetaLoRAStack(
            paligemma_model.layers,
            text_meta_lora_rank,
        )
        self.vl_meta_lora = MetaLoRAStack(
            paligemma_model.layers,
            vl_meta_lora_rank,
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
            or int(task_span_mask.sum(dim=1).max()) > maximum_task_tokens
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
        for row in range(batch):
            selected = language_tokens[row, task_span_mask[row]]
            text_tokens[row, 1 : selected.numel() + 1] = selected
            text_padding[row, 1 : selected.numel() + 1] = True
        with torch.no_grad():
            text_embeds = bridge.embed_language_tokens(text_tokens)
        text_attention = torch.zeros_like(text_padding)
        mask = core._prepare_attention_masks_4d(
            make_att_2d_masks(text_padding, text_attention)
        )
        positions = torch.cumsum(text_padding, dim=1) - 1
        target_dtype = language_model.layers[0].self_attn.q_proj.weight.dtype
        with self.text_meta_lora.installed(language_model):
            (text_hidden, suffix_hidden), _ = bridge.forward(
                attention_mask=mask,
                position_ids=positions,
                past_key_values=None,
                inputs_embeds=[text_embeds.to(target_dtype), None],
                use_cache=False,
                adarms_cond=[None, None],
            )
        if (
            suffix_hidden is not None
            or text_hidden.shape != (
                batch,
                maximum_task_tokens + 1,
                self.image_width,
            )
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
        maximum_task_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

        bridge = core.paligemma_with_expert
        language_model = bridge.paligemma.model.language_model
        expert_model = bridge.gemma_expert.model
        images = self._prepare_images(frames)
        with torch.no_grad():
            image_tokens = bridge.embed_image(images)
            text_tokens = bridge.embed_language_tokens(language_tokens)
        if (
            image_tokens.shape[1:] != (
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
        mask = core._prepare_attention_masks_4d(
            make_att_2d_masks(padding, attention)
        )
        positions = torch.cumsum(padding, dim=1) - 1
        target_dtype = language_model.layers[0].self_attn.q_proj.weight.dtype
        with (
            self.vl_meta_lora.installed(language_model),
            self.action_meta_lora.installed(expert_model),
        ):
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

        language_hidden = prefix_hidden[:, self.NATIVE_IMAGE_TOKENS :]
        packed_language = self._pack_hidden(
            language_hidden,
            task_span_mask,
            maximum_task_tokens,
        )
        evidence = self.language_projection(packed_language)
        interaction = self.interaction_projection(suffix_hidden.mean(dim=1))
        return evidence, interaction

    def _validate_forward_batch(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
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
            or task_span_mask.shape != language_tokens.shape
            or task_span_mask.dtype != torch.bool
            or bool((task_span_mask & ~language_mask).any())
            or not bool(task_span_mask.any(dim=1).all())
            or bool(task_span_mask[:, 0].any())
            or int(frame_condition_ids.min()) < 0
            or int(frame_condition_ids.max()) >= conditions
        ):
            raise VideoProgramError("invalid frame-language semantic batch")
        counts = torch.bincount(frame_condition_ids, minlength=conditions)
        expected = torch.repeat_interleave(
            torch.arange(conditions, device=frames.device),
            counts,
        )
        if bool((counts <= 0).any()) or not torch.equal(
            frame_condition_ids,
            expected,
        ):
            raise VideoProgramError(
                "semantic frames must be contiguous by video condition"
            )
        core = policy.model
        if (
            int(core.config.chunk_size) != self.action_horizon
            or int(core.config.max_action_dim) != self.padded_action_dim
        ):
            raise VideoProgramError("PI05 Action Expert topology changed")
        task_counts = task_span_mask.sum(dim=1)
        maximum_task_tokens = int(task_counts.max())
        valid_task_tokens = (
            torch.arange(
                maximum_task_tokens,
                device=frames.device,
            )[None]
            < task_counts[:, None]
        )
        return core, valid_task_tokens, task_counts

    def forward(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return aligned text/video evidence and one interaction per frame."""

        core, valid_task_tokens, _ = self._validate_forward_batch(
            policy,
            frames,
            frame_condition_ids,
            language_tokens,
            language_mask,
            task_span_mask,
        )
        maximum_task_tokens = valid_task_tokens.shape[1]

        def invoke_text(
            token_values: torch.Tensor,
            span_values: torch.Tensor,
        ) -> torch.Tensor:
            return self._encode_text(
                core,
                token_values,
                span_values,
                maximum_task_tokens,
            )

        should_checkpoint = (
            self.activation_checkpointing
            and self.training
            and torch.is_grad_enabled()
        )
        if should_checkpoint:
            text_queries = checkpoint(
                invoke_text,
                language_tokens,
                task_span_mask,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            text_queries = invoke_text(language_tokens, task_span_mask)

        evidence_rows = []
        interaction_rows = []
        for start in range(0, frames.shape[0], self.max_frames_per_encoder_call):
            stop = min(start + self.max_frames_per_encoder_call, frames.shape[0])
            rows = torch.arange(start, stop, device=frames.device)
            selected = frame_condition_ids.index_select(0, rows)
            arguments = (
                frames.index_select(0, rows),
                language_tokens.index_select(0, selected),
                language_mask.index_select(0, selected),
                task_span_mask.index_select(0, selected),
            )

            def invoke_frames(
                frame_values: torch.Tensor,
                token_values: torch.Tensor,
                mask_values: torch.Tensor,
                span_values: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return self._encode_microbatch(
                    core,
                    frame_values,
                    token_values,
                    mask_values,
                    span_values,
                    maximum_task_tokens,
                )

            if should_checkpoint:
                evidence, interaction = checkpoint(
                    invoke_frames,
                    *arguments,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                evidence, interaction = invoke_frames(*arguments)
            evidence_rows.append(evidence)
            interaction_rows.append(interaction)
        return (
            text_queries,
            torch.cat(evidence_rows, dim=0),
            torch.cat(interaction_rows, dim=0),
            valid_task_tokens,
        )
