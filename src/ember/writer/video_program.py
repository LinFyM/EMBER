"""Language-aligned camera evidence and full Action-Expert horizon read."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ember.writer.meta_lora import MetaLoRAStack
from ember.writer.temporal import RMSNorm


VIDEO_READ_MODES = (("agentview", "repeated_full"), ("dual", "repeated_full"))


class VideoProgramError(RuntimeError):
    """Raised when the sealed teacher-video semantic interface changes."""


def _frame_shard(total_frames: int, size: int, rank: int) -> tuple[int, int]:
    """Contiguous balanced slices, including empty ranks for short videos."""
    whole, remainder = divmod(total_frames, size)
    start = rank * whole + min(rank, remainder)
    return start, start + whole + int(rank < remainder)


class _GatherFrameRows(torch.autograd.Function):
    """Gather full frame rows; return the SUM of consumer cotangents to owners."""

    @staticmethod
    def forward(ctx, local_rows, group, total_frames):
        size, rank = dist.get_world_size(group), dist.get_rank(group)
        bounds = [_frame_shard(total_frames, size, index) for index in range(size)]
        ctx.start, ctx.stop = bounds[rank]
        ctx.group = group
        if local_rows.shape[0] != ctx.stop - ctx.start:
            raise VideoProgramError("native frame shard and gathered rows disagree")
        padded = local_rows.new_zeros((max(stop - start for start, stop in bounds), *local_rows.shape[1:]))
        padded[:local_rows.shape[0]].copy_(local_rows)
        gathered = [torch.empty_like(padded) for _ in range(size)]
        dist.all_gather(gathered, padded, group=group)
        return torch.cat([value[:stop - start] for value, (start, stop) in zip(gathered, bounds, strict=True)])

    @staticmethod
    def backward(ctx, gradient):
        # The decoder is replicated with disjoint query cotangents. Averaging here
        # would change the objective; parameter gradients are summed by the trainer.
        combined = gradient.contiguous().clone()
        dist.all_reduce(combined, op=dist.ReduceOp.SUM, group=ctx.group)
        return combined[ctx.start:ctx.stop], None, None


class TaskQueriedPatchGrounding(torch.nn.Module):
    """Read per-frame image-position content with text-only task queries."""

    def __init__(self, *, width: int, heads: int, image_tokens: int = 512) -> None:
        super().__init__()
        if min(width, heads) <= 0 or width % heads or image_tokens not in (256, 512):
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


class Pi05LanguageAxialEncoder(torch.nn.Module):
    """Produce text queries, aligned video evidence, and Action-Expert probes."""

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
        camera_view: str = "agentview",
        horizon_read: str = "repeated_full",
    ) -> None:
        super().__init__()
        dimensions = (
            image_width,
            expert_width,
            program_width,
            text_meta_lora_rank,
            vl_meta_lora_rank,
            action_meta_lora_rank,
            patch_grounding_heads,
            max_frames_per_encoder_call,
            action_horizon,
            padded_action_dim,
        )
        if (
            any(value <= 0 for value in dimensions)
            or action_horizon != 50
            or padded_action_dim != 32
            or (camera_view, horizon_read) not in VIDEO_READ_MODES
        ):
            raise VideoProgramError("invalid PI05 language-axial dimensions")
        self.image_width = int(image_width)
        self.expert_width = int(expert_width)
        self.program_width = int(program_width)
        self.max_frames_per_encoder_call = int(max_frames_per_encoder_call)
        self.action_horizon = int(action_horizon)
        self.padded_action_dim = int(padded_action_dim)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.camera_view = camera_view
        self.camera_count = 2 if camera_view == "dual" else 1
        self.image_tokens = self.camera_count * self.PATCHES_PER_CAMERA
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
        self.patch_grounding = TaskQueriedPatchGrounding(
            width=program_width,
            heads=patch_grounding_heads,
            image_tokens=self.image_tokens,
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

    def _valid_frame_layout(self, frames: torch.Tensor) -> bool:
        channels = (2, 3) if self.camera_count == 2 else (3,)
        return frames.ndim == len(channels) + 3 and frames.shape[1:1 + len(channels)] == channels

    def _prepare_images(self, frames: torch.Tensor) -> torch.Tensor:
        from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

        if (
            not self._valid_frame_layout(frames)
            or frames.shape[0] <= 0
            or frames.dtype != torch.uint8
        ):
            raise VideoProgramError("teacher frames changed shape or dtype")
        pixels = frames.flatten(0, 1) if self.camera_count == 2 else frames
        value = pixels.to(torch.float32).div_(255.0).permute(0, 2, 3, 1)
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
        text_queries: torch.Tensor,
        valid_task_tokens: torch.Tensor,
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
            image_tokens.shape != (
                frames.shape[0] * self.camera_count,
                self.PATCHES_PER_CAMERA,
                self.image_width,
            )
            or text_tokens.shape[:2] != language_tokens.shape
        ):
            raise VideoProgramError("PI05 prefix embedding layout changed")
        # Agentview is real in both modes; dual appends synchronized wrist patches.
        image_tokens = image_tokens.reshape(
            frames.shape[0], self.image_tokens, self.image_width
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

        language_hidden = prefix_hidden[:, self.image_tokens :]
        packed_language = self._pack_hidden(
            language_hidden,
            task_span_mask,
            maximum_task_tokens,
        )
        multimodal_evidence = self.language_projection(packed_language)
        patch_content = self.language_projection(
            prefix_hidden[:, : self.image_tokens]
        )
        patch_evidence = self.patch_grounding(
            text_queries,
            patch_content,
            valid_task_tokens,
        )
        evidence = multimodal_evidence + patch_evidence
        # H keeps the native stream dtype; projected E may use the autocast dtype.
        return evidence, suffix_hidden.to(target_dtype)

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
            not self._valid_frame_layout(frames)
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

    def _collect_frame_rows(self, rows, reference, shape, total_frames, group, *, dtype=None):
        if rows:
            local = torch.cat(rows, dim=0)
        else:
            # Empty owners still participate in both gather backwards without
            # invoking the native policy on invented frames.
            dtype = reference.dtype if dtype is None else dtype
            local = reference.new_empty((0, *shape), dtype=dtype)
            local = local + reference.reshape(-1)[0].to(dtype) * 0.0
        if group is None or dist.get_world_size(group) == 1:
            return local
        # Collectives must stay outside every activation-checkpoint closure.
        return _GatherFrameRows.apply(local, group, total_frames)

    def forward(
        self,
        policy: torch.nn.Module,
        frames: torch.Tensor,
        frame_condition_ids: torch.Tensor,
        language_tokens: torch.Tensor,
        language_mask: torch.Tensor,
        task_span_mask: torch.Tensor,
        *,
        frame_parallel_group=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return text, E, initial mean-projected p, full native H, and token mask."""

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

        local_start, local_stop = 0, frames.shape[0]
        if frame_parallel_group is not None:
            size, rank = dist.get_world_size(frame_parallel_group), dist.get_rank(frame_parallel_group)
            if size <= 0 or rank < 0:
                raise VideoProgramError("native frame rank is outside its process group")
            local_start, local_stop = _frame_shard(frames.shape[0], size, rank)
        evidence_rows = []
        horizon_rows = []
        for start in range(local_start, local_stop, self.max_frames_per_encoder_call):
            stop = min(start + self.max_frames_per_encoder_call, local_stop)
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
            ) -> tuple[torch.Tensor, torch.Tensor]:
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
                evidence, horizon = checkpoint(
                    invoke_frames,
                    *arguments,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                evidence, horizon = invoke_frames(*arguments)
            evidence_rows.append(evidence)
            horizon_rows.append(horizon)
        evidence = self._collect_frame_rows(
            evidence_rows, text_queries, (maximum_task_tokens, self.program_width),
            frames.shape[0], frame_parallel_group,
        )
        horizon = self._collect_frame_rows(
            horizon_rows, text_queries, (self.action_horizon, self.expert_width),
            frames.shape[0], frame_parallel_group,
            dtype=core.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype,
        )
        # Only the initial A state is a mean. Every subsequent read receives H50.
        initial = self.interaction_projection(horizon.to(torch.float32).mean(dim=1).to(horizon.dtype))
        return (
            text_queries,
            evidence,
            initial,
            horizon,
            valid_task_tokens,
        )
