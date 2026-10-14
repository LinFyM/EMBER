"""Pure source-image prefix and the matched fresh V/VL compilation owner."""
from __future__ import annotations

import time
import torch
from torch import nn

from ember.operator_writer.native import _teacher_attention_kernel
from ember.writer.runtime import autocast
from .model import FactorEncoder, attention, positions

COMMON_SEED = 2026101417
LANGUAGE_SEED = 2026101418
HISTORY_SEED = 2026101419
FEATURE_SCHEMA = 'ember_image_only_gemma_compiler_v1'
SCHEMA = 'ember_image_gemma_history_compiler_v2'


class HistoryReader(nn.Module):
    """Actual MT events without an execution-hidden or variable-parent branch."""
    def __init__(self):
        super().__init__()
        self.pre_post = nn.Embedding(2, 256)
        self.facts = nn.Linear(63, 256)
        self.event_query = nn.Parameter(torch.randn(1, 256) / 16)
        self.event_attention, self.correspondence = attention(), attention()
        layer = nn.TransformerEncoderLayer(256, 4, 1024, dropout=0., batch_first=True, norm_first=True)
        self.history_time = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)

    def forward(self, observations, history, teaching):
        pair, facts = history['prepost'], history['facts']
        if pair.ndim != 2 or pair.shape[1] != 2 or facts.shape != (len(pair), 63) or not len(pair):
            raise ValueError('complete nonempty MT events and all63 actual facts required')
        memory = observations[pair] + self.pre_post.weight[None, :, None]
        q = self.event_query[None] + self.facts(facts)[:, None] + positions(history['record_indices'])[:, None]
        event = (q + self.event_attention(q, memory.flatten(1, 2), memory.flatten(1, 2), need_weights=False)[0])[:, 0]
        related = event + self.correspondence(event[None], teaching[None], teaching[None], need_weights=False)[0][0]
        return self.history_time(related[None])[0]


class ImageReader(nn.Module):
    """Eight spatial slots per real frame followed by four ordered video layers."""
    def __init__(self):
        super().__init__()
        self.image = nn.Linear(2048, 256)
        self.spatial = nn.Embedding(512, 256)
        self.frame_queries = nn.Parameter(torch.randn(8, 256) / 16)
        self.frame_attention = attention()
        layer = nn.TransformerEncoderLayer(256, 4, 1024, dropout=0., batch_first=True, norm_first=True)
        self.video_time = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(256)
        self.language = self.correspondence = None

    def teaching(self, images, indices, language=None):
        if images.ndim != 3 or images.shape[1:] != (512, 2048) or len(images) != len(indices):
            raise ValueError('Reader requires all synchronized image-prefix frames and original indices')
        memory = self.image(images) + self.spatial.weight
        tokens = None
        if self.language is not None:
            if language is None or language.ndim != 2 or language.shape[1] != 2048 or not len(language):
                raise ValueError('VL requires the exact valid frozen language embeddings')
            tokens = self.language(language)
            memory = torch.cat((memory, tokens[None].expand(len(images), -1, -1)), 1)
        elif language is not None:
            raise ValueError('V has no language input')
        query = self.frame_queries[None].expand(len(images), -1, -1)
        slots = query + self.frame_attention(query, memory, memory, need_weights=False)[0]
        ordered = self.video_time((slots + positions(indices)[:, None]).flatten(0, 1)[None])[0]
        if tokens is not None:
            tokens = tokens + self.correspondence(tokens[None], ordered[None], ordered[None], need_weights=False)[0][0]
            ordered = torch.cat((ordered, tokens))
        return ordered

    def forward(self, images, indices, history, language=None):
        if set(history) != {'images', 'prepost', 'facts', 'record_indices'}:
            raise ValueError('H model inputs exclude hidden, labels, identities and future outcomes')
        video = self.teaching(images, indices, language)
        # Project every actual observation once per forward, with gradients for all uses.
        observations = self.image(history['images']) + self.spatial.weight
        events = self.history(observations, history, video)
        return self.norm(torch.cat((video, events)))


class ImageCompiler(nn.Module):
    """One complete MT-centered endpoint; arm switches only the direct L entrance."""
    def __init__(self, layout, arm):
        super().__init__()
        if arm not in ('V', 'VL'):
            raise ValueError('only the two registered matched arms exist')
        self.arm, self.layout = arm, layout
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(COMMON_SEED)
            self.reader = ImageReader()
            self.decoder = FactorEncoder(layout, layers=6, endpoint=True)
        if arm == 'VL':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(LANGUAGE_SEED)
                self.reader.language, self.reader.correspondence = nn.Linear(2048, 256), attention()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(HISTORY_SEED)
            self.reader.history = HistoryReader()

    def forward(self, images, indices, history, language=None, *, checkpoint_layers=False):
        memory = self.reader(images, indices, history, language)
        parent = torch.zeros_like(self.layout.center)
        output = self.decoder(parent, memory=memory, checkpoint_layers=checkpoint_layers)
        return self.layout.unpack(self.layout.center + self.layout.scale * output.float())


@torch.no_grad()
def pure_features(runtime, task, demo, *, frame_chunk=32):
    """No L or suffix enters source Gemma; VL words are embedded separately."""
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, resize_with_pad_torch
    video = runtime.videos.load(task, demo)
    frames = torch.from_numpy(video.frames).to(runtime.device)
    if frames.ndim != 5 or frames.shape[1:3] != (2, 3) or frames.dtype != torch.uint8:
        raise ValueError('source image prefix needs real ordered dual RGB')
    if runtime._generator is not None or runtime.execution._active_state is not None:
        raise ValueError('image-only source must not initialize old Meta/G or a task execution context')
    core, bridge = runtime.policy.model, runtime.policy.model.paligemma_with_expert
    dtype = bridge.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
    runtime.policy.eval().requires_grad_(False)
    images, chunks = [], []
    started = time.monotonic()
    for first in range(0, len(frames), frame_chunk):
        selected = frames[first:first + frame_chunk]
        with autocast(runtime.device):
            pixels = selected.flatten(0, 1).float().div(255).permute(0, 2, 3, 1)
            pixels = (resize_with_pad_torch(pixels, 224, 224) * 2 - 1).permute(0, 3, 1, 2)
            raw = bridge.embed_image(pixels).reshape(len(selected), 512, 2048)
            padding = torch.ones(len(selected), 512, dtype=torch.bool, device=runtime.device)
            mask = core._prepare_attention_masks_4d(make_att_2d_masks(padding, torch.zeros_like(padding)))
            with _teacher_attention_kernel(mask):
                (z, suffix), _ = bridge(attention_mask=mask, position_ids=torch.arange(512, device=runtime.device)[None].expand(len(selected), -1),
                    past_key_values=None, inputs_embeds=[raw.to(dtype), None], use_cache=False, adarms_cond=[None, None])
            if suffix is not None or z.shape != (len(selected), 512, 2048) or not torch.isfinite(z).all():
                raise ValueError('pure image-prefix post-norm output changed')
            images.append(z.detach().to(dtype=torch.bfloat16, device='cpu').contiguous())
        chunks.append(len(selected))
    words, mask, _ = runtime.tokenizer([runtime.tasks[task].authority.language])
    with autocast(runtime.device):
        language = bridge.embed_language_tokens(words)[0][mask[0]].detach().cpu().contiguous()
    return dict(z_img=torch.cat(images), indices=torch.from_numpy(video.frame_indices.copy()), language=language,
        source=dict(model=runtime.source['model_path'], dual_rgb=True, stride=5, post_norm=True,
            Gemma_input='512 image tokens only', L_in_Gemma=False, action_suffix=False, probe=False,
            old_meta_loaded=False, raw_frames=video.raw_frame_count, frame_chunks=chunks, seconds=time.monotonic()-started))
