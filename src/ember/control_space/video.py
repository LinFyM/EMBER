"""Frozen Reader900 ordered memory; original RGB teaching only, no legacy head."""
from pathlib import Path
import time

import torch
from torch import nn

from ember.pi05_source_checkpoint import read_json
from ember.proposal_writer.model import attention, positions
from ember.reference.contract import ASSETS, ROOT as REFERENCE_ROOT, SOURCE, TASKS
from ember.writer.runtime import autocast
from .cache import ReferencedMemoryCache
from .contract import PREVIOUS_ROOT

VIDEO_ROOT = REFERENCE_ROOT.parent/'video_initial_seed7'
READER_CHECKPOINT = VIDEO_ROOT/'checkpoints/update_00000900/state.pt'
SOURCE_MODEL = str(ASSETS/SOURCE/'policy')
VALIDATION = (3, 6, 11, 16, 23, 26, 31, 39)
FEATURE_SCHEMA = 'ember_source_image_prefix_features_v1'
LEGACY_SCHEMA = 'ember_image_only_gemma_compiler_v1'
PREPROCESS = dict(camera_order=['agentview', 'eye_in_hand'], rotation_degrees=180,
    resize='resize_with_pad_torch', model_resolution=224, frame_stride=5, terminal_included=True,
    tokens_per_frame=512, token_width=2048)


class MemoryReader(nn.Module):
    """The strict image/spatial/frame/time/norm subset of Reader900."""
    def __init__(self):
        super().__init__()
        self.image = nn.Linear(2048, 256)
        self.spatial = nn.Embedding(512, 256)
        self.frame_queries = nn.Parameter(torch.randn(8, 256)/16)
        self.frame_attention = attention()
        layer = nn.TransformerEncoderLayer(256, 4, 1024, dropout=0., batch_first=True, norm_first=True)
        self.video_time = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(256)

    def ordered_memory(self, images, indices):
        if (images.ndim != 3 or images.shape[1:] != (512, 2048) or not len(images)
                or indices.ndim != 1 or len(images) != len(indices)
                or indices.dtype not in (torch.int32, torch.int64)):
            raise ValueError('Reader requires full image-prefix frames and original integer indices')
        if not torch.is_autocast_enabled(images.device.type): images = images.to(self.image.weight.dtype)
        memory = self.image(images)+self.spatial.weight
        query = self.frame_queries[None].expand(len(images), -1, -1)
        slots = query+self.frame_attention(query, memory, memory, need_weights=False)[0]
        ordered = (slots+positions(indices.to(images.device))[:, None]).flatten(0, 1)[None]
        return self.norm(self.video_time(ordered))

    forward = ordered_memory


def load_memory_reader(device='cpu'):
    # Formal, trusted local CP contains NumPy RNG; mmap avoids materializing its
    # optimizer. Only the explicitly owned Reader tensors are selected or moved.
    saved = torch.load(READER_CHECKPOINT, map_location='cpu', weights_only=False, mmap=True)
    if (saved['schema'], saved['phase'], saved['next_update']) != ('ember_source_video_initial_r8_seed7_v1', 'video_initial', 900):
        raise ValueError('ordered memory requires the actual Reader900 checkpoint')
    with torch.random.fork_rng(devices=[]): reader = MemoryReader()
    names = set(reader.state_dict())
    owned = {key: value for key, value in saved['reader'].items() if key in names}
    reader.load_state_dict(owned, strict=True)
    del saved, owned
    return reader.to(device).eval().requires_grad_(False)


def _indices(indices, raw_count):
    value = torch.as_tensor(indices)
    if value.ndim != 1 or not len(value) or type(raw_count) is not int or raw_count <= 0:
        raise ValueError('teaching needs full original frame indices')
    expected = list(range(0, raw_count, 5))
    if expected[-1] != raw_count-1: expected.append(raw_count-1)
    if value.dtype not in (torch.int32, torch.int64) or value.tolist() != expected:
        raise ValueError('teaching lost stride5/start/terminal')
    return value.to(dtype=torch.int64, device='cpu')


def load_cached(entry):
    value = torch.load(entry['cache_path'], map_location='cpu', weights_only=True, mmap=True)
    source = value['source']; images = value['z_img']
    schema=value.get('schema_version')
    required = dict(model=SOURCE_MODEL, dual_rgb=True, stride=5, post_norm=True,
        Gemma_input='512 image tokens only', L_in_Gemma=False, action_suffix=False, probe=False, old_meta_loaded=False)
    identity = dict(kind='original', task=int(entry['task']), demo=int(entry['demo']))
    actual=(dict(kind='original',task=value.get('task'),demo=value.get('demo')) if schema==LEGACY_SCHEMA else value.get('video'))
    # The exact legacy schema is already referenced by the frozen §13 manifest
    # and its pure_features producer. Only z_img/indices are selected; stored L
    # is mmap-only and never read or returned. No general unknown-schema fallback.
    if (not value.get('complete') or schema not in (FEATURE_SCHEMA,LEGACY_SCHEMA) or actual!=identity
            or any(source.get(key)!=item for key,item in required.items())
            or (schema==FEATURE_SCHEMA and source.get('preprocess')!=PREPROCESS)):
        raise ValueError(f'original pure-source prefix identity/schema mismatch: {identity}, {schema}')
    indices = _indices(value['indices'], source['raw_frames'])
    if (images.dtype != torch.bfloat16 or images.shape != (len(indices), 512, 2048)
            or len(indices) != entry['frames'] or not bool(images.isfinite().all())):
        raise ValueError('original teaching lost complete finite BF16 prefix frames')
    return dict(z_img=images, indices=indices, source=source)


@torch.no_grad()
def extract(runtime, video, frame_chunk=32):
    """Exactly the existing real 512-token image-only Gemma path, RAM only."""
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, resize_with_pad_torch
    from ember.operator_writer.native import _teacher_attention_kernel
    frames = torch.from_numpy(video.frames)
    if (frames.dtype != torch.uint8 or frames.ndim != 5 or frames.shape[1:3] != (2, 3)
            or len(frames) != len(video.frame_indices) or not len(frames)
            or type(frame_chunk) is not int or frame_chunk <= 0):
        raise ValueError('pure image prefix requires all ordered dual-camera uint8 RGB')
    indices = _indices(video.frame_indices, video.raw_frame_count)
    if getattr(runtime, '_generator', None) is not None or runtime.execution._active_state is not None:
        raise ValueError('pure source image extraction requires a closed adapter context')
    if str(runtime.source['model_path']) != SOURCE_MODEL: raise ValueError('teaching source changed')
    core, bridge = runtime.policy.model, runtime.policy.model.paligemma_with_expert
    dtype = bridge.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
    images = []; started = time.monotonic()
    for first in range(0, len(frames), frame_chunk):
        selected = frames[first:first+frame_chunk].to(runtime.device)
        with autocast(torch.device(runtime.device)):
            pixels = selected.flatten(0, 1).float().div(255).permute(0, 2, 3, 1)
            pixels = (resize_with_pad_torch(pixels, 224, 224)*2-1).permute(0, 3, 1, 2)
            raw = bridge.embed_image(pixels).reshape(len(selected), 512, 2048)
            padding = torch.ones(len(selected), 512, dtype=torch.bool, device=runtime.device)
            mask = core._prepare_attention_masks_4d(make_att_2d_masks(padding, torch.zeros_like(padding)))
            with _teacher_attention_kernel(mask):
                (z, suffix), _ = bridge(attention_mask=mask,
                    position_ids=torch.arange(512, device=runtime.device)[None].expand(len(selected), -1),
                    past_key_values=None, inputs_embeds=[raw.to(dtype), None], use_cache=False, adarms_cond=[None, None])
            if suffix is not None or z.shape != (len(selected), 512, 2048) or not bool(z.isfinite().all()):
                raise ValueError('pure image-prefix post-norm output changed')
            images.append(z.detach().to(dtype=torch.bfloat16, device='cpu').contiguous())
    return dict(z_img=torch.cat(images), indices=indices,
        source=dict(model=SOURCE_MODEL, preprocess=PREPROCESS, raw_frames=video.raw_frame_count,
            seconds=time.monotonic()-started))


class MemoryStore:
    """One root-wide 1GiB cache for all original training and held M tensors."""
    def __init__(self, root, device='cpu', runtime=None, frame_chunk=32):
        self.root, self.device, self.runtime = Path(root), torch.device(device), runtime
        if type(frame_chunk) is not int or frame_chunk <= 0: raise ValueError('frame_chunk must be positive')
        self.frame_chunk, self.reader = frame_chunk, None
        manifest = read_json(VIDEO_ROOT/'features/manifest.json')
        originals = [entry for entry in manifest['entries'] if entry['kind'] == 'original']
        self.originals = {(int(entry['task']), int(entry['demo'])): entry for entry in originals}
        expected = {(task, demo) for task in TASKS for demo in range(50)}
        if set(self.originals) != expected or len(originals) != 1800 or sum(entry['frames'] for entry in originals) != 59114:
            raise ValueError('original-only teaching pool must contain exactly 1800 videos/59114 frames')
        self.training_entries = tuple(dict(task=task, demo=demo, frames=self.originals[task, demo]['frames'])
            for task, demo in sorted(self.originals))
        self.panels = {(int(row['task']), int(row['init_state_id'])): row
            for row in read_json(VIDEO_ROOT/'panels.json')['validation']}
        if (set(self.panels) != {(task, init) for task in VALIDATION for init in range(50)}
                or any({self.panels[task, init]['teacher_demo'] for init in range(50)} != set(range(50))
                    for task in VALIDATION)):
            raise ValueError('held RGB mapping must preserve all 400 unique task-video conditions')
        self.cache = ReferencedMemoryCache(self.root/'cache/M',PREVIOUS_ROOT/'cache/M')
        self.videos, self.measurements = None, []

    def _held_videos(self):
        if self.runtime is None: raise ValueError('held image extraction needs the caller-owned frozen source Runtime')
        if self.videos is None:
            from ember.writer.data import RawTeacherVideoStore
            from ember.writer.learning_data import load_learning_tasks
            metadata = load_learning_tasks(ASSETS, VALIDATION, role='validation', protocol_path='configs/libero_24_8_8_coverage_v1/protocol.json')
            self.videos = RawTeacherVideoStore([row.authority for row in metadata.values()], frame_stride=5, camera_view='dual')
        return self.videos

    def _memory(self, key, metadata, frames, teacher):
        detail = {}
        @torch.no_grad()
        def build():
            if self.reader is None:self.reader=load_memory_reader(self.device)
            prefix = teacher()
            if len(prefix['indices']) != frames: raise ValueError('M input frame count differs from its observation identity')
            detail.update(indices=prefix['indices'].tolist(), raw_frames=prefix['source']['raw_frames'],
                prefix_seconds=prefix['source'].get('seconds', 0.) if metadata['held'] else 0.)
            with autocast(self.device):
                result = self.reader.ordered_memory(prefix['z_img'].to(self.device), prefix['indices'].to(self.device))
            return result.float()  # Prefix has no persistent held file or retained store reference.
        value, measure = self.cache.get(key, metadata, frames*8*256*4, build)
        if (value.shape != (1, frames*8, 256) or value.dtype != torch.float32 or not bool(value.isfinite().all())):
            raise ValueError('M cache lost the complete finite ordered memory')
        self.measurements.append(dict(key=key, identity=metadata, frames=frames, frame_equivalent=frames,
            M_builds=int(not measure['hit']), M_hits=int(measure['hit']), prefix_frames_built=frames if metadata['held'] and not measure['hit'] else 0,
            prefix_frames_reread=frames if not metadata['held'] and not measure['hit'] else 0, new_prefix_files=0, **detail, **measure))
        return value

    def training(self, task, demo):
        entry = self.originals[(int(task), int(demo))]
        identity = dict(held=False, task=int(task), demo=int(demo), cache_path=entry['cache_path'],
            cache_bytes=Path(entry['cache_path']).stat().st_size, source=SOURCE_MODEL,
            reader=str(READER_CHECKPOINT), preprocess=PREPROCESS, frames=entry['frames'])
        return self._memory(f'original_t{int(task):04d}_d{int(demo):02d}', identity, entry['frames'], lambda: load_cached(entry))

    def held(self, task, init, panel):
        task, init = int(task), int(init)
        row = self.panels[(task, init)]
        fields = ('task', 'init_state_id', 'teacher_demo', 'noise_root')
        if task not in VALIDATION or any(panel[key] != row[key] for key in fields):
            raise ValueError('held teaching must match the frozen validation400 RGB-only mapping')
        demo = int(row['teacher_demo']); videos = self._held_videos()
        authority = videos.authorities[task]; raw_count, frames = videos.frame_counts(task, demo)
        identity = dict(held=True, task=task, demo=demo, authority=str(authority.path), authority_bytes=authority.expected_bytes,
            raw_frames=raw_count, frames=frames, source=SOURCE_MODEL, reader=str(READER_CHECKPOINT), preprocess=PREPROCESS)
        return self._memory(f'held_t{task:04d}_d{demo:02d}', identity, frames,
            lambda: extract(self.runtime, videos.load(task, demo), self.frame_chunk))

    def drain_measurements(self):
        rows, self.measurements = self.measurements, []
        return rows

    def close(self):
        if self.videos is not None: self.videos.close(); self.videos = None
