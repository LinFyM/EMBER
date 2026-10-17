"""Frozen Reader900 ordered memory; original RGB teaching only, no legacy head."""
from pathlib import Path

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


class MemoryStore:
    """Read-only original teaching memory; no held/producers or old decoder."""
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
        self.cache = ReferencedMemoryCache(self.root/'cache/M',PREVIOUS_ROOT/'cache/M')
        self.measurements = []

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

    def drain_measurements(self):
        rows, self.measurements = self.measurements, []
        return rows

    def close(self):
        self.reader = None
