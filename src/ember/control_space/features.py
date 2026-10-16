"""Read-only full source image-prefix schema and bounded immutable cache views."""
from __future__ import annotations

from pathlib import Path

import torch
from functools import lru_cache


FEATURE_SCHEMA = 'ember_source_image_prefix_features_v1'
LEGACY_SCHEMA = 'ember_image_only_gemma_compiler_v1'
PREPROCESS = dict(camera_order=['agentview', 'eye_in_hand'], rotation_degrees=180,
    resize='resize_with_pad_torch', model_resolution=224, frame_stride=5, terminal_included=True,
    tokens_per_frame=512, token_width=2048)


def _indices(indices, raw_count):
    value = torch.as_tensor(indices)
    if (value.ndim != 1 or not len(value) or value.dtype not in (torch.int32, torch.int64)
            or type(raw_count) is not int or raw_count <= 0):
        raise ValueError('full video requires original nonempty integer frame indices')
    expected = list(range(0, raw_count, 5))
    if expected[-1] != raw_count - 1:expected.append(raw_count - 1)
    if value.tolist() != expected:raise ValueError('video lost stride5/start/terminal or was sampled twice')
    return value.to(dtype=torch.int64, device='cpu')


@lru_cache(maxsize=4)
def _read_cached(path):
    """Legacy language storage is memory-mapped but never selected, read or returned."""
    value = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    schema = value.get('schema_version');source = value['source']
    required = dict(dual_rgb=True, stride=5, post_norm=True, Gemma_input='512 image tokens only',
        L_in_Gemma=False, action_suffix=False, probe=False, old_meta_loaded=False)
    if (not value.get('complete') or schema not in (FEATURE_SCHEMA, LEGACY_SCHEMA)
            or not source.get('model') or any(source.get(k) != v for k,v in required.items())
            or (schema == FEATURE_SCHEMA and source.get('preprocess') != PREPROCESS)):
        raise ValueError('feature cache is not the registered frozen pure source image prefix')
    images = value['z_img'];indices = _indices(value['indices'], source['raw_frames'])
    if (images.dtype != torch.bfloat16 or images.shape != (len(indices), 512, 2048)
            or not bool(torch.isfinite(images).all())):
        raise ValueError('feature cache lost complete finite BF16 image tokens')
    video = value.get('video')
    if schema == LEGACY_SCHEMA:
        video = dict(kind='original', task=value['task'], demo=value['demo'])
    return dict(z_img=images, indices=indices, source=dict(source), video=video)


def load_cached(path, device='cpu'):
    value = _read_cached(path)
    return dict(z_img=value['z_img'].to(device), indices=value['indices'].to(device), source=value['source'])
