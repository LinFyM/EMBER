"""Pure frozen source image-prefix features, raw-video loading and atomic cache."""
from __future__ import annotations

from pathlib import Path
import os
import time

import numpy as np
import torch

from ember.operator_writer.native import _teacher_attention_kernel
from ember.writer.data import RawTeacherVideo
from ember.writer.runtime import autocast

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


def load_video(runtime, descriptor):
    """Original store already rotates; producer archives contain unrotated raw RGB."""
    if descriptor['kind'] == 'original':
        return runtime.videos.load(int(descriptor['task']), int(descriptor['demo']))
    if descriptor['kind'] != 'producer':raise ValueError('only original or actual producer RGB may teach')
    with np.load(descriptor['path'], allow_pickle=False) as archive:
        if set(archive.files) != {'frames', 'frame_indices'}:
            raise ValueError('producer teaching archive must contain RGB and its clock only')
        frames, indices = archive['frames'], archive['frame_indices']
    if (frames.dtype != np.uint8 or frames.ndim != 5 or frames.shape[1:] != (2, 256, 256, 3)
            or not len(frames) or len(frames) != len(indices)
            or not np.array_equal(indices, np.asarray(descriptor['frame_indices']))):
        raise ValueError('producer RGB/camera order or registered control clock changed')
    raw_count = int(indices[-1]) + 1
    _indices(indices, raw_count)
    frames = np.ascontiguousarray(frames[:, :, ::-1, ::-1, :].transpose(0, 1, 4, 2, 3))
    return RawTeacherVideo(frames, np.asarray(indices, dtype=np.int64), raw_count)


@torch.no_grad()
def extract(runtime, video, frame_chunk=32):
    """Run all512 real image tokens through frozen Gemma, with no L or suffix."""
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, resize_with_pad_torch
    frames = torch.from_numpy(video.frames)
    if (frames.dtype != torch.uint8 or frames.ndim != 5 or frames.shape[1:3] != (2, 3)
            or len(frames) != len(video.frame_indices) or not len(frames)
            or type(frame_chunk) is not int or frame_chunk <= 0):
        raise ValueError('source image prefix requires full ordered dual-camera uint8 RGB')
    indices = _indices(video.frame_indices, video.raw_frame_count)
    if getattr(runtime, '_generator', None) is not None or runtime.execution._active_state is not None:
        raise ValueError('pure source image extraction cannot overlap learned or execution adapters')
    core, bridge = runtime.policy.model, runtime.policy.model.paligemma_with_expert
    dtype = bridge.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
    runtime.policy.eval().requires_grad_(False)
    images, chunks = [], []
    started = time.monotonic()
    for first in range(0, len(frames), frame_chunk):
        selected = frames[first:first + frame_chunk].to(runtime.device)
        with autocast(torch.device(runtime.device)):
            pixels = selected.flatten(0, 1).float().div(255).permute(0, 2, 3, 1)
            pixels = (resize_with_pad_torch(pixels, 224, 224) * 2 - 1).permute(0, 3, 1, 2)
            raw = bridge.embed_image(pixels).reshape(len(selected), 512, 2048)
            padding = torch.ones(len(selected), 512, dtype=torch.bool, device=runtime.device)
            mask = core._prepare_attention_masks_4d(make_att_2d_masks(padding, torch.zeros_like(padding)))
            with _teacher_attention_kernel(mask):
                (z, suffix), _ = bridge(attention_mask=mask,
                    position_ids=torch.arange(512, device=runtime.device)[None].expand(len(selected), -1),
                    past_key_values=None, inputs_embeds=[raw.to(dtype), None], use_cache=False, adarms_cond=[None, None])
            if suffix is not None or z.shape != (len(selected), 512, 2048) or not bool(torch.isfinite(z).all()):
                raise ValueError('pure image-prefix post-norm output changed')
            images.append(z.detach().to(dtype=torch.bfloat16, device='cpu').contiguous())
        chunks.append(len(selected))
    return dict(z_img=torch.cat(images), indices=indices,
        source=dict(model=str(runtime.source['model_path']), dual_rgb=True, stride=5, post_norm=True,
            Gemma_input='512 image tokens only', L_in_Gemma=False, action_suffix=False, probe=False,
            old_meta_loaded=False, preprocess=dict(PREPROCESS), input_resolution=list(frames.shape[-2:]),
            raw_frames=video.raw_frame_count, frame_chunks=chunks, seconds=time.monotonic()-started))


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


def _identity(descriptor):
    if descriptor['kind'] == 'original':
        return dict(kind='original', task=int(descriptor['task']), demo=int(descriptor['demo']))
    if descriptor['kind'] == 'producer':
        return dict(kind='producer', path=str(Path(descriptor['path']).resolve()),
            frame_indices=list(map(int, descriptor['frame_indices'])))
    raise ValueError('unknown teaching descriptor')


def get_or_build(runtime, descriptor, destination, frame_chunk=32):
    destination = Path(destination);identity = _identity(descriptor)
    cached = Path(descriptor.get('cache_path', destination))
    if cached.exists():
        value = _read_cached(cached)
        if (value['video'] != identity
                or Path(value['source']['model']).resolve() != Path(runtime.source['model_path']).resolve()):
            raise ValueError('cache belongs to another teaching video or frozen source')
        return {key:value[key] for key in ('z_img','indices','source')}
    if cached != destination:
        raise FileNotFoundError('referenced immutable feature cache is missing: '+str(cached))
    value = extract(runtime, load_video(runtime, descriptor), frame_chunk=frame_chunk)
    record = dict(value,video=identity,schema_version=FEATURE_SCHEMA,complete=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f'.{os.getpid()}.partial')
    try:
        torch.save(record, temporary);temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return value


def worker(args):
    """One persistent source replica builds disjoint useful videos, then releases."""
    import json
    from ember.pi05_source_setup import initialize_distributed
    from ember.pi05_source_checkpoint import write_json_atomic
    from ember.reference.accounting import host_resources
    from .data import make_runtime,feature_entries
    context=initialize_distributed(require_numa=True,defer_process_group=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=True
    runtime=make_runtime(args.root,context.device)
    entries=[e for i,e in enumerate(feature_entries(args.root)) if i%args.feature_shards==args.feature_shard]
    # Useful long videos expose real frame-chunk headroom without a disposable probe.
    longest=sorted((e for e in entries if not Path(e['cache_path']).exists()),key=lambda e:-e['frames'])[:6]
    first={e['key'] for e in longest};entries=longest+[e for e in entries if e['key'] not in first]
    measures=[];failures=[];built=reused=frames=0;started=time.monotonic();best=8
    try:
        for entry in entries:
            destination=Path(entry['cache_path']);exists=destination.exists();candidate=best
            if not exists and len(measures)<6:candidate=(4,8,16,32,64,128)[len(measures)]
            tick=time.monotonic();torch.cuda.reset_peak_memory_stats();oom=[]
            while True:
                try:value=get_or_build(runtime,entry,destination,frame_chunk=candidate);break
                except torch.cuda.OutOfMemoryError:
                    oom.append(candidate);torch.cuda.empty_cache()
                    if candidate<=2:raise
                    candidate=max(2,candidate//2)
            seconds=time.monotonic()-tick;count=len(value['indices']);frames+=count
            if oom:
                failures.append(dict(key=entry['key'],failed_frame_chunks=oom,recovered_frame_chunk=candidate,seconds_including_retries=seconds))
                best=min(best,candidate)
            if exists:reused+=1
            else:
                built+=1
                measure=dict(key=entry['key'],requested_frame_chunk=candidate,actual_max_frame_chunk=max(value['source']['frame_chunks']),
                    frames=count,seconds=seconds,frames_per_second=count/max(seconds,1e-6),
                    peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,OOM_attempts=oom)
                if len(measures)<6:
                    measures.append(measure);finite=[m for m in measures if not m['OOM_attempts']]
                    best=max(finite or measures,key=lambda m:m['frames_per_second'])['requested_frame_chunk']
                    print('EMBER_WORK_EVENT '+json.dumps(dict(kind='feature_observation',**measure)),flush=True)
            del value
            # Coordinator waits on actual completed files/child events, never timed cache polls.
            print('EMBER_WORK_EVENT '+json.dumps(dict(kind='feature_complete',key=entry['key'],path=str(destination))),flush=True)
        result=dict(complete=True,shard=args.feature_shard,shards=args.feature_shards,new_videos=built,reused_videos=reused,
            frames=frames,seconds=time.monotonic()-started,chosen_frame_chunk=best,measurements=measures,OOM_recoveries=failures,resources=host_resources())
        write_json_atomic(args.root/'features'/f'worker_{args.feature_shard:02d}.json',result)
        return result
    finally:runtime.videos.close();runtime.execution.close()
