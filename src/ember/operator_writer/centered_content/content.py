"""One-pass fixed role density, pixel crop and real prefix content encoding."""
from contextlib import contextmanager
import gc
import time
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.runtime import autocast
from ember.writer.materialization import file_record
from ..conditional_read_write import rms_zero
from ..native import _teacher_attention_kernel
from ..data import FormalData
from .common import ROOT, OLD, ASSET, TASKS, HELD_TEACHERS, compact, save_compact, initialize
from .model import FrozenLocator


def grid(side, centers, widths):
    points = -1 + (2 * torch.arange(side, device=centers.device).float() + 1) / side
    y, x = torch.meshgrid(points, points, indexing='ij')
    return centers[..., None, None, :] + widths[..., None, None, :] * torch.stack((x, y), -1)


def crop_geometry(alpha):
    """One shared frame box per camera; fifty slot densities stay separate."""
    t = len(alpha); a = alpha.float().reshape(t, 50, 2, 16, 16)
    mass = a.sum((-1, -2))
    average = a.mean(1); prob = average / average.sum((-1, -2), keepdim=True).clamp_min(1e-30)
    points = -1 + (2 * torch.arange(16, device=alpha.device).float() + 1) / 16
    y, x = torch.meshgrid(points, points, indexing='ij'); coords = torch.stack((x, y), -1)
    center = (prob[..., None] * coords).sum((-3, -2))
    variance = (prob[..., None] * (coords - center[..., None, None, :]).square()).sum((-3, -2))
    half = (1.25 * (3 * variance).clamp_min(0).sqrt() + 1 / 16).clamp(2 / 16, 1)
    center = torch.maximum(torch.minimum(center, 1 - half), -1 + half)
    return center, half, mass, variance


def transform_density(alpha, center, half, mass):
    t = len(alpha)
    values = alpha.float().reshape(t, 50, 2, 16, 16).permute(0, 2, 1, 3, 4).reshape(t * 2, 50, 16, 16)
    sampled = F.grid_sample(values, grid(16, center, half).reshape(t * 2, 16, 16, 2),
                            mode='bilinear', padding_mode='border', align_corners=False)
    sampled = sampled.reshape(t, 2, 50, 16, 16).permute(0, 2, 1, 3, 4)
    sampled = sampled / sampled.sum((-1, -2), keepdim=True).clamp_min(1e-30) * mass[..., None, None]
    return sampled.reshape(t, 50, 512)


@contextmanager
def prefix_keys(policy):
    layers = policy.model.paligemma_with_expert.paligemma.model.language_model.layers
    result, handles = {}, []
    for i, layer in enumerate(layers):
        def hook(module, args, output, i=i):
            if output.shape[1] < 512 or output.shape[-1] != 256 or i in result:
                raise ValueError('real prefix18 pre-RoPE single-KV capture changed')
            result[i] = output[:, :512].detach()
        handles.append(layer.self_attn.k_proj.register_forward_hook(hook))
    try:
        yield result
    finally:
        for handle in handles:
            handle.remove()


def encode_prefix(policy, images, tokens, token_mask):
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
    core = policy.model; bridge = core.paligemma_with_expert; count = len(images)
    image_tokens = bridge.embed_image(images.flatten(0, 1))
    if image_tokens.shape[1:] != (256, 2048):
        raise ValueError('actual dual image token layout changed')
    language = bridge.embed_language_tokens(tokens.expand(count, -1))
    prefix = torch.cat((image_tokens.reshape(count, 512, 2048), language), 1)
    padding = torch.cat((torch.ones(count, 512, dtype=torch.bool, device=images.device), token_mask.expand(count, -1)), 1)
    positions = torch.cumsum(padding, 1) - 1
    mask = core._prepare_attention_masks_4d(make_att_2d_masks(padding, torch.zeros_like(padding)))
    # All visible prefix tokens are bidirectional; no suffix exists or is called.
    dtype = bridge.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
    config = bridge.paligemma.model.language_model.config
    original_implementation = config._attn_implementation
    config._attn_implementation = 'eager'
    try:
        with prefix_keys(policy) as captured, _teacher_attention_kernel(mask):
            bridge.forward(attention_mask=mask, position_ids=positions, past_key_values=None,
                           inputs_embeds=[prefix.to(dtype), None], use_cache=False, adarms_cond=[None, None])
    finally:
        config._attn_implementation = original_implementation
    if set(captured) != set(range(18)):
        raise ValueError('real content prefix missed a layer')
    return captured


@torch.no_grad()
def create(runtime, spec, identity):
    from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch
    if (ROOT / 'content/manifest.json').exists():
        raise ValueError('fixed content already complete; no repeated encoding')
    locator = FrozenLocator(runtime.device); records = []
    mode = runtime.writer.mode
    del runtime.writer
    runtime.writer = SimpleNamespace(mode=mode)  # FormalData.condition only consumes this mode flag.
    gc.collect(); torch.cuda.empty_cache()
    for tasks, role in ((TASKS, 'train'), ((16,), 'validation')):
        data = FormalData(ASSET, spec, query_labels=False, task_ids=tasks, role=role)
        try:
            for task in tasks:
                for teacher in ((0, 1) if task != 16 else HELD_TEACHERS):
                    path = ROOT / 'content' / f'task{task:03d}_teacher{teacher:02d}.pt'
                    if path.exists():
                        # Successful inputs are immutable; recovery only skips the
                        # recorded completed condition, without a second forward.
                        records.append(read_json(path.with_suffix('.json'))); continue
                    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
                    old = OLD / 'native' / path.name
                    fixed = torch.load(old, map_location='cpu', weights_only=True, mmap=True)
                    k, c = fixed['K'].to(runtime.device), fixed['c'].to(runtime.device)
                    condition, raw_count, sampled = data.condition(runtime, task, teacher)
                    frames, indices, tokens, mask = condition
                    if not torch.equal(indices.cpu(), fixed['frame_indices']) or sampled != len(k):
                        raise ValueError('RGB/frame alignment differs from original native')
                    with autocast(runtime.device):
                        alpha = locator(c, k).float()
                        raw_p = torch.stack([torch.einsum('tjp,tpd->tjd', alpha, rms_zero(k[:-1, l].float())).float()
                                             for l in range(18)])
                    center, half, mass, var = crop_geometry(alpha)
                    alpha_crop = transform_density(alpha, center, half, mass)
                    mass_error = (alpha_crop.reshape(sampled - 1, 50, 2, 256).sum(-1) - mass).abs().max()
                    if mass_error > 2e-5:
                        raise ValueError('crop changed original per-slot camera mass')
                    pixels = frames[:-1].flatten(0, 1).float().div(255).permute(0, 2, 3, 1)
                    original = (resize_with_pad_torch(pixels, 224, 224) * 2 - 1).permute(0, 3, 1, 2)
                    local = F.grid_sample(original, grid(224, center, half).reshape(-1, 224, 224, 2),
                                          mode='bilinear', padding_mode='border', align_corners=False).reshape(-1, 2, 3, 224, 224)
                    del k, c, fixed, original, pixels
                    # Actual videos are <=128 sampled frames: exhaust each legal
                    # video's origins, no unconsumed whole-prefix cache retained.
                    frame_chunk = 128; folded = [[] for _ in range(18)]
                    for start in range(0, len(local), frame_chunk):
                        with autocast(runtime.device):
                            capture = encode_prefix(runtime.policy, local[start:start + frame_chunk], tokens, mask)
                            for l in range(18):
                                p = torch.einsum('tjp,tpd->tjd', alpha_crop[start:start + frame_chunk], rms_zero(capture[l].float()))
                                folded[l].append(p.float().cpu())
                        del capture, p
                    centered = torch.stack([torch.cat(v) for v in folded])
                    value = dict(schema='native_role_centered_content_v1', task=task, teacher=teacher,
                        raw_key=raw_p, centered_rgb=centered, alpha=alpha, alpha_crop=alpha_crop,
                        crop_center=center, crop_half_width=half, camera_mass=mass, original_variance=var,
                        local_RGB=local, frame_indices=indices, origin_indices=indices[:-1],
                        full_image_area_fraction=half.prod(-1), source_native=str(old), source_locator=str(OLD / 'R/checkpoint64/binding.safetensors'),
                        selection_gradient=False, original_c_d_preserved=True, extra_suffix=0)
                    artifact = save_compact(value, path)
                    row = dict(task=task, teacher=teacher, artifact=artifact, sampled_frames=sampled,
                        origins=sampled - 1, reading_git=identity['commit'], frame_chunk=frame_chunk,
                        source_loads=1, actual_prefix_frames=sampled - 1, extra_suffix=0, additional_native=0,
                        seconds=time.monotonic() - started, peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
                        camera_mass_max_error=float(mass_error), area_fraction_mean=value['full_image_area_fraction'].mean(0).cpu().tolist(),
                        area_fraction_min=float(value['full_image_area_fraction'].min()), area_fraction_max=float(value['full_image_area_fraction'].max()))
                    write_json_atomic(path.with_suffix('.json'), row); records.append(row)
                    if len(records) == 1:
                        origins = read_json(ROOT / 'launch/resource_estimate.json')['origin_frames']
                        projected = artifact['bytes'] / (sampled - 1) * origins
                        write_json_atomic(ROOT / 'launch/first_content_storage.json', dict(**artifact,
                            estimated_all25_content_GiB=projected / 2**30, logical_storage_owned=True,
                            estimate_other_GiB=15, peak_estimate_GiB=15 + projected / 2**30))
                        if projected / 2**30 + 15 >= 24:
                            raise ValueError('first legal artifact predicts storage above24GiB')
                    print(dict(event='content', task=task, teacher=teacher, seconds=row['seconds'], bytes=artifact['bytes']), flush=True)
                    del condition, frames, indices, tokens, mask, value, alpha, alpha_crop, local, raw_p, centered, folded
        finally:
            data.close()
    if len(records) != 25:
        raise ValueError('fixed25 content panel incomplete')
    write_json_atomic(ROOT / 'content/manifest.json', dict(complete=True, records=records, reading_git=identity,
        input_teacher_fields=['dual_RGB', 'exact_language'], new_native_reads=0, crop_suffix_calls=0,
        prefix_mask='bidirectional visible prefix; no suffix', locator_parameters=327680,
        content_consumer='18 full50-slot P_l per origin; no quality selection', camera_mass_preserved=True))


def main():
    runtime, spec, identity, _ = initialize()
    create(runtime, spec, identity)
    write_json_atomic(ROOT / 'content/consumer_completion.json', dict(complete=True, reading_git=identity))


if __name__ == '__main__':
    main()
