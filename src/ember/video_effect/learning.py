"""Frozen source patches and the unique, task-balanced Phi learning consumer."""
from __future__ import annotations

import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ember.pi05_source_checkpoint import capture_rng, restore_rng, read_json, write_json_atomic
from ember.pi05_source_setup import initialize_deferred_process_group
from ember.proposal_writer.learning import memory_peak, synchronize_gradients
from ember.writer.runtime import autocast
from .perception import RelationPerception, loss
from .labels import TRAIN_TASKS, TRAIN_DEMOS, load_geometry_labels, label_metadata

SCHEMA = 'ember_video_effect_perception_v1'
PROBES = (0, 1, 12, 13, 20, 21, 32, 34)


def feature_file(root, task, demo):
    return Path(root) / 'patches' / f'task_{task:04d}_demo_{demo:02d}.pt'


@torch.no_grad()
def source_patches(runtime, task, demo, chunk):
    """Real spatial embed_image output; no action suffix, meta or Gemma proxy."""
    if (task, demo) not in {(t, d) for t in TRAIN_TASKS for d in TRAIN_DEMOS} | {
            (t, d) for t in PROBES for d in (0, 1)}:
        raise ValueError('video is outside the unique registered input pool')
    video = runtime.videos.load(task, demo)
    pieces = []
    started = time.monotonic()
    bridge = runtime.policy.model.paligemma_with_expert
    for first in range(0, len(video.frames), chunk):
        images = torch.from_numpy(video.frames[first:first + chunk]).to(runtime.device).float().div(255)
        pixels, _ = runtime.policy._preprocess_images({
            'observation.images.base_0_rgb': images[:, 0],
            'observation.images.left_wrist_0_rgb': images[:, 1]})
        with autocast(runtime.device):
            value = torch.cat([bridge.embed_image(p) for p in pixels[:2]], 1)
        if value.shape[1:] != (512, 2048) or not torch.isfinite(value).all():
            raise ValueError('real dual-image spatial patch consumer changed')
        pieces.append(value.detach().to(device='cpu', dtype=torch.bfloat16).contiguous())
    words, mask, _ = runtime.tokenizer([runtime.tasks[task].authority.language])
    with autocast(runtime.device):
        language = bridge.embed_language_tokens(words)[0].detach().cpu()
    return dict(patches=torch.cat(pieces), language=language, language_mask=mask[0].cpu(),
        frame_indices=torch.from_numpy(video.frame_indices.copy()),
        source=dict(model=runtime.source['model_path'], exact_language=runtime.tasks[task].authority.language,
            image_patch_shape=[512, 2048], action_suffix=False, execution_hidden=False,
            learned_meta=False, post_image_Gemma=False, frame_stride=5, dual_rgb=True,
            raw_frames=video.raw_frame_count, seconds=time.monotonic() - started, chunk=chunk))


def logical_stream():
    """Twenty frozen permutations; every task has exactly eight conditions/epoch."""
    pool = [(t, d) for t in TRAIN_TASKS for d in TRAIN_DEMOS]
    stream = []
    for epoch in range(20):
        order = np.random.default_rng(np.random.SeedSequence([20261015, epoch])).permutation(len(pool))
        stream.extend([pool[int(i)] for i in order])
    return [stream[i:i + 4] for i in range(0, len(stream), 4)]


def _save(root, update, model, optimizer, scheduler, context, stream, retained):
    rng = capture_rng(context)
    ranks = [None] * context.world_size if context.is_main else None
    if context.world_size > 1:
        dist.gather_object(rng, ranks, dst=0)
    else:
        ranks = [rng]
    if context.is_main:
        retained[:context.world_size] = ranks
        path = Path(root) / 'Phi' / 'checkpoints' / f'update_{update:08d}'
        path.mkdir(parents=True, exist_ok=False)
        torch.save(dict(schema=SCHEMA, Phi=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), next_update=update, sampler=dict(stream=stream, next_update=update),
            rank_rng=ranks, retained_rank_rng=retained,
            topology=dict(world_size=context.world_size, logical_batch=4),
            parameter_names=list(dict(model.named_parameters()))), path / 'state.pt')
        write_json_atomic(path / 'manifest.json', dict(schema=SCHEMA, complete=True, next_update=update,
            full_optimizer_sampler_RNG=True, logical_batch=4, world_size=context.world_size,
            only_compilation_endpoint=1120, recovery_only=update != 1120))
    if context.world_size > 1:
        dist.barrier()


def _load_inputs(args,context,stream,start):
    inputs = {}
    labels = {}
    required = {(t,d) for t in TRAIN_TASKS for d in TRAIN_DEMOS}
    if args.profile:
        required = {entry for pos,entry in enumerate(stream[start]) if pos % context.world_size==context.rank}
    for task,demo in sorted(required):
        value = torch.load(feature_file(args.root, task, demo), map_location='cpu', weights_only=False)
        if value['patches'].shape[1:] != (512, 2048):
            raise ValueError('Phi input is not full source spatial patches')
        inputs[task, demo] = value
        labels[task, demo] = load_geometry_labels(task, demo, value['frame_indices'])
    return inputs,labels


def train(args, context):
    """One implementation for physical 1–4 ranks, preserving logical B4."""
    if context.world_size not in (1, 2, 3, 4):
        raise ValueError('logical B4 requires 1–4 physical ranks')
    torch.manual_seed(20261015)
    model = RelationPerception().to(context.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9, .95), eps=1e-8, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _u: 1.)
    stream, start, retained = logical_stream(), 0, [None] * 4
    if args.resume:
        saved = torch.load(args.resume / 'state.pt', map_location='cpu', weights_only=False)
        if (saved['schema'] != SCHEMA or saved['sampler']['stream'] != stream
                or saved['parameter_names'] != list(dict(model.named_parameters()))
                or not read_json(args.resume / 'manifest.json')['complete']):
            raise ValueError('Phi complete checkpoint or registered input stream changed')
        model.load_state_dict(saved['Phi']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); start = saved['next_update']
        retained = list(saved['retained_rank_rng'])
        if retained[context.rank] is None:
            torch.manual_seed(20261015 + context.rank)
            retained[context.rank] = capture_rng(context)
        restore_rng(retained[context.rank], context)
    else:
        torch.manual_seed(20261015 + context.rank)
        np.random.seed(20261015 + context.rank); random.seed(20261015 + context.rank)
        retained[context.rank] = capture_rng(context)
    inputs,labels=_load_inputs(args,context,stream,start)
    initialize_deferred_process_group(context, rendezvous_root=args.root / 'rendezvous')
    path = (args.root / 'profiles' / f'world_{context.world_size}' if args.profile else args.root / 'Phi') / f'updates_rank_{context.rank:02d}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    limit = start + 1 if args.profile else 1120
    model.train(); started = time.monotonic()
    for u in range(start, limit):
        tick = time.monotonic(); optimizer.zero_grad(set_to_none=True)
        components = []
        torch.cuda.reset_peak_memory_stats(context.device)
        for pos, (task, demo) in enumerate(stream[u]):
            if pos % context.world_size != context.rank:
                continue
            item = inputs[task, demo]
            with autocast(context.device):
                prediction = model(item['patches'].to(context.device), item['language'].to(context.device),
                    item['language_mask'].to(context.device), item['frame_indices'].to(context.device))
                objective, metrics = loss(prediction, labels[task, demo])
            if not torch.isfinite(objective):
                raise ValueError('nonfinite physical perception objective')
            (objective / 4).backward()
            components.append(dict(task=task, demo=demo, **metrics))
            del prediction, objective
        synchronize_gradients(model, context)
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(gradient):
            raise ValueError('nonfinite shared perception credit')
        optimizer.step(); scheduler.step()
        torch.cuda.synchronize(context.device)
        record = dict(update=u + 1, conditions=stream[u], local_components=components,
            gradient_norm=float(gradient), peak=memory_peak(context.device),
            seconds=time.monotonic() - tick, world=context.world_size, logical_B=4,
            discarded_profile=bool(args.profile))
        if u==start:
            record['actual_perception_credit_by_module']={name:sum(float(p.grad.square().sum()) for p in module.parameters()
                if p.grad is not None)**.5 for name,module in model.named_children()}
        with path.open('a') as log:
            import json
            log.write(json.dumps(record) + '\n')
        if u + 1 in (560, 1120) and not args.profile:
            _save(args.root, u + 1, model, optimizer, scheduler, context, stream, retained)
    if args.profile:
        write_json_atomic(args.root / 'profiles' / f'world_{context.world_size}_rank_{context.rank}.json',
            dict(**record, elapsed_seconds=time.monotonic() - started,
                fresh_weights_optimizer_sampler_RNG_required_after_profile=True))
    elif context.is_main:
        write_json_atomic(args.root / 'Phi' / 'complete.json', dict(complete=True, next_update=1120,
            checkpoint=str(args.root / 'Phi/checkpoints/update_00001120'), unique_compilation_point=True))


def frozen_phi(root, device):
    checkpoint = Path(root) / 'Phi/checkpoints/update_00001120'
    manifest = read_json(checkpoint / 'manifest.json')
    if not manifest['complete'] or manifest['next_update'] != 1120:
        raise ValueError('compilation requires the unique full Phi endpoint')
    saved = torch.load(checkpoint / 'state.pt', map_location='cpu', weights_only=False)
    model = RelationPerception().to(device)
    model.load_state_dict(saved['Phi'])
    return model.eval().requires_grad_(False)
