"""Bounded privileged probability routing; retired after the registered36 rows."""
from collections import OrderedDict
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from ember.operator_writer.bank import FrozenOperatorAdapter

OLD = Path('/data1/user/ymdai/ember_runs/self_image_attention_transfer_20261003/frozen_numeric_check')


def historical_module(name, frozen=OLD):
    qualified = 'ember.pi05_eval.' + name
    if qualified not in sys.modules:
        path = frozen / 'src/ember/pi05_eval' / (name + '.py')
        spec = importlib.util.spec_from_file_location(qualified, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    return sys.modules[qualified]


historical_module('image_attention_contract')
AttentionOwner = historical_module('image_attention_transfer').AttentionTransferAdapter


def redistribute(a, f, receiver):
    """FP32 formula; f is unnormalized visible coverage, never the old object q."""
    image = a[..., :512]
    if receiver is None:
        return a, torch.zeros_like(image[..., 0])
    fr = f[:, receiver]
    area = fr.sum(-1, keepdim=True)
    d = (f.sum(1) - fr)[:, None, None]
    possible = (image * d).sum(-1)
    m = torch.where(area[:, None] > 0, possible, 0)
    recipient = fr / area.clamp_min(torch.finfo(fr.dtype).tiny)
    routed = image * (1 - d) + m[..., None] * recipient[:, None, None]
    result = a.clone()
    result[..., :512] = torch.where(area[:, None, None] > 0, routed, image)
    return result, m


def render_segmentation(sim, camera):
    """Native robosuite ID-buffer decoding with explicit int32 for NumPy2."""
    from robosuite.utils.binding_utils import _MjSim_render_lock
    with _MjSim_render_lock:
        context = sim._render_context_offscreen
        context.render(256,256,camera_id=sim.model.camera_name2id(camera),segmentation=True)
        rgb = context.read_pixels(256,256).astype(np.int32)
        ids = rgb[...,0] + rgb[...,1]*256 + rgb[...,2]*65536
        ids[ids >= context.scn.ngeom+1] = 0
        registry = np.full((context.scn.ngeom+1,2),-1,dtype=np.int32)
        for geom in context.scn.geoms[:context.scn.ngeom]:
            if geom.segid != -1:
                registry[geom.segid+1] = (geom.objtype,geom.objid)
        return registry[ids]


def visible_coverage(env, observation, *, verify=False):
    """Current simulator visual geoms, original sensor convention and real224 mapping."""
    import mujoco
    from robosuite import macros
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

    owner, model = env.env, env.env.sim.model
    roots = {model.body_id2name(int(model.jnt_bodyid[j])).removesuffix('_main'):
             int(model.jnt_bodyid[j]) for j in range(model.njnt) if int(model.jnt_type[j]) == 0}
    names = sorted(n for n in roots if not any(s in n for s in ('robot', 'basket', 'tray')))
    assert 'butter_1' in names and 'orange_juice_1' in names
    groups = []
    for name in names:
        members = []
        for body in range(model.nbody):
            parent = body
            while parent and parent != roots[name]:
                parent = int(model.body_parentid[parent])
            if parent == roots[name]:
                members.append(body)
        groups.append(np.flatnonzero(np.isin(model.geom_bodyid, members)))
    convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    masks, pictures, checks = [], [], []
    for camera, obskey in [('agentview', 'agentview_image'), ('robot0_eye_in_hand', 'robot0_eye_in_hand_image')]:
        segmentation = render_segmentation(owner.sim, camera)[::convention]
        if verify:
            rgb = owner.sim.render(256, 256, camera_name=camera)[::convention]
            equal = np.array_equal(rgb, observation[obskey])
            assert equal, 'current segmentation render and actual own RGB are not synchronized'
            pictures.append(rgb[::-1, ::-1].copy())
            checks.append(equal)
        geom = segmentation[::-1, ::-1]
        masks.append(np.stack([(geom[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)) &
                               np.isin(geom[..., 1], group) for group in groups]))
    raw = torch.as_tensor(np.stack(masks).copy(), dtype=torch.float32)
    # Same square256->224 bilinear, align_corners=False as native resize_with_pad_torch.
    resized = F.interpolate(raw, size=(224, 224), mode='bilinear', align_corners=False)
    f = F.avg_pool2d(resized, 14).flatten(-2).permute(1, 0, 2).reshape(len(names), 512)
    assert (f >= 0).all() and (f <= 1).all() and (f.sum(0) <= 1.000001).all()
    return names, f, dict(camera_names=['agentview','robot0_eye_in_hand'],
        source='current own sim visual segmentation, geom type/id -> free-body descendants',
        entity_root_body_ids={n:roots[n] for n in names},
        entity_geom_ids={n:g.tolist() for n,g in zip(names,groups,strict=True)},
        decoder='native robosuite segmentation ID RGB buffer; explicit int32 decode for NumPy2',
        sensor_convention=macros.IMAGE_CONVENTION, canonical_rotate180=True,
        resize='native bilinear256->224, align_corners=False; avg14x14->16x16',
        current_RGB_equal=checks, pixels=np.stack(pictures) if verify else None,
        raw_masks=np.stack(masks) if verify else None)


class VisibleObjectAdapter(AttentionOwner):
    """Reuse the archived attention hook owner and canonical complete-LoRA inference."""

    def __init__(self, *, banks, cases, arm, **kwargs):
        FrozenOperatorAdapter.__init__(self, **kwargs)
        from safetensors.torch import load_file
        self.banks, self.cases, self.arm = banks, {c['case_id']:c for c in cases}, arm
        self.commons = {k:load_file(v['shared']['path'], device='cpu') for k,v in banks.items()}
        self.caches = {k:OrderedDict() for k in banks}
        layers = self.policy.model.paligemma_with_expert.gemma_expert.model.layers
        assert len(layers) == 18 and all(l.self_attn.num_key_value_groups == 8 for l in layers)
        self.layer_ids = {id(l.self_attn):i for i,l in enumerate(layers)}
        self.phase, self.flow_step, self.layout = None, 0, None
        self.donor_pi, self.donor_mass = {}, {}
        self.effects, self.packing = {}, {}

    def _state(self, case_id):
        case = self.cases[case_id]
        model, key = case['model'], case['condition_id']
        saved = self.bank, self.conditions, self.states, self.common
        self.bank = self.banks[model]
        self.conditions = {r['condition_id']:r for r in self.bank['conditions']}
        self.states, self.common = self.caches[model], self.commons[model]
        try:
            return FrozenOperatorAdapter._state(self, key)
        finally:
            self.bank, self.conditions, self.states, self.common = saved

    def set_visibility(self, slots, names, coverage):
        self.slot_ids = [s['readout_case']['case_id'] for s in slots]
        self.names = names
        self.f = torch.stack(coverage).to('cuda:0')
        receiver = {'parent':None, 'route_butter':'butter_1', 'route_orange_juice':'orange_juice_1'}[self.arm]
        self.receiver = None if receiver is None else names.index(receiver)
        self.visible_area = self.f.sum(-1).cpu().numpy()

    def _attention(self, module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        layer = self.layer_ids.get(id(module))
        if layer is None or self.phase is None:
            return self.base_attention(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
        assert not module.training and dropout == 0 and query.shape[1:3] == (8, 50)
        self.layers_seen.add(layer)
        key = self.repeat_kv(key, module.num_key_value_groups)
        value = self.repeat_kv(value, module.num_key_value_groups)
        logits = (query @ key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            logits = logits + attention_mask
        a = logits.softmax(-1, dtype=torch.float32)
        routed, m = redistribute(a, self.f, self.receiver)
        original = a.to(query.dtype) @ value
        actual = routed.to(query.dtype) @ value
        ideal_delta = (routed[..., :512] - a[..., :512]) @ value[..., :512, :].float()
        before = torch.einsum('bhsp,bep->bhe', a[..., :512], self.f) / 50
        after = torch.einsum('bhsp,bep->bhe', routed[..., :512], self.f) / 50
        stats = torch.cat([before.mean(1), after.mean(1), torch.stack([
            m.mean((1,2)), m.amin((1,2)), m.amax((1,2)),
            a[..., :512].sum(-1).mean((1,2)),
            (routed[..., :512].sum(-1)-a[..., :512].sum(-1)).abs().amax((1,2)),
            (routed[..., 512:]-a[..., 512:]).abs().amax((1,2,3)),
            (routed.sum(-1)-a.sum(-1)).abs().amax((1,2)), routed.amin((1,2,3)),
            ideal_delta.square().mean((1,2,3)).sqrt(),
            (actual.float()-original.float()).square().mean((1,2,3)).sqrt(),
            original.float().square().mean((1,2,3)).sqrt()], -1)], -1)
        self.current_stats[:, layer, self.flow_step] = stats
        return actual.transpose(1,2).contiguous(), routed.to(query.dtype)

    def _denoise(self, original, recipient, donor, **arguments):
        self.phase, self.layers_seen = 'route', set()
        result = original(**arguments)
        assert self.layers_seen == set(range(18))
        self.phase = None
        self.flow_step += 1
        return result

    @torch.no_grad()
    def predict_action_chunk(self, prepared, batch, *, noise, num_steps):
        assert [p.key for p in prepared] == self.slot_ids and num_steps == 10
        start = time.perf_counter()
        self.flow_step = 0
        self.current_stats = torch.zeros(len(prepared),18,10,2*len(self.names)+11,device=noise.device)
        with self._intervene(None, None):
            result = FrozenOperatorAdapter.predict_action_chunk(self, prepared, batch, noise=noise, num_steps=num_steps)
        from transformers.models.gemma import modeling_gemma
        assert modeling_gemma.eager_attention_forward is self.base_attention and self.flow_step == 10
        assert self.layout['image_block_lengths'] == [256,256,256] and self.layout['visible_tokens'] == 512
        values = self.current_stats.cpu().numpy()
        assert np.isfinite(values).all()
        assert values[...,2*len(self.names)+4:2*len(self.names)+7].max() < 3e-6
        assert values[...,2*len(self.names)+7].min() >= 0
        for i, case_id in enumerate(self.slot_ids):
            entry = self.effects.setdefault(case_id, dict(metrics=[], visible_area=[]))
            entry['metrics'].append(values[i])
            entry['visible_area'].append(self.visible_area[i])
        torch.cuda.synchronize()
        n = len(prepared)
        record = self.packing.setdefault(n, dict(calls=0,seconds=0.,reserved_peak_GiB=0.))
        record['calls'] += 1
        record['seconds'] += time.perf_counter()-start
        record['reserved_peak_GiB'] = max(record['reserved_peak_GiB'],torch.cuda.max_memory_reserved()/2**30)
        return result

    def save_effects(self, case_id, directory):
        entry = self.effects.pop(case_id)
        path = directory / 'attention_effects.npz'
        metric_names = ([f'before_mass/{n}' for n in self.names]+[f'after_mass/{n}' for n in self.names]+[
            'm_mean','m_min','m_max','image_mass_mean','image_mass_error','other_probability_error',
            'total_probability_error','probability_min','ideal_delta_y_RMS','actual_delta_y_RMS','original_y_RMS'])
        np.savez_compressed(path, stats=np.stack(entry['metrics']),visible_area=np.stack(entry['visible_area']),
            metric_names=np.asarray(metric_names),entity_names=np.asarray(self.names))
        return dict(path=str(path),bytes=path.stat().st_size,axis_order='replan,layer,flow,metric',
            head_slot_reduction='18 layers individually; all8 heads/all50 slots mean; m min/max retained',
            layout=self.layout,hooks_restored=True,receiver=self.receiver,arm=self.arm)
