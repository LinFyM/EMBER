"""Fixed task16 native key capture; uses the existing source/prefix owners."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import time

import h5py
import numpy as np
import torch

from ember.ecp.policy_effects import ExecutionPolicyPrefix, prepare_prefix_features_and_cache
from ember.pi05_processing import Pi05TeacherPrefixTokenizer, quat2axisangle
from ember.pi05_source_checkpoint import read_json, write_json_atomic

ROOT = Path('/data1/user/ymdai/ember_runs/native_role_address_20261004')
ASSET = Path('/data1/user/ymdai/projects/EMBER')
REPO = Path(__file__).resolve().parents[3]
PARENT_BANK = Path('/data1/user/ymdai/ember_runs/conditional_A_reexpression_diagnostic_20261002/Original/bank/panel_teacher0.json')
TEACHERS = (47, 33, 28, 46, 32, 1, 24, 43)
LANGUAGE = 'pick up the butter and place it in the basket'


def roi_weights(box):
    x0, y0, x1, y1 = map(float, box)
    assert 0 <= x0 < x1 <= 256 and 0 <= y0 < y1 <= 256
    lo = np.arange(16, dtype=float) * 16
    wx = np.maximum(0, np.minimum(lo+16, x1)-np.maximum(lo, x0))
    wy = np.maximum(0, np.minimum(lo+16, y1)-np.maximum(lo, y0))
    weights = (wy[:, None]*wx[None, :]).reshape(-1)/((x1-x0)*(y1-y0))
    assert abs(weights.sum()-1) < 1e-12
    return weights


def inputs():
    visible = ROOT/'analysis/visible_input'
    source, roi = read_json(visible/'sources.json'), read_json(visible/'roi_registration.json')
    assert source['language'] == LANGUAGE and len(source['records']) == len(roi['records']) == 24
    out = []
    for record, registered in zip(source['records'], roi['records'], strict=True):
        assert all(record[k] == registered[k] for k in ('init', 'teacher', 'source'))
        assert record['teacher'] == TEACHERS[record['init']] and registered['camera'] == 0
        kind, item = record['source']['kind'], {**record, 'roi': registered['boxes_xyxy']}
        src = record['source']
        if kind == 'teacher':
            assert src['frame'] == 0 and src['demo'] == record['teacher']
            with h5py.File(src['hdf5'], 'r') as f:
                rgb = np.stack([f[f'data/demo_{src["demo"]}/obs/{k}'][0]
                    for k in ('agentview_rgb', 'eye_in_hand_rgb')])[:, ::-1, ::-1].copy()
            assert rgb.shape == (2, 128, 128, 3)
        else:
            assert kind in ('original', 'swapped') and src['already_rotated_180']
            with np.load(src['npz'], allow_pickle=False) as f:
                rgb = f['after_rgb'].copy()
                state = np.concatenate((f['after_eef_pos'], quat2axisangle(f['after_eef_quat']),
                    f['after_gripper_qpos'])).astype(np.float32)
            assert rgb.shape == (2, 256, 256, 3) and state.shape == (8,)
            item['state8'] = torch.from_numpy(state)
        assert rgb.dtype == np.uint8
        item['rgb'] = torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous()
        item['weights'] = {name:torch.from_numpy(roi_weights(box)) for name, box in item['roi'].items()}
        out.append(item)
    for init in range(8):
        pair = [x for x in out if x['init'] == init and x['source']['kind'] != 'teacher']
        assert torch.equal(pair[0]['state8'], pair[1]['state8'])
    return out


@contextmanager
def capture_hooks(policy):
    """Actual prefix projection outputs; fail if any Action Expert is executed."""
    layers = policy.model.paligemma_with_expert.paligemma.model.language_model.layers
    assert len(layers) == 18
    captured, metadata, handles = {}, {}, []

    def projection(index, _module, _args, output):
        assert index not in captured and output.shape[-1] == 256
        captured[index] = output.detach()

    def prefix_args(_module, _args, kwargs):
        assert not metadata and kwargs['use_cache']
        metadata.update(mask=kwargs['attention_mask'].detach(), positions=kwargs['position_ids'].detach())

    def prohibited(*_args, **_kwargs):
        raise RuntimeError('contract forbids suffix/action execution')

    try:
        for i, layer in enumerate(layers):
            attention = layer.self_attn
            assert attention.config.num_key_value_heads == 1 and attention.config.num_attention_heads == 8
            assert attention.head_dim == 256 and attention.scaling == 1/16
            handles.append(attention.k_proj.register_forward_hook(lambda *args, index=i:projection(index, *args)))
        # The bridge calls language_model.forward directly; its module hooks would
        # be bypassed. Decoder layer zero receives the actual mask/positions.
        handles.append(layers[0].register_forward_pre_hook(prefix_args, with_kwargs=True))
        for module in (policy.model.paligemma_with_expert.gemma_expert.model,
                       policy.model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj,
                       policy.model.action_in_proj, policy.model.action_out_proj):
            handles.append(module.register_forward_pre_hook(prohibited))
        yield captured, metadata
    finally:
        for handle in handles:
            handle.remove()


def prefix_batch(policy, processor, tokenizer, records):
    from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    frames = torch.stack([x['rgb'] for x in records]).to('cuda')
    count = len(records)
    if records[0]['source']['kind'] == 'teacher':
        assert all(x['source']['kind'] == 'teacher' for x in records)
        pixels = frames.flatten(0, 1).float().div(255).permute(0, 2, 3, 1)
        resized = (resize_with_pad_torch(pixels, 224, 224)*2-1).permute(0, 3, 1, 2)
        resized = resized.reshape(count, 2, 3, 224, 224)
        images = [resized[:, i] for i in range(2)]
        masks = [torch.ones(count, dtype=torch.bool, device='cuda') for _ in range(2)]
        tokens, token_mask, _ = tokenizer([LANGUAGE]*count)
    else:
        states = torch.stack([x['state8'] for x in records]).to('cuda')
        tokens, token_mask = processor._tokenize_prompts(states, [LANGUAGE]*count)
        batch = {'observation.images.base_0_rgb':frames[:, 0].float()/255,
                 'observation.images.left_wrist_0_rgb':frames[:, 1].float()/255,
                 OBS_LANGUAGE_TOKENS:tokens, OBS_LANGUAGE_ATTENTION_MASK:token_mask}
        images, masks = policy._preprocess_images(batch)
        assert len(images) == 3 and not masks[2].any() and all(x.all() for x in masks[:2])
    embeddings, padding, attention = policy.model.embed_prefix(images, masks, tokens, token_mask)
    assert not attention.any() and padding[:, :512].all()
    return ExecutionPolicyPrefix(embeddings, padding), tokens, token_mask


def capture_batch(policy, processor, tokenizer, records):
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
    from ember.operator_writer.native import _teacher_attention_kernel
    started = time.monotonic()
    with torch.no_grad():
        prefix, tokens, token_mask = prefix_batch(policy, processor, tokenizer, records)
        padding, length = prefix.padding, prefix.padding.shape[1]
        mask2d = make_att_2d_masks(padding, torch.zeros_like(padding))
        mask4d = policy.model._prepare_attention_masks_4d(mask2d)
        # Verify the full native block mask without creating/embedding action inputs.
        suffix_pad = torch.ones((len(records), 50), dtype=torch.bool, device='cuda')
        suffix_att = torch.zeros_like(suffix_pad); suffix_att[:, 0] = True
        native_mask = make_att_2d_masks(torch.cat((padding, suffix_pad), 1),
            torch.cat((torch.zeros_like(padding), suffix_att), 1))
        assert not native_mask[:, :length, length:].any()
        assert torch.equal(native_mask[:, :length, :length], mask2d)
        with capture_hooks(policy) as (pre, meta), _teacher_attention_kernel(mask4d):
            features, cache = prepare_prefix_features_and_cache(policy, prefix)
        assert len(pre) == 18 and torch.equal(meta['positions'], padding.cumsum(1)-1)
        assert torch.equal(meta['mask'] >= 0, mask2d[:, None])
        pre_keys = torch.stack([pre[i][:, :512] for i in range(18)], 1)
        assert len(cache.layers) == 18
        post_keys = torch.stack([cache.layers[i].keys[:, 0, :512] for i in range(18)], 1)
        assert pre_keys.shape == post_keys.shape == (len(records), 18, 512, 256)
        assert torch.isfinite(pre_keys).all() and torch.isfinite(post_keys).all()
        query_positions = padding.sum(1)[:, None] + torch.arange(50, device='cuda')[None]
        bridge = policy.model.paligemma_with_expert
        # Native joint teacher uses PaliGemma rotary; official self cached suffix uses expert rotary.
        rotary = (bridge.paligemma.model.language_model.rotary_emb if records[0]['source']['kind'] == 'teacher'
                  else bridge.gemma_expert.model.rotary_emb)
        cos, sin = rotary(prefix.embeddings[:, :50].float(), query_positions)
        for i, item in enumerate(records):
            path = ROOT/f'raw/init{item["init"]}_{item["source"]["kind"]}.pt'
            assert not path.exists(), 'fixed prefix already retained; do not duplicate'
            payload = {**item, 'k_pre':pre_keys[i].cpu(), 'k_post':post_keys[i].cpu(),
                'prefix_mask':padding[i].cpu(), 'attention_mask':(meta['mask'][i, 0]>=0).cpu(),
                'position_ids':meta['positions'][i].cpu(), 'prefix_valid_length':int(padding[i].sum()),
                'prefix_physical_length':length, 'query_positions':query_positions[i].cpu(),
                'query_cos':cos[i].cpu(), 'query_sin':sin[i].cpu(),
                'tokens':tokens[i].cpu(), 'token_mask':token_mask[i].cpu(),
                'camera_token_ranges':[[0,256],[256,512]],
                'third_camera_masked':item['source']['kind'] != 'teacher',
                'KV_heads':1, 'shared_query_heads':8, 'head_dim':256,
                'R_source':'native joint PaliGemma' if item['source']['kind']=='teacher' else 'official cached Action Expert',
                'R_precision':'official rotary module at float32 mathematical positions; no suffix forward',
                'prefix_consumer_only':True, 'suffix_forward_calls':0}
            torch.save(payload, path)
        del features, cache, pre, pre_keys, post_keys
    torch.cuda.synchronize()
    return dict(kind='teacher' if records[0]['source']['kind']=='teacher' else 'self',
        prefixes=len(records), physical_batch=len(records),
        seconds=time.monotonic()-started, actual_prefix_length=length,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
        packing='all registered inputs of this prefix shape; no additional samples/profile')


def half_turn(x):
    return np.concatenate((-x[..., 128:], x[..., :128]), axis=-1)


def covectors(raw):
    keys = raw['k_post'].float().numpy().astype(np.float64)
    weights = (raw['weights']['butter']-raw['weights']['orange_juice']).numpy()
    delta = np.einsum('lpd,p->ld', keys[:, :256], weights)
    cos, sin = raw['query_cos'].double().numpy(), raw['query_sin'].double().numpy()
    return (delta[:, None]*cos[None]-half_turn(delta[:, None])*sin[None])/16


def readback():
    arrays, rows = {}, []
    for init in range(8):
        raw = {kind:torch.load(ROOT/f'raw/init{init}_{kind}.pt', map_location='cpu', weights_only=False)
            for kind in ('teacher','original','swapped')}
        dt = covectors(raw['teacher']); b = dt.mean(1); norms = np.linalg.norm(b, axis=-1)
        u = np.divide(b, norms[:, None], out=np.full_like(b, np.nan), where=norms[:, None]!=0)
        arrays[f'init{init}_teacher_unit'] = u
        arrays[f'init{init}_teacher_covectors'] = dt
        arrays[f'init{init}_teacher_reference'] = norms
        teacher_dot = np.einsum('lid,ld->li', dt, u)
        assert np.allclose(teacher_dot.mean(1)[norms!=0], norms[norms!=0], rtol=1e-10, atol=1e-12)
        arrays[f'init{init}_teacher_query_cosine'] = np.divide(teacher_dot, np.linalg.norm(dt,axis=-1),
            out=np.full_like(teacher_dot,np.nan), where=np.linalg.norm(dt,axis=-1)!=0)
        wt = (raw['teacher']['weights']['butter']-raw['teacher']['weights']['orange_juice']).numpy()
        pre_t = np.einsum('lpd,p->ld',raw['teacher']['k_pre'][:, :256].float().numpy(),wt)
        for kind in ('original','swapped'):
            ds = covectors(raw[kind]); m = np.einsum('ld,lid->li',u,ds)
            cs = np.divide(m,np.linalg.norm(ds,axis=-1),out=np.full_like(m,np.nan),where=np.linalg.norm(ds,axis=-1)!=0)
            w = (raw[kind]['weights']['butter']-raw[kind]['weights']['orange_juice']).numpy()
            pre_s = np.einsum('lpd,p->ld',raw[kind]['k_pre'][:, :256].float().numpy(),w)
            denominator = np.linalg.norm(pre_t,axis=-1)*np.linalg.norm(pre_s,axis=-1)
            content = np.divide((pre_t*pre_s).sum(-1),denominator,out=np.full(18,np.nan),where=denominator!=0)
            arrays[f'init{init}_{kind}_m'] = m; arrays[f'init{init}_{kind}_unit_cosine'] = cs
            arrays[f'init{init}_{kind}_pre_RoPE_cosine'] = content
            for layer in range(18):
                rows.append(dict(init=init,teacher=TEACHERS[init],layout=kind,layer=layer,
                    teacher_reference=float(norms[layer]), teacher_zero=bool(norms[layer]==0),
                    m={str(n):dict(mean=float(m[layer,:n].mean()),min=float(m[layer,:n].min()),
                        max=float(m[layer,:n].max()),positive_slots=int((m[layer,:n]>0).sum())) for n in (5,50)},
                    unit_direction_cosine={str(n):float(cs[layer,:n].mean()) for n in (5,50)},
                    pre_RoPE_content_cosine=float(content[layer])))
        del raw
    np.savez_compressed(ROOT/'analysis/all_readout.npz',**arrays)
    write_json_atomic(ROOT/'readback.json',dict(complete=True,prefixes=24,rows=rows,
        m_shape=[8,2,18,50],all_pairs_layers_slots_retained=True,source=str(ROOT/'raw'),
        raw_arrays=str(ROOT/'analysis/all_readout.npz'),new_behavior=False,
        quantity='directional derivative of ROI weighted average logit contrast, not mass log-odds or action effect',
        selection_threshold_fit_sign_tuning=False,scientific_interpretation_owner='main'))
    return len(rows)


def main():
    # Public facade establishes the canonical import order; no evaluator is constructed.
    import ember.pi05_evaluation
    from ember.pi05_eval.worker_setup import load_policy
    from ember.operator_writer.run import frozen_git
    from ember.writer.topology import bind_current_process_to_cuda_numa
    identity = frozen_git()
    assert not (ROOT/'consumer_completion.json').exists() and not list((ROOT/'raw').glob('*.pt')), \
        'fixed24 already started/completed; do not duplicate model reads'
    torch.cuda.set_device(0); bind_current_process_to_cuda_numa(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    parent = read_json(PARENT_BANK); spec = read_json(Path(parent['spec']['path']))
    source = parent['source']; config = spec['source']
    stats = read_json(ASSET/config['normalization'])['stats']
    tokenizer_path = ASSET/config['tokenizer']
    recipe = read_json(ASSET/'configs/pi05_target_evaluation_v1.json')['policy']
    assert recipe['chunk_size'] == 50 and recipe['state_dim'] == 8
    records = inputs(); (ROOT/'raw').mkdir(exist_ok=True)
    write_json_atomic(ROOT/'run_contract.json',dict(study=ROOT.name,git=identity,source=source,
        normalization=str(ASSET/config['normalization']),tokenizer=str(tokenizer_path),
        exact_language=LANGUAGE,inputs=read_json(ROOT/'analysis/visible_input/sources.json'),
        registered_ROI=str(ROOT/'analysis/visible_input/roi_registration.json'),source_loading_count=1,
        teacher_fields='only first dual RGB',self_fields='saved after_rgb/eef_pos/quat/gripper_qpos',
        prefix_owner='ember.ecp.policy_effects.prepare_prefix_features_and_cache',suffix_calls=0,Writer_calls=0,
        precision='canonical prefix owner default BF16 autocast; frozen source recipe BF16/TF32',
        hard_GPUh=.5,hard_GiB=4,teacher_batch=8,self_batch=16))
    policy, processor, _ = load_policy(Path(source['model_path']),stats,tokenizer_path,recipe)
    policy.requires_grad_(False)
    tokenizer = Pi05TeacherPrefixTokenizer(tokenizer_path,200,'cuda:0')
    assert not any(x.requires_grad for x in policy.parameters())
    batches = []
    for teacher in (True,False):
        group = [x for x in records if (x['source']['kind']=='teacher') == teacher]
        batches.append(capture_batch(policy,processor,tokenizer,group))
        write_json_atomic(ROOT/'capture_batches.json',batches)
    del policy,processor,tokenizer
    torch.cuda.empty_cache()
    result_rows = readback()
    write_json_atomic(ROOT/'consumer_completion.json',dict(complete=True,prefixes=24,
        result_rows=result_rows,git=identity,source_loading_count=1,source_frozen=True,
        suffix_calls=0,Writer_calls=0,environment_steps=0,training=0,new_behavior=False,batches=batches))
    print(json.dumps(dict(event='fixed24_complete',prefixes=24,rows=result_rows)),flush=True)
