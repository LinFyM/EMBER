"""Temporary §134 passive reading of exactly101 sealed first replans."""
from __future__ import annotations

from contextlib import contextmanager
import gc
from pathlib import Path
import re
import time

import torch
from safetensors.torch import load_file

import ember.pi05_evaluation  # canonical import owner before evaluation leaves
from ember.batched_lora import BatchedLoRAInference
from ember.lora import LORA_A_SUFFIX, inject_task_lora, validate_lora_state
from ember.pi05_eval.worker_setup import load_policy, validate_worker_assets
from ember.pi05_eval_contract import git_state, git_state_is_clean_pushed_or_frozen_authority
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.topology import bind_current_process_to_cuda_numa
from .self_call_labels import ROOT, ASSET, initial


def q_layer(name):
    match = re.search(r'gemma_expert\.model\.layers\.(\d+)\.self_attn\.q_proj$', name)
    return int(match[1]) if match else None


class SelfCallObserver:
    """Measure actual Q/K; call the original attention unchanged."""

    def __init__(self, policy, f, q, factors, residual, r_indices):
        layers = policy.model.paligemma_with_expert.gemma_expert.model.layers
        if len(layers) != 18 or any(l.self_attn.num_key_value_groups != 8 for l in layers):
            raise ValueError('actual18-layer/eight-head single-KV GQA changed')
        self.layers = {id(layer.self_attn): i for i, layer in enumerate(layers)}
        self.f, self.q = f.float(), q.float()
        self.a_factors, self.residual, self.r_indices = factors, residual, r_indices
        self.names = {q_layer(n): n for n in factors if q_layer(n) is not None}
        self.inputs, self.positions, self.steps = {}, {}, []

    def attention(self, module, query, key, value, mask, scaling, dropout=0., **kwargs):
        layer = self.layers.get(id(module))
        if layer is not None:
            if query.shape[1:] != (8, 50, 256) or key.shape[1] != 1 or scaling != 1 / 16:
                raise ValueError('actual own action query/key/scaling topology changed')
            # All statistics are FP32; the original SDPA below keeps its own dtype.
            with torch.autocast('cuda', enabled=False):
                logits = (query[:, :, :5].float() @ key.float().transpose(-1, -2)) * scaling
                if mask is not None:
                    logits = logits + mask[:, :, :5].float()
                image = logits[..., :512]
                logq = self.q.clamp_min(1e-30).log()[:, None, None]
                scores = torch.logsumexp(image[..., None, :] + logq, -1)
                pi = logits.softmax(-1)
                mass = pi[..., :512].sum(-1)
                mu = torch.einsum('bhip,bep->bhie', pi[..., :512], self.f)
                record = dict(scores=scores.cpu(), image_mass=mass.cpu(), entity_mass=mu.cpu())
                if self.r_indices.numel():
                    record.update(self.direct(layer, key[..., :512, :], image, scores))
                self.current[layer] = record
        return self.base(module, query, key, value, mask, scaling, dropout=dropout, **kwargs)

    def direct(self, layer, key, image, scores):
        from transformers.models.gemma import modeling_gemma
        ids = self.r_indices
        h = self.inputs[layer][ids, :5].float()
        factor = self.a_factors[self.names[layer]][ids].float()
        a = torch.bmm(h, factor.transpose(1, 2))
        r = self.residual[str(layer)].float().reshape(len(ids), 8, 256, 128)
        q = self.q[ids]
        key = key[ids, 0].float()
        contrast = q[:, :1] - q
        mean_key = torch.einsum('bep,bpd->bed', contrast, key)
        cos, sin = self.positions['cos'][ids, :5].float(), self.positions['sin'][ids, :5].float()
        rotated_mean = mean_key[:, None] * cos[:, :, None] - \
                       modeling_gemma.rotate_half(mean_key[:, None]) * sin[:, :, None]
        d = rotated_mean / 16
        w = torch.einsum('bhdr,bied->bhier', r, d)
        m = torch.einsum('bhier,bir->bhie', w, a)
        delta = torch.einsum('bhdr,bir->bhid', r, a)
        delta = delta * cos[:, None] + modeling_gemma.rotate_half(delta) * sin[:, None]
        effect = (delta @ key[:, None].transpose(-1, -2)) / 16
        logq = q.clamp_min(1e-30).log()[:, None, None]
        density_without = torch.logsumexp((image[ids] - effect)[..., None, :] + logq, -1)
        return dict(a=a.cpu(), w=w.cpu(), m=m.cpu(),
            direct_R_local_density=(scores[ids] - density_without).cpu(),
            direct_R_entity_mean=torch.einsum('bhip,bep->bhie', effect, q).cpu(),
            direct_R_image_mean=effect.mean(-1).cpu(), direct_R_image_RMS=effect.square().mean(-1).sqrt().cpu(),
            actual_query_cos=cos.cpu(), actual_query_sin=sin.cpu())

    @contextmanager
    def capture(self, policy):
        from transformers.models.gemma import modeling_gemma
        original, rotate = policy.model.denoise_step, modeling_gemma.apply_rotary_pos_emb
        self.base = modeling_gemma.eager_attention_forward
        handles = []
        for layer, name in self.names.items():
            def capture_input(module, args, layer=layer):
                self.inputs[layer] = args[0]
            handles.append(policy.get_submodule(name).register_forward_pre_hook(capture_input))
        def rotated(query, key, cos, sin, unsqueeze_dim=1):
            if query.shape[-2] == 50 and query.shape[1] == 8:
                self.positions['cos'], self.positions['sin'] = cos, sin
            return rotate(query, key, cos, sin, unsqueeze_dim=unsqueeze_dim)
        def denoise(*args, **kwargs):
            self.current = {}; self.inputs.clear(); self.positions.clear()
            result = original(*args, **kwargs)
            if set(self.current) != set(range(18)):
                raise ValueError('same official flow missed a real layer')
            self.steps.append({name: torch.stack([self.current[i][name] for i in range(18)], 1)
                               for name in self.current[0]})
            self.inputs.clear(); self.current.clear(); self.positions.clear()
            return result
        policy.model.denoise_step = denoise
        modeling_gemma.apply_rotary_pos_emb = rotated
        modeling_gemma.eager_attention_forward = self.attention
        try:
            yield self
        finally:
            policy.model.denoise_step = original
            modeling_gemma.apply_rotary_pos_emb = rotate
            modeling_gemma.eager_attention_forward = self.base
            for handle in handles:
                handle.remove()
            self.inputs.clear(); self.positions.clear()


def pack(rows, labels, device):
    observations, old_actions, seeds, states = [], [], [], []
    count = max(len(labels[r['scene_key']]['entities']) for r in rows)
    f, q = torch.zeros(len(rows), count, 512), torch.zeros(len(rows), count, 512)
    for i, row in enumerate(rows):
        obs, state, old, seed = initial(row)
        if seed != row['noise_seed']:
            raise ValueError('actual recorded Gaussian seed differs')
        observations.append(obs); states.append(state); old_actions.append(old); seeds.append(seed)
        import numpy as np
        data = np.load(labels[row['scene_key']]['path'])
        f[i, :len(data['f'])], q[i, :len(data['q'])] = torch.from_numpy(data['f']), torch.from_numpy(data['q'])
    batch = {k: torch.cat([o[k] for o in observations]).to(device) for k in observations[0]}
    noise = torch.stack([torch.randn((50, 32), generator=torch.Generator(device='cpu').manual_seed(seed)) for seed in seeds]).to(device)
    return batch, noise, f.to(device), q.to(device), torch.cat(old_actions), states


def factors_for(rows, lora, device):
    unique = {r['factors'] for r in rows}
    states = {p: load_file(p, device='cpu') for p in unique}
    for state in states.values():
        validate_lora_state(state, lora)
    ordered = [states[r['factors']] for r in rows]
    a = {t.name: torch.stack([s[t.name + LORA_A_SUFFIX] for s in ordered]).to(device)
         for t in lora.targets if q_layer(t.name) is not None}
    r_ids = torch.tensor([i for i, row in enumerate(rows) if row['arm'] == 'R'], device=device, dtype=torch.long)
    residual = {}
    if r_ids.numel():
        bindings = {p: torch.load(p, map_location='cpu', weights_only=True)['R']
                    for p in {row['binding'] for row in rows if row['arm'] == 'R'}}
        if any(set(v) != {str(i) for i in range(18)} for v in bindings.values()):
            raise ValueError('R must contain all eighteen actual matrices')
        for i in range(18):
            residual[str(i)] = torch.stack([bindings[rows[j]['binding']][str(i)] for j in r_ids.tolist()]).to(device)
            if residual[str(i)].shape[1:] != (2048, 128) or not torch.isfinite(residual[str(i)]).all():
                raise ValueError('actual sealed R shape/finite changed')
    return ordered, a, residual, r_ids


@torch.no_grad()
def consume(policy, lora, selected, labels, batch_number):
    device = torch.device('cuda:0')
    batch, noise, f, q, original, raw_states = pack(selected, labels, device)
    states, a, residual, ids = factors_for(selected, lora, device)
    observer = SelfCallObserver(policy, f, q, a, residual, ids)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.monotonic()
    batched = BatchedLoRAInference(policy, lora)
    try:
        with batched.activate(states), observer.capture(policy), torch.autocast('cuda', dtype=torch.bfloat16):
            generated = policy.predict_action_chunk(batch, noise=noise, num_steps=10)
    finally:
        batched.close()
    torch.cuda.synchronize(); seconds = time.monotonic() - started
    if len(observer.steps) != 10 or generated.shape != (len(selected), 50, 7) or not torch.isfinite(generated).all():
        raise ValueError('actual once-only official ten-flow consumer incomplete')
    raw = {k: torch.stack([s[k] for s in observer.steps], 1) for k in observer.steps[0]}
    r_index = {j: i for i, j in enumerate(ids.tolist())}
    records = []
    for i, row in enumerate(selected):
        label = labels[row['scene_key']]; entities = len(label['entities'])
        result = dict(schema='native_role_self_call_raw_v1', row=row, entities=label['entities'],
            candidates=label['candidates'], target_index=0, valid_rho=label['valid_rho'],
            action_generated_normalized=generated[i].float().cpu(), original_first_chunk=original[i],
            original_state8=raw_states[i], noise_seed=row['noise_seed'], first_replan=0,
            tau=torch.arange(10, 0, -1).float() / 10, physical_batch=len(selected),
            role_scores=raw['scores'][i, ..., :entities], image_mass=raw['image_mass'][i],
            entity_mass=raw['entity_mass'][i, ..., :entities],
            entity_mass_fraction=raw['entity_mass'][i, ..., :entities] / raw['image_mass'][i, ..., None].clamp_min(1e-30),
            f=f[i, :entities].cpu(), q=q[i, :entities].cpu(),
            extra_preprocessing=False, additional_forward=False, environment_steps=0,
            hook_scope='measurement only; real SDPA/value/prefix/mask/GQA/flow unchanged')
        if i in r_index:
            j = r_index[i]
            result.update({k: raw[k][j] for k in ('a', 'direct_R_image_mean', 'direct_R_image_RMS', 'actual_query_cos', 'actual_query_sin')})
            result.update({k: raw[k][j, ..., :entities, :] for k in ('w',)})
            result.update({k: raw[k][j, ..., :entities] for k in ('m', 'direct_R_local_density', 'direct_R_entity_mean')})
        path = ROOT / 'predictions' / (row['case_id'] + '.pt')
        if path.exists():
            raise ValueError('a successful legal query may not be rerun')
        torch.save(result, path)
        records.append(dict(case_id=row['case_id'], path=str(path), bytes=path.stat().st_size))
    timing = dict(batch_number=batch_number, actual_inputs=len(selected), actual_R_inputs=len(ids), seconds=seconds,
        queries_per_second=len(selected) / seconds, peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
        peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
        source_resident_GiB=sum(p.numel() * p.element_size() for p in policy.parameters()) / 2**30,
        no_Writer_native_or_unused_K_cache=True, hooks_restored=True)
    del raw, observer, batch, states, a, residual, generated
    gc.collect(); torch.cuda.empty_cache()
    return records, timing


def main():
    identity = git_state(Path(__file__).resolve().parents[3])
    if identity['branch'] or not git_state_is_clean_pushed_or_frozen_authority(identity):
        raise ValueError('actual consumers require clean pushed detached code')
    if (ROOT / 'consumer_completion.json').exists() or list((ROOT / 'predictions').glob('*.pt')):
        raise ValueError('prior successful inputs exist; no duplicate launch')
    torch.cuda.set_device(0); torch.set_num_threads(8)
    affinity = bind_current_process_to_cuda_numa(0)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    manifest = read_json(ROOT / 'inputs.json'); rows = manifest['rows']
    labels = {r['scene_key']: r for r in read_json(ROOT / 'labels/manifest.json')['records']}
    original_contract = read_json(Path(rows[0]['contract']))
    # New frozen readout consumes only the immutable asset contract. It neither
    # reopens the retired closed-loop launcher nor changes its guard.
    contract = {key: original_contract[key] for key in ('model', 'normalization', 'tokenizer', 'policy')}
    contract['native_role_self_call'] = dict(study=ROOT.name, inputs=101, environment_steps=0)
    write_json_atomic(ROOT / 'launch/asset_contract.json', contract)
    model, stats, tokenizer = validate_worker_assets(contract)
    policy, _, _ = load_policy(model, stats, tokenizer, contract['policy'])
    lora = load_pi05_lora_contract(ASSET / 'configs/pi05_lora_rank128_aligned.json')
    if lora.alpha != lora.rank or lora.rank != 128 or len(lora.targets) != 38:
        raise ValueError('actual complete38/rank128/scale1 contract changed')
    inject_task_lora(policy, lora); policy.requires_grad_(False); policy.eval()
    train = [r for r in rows if r['panel'] == 'train']; held = [r for r in rows if r['panel'] != 'train']
    records, timings = [], []
    first, timing = consume(policy, lora, train, labels, 0)
    records.extend(first); timings.append(timing)
    # The first32 legal queries supply real peak/throughput. Expand the remaining
    # batch only when the measured peak predicts true safe live headroom.
    free, total = torch.cuda.mem_get_info()
    fixed = timing['source_resident_GiB']; variable = max(timing['peak_reserved_GiB'] - fixed, .01)
    projected = fixed + variable * len(held) / len(train)
    held_batch = len(held) if projected + 4 < total / 2**30 else min(48, len(held))
    selection = dict(first_actual_batch=32, remaining_inputs=69, selected_held_batch=held_batch,
        projected_69_peak_GiB=projected, live_free_GiB=free / 2**30, total_GiB=total / 2**30,
        reason='expand actual remaining inputs using measured first32 peak; no consumed-query reprofiling',
        scientific_inputs_added=0, duplicate_successful_reads=0)
    write_json_atomic(ROOT / 'launch/packing_selection.json', selection)
    write_json_atomic(ROOT / 'predictions/manifest.json', dict(records=records, timings=timings, complete=False))
    for offset in range(0, len(held), held_batch):
        current, timing = consume(policy, lora, held[offset:offset + held_batch], labels, len(timings))
        records.extend(current); timings.append(timing)
        write_json_atomic(ROOT / 'predictions/manifest.json', dict(records=records, timings=timings, complete=False))
    if len(records) != 101:
        raise ValueError('fixed101 first queries incomplete')
    write_json_atomic(ROOT / 'predictions/manifest.json', dict(records=records, timings=timings, complete=True))
    write_json_atomic(ROOT / 'consumer_completion.json', dict(complete=True, inputs=101, unique_states=25,
        real_flow_calls=1010, new_environment_steps=0, new_closed_loop_rows=0, new_native_or_Writer_calls=0,
        source_loads=1, reading_git=identity, timings=timings, affinity=affinity, packing=selection,
        measurement_dtype='FP32; native policy BF16/TF32', all_hooks_restored=True))
    print({'complete': True, 'queries': 101, 'timings': timings}, flush=True)


if __name__ == '__main__':
    main()
