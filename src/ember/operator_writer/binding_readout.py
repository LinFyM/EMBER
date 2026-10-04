"""Once-only B20 ten-step actions and same-forward Q/role/direct-R readback."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from ember.batched_lora import BatchedLoRAInference

from ember.lora import LORA_A_SUFFIX, copy_task_lora_state_
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record
from ember.writer.runtime import autocast
from .binding_run import ROOT, ASSET, TASKS, HELD_TEACHERS, initialize, load_native, load_labels
from .role_binding import RoleBinding, RoleObserver, compile_fixed, q_layer
from .binding_credit import raw_batch, samples, labels_for
from .binding_evaluation import HELD_REFERENCE, TRAIN_REFERENCE, evaluate
from .data import FormalData


@contextmanager
def q_readback(policy, q, factors, residual):
    from transformers.models.gemma import modeling_gemma
    original, rotate = policy.model.denoise_step, modeling_gemma.apply_rotary_pos_emb
    records, inputs, positions = [], {}, {}
    handles = []
    names = {q_layer(n.removesuffix(LORA_A_SUFFIX)): n.removesuffix(LORA_A_SUFFIX)
             for n in factors if n.endswith(LORA_A_SUFFIX) and q_layer(n.removesuffix(LORA_A_SUFFIX)) is not None}
    for layer, name in names.items():
        def capture(module, args, layer=layer):
            inputs[layer] = args[0]
        handles.append(policy.get_submodule(name).register_forward_pre_hook(capture))
    def rotated(query, key, cos, sin, unsqueeze_dim=1):
        if query.shape[-2] == 50 and query.shape[1] == 8:
            positions['cos'], positions['sin'] = cos, sin
        return rotate(query, key, cos, sin, unsqueeze_dim=unsqueeze_dim)
    def denoise(*args, **kwargs):
        observer = RoleObserver(policy, q)
        local_logits, local_density = {}, {}
        native_attention = observer.attention
        def attention(module, query, key, value, mask, scaling, dropout=0., **extras):
            layer = observer.layers.get(id(module))
            output = native_attention(module, query, key, value, mask, scaling, dropout=dropout, **extras)
            if residual is not None and layer is not None:
                name = names[layer]; h = inputs[layer].float()
                a = factors[name + LORA_A_SUFFIX].float()
                r = residual[str(layer)].to(device=h.device, dtype=torch.float32)
                delta = torch.bmm(torch.bmm(h, a.transpose(1, 2)), r.transpose(1, 2))
                delta = delta.reshape(len(h), 50, 8, 256).transpose(1, 2)
                cos, sin = positions['cos'][:, None], positions['sin'][:, None]
                delta = delta * cos + modeling_gemma.rotate_half(delta) * sin
                effect = (delta[:, :, :5] @ key[..., :512, :].float().transpose(-1, -2)) * scaling
                original_logits = (query[:, :, :5].float() @ key[..., :512, :].float().transpose(-1, -2)) * scaling
                logq = q.float().clamp_min(1e-30).log()[:, None, None]
                baseline_density = torch.logsumexp((original_logits - effect)[..., None, :] + logq, -1)
                local_logits[layer] = effect.detach().cpu()
                local_density[layer] = (observer.scores[layer] - baseline_density).detach().cpu()
            return output
        observer.attention = attention
        with observer.capture():
            value = original(*args, **kwargs)
        if set(observer.scores) != set(range(18)):
            raise ValueError('same official ten-flow role capture incomplete')
        records.append(dict(scores=torch.stack([observer.scores[i].detach().cpu() for i in range(18)], 1),
                            image_mass=torch.stack([observer.mass[i].detach().cpu() for i in range(18)], 1),
                            direct_R_image_logits=(torch.stack([local_logits[i] for i in range(18)], 1) if residual is not None else None),
                            direct_R_local_density=(torch.stack([local_density[i] for i in range(18)], 1) if residual is not None else None)))
        return value
    modeling_gemma.apply_rotary_pos_emb = rotated
    policy.model.denoise_step = denoise
    try:
        yield records
    finally:
        policy.model.denoise_step = original
        modeling_gemma.apply_rotary_pos_emb = rotate
        for handle in handles:
            handle.remove()


@torch.no_grad()
def materialize(runtime, spec, arm, identity):
    out = ROOT / 'banks' / arm; out.mkdir(exist_ok=False)
    binding = None
    if arm != 'parent':
        runtime.writer.load_state_dict(load_file(str(ROOT / arm / 'checkpoint64/writer.safetensors'), device='cuda:0'), strict=True)
    if arm == 'R':
        binding = RoleBinding(runtime.device)
        binding.load_state_dict(load_file(str(ROOT / 'R/checkpoint64/binding.safetensors'), device='cuda:0'), strict=True)
    original = read_json(HELD_REFERENCE)['adapter']
    bank = deepcopy(original)
    bank.update(conditions=[], tasks=[], native_role_binding_compilation=dict(study=ROOT.name, arm=arm),
                training_git='85919994aef11c17b49b7d0e70a2c110158bff61' if arm == 'parent' else read_json(ROOT / arm / 'training_contract.json')['git']['commit'],
                materialization_git=identity['commit'], native_reading=file_record(ROOT / 'native/manifest.json'),
                condition_factors='complete_A0_plus_S_B0_plus_M')
    if arm != 'parent':
        bank['checkpoint'] = str(ROOT / arm / 'checkpoint64')
        bank['checkpoint_manifest'] = file_record(ROOT / arm / 'checkpoint64/checkpoint_manifest.json')
    for tasks, role in ((TASKS, 'train'), *((((16,), 'validation'),) if arm != 'parent' else ())):
        data = FormalData(ASSET, spec, query_labels=False, task_ids=tasks, role=role)
        try:
            for task in tasks:
                meta = data.tasks[task]
                teachers = (0, 1) if task != 16 else HELD_TEACHERS
                bank['tasks'].append(dict(global_task_id=task, suite=meta.suite, task_id=meta.suite_task_id,
                    language=meta.authority.language, split_role=role,
                    episodes=[dict(init_state_id=i, condition_id=f'task{task:03d}_teacher00',
                                   teacher_demo_indices=[0], video_ordinal=None) for i in (32, 33)] if task != 16 else []))
                for teacher in teachers:
                    path = ROOT / 'native' / f'task{task:03d}_teacher{teacher:02d}.pt'
                    fixed = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
                    from .binding_run import fixed_to_gpu
                    fixed = fixed_to_gpu(fixed, runtime.device)
                    factors, residual = compile_fixed(runtime, fixed, binding)
                    key = f'task{task:03d}_teacher{teacher:02d}'
                    factor_path = out / (key + '.safetensors')
                    save_file({k: v.cpu().contiguous() for k, v in factors.items()}, str(factor_path))
                    if residual is not None:
                        torch.save(residual, out / (key + '_binding.pt'))
                    bank['conditions'].append(dict(condition_id=key, global_task_id=task, teacher_demo=teacher,
                        factors=file_record(factor_path), sampled_frames=len(fixed['H']),
                        native_reading=file_record(path), full38=True))
                    del fixed, factors, residual
        finally:
            data.close()
    bank['manifest'] = dict(path=str(out / 'manifest.json'))
    write_json_atomic(out / 'manifest.json', bank)
    return bank


@torch.no_grad()
def readout(runtime, spec, arm, bank):
    data = FormalData(ASSET, spec, query_labels=True, task_ids=TASKS)
    labels = load_labels(); manifest = read_json(ROOT / 'query_manifest.json')
    records, pair_timings = [], []
    runtime.restore_identity()
    batched = BatchedLoRAInference(runtime.policy, runtime.lora)
    torch.cuda.reset_peak_memory_stats()
    label_records = {(r['task'], r['demo']): r for r in read_json(ROOT / 'labels/manifest.json')['records']}
    try:
        for task in TASKS:
            queries = manifest['B20'][str(task)]
            raw = raw_batch(data, task, queries); batch = runtime.processor.training_batch(raw)
            q = labels_for(labels, task, queries, runtime.device).repeat(2, 1, 1)
            conditions, factor_states, residuals = [], [], []
            for teacher in (0, 1):
                key = f'task{task:03d}_teacher{teacher:02d}'
                condition = next(r for r in bank['conditions'] if r['condition_id'] == key)
                factors = load_file(condition['factors']['path'], device='cuda:0')
                residual = (torch.load(ROOT / 'banks' / arm / (key + '_binding.pt'), map_location='cpu',
                                       weights_only=True)['R'] if arm == 'R' else None)
                conditions.append(condition); factor_states.append(factors); residuals.append(residual)
            packed = {k: torch.cat((v, v)) if isinstance(v, torch.Tensor) and v.ndim and len(v) == 20
                      else v * 2 if isinstance(v, (list, tuple)) and len(v) == 20 else v
                      for k, v in batch.items() if k != 'action'}
            # Exactly the existing two conditions × B20, ordered teacher0 then1;
            # physical batching changes neither data, noise nor ten-flow scope.
            local_a = {name: torch.cat([state[name][None].expand(20, *state[name].shape) for state in factor_states])
                       for name in factor_states[0] if name.endswith(LORA_A_SUFFIX)}
            local_r = ({layer: torch.cat([value[layer][None].expand(20, *value[layer].shape) for value in residuals])
                        for layer in residuals[0]} if arm == 'R' else None)
            with autocast(runtime.device):
                noise20 = samples(runtime, batch, queries).arguments[5]
                noise = torch.cat((noise20, noise20))
                torch.cuda.synchronize(); started = time.monotonic()
                with batched.activate([factor_states[0]] * 20 + [factor_states[1]] * 20), \
                     q_readback(runtime.policy, q, local_a, local_r) as role:
                    generated = runtime.policy.predict_action_chunk(packed, noise=noise, num_steps=10)
            torch.cuda.synchronize(); elapsed = time.monotonic() - started
            pair_timings.append(dict(task=task, actual_queries=40, seconds=elapsed,
                                     queries_per_second=40 / elapsed))
            if len(role) != 10 or len(generated) != 40:
                raise ValueError('B20 paired packing must use40 queries and ten actual official calls')
            for teacher, condition in enumerate(conditions):
                key = condition['condition_id']; selected = slice(teacher * 20, (teacher + 1) * 20)
                row = dict(schema='native_role_binding_B20_v1', arm=arm, task=task, teacher=teacher,
                    queries=queries, query_offset=1, official_flow_steps=10,
                    action_generated_normalized=generated[selected].float().cpu(), action_target=batch['action'].cpu(),
                    valid_mask=~raw['action_is_pad'], noise=noise[selected].cpu(), role_patch_weights=q[selected].cpu(),
                    role_scores=torch.stack([r['scores'][selected] for r in role], 1),
                    image_attention_mass=torch.stack([r['image_mass'][selected] for r in role], 1),
                    target_entity_index=0, entities=label_records[task, queries[0]['demo']]['entities'],
                    extra_forward=False, Writer_removed=True, physical_query_batch=40)
                if arm == 'R':
                    # Keep compact same-h effects; full logits reduce over image
                    # tokens only after recording each layer/head/slot contribution.
                    effect = torch.stack([r['direct_R_image_logits'][selected] for r in role], 1)
                    row.update(direct_R_image_logit_mean=effect.mean(-1), direct_R_image_logit_RMS=effect.square().mean(-1).sqrt(),
                        direct_R_entity_logit_mean=torch.einsum('btlhsp,bop->btlhso', effect, q[selected].cpu()),
                        direct_R_local_density_difference=torch.stack([r['direct_R_local_density'][selected] for r in role], 1),
                        direct_R_scope='same actual h/own K; local subtraction only, not a removed-R policy')
                path = ROOT / 'predictions' / arm / (key + '_B20.pt'); path.parent.mkdir(exist_ok=True)
                torch.save(row, path)
                records.append(dict(task=task, teacher=teacher, predictions=file_record(path), queries=20))
    finally:
        batched.close(); data.close(); runtime.restore_identity()
    write_json_atomic(ROOT / 'predictions' / arm / 'manifest.json', dict(records=records, queries=320, forward_steps=10,
        additional_FM_forward=0, physical_query_batch=40, paired_teachers=2, no_extra_queries=True,
        actual_pair_timings=pair_timings, peak_reserved_GiB=torch.cuda.max_memory_reserved() / 2**30,
        larger_same_task_batch_unavailable='all20 authorized queries ×both teachers already packed; no additional query/profile'))


def main(arm):
    if (ROOT / 'evaluation' / arm / 'consumer_completion.json').exists():
        raise ValueError('this endpoint has already finished; do not duplicate it')
    runtime, spec, identity, affinity = initialize()
    torch.set_grad_enabled(False)
    bank = materialize(runtime, spec, arm, identity)
    # The deployed policy consumes only the fixed complete factor banks.
    del runtime.writer
    if arm != 'parent':
        readout(runtime, spec, arm, bank)
    evaluate(runtime, bank, arm, identity)
    write_json_atomic(ROOT / 'evaluation' / arm / 'consumer_completion.json',
        dict(status='complete', arm=arm, reading_git=identity, Writer_removed=True,
             rows=32 if arm == 'parent' else 55, full=8 if arm == 'parent' else 31))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('parent', 'F', 'G', 'R'), required=True)
    main(parser.parse_args().arm)
