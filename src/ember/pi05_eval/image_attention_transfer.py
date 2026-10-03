"""Task-scoped same-input image attention intervention; retired after readback."""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import torch

from ember.lora import copy_task_lora_state_
from ember.operator_writer.bank import FrozenOperatorAdapter, PreparedOperatorLoRA, episode_evidence
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_source_checkpoint import write_json_atomic
from ember.writer.materialization import file_record
from .image_attention_contract import ROOT, TAG, inspected_bank, registration, route

METRICS = ('recipient_image_mass', 'donor_image_mass', 'conditional_pi_L1',
           'actual_value_delta_RMS', 'original_attention_output_RMS',
           'image_mass_error_fp32', 'non_image_probability_error', 'total_probability_error_fp32')


def image_probabilities(logits, image_mask):
    """Stable conditional image probabilities, including arbitrarily small mass."""
    selected = logits[..., :image_mask.shape[-1]].masked_fill(~image_mask[:, None, None, :], -torch.inf)
    return torch.softmax(selected, dim=-1, dtype=torch.float32)


def replace_image_distribution(probabilities, donor_pi, image_mask):
    length = image_mask.shape[-1]
    visible = image_mask[:, None, None, :]
    mass = probabilities[..., :length].masked_fill(~visible, 0).sum(-1, keepdim=True)
    result = probabilities.clone()
    result[..., :length] = torch.where(visible, mass * donor_pi, probabilities[..., :length])
    return result, mass


class AttentionTransferAdapter(FrozenOperatorAdapter):
    """One resident source and the original complete bank factor loader."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.banks = {name: inspected_bank(name) for name in ('C', 'N')}
        self.caches = {name: OrderedDict() for name in ('C', 'N')}
        self.arm = kwargs['evaluation_adapter']['self_image_attention_transfer']['arm']
        self.output = ROOT / 'evaluation' / self.arm
        self.totals, self.calls = {}, {}
        self.phase, self.flow_step, self.checking = None, 0, False
        self.donor_pi, self.donor_mass = {}, {}
        self.layout = None
        self.packing = {}
        layers = self.policy.model.paligemma_with_expert.gemma_expert.model.layers
        if len(layers) != 18 or any(layer.self_attn.num_key_value_groups != 8 for layer in layers):
            raise Pi05EvaluationError('native eighteen-layer GQA topology changed')
        self.layer_ids = {id(layer.self_attn): i for i, layer in enumerate(layers)}

    def select_panel(self, contract):
        self.arm = contract['self_image_attention_transfer']['arm']
        self.output = Path(contract['output_dir'])
        self.bank = contract['adapter']

    def _state_for(self, name, key):
        saved = self.bank, self.conditions, self.states
        bank = self.banks[name]
        self.bank, self.conditions, self.states = bank, {r['condition_id']: r for r in bank['conditions']}, self.caches[name]
        try:
            return super()._state(key)
        finally:
            self.bank, self.conditions, self.states = saved

    def prepare_episode(self, *, suite, task_id, init_state_id):
        task = self.tasks[suite, task_id]
        episode = next(e for e in task['episodes'] if e['init_state_id'] == init_state_id)
        recipient, _ = route(self.arm)
        return PreparedOperatorLoRA(episode['condition_id'], episode_evidence(self.banks[recipient], task, episode))

    def _prefix(self, original, images, img_masks, tokens, masks):
        expert = self.policy.model.paligemma_with_expert
        original_image = expert.embed_image
        lengths = []

        def record_image(image):
            result = original_image(image)
            lengths.append(result.shape[1])
            return result

        expert.embed_image = record_image
        try:
            result = original(images, img_masks, tokens, masks)
        finally:
            expert.embed_image = original_image
        if len(lengths) != len(img_masks) or len(lengths) != 3:
            raise Pi05EvaluationError('actual image token blocks or padded camera changed')
        self.image_mask = torch.cat([mask[:, None].expand(-1, n) for mask, n in zip(img_masks, lengths, strict=True)], 1).bool()
        camera_visible = torch.stack(img_masks, dim=1).bool()
        if (not torch.all(camera_visible[:, :2]) or torch.any(camera_visible[:, 2])
                or not torch.equal(self.image_mask, result[1][:, :sum(lengths)])):
            raise Pi05EvaluationError('two visible camera masks disagree with actual prefix')
        self.layout = {'image_block_lengths': lengths, 'camera_visible': camera_visible[0].tolist(),
                       'visible_tokens': int(self.image_mask[0].sum()), 'prefix_tokens': result[0].shape[1],
                       'language_tokens': tokens.shape[1], 'image_positions': 'actual_embed_image_outputs_and_img_masks'}
        return result

    def _attention(self, module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        layer = self.layer_ids.get(id(module))
        if layer is None or self.phase is None:
            return self.base_attention(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
        if module.training or dropout or query.shape[-2] != 50:
            raise Pi05EvaluationError('native evaluation slots or dropout changed')
        self.layers_seen.add(layer)
        key = self.repeat_kv(key, module.num_key_value_groups)
        value = self.repeat_kv(value, module.num_key_value_groups)
        logits = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            logits = logits + attention_mask
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        native_probs = probs.to(query.dtype)
        output = torch.matmul(native_probs, value)
        pi = image_probabilities(logits, self.image_mask)
        n = self.image_mask.shape[-1]
        visible = self.image_mask[:, None, None, :]
        mass = probs[..., :n].masked_fill(~visible, 0).sum(-1, keepdim=True)
        if self.phase == 'donor':
            self.donor_pi[layer], self.donor_mass[layer] = pi, mass
        else:
            mixed, mass = replace_image_distribution(probs, self.donor_pi[layer], self.image_mask)
            native_probs = mixed.to(query.dtype)
            changed = torch.matmul(native_probs, value)
            if not self.checking:
                value_delta = (changed.float() - output.float()).square().mean((1, 2, 3)).sqrt()
                non_image_diff = (mixed - probs).masked_fill(
                    torch.nn.functional.pad(visible, (0, probs.shape[-1] - n)), 0).abs().amax((1, 2, 3))
                mixed_mass = mixed[..., :n].masked_fill(~visible, 0).sum(-1, keepdim=True)
                stats = torch.stack((mass.mean((1, 2, 3)), self.donor_mass[layer].mean((1, 2, 3)),
                                     (self.donor_pi[layer] - pi).abs().sum(-1).mean((1, 2)),
                                     value_delta, output.float().square().mean((1, 2, 3)).sqrt(),
                                     (mixed_mass - mass).abs().amax((1, 2, 3)), non_image_diff,
                                     (mixed.sum(-1) - probs.sum(-1)).abs().amax((1, 2))), -1)
                self.current_stats[:, layer, self.flow_step] += stats
            output = changed
        return output.transpose(1, 2).contiguous(), native_probs

    def _forward(self, original, packed, phase, arguments):
        self.phase = phase
        self.layers_seen = set()
        with self.batched.activate_packed(packed):
            result = original(**arguments)
        # Every layer's donor conditional pi is needed, even if its image mass is tiny.
        if set(self.donor_pi) != set(range(18)) or self.layers_seen != set(range(18)):
            raise Pi05EvaluationError('donor did not cover all eighteen action layers')
        return result

    def _self_check(self, original, packed, arguments):
        from transformers.cache_utils import DynamicCache

        claim = ROOT / 'launch/interface_self_donor_claim'
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return
        os.close(fd)
        sliced = dict(arguments)
        for name in ('prefix_pad_masks', 'x_t', 'timestep'):
            sliced[name] = arguments[name][:1]
        sliced['past_key_values'] = DynamicCache(tuple(
            (k[:1], v[:1], sliding) for k, v, sliding in arguments['past_key_values']))
        single = {k: v[:1] for k, v in packed.items()}
        image_mask = self.image_mask
        self.image_mask, self.checking = image_mask[:1], True
        try:
            self.donor_pi, self.donor_mass = {}, {}
            before = self._forward(original, single, 'donor', sliced)
            after = self._forward(original, single, 'recipient', sliced)
            delta = (after - before).float()
            rms, maximum = float(delta.square().mean().sqrt()), float(delta.abs().max())
            baseline_rms = float(before.float().square().mean().sqrt())
            baseline_maximum = float(before.float().abs().max())
            unit = torch.finfo(torch.bfloat16).eps
            rms_tolerance = unit * max(1.0, baseline_rms)
            maximum_tolerance = unit * max(1.0, baseline_maximum)
            record = {'schema_version': TAG, 'same_registered_query': True, 'queries': 1,
                      'layers': 18, 'self_donor_velocity_RMS': rms, 'self_donor_velocity_max_abs': maximum,
                      'source_prefix_shared': True, 'layout': self.layout,
                      'baseline_velocity_RMS': baseline_rms, 'baseline_velocity_max_abs': baseline_maximum,
                      'tolerance_basis': 'one BF16 epsilon times max(1, baseline output scale)',
                      'allowed_normal_BF16_RMS': rms_tolerance, 'allowed_normal_BF16_max_abs': maximum_tolerance,
                      'cuda_peak_reserved_GiB': torch.cuda.max_memory_reserved() / 2**30,
                      'passed': rms < rms_tolerance and maximum < maximum_tolerance}
            write_json_atomic(ROOT / 'analysis/interface_self_donor.json', record)
            if not record['passed']:
                raise Pi05EvaluationError('self-donor changes actual velocity beyond normal BF16 tolerance')
        finally:
            self.image_mask, self.checking = image_mask, False
            self.phase = None

    def _denoise(self, original, recipient, donor, **arguments):
        if self.arm == 'C_from_N' and self.flow_step == 0:
            self._self_check(original, recipient, arguments)
        self.donor_pi, self.donor_mass = {}, {}
        self._forward(original, donor, 'donor', arguments)
        result = self._forward(original, recipient, 'recipient', arguments)
        self.phase = None
        self.donor_pi, self.donor_mass = {}, {}
        self.flow_step += 1
        return result

    @contextmanager
    def _intervene(self, recipient, donor):
        from transformers.models.gemma import modeling_gemma

        model = self.policy.model
        original_prefix, original_denoise = model.embed_prefix, model.denoise_step
        self.base_attention, self.repeat_kv = modeling_gemma.eager_attention_forward, modeling_gemma.repeat_kv
        model.embed_prefix = lambda *a, **kw: self._prefix(original_prefix, *a, **kw)
        model.denoise_step = lambda **kw: self._denoise(original_denoise, recipient, donor, **kw)
        modeling_gemma.eager_attention_forward = self._attention
        try:
            yield
        finally:
            model.embed_prefix, model.denoise_step = original_prefix, original_denoise
            modeling_gemma.eager_attention_forward = self.base_attention
            self.phase = None
            self.donor_pi, self.donor_mass = {}, {}

    @torch.no_grad()
    def predict_action_chunk(self, prepared, batch, *, noise, num_steps):
        if len(prepared) != noise.shape[0] or num_steps != 10:
            raise Pi05EvaluationError('official same-query ten-flow batch changed')
        recipient_name, donor_name = route(self.arm)
        start = time.perf_counter()
        copy_task_lora_state_(self.policy, self.identity, self.lora)
        recipient = self.batched.pack_states([self._state_for(recipient_name, p.key) for p in prepared])
        self.flow_step = 0
        self.current_stats = torch.zeros(len(prepared), 18, 10, len(METRICS), device=noise.device)
        if donor_name is None:
            with self.batched.activate_packed(recipient):
                result = self.policy.predict_action_chunk(batch, noise=noise, num_steps=num_steps)
        else:
            donor = self.batched.pack_states([self._state_for(donor_name, p.key) for p in prepared])
            with self._intervene(recipient, donor):
                result = self.policy.predict_action_chunk(batch, noise=noise, num_steps=num_steps)
            if self.flow_step != 10:
                raise Pi05EvaluationError('donor/recipient flow-step count changed')
        for index, p in enumerate(prepared):
            key = (self.arm, p.key)
            self.totals[key] = self.totals.get(key, 0) + self.current_stats[index]
            self.calls[key] = self.calls.get(key, 0) + 1
        torch.cuda.synchronize()
        sample = self.packing.setdefault(f'{self.arm}/batch{len(prepared)}', {'chunks': 0, 'seconds': 0.0})
        sample['chunks'] += 1
        sample['seconds'] += time.perf_counter() - start
        return result

    def episode_evidence(self, prepared):
        key = (self.arm, prepared.key)
        calls = self.calls[key]
        output = self.output / 'attention_effects' / f'{prepared.key}.json'
        record = {'schema_version': TAG, 'registration': registration(self.arm),
                  'condition_id': prepared.key, 'replans': calls, 'layout': self.layout,
                  'metric_names': list(METRICS), 'per_layer_flow_mean': (self.totals[key] / calls).cpu().tolist(),
                  'same_input_donor_suffix_calls': calls * 10 if route(self.arm)[1] else 0,
                  'prefix_prefills_per_replan': 1, 'donor_integrations': 0,
                  'native_precision': 'FP32 softmax then query dtype matmul; original RoPE/GQA/scaling/mask',
                  'hooks_restored_after_each_chunk': True}
        write_json_atomic(output, record)
        return {**prepared.evidence, 'self_image_attention_transfer':
                {'registration': registration(self.arm), 'effect': file_record(output)}}

    def close(self):
        super().close()
        for cache in self.caches.values():
            cache.clear()
