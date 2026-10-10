"""Canonical assets, source-only execution and versioned dual-meta features."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import time

import torch
from safetensors.torch import load_file

from ember.lora import validate_lora_state
from ember.pi05_eval_contract import inspect_source_checkpoint, load_evaluation_authorities
from ember.pi05_source_setup import load_policy
from ember.pi05_source_checkpoint import read_json
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_lora import load_pi05_lora_contract, derive_pi05_lora_rank
from ember.writer.data import RawTeacherVideoStore
from ember.writer.learning_data import load_learning_tasks
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.runtime import autocast
from ember.writer.practice.execution import native_actions
from ember.writer.practice import processed

from .contract import SOURCE, MT_PATH, TASKS, ASSET_ROOT, RUN_ROOT
from .native import MetaReader, PolicyContexts
from .model import FactorLayout, Generator, Actor


class Runtime:
    """One physical frozen source; owned ψ and θ are independent small modules."""
    def __init__(self, device='cuda:0', *, asset_root=ASSET_ROOT, root=RUN_ROOT, frame_chunk=4, response_microbatch=16,
                 feature_cache_bytes=2 * 1024**3):
        self.device, self.asset_root = torch.device(device), Path(asset_root)
        authorities = load_evaluation_authorities(self.asset_root / SOURCE['evaluation_config'], self.asset_root)
        checkpoint = self.asset_root / SOURCE['checkpoint']
        self.source = inspect_source_checkpoint(authorities, checkpoint.parent.parent, checkpoint, evaluation_mode='formal')
        self.policy = load_policy(Path(self.source['model_path']), authorities.source_base_config, self.device)
        self.root = Path(root)
        self.expert_lora = load_pi05_lora_contract(self.asset_root / SOURCE['lora_contract'])
        self.lora = derive_pi05_lora_rank(self.expert_lora, rank=8)
        self.identity = prepare_frozen_writer_policy(self.policy, self.lora)
        self.policy.model.gradient_checkpointing_disable()
        base_mt = {k: v.float() for k, v in load_file(str(MT_PATH), device=str(self.device)).items()}
        validate_lora_state(base_mt, self.expert_lora)
        self.mt = load_file(str(self.root/'initial.safetensors'), device=str(self.device))
        validate_lora_state(self.mt, self.lora)  # Residual Λ0; effective policy is MT300.
        stats = read_json(self.asset_root / SOURCE['normalization'])['stats']
        tokenizer = self.asset_root / SOURCE['tokenizer']
        self.processor = Pi05LiberoProcessor(stats, tokenizer, 200, str(self.device))
        self.tokenizer = Pi05TeacherPrefixTokenizer(tokenizer, 200, str(self.device))
        self.tasks = load_learning_tasks(self.asset_root, TASKS, role='train', protocol_path=SOURCE['data_protocol'])
        self.videos = RawTeacherVideoStore([t.authority for t in self.tasks.values()], frame_stride=5, camera_view='dual')
        base_path=self.root/'base/merged_targets.safetensors'
        merged=load_file(str(base_path),device=str(self.device)) if base_path.exists() else None
        self.execution = PolicyContexts(self.policy, self.lora, self.expert_lora, base_mt, self.mt, merged)
        self.prox_scales = self.execution.prox
        coordinates = read_json(self.root / 'coordinates.json')
        if coordinates['task_rank'] != 8 or coordinates['valid_coordinates'] != self.lora.parameter_count:
            raise ValueError('current run has incompatible task coordinates')
        if any(abs(v / coordinates['S_prox'][k] - 1) > 1e-5 for k, v in self.prox_scales.items()):
            raise ValueError('fixed proximal units differ from actual saved A0/MT dense update')
        self.bank_scale_path = self.root / 'bank_scale.json'
        scales = read_json(self.bank_scale_path)['S_G'] if self.bank_scale_path.exists() else None
        torch.manual_seed(7)
        meta = MetaReader(self.policy, self.execution, frame_chunk=frame_chunk)
        layout = FactorLayout(self.lora, self.mt, scales)
        self.generator = Generator(layout, meta).to(self.device)
        self.actor = Actor(self.generator.layout).to(self.device)
        self.response_microbatch = response_microbatch
        self.version, self.psi_frozen = 0, False
        self.feature_cache, self.feature_cache_size = OrderedDict(), 0
        self.raw_cache, self.raw_cache_size = OrderedDict(), 0
        self.feature_cache_bytes = feature_cache_bytes
        self.components = dict(meta_read_seconds=0., meta_read_frames=0, teacher_cache_hits=0,
                               response_seconds=0., response_observations=0, image_observations=0)

    def task_parameters(self, state):
        return self.execution.task_parameters(state)

    def condition(self, task_id, demo, *, video_task=None):
        video_task = task_id if video_task is None else video_task
        if task_id not in TASKS or video_task not in TASKS or not 0 <= demo < 50:
            raise ValueError('pilot teaching condition outside registered non-held pool')
        raw = self.videos.load(video_task, demo)
        tokens, mask, _ = self.tokenizer([self.tasks[task_id].authority.language])
        return (torch.from_numpy(raw.frames).to(self.device),
                torch.from_numpy(raw.frame_indices).to(self.device), tokens, mask)

    def teaching(self, task_id, demo, *, video_task=None):
        key = self.version, task_id, video_task, demo
        if self.psi_frozen and key in self.feature_cache:
            self.components['teacher_cache_hits'] += 1
            self.feature_cache.move_to_end(key)
            return {k: v.to(self.device) for k, v in self.feature_cache[key].items()}
        condition = self.condition(task_id, demo, video_task=video_task)
        started = time.monotonic()
        raw_key=task_id,video_task,demo
        if raw_key in self.raw_cache:
            self.raw_cache.move_to_end(raw_key)
            raw=tuple(v.to(self.device) for v in self.raw_cache[raw_key])
        else:
            with autocast(self.device):
                raw=self.generator.meta.encode_raw(condition)
            cpu=tuple(v.detach().cpu() for v in raw)
            size=sum(v.nbytes for v in cpu)
            while self.raw_cache and self.raw_cache_size+size>self.feature_cache_bytes:
                _,previous=self.raw_cache.popitem(last=False)
                self.raw_cache_size-=sum(v.nbytes for v in previous)
            if size<=self.feature_cache_bytes:
                self.raw_cache[raw_key]=cpu;self.raw_cache_size+=size
        with torch.set_grad_enabled(not self.psi_frozen), autocast(self.device):
            features = self.generator.meta(condition,raw_prefix=raw)
        self.components['meta_read_seconds'] += time.monotonic() - started
        self.components['meta_read_frames'] += len(condition[0])
        if self.psi_frozen:
            features = {k: v.detach() for k, v in features.items()}
            cpu = {k: v.cpu() for k, v in features.items()}
            size = sum(v.nbytes for v in cpu.values())
            while self.feature_cache and size + self.feature_cache_size > self.feature_cache_bytes:
                _, old = self.feature_cache.popitem(last=False)
                self.feature_cache_size -= sum(v.nbytes for v in old.values())
            if size <= self.feature_cache_bytes:
                self.feature_cache[key] = cpu
                self.feature_cache_size += size
        return features

    def mark_psi_update(self):
        if self.psi_frozen:
            raise ValueError('π training cannot modify its fixed generation kernel ψ')
        self.version += 1
        self.feature_cache.clear()
        self.feature_cache_size = 0

    def freeze_psi(self):
        if self.psi_frozen:return
        self.psi_frozen = True
        self.generator.eval().requires_grad_(False)
        self.feature_cache.clear()
        self.feature_cache_size = 0

    @torch.no_grad()
    def observation_features(self, observations):
        result = {}
        items = list(observations.items())
        for start in range(0, len(items), 16):
            selected = items[start:start + 16]
            if not selected:
                continue
            images = torch.stack([v['images'] for _, v in selected]).to(self.device).float().div(255)
            batch = {'observation.images.base_0_rgb': images[:, 0],
                     'observation.images.left_wrist_0_rgb': images[:, 1]}
            pixels, _ = self.policy._preprocess_images(batch)
            with autocast(self.device):
                phi = torch.cat([self.policy.model.paligemma_with_expert.embed_image(v) for v in pixels[:2]], 1)
            result.update({key: value.detach().cpu().clone() for (key, _), value in zip(selected, phi, strict=True)})
            self.components['image_observations'] += len(selected)
        return result

    def native_actions(self, states, batch, noise, *, batch_indices=None, checkpointed=True):
        if self.generator.meta.active.get() is not None:
            raise ValueError('read-meta leaked into native task response')
        with autocast(self.device):
            return native_actions(self, states, batch, noise, batch_indices=batch_indices, checkpointed=checkpointed)

    @torch.no_grad()
    def responses(self, states, history, language, *, parent=None):
        support = history.response_support(self, language)
        if support is None:
            return {}
        started, absolute = time.monotonic(), {}
        for key, state in states.items():
            chunks = []
            count = len(support['noise'])
            for start in range(0, count, self.response_microbatch):
                stop = min(count, start + self.response_microbatch)
                batch = {k: v[start:stop] for k, v in support['batch'].items()}
                owners = torch.zeros(stop - start, dtype=torch.long, device=self.device)
                chunks.append(self.native_actions([state], batch, support['noise'][start:stop],
                                                   batch_indices=owners, checkpointed=False).float().cpu())
            absolute[key] = torch.cat(chunks).flatten(1)
        self.components['response_seconds'] += time.monotonic() - started
        self.components['response_observations'] += len(support['noise']) * len(states)
        return {key: torch.cat((value, value - absolute[parent] if parent is not None else torch.zeros_like(value)), 1)
                for key, value in absolute.items()}

    def close(self):
        self.videos.close()
        self.generator.meta.close()
        self.execution.close()
