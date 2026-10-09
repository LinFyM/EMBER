"""Frozen native assets, bounded legal teacher features, and shared replay."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import time

import torch
from safetensors.torch import load_file

from ember.batched_lora import BatchedLoRAInference
from ember.operator_writer.native import read_frozen_teacher_features
from ember.pi05_eval_contract import inspect_source_checkpoint, load_evaluation_authorities
from ember.pi05_lora import derive_pi05_lora_rank, load_pi05_lora_contract
from ember.pi05_processing import Pi05LiberoProcessor, Pi05TeacherPrefixTokenizer
from ember.pi05_source_checkpoint import read_json
from ember.pi05_source_setup import load_policy
from ember.writer.data import RawTeacherVideoStore
from ember.writer.functional import prepare_frozen_writer_policy
from ember.writer.learning_data import load_learning_tasks
from ember.writer.runtime import autocast

from .model import ExperienceCompiler


class Runtime:
    """One frozen source and one trainable Compiler; no condition optimizer."""

    def __init__(self, asset_root, device, *, frame_chunk=8, decoder_chunk=4096,
                 experience_chunk=16, native_frame_chunk=None, cache_bytes=2 * 1024**3):
        from .contract import SOURCE, MT_PATH, TASKS36 as TASKS

        self.asset_root, self.device = Path(asset_root), torch.device(device)
        self.frame_chunk, self.cache_bytes = frame_chunk, cache_bytes
        self.native_frame_chunk = frame_chunk if native_frame_chunk is None else native_frame_chunk
        authorities = load_evaluation_authorities(self.asset_root / SOURCE['evaluation_config'], self.asset_root)
        checkpoint = self.asset_root / SOURCE['checkpoint']
        self.source = inspect_source_checkpoint(authorities, checkpoint.parent.parent, checkpoint,
                                                evaluation_mode='formal')
        tokenizer = self.asset_root / SOURCE['tokenizer']
        stats = read_json(self.asset_root / SOURCE['normalization'])['stats']
        self.policy = load_policy(Path(self.source['model_path']), authorities.source_base_config, self.device)
        self.processor = Pi05LiberoProcessor(stats, tokenizer, 200, str(self.device))
        self.tokenizer = Pi05TeacherPrefixTokenizer(tokenizer, 200, str(self.device))
        self.lora = derive_pi05_lora_rank(load_pi05_lora_contract(self.asset_root / SOURCE['lora_contract']), rank=128)
        prepare_frozen_writer_policy(self.policy, self.lora)
        self.policy.model.gradient_checkpointing_disable()
        self.mt = load_file(str(MT_PATH), device=str(self.device))
        self.compiler = ExperienceCompiler(self.mt, [t.name for t in self.lora.targets],
            decoder_chunk=decoder_chunk, frame_chunk=frame_chunk, experience_chunk=experience_chunk).to(self.device)
        self.execution = BatchedLoRAInference(self.policy, self.lora)
        self.probe = torch.randn((50, 32), generator=torch.Generator().manual_seed(1729)).to(self.device)
        self.tasks = load_learning_tasks(self.asset_root, TASKS, protocol_path=SOURCE['data_protocol'])
        self.stores, self.cache = {}, OrderedDict()
        self.cache_size = 0
        self.feature_seconds = 0.
        self.native_teacher_frames = 0
        self.last_teacher_cost = {}

    def _store(self, role):
        if role not in self.stores:
            if role == 'train':
                tasks = self.tasks
            elif role == 'validation':
                from ember.task_protocol import load_task_authorities
                from .contract import SOURCE
                _, manifest = load_task_authorities(self.asset_root, SOURCE['data_protocol'])
                ids = [r['global_task_id'] for r in manifest['tasks'] if r['split_role'] == role]
                tasks = load_learning_tasks(self.asset_root, ids, role=role, protocol_path=SOURCE['data_protocol'])
            else:
                raise ValueError('Test is sealed; Compiler teacher store supports train or validation only')
            self.stores[role] = (tasks, RawTeacherVideoStore([t.authority for t in tasks.values()],
                                                         frame_stride=5, camera_view='dual'))
        return self.stores[role]

    @torch.no_grad()
    def teacher(self, task_id, demo, role='train'):
        key = role, int(task_id), int(demo)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.last_teacher_cost = dict(cache_hit=True, native_encoder_seconds=0., native_encoded_frames=0)
            return self.cache[key]
        tasks, store = self._store(role)
        raw = store.load(task_id, demo)
        tokens, mask, task_span = self.tokenizer([tasks[task_id].authority.language])
        started = time.monotonic()
        with autocast(self.device):
            phi, hidden = read_frozen_teacher_features(self.policy, self.mt, self.probe,
                (torch.from_numpy(raw.frames).to(self.device), torch.from_numpy(raw.frame_indices), tokens, mask),
                frame_chunk=self.native_frame_chunk)
            language = self.policy.model.paligemma_with_expert.embed_language_tokens(tokens)
            language = language[0, task_span[0]].float().mean(0)
        features = dict(phi=phi.cpu(), hidden=hidden.cpu(), language=language.cpu(),
                        indices=torch.from_numpy(raw.frame_indices))
        elapsed = time.monotonic() - started
        self.feature_seconds += elapsed
        self.last_teacher_cost = dict(cache_hit=False, native_encoder_seconds=elapsed,
                                      native_encoded_frames=len(raw.frames))
        self.native_teacher_frames += len(raw.frames)
        size = sum(v.nbytes for v in features.values())
        while self.cache and self.cache_size + size > self.cache_bytes:
            _, old = self.cache.popitem(last=False)
            self.cache_size -= sum(v.nbytes for v in old.values())
        if size <= self.cache_bytes:
            self.cache[key], self.cache_size = features, self.cache_size + size
        return features

    @torch.no_grad()
    def frozen_images(self, raw):
        """Native frozen image embeddings of actual rotated RGB, never a probe H."""
        images = raw.to(self.device).float().div(255)
        batch = {'observation.images.base_0_rgb': images[:, 0],
                 'observation.images.left_wrist_0_rgb': images[:, 1]}
        pixels, _ = self.policy._preprocess_images(batch)
        with autocast(self.device):
            result = [self.policy.model.paligemma_with_expert.embed_image(value) for value in pixels[:2]]
        return torch.cat(result, 1).detach().cpu()

    def replay(self, teacher, chain, masked=False):
        """All Q revisions stay differentiable; raw E/H alone stop gradient."""
        with autocast(self.device):
            q = self.compiler.initial(teacher)
            states = [self.compiler.decode(q)]
            for endpoint in chain.endpoints:
                experience = {} if masked else chain.experience(endpoint)
                q = self.compiler.revise(q, teacher, experience)
                states.append(self.compiler.decode(q))
        return states

    def load_checkpoint(self, checkpoint):
        self.compiler.load_state_dict(load_file(str(Path(checkpoint) / 'ecp.safetensors'),
                                               device=str(self.device)), strict=True)

    def close(self):
        self.execution.close()
        for _, store in self.stores.values():
            store.close()
