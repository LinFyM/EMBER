"""Frozen assets and observed complete-factor editing; no hidden persistent Q."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import socket
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

    def __init__(self, asset_root, device, *, frame_chunk=8,
                 experience_chunk=16, native_frame_chunk=None, cache_bytes=2 * 1024**3,
                 cache_root=None, feature_cache_bytes=2 * 1024**3, support_microbatch=4):
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
            frame_chunk=frame_chunk, experience_chunk=experience_chunk).to(self.device)
        self.execution = BatchedLoRAInference(self.policy, self.lora)
        self.support_microbatch = support_microbatch
        self.neural_reads, self.neural_read_frames = 0, 0
        self.probe = torch.randn((50, 32), generator=torch.Generator().manual_seed(1729)).to(self.device)
        self.tasks = load_learning_tasks(self.asset_root, TASKS, protocol_path=SOURCE['data_protocol'])
        self.stores, self.cache = {}, OrderedDict()
        self.cache_size = 0
        self.feature_seconds = 0.
        self.native_teacher_frames = 0
        self.last_teacher_cost = {}
        from .contract import RUN_ROOT
        from .storage import FeatureCache, RecordWriter
        cache_root = Path(cache_root) if cache_root is not None else RUN_ROOT / 'frozen_features'
        # Frozen derivatives are reconstructible; bounded host shards avoid
        # duplicating raw facts or copying the historical large cache.
        self.features = FeatureCache(cache_root / socket.gethostname(), max_bytes=feature_cache_bytes)
        self.io = RecordWriter()
        self.image_cache_hits, self.image_encoded_observations = 0, 0

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
        disk_key = f'teacher_{role}_{int(task_id):03d}_{int(demo):02d}'
        disk = self.features.get(disk_key)
        if disk is not None:
            self.last_teacher_cost = dict(cache_hit=True, disk_cache_hit=True,
                                          native_encoder_seconds=0., native_encoded_frames=0)
            self._remember(key, disk)
            return disk
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
        from .storage import tensor_bytes
        self.io.submit(self.features.put, disk_key, features, byte_cost=tensor_bytes(features))
        self._remember(key, features)
        return features

    def _remember(self, key, features):
        size = sum(v.nbytes for v in features.values())
        while self.cache and self.cache_size + size > self.cache_bytes:
            _, old = self.cache.popitem(last=False)
            self.cache_size -= sum(v.nbytes for v in old.values())
        if size <= self.cache_bytes:
            self.cache[key], self.cache_size = features, self.cache_size + size

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

    def native_actions(self, states, batch, noise, *, batch_indices=None, checkpointed=True):
        from .execution import native_actions
        with autocast(self.device):
            return native_actions(self, states, batch, noise, batch_indices=batch_indices,
                                  checkpointed=checkpointed)

    def _context(self, item):
        self.neural_reads += 1
        self.neural_read_frames += len(item['teacher']['indices'])
        return self.compiler.context(item['teacher'], item['experience'], item['support']['indices'])

    def edit(self, incoming, teacher, experience, *, support):
        return self.edit_many([dict(incoming=incoming, teacher=teacher, experience=experience,
                                    support=support)])[0]

    def edit_many(self, items):
        """Different conditions share native batches; each retains its own M mean."""
        from .execution import factor_vjp, join_supports
        started = time.monotonic()
        with torch.no_grad(), autocast(self.device):
            contexts = [self._context(item) for item in items]
        if hasattr(self, 'profile_root'):
            torch.cuda.synchronize(self.device)
        context_seconds = time.monotonic() - started
        support, counts = join_supports([item['support'] for item in items])
        bounds = [0]
        for count in counts:
            bounds.append(bounds[-1] + count)
        def pressure(actions, start, stop):
            result = torch.empty_like(actions)
            for i, context in enumerate(contexts):
                low, high = max(start, bounds[i]), min(stop, bounds[i + 1])
                if low < high:
                    with torch.enable_grad(), autocast(self.device):
                        result[low - start:high - start] = self.compiler.action_cotangent(
                            context[low - bounds[i]:high - bounds[i]], actions[low - start:high - start],
                            create_graph=False).detach() / counts[i]
            return result
        started = time.monotonic()
        incoming = [item['incoming'] for item in items]
        pullbacks, actions = factor_vjp(self, incoming, support, pressure,
            microbatch=self.support_microbatch, return_actions=True)
        outgoing = []
        for i, (state, pullback) in enumerate(zip(incoming, pullbacks, strict=True)):
            change = self.compiler.precondition(pullback)
            edited = {key: value.detach().to(self.device).float() - change[key] for key, value in state.items()}
            if any(not bool(torch.isfinite(value).all()) for value in edited.values()):
                raise RuntimeError('functional edit produced nonfinite complete factors')
            outgoing.append(edited)
            items[i]['support']['actions'] = actions[bounds[i]:bounds[i + 1]].detach()
        self.last_revision_cost = dict(context_seconds=context_seconds,
            native_F_VJP_seconds=time.monotonic() - started, support_points=sum(counts),
            event_support_counts=counts, group_event_count=len(items),
            physical_support_microbatch=self.support_microbatch)
        return outgoing

    def backward_revision(self, incoming, teacher, experience, support, credit, *, microbatch=None):
        return self.backward_revisions([dict(incoming=incoming, teacher=teacher,
            experience=experience, support=support)], [credit], microbatch=microbatch)[0]

    def backward_revisions(self, items, credits, *, microbatch=None):
        """Exact -J(P^T v)/M, batched across distinct actual incoming adapters."""
        from .execution import factor_jvp, join_supports
        chunk = self.support_microbatch if microbatch is None else microbatch
        support, counts = join_supports([item['support'] for item in items])
        started = time.monotonic()
        tangent = factor_jvp(self, [item['incoming'] for item in items], support,
            [self.compiler.precondition(credit) for credit in credits], microbatch=chunk)
        jvp_seconds = time.monotonic() - started
        started = time.monotonic()
        results, scales, cursor = [], [], 0
        for item, count in zip(items, counts, strict=True):
            qbar = -tangent[cursor:cursor + count] / count
            with torch.enable_grad(), autocast(self.device):
                context = self._context(item)
                q = self.compiler.action_cotangent(context, item['support']['actions'], create_graph=True)
                torch.autograd.backward(q, qbar.to(q))
            scales.append(dict(q_RMS=float(q.detach().float().square().mean().sqrt()),
                               qbar_RMS=float(qbar.float().square().mean().sqrt())))
            results.append(qbar.detach())
            cursor += count
        if hasattr(self, 'profile_root'):
            torch.cuda.synchronize(self.device)
        self.last_adjoint_cost = dict(native_JVP_seconds=jvp_seconds,
            shared_context_energy_backward_seconds=time.monotonic() - started,
            group_event_count=len(items), event_support_counts=counts, event_pressure_scales=scales)
        return results

    @torch.no_grad()
    def observation_features(self, observations):
        result, missing, keys = {}, [], []
        cached_features = self.features.get_many(observations)
        for key, raw in observations.items():
            cached = cached_features.get(key)
            if cached is None:
                missing.append(raw['images'])
                keys.append(key)
            else:
                result[key] = cached['phi']
                self.image_cache_hits += 1
        if missing:
            self.image_encoded_observations += len(missing)
            for start in range(0, len(missing), self.native_frame_chunk):
                phi = self.frozen_images(torch.stack(missing[start:start + self.native_frame_chunk]))
                entries = {}
                for key, value in zip(keys[start:start + len(phi)], phi, strict=True):
                    result[key] = value
                    entries[key] = {'phi': value}
                from .storage import tensor_bytes
                self.io.submit(self.features.put_many, entries, byte_cost=tensor_bytes(entries))
        return result

    def load_checkpoint(self, checkpoint):
        self.compiler.load_state_dict(load_file(str(Path(checkpoint) / 'ecp.safetensors'),
                                               device=str(self.device)), strict=True)

    def close(self):
        self.io.close()
        self.execution.close()
        for _, store in self.stores.values():
            store.close()
