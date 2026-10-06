"""Non-held privileged F diagnostic within the original ten-step PI05 ODE."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX
from ember.operator_writer.data import FormalData, TASKS
from ember.operator_writer.scope import STATES, task_keys_from_ids
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_source_checkpoint import read_json

from .materialization import model_weights
from .readout import (F_EVAL_SCHEMA, F_KIND, ROOT, TASK, feedback_episode_evidence)


@dataclass(frozen=True)
class PreparedFeedback:
    key: str
    task: int
    demo: int
    evidence: dict


def _teacher_batch(rows: list[dict], device):
    """Pad only authorized GT trajectories; padded frames have no valid Values."""
    keys = set(rows[0])
    lengths = [len(row["frame_indices"]) for row in rows]
    if (any(set(row) != keys for row in rows) or min(lengths) < 1
            or "valid" not in keys or "presence" not in keys):
        raise Pi05EvaluationError("F teacher geometry/mask schema changed")
    padded = {}
    for key in keys:
        tensors = [torch.as_tensor(row[key], device=device) for row in rows]
        if any(value.ndim < 1 or value.shape[0] != length
               for value, length in zip(tensors, lengths, strict=True)):
            raise Pi05EvaluationError("F teacher fields lost frame alignment")
        output = tensors[0].new_zeros((len(rows), max(lengths), *tensors[0].shape[1:]))
        for index, value in enumerate(tensors):
            output[index, :len(value)] = value
        padded[key] = output
    indices = padded.pop("frame_indices")
    return padded, indices


class FrozenFeedbackAdapter:
    """F receives training GT only and modifies the same source denoise velocity."""
    def __init__(self, *, policy, source, evaluation_adapter, task_keys, device, require_formal):
        from .feedback import FeedbackFunction
        from .labels import LabelStore
        from .runtime import semantic_vectors

        bank = evaluation_adapter
        if (bank.get("kind") != F_KIND or bank.get("schema_version") != F_EVAL_SCHEMA
                or bank.get("study_id") != TASK or bank.get("readout_model") != "F"
                or bank.get("macro") != 450 or bank.get("panel") != "seen144"
                or bank.get("source") != source or not require_formal
                or set(task_keys) != set(task_keys_from_ids(TASKS))
                or any(tuple(row["init_state_id"] for row in task["episodes"]) != STATES
                       for task in bank["tasks"])):
            raise Pi05EvaluationError("F is restricted to the registered training36 seen144 diagnostic")
        if any(name.endswith((LORA_A_SUFFIX, LORA_B_SUFFIX)) for name, _ in policy.named_parameters()):
            raise Pi05EvaluationError("privileged F must read the unadapted original source, without LoRA")
        self.bank, self.policy, self.device = bank, policy, torch.device(device)
        self.tasks = {(row["suite"], row["task_id"]): row for row in bank["tasks"]}
        if set(self.tasks) != set(task_keys) or {row["global_task_id"] for row in bank["tasks"]} != set(TASKS):
            raise Pi05EvaluationError("F worker task registry crossed its fixed non-held allowlist")
        spec, asset_root = read_json(Path(bank["spec"]["path"])), Path(bank["asset_root"])
        self.data = FormalData(asset_root, spec, query_labels=False, task_ids=TASKS, role="train")
        self.labels = LabelStore(self.data, asset_root, cache_root=ROOT / "labels")
        semantic_path = ROOT / "labels/semantic_vectors.json"
        vectors = (read_json(semantic_path) if semantic_path.is_file() else
                   semantic_vectors(policy, self.labels.semantic_names,
                                    tokenizer_path=asset_root / spec["source"]["tokenizer"]))
        if isinstance(vectors, dict) and "vectors" in vectors:
            vectors = vectors["vectors"]
        self.labels.set_semantics({name: np.asarray(vector, dtype=np.float32) for name, vector in vectors.items()})
        self.feedback = FeedbackFunction().to(self.device)
        self.feedback.load_state_dict(model_weights(Path(bank["checkpoint"]), "F", device=self.device), strict=True)
        self.feedback.requires_grad_(False).eval()
        self.policy.requires_grad_(False).eval()
        self.teachers: OrderedDict[tuple[int, int], dict] = OrderedDict()
        self.context = None

    def prepare_episode(self, *, suite, task_id, init_state_id):
        if init_state_id not in STATES:
            raise Pi05EvaluationError("privileged F cannot evaluate held or unregistered states")
        task = self.tasks[(suite, task_id)]
        episode = next(row for row in task["episodes"] if row["init_state_id"] == init_state_id)
        return PreparedFeedback(episode["condition_id"], task["global_task_id"], episode["teacher_demo_indices"][0],
                                feedback_episode_evidence(self.bank, task, episode))

    def observe(self, prepared, envs, observations):
        if not prepared or not len(prepared) == len(envs) == len(observations):
            raise Pi05EvaluationError("F current GT lost the paired live environment")
        current = []
        for item, env, observation in zip(prepared, envs, observations, strict=True):
            if item.task not in TASKS:
                raise Pi05EvaluationError("F current registry is outside training36")
            owner = getattr(env, "env", env)
            current.append(self.labels.registries[item.task].runtime(owner.sim, observation))
        self.context = ([item.key for item in prepared], current)

    def _teacher(self, item):
        key = (item.task, item.demo)
        if key not in self.teachers:
            self.teachers[key] = self.labels.teacher(item.task, item.demo)
        self.teachers.move_to_end(key)
        return self.teachers[key]

    @torch.no_grad()
    def predict_action_chunk(self, prepared, batch, *, noise, num_steps):
        if (num_steps != 10 or noise.shape != (len(prepared), 50, 32)
                or self.context is None or self.context[0] != [item.key for item in prepared]):
            raise Pi05EvaluationError("F requires the original paired ten-step full-horizon ODE")
        current = {key: torch.as_tensor(np.stack([row[key] for row in self.context[1]]), device=self.device)
                   for key in self.context[1][0] if key != "frame_indices"}
        teacher, indices = _teacher_batch([self._teacher(item) for item in prepared], self.device)
        calls, h0 = 0, None

        def capture_hidden(_module, arguments):
            nonlocal h0
            h0 = arguments[0]

        def feedback_velocity(_module, _arguments, v0):
            nonlocal calls, h0
            if h0 is None or h0.shape != (len(prepared), 50, 1024) or v0.shape != noise.shape:
                raise Pi05EvaluationError("F hook did not capture the same complete source H0/v0")
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda"):
                result = self.feedback(teacher, current, h0, v0, indices)
            h0, calls = None, calls + 1
            if result.shape != v0.shape or not torch.isfinite(result).all():
                raise Pi05EvaluationError("F produced an invalid full source velocity")
            return result.to(v0.dtype)

        projection = self.policy.model.action_out_proj
        before = projection.register_forward_pre_hook(capture_hidden)
        after = projection.register_forward_hook(feedback_velocity)
        try:
            result = self.policy.predict_action_chunk(batch, noise=noise, num_steps=num_steps)
            if calls != num_steps:
                raise Pi05EvaluationError("F must run exactly once per original source denoise")
            return result
        finally:
            before.remove()
            after.remove()
            self.context = None
            while len(self.teachers) > max(8, len(prepared)):
                self.teachers.popitem(last=False)

    def close(self):
        self.data.close()
        self.teachers.clear()
        self.context = None
