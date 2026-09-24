"""Frozen PI0.5 10-flow predictions across sealed task LoRA conditions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors.torch import load_file

from ember.batched_lora import BatchedLoRAInference
from ember.lora import identity_lora_state, inject_task_lora, validate_lora_state
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.crossed_video_field import QUERY_SCHEMA, SCHEMA
from ember.pi05_eval.worker_setup import load_policy
from ember.pi05_evaluation import make_policy_noise
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_processing import libero_policy_input


PREDICTION_SCHEMA = "ember_crossed_video_field_prediction_v1"
def condition_plan(manifest: Mapping[str, Any], query: Mapping[str, Any]) -> list[dict[str, Any]]:
    global_id = str(query["global_task_id"])
    state = str(query["init_state_id"])
    banks = manifest["banks"]
    pool = banks["C_correct"]["conditions"][global_id]["pool_by_video_demo"]
    if (len(pool) != 50
            or [item["teacher_demo_indices"] for item in pool] != [[i] for i in range(50)]):
        raise Pi05EvaluationError("crossed-video correct condition pool changed")
    references = [
        {"kind": "B_language",
         **banks["B"]["conditions"][global_id]["per_state"][state]},
        {"kind": "Source1000", "condition_id": "Source1000",
         "adapter": None, "global_task_id": int(global_id), "teacher_demo_indices": []},
        {"kind": "C_wrong",
         **banks["C_wrong"]["conditions"][global_id]["per_state"][state]},
    ]
    plan = [{"kind": "C_correct", "video_demo": i, **row} for i, row in enumerate(pool)]
    plan.extend(references)
    if (len(plan) != 53 or plan[50]["condition_id"] != query["B_condition_id"]
            or plan[52]["condition_id"] != query["wrong_condition_id"]
            or query["C_diagonal_condition_id"] not in
            {row["condition_id"] for row in plan[:50]}
            or query["C_scheduled_other_condition_id"] not in
            {row["condition_id"] for row in plan[:50]}):
        raise Pi05EvaluationError("crossed-video query/condition mapping changed")
    return plan


def scaled_osc_actions(actions: np.ndarray, audit: Mapping[str, Any]) -> np.ndarray:
    if (actions.ndim != 3 or actions.shape[1:] != (50, 7)
            or audit["controller_class"] != "OperationalSpaceController"
            or audit["impedance_mode"] != "fixed"
            or audit["control_delta"] is not True
            or audit["arm_control_dim"] != 6 or audit["robot_action_dim"] != 7):
        raise Pi05EvaluationError("crossed-video OSC action contract changed")
    low = np.asarray(audit["controller_input_min"], dtype=np.float32)
    high = np.asarray(audit["controller_input_max"], dtype=np.float32)
    out_low = np.asarray(audit["controller_output_min"], dtype=np.float32)
    out_high = np.asarray(audit["controller_output_max"], dtype=np.float32)
    if (low.shape != high.shape or low.shape != out_low.shape or low.shape != out_high.shape
            or low.shape != (6,) or not np.all(high > low)):
        raise Pi05EvaluationError("crossed-video OSC scale authority changed")
    clipped = np.clip(actions[:, :5, :6], low, high)
    return ((clipped - (high + low) / 2.0) *
            (np.abs(out_high - out_low) / np.abs(high - low)) +
            (out_high + out_low) / 2.0).astype(np.float32)


def _query_input(query: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(query["query_path"])
    if not path.is_file() or path.stat().st_size != int(query["query_bytes"]):
        raise Pi05EvaluationError("crossed-video frozen query asset changed")
    with np.load(path, allow_pickle=False) as saved:
        if saved["schema_version"].item() != QUERY_SCHEMA:
            raise Pi05EvaluationError("crossed-video frozen query schema changed")
        obs = {
            "agentview_image": saved["agentview_image"].copy(),
            "robot0_eye_in_hand_image": saved["wrist_image"].copy(),
            "robot0_eef_pos": saved["eef_pos"].copy(),
            "robot0_eef_quat": saved["eef_quat"].copy(),
            "robot0_gripper_qpos": saved["gripper_qpos"].copy(),
        }
        original_state = saved["state8"].copy()
    raw = libero_policy_input(obs, query["language"])
    if (obs["agentview_image"].shape != (256, 256, 3)
            or obs["robot0_eye_in_hand_image"].shape != (256, 256, 3)
            or not np.array_equal(np.asarray(raw["observation.state"]), original_state)):
        raise Pi05EvaluationError("crossed-video same-query RGB/state changed")
    return raw


class CrossedVideoPredictor:
    """One canonical policy with per-sample sealed LoRA and identity Source."""

    def __init__(self, manifest: Mapping[str, Any], *, batch_size: int) -> None:
        if (manifest["schema_version"] != SCHEMA or batch_size < 1 or batch_size > 8
                or manifest["policy"]["chunk_size"] != 50
                or manifest["policy"]["num_inference_steps"] != 10
                or manifest["policy"]["action_dim"] != 7):
            raise Pi05EvaluationError("crossed-video predictor configuration changed")
        normalization = json.loads(Path(manifest["normalization"]["path"]).read_text())
        if (Path(manifest["normalization"]["path"]).stat().st_size !=
                manifest["normalization"]["bytes"]):
            raise Pi05EvaluationError("crossed-video source normalization changed")
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        policy, processor, postprocess = load_policy(
            Path(manifest["model"]["model_path"]), normalization["stats"],
            Path(manifest["tokenizer"]["path"]), manifest["policy"],
        )
        policy.requires_grad_(False).eval()
        lora = load_pi05_lora_contract(
            Path(manifest["banks"]["C_correct"]["lora_contract"]["path"]))
        inject_task_lora(policy, lora)
        policy.requires_grad_(False).eval()
        if any(parameter.requires_grad for parameter in policy.parameters()):
            raise Pi05EvaluationError("crossed-video source policy has trainable parameters")
        self.policy = policy
        self.processor = processor
        self.postprocess = postprocess
        self.lora = lora
        self.batched = BatchedLoRAInference(policy, lora)
        self.identity = identity_lora_state(lora)
        self.batch_size = batch_size
        self.cache: dict[str, dict[str, torch.Tensor]] = {}
        self.cached_task: int | None = None
        self.device = torch.device("cuda:0")
        if int(policy.config.max_action_dim) != 32:
            raise Pi05EvaluationError("crossed-video internal action dimension changed")
        torch.cuda.reset_peak_memory_stats(self.device)

    def close(self) -> None:
        self.batched.close()
        self.cache.clear()

    def _state(self, condition: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
        if condition["kind"] == "Source1000":
            return self.identity
        record = condition["adapter"]
        key = str(record["path"])
        if key not in self.cache:
            path = Path(key)
            if not path.is_file() or path.stat().st_size != int(record["bytes"]):
                raise Pi05EvaluationError("crossed-video adapter changed after manifest")
            state = load_file(str(path), device="cpu")
            validate_lora_state(state, self.lora)
            if any(value.dtype != torch.float32 or not torch.isfinite(value).all()
                   for value in state.values()):
                raise Pi05EvaluationError("crossed-video adapter tensor invalid")
            self.cache[key] = state
        return self.cache[key]

    def _forward_group(self, group: list[dict[str, Any]],
                       batch_input: Mapping[str, torch.Tensor], noise: torch.Tensor,
                       *, pilot: bool) -> tuple[np.ndarray, np.ndarray, float | None]:
        size = len(group)
        repeated = {key: value.expand(size, *value.shape[1:]).contiguous()
                    for key, value in batch_input.items()}
        repeated_noise = noise.expand(size, -1, -1).contiguous()
        states = [self._state(condition) for condition in group]
        with self.batched.activate(states):
            predicted = self.policy.predict_action_chunk(
                repeated, noise=repeated_noise, num_steps=10,
            )
        if predicted.shape != (size, 50, 7) or not torch.isfinite(predicted).all():
            raise Pi05EvaluationError("crossed-video normalized PI05 output invalid")
        env_actions = self.postprocess(predicted)
        if env_actions.shape != (size, 50, 7) or not torch.isfinite(env_actions).all():
            raise Pi05EvaluationError("crossed-video environment PI05 output invalid")
        source_check = None
        if pilot and any(item["kind"] == "Source1000" for item in group):
            source_index = next(i for i, item in enumerate(group)
                                if item["kind"] == "Source1000")
            direct = self.policy.predict_action_chunk(
                repeated, noise=repeated_noise, num_steps=10,
            )
            source_check = float((direct[source_index].float() -
                                  predicted[source_index].float()).abs().max().item())
            if source_check > 0.05:
                raise Pi05EvaluationError(
                    f"crossed-video Source adapter clearing failed: {source_check}")
        return (predicted.to(torch.float32).cpu().numpy(),
                env_actions.to(torch.float32).cpu().numpy(), source_check)

    def predict_query(self, manifest: Mapping[str, Any], query: Mapping[str, Any],
                      output: Path, *, pilot: bool) -> dict[str, Any]:
        if self.cached_task != int(query["global_task_id"]):
            self.cache.clear()
            self.cached_task = int(query["global_task_id"])
        plan = condition_plan(manifest, query)
        raw = _query_input(query)
        processed = self.processor(raw)
        batch_input = {key: value for key, value in processed.items()
                       if isinstance(value, torch.Tensor)}
        if any(value.shape[0] != 1 for value in batch_input.values()):
            raise Pi05EvaluationError("crossed-video frozen query preprocessing changed")
        self.policy.reset()
        noise, seeds = make_policy_noise(
            [{"init_state_id": int(query["init_state_id"]),
              "replan_index": int(query["global_replan_index"])}],
            root_seed=7, suite=query["suite"], task_id=int(query["task_id"]),
            chunk_size=50, max_action_dim=32, device=self.device,
        )
        if seeds != (int(query["policy_noise_seed"]),) or noise.shape != (1, 50, 32):
            raise Pi05EvaluationError("crossed-video stateless policy RNG changed")
        normalized = np.empty((53, 50, 7), dtype=np.float32)
        actions = np.empty_like(normalized)
        source_check = None
        started = time.monotonic()
        with torch.inference_mode():
            for start in range(0, 53, self.batch_size):
                group = plan[start:start + self.batch_size]
                size = len(group)
                group_normalized, group_actions, clear = self._forward_group(
                    group, batch_input, noise, pilot=pilot)
                normalized[start:start+size] = group_normalized
                actions[start:start+size] = group_actions
                if clear is not None:
                    source_check = clear
        osc = scaled_osc_actions(actions, query["osc_channel_audit"])
        if osc.shape != (53, 5, 6) or not np.isfinite(osc).all():
            raise Pi05EvaluationError("crossed-video scaled OSC output invalid")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            np.savez_compressed(
                handle, schema_version=np.asarray(PREDICTION_SCHEMA),
                normalized_actions=normalized, environment_actions=actions,
                osc_scaled_first5=osc,
                policy_noise=noise[0].to(torch.float32).cpu().numpy(),
                condition_ids=np.asarray([item["condition_id"] for item in plan]),
                condition_kinds=np.asarray([item["kind"] for item in plan]),
            )
        return {
            "schema_version": PREDICTION_SCHEMA,
            "query_id": query["query_id"], "query_path": query["query_path"],
            "prediction_path": str(output), "prediction_bytes": output.stat().st_size,
            "condition_ids": [item["condition_id"] for item in plan],
            "condition_kinds": [item["kind"] for item in plan],
            "policy_noise_seed": int(query["policy_noise_seed"]),
            "global_replan_index": int(query["global_replan_index"]),
            "predictions": 53, "correct_predictions": 50,
            "reference_predictions": 3, "batch_size": self.batch_size,
            "source_clear_max_abs_normalized": source_check,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(self.device) / 2**20,
            "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved(self.device) / 2**20,
            "seconds": time.monotonic() - started,
            "no_grad": True, "inference_mode": True,
            "predicted_actions_executed_in_environment": False,
        }
