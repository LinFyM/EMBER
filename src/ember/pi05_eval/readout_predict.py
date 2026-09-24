"""Same-query four-cell inference and actual action-out flow observation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.crossed_video_predict import (
    CrossedVideoPredictor, _query_input, scaled_osc_actions,
)
from ember.pi05_eval.readout_state import GROUPS, READOUT, load_masked_state, mask_state
from ember.pi05_evaluation import make_policy_noise


SCHEMA = "ember_readout_realization_prediction_v1"


def _flow_components(base: np.ndarray, total: np.ndarray, noise: np.ndarray,
                     normalized: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if (base.shape != (10, 4, 50, 32) or total.shape != base.shape
            or normalized.shape != (4, 50, 7)
            or not all(np.isfinite(value).all() for value in (base, total, normalized))):
        raise Pi05EvaluationError("readout flow/action shape or finite contract failed")
    delta = total - base
    off = max(float(np.abs(delta[:, index]).max()) for index in (0, 2))
    reconstructed = noise[None] - 0.1 * total.sum(axis=0)
    components = {}
    residuals = {"euler": float(np.abs(reconstructed[:, :, :7] - normalized).max()),
                 "readout_off_delta": off}
    for label, left, right in (("11_10", 3, 2), ("01_00", 1, 0)):
        direct = -0.1 * delta[:, left].sum(axis=0)
        feedback = -0.1 * (base[:, left] - base[:, right]).sum(axis=0)
        components[f"direct_{label}"] = direct
        components[f"feedback_{label}"] = feedback
        residuals[label] = float(np.abs(
            reconstructed[left] - reconstructed[right] - direct - feedback).max())
    if off > 1e-5 or max(residuals["euler"], residuals["11_10"], residuals["01_00"]) > 1e-4:
        raise Pi05EvaluationError("readout off/Euler/decomposition contract failed")
    return components, residuals


class ActionOutObserver:
    """Observe the real hidden and consumed velocity after batched-LoRA hooks."""

    def __init__(self, policy: Any) -> None:
        module = dict(policy.named_modules())[READOUT]
        if torch.count_nonzero(module.lora_B["default"].weight).item() != 0:
            raise Pi05EvaluationError("physical action-out LoRA is not identity")
        self.events: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.enabled = False

        def observe(layer: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            if not self.enabled:
                return None
            if len(inputs) != 1 or inputs[0].shape[-1] != 1024 or output.shape[-2:] != (50, 32):
                raise Pi05EvaluationError("actual action-out hidden/velocity shape changed")
            base = layer.base_layer(inputs[0])
            if base.shape != output.shape or not torch.isfinite(base).all() or not torch.isfinite(output).all():
                raise Pi05EvaluationError("actual action-out base/total invalid")
            self.events.append((base.detach().float().cpu(), output.detach().float().cpu()))
            return None

        self.handle = module.register_forward_hook(observe)

    def clear(self) -> None:
        self.events.clear()

    def close(self) -> None:
        self.handle.remove()
        self.events.clear()


class ReadoutPredictor(CrossedVideoPredictor):
    """Reuse the sealed crossed-video loader, preprocessing and LoRA runtime."""

    def __init__(self, crossed_manifest: Mapping[str, Any],
                 derived_manifest: Mapping[str, Any]) -> None:
        super().__init__(crossed_manifest, batch_size=4)
        self.derived_manifest = derived_manifest
        self.derived_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
        self.observer = ActionOutObserver(self.policy)

    def close(self) -> None:
        self.observer.close()
        self.derived_cache.clear()
        super().close()

    def _derived(self, condition_id: str, group: str) -> dict[str, torch.Tensor]:
        key = (condition_id, group)
        if key not in self.derived_cache:
            self.derived_cache[key] = load_masked_state(
                self.derived_manifest, condition_id=condition_id, group=group,
                lora=self.lora)
        return self.derived_cache[key]

    def _observe_four(self, states: list[dict[str, torch.Tensor]],
                      batch: Mapping[str, torch.Tensor], noise: torch.Tensor,
                      *, pilot: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float | None]:
        with torch.inference_mode():
            self.observer.clear()
            self.observer.enabled = True
            try:
                with self.batched.activate(states):
                    predicted = self.policy.predict_action_chunk(
                        batch, noise=noise, num_steps=10)
            finally:
                self.observer.enabled = False
            if len(self.observer.events) != 10 or predicted.shape != (4, 50, 7):
                raise Pi05EvaluationError("readout did not observe ten real flow events")
            base = torch.stack([event[0] for event in self.observer.events]).numpy()
            total = torch.stack([event[1] for event in self.observer.events]).numpy()
            normalized = predicted.float().cpu().numpy()
            actions = self.postprocess(predicted).float().cpu().numpy()
            observer_difference = None
            if pilot:
                with self.batched.activate(states):
                    unobserved = self.policy.predict_action_chunk(
                        batch, noise=noise, num_steps=10)
                observer_difference = float((unobserved.float() - predicted.float()).abs().max().item())
        return normalized, actions, base, total, observer_difference

    def predict_query(self, manifest: Mapping[str, Any], query: Mapping[str, Any],
                      output: Path, *, pilot: bool) -> dict[str, Any]:
        started = time.monotonic()
        global_id = str(query["global_task_id"])
        state_id = str(query["init_state_id"])
        condition = manifest["banks"]["C_correct"]["conditions"][global_id]["per_state"][state_id]
        condition_id = condition["condition_id"]
        if condition_id != query["C_diagonal_condition_id"] or condition_id not in self.derived_manifest["conditions"]:
            raise Pi05EvaluationError("readout query and original correct video are not paired")
        if self.cached_task != int(query["global_task_id"]):
            self.cache.clear()
            self.derived_cache.clear()
            self.cached_task = int(query["global_task_id"])
        original = self._state({"kind": "C_correct", **condition})
        states = [
            mask_state(original, self.lora, "00"),
            self._derived(condition_id, "01"),
            self._derived(condition_id, "10"),
            original,
        ]
        raw = _query_input(query)
        processed = self.processor(raw)
        batch = {key: value.expand(4, *value.shape[1:]).contiguous()
                 for key, value in processed.items() if isinstance(value, torch.Tensor)}
        if not batch or any(value.shape[0] != 4 for value in batch.values()):
            raise Pi05EvaluationError("readout frozen query preprocessing changed")
        self.policy.reset()
        noise, seeds = make_policy_noise(
            [{"init_state_id": int(query["init_state_id"]),
              "replan_index": int(query["global_replan_index"])}],
            root_seed=7, suite=query["suite"], task_id=int(query["task_id"]),
            chunk_size=50, max_action_dim=32, device=self.device,
        )
        if seeds != (int(query["policy_noise_seed"]),) or noise.shape != (1, 50, 32):
            raise Pi05EvaluationError("readout paired stateless noise changed")
        repeated_noise = noise.expand(4, -1, -1).contiguous()
        normalized, actions, base, total, observer_difference = self._observe_four(
            states, batch, repeated_noise, pilot=pilot)
        if actions.shape != (4, 50, 7) or not np.isfinite(actions).all():
            raise Pi05EvaluationError("readout environment action invalid")
        noise_np = noise[0].float().cpu().numpy()
        components, residuals = _flow_components(base, total, noise_np, normalized)
        osc = scaled_osc_actions(actions, query["osc_channel_audit"])
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            np.savez_compressed(
                handle, schema_version=np.asarray(SCHEMA),
                groups=np.asarray(GROUPS), condition_id=np.asarray(condition_id),
                normalized_actions=normalized, environment_actions=actions,
                osc_scaled_first5=osc, policy_noise=noise_np,
                action_out_base_velocity=base, action_out_total_velocity=total,
                **components,
            )
        return {
            "schema_version": SCHEMA, "query_id": query["query_id"],
            "query_path": query["query_path"], "prediction_path": str(output),
            "prediction_bytes": output.stat().st_size,
            "condition_id": condition_id, "groups": list(GROUPS), "predictions": 4,
            "policy_noise_seed": seeds[0], "flow_events": len(self.observer.events),
            "euler_max_abs_residual": residuals["euler"],
            "decomposition_11_10_max_abs_residual": residuals["11_10"],
            "decomposition_01_00_max_abs_residual": residuals["01_00"],
            "readout_off_max_abs_delta": residuals["readout_off_delta"],
            "pilot_observer_on_off_max_abs_normalized": observer_difference,
            "pilot_extra_instrument_smoke_forwards": 1 if pilot else 0,
            "seconds": time.monotonic() - started,
            "no_grad": True, "inference_mode": True,
            "predicted_actions_executed_in_environment": False,
        }
