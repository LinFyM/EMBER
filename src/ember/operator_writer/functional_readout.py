"""Saved A28 sample and frozen official FM consumers shared by registered readouts."""
from __future__ import annotations
from pathlib import Path
import torch
from ember.lora import validate_lora_state
from ember.operator_writer import joint_readout as readout
from ember.writer.function_credit import FlowSample, NativeFlowPrediction
from ember.writer.runtime import autocast

def fixed_flow(task: int, panel: dict) -> tuple[dict, Path]:
    path = readout.FIXED_PANEL / f"task{task:03d}_A_query_flow_target.pt"
    flow = torch.load(path, map_location="cpu", weights_only=False)
    shapes = {"action": (28, 50, 7), "noise": (28, 50, 32),
              "time": (28,), "FM_target": (28, 50, 32)}
    if (tuple(panel["teachers"]) != readout.TEACHERS[task] or flow["queries"] != panel["A"]
            or flow["flow_seed"] != panel["A_flow_seed"]
            or len(flow["queries"]) != 28
            or set(query["demo"] for query in flow["queries"]) & set(panel["teachers"])
            or any(tuple(flow[key].shape) != shape or not torch.isfinite(flow[key]).all()
                   for key, shape in shapes.items())):
        raise ValueError("fixed A28 query/Gaussian/tau/target source changed")
    return flow, path


@torch.no_grad()
def fm_prediction(runtime, state: dict, batch: dict, flow: dict, microbatch: int) -> torch.Tensor:
    """Read the original saved FM sample; never call the 10-step action sampler."""
    from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    validate_lora_state(state, runtime.lora)
    runtime.restore_identity()
    owner = NativeFlowPrediction(runtime.policy)
    predictions = []
    for start in range(0, 28, microbatch):
        stop = min(start + microbatch, 28)
        sliced = {key: (value[start:stop] if isinstance(value, torch.Tensor)
                       and value.ndim and len(value) == 28 else value) for key, value in batch.items()}
        sliced[ACTION] = flow["action"][start:stop].to(runtime.device)
        images, masks = runtime.policy._preprocess_images(dict(sliced))
        sample = FlowSample((images, masks, sliced[OBS_LANGUAGE_TOKENS],
                            sliced[OBS_LANGUAGE_ATTENTION_MASK], runtime.policy.prepare_action(sliced),
                            flow["noise"][start:stop].to(runtime.device),
                            flow["time"][start:stop].to(runtime.device)),
                           flow["FM_target"][start:stop].to(runtime.device), 7)
        with autocast(runtime.device):
            prediction = torch.func.functional_call(owner, {"policy." + key: value
                                                            for key, value in state.items()},
                                                    (sample,), strict=False)
        predictions.append(prediction[..., :7].detach().float().cpu())
    result = torch.cat(predictions)
    if result.shape != (28, 50, 7) or not torch.isfinite(result).all():
        raise ValueError("fixed A28 FM consumer lost finite 28x50x7 velocity")
    return result


def risk(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    error = (prediction.float() - target.float()).square()
    return {"full50": float(error.mean()), "first5": float(error[:, :5].mean()),
            "full50_motion6": float(error[..., :6].mean()),
            "full50_gripper1": float(error[..., 6].mean()),
            "first5_motion6": float(error[:, :5, :6].mean()),
            "first5_gripper1": float(error[:, :5, 6].mean()),
            "per_query_full50": error.mean(dim=(1, 2)).tolist(),
            "per_query_first5": error[:, :5].mean(dim=(1, 2)).tolist()}
