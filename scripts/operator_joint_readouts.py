"""Registered fresh study consumers; materialization and fixed FM, no trainer or evaluator."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback

import torch
from safetensors.torch import load_file

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.operator_writer import joint_readout as readout
from ember.operator_writer.data import FormalData
from ember.operator_writer.run import build_runtime, frozen_git
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.function_credit import FlowSample, NativeFlowPrediction
from ember.writer.materialization import file_record
from ember.writer.runtime import autocast


ASSET = Path("/data1/user/ymdai/projects/EMBER")
SELF_READ_SITES = {"Q8": "model.paligemma_with_expert.gemma_expert.model.layers.8.self_attn.q_proj",
                   "V8": "model.paligemma_with_expert.gemma_expert.model.layers.8.self_attn.v_proj",
                   "action_out": "model.action_out_proj"}


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


def a28(args) -> None:
    mode = {"joint": "joint", "context": "context", "self_read": "self_read", "T450": "T450_public"}[args.model]
    spec, training, spec_path = readout.source_record(mode, args.checkpoint)
    output = readout.study_root(mode, args.checkpoint) / "analysis/A28" / args.model
    if output.exists():
        raise ValueError("published fixed A28 readout already exists")
    panels = read_json(readout.FIXED_PANEL / "fixed_panels.json")
    # Resolve all immutable samples before loading a GPU policy.
    samples = {task: fixed_flow(task, panels[str(task)]) for task in readout.TRAIN_TASKS}
    reading_git = frozen_git()
    output.mkdir(parents=True)
    started, rows, native_records = time.monotonic(), [], []
    provenance = {"checkpoint": str(args.checkpoint.resolve()), "training_git": training["git"]["commit"],
                  "reading_git": reading_git, "training_spec": training["spec"], "reading_spec": file_record(spec_path)}
    write_json_atomic(output / "run_contract.json", {
        "study": readout.study_id(mode, args.checkpoint), "model": args.model, "checkpoint": str(args.checkpoint.resolve()),
        "training_git": training["git"]["commit"], "reading_git": reading_git,
        "training_spec": training["spec"], "reading_spec": file_record(spec_path),
        "fixed_panels": file_record(readout.FIXED_PANEL / "fixed_panels.json"),
        "flow_reuse": "saved Gaussian/tau/FM_target; single FM velocity forward",
        "query_offset": 1, "flow_batch_offset": 0, "updates": 0, "environment_episodes": 0,
        "microbatch": args.microbatch, "frame_chunk": args.native_frame_chunk})
    data = None
    try:
        from ember.writer.materialization_workers import _configure_device

        _configure_device(torch.device(args.device), args.cpu_threads)
        runtime = build_runtime(args.asset_root, spec, torch.device(args.device),
                                mode if mode in ("context", "self_read") else "T")
        runtime.writer.load_state_dict(load_file(str(args.checkpoint / "ecp.safetensors"),
                                                 device=args.device), strict=True)
        runtime.writer.requires_grad_(False).eval()
        runtime.policy.eval()
        data = FormalData(args.asset_root, spec, task_ids=readout.TRAIN_TASKS)
        for task in readout.TRAIN_TASKS:
            panel, (flow, reference) = panels[str(task)], samples[task]
            event = {"task": task, "teacher_demo": panel["teachers"][0], "queries": panel["A"]}
            batch = runtime.processor.training_batch(data.batch(event))
            target = flow["FM_target"][..., :7]
            common = runtime.writer.public_state()
            _save_readout(output, "public", task, None,
                          fm_prediction(runtime, common, batch, flow, args.microbatch), target,
                          reference, flow, rows)
            for teacher in panel["teachers"]:
                condition, _, _ = data.condition(runtime, task, teacher)
                with torch.no_grad():
                    state, native = runtime.compile(condition, frame_chunk=args.native_frame_chunk,
                                                    retain_native=mode == "self_read")
                native_ref = None
                if mode == "self_read":
                    native_ref = _save_native_evidence(output, task, teacher, condition[1], native, common, provenance)
                    native_records.append({"task": task, "teacher": teacher,
                                           "schema_version": "ember_self_read_native_evidence_v1", "raw": native_ref})
                    _save_readout(output, "intermediate", task, teacher,
                                  fm_prediction(runtime, native["passes"][0]["state"], batch, flow, args.microbatch),
                                  target, reference, flow, rows, native_ref=native_ref)
                _save_readout(output, "full", task, teacher,
                              fm_prediction(runtime, state, batch, flow, args.microbatch), target,
                              reference, flow, rows, native_ref=native_ref)
        write_json_atomic(output / "rows.json", rows)
        complete = {"status": "complete", "rows": len(rows), "public": 4, "full": 8,
                    "updates": 0, "environment_episodes": 0, "seconds": time.monotonic() - started}
        if mode == "self_read":
            write_json_atomic(output / "native_records.json", native_records)
            complete.update(intermediate=8, native_records=len(native_records))
        write_json_atomic(output / "completion.json", complete)
    except Exception:
        write_json_atomic(output / "failure.json", {"rows": len(rows),
                          "seconds": time.monotonic() - started, "traceback": traceback.format_exc()})
        raise
    finally:
        if data is not None:
            data.close()


def _save_native_evidence(output, task, teacher, frame_indices, native, common, provenance):
    """Observe two already computed passes at the three preregistered sites."""
    passes = native["passes"]
    indices = torch.as_tensor(frame_indices, dtype=torch.int64).cpu()
    if len(passes) != 2 or any(item["h"].shape != (len(indices), 50, 1024) for item in passes):
        raise ValueError("self-read A28 lost its two actual native passes")
    targets = {}
    for label, name in SELF_READ_SITES.items():
        address = common[name + LORA_A_SUFFIX].detach().float().cpu()
        targets[label] = {"name": name}
        for index, item in enumerate(passes):
            inputs = item["x"][name].detach().float().cpu()
            if inputs.shape != (len(indices), 50, address.shape[1]):
                raise ValueError("self-read passive native site input shape changed")
            targets[label][f"AX{index}"] = torch.nn.functional.linear(inputs, address)
            targets[label][f"M{index}"] = (item["state"][name + LORA_B_SUFFIX].detach().float().cpu()
                                          - common[name + LORA_B_SUFFIX].detach().float().cpu())
    path = output / f"native_task{task:03d}_teacher{teacher:02d}.pt"
    torch.save({"schema_version": "ember_self_read_native_evidence_v1", **provenance,
                "task": task, "teacher": teacher, "frame_indices": indices,
                "H0": passes[0]["h"].detach().cpu(), "H1": passes[1]["h"].detach().cpu(),
                "targets": targets, "probe_seed": 1729, "tau": 1.0, "frame_stride": 5,
                "AX_definition": "unnormalized A X, FP32 from retained actual native inputs",
                "M_definition": "actual pass B minus the unchanged public B0",
                "state_formula": {"pass0": "beta+M0", "pass1": "beta+M1", "deployed": "beta+M1"}}, path)
    return file_record(path)


def _save_readout(output, kind, task, teacher, prediction, target, reference, flow, rows, *, native_ref=None):
    path = output / (f"public_task{task:03d}.pt" if teacher is None
                     else f"{kind}_task{task:03d}_teacher{teacher:02d}.pt")
    record = {"kind": kind, "task": task, "teacher": teacher, "risk": risk(prediction, target),
              "queries": flow["queries"], "flow_seed": flow["flow_seed"],
              "target_ref": file_record(reference), "target_field": "FM_target[..., :7]",
              "time_field": "time", "noise_field": "noise"}
    if native_ref is not None:
        record["native_ref"] = native_ref
    torch.save({**record, "prediction": prediction}, path)
    rows.append({**record, "raw": file_record(path)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("materialize", "a28"))
    parser.add_argument("--mode", choices=readout.MODES)
    parser.add_argument("--model", choices=("joint", "T450", "context", "self_read"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=ASSET)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--devices", nargs="+")
    parser.add_argument("--microbatch", type=int, choices=(7, 14, 28), default=28)
    parser.add_argument("--native-frame-chunk", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--resume-materialization-git")
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.native_frame_chunk < 1:
        parser.error("packing parameters must be positive")
    if args.phase == "a28":
        if args.model is None or args.devices is not None:
            parser.error("a28 requires one model and one device")
        a28(args)
    else:
        if args.mode is None:
            parser.error("materialize requires its registered mode")
        print(readout.materialize(args.mode, args.checkpoint, args.asset_root,
                                 devices=args.devices or [args.device],
                                 native_frame_chunk=args.native_frame_chunk, cpu_threads=args.cpu_threads,
                                 resume_materialization_git=args.resume_materialization_git))


if __name__ == "__main__":
    main()
