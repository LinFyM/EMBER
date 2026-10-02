"""Registered fresh study consumers; materialization and fixed FM, no trainer or evaluator."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import traceback

import torch
from safetensors.torch import load_file

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX
from ember.operator_writer import joint_readout as readout
from ember.operator_writer.data import FormalData
from ember.operator_writer.run import build_runtime, frozen_git
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.materialization import file_record


ASSET = Path("/data1/user/ymdai/projects/EMBER")
SELF_READ_SITES = {"Q8": "model.paligemma_with_expert.gemma_expert.model.layers.8.self_attn.q_proj",
                   "V8": "model.paligemma_with_expert.gemma_expert.model.layers.8.self_attn.v_proj",
                   "action_out": "model.action_out_proj"}


from ember.operator_writer.functional_readout import fixed_flow, fm_prediction, risk


def a28(args) -> None:
    mode = {"joint": "joint", "context": "context", "self_read": "self_read", "T450": "T450_public", "conditional_read_write": readout.CONDITIONAL_MODE}[args.model]
    arm = getattr(args, "arm", None)
    arm_options = {"arm": arm} if arm is not None else {}
    spec, training, spec_path = readout.a28_source_record(mode, args.checkpoint, **arm_options)
    output = readout.study_root(mode, args.checkpoint, **arm_options) / "analysis/A28" / args.model
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
    if arm is not None:
        provenance["support_diversity_arm"] = arm
    write_json_atomic(output / "run_contract.json", {
        "study": readout.study_id(mode, args.checkpoint, **arm_options), "model": args.model, **provenance,
        "fixed_panels": file_record(readout.FIXED_PANEL / "fixed_panels.json"),
        "flow_reuse": "saved Gaussian/tau/FM_target; single FM velocity forward",
        "query_offset": 1, "flow_batch_offset": 0, "updates": 0, "environment_episodes": 0,
        "microbatch": args.microbatch, "frame_chunk": args.native_frame_chunk})
    data = None
    try:
        from ember.writer.materialization_workers import _configure_device

        _configure_device(torch.device(args.device), args.cpu_threads)
        runtime = build_runtime(args.asset_root, spec, torch.device(args.device),
                                mode if mode in ("context", "self_read", readout.CONDITIONAL_MODE) else "T")
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
                          reference, flow, rows, arm=arm)
            for teacher in panel["teachers"]:
                condition, _, _ = data.condition(runtime, task, teacher)
                with torch.no_grad():
                    capture = ({"capture_mechanism": True} if mode == readout.CONDITIONAL_MODE
                               else {"retain_native": mode == "self_read"})
                    state, native = runtime.compile(condition, frame_chunk=args.native_frame_chunk, **capture)
                native_ref = None
                if mode == "self_read":
                    native_ref = _save_native_evidence(output, task, teacher, condition[1], native, common, provenance)
                    native_records.append({"task": task, "teacher": teacher,
                                           "schema_version": "ember_self_read_native_evidence_v1", "raw": native_ref})
                    _save_readout(output, "intermediate", task, teacher,
                                  fm_prediction(runtime, native["passes"][0]["state"], batch, flow, args.microbatch),
                                  target, reference, flow, rows, native_ref=native_ref)
                if mode == readout.CONDITIONAL_MODE:
                    native_ref = _save_conditional_evidence(output, task, teacher, condition[1], native, provenance)
                    native_records.append({"task": task, "teacher": teacher,
                                           "schema_version": "ember_conditional_read_write_evidence_v1",
                                           "raw": native_ref})
                _save_readout(output, "full", task, teacher,
                              fm_prediction(runtime, state, batch, flow, args.microbatch), target,
                              reference, flow, rows, native_ref=native_ref, arm=arm)
        write_json_atomic(output / "rows.json", rows)
        complete = {"status": "complete", "rows": len(rows), "public": 4, "full": 8,
                    "updates": 0, "environment_episodes": 0, "seconds": time.monotonic() - started}
        if arm is not None:
            complete["support_diversity_arm"] = arm
        if mode in ("self_read", readout.CONDITIONAL_MODE):
            write_json_atomic(output / "native_records.json", native_records)
            complete["native_records"] = len(native_records)
            if mode == "self_read":
                complete["intermediate"] = 8
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


def _save_conditional_evidence(output, task, teacher, frame_indices, native, provenance):
    """Save the preregistered fields from this one already completed compilation."""
    indices = torch.as_tensor(frame_indices, dtype=torch.int64).cpu()
    mechanism = native["mechanism"]
    if (len(native["passes"]) != 1 or native["h"].shape != (len(indices), 50, 1024)
            or any(mechanism[key].shape != native["h"].shape for key in ("c", "d"))):
        raise ValueError("conditional A28 lost its actual full-grid single native compile")
    targets = {}
    for label, name in SELF_READ_SITES.items():
        fields = mechanism["targets"][name]
        x = native["x"][name]
        if (x.shape != (len(indices), 50, fields["A0"].shape[1])
                or fields["A0"].shape != fields["S"].shape
                or fields["B0"].shape != fields["M"].shape):
            raise ValueError("conditional passive site factor/input shape changed")
        targets[label] = {"name": name, "X": x.detach().float().cpu(),
                          **{key: fields[key].detach().float().cpu()
                             for key in ("A0", "S", "B0", "M")}}
    path = output / f"native_task{task:03d}_teacher{teacher:02d}.pt"
    torch.save({"schema_version": "ember_conditional_read_write_evidence_v1", **provenance,
                "task": task, "teacher": teacher, "frame_indices": indices,
                "H": native["h"].detach().float().cpu(),
                "c": mechanism["c"].detach().float().cpu(),
                "d": mechanism["d"].detach().float().cpu(), "targets": targets,
                "probe_seed": 1729, "tau": 1.0, "frame_stride": 5,
                "native_passes": 1, "teacher_state": "State_prompt_segment_omitted",
                "transition": "X[t-1] addressed with c[t],d[t]",
                "state_formula": "A=A0+S; B=B0+M; M_initial=0"}, path)
    return file_record(path)


def _save_readout(output, kind, task, teacher, prediction, target, reference, flow, rows, *, native_ref=None, arm=None):
    path = output / (f"public_task{task:03d}.pt" if teacher is None
                     else f"{kind}_task{task:03d}_teacher{teacher:02d}.pt")
    record = {"kind": kind, "task": task, "teacher": teacher, "risk": risk(prediction, target),
              "queries": flow["queries"], "flow_seed": flow["flow_seed"],
              "target_ref": file_record(reference), "target_field": "FM_target[..., :7]",
              "time_field": "time", "noise_field": "noise"}
    if arm is not None:
        record["support_diversity_arm"] = arm
    if native_ref is not None:
        record["native_ref"] = native_ref
    torch.save({**record, "prediction": prediction}, path)
    rows.append({**record, "raw": file_record(path)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("materialize", "a28"))
    parser.add_argument("--mode", choices=readout.MODES)
    parser.add_argument("--model", choices=("joint", "T450", "context", "self_read", "conditional_read_write"))
    parser.add_argument("--arm", choices=("C12", "D71"))
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
                                 resume_materialization_git=args.resume_materialization_git, arm=args.arm))


if __name__ == "__main__":
    main()
