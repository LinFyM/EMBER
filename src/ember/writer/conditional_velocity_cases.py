"""Two train-only canonical episodes for the compiled conditional-velocity LoRA."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from ember.lora import copy_task_lora_state_, task_lora_state_dict
from ember.pi05_assets import prepare_libero_config
from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_evaluation import rollout_shard
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.conditional_velocity_engineering import build_runtime, contract, require_frozen
from ember.writer.data import RawTeacherVideoStore
from ember.writer.learning_data import load_learning_tasks


SEEN_REFERENCE = Path(
    "/data0/user/ymdai/ember_runs/learned_initial_content_causality_20260926"
    "/evaluation/complete/S0_630_seen_correct/run_contract.json"
)
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _case_contract(reference: dict, root: Path, task_id: int):
    suite, local = SUITES[task_id // 10], task_id % 10
    tasks = [row for row in reference["tasks"] if (row["suite"], row["task_id"]) == (suite, local)]
    if reference["role"] != "development_train" or len(tasks) != 1 or tasks[0]["split_role"] != "train":
        raise ValueError("canonical velocity case crosses train-only task authority")
    task = {**tasks[0], "init_state_ids": [0]}
    capture = {"mode": "compact", "full_conditions": ([{"suite": suite,
              "task_id": local, "init_state_id": 0}] if task_id == 2 else []),
               "trajectory_root": str(root / "engineering/cases/trajectories"),
               "passive_trace": {"trace_root": str(root / "engineering/cases/continuous_traces")}}
    contract = {"environment": reference["environment"], "policy": reference["policy"],
                "rng": reference["rng"], "libero_paths": reference["libero_paths"],
                "parallel": {"envs_per_replica": 1},
                "diagnostic_occupancy_capture": capture,
                "diagnostic_stage_predicates": {"full_conditions_only": False},
                "adapter": None, "role": "development_train"}
    return contract, task


def _runtime(args, spec, training_reference, seen):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(spec["flow_seed"])
    runtime = build_runtime(args.asset_root, training_reference, device,
                            evaluation_policy=seen["policy"])
    if Path(runtime.source["model_path"]) != Path(seen["model"]["model_path"]):
        raise ValueError("conditional velocity and evaluator source checkpoints differ")
    runtime.state.load_state_dict(load_file(str(args.checkpoint / "ecp.safetensors"), device="cuda:0"),
                                  strict=True)
    runtime.state.eval()
    return runtime


def _run_case(runtime, identity, videos, tasks, pool, seen, root, checkpoint, task_id):
    path = root / "engineering/cases" / f"global_{task_id:03d}.json"
    if path.exists():
        raise ValueError("canonical case already exists; refusing duplicate episode")
    case_contract, task = _case_contract(seen, root, task_id)
    started = time.perf_counter()
    # Teaching native reads always use the frozen source identity, never the
    # previously deployed case's LoRA.
    copy_task_lora_state_(runtime.policy, identity, runtime.lora)
    condition, raw_frames = runtime.condition(videos, task_id, 0, tasks[task_id].authority.language)
    with torch.no_grad():
        generated, coefficient = runtime.compile(condition)
    copy_task_lora_state_(runtime.policy, generated, runtime.lora)
    compile_seconds = time.perf_counter() - started
    envs, init_states = pool.switch(task)
    rows = rollout_shard(
        envs=envs, init_states=init_states, task=task, state_ids=[0],
        contract=case_contract, policy=runtime.policy, preprocess=runtime.processor,
        postprocess=runtime.processor.unnormalize_action,
    )
    if len(rows) != 1:
        raise ValueError("canonical evaluator returned wrong episode count")
    row = rows[0]
    trace = row["continuous_control_trace"]["trace"]
    trajectory = row["occupancy_trajectory"]
    if (trace["steps"] != row["steps"] or trace["samples"] != row["steps"] + 1
            or not Path(trace["path"]).is_file()
            or trajectory["capture_level"] != ("full" if task_id == 2 else "compact")):
        raise ValueError("canonical velocity action/state T+1 capture changed")
    row["conditional_velocity_engineering"] = {
        "global_task": task_id, "teacher_demo": 0, "raw_frames": raw_frames,
        "sampled_frames": len(condition[0]), "compile_seconds": compile_seconds,
        "wall_seconds_including_compile": time.perf_counter() - started,
        "coefficient_norm": float(coefficient.detach().float().norm()),
        "checkpoint": str(checkpoint), "scientific_qualification": False,
    }
    write_json_atomic(path, row)
    return {"path": str(path), "task": task_id, "steps": row["steps"],
            "wall_seconds": row["conditional_velocity_engineering"]["wall_seconds_including_compile"],
            "trace": trace, "capture_level": trajectory["capture_level"]}


def cases(args):
    spec, training_reference = contract()
    root = Path(spec["run_root"])
    if args.output != root or args.checkpoint is None:
        raise ValueError("canonical cases require registered study root and checkpoint")
    require_frozen(root)
    seen = read_json(SEEN_REFERENCE)
    if (seen["policy"]["num_inference_steps"] != 10
            or seen["policy"]["replan_steps"] != 5
            or seen["environment"]["render_resolution"] != 256
            or seen["rng"]["inference_seed"] != 7):
        raise ValueError("canonical train-only evaluator contract changed")
    runtime = _runtime(args, spec, training_reference, seen)
    identity = task_lora_state_dict(runtime.policy, clone=True)
    tasks = load_learning_tasks(args.asset_root, (2, 32), role="train",
                                protocol_path=spec["protocol"])
    videos = RawTeacherVideoStore(tuple(task.authority for task in tasks.values()),
                                  frame_stride=5, camera_view="agentview")
    os.environ["EMBER_LIBERO_ASSETS_ROOT"] = seen["libero_paths"]["assets"]
    installed = prepare_libero_config(root / "engineering/libero_config")
    if installed != seen["libero_paths"]:
        raise ValueError("canonical LIBERO assets changed")
    os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                      MUJOCO_EGL_DEVICE_ID=str(args.physical_gpu))
    first_contract, _ = _case_contract(seen, root, 2)
    pool = PersistentTaskEnvironmentPool(first_contract, physical_gpu_id=args.physical_gpu)
    results = []
    try:
        for task_id in (2, 32):
            results.append(_run_case(runtime, identity, videos, tasks, pool, seen, root,
                                     args.checkpoint, task_id))
    finally:
        pool.close()
        videos.close()
    write_json_atomic(root / "engineering/interface_cases.json", {
        "schema_version": "ember_conditional_velocity_interface_cases_v1",
        "status": "complete", "cases": results, "count": 2,
        "scientific_qualification": False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=Path("/data1/user/ymdai/projects/EMBER"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    cases(parser.parse_args())


if __name__ == "__main__":
    main()
