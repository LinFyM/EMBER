"""One longest fit-video Reader profile and six canonical train-only episodes."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from ember.pi05_assets import prepare_libero_config
from ember.pi05_eval.environment_pool import PersistentTaskEnvironmentPool
from ember.pi05_evaluation import rollout_shard
from ember.pi05_source_checkpoint import read_json, write_json_atomic
from ember.writer.learning_data import WriterTrainingData
from ember.writer.native_reader_engineering import (_contract, _job, build_runtime,
                                                    require_frozen_data1)


SEEN_REFERENCE = Path(
    "/data0/user/ymdai/ember_runs/learned_initial_content_causality_20260926"
    "/evaluation/complete/S0_630_seen_correct/run_contract.json"
)
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _runtime(args, config, *, evaluation_policy=None):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(int(config["optimization"]["seed"]))
    torch.cuda.manual_seed(int(config["optimization"]["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    runtime = build_runtime(args.asset_root, config, device, args.mode,
                            evaluation_policy=evaluation_policy)
    state_path = args.checkpoint / "ecp.safetensors"
    if not state_path.is_file():
        raise ValueError("Reader cases require one full engineering checkpoint")
    runtime.state.load_state_dict(load_file(str(state_path), device="cuda:0"), strict=True)
    runtime.state.eval()
    return runtime


def profile(args) -> None:
    spec, config = _contract()
    require_frozen_data1(spec, output=args.output, exact_root=True)
    if args.mode != "R_V":
        raise ValueError("the single registered longest-video profile is R_V")
    data = WriterTrainingData(args.asset_root, config["data"], camera_view="agentview",
                              planned_updates=630, use_videos=True)
    try:
        longest = max(
            (data.videos.frame_counts(task, demo)[1],
             data.videos.frame_counts(task, demo)[0], task, demo)
            for task in spec["fit_task_ids"] for demo in range(46)
        )
        sampled, raw, task, demo = longest
        draw = next(row for _ in range(630) for row in data.next_iteration()
                    if row["task"] == task and row["video_demos"] == (demo,))
        runtime = _runtime(args, config)
        runtime.state.train()
        runtime.state.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(runtime.device)
        started = time.perf_counter()
        result = _job(runtime, data, draw, config, microbatch=args.microbatch,
                      condition_demo=demo)
        torch.cuda.synchronize(runtime.device)
        seconds = time.perf_counter() - started
        record = {"schema_version": "ember_native_reader_profile_v1",
                  "status": "complete", "mode": "R_V", "task": task,
                  "teacher_demo": demo, "raw_frames": raw, "stride5_frames": sampled,
                  "queries": result["queries"], "seconds": seconds,
                  "memory_seconds": result["memory_seconds"],
                  "peak_allocated_gib": torch.cuda.max_memory_allocated(runtime.device) / 2**30,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved(runtime.device) / 2**30,
                  "source_trainable": sum(p.numel() for p in runtime.policy.parameters()
                                          if p.requires_grad),
                  "reader_grad_norm": sum(float(p.grad.float().square().sum())
                                          for p in runtime.state.reader.parameters()
                                          if p.grad is not None) ** 0.5,
                  "checkpoint": str(args.checkpoint), "scientific_qualification": False}
        write_json_atomic(args.output / "engineering/profile.json", record)
    finally:
        data.close()


def _case_contract(reference: dict, args, *, mode: str, global_task: int) -> tuple[dict, dict]:
    suite, local = SUITES[global_task // 10], global_task % 10
    tasks = [row for row in reference["tasks"] if (row["suite"], row["task_id"]) == (suite, local)]
    if reference["role"] != "development_train" or len(tasks) != 1 or tasks[0]["split_role"] != "train":
        raise ValueError("Reader interface case is not a registered train task")
    task = {**tasks[0], "init_state_ids": [0]}
    output = args.output / "engineering" / "cases" / mode
    capture = {"mode": "compact", "full_conditions": ([{"suite": suite,
              "task_id": local, "init_state_id": 0}] if global_task == 2 else []),
               "trajectory_root": str(output / "trajectories"),
               "passive_trace": {"trace_root": str(output / "continuous_traces")}}
    contract = {"environment": reference["environment"], "policy": reference["policy"],
                "rng": reference["rng"], "libero_paths": reference["libero_paths"],
                "parallel": {"envs_per_replica": 1},
                "diagnostic_occupancy_capture": capture,
                "diagnostic_stage_predicates": {"full_conditions_only": False},
                "adapter": None, "role": "development_train"}
    return contract, task


def cases(args) -> None:
    spec, config = _contract()
    require_frozen_data1(spec, output=args.output, exact_root=True)
    if args.mode not in spec["modes"]:
        raise ValueError("unknown Reader interface mode")
    reference = read_json(SEEN_REFERENCE)
    if (reference["policy"]["num_inference_steps"] != 10
            or reference["policy"]["replan_steps"] != 5
            or reference["environment"]["render_resolution"] != 256
            or reference["rng"]["inference_seed"] != 7):
        raise ValueError("canonical seen-task evaluator contract changed")
    runtime = _runtime(args, config, evaluation_policy=reference["policy"])
    if Path(runtime.source["model_path"]) != Path(reference["model"]["model_path"]):
        raise ValueError("Reader and canonical evaluator source checkpoints differ")
    data = WriterTrainingData(args.asset_root, config["data"], camera_view="agentview",
                              planned_updates=4, use_videos=args.mode == "R_V")
    os.environ["EMBER_LIBERO_ASSETS_ROOT"] = reference["libero_paths"]["assets"]
    config_path = args.output / "engineering/libero_config"
    installed_paths = prepare_libero_config(config_path)
    if installed_paths != reference["libero_paths"]:
        raise ValueError("Reader canonical LIBERO assets differ from sealed seen reference")
    os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                      MUJOCO_EGL_DEVICE_ID=str(args.physical_gpu))
    first_contract, _ = _case_contract(reference, args, mode=args.mode, global_task=2)
    pool = PersistentTaskEnvironmentPool(first_contract, physical_gpu_id=args.physical_gpu)
    try:
        for global_task in (2, 12, 22):
            path = args.output / "engineering" / "cases" / args.mode / f"global_{global_task:03d}.json"
            if path.exists():
                old = read_json(path)
                if old["native_reader_engineering"]["global_task"] != global_task:
                    raise ValueError("existing Reader interface case changed identity")
                continue
            contract, task = _case_contract(reference, args, mode=args.mode, global_task=global_task)
            started = time.perf_counter()
            with torch.no_grad():
                memory = runtime.memory(runtime.condition(data, global_task, 46))
            memory_seconds = time.perf_counter() - started
            envs, init_states = pool.switch(task)
            with runtime.state.reader.installed(runtime.policy, memory) as counts:
                rows = rollout_shard(
                    envs=envs, init_states=init_states, task=task, state_ids=[0],
                    contract=contract, policy=runtime.policy, preprocess=runtime.processor,
                    postprocess=runtime.processor.unnormalize_action,
                )
            if len(rows) != 1 or not counts["q"] or not counts["v"]:
                raise ValueError("Reader canonical episode or native target calls missing")
            row = rows[0]
            trace = row["continuous_control_trace"]["trace"]
            trajectory = row["occupancy_trajectory"]
            if (trace["steps"] != row["steps"] or trace["samples"] != row["steps"] + 1
                    or not Path(trace["path"]).is_file()
                    or trajectory["capture_level"] != ("full" if global_task == 2 else "compact")):
                raise ValueError("Reader canonical T+1 or full-case capture missing")
            row["native_reader_engineering"] = {
                "mode": args.mode, "global_task": global_task, "teacher_demo": 46,
                "memory_seconds": memory_seconds,
                "wall_seconds_including_memory": time.perf_counter() - started,
                "native_target_calls": counts, "checkpoint": str(args.checkpoint),
                "scientific_qualification": False,
            }
            write_json_atomic(path, row)
    finally:
        pool.close()
        data.close()


def summarize(args) -> None:
    spec, _ = _contract()
    cases = []
    for entry in spec["engineering"]["smoke_episodes"]:
        path = (args.output / "engineering" / "cases" / entry["mode"]
                / f"global_{entry['global_task_id']:03d}.json")
        row = read_json(path)
        if (row["native_reader_engineering"]["mode"] != entry["mode"]
                or row["native_reader_engineering"]["global_task"] != entry["global_task_id"]
                or row["init_state_id"] != 0
                or row["continuous_control_trace"]["trace"]["samples"] != row["steps"] + 1):
            raise ValueError("Reader interface case differs from registered six rows")
        cases.append({"path": str(path), "mode": entry["mode"],
                      "global_task": entry["global_task_id"], "steps": row["steps"],
                      "wall_seconds": row["native_reader_engineering"]["wall_seconds_including_memory"],
                      "full": entry["full"], "trace": row["continuous_control_trace"]["trace"]})
    write_json_atomic(args.output / "engineering/interface_cases.json", {
        "schema_version": "ember_native_reader_interface_cases_v1",
        "status": "complete", "cases": cases, "count": len(cases),
        "scientific_qualification": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("profile", "cases", "summarize"))
    parser.add_argument("--mode", choices=("R_V", "R_L"), default="R_V")
    parser.add_argument("--asset-root", type=Path, default=Path("/data1/user/ymdai/projects/EMBER"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args()
    if args.phase == "profile":
        profile(args)
    elif args.phase == "cases":
        if args.physical_gpu is None:
            raise ValueError("canonical interface cases require actual physical GPU index")
        cases(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
