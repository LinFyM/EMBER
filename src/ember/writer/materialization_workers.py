"""Resident condition compilers; the caller alone owns complete bank manifests."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from multiprocessing.util import Finalize
import os

import torch
from safetensors.torch import load_file


def execution_devices(device=None, devices=None) -> tuple[torch.device, ...]:
    if device is not None and devices is not None:
        raise ValueError("choose one materialization device or one device list")
    selected = tuple(torch.device(value) for value in (devices if devices is not None else (device or "cuda:0",)))
    selected = tuple(torch.device("cuda:0") if value == torch.device("cuda") else value for value in selected)
    if (not 1 <= len(selected) <= 6 or len(set(selected)) != len(selected)
            or any(value.type not in {"cpu", "cuda"} for value in selected)
            or (len(selected) > 1 and any(value.type != "cuda" for value in selected))):
        raise ValueError("materialization needs one CPU/device or 1..6 distinct same-node CUDA devices")
    return selected


def _configure_device(device, cpu_threads):
    torch.set_num_threads(cpu_threads)
    if device.type != "cuda":
        return sorted(os.sched_getaffinity(0))
    from ember.writer.topology import bind_current_process_to_cuda_numa

    torch.cuda.set_device(device)
    affinity = bind_current_process_to_cuda_numa(torch.cuda.current_device())
    if not affinity:
        raise ValueError("materialization requires GPU-local NUMA placement")
    torch.backends.cuda.matmul.allow_tf32 = True
    return list(affinity)


class ResidentCompiler:
    """One frozen source and Writer per device, with request-local video state."""

    def __init__(self, asset_root, config, device, cpu_threads):
        from ember.writer.runtime import build_runtime

        affinity = _configure_device(device, cpu_threads)
        self.runtime = build_runtime(asset_root, config, device)
        self.device, self.request, self.checkpoint, self.store = device, None, None, None
        print(json.dumps({"materialization_worker": {"pid": os.getpid(), "device": str(device),
                                                     "cpu_affinity": affinity}}), flush=True)

    def prepare(self, request):
        from ember.writer import materialization as bank

        checkpoint, run, record, tasks, output = request
        if self.request == output:
            return
        self.close()
        runtime = self.runtime
        if not bank.source_matches(runtime.source, run["source"]):
            raise ValueError("Writer runtime uses a different frozen source checkpoint")
        if self.checkpoint != checkpoint:
            runtime.state.load_state_dict(load_file(str(checkpoint / "ecp.safetensors"), device=str(self.device)), strict=True)
            runtime.state.requires_grad_(False).eval()
            runtime.policy.eval()
            expected = torch.randn(50, 32, generator=torch.Generator().manual_seed(int(run["config"]["observer"]["probe_seed"])))
            if not torch.equal(runtime.state.probe.cpu(), expected):
                raise ValueError("checkpoint public probe differs from its declared seed")
            if runtime.lora.rank != 16 or len(runtime.lora.targets) != 38:
                raise ValueError("materialization must produce one complete 38-target rank16 LoRA")
            self.checkpoint = checkpoint
        # Only model weights persist. Adapted Z/KV/H and source coordinates are
        # local to bank._compile_condition and never retained across conditions.
        self.store = bank.RawTeacherVideoStore(tuple(task.authority for task in tasks.values()), frame_stride=5,
                                              camera_view=run["config"]["observer"].get("camera_view", "agentview"))
        self.tasks, self.output, self.record, self.request = tasks, output, record, output

    def compile(self, job):
        from ember.writer.materialization import _compile_condition

        control = job.get("control")
        extras = {"control": control, "video_task": self.tasks[control["video_global_task_id"]]} if control else {}
        return _compile_condition(self.runtime, self.store, self.tasks[job["task"]], job["demos"],
                                  self.output, self.record, **extras)

    def close(self):
        if self.store is not None:
            self.store.close()
            self.store = None
        self.request = None


_resident = None


def _initialize_worker(device_queue, ready, asset_root, config, cpu_threads):
    global _resident
    _resident = ResidentCompiler(asset_root, config, device_queue.get(), cpu_threads)
    Finalize(None, _resident.close, exitpriority=1)
    ready.wait()


def _worker_ready():
    return True


def _compile_job(request, job):
    _resident.prepare(request)
    return _resident.compile(job)


class MaterializationWorkers:
    """Dynamic condition scheduling with one persistent spawn process per GPU."""

    def __init__(self, *, asset_root, config, devices, cpu_threads):
        self.asset_root, self.config, self.devices = asset_root, config, devices
        self.cpu_threads, self.local, self.executor, self.queue = cpu_threads, None, None, None

    def __enter__(self):
        if len(self.devices) == 1:
            self.local = ResidentCompiler(self.asset_root, self.config, self.devices[0], self.cpu_threads)
        else:
            context = mp.get_context("spawn")
            self.queue = context.Queue()
            for device in self.devices:
                self.queue.put(device)
            self.executor = ProcessPoolExecutor(max_workers=len(self.devices), mp_context=context,
                initializer=_initialize_worker,
                initargs=(self.queue, context.Barrier(len(self.devices)), self.asset_root, self.config, self.cpu_threads))
            try:
                ready = [self.executor.submit(_worker_ready) for _ in self.devices]
                for future in ready:
                    future.result()
            except BaseException:
                self.__exit__()
                raise
        return self

    def compile(self, request, jobs):
        if self.local is not None:
            self.local.prepare(request)
            for job in jobs:
                yield job, self.local.compile(job)
            return
        futures = {self.executor.submit(_compile_job, request, job): job for job in jobs}
        try:
            for future in as_completed(futures):
                yield futures[future], future.result()
        finally:
            for future in futures:
                future.cancel()

    def __exit__(self, *_error):
        if self.local is not None:
            self.local.close()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        if self.queue is not None:
            self.queue.close()
            self.queue.join_thread()
