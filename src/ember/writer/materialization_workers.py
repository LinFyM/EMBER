"""Resident condition compilers; the caller alone owns complete bank manifests."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from multiprocessing.util import Finalize
import os

import torch


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


_resident = None


def _initialize_worker(device_queue, ready, asset_root, config, cpu_threads, compiler_factory):
    global _resident
    _resident = compiler_factory(asset_root, config, device_queue.get(), cpu_threads)
    Finalize(None, _resident.close, exitpriority=1)
    ready.wait()


def _worker_ready():
    return True


def _compile_job(request, job):
    _resident.prepare(request)
    return _resident.compile(job)


class MaterializationWorkers:
    """Dynamic condition scheduling with one persistent spawn process per GPU."""

    def __init__(self, *, asset_root, config, devices, cpu_threads, compiler_factory):
        self.asset_root, self.config, self.devices = asset_root, config, devices
        self.cpu_threads, self.local, self.executor, self.queue = cpu_threads, None, None, None
        self.compiler_factory = compiler_factory

    def __enter__(self):
        if len(self.devices) == 1:
            self.local = self.compiler_factory(self.asset_root, self.config, self.devices[0], self.cpu_threads)
        else:
            context = mp.get_context("spawn")
            self.queue = context.Queue()
            for device in self.devices:
                self.queue.put(device)
            self.executor = ProcessPoolExecutor(max_workers=len(self.devices), mp_context=context,
                initializer=_initialize_worker,
                initargs=(self.queue, context.Barrier(len(self.devices)), self.asset_root, self.config,
                          self.cpu_threads, self.compiler_factory))
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
