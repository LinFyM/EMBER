from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import random
import time

import h5py
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, default_collate

from ember.lora import LoRATarget, inject_task_lora, task_lora_state_dict
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import DistributedContext, canonical_hash, restore_rng
from ember.source_sft.checkpoint import load_source_sft_checkpoint, save_source_sft_checkpoint
from ember.source_sft.contract import (
    Pi05SourceSFTError, load_source_sft_config, load_training_data, resolve_runtime,
)
from ember.source_sft.sampler import HierarchicalMixedBatchSampler
from ember.source_sft.training import _one_step, _scheduler
from ember.source_sft.validation import _validation_tasks

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pi05_source_sft_aligned.json"


def test_sft_loader_future_labels_and_held_wall(tmp_path, monkeypatch):
    config = load_source_sft_config(CONFIG)
    path = tmp_path / "train.hdf5"
    actions = np.zeros((4, 7), dtype=np.float32)
    actions[:, 0] = [.1, .2, .3, .4]
    state = np.zeros((4, 6), dtype=np.float32)
    state[:, 0] = actions[:, 0].cumsum()
    with h5py.File(path, "w") as handle:
        for demo in range(50):
            group = handle.create_group(f"data/demo_{demo}")
            group.create_dataset("actions", data=actions)
            obs = group.create_group("obs")
            obs.create_dataset("ee_states", data=state)
            obs.create_dataset("gripper_states", data=np.zeros((4, 2), dtype=np.float32))
            for camera in ("agentview_rgb", "eye_in_hand_rgb"):
                obs.create_dataset(camera, data=np.zeros((4, 2, 2, 3), dtype=np.uint8))
    # Use the actual train24 selector with a tiny manifest. Held files do not
    # exist: touching them during construction would fail this test.
    import ember.source_sft.contract as module
    manifest = {"tasks": [dict(global_task_id=i, suite="tiny", task_id=i,
        split_role="train" if i < 24 else "validation" if i < 32 else "test", language="move",
        hdf5=dict(relative_path=path.name if i < 24 else f"held_{i}.hdf5",
                  bytes=path.stat().st_size, sha256="0" * 64)) for i in range(40)]}
    monkeypatch.setattr(module, "read_json", lambda _: manifest)
    args = SimpleNamespace(data_root=tmp_path, stage="development")
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    dataset, tasks, checked = load_training_data(args, config, context)
    try:
        assert len(tasks) == checked["tasks_checked"] == 24
        assert checked["full_sha256_verified"] is False
        assert len(dataset) == 24 * 50 * 3
        assert {demo for _, demo, _ in dataset.frame_index} == set(range(50))
        row = dataset[1]
        np.testing.assert_allclose(row["observation.state"][:3] + row["action"][0, :3], state[2, :3])
        assert row["action_start_index"] == 2
        assert dataset[2]["action_is_pad"].tolist() == [False] + [True] * 49
    finally:
        dataset.close()
    with pytest.raises(Pi05SourceSFTError, match="forbids validation-action reads"):
        _validation_tasks({"information_wall": config["information_wall"]}, {}, tmp_path)


def test_historical_lr_axis_is_not_compressed_to_450():
    config = load_source_sft_config(CONFIG)["optimization"]["scheduler"]
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=config["peak_lr"])
    scheduler = _scheduler(optimizer, config)
    # Values copied from the actual historical rank128 run's metrics, including
    # the absolute-step cosine transition between updates 100 and 101.
    anchors = {1: 2.97029702970297e-6, 100: .000297029702970297,
               101: .00028896253221413656, 400: .00015556941220495188,
               425: .00014135429456881541, 450: .00012727059253548267}
    for update in range(1, 451):
        if update in anchors:
            assert optimizer.param_groups[0]["lr"] == pytest.approx(anchors[update], rel=1e-12)
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(.0001267119033076614)


@pytest.mark.parametrize("world,micro,accumulation", [(1, 32, 18), (2, 64, 5), (4, 64, 3)])
def test_profile_resolves_physical_packing_without_changing_science(world, micro, accumulation):
    config = load_source_sft_config(CONFIG)
    args = SimpleNamespace(stage="development", mode="profile", total_steps=3, batch_size=micro,
                           checkpoint_steps="1,3", stop_after_step=1, gradient_accumulation_steps=None)
    assert resolve_runtime(args, config, SimpleNamespace(world_size=world)) == (3, micro, (1, 3))
    assert args.gradient_accumulation_steps == accumulation
    args.mode = "formal"
    with pytest.raises(Pi05SourceSFTError, match="physical profile is not sealed"):
        resolve_runtime(args, config, SimpleNamespace(world_size=world))


class _TinyDataset:
    def __init__(self):
        self.frame_index = tuple((task, demo, frame) for task in range(2) for demo in range(20) for frame in range(3))
        self.task_episode_rows = {task: {demo: tuple(index for index, row in enumerate(self.frame_index)
            if row[:2] == (task, demo)) for demo in range(20)} for task in range(2)}

    def __getitem__(self, row):
        task, demo, frame = self.frame_index[row]
        x = torch.tensor([row / 100, task * .3, (demo + frame + 1) / 9], dtype=torch.float64)
        return {"x": x, "action": torch.cat((x.sin(), x[:1].cos())), "task_id": task}

    def __len__(self):
        return len(self.frame_index)


class _TinyPolicy(torch.nn.Module):
    def __init__(self, stochastic=False):
        super().__init__()
        self.proj = torch.nn.Linear(3, 4, bias=False)
        self.frozen_bias = torch.nn.Parameter(torch.zeros(4))
        self.stochastic = stochastic

    def forward(self, batch):
        x = batch["x"]
        if self.stochastic:
            x = x + torch.randn_like(x) * .03
        loss = ((self.proj(x) + self.frozen_bias - batch["action"]) ** 2).mean()
        return loss, {}


def _runtime(*, world=1, rank=0, micro=4, start=0, stochastic=False):
    config = load_source_sft_config(CONFIG)
    lora = replace(load_pi05_lora_contract(ROOT / "configs/pi05_lora_rank128_aligned.json"),
                   targets=(LoRATarget("proj", 3, 4),), rank=2, alpha=2)
    torch.manual_seed(17)
    policy = inject_task_lora(_TinyPolicy(stochastic), lora).double()
    optimizer = torch.optim.AdamW(task_lora_state_dict(policy).values(),
        lr=config["optimization"]["scheduler"]["peak_lr"], betas=(.9, .95), eps=1e-8, weight_decay=1e-4)
    scheduler = _scheduler(optimizer, config["optimization"]["scheduler"])
    config["optimization"]["optimizer"]["gradient_clip_norm"] = .05
    dataset = _TinyDataset()
    accumulation = ((20 + world - 1) // world + micro - 1) // micro
    sampler = HierarchicalMixedBatchSampler(dataset, task_ids=(0, 1), logical_world_size=2,
        logical_per_rank_batch_size=10, per_rank_batch_size=micro, gradient_accumulation_steps=accumulation,
        start_step=start, stop_step=3, rank=rank, world_size=world, seed=20260723)
    iterator = iter(DataLoader(dataset, batch_sampler=sampler, generator=torch.Generator().manual_seed(123)))
    return SimpleNamespace(policy=policy, wrapped=policy, lora_contract=lora, optimizer=optimizer,
        scheduler=scheduler, sampler=sampler, iterator=iterator, config=config, resume_step=start,
        dataset=dataset, task_ids=(0, 1), context=DistributedContext(rank, rank, world, torch.device("cpu")),
        processor=SimpleNamespace(training_batch=lambda batch: {key: batch[key] for key in ("x", "action")}))


def _cpu_memory_stats(monkeypatch):
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: 0)


def _ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", world_size=3, rank=rank)
    try:
        with pytest.MonkeyPatch.context() as patch:
            _cpu_memory_stats(patch)
            runtime = _runtime(world=3, rank=rank)
            runtime.wrapped = DDP(runtime.policy, gradient_as_bucket_view=True, broadcast_buffers=False)
            calls = []
            from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
            def hook(state, bucket):
                calls.append(1)
                return allreduce_hook(state, bucket)
            runtime.wrapped.register_comm_hook(dist.group.WORLD, hook)
            reference = _runtime(micro=20)
            frozen = runtime.policy.proj.base_layer.weight.detach().clone()
            for step in range(2):
                row = _one_step(runtime, step, time.monotonic())
                reference.optimizer.zero_grad()
                batch = default_collate([reference.dataset[i] for i in reference.sampler.global_rows_for_step(step)])
                loss, _ = reference.policy(batch)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(task_lora_state_dict(reference.policy).values(), .05)
                reference.optimizer.step()
                reference.scheduler.step()
                assert row["mean_action_loss"] == pytest.approx(float(loss.detach()), rel=1e-10)
                assert row["gradient_norm_before_clip_max"] == pytest.approx(float(norm), rel=1e-10)
                for actual, expected in zip(task_lora_state_dict(runtime.policy).values(), task_lora_state_dict(reference.policy).values()):
                    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-10)
            assert len(calls) == 3  # first real backward establishes views, then one sync/update
            assert runtime.scheduler.last_epoch == 2
            assert {int(state["step"]) for state in runtime.optimizer.state.values()} == {2}
            torch.testing.assert_close(runtime.policy.proj.base_layer.weight, frozen)
            assert runtime.policy.frozen_bias.grad is None
    finally:
        dist.destroy_process_group()


def test_cpu_ddp_uneven_ranks_accumulate_one_global_mean(tmp_path):
    mp.start_processes(_ddp_worker, args=(str(tmp_path / "gloo"),), nprocs=3, join=True, start_method="spawn")


def test_checkpoint_restores_optimizer_scheduler_rng_and_logical_cursor(tmp_path, monkeypatch):
    _cpu_memory_stats(monkeypatch)
    # Exercise the real RNG schema with CPU randomness; CUDA RNG is checked by
    # the required real-device resume profile, not simulated as GPU evidence.
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda _: torch.zeros(1, dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: None)
    first = _runtime(stochastic=True)
    _one_step(first, 0, time.monotonic())
    contract = {"stage": "development", "runtime": {"total_steps": 3, "dataloader_generator_seed_base": 123}}
    checkpoint = save_source_sft_checkpoint(output_dir=tmp_path, step=1, context=first.context,
        policy=first.policy, optimizer=first.optimizer, scheduler=first.scheduler, sampler=first.sampler,
        contract=contract, mode="profile", metrics_rows=1)
    expected_random = (random.random(), np.random.random(), torch.rand(()))
    expected_row = _one_step(first, 1, time.monotonic())
    resumed = _runtime(start=1, stochastic=True)
    kwargs = dict(checkpoint=checkpoint, context=resumed.context, policy=resumed.policy,
        lora_contract=resumed.lora_contract, optimizer=resumed.optimizer, scheduler=resumed.scheduler,
        sampler=resumed.sampler, dataloader_generator_seed=123, contract_sha256=canonical_hash(contract))
    step, rng, rows = load_source_sft_checkpoint(**kwargs)
    restore_rng(rng, resumed.context)
    assert (step, rows) == (1, 1)
    assert (random.random(), np.random.random(), torch.rand(())) == expected_random
    actual_row = _one_step(resumed, 1, time.monotonic())
    for key in ("applied_lr", "next_lr", "global_action_queries", "micro_step", "mean_action_loss"):
        assert actual_row[key] == pytest.approx(expected_row[key])
    for actual, expected in zip(task_lora_state_dict(resumed.policy).values(), task_lora_state_dict(first.policy).values()):
        torch.testing.assert_close(actual, expected)
    assert resumed.scheduler.state_dict() == first.scheduler.state_dict()
    for actual, expected in zip(resumed.optimizer.state.values(), first.optimizer.state.values()):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual[key], expected[key])
    changed = _runtime(micro=5, start=1)
    with pytest.raises(Pi05SourceSFTError, match="resume state changed"):
        load_source_sft_checkpoint(**{**kwargs, "sampler": changed.sampler})
