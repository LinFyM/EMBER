from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import DistributedContext
from ember.writer.as_config import load_writer_config, parse_macro_boundaries
from ember.writer.as_contract import build_contract, inspect_video_data, publish_contract
from ember.writer.as_sampling import MixedTaskBatchSampler, TeacherVideoSchedule
from ember.writer.as_step import (
    _pack_condition,
    _task_gradient,
    accumulate_flat_gradient,
    assign_flat_gradient,
    gather_full24_records,
    parameter_layout,
    reduce_full24_gradient,
)
from ember.writer.checkpoint import (
    DEPLOYMENT_CHECKPOINT_KIND,
    load_writer_checkpoint,
    load_writer_deployment_state_,
    save_writer_checkpoint,
)
from ember.writer.data import RawTeacherVideo
from ember.writer.errors import WriterModelError
from ember.writer.live_adapter import FrozenDynamicKTaskAdapter, condition_video_offsets
from ember.writer.update_schedule import build_exposure_scheduler
import ember.writer.live_adapter as live_adapter_module
import ember.writer.as_contract as as_contract_module


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_CONFIG = (
    REPO_ROOT / "configs/pi05_writer_layer_matched_memory_program_compiler_v5.json"
)


@dataclass
class _DatasetStub:
    task_episode_rows: dict[int, dict[int, tuple[int, ...]]]
    frame_index: tuple[tuple[int, int, int], ...]


def _dataset_stub() -> _DatasetStub:
    rows: dict[int, dict[int, tuple[int, ...]]] = {}
    frame_index = []
    cursor = 0
    for task_id in range(24):
        rows[task_id] = {}
        for demo in range(50):
            rows[task_id][demo] = (cursor,)
            frame_index.append((task_id, demo, 0))
            cursor += 1
    return _DatasetStub(rows, tuple(frame_index))


def test_video_data_inspection_returns_runtime_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = []
    for task_id, lengths in ((3, [6, 11]), (7, [9, 16])):
        relative = f"task_{task_id}.hdf5"
        path = tmp_path / relative
        path.write_bytes(b"video-authority")
        rows.append(
            {
                "global_task_id": task_id,
                "hdf5": {"relative_path": relative, "bytes": path.stat().st_size},
                "demonstrations": {"episode_lengths": lengths},
            }
        )
    monkeypatch.setattr(as_contract_module, "authority_path", lambda *_: tmp_path)
    monkeypatch.setattr(
        as_contract_module, "read_json", lambda _: {"tasks": rows}
    )
    result = inspect_video_data(
        tmp_path,
        {"data": {"demo_indices": [0, 1]}, "writer": {"frame_stride": 5}},
        (3, 7),
    )
    assert result["task_ids"] == [3, 7]
    assert result["sampled_frame_counts_by_task"] == {
        "3": {"0": 2, "1": 3},
        "7": {"0": 3, "1": 4},
    }
    assert result["max_sampled_frames"] == 4


def test_lmmpc_v5_config_is_fresh_rank16_without_text_or_vl_meta_lora() -> None:
    config = load_writer_config(ACTIVE_CONFIG)
    lora = load_pi05_lora_contract(REPO_ROOT / "configs/pi05_lora_v1.json")
    assert (lora.rank, lora.alpha, lora.parameter_count) == (16, 16, 1_287_168)
    assert lora.state_tensor_count == 76
    assert config["method"]["development_initialization"] == "fresh_writer"
    assert config["writer"]["memory_token_count"] == 16
    assert config["writer"]["m2p"].startswith("same_20x16_grid")
    assert "text_meta_lora_rank" not in config["writer"]
    assert "vl_meta_lora_rank" not in config["writer"]
    assert config["writer"]["video_set_max_relative_correction"] == 0.5
    assert config["writer"]["m2p_max_relative_correction"] == 0.5
    assert config["writer"]["procedure_memory_readout"].startswith(
        "core_conditioned_policy_address_queries"
    )
    assert config["data"]["dynamic_k_max"] == 4
    assert config["optimization"]["functional_policy_microbatch_size"] == 5
    assert config["formal_run"]["status"] == "unsealed_pending_live_profile"
    assert config["formal_run"]["checkpoint_macros"] == [25, 50, 75, 100]
    assert config["optimization"]["distributed"]["fresh_world_sizes"] == list(
        range(1, 7)
    )
    assert parse_macro_boundaries([25, 50, 75, 100], 100) == (25, 50, 75, 100)
    assert parse_macro_boundaries("1,2,3", 3) == (1, 2, 3)


def test_dynamic_k_schedule_balances_each_macro_and_each_task_cycle() -> None:
    task_ids = tuple(range(24))
    schedule = TeacherVideoSchedule(
        task_ids=task_ids,
        demo_indices=range(50),
        seed=20260722,
        videos_per_visit=4,
        dynamic_k_max=4,
    )
    for macro in range(9):
        counts = [
            schedule.shot_count_for_task_visit(task_id, macro) for task_id in task_ids
        ]
        assert {k: counts.count(k) for k in range(1, 5)} == {
            1: 6,
            2: 6,
            3: 6,
            4: 6,
        }
    for task_id in task_ids:
        assert {
            schedule.shot_count_for_task_visit(task_id, macro) for macro in range(4)
        } == {1, 2, 3, 4}


def test_dynamic_k_videos_are_unique_and_cross_episode() -> None:
    schedule = TeacherVideoSchedule(
        task_ids=tuple(range(24)),
        demo_indices=range(50),
        seed=20260722,
        videos_per_visit=4,
        dynamic_k_max=4,
    )
    excluded = tuple(range(20))
    for task_id in schedule.task_ids:
        for macro in range(4):
            selected = schedule.demos_for_task_visit(task_id, macro, excluded=excluded)
            assert len(selected) == schedule.shot_count_for_task_visit(task_id, macro)
            assert len(selected) == len(set(selected))
            assert not set(selected) & set(excluded)


@pytest.mark.parametrize("world_size", range(1, 7))
def test_sampler_covers_full24_with_uneven_world_sizes(world_size: int) -> None:
    dataset = _dataset_stub()
    task_ids = tuple(range(24))
    schedule = TeacherVideoSchedule(
        task_ids=task_ids,
        demo_indices=range(50),
        seed=20260722,
        videos_per_visit=4,
        dynamic_k_max=4,
    )
    samplers = [
        MixedTaskBatchSampler(
            dataset,  # type: ignore[arg-type]
            task_ids=task_ids,
            per_rank_batch_size=20,
            start_step=0,
            stop_step=2,
            rank=rank,
            world_size=world_size,
            seed=20260721,
            tasks_per_rank_per_update=(24 + world_size - 1) // world_size,
            video_schedule=schedule,
            task_video_costs={
                task_id: {demo: 1 + task_id + demo for demo in range(50)}
                for task_id in task_ids
            },
            assignment_strategy="cost_balanced_long_first_dynamic_uneven",
        )
        for rank in range(world_size)
    ]
    for macro in range(2):
        shards = [sampler.tasks_for_step(macro) for sampler in samplers]
        assert {task for shard in shards for task in shard} == set(task_ids)
        assert sum(map(len, shards)) == 24
        assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_flat_gradient_accumulates_tasks_then_divides_once_by_24() -> None:
    module = torch.nn.Sequential(
        torch.nn.Linear(2, 3, bias=False),
        torch.nn.Linear(3, 1, bias=False),
    )
    layout = parameter_layout(module)
    flat = torch.zeros(layout[-1].stop)
    for task in range(24):
        gradients = tuple(
            torch.full_like(item.parameter, float(task + 1)) for item in layout
        )
        accumulate_flat_gradient(flat, gradients, layout)
    expected = sum(range(1, 25)) / 24
    reduce_full24_gradient(flat, world_size=1)
    assert torch.all(flat == expected)
    assign_flat_gradient(flat, layout)
    assert all(torch.all(item.parameter.grad == expected) for item in layout)


def test_flat_gradient_contract_rejects_non_full24_reduction() -> None:
    with pytest.raises(WriterModelError, match="full24"):
        reduce_full24_gradient(torch.ones(2), world_size=1, global_task_count=23)


def test_full24_task_evidence_is_gathered_and_sorted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = [{"task_id": task, "functional_loss": 1.0} for task in range(12)]
    second = [{"task_id": task, "functional_loss": 2.0} for task in range(12, 24)]

    def gather(shards: list[object], local: object) -> None:
        assert local == first
        shards[:] = [first, second]

    monkeypatch.setattr("ember.writer.as_step.dist.all_gather_object", gather)
    records = gather_full24_records(first, world_size=2, task_ids=tuple(range(24)))
    assert [row["task_id"] for row in records] == list(range(24))


def test_task_gradient_uses_functional_lora_cotangent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))

    class _Writer:
        @staticmethod
        def forward_training(*_args: object, **_kwargs: object):
            return {
                "fixed_a": torch.ones(2),
                "dynamic_b": parameter * torch.ones(2),
            }

    monkeypatch.setattr(
        "ember.writer.as_step.functional_lora_loss_gradient",
        lambda *_args, **_kwargs: (
            torch.tensor(1.0),
            {},
            {"fixed_a": torch.ones(2), "dynamic_b": torch.full((2,), 3.0)},
        ),
    )
    runtime = SimpleNamespace(
        writer=_Writer(),
        policy=torch.nn.Identity(),
        lora_contract=object(),
        context=SimpleNamespace(device=torch.device("cpu")),
        config={
            "conditioning_training": {
                "policy_flow_time_sampling_scheme": None,
                "policy_flow_noise_sampling_scheme": None,
            },
            "optimization": {"functional_policy_microbatch_size": 2},
        },
        gradient_layout=(SimpleNamespace(parameter=parameter, start=0, stop=1),),
    )
    flat = torch.zeros(1)
    _task_gradient(runtime, (None,) * 7, {}, 7, flat)
    assert flat.item() == pytest.approx(6.0)
    assert parameter.grad is None


def test_task_gradient_rejects_a_writer_without_trainable_lora_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))

    class _Writer:
        @staticmethod
        def forward_training(*_args: object, **_kwargs: object):
            return {"dynamic": torch.ones(1)}

    monkeypatch.setattr(
        "ember.writer.as_step.functional_lora_loss_gradient",
        lambda *_args, **_kwargs: (
            torch.tensor(1.0),
            {},
            {"dynamic": torch.ones(1)},
        ),
    )
    runtime = SimpleNamespace(
        writer=_Writer(),
        policy=torch.nn.Identity(),
        lora_contract=object(),
        context=SimpleNamespace(device=torch.device("cpu")),
        config={
            "conditioning_training": {
                "policy_flow_time_sampling_scheme": None,
                "policy_flow_noise_sampling_scheme": None,
            },
            "optimization": {"functional_policy_microbatch_size": 1},
        },
        gradient_layout=(SimpleNamespace(parameter=parameter, start=0, stop=1),),
    )
    with pytest.raises(WriterModelError, match="trainable output"):
        _task_gradient(runtime, (None,) * 7, {}, 7, torch.zeros(1))


def test_scheduler_applies_warmup_lr_before_first_optimizer_step() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=3e-4)
    scheduler = build_exposure_scheduler(
        optimizer,
        {
            "kind": "cosine_decay_with_warmup",
            "peak_lr": 3e-4,
            "warmup_steps": 17,
            "decay_steps": 400,
            "decay_lr": 1e-5,
        },
        400,
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4 / 17)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2 * 3e-4 / 17)


def test_ragged_offsets_remain_cpu_long_while_frames_follow_device() -> None:
    class _Store:
        @staticmethod
        def load(task_id: int, demo: int) -> RawTeacherVideo:
            del task_id
            count = demo + 2
            return RawTeacherVideo(
                frames=np.zeros((count, 3, 4, 4), dtype=np.uint8),
                frame_indices=np.arange(count, dtype=np.int64),
                raw_frame_count=count,
            )

    runtime = SimpleNamespace(
        video_store=_Store(),
        context=SimpleNamespace(device=torch.device("cpu")),
        config={"writer": {"backbone_total_frames_per_condition": 420}},
        language_tokens={
            7: (
                torch.ones((1, 3), dtype=torch.long),
                torch.ones((1, 3), dtype=torch.bool),
                torch.ones((1, 3), dtype=torch.bool),
            )
        },
    )
    packed, metrics = _pack_condition(runtime, 7, (0, 2))
    frames, indices, video_offsets, condition_offsets, *_ = packed
    assert frames.device.type == indices.device.type == "cpu"
    assert video_offsets.device.type == condition_offsets.device.type == "cpu"
    assert video_offsets.dtype == condition_offsets.dtype == torch.long
    assert video_offsets.tolist() == [0, 2, 6]
    assert condition_offsets.tolist() == [0, 2]
    assert metrics["total_sampled_frames"] == 6


def test_publish_contract_accepts_only_exact_resume_contract(tmp_path: Path) -> None:
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    contract = {"schema_version": "test", "runtime": {"world_size": 1}}
    args = argparse.Namespace(
        output_dir=tmp_path / "run", resume=None, stop_after_macro=1
    )
    publish_contract(args, context, contract)
    args.resume = tmp_path / "run/checkpoints/macro_00000001"
    publish_contract(args, context, contract)
    with pytest.raises(WriterModelError, match="contract changed"):
        publish_contract(args, context, {**contract, "changed": True})


def test_diagnostic_contract_records_frozen_head_intervention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        as_contract_module,
        "git_state",
        lambda _: {"branch": "", "commit": "a" * 40},
    )
    source_checkpoint = tmp_path / "checkpoints/macro_00000025"
    args = argparse.Namespace(
        mode="formal",
        config=tmp_path / "config.json",
        num_workers=0,
        diagnostic_fork_resume=source_checkpoint,
    )
    contract = build_contract(
        args=args,
        config={
            "authorities": {},
            "information_wall": {},
            "writer": {},
            "data": {},
            "conditioning_training": {},
            "optimization": {},
        },
        context=DistributedContext(0, 0, 1, torch.device("cpu")),
        source={},
        tokenizer={},
        video_data={},
        data_validation={},
        task_ids=tuple(range(24)),
        trainable={"writer_frozen_parameter_count": 1},
        total_macros=100,
        batch_size=20,
        checkpoint_macros=(25, 50, 75, 100),
    )
    assert contract["diagnostic_intervention"] == {
        "kind": "freeze_factor_heads_after_macro25",
        "source_checkpoint": str(source_checkpoint.resolve()),
        "frozen_module": "factor_heads",
        "unchanged_components": (
            "Program Reader K-set M2P objective data optimizer scheduler and RNG"
        ),
        "deployment_method": False,
    }


def test_hashless_checkpoint_restores_training_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ember.writer.checkpoint as checkpoint_module

    monkeypatch.setattr(
        checkpoint_module, "capture_rng", lambda context: {"rank": context.rank}
    )
    monkeypatch.setattr(checkpoint_module, "restore_rng", lambda state, context: None)
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    writer = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(writer.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    contract = {"schema_version": "launch_v1"}
    expected = writer.weight.detach().clone()
    checkpoint = save_writer_checkpoint(
        output_dir=tmp_path,
        macro=1,
        context=context,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        contract=contract,
        metrics_rows=1,
    )
    with torch.no_grad():
        writer.weight.zero_()
    macro, rows = load_writer_checkpoint(
        checkpoint=checkpoint,
        context=context,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        contract=contract,
    )
    assert (macro, rows) == (1, 1)
    assert torch.equal(writer.weight, expected)


def test_deployment_loads_only_writer_safetensors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ember.writer.checkpoint as checkpoint_module

    expected = torch.nn.Linear(3, 2)
    state_path = tmp_path / "writer.safetensors"
    save_file(
        {name: value.detach().clone() for name, value in expected.state_dict().items()},
        str(state_path),
    )
    observed = torch.nn.Linear(3, 2)
    monkeypatch.setattr(
        checkpoint_module.torch,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("deployment must not load optimizer or RNG state")
        ),
    )
    load_writer_deployment_state_(
        writer=observed,
        writer_asset={
            "kind": DEPLOYMENT_CHECKPOINT_KIND,
            "writer_state": {"path": str(state_path), "bytes": state_path.stat().st_size},
        },
        device=torch.device("cpu"),
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            observed.state_dict().values(), expected.state_dict().values(), strict=True
        )
    )


@pytest.mark.parametrize(
    ("evaluation_k", "expected"),
    ((1, [0, 1, 2, 3, 4]), (4, [0, 4, 8, 12, 16])),
)
def test_evaluation_offsets_assign_fixed_k_videos_per_condition(
    evaluation_k: int, expected: list[int]
) -> None:
    offsets = condition_video_offsets(4, evaluation_k)
    assert offsets.tolist() == expected
    assert offsets.dtype == torch.long and offsets.device.type == "cpu"


def test_live_evaluator_supplies_k4_as_one_ragged_writer_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FrozenDynamicKTaskAdapter.__new__(FrozenDynamicKTaskAdapter)
    adapter.evaluation_k = 4
    adapter._physical_lora_is_identity = True
    adapter.device = torch.device("cpu")
    adapter.policy = object()
    adapter.identity_state = {}
    adapter.lora_contract = object()
    adapter.tokenizer = lambda languages: (
        torch.zeros((len(languages), 3), dtype=torch.long),
        torch.ones((len(languages), 3), dtype=torch.bool),
        torch.ones((len(languages), 3), dtype=torch.bool),
    )
    requests = (
        {"suite": "libero_spatial", "task_id": 1, "init_state_id": 0},
        {"suite": "libero_object", "task_id": 1, "init_state_id": 0},
    )

    def episode_input(**identity: object):
        condition = 0 if identity["suite"] == "libero_spatial" else 1
        videos = tuple(
            RawTeacherVideo(
                frames=np.full((2, 3, 2, 2), condition * 4 + video, dtype=np.uint8),
                frame_indices=np.asarray((0, 5), dtype=np.int64),
                raw_frame_count=6,
            )
            for video in range(4)
        )
        return (
            {"lora_reference": f"condition-{condition}"},
            videos,
            f"task-{condition}",
            (2, 2, 2, 2),
        )

    adapter._episode_input = episode_input
    captured = {}

    def writer(
        frames: torch.Tensor,
        _indices: torch.Tensor,
        video_offsets: torch.Tensor,
        ownership: torch.Tensor,
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, torch.Tensor]:
        captured.update(frames=frames, video_offsets=video_offsets, ownership=ownership)
        return {"generated": torch.zeros((2, 1))}

    adapter.writer = writer
    monkeypatch.setattr(live_adapter_module, "validate_lora_state", lambda *_args: None)
    prepared = adapter.prepare_episodes(requests)
    assert len(prepared) == 2
    assert captured["video_offsets"].tolist() == list(range(0, 17, 2))
    assert captured["ownership"].tolist() == [0, 4, 8]


def test_evaluator_resolves_only_the_lmmpc_rank16_authority() -> None:
    import importlib

    from ember.eval_adapters import DYNAMIC_K_WRITER_KIND

    writer_lora_contract = importlib.import_module(
        "ember.pi05_eval.run_contract"
    )._writer_lora_contract
    lora = writer_lora_contract(
        SimpleNamespace(repo_root=REPO_ROOT),
        {
            "kind": DYNAMIC_K_WRITER_KIND,
            "config": {"path": str(ACTIVE_CONFIG)},
            "lora_contract": {
                "reference": "configs/pi05_lora_v1.json:76tensors:1287168parameters"
            },
        },
    )
    assert (lora.rank, lora.parameter_count, lora.state_tensor_count) == (
        16,
        1_287_168,
        76,
    )


def test_v5_generation_profile_is_sealed_from_live_evidence() -> None:
    from ember.writer.evaluation import DYNAMIC_K_GENERATION_PROFILES

    assert set(DYNAMIC_K_GENERATION_PROFILES) == {4}
    profile = DYNAMIC_K_GENERATION_PROFILES[4]
    assert profile["schema_version"] == "ember_pi05_writer_generation_profile_v2"
    assert (
        profile["evidence_path"]
        == "runs/acceptance/"
        "pi05_lmmpc_v5_k4_generation_profile_aecbce5_macro2_gpu02p7_"
        "20260818/writer_generation_profile.json"
    )
    assert profile["evidence_bytes"] == 10775
    assert profile["authority_commit"] == (
        "aecbce5b4301f98ecaafea650e099b6326c5c98d"
    )
    assert profile["profiled_writer_model_batch_sizes"] == [8, 16, 32]
    assert profile["supported_writer_model_batch_sizes"] == [8, 16, 32]
    assert profile["selected_writer_model_batch_size"] == 32
    assert profile["panel_entry_count"] == 32
    assert profile["panel_total_sampled_frames"] == 4438
    assert profile["longest_sampled_video_frames"] == 226
    assert [
        row["batch_size"] for row in profile["writer_generation_measurements"]
    ] == [8, 16, 32]
    assert all(
        row["stable"] for row in profile["writer_generation_measurements"]
    )
    assert profile["writer_generation_measurements"][2]["loras_per_second"] > (
        profile["writer_generation_measurements"][0]["loras_per_second"]
    )
    assert profile["oom_count"] == profile["nonfinite_count"] == 0
