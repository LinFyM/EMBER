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
from ember.writer.as_config import load_writer_config, parse_macro_boundaries
from ember.writer.as_contract import publish_contract
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
from ember.writer.update_schedule import build_exposure_scheduler
import ember.writer.live_adapter as live_adapter_module
from ember.writer.live_adapter import (
    FrozenDynamicKTaskAdapter,
    condition_video_offsets,
)
from ember.pi05_source_checkpoint import DistributedContext


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_rank8_task_grounded_visual_value_config_is_mechanical_and_loadable() -> None:
    config = load_writer_config(
        REPO_ROOT
        / (
            "configs/pi05_as_writer_dynamic_k_task_grounded_visual_value_"
            "rank8_v1.json"
        )
    )
    lora = load_pi05_lora_contract(
        REPO_ROOT / "configs/pi05_lora_rank8_writer_v1.json"
    )
    assert (lora.rank, lora.alpha, lora.parameter_count) == (8, 8, 643_584)
    assert lora.state_tensor_count == 76
    assert config["data"]["dynamic_k_max"] == 4
    assert config["writer"]["temporal_semantic_address"] == (
        "per_video_absolute_mean_memory_to_temporal_query_only"
    )
    assert config["writer"]["task_grounded_visual_evidence"].startswith(
        "same_joint_forward_task_query_to_raw_patch_value"
    )
    assert config["writer"]["lora_a"] == (
        "fixed_template_without_dynamic_a_head"
    )
    assert config["writer"]["lora_b_readout"].startswith(
        "four_bias_free_zero_initialized_direct_shape_family_linears"
    )
    assert config["optimization"]["distributed"]["fresh_world_sizes"] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert parse_macro_boundaries("every:25", 400) == tuple(
        range(25, 401, 25)
    )
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
            schedule.shot_count_for_task_visit(task_id, macro)
            for task_id in task_ids
        ]
        assert {k: counts.count(k) for k in range(1, 5)} == {
            1: 6,
            2: 6,
            3: 6,
            4: 6,
        }
    for task_id in task_ids:
        assert {
            schedule.shot_count_for_task_visit(task_id, macro)
            for macro in range(4)
        } == {1, 2, 3, 4}


def test_dynamic_k_videos_are_unique_and_in_action_episode_complement() -> None:
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
            selected = schedule.demos_for_task_visit(
                task_id, macro, excluded=excluded
            )
            assert len(selected) == schedule.shot_count_for_task_visit(
                task_id, macro
            )
            assert len(selected) == len(set(selected))
            assert not set(selected) & set(excluded)


@pytest.mark.parametrize("world_size", range(1, 7))
def test_dynamic_k_sampler_covers_full24_with_uneven_world_sizes(
    world_size: int,
) -> None:
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
        for sampler, shard in zip(samplers, shards, strict=True):
            for task_id in shard:
                excluded = sampler.action_demo_indices_for_task_visit(
                    task_id, macro
                )
                selected = schedule.demos_for_task_visit(
                    task_id, macro, excluded=excluded
                )
                assert len(excluded) == 20
                assert not set(selected) & set(excluded)
        costs, _, _ = samplers[0]._cost_order_for_task_cycle(macro)
        task_id = task_ids[0]
        excluded = samplers[0].action_demo_indices_for_task_visit(task_id, macro)
        videos = schedule.demos_for_task_visit(
            task_id, macro, excluded=excluded
        )
        assert costs[task_id] == 20 + sum(
            1 + task_id + demo for demo in videos
        )


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
    records = gather_full24_records(
        first,
        world_size=2,
        task_ids=tuple(range(24)),
    )
    assert [row["task_id"] for row in records] == list(range(24))


def test_task_gradient_skips_fixed_template_a_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))

    class _Writer:
        @staticmethod
        def forward_training(*_args: object, **_kwargs: object):
            return {
                "fixed_a": torch.ones(2),
                "dynamic_b": parameter * torch.ones(2),
            }, parameter * 0.0

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
                "singleton_to_full_consistency": {"weight": 0.05},
            },
            "optimization": {"functional_policy_microbatch_size": 2},
        },
        gradient_layout=(SimpleNamespace(parameter=parameter, start=0, stop=1),),
    )
    flat = torch.zeros(1)
    packed = (None, None, None, torch.tensor([0, 1]), None, None, None)
    _task_gradient(runtime, packed, {}, 7, flat)
    assert flat.item() == pytest.approx(6.0)
    assert parameter.grad is None


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
        config={"writer": {"backbone_total_frames_per_condition": 64}},
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
        output_dir=tmp_path / "run",
        resume=None,
        stop_after_macro=1,
    )
    publish_contract(args, context, contract)
    args.resume = tmp_path / "run/checkpoints/macro_00000001"
    publish_contract(args, context, contract)
    with pytest.raises(WriterModelError, match="contract changed"):
        publish_contract(args, context, {**contract, "changed": True})


def test_prepare_runtime_rejects_cross_run_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ember.writer.training import prepare_runtime

    monkeypatch.setattr(
        "ember.writer.training.load_writer_config",
        lambda _path: {
            "profile_defaults": {
                "allowed_world_sizes": [1],
                "total_macros": 2,
                "per_task_action_batch_size": 20,
                "checkpoint_macros": [2],
                "stop_after_macro": 2,
            }
        },
    )
    monkeypatch.setattr(
        "ember.writer.training.resolve_mode_config", lambda config, _mode: config
    )
    args = argparse.Namespace(
        config=tmp_path / "config.json",
        mode="profile",
        total_macros=None,
        batch_size=None,
        checkpoint_macros=None,
        stop_after_macro=None,
        resume=tmp_path / "run_a/checkpoints/macro_00000001",
        output_dir=tmp_path / "run_b",
    )
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    with pytest.raises(WriterModelError, match="another run"):
        prepare_runtime(args, context)


def test_hashless_checkpoint_restores_training_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ember.writer.checkpoint as checkpoint_module

    monkeypatch.setattr(checkpoint_module, "capture_rng", lambda context: {"rank": context.rank})
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


def test_deployment_checkpoint_loads_only_writer_safetensors(
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
    with torch.no_grad():
        observed.weight.zero_()
        observed.bias.zero_()
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
            "writer_state": {
                "path": str(state_path),
                "bytes": state_path.stat().st_size,
            },
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
        captured.update(
            frames=frames,
            video_offsets=video_offsets,
            ownership=ownership,
        )
        return {"generated": torch.zeros((2, 1))}

    adapter.writer = writer
    monkeypatch.setattr(live_adapter_module, "validate_lora_state", lambda *_args: None)
    prepared = adapter.prepare_episodes(requests)

    assert len(prepared) == 2
    assert captured["video_offsets"].tolist() == list(range(0, 17, 2))
    assert captured["ownership"].tolist() == [0, 4, 8]
    assert [int(captured["frames"][index * 2, 0, 0, 0]) for index in range(8)] == list(
        range(8)
    )
    assert [row["sampled_frames"] for row in adapter.last_generation_batch_profile()] == [
        8,
        8,
    ]


def test_dynamic_k_evaluation_request_is_not_the_legacy_writer() -> None:
    from ember.eval_adapters import DYNAMIC_K_WRITER_KIND, adapter_requests

    args = argparse.Namespace(
        source_sft_config=None,
        source_sft_checkpoint=None,
        task_expert_config=None,
        task_expert_bank_root=None,
        task_expert_step=None,
        expert_manifold_config=None,
        expert_manifold_checkpoint=None,
        expert_manifold_video_data_root=None,
        expert_manifold_video_condition=None,
        dynamic_k_writer_config=Path("config.json"),
        dynamic_k_writer_checkpoint=Path("macro_00000050"),
        dynamic_k_writer_video_data_root=Path("videos"),
        dynamic_k_writer_video_condition="correct",
    )
    assert adapter_requests(args) == (DYNAMIC_K_WRITER_KIND, False)


def test_evaluator_resolves_the_dynamic_k_rank8_lora_authority() -> None:
    import importlib

    from ember.eval_adapters import DYNAMIC_K_WRITER_KIND

    importlib.import_module("ember.pi05_eval_contract")
    writer_lora_contract = importlib.import_module(
        "ember.pi05_eval.run_contract"
    )._writer_lora_contract

    config = (
        REPO_ROOT
        / (
            "configs/pi05_as_writer_dynamic_k_task_grounded_visual_value_"
            "rank8_v1.json"
        )
    )
    lora = writer_lora_contract(
        SimpleNamespace(repo_root=REPO_ROOT),
        {
            "kind": DYNAMIC_K_WRITER_KIND,
            "config": {"path": str(config)},
            "lora_contract": {
                "reference": (
                    "configs/pi05_lora_rank8_writer_v1.json:"
                    "76tensors:643584parameters"
                )
            },
        },
    )
    assert (lora.rank, lora.parameter_count, lora.state_tensor_count) == (
        8,
        643_584,
        76,
    )


def test_task_grounded_visual_value_k1_deployment_profile_is_sealed() -> None:
    from ember.writer.evaluation import (
        DYNAMIC_K_GENERATION_BATCH_SIZE,
        DYNAMIC_K_GENERATION_PROFILES,
    )

    assert set(DYNAMIC_K_GENERATION_PROFILES) == {1}
    assert DYNAMIC_K_GENERATION_PROFILES[1] == {
        "schema": "ember_pi05_writer_generation_profile_v2",
        "path": (
            "runs/outputs/"
            "pi05_dynamic_k_task_grounded_visual_value_rank8_k1_writer_"
            "generation_profile_val8x4_correct_gpu02p1_caa2e30_macro0025_"
            "retry1_20260813/writer_generation_profile.json"
        ),
        "selected_writer_model_batch_size": DYNAMIC_K_GENERATION_BATCH_SIZE,
    }
