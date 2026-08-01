from __future__ import annotations

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.pi05_source_checkpoint import DistributedContext, write_json_atomic
from ember.writer.as_contract import (
    build_contract,
    load_writer_config,
    parse_checkpoint_steps,
    reconcile_resume_contract,
    resolve_runtime,
    resume_step,
    writer_split_roles,
)
from ember.writer.as_sampling import TeacherVideoSchedule
from ember.writer.task_gradient import FlatParameter
from ember.writer.model import WriterModelError
from ember.writer import as_step


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT
    / "configs/pi05_as_writer_contextual_value_dual_read_full24_decay400_v1.json"
)
OLD_RECIPE_CONFIG = (
    REPO_ROOT
    / "configs/pi05_as_writer_language_axial_v6_old_recipe_v1.json"
)


def test_cvadr_full24_config_seals_architecture_and_information_wall() -> None:
    config = load_writer_config(CONFIG)
    writer = config["writer"]
    assert writer["architecture"] == "pi05_contextual_value_asymmetric_dual_read_v1"
    assert writer["teacher_state_input"] is False
    assert writer["teacher_prompt"] == "Task: {cleaned_task};\nAction: "
    assert writer["text_meta_lora_rank"] == 4
    assert writer["patch_grounding_heads"] == 8
    assert writer["patch_grounding_value"].startswith("raw_shared")
    assert writer["frame_batching_contract"].startswith("encode_one_video")
    assert writer["vl_meta_lora_rank"] == 4
    assert writer["action_meta_lora_rank"] == 4
    assert writer["action_horizon"] == 50
    assert writer["target_count"] == 38
    assert writer["public_rank"] == 16
    assert writer["frame_stride"] == 5
    assert writer["max_frames_per_encoder_call"] == 32
    assert writer["action_expert_probe"].startswith("one_forward_fixed")
    assert writer["interaction_reduction"].startswith("mean_50")
    assert writer["absolute_semantic_value"].startswith("X_f_equals")
    assert writer["semantic_core"].startswith("mean_X_plus")
    assert writer["semantic_core_frame_order"].startswith("none_strict")
    assert writer["semantic_core_blocks"] == 2
    assert writer["program_attention"].startswith("interval_local")
    assert writer["program_blocks"] == 2
    assert writer["program_memory_path"].startswith("single_causal")
    assert writer["program_value_path"].startswith("same_single_contextual")
    assert writer["program_grid"].startswith("outgoing_native_Action")
    assert writer["program_terminal_policy"].startswith("F_minus_1")
    assert writer["program_coordinate_reader"].startswith("38x16_target_rank")
    assert writer["core_reader"].startswith("38_target_only")
    assert writer["coordinate_mixer"] == "none"
    assert writer["factor_hidden_width"] == 256
    assert writer_split_roles(config) == ("train",)
    conditioning = config["conditioning_training"]
    assert conditioning["teacher_videos_per_task_visit"] == 1
    assert (
        conditioning["logical_pair_batch"]
        == "per_task_action_batch"
    )
    assert conditioning["tasks_per_rank_per_optimizer_update"] == 6
    assert conditioning["global_tasks_per_optimizer_update"] == 24
    assert conditioning["action_video_assignment"] == "all_actions_share_single_video_lora"
    assert conditioning["pair_loss_reduction"] == (
        "mean_within_task_then_equal_mean_over_24_tasks"
    )
    assert conditioning["policy_noise_contract"].startswith("one independent")
    assert conditioning["ddp_gradient_sync"].startswith("none_during")
    assert conditioning["gradient_composition"] == (
        "exact_raw_equal_weight_full24_mean_without_projection"
    )
    assert "no_conflict_fallback" not in conditioning
    assert "exact normalized-progress strata" in config["data"][
        "action_query_sampling"
    ]
    assert config["information_wall"]["test_actions_read"] == 0
    assert config["information_wall"]["test_video_values_read"] == 0
    assert "state" in config["information_wall"]["writer_forbidden_inputs"]
    assert config["profile_defaults"]["expected_world_size"] == 4
    assert config["profile_defaults"]["status"] == (
        "pending_cvadr_live_105_frame_profile"
    )
    assert config["profile_evidence"]["status"] == (
        "pending_cvadr_live_105_frame_profile"
    )
    assert config["profile_evidence"]["allowed_physical_gpu_ids"] == [4, 5, 6, 7]
    assert config["profile_evidence"]["primary_candidate"][
        "per_task_action_batch_size"
    ] == 20
    assert config["profile_evidence"]["oom_fallback_only"][
        "per_task_action_batch_size"
    ] == 16
    assert config["profile_evidence"]["inference_profile"] is None
    assert config["profile_evidence"]["teacher_videos_per_task_visit"] == 1
    assert config["specificity_gate"]["status"] == "pending_absolute_gate"
    assert config["optimization"]["scheduler"]["decay_steps"] == 400
    assert config["data"]["teacher_video_seed"] == 20260722
    assert config["profile_evidence"]["formal_teacher_video_seed_after_profile_seal"] == 20260722
    assert config["formal_run"]["status"] == "pending_profile"
    assert config["formal_run"]["launch_state"].startswith("blocked_until")
    assert config["profile_evidence"]["selected"] is None
    assert config["profile_evidence"]["exact_resume_smoke"] is None
    assert config["formal_run"]["total_steps"] == 400
    assert config["formal_run"]["per_rank_batch_size"] == 20
    assert config["formal_run"]["selected_stop_step"] == 200
    assert config["formal_run"]["stage_stop_steps"] == [200, 400]
    assert config["formal_run"]["segment_definition"].startswith(
        "fresh_cvadr_raw_full24"
    )
    assert "without_runtime_full_data_sha" in config["formal_run"][
        "data_integrity_check"
    ]


def test_v6_recipe_overlay_is_provenance_not_an_active_writer_path() -> None:
    with pytest.raises(WriterModelError, match="unsupported PI05 AS-Writer"):
        load_writer_config(OLD_RECIPE_CONFIG)


def test_checkpoint_schedule_and_cursor_are_fail_closed() -> None:
    assert parse_checkpoint_steps("2,4,4", 4) == (2, 4)
    assert parse_checkpoint_steps("every:2", 6) == (2, 4, 6)
    assert resume_step(Path("/tmp/step_00000004")) == 4
    with pytest.raises(WriterModelError, match="must end at total_steps"):
        parse_checkpoint_steps("2,3", 4)
    with pytest.raises(WriterModelError, match="not a step checkpoint"):
        resume_step(Path("/tmp/trainer_state.pt"))
    with pytest.raises(WriterModelError, match="must divide"):
        parse_checkpoint_steps("every:4", 6)


def test_profile_and_formal_runtime_require_four_symmetric_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_writer_config(CONFIG)
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=4,
        device=torch.device("cpu"),
        numa_node=0,
        cpu_affinity=(0,),
    )
    profile = argparse.Namespace(
        mode="profile",
        total_steps=None,
        batch_size=None,
        checkpoint_steps=None,
        stop_after_step=None,
        resume=None,
        skip_data_sha=False,
    )
    assert resolve_runtime(profile, config, context) == (
        3,
        20,
        (1, 2, 3),
    )
    assert profile.stop_after_step == 3
    b16 = copy.copy(profile)
    b16.batch_size = 16
    b16.stop_after_step = None
    assert resolve_runtime(b16, config, context) == (3, 16, (1, 2, 3))
    invalid_batch = copy.copy(profile)
    invalid_batch.batch_size = 19
    invalid_batch.stop_after_step = None
    with pytest.raises(WriterModelError, match="hardware-friendly"):
        resolve_runtime(invalid_batch, config, context)
    oversized_batch = copy.copy(profile)
    oversized_batch.batch_size = 40
    oversized_batch.stop_after_step = None
    with pytest.raises(WriterModelError, match="hardware-friendly"):
        resolve_runtime(oversized_batch, config, context)
    wrong_world = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        numa_node=0,
        cpu_affinity=(0,),
    )
    with pytest.raises(WriterModelError, match="exactly 4"):
        resolve_runtime(profile, config, wrong_world)
    formal = argparse.Namespace(
        mode="formal",
        total_steps=None,
        batch_size=None,
        checkpoint_steps=None,
        stop_after_step=None,
        resume=None,
        skip_data_sha=False,
    )
    unsealed = copy.deepcopy(config)
    unsealed["formal_run"]["status"] = "pending_live_profile"
    with pytest.raises(WriterModelError, match="not sealed"):
        resolve_runtime(formal, unsealed, context)
    sealed = copy.deepcopy(config)
    sealed["formal_run"]["status"] = "sealed"
    monkeypatch.setattr(
        "ember.writer.as_contract.git_state",
        lambda _root: {
            "dirty_paths": ["task-scoped-change"],
            "commit": "local",
            "origin_main": "remote",
        },
    )
    with pytest.raises(WriterModelError, match="clean worktree"):
        resolve_runtime(formal, sealed, context)
    monkeypatch.setattr(
        "ember.writer.as_contract.git_state",
        lambda _root: {
            "dirty_paths": [],
            "commit": "local",
            "origin_main": "remote",
        },
    )
    with pytest.raises(WriterModelError, match="must be pushed"):
        resolve_runtime(formal, sealed, context)
    monkeypatch.setattr(
        "ember.writer.as_contract.git_state",
        lambda _root: {
            "dirty_paths": [],
            "commit": "same",
            "origin_main": "same",
        },
    )
    formal.skip_data_sha = True
    assert resolve_runtime(formal, sealed, context) == (
        400,
        20,
        tuple(range(25, 401, 25)),
    )
    assert formal.stop_after_step == 200


def test_cvadr_launch_contract_records_raw_mean_collectives_not_ddp_accumulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_writer_config(CONFIG)
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=4,
        device=torch.device("cuda:0"),
        numa_node=1,
        cpu_affinity=(48,),
    )
    monkeypatch.setattr(
        "ember.writer.as_contract.git_state",
        lambda _root: {"branch": "main", "commit": "a" * 40},
    )

    def gather_topology(output: list[object], local: object) -> None:
        output[:] = [local, local, local, local]

    monkeypatch.setattr(
        "ember.writer.as_contract.dist.all_gather_object", gather_topology
    )
    args = argparse.Namespace(mode="profile", config=CONFIG, num_workers=2)
    contract = build_contract(
        args=args,
        config=config,
        context=context,
        source={},
        tokenizer={},
        video_data={"sampled_frame_cost_sha256": "b" * 64},
        data_validation={},
        task_ids=tuple(range(24)),
        trainable={},
        total_steps=3,
        batch_size=20,
        batch_cycle=(20,),
        checkpoint_steps=(1, 2, 3),
        initialization={},
    )
    runtime = contract["runtime"]
    assert runtime["optimizer_gradient_accumulation"] is False
    assert runtime["loss_reduction"] == (
        "mean_within_each_task_then_equal_mean_across_all_tasks"
    )
    assert runtime["task_gradients_per_rank_per_macro"] == 6
    assert runtime["global_task_gradients_per_macro"] == 24
    assert runtime["distributed_full_task_gradient_matrix_materialized"] is False
    assert runtime["gradient_task_id_allgathers_per_macro"] == 1
    assert runtime["gradient_gram_chunk_elements"] == 1_048_576
    assert (
        runtime["gradient_gram_chunk_allgathers_per_macro"]
        == "runtime_enumerated_from_parameter_block_layout"
    )
    assert runtime["gradient_composition"] == (
        "exact_raw_equal_weight_full24_mean_without_projection"
    )
    assert runtime["gradient_projection"] == "none"
    assert "gradient_direction_allreduces_per_macro" not in runtime
    assert "gradient_weight_broadcasts_per_macro" not in runtime
    assert runtime["single_video_gradient_direction_sketch"].startswith(
        "fixed_countsketch_32"
    )
    assert runtime["diagnostic_tensor_allgathers_per_macro"] == 1
    assert runtime["ddp_no_sync_microtasks_per_macro"] == 0
    assert runtime["ddp_gradient_synchronizations_per_macro"] == 0


def test_single_video_schedule_is_reproducible_and_cycle_complete() -> None:
    schedule = TeacherVideoSchedule(
        task_ids=(3, 7),
        demo_indices=tuple(range(50)),
        seed=29,
    )
    first = schedule.demo_for_task_visit(3, 7)
    replay = TeacherVideoSchedule(
        task_ids=(3, 7),
        demo_indices=tuple(range(50)),
        seed=29,
    ).demo_for_task_visit(3, 7)
    cycle = [schedule.demo_for_task_visit(3, visit) for visit in range(50)]
    assert first == replay
    assert len(set(cycle)) == 50
    with pytest.raises(WriterModelError, match="outside the schedule"):
        schedule.demo_for_task_visit(99, 7)


def test_code_compatible_resume_allows_only_recorded_commit_change(
    tmp_path: Path,
) -> None:
    existing = {
        "schema_version": "contract",
        "git": {"branch": "main", "commit": "old"},
        "runtime": {"selected_stop_step": 500, "total_steps": 1200},
    }
    write_json_atomic(tmp_path / "run_contract.json", existing)
    args = argparse.Namespace(
        output_dir=tmp_path,
        resume=tmp_path / "checkpoints/step_00000500",
        allow_contract_compatible_code_resume=True,
    )
    candidate = {**existing, "git": {"branch": "main", "commit": "new"}}
    assert reconcile_resume_contract(args, candidate) == existing
    changed = {
        **candidate,
        "runtime": {"selected_stop_step": 500, "total_steps": 1300},
    }
    with pytest.raises(WriterModelError, match="scientific contract"):
        reconcile_resume_contract(args, changed)


def test_resume_allows_only_monotonic_stage_extension(tmp_path: Path) -> None:
    existing = {
        "schema_version": "contract",
        "git": {"branch": "main", "commit": "same"},
        "runtime": {"selected_stop_step": 200, "total_steps": 2400},
    }
    write_json_atomic(tmp_path / "run_contract.json", existing)
    args = argparse.Namespace(
        output_dir=tmp_path,
        resume=tmp_path / "checkpoints/step_00000200",
        allow_contract_compatible_code_resume=False,
    )
    extended = {
        **existing,
        "runtime": {"selected_stop_step": 400, "total_steps": 2400},
    }
    assert reconcile_resume_contract(args, extended) == existing
    args.allow_contract_compatible_code_resume = True
    code_extended = {
        **extended,
        "git": {"branch": "main", "commit": "new"},
    }
    assert reconcile_resume_contract(args, code_extended) == existing

    shortened = {
        **existing,
        "runtime": {"selected_stop_step": 100, "total_steps": 2400},
    }
    with pytest.raises(WriterModelError, match="cannot shorten"):
        reconcile_resume_contract(args, shortened)

    changed_axis = {
        **existing,
        "runtime": {"selected_stop_step": 400, "total_steps": 2600},
    }
    with pytest.raises(WriterModelError, match="scientific contract"):
        reconcile_resume_contract(args, changed_axis)


def test_retired_writer_configs_are_not_active() -> None:
    for name in (
        "writer_cold_start_v1.json",
        "pi05_as_writer_v1.json",
        "pi05_as_writer_v3_normal_only.json",
        "pi05_as_writer_recenter.json",
        "pi05_as_writer_core_program.json",
    ):
        assert not (REPO_ROOT / "configs" / name).exists()
    with pytest.raises(WriterModelError, match="unsupported PI05 AS-Writer"):
        load_writer_config(REPO_ROOT / "configs/pi05_as_writer_v2.json")


def test_raw_full_task_step_collects_task_gradients_and_updates_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_ids = (10, 20, 30, 40, 50, 60)

    class Sampler:
        per_rank_batch_size = 2
        task_video_costs = {
            task_id: {0: 10 + task_id} for task_id in task_ids
        }

        @staticmethod
        def task_visit_for_step(step: int, microtask: int) -> tuple[int, int]:
            return task_ids[microtask], step

        @staticmethod
        def batch_size_for_step(_step: int) -> int:
            return 2

    class Schedule:
        @staticmethod
        def demo_for_task_visit(_task_id: int, _visit: int) -> int:
            return 0

    class Scheduler:
        steps = 0

        def step(self) -> None:
            self.steps += 1

    writer = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        writer.weight.fill_(1.0)
    scheduler = Scheduler()
    calls: list[int] = []

    def fake_pack(_runtime, *, task_id, teacher_demo, action_batch_size):
        return (task_id,), {
            "teacher_demo_index": teacher_demo,
            "teacher_video_sampled_frames": 10 + task_id,
            "unique_teacher_video_conditions": 1,
            "actions_per_video": action_batch_size,
        }

    def fake_differentiate(
        _runtime,
        packed,
        _policy_batch,
        flat_gradient,
        policy_rng_seed,
    ):
        assert policy_rng_seed is None
        calls.append(int(packed[0]))
        flat_gradient.fill_(1.0)
        return (
            torch.tensor(float(packed[0])),
            {"ok": True},
            flat_gradient,
        )

    captured: dict[str, object] = {}

    def fake_metrics(_runtime, **kwargs):
        captured.update(kwargs)
        return {"optimizer_step": int(kwargs["step"]) + 1}

    monkeypatch.setattr(as_step, "_pack_raw_conditions", fake_pack)
    monkeypatch.setattr(as_step, "_differentiate_conditions", fake_differentiate)
    monkeypatch.setattr(as_step, "_step_metrics", fake_metrics)
    runtime = SimpleNamespace(
        optimizer=torch.optim.SGD(writer.parameters(), lr=0.1),
        scheduler=scheduler,
        config={
            "conditioning_training": {
                "method": (
                    "raw_task_complete_single_video_multi_action_"
                    "positive_functional_loss"
                )
            },
            "optimization": {
                "seed": 7,
                "optimizer": {"gradient_clip_norm": 1.0},
            },
        },
        context=SimpleNamespace(device=torch.device("cpu"), world_size=1, rank=0),
        task_ids=task_ids,
        tasks_per_rank_per_update=6,
        iterator=iter(
            {"task_id": torch.tensor([task_id, task_id])}
            for task_id in task_ids
        ),
        sampler=Sampler(),
        video_schedule=Schedule(),
        videos_per_task_visit=1,
        processor=SimpleNamespace(training_batch=lambda batch: batch),
        writer=writer,
        gradient_layout=(
            FlatParameter(
                name="factor_heads.weight",
                parameter=writer.weight,
                start=0,
                stop=1,
                block="factor",
            ),
        ),
        policy=torch.nn.Identity(),
    )
    row = as_step.run_writer_step(runtime, step=0, started=0.0)
    assert row["optimizer_step"] == 1
    assert calls == list(task_ids)
    assert writer.weight.item() == pytest.approx(0.9)
    assert scheduler.steps == 1
    assert len(captured["records"]) == 6
    composition = captured["gradient_composition"]
    assert composition["schema_version"] == "ember_raw_full_task_gradient_v1"
    assert composition["raw_mean_to_average_task_energy_ratio"] == pytest.approx(1.0)
    assert "projected_gradient_gram" not in composition


def test_single_video_diagnostics_keep_all_task_losses_and_gradient_observables() -> None:
    runtime = SimpleNamespace(
        context=SimpleNamespace(device=torch.device("cpu"), world_size=1),
    )
    records = [
        {"task_id": 20, "loss": 0.2},
        {"task_id": 10, "loss": 0.1},
    ]
    assignments = [
        {"task_id": 10, "teacher_demo_index": 3, "teacher_video_sampled_frames": 9},
        {"task_id": 20, "teacher_demo_index": 4, "teacher_video_sampled_frames": 11},
    ]
    composition = {
        "task_ids": [10, 20],
        "raw_gradient_gram": [[4.0, -1.0], [-1.0, 9.0]],
        "raw_candidate_task_dots": [1.5, 4.0],
    }
    rows = as_step._global_single_video_diagnostics(
        runtime,
        records,
        assignments,
        composition,
        {
            "semantic_frontend": torch.tensor(
                [[20.0, 21.0], [10.0, 11.0]]
            ),
            "program": torch.tensor([[22.0, 23.0], [12.0, 13.0]]),
        },
    )
    assert [row["task_id"] for row in rows] == [10, 20]
    assert [row["functional_action_loss"] for row in rows] == pytest.approx(
        [0.1, 0.2]
    )
    assert [row["raw_task_gradient_norm"] for row in rows] == pytest.approx(
        [2.0, 3.0]
    )
    assert rows[0]["teacher_demo_index"] == 3
    assert rows[1]["raw_task_dot_candidate_direction"] == 4.0
    assert "projected_task_dot_candidate_direction" not in rows[1]
    assert rows[0]["raw_task_gradient_direction_sketch"] == {
        "program": [12.0, 13.0],
        "semantic_frontend": [10.0, 11.0],
    }
