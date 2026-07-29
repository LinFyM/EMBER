from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.pi05_source_checkpoint import DistributedContext, write_json_atomic
from ember.writer.as_contract import (
    load_writer_config,
    parse_checkpoint_steps,
    reconcile_resume_contract,
    resolve_runtime,
    resume_step,
    writer_split_roles,
)
from ember.writer.as_sampling import TeacherVideoSchedule
from ember.writer.model import WriterModelError
from ember.writer import as_step


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/pi05_as_writer_language_axial_v6.json"
OLD_RECIPE_CONFIG = (
    REPO_ROOT
    / "configs/pi05_as_writer_language_axial_v6_old_recipe_v1.json"
)


def test_language_axial_config_seals_architecture_and_information_wall() -> None:
    config = load_writer_config(CONFIG)
    writer = config["writer"]
    assert (
        writer["architecture"]
        == "pi05_task_grounded_semantic_set_visual_transition_causal_procedure_slot_fusion_v6"
    )
    assert writer["teacher_state_input"] is False
    assert writer["teacher_prompt"] == "Task: {cleaned_task};\nAction: "
    assert writer["text_branch_input"].startswith("bos_plus_exact")
    assert "task_queried_image_position_content" in writer["multimodal_core_value"]
    assert writer["patch_grounding_heads"] == 8
    assert "no_value_projection" in writer["patch_grounding_value"]
    assert writer["frame_batching_contract"].startswith("encode_one_video")
    assert writer["text_meta_lora_rank"] == 4
    assert writer["vl_meta_lora_rank"] == 4
    assert writer["action_meta_lora_rank"] == 4
    assert writer["action_horizon"] == 50
    assert writer["query_count"] == 320
    assert writer["frame_stride"] == 5
    assert writer["max_frames_per_encoder_call"] == 32
    assert writer["semantic_set_fusion"].startswith("valid_frame_mean")
    assert "centered" in writer["semantic_set_value"]
    assert "no_value_projection" in writer["semantic_set_value"]
    assert writer["semantic_core_blocks"] == 2
    assert writer["procedure_attention"] == "global_causal_pre_norm_with_valid_mask"
    assert writer["procedure_blocks"] == 2
    assert writer["visual_transition_heads"] == 8
    assert writer["visual_transition_first_frame"] == "exact_zero"
    assert "actual_arm_input_order" in writer["visual_transition_source"]
    assert "no_value_projection" in writer["visual_transition_value"]
    assert writer["slot_fusion"].startswith("zero_initialized")
    assert writer["post_fusion_blocks"] == 1
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
    assert conditioning["ddp_gradient_sync"].startswith("first_five")
    assert config["information_wall"]["test_actions_read"] == 0
    assert config["information_wall"]["test_video_values_read"] == 0
    assert "state" in config["information_wall"]["writer_forbidden_inputs"]
    assert config["profile_defaults"]["expected_world_size"] == 4
    assert config["profile_evidence"]["status"] == "sealed_b20"
    assert config["profile_evidence"]["allowed_physical_gpu_ids"] == [4, 5, 6, 7]
    assert config["profile_evidence"]["initial_candidate"] == {
        "max_frames_per_encoder_call": 32,
        "per_task_action_batch_size": 20,
    }
    assert config["profile_evidence"]["only_fallback_candidate"] == {
        "max_frames_per_encoder_call": 32,
        "per_task_action_batch_size": 16,
    }
    assert config["profile_evidence"]["selected"]["per_task_action_batch_size"] == 20
    assert config["profile_evidence"]["selected"]["contains_real_105_frame_video"] is True
    assert config["profile_evidence"]["upper_bound"].startswith(
        "larger_batches_not_scanned"
    )
    assert config["profile_evidence"]["exact_resume_smoke"]["status"].startswith(
        "pass_macro_boundary"
    )
    assert config["profile_evidence"]["real_transition_evidence"]["status"].startswith(
        "pass_real_profile"
    )
    assert config["profile_evidence"]["inference_profile"] is None
    assert config["profile_evidence"]["teacher_videos_per_task_visit"] == 1
    assert config["specificity_gate"]["status"] == "pending"
    assert (
        config["profile_evidence"][
            "writer_video_conditions_per_rank_per_macro_update"
        ]
        == 6
    )
    assert config["formal_run"]["status"] == "sealed"
    assert config["formal_run"]["total_steps"] == 2400
    assert config["formal_run"]["per_rank_batch_size"] == 20
    assert config["formal_run"]["selected_stop_step"] == 200
    assert config["formal_run"]["stage_stop_steps"] == "every:200"
    assert "live_b20_throughput" in config["formal_run"]["segment_definition"]
    assert "without_runtime_full_data_sha" in config["formal_run"][
        "data_integrity_check"
    ]


def test_old_recipe_overlay_changes_training_without_changing_v6() -> None:
    config = load_writer_config(OLD_RECIPE_CONFIG)
    assert (
        config["writer"]["architecture"]
        == "pi05_task_grounded_semantic_set_visual_transition_causal_procedure_slot_fusion_v6"
    )
    assert config["conditioning_training"] == {
        "method": "single_video_multi_action_positive_functional_loss",
        "update_topology": "rank_rotating_one_task_per_rank",
        "writer_language_contract": (
            "correct_task_language_state_free_teacher_action_suffix"
        ),
        "policy_language_contract": "correct_action_query_task_language",
        "action_query_batch_owner": (
            "one physical action batch per rank with no optimizer gradient "
            "accumulation"
        ),
        "task_assignment": (
            "one task per rank per optimizer step with globally balanced task "
            "rotation"
        ),
        "tasks_per_rank_per_optimizer_update": 1,
        "global_tasks_per_optimizer_update": 4,
        "teacher_videos_per_task_visit": 1,
        "action_video_assignment": "all_actions_share_single_video_lora",
        "logical_pair_batch": "per_rank_action_batch",
        "policy_noise_contract": (
            "one independent policy flow noise and time draw per action query"
        ),
        "pair_loss_reduction": "mean_over_rank_local_action_batch",
        "task_loss_scale_before_backward": "one",
        "ddp_gradient_sync": "one_synchronized_backward_per_optimizer_step",
        "optimizer_steps_per_macro_update": 1,
        "checkpoint_boundary": "complete_optimizer_update_only",
        "normal_loss_weight": 1.0,
    }
    assert config["optimization"]["scheduler"] == {
        "kind": "cosine_decay_with_warmup",
        "peak_lr": 0.0003,
        "warmup_steps": 100,
        "decay_steps": 12000,
        "decay_lr": 1e-05,
    }
    assert config["_config_derivation"]["base_sha256"] == (
        "812793661ea20b7207f15e6a4ae13d506f69d0d3003c72f1bbcc16837aaf33fb"
    )


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
    with pytest.raises(WriterModelError, match="B20"):
        resolve_runtime(invalid_batch, config, context)
    oversized_batch = copy.copy(profile)
    oversized_batch.batch_size = 21
    oversized_batch.stop_after_step = None
    with pytest.raises(WriterModelError, match="B20"):
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
    pending = copy.deepcopy(config)
    pending["formal_run"]["status"] = "pending_profile"
    with pytest.raises(WriterModelError, match="not sealed"):
        resolve_runtime(formal, pending, context)
    monkeypatch.setattr(
        "ember.writer.as_contract.git_state",
        lambda _root: {
            "dirty_paths": [],
            "commit": "same",
            "origin_main": "same",
        },
    )
    formal.skip_data_sha = True
    assert resolve_runtime(formal, config, context) == (
        2400,
        20,
        tuple(range(25, 2401, 25)),
    )
    assert formal.stop_after_step == 200


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
    ):
        assert not (REPO_ROOT / "configs" / name).exists()
    with pytest.raises(WriterModelError, match="unsupported PI05 AS-Writer"):
        load_writer_config(REPO_ROOT / "configs/pi05_as_writer_v2.json")


def test_task_complete_step_scales_six_losses_and_syncs_only_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_ids = (10, 20, 30, 40, 50, 60)

    class Wrapped:
        active = False
        entries = 0

        @contextmanager
        def no_sync(self):
            self.entries += 1
            self.active = True
            try:
                yield
            finally:
                self.active = False

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
    wrapped = Wrapped()
    scheduler = Scheduler()
    calls: list[tuple[bool, float, int]] = []

    def fake_pack(_runtime, *, task_id, teacher_demo, action_batch_size):
        return (task_id,), {
            "teacher_demo_index": teacher_demo,
            "teacher_video_sampled_frames": 10 + task_id,
            "unique_teacher_video_conditions": 1,
            "actions_per_video": action_batch_size,
        }

    def fake_differentiate(
        runtime,
        packed,
        _policy_batch,
        *,
        loss_scale,
    ):
        calls.append((wrapped.active, loss_scale, int(packed[0])))
        contribution = torch.full_like(writer.weight, loss_scale)
        writer.weight.grad = (
            contribution
            if writer.weight.grad is None
            else writer.weight.grad + contribution
        )
        return torch.tensor(float(packed[0])), {"ok": True}

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
                    "task_complete_single_video_multi_action_positive_"
                    "functional_loss"
                )
            },
            "optimization": {
                "optimizer": {"gradient_clip_norm": 1.0}
            },
        },
        tasks_per_rank_per_update=6,
        iterator=iter(
            {"task_id": torch.tensor([task_id, task_id])}
            for task_id in task_ids
        ),
        sampler=Sampler(),
        video_schedule=Schedule(),
        videos_per_task_visit=1,
        processor=SimpleNamespace(training_batch=lambda batch: batch),
        wrapped_writer=wrapped,
        writer=writer,
        policy=torch.nn.Identity(),
    )
    row = as_step.run_writer_step(runtime, step=0, started=0.0)
    assert row["optimizer_step"] == 1
    assert wrapped.entries == 5
    assert [active for active, _, _ in calls] == [True] * 5 + [False]
    assert [task_id for _, _, task_id in calls] == list(task_ids)
    assert all(scale == pytest.approx(1.0 / 6.0) for _, scale, _ in calls)
    assert writer.weight.item() == pytest.approx(0.9)
    assert scheduler.steps == 1
    assert len(captured["records"]) == 6
