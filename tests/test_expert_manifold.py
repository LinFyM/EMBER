from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest
import torch

from ember.expert_manifold.contract import (
    ExpertTask,
    ExpertManifoldError,
    load_task_expert_config,
    parse_resume_task,
    parse_task_indices,
    resolve_runtime,
    validate_formal_task_assignment,
    worker_stage_resume_step,
)
from ember.writer.data import WriterTaskAuthority
from ember.expert_manifold.expert_training import _scheduler
from ember.expert_manifold.evaluation import (
    FrozenTaskExpertAdapter,
    TASK_EXPERT_ADAPTER_KIND,
    TASK_EXPERT_EPISODE_SCHEMA,
    _evaluation_task_rows,
    inspect_projected_task_expert_bank,
    inspect_task_expert_bank,
    inspect_task_expert_evaluation,
    validate_task_expert_episode,
)
from ember.expert_manifold.projection import (
    PROJECTED_TASK_EXPERT_ADAPTER_SCHEMA,
    _projection_contract,
)
from ember.expert_manifold.meta_contract import meta_expert_rows
from ember.expert_manifold.diagnostic_contract import (
    validation_expert_rows,
    validation_worker_assignments,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import (
    read_json,
    write_json_atomic,
)
from ember.expert_manifold.sampler import TaskLocalEpochSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/pi05_video_expert_manifold_v1.json"
META_CONFIG = REPO_ROOT / "configs/pi05_nonheld_meta_expert_bank_v1.json"
VALIDATION_CONFIG = REPO_ROOT / "configs/pi05_validation_expert_diagnostic_v1.json"
ECP_PARTICLE_CONFIG = REPO_ROOT / "configs/pi05_ecp_stage1a_particle_experts_v1.json"


def test_task_expert_config_contains_only_the_retained_train24_authority() -> None:
    config = load_task_expert_config(CONFIG)
    assert config["status"] == "sealed_task_expert_reference"
    assert config["task_experts"]["task_count"] == 24
    assert config["task_experts"]["task_parameter_sharing"] == "none"
    assert "topological_writer" not in config
    assert "meta_training" not in config
    assert config["information_wall"]["validation_actions_read"] == 0
    assert config["task_experts"]["profile_defaults"]["scheduler_total_steps"] == 2000


def test_ecp_particle_experts_are_independent_fixed_step_held5_lineages() -> None:
    config = load_task_expert_config(ECP_PARTICLE_CONFIG)
    experts = config["task_experts"]
    assert experts["optimization"]["seed"] != 7
    assert experts["formal_run"]["checkpoint_selection"].startswith("fixed_step2000")
    for index in experts["formal_run"]["task_indices"]:
        validate_formal_task_assignment(config, (index,))
    with pytest.raises(ExpertManifoldError):
        validate_formal_task_assignment(config, (1,))


def test_nonheld_meta_bank_supports_its_fixed_train_and_validation_panels() -> None:
    rows = meta_expert_rows(load_task_expert_config(META_CONFIG))
    all_rows = _evaluation_task_rows(
        rows, is_meta=True, evaluation_role="nonheld_meta"
    )
    train_rows = _evaluation_task_rows(
        rows, is_meta=True, evaluation_role="nonheld_meta_train"
    )
    validation_rows = _evaluation_task_rows(
        rows, is_meta=True, evaluation_role="nonheld_meta_validation"
    )
    assert len(all_rows) == 71
    assert len(train_rows) == 56
    assert len(validation_rows) == 15
    assert {int(row["task_id"]) for row in train_rows}.isdisjoint(
        int(row["task_id"]) for row in validation_rows
    )


def test_validation_expert_diagnostic_is_sealed_away_from_shared_training() -> None:
    config = load_task_expert_config(VALIDATION_CONFIG)
    rows = validation_expert_rows(config)
    selected = _evaluation_task_rows(
        rows,
        is_meta=False,
        evaluation_role="validation",
        is_validation_diagnostic=True,
    )
    assert len(selected) == 8
    assert {row["split_role"] for row in selected} == {"validation_diagnostic"}
    assert config["information_wall"]["writer_or_decoder_gradient_use"] is False
    assert config["task_experts"]["formal_run"]["diagnostic_evaluation_step"] == 2000
    assert set(
        value
        for assignment in validation_worker_assignments(
            config["task_experts"]["formal_run"]
        )
        for value in assignment
    ) == set(range(8))


def test_projected_meta_subpanel_validates_the_complete_projection_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = tuple(
        {
            "suite": "libero_90",
            "task_id": task_id,
            "split_role": (
                "meta_train" if task_id < 56 else "meta_validation_oracle"
            ),
        }
        for task_id in range(71)
    )
    full_adapter = {
        "tasks": list(rows),
        "information_wall": {
            "evaluation_role": "nonheld_meta",
            "evaluated_task_count": 71,
        },
    }
    inspected: list[str] = []

    def inspect_bank(**kwargs: Any) -> dict[str, object]:
        inspected.append(str(kwargs["evaluation_role"]))
        assert len(tuple(kwargs["task_keys"])) == 71
        return full_adapter

    monkeypatch.setattr(
        "ember.expert_manifold.evaluation.load_task_expert_config",
        lambda _path: {"schema_version": "ember_pi05_nonheld_meta_expert_bank_v1"},
    )
    monkeypatch.setattr(
        "ember.expert_manifold.evaluation._expert_task_rows", lambda _config: rows
    )
    monkeypatch.setattr(
        "ember.expert_manifold.evaluation.inspect_task_expert_bank", inspect_bank
    )
    monkeypatch.setattr(
        "ember.expert_manifold.evaluation.inspect_projected_task_expert_bank",
        lambda base, _manifest: {**base, "arm": "projected"},
    )
    result = inspect_task_expert_evaluation(
        config_path=tmp_path / "meta.json",
        bank_root=tmp_path / "bank",
        step=1000,
        source={},
        task_keys=[("libero_90", task_id) for task_id in range(56)],
        evaluation_role="nonheld_meta_train",
        require_formal=True,
        projection_manifest=tmp_path / "projection.json",
    )
    assert inspected == ["nonheld_meta"]
    assert len(result["tasks"]) == 56
    assert result["information_wall"]["evaluation_role"] == "nonheld_meta_train"


@pytest.mark.parametrize(
    ("projection_kind", "member", "arm"),
    (
        ("stable_shared_prior_baseline", "shared", "stable_shared_prior_baseline"),
        (
            "stable_shared_prior_task_residual_decoder",
            "earliest",
            "stable_shared_prior_residual_earliest_projection",
        ),
    ),
)
def test_phase_projection_accepts_one_merged_shared_prior_surface(
    tmp_path: Path,
    projection_kind: str,
    member: str,
    arm: str,
) -> None:
    assets = {}
    for name in (
        "decoder_checkpoint",
        "code_artifact",
        "training_result",
        "shared_prior_adapter",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        assets[name] = {"path": str(path), "bytes": path.stat().st_size}
    contract = _projection_contract(
        {
            "schema_version": (
                "ember_phase_aligned_functional_decoder_train24_projection_v1"
            ),
            "projection_kind": projection_kind,
            **assets,
            "optimization": {
                "decoder_frozen": True,
                "held_code_gradient_steps": 0,
                "final_lora_averaging": False,
                "single_complete_lora": True,
                "second_adapter_deployed": False,
                "rank_partition": {
                    "shared": [0, 12],
                    "task_residual": [12, 16],
                    "merge": "exact_effective_delta_sum",
                },
                "code_member": member,
            },
        }
    )
    assert contract["arm"] == arm
    assert contract["asset"]["shared_prior_adapter"] == assets[
        "shared_prior_adapter"
    ]


def test_ecp_stage1_projection_accepts_one_complete_privileged_lora_surface(
    tmp_path: Path,
) -> None:
    assets = {}
    for name in (
        "stage1_config",
        "stage1_checkpoint",
        "base_projection_manifest",
        "policy_support_bank",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        assets[name] = {"path": str(path), "bytes": path.stat().st_size}
    contract = _projection_contract(
        {
            "schema_version": "ember_ecp_stage1_process_value_selector_projection_v10",
            "projection_kind": "ecp_stage1_privileged_process_value_selector_compiler",
            **assets,
            "optimization": {
                "task_visits": 228,
                "held_shared_gradient_steps": 0,
                "compiler_frozen_for_materialization": True,
                "single_complete_lora": True,
                "final_lora_averaging": False,
                "rank": 16,
                "all_ranks_writable": True,
                "parameterization": "prior-only exact template; full-process process-value-only bounded rank-one retraction",
                "content_address_separated": True,
                "query_content_modulated": True,
                "policy_support_teacher": True,
                "raw_factor_addition": False,
                "fixed_rank_partition": False,
                "second_adapter_deployed": False,
                "objective_phase": "policy_support",
            },
            "information_wall": {
                "privileged_q_pi": True,
                "second_adapter_deployed": False,
            },
        }
    )
    assert contract["arm"] == "ecp_stage1_q_pi_process_value_selector_tv228"
    assert contract["asset"]["single_complete_lora"] is True


def test_ecp_stage1_outcome_projection_keeps_the_same_single_lora_surface(
    tmp_path: Path,
) -> None:
    assets = {}
    for name in (
        "stage1_config",
        "stage1_checkpoint",
        "base_projection_manifest",
        "policy_support_bank",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        assets[name] = {"path": str(path), "bytes": path.stat().st_size}
    contract = _projection_contract(
        {
            "schema_version": "ember_ecp_stage1_outcome_binding_projection_v11",
            "projection_kind": "ecp_stage1_privileged_outcome_binding_compiler",
            **assets,
            "optimization": {
                "outcome_macro": 1,
                "held_shared_gradient_steps": 0,
                "compiler_frozen_for_materialization": True,
                "single_complete_lora": True,
                "final_lora_averaging": False,
                "rank": 16,
                "all_ranks_writable": True,
                "parameterization": "prior-only exact template; full-process process-value-only bounded rank-one retraction",
                "content_address_separated": True,
                "query_content_modulated": True,
                "policy_support_teacher": True,
                "raw_factor_addition": False,
                "fixed_rank_partition": False,
                "second_adapter_deployed": False,
                "objective_phase": "outcome_calibrated_policy_support",
            },
            "information_wall": {
                "privileged_q_pi": True,
                "second_adapter_deployed": False,
            },
        }
    )
    assert contract["arm"] == "ecp_stage1_q_pi_outcome_binding_m1"
    assert contract["asset"]["single_complete_lora"] is True


def test_ecp_stage1_program_locked_compiler_projection_is_one_complete_lora(
    tmp_path: Path,
) -> None:
    assets = {}
    for name in (
        "stage1_config",
        "stage1_checkpoint",
        "base_projection_manifest",
        "policy_support_bank",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        assets[name] = {"path": str(path), "bytes": path.stat().st_size}
    contract = _projection_contract(
        {
            "schema_version": "ember_ecp_stage1_program_locked_compiler_projection_v20",
            "projection_kind": "ecp_stage1_privileged_program_locked_compiler_identification",
            **assets,
            "optimization": {
                "task_visits": 114,
                "held_shared_gradient_steps": 0,
                "compiler_trainable_during_training": True,
                "visible_program_frozen_during_training": True,
                "policy_teacher_frozen_during_training": True,
                "compiler_frozen_for_materialization": True,
                "single_complete_lora": True,
                "final_lora_averaging": False,
                "rank": 16,
                "all_ranks_writable": True,
                "parameterization": "prior-only exact template; full-process process-value-only bounded rank-one retraction",
                "content_address_separated": True,
                "query_content_modulated": True,
                "policy_support_teacher": True,
                "raw_factor_addition": False,
                "fixed_rank_partition": False,
                "second_adapter_deployed": False,
                "objective_phase": "task_balanced_program_locked_compiler_identification",
            },
            "information_wall": {
                "privileged_q_pi": True,
                "second_adapter_deployed": False,
            },
        }
    )
    assert contract["arm"] == "ecp_stage1_q_pi_program_locked_compiler_tv114"
    assert contract["asset"]["single_complete_lora"] is True


def test_profile_runtime_supports_fresh_then_exact_resume_boundary() -> None:
    config = load_task_expert_config(CONFIG)
    fresh = Namespace(mode="profile", batch_size=None, stop_after_step=1, resume=None)
    resumed = Namespace(
        mode="profile", batch_size=None, stop_after_step=3, resume=Path("x")
    )
    assert resolve_runtime(fresh, config) == (3, 16, (1, 3), 1)
    assert resolve_runtime(resumed, config) == (3, 16, (1, 3), 3)


def test_formal_experts_use_the_sealed_live_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_task_expert_config(CONFIG)
    monkeypatch.setattr(
        "ember.expert_manifold.contract.git_state",
        lambda _root: {
            "branch": "main",
            "dirty_paths": [],
            "commit": "sealed",
            "upstream": "origin/main",
            "upstream_commit": "sealed",
            "authority_ref": "origin/main",
            "authority_contains_commit": True,
        },
    )
    args = Namespace(mode="formal", batch_size=None, stop_after_step=1000, resume=None)
    assert resolve_runtime(args, config) == (
        2000,
        16,
        (250, 500, 1000, 1500, 2000),
        1000,
    )


def test_task_local_sampler_is_step_exact_across_epoch_boundary() -> None:
    sampler = TaskLocalEpochSampler(range(11), task_id=7, batch_size=4, seed=19)
    uninterrupted = tuple(
        value for step in range(9) for value in sampler.batch_for_step(step)
    )
    resumed = tuple(
        value for step in range(3, 9) for value in sampler.batch_for_step(step)
    )
    assert resumed == uninterrupted[12:]
    assert len(set(uninterrupted[:11])) == 11
    assert len(set(uninterrupted[11:22])) == 11


def test_task_assignment_and_resume_identity_are_explicit() -> None:
    assert parse_task_indices("0,6,12,18", 24) == (0, 6, 12, 18)
    with pytest.raises(ExpertManifoldError):
        parse_task_indices("6,0", 24)
    assert parse_resume_task(
        Path("worker/task_06_global_12/checkpoints/step_00000003")
    ) == (6, 3)


def test_task_expert_scheduler_warms_then_decays() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=5e-5)
    scheduler = _scheduler(
        optimizer,
        total_steps=100,
        warmup_steps=25,
        peak_lr=5e-5,
        decay_lr=1e-7,
    )
    values = [optimizer.param_groups[0]["lr"]]
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        values.append(optimizer.param_groups[0]["lr"])
    assert values[0] < values[24] <= values[25]
    assert values[25] > values[75] > values[-1]
    assert values[-1] == pytest.approx(1e-7)


def test_worker_stage_resume_requires_complete_same_step_bank(tmp_path: Path) -> None:
    tasks = tuple(
        ExpertTask(
            ordinal=ordinal,
            global_task_id=ordinal + 10,
            suite="suite",
            task_id=ordinal,
            split_role="train",
            language=f"task {ordinal}",
            authority=WriterTaskAuthority(
                task_id=ordinal + 10,
                language=f"task {ordinal}",
                path=tmp_path / f"{ordinal}.hdf5",
                expected_bytes=1,
            ),
        )
        for ordinal in range(2)
    )
    rows = []
    for task in tasks:
        checkpoint = (
            tmp_path
            / f"task_{task.ordinal:02d}_global_{task.global_task_id:02d}"
            / "checkpoints"
            / "step_00001000"
        )
        checkpoint.mkdir(parents=True)
        rows.append(
            {
                "task_ordinal": task.ordinal,
                "global_task_id": task.global_task_id,
                "completed_steps": 1000,
            }
        )
    from ember.pi05_source_checkpoint import write_json_atomic

    write_json_atomic(
        tmp_path / "worker_summary.json",
        {
            "schema_version": "ember_pi05_task_expert_worker_summary_v1",
            "tasks": rows,
            "completed_task_count": 2,
            "selected_stop_step": 1000,
        },
    )
    assert worker_stage_resume_step(tmp_path, tmp_path, tasks) == 1000


def test_task_expert_episode_evidence_is_task_and_state_exact() -> None:
    record = {
        "suite": "libero_goal",
        "task_id": 2,
        "ordinal": 7,
        "global_task_id": 22,
        "language": "perform task",
        "step": 1000,
        "checkpoint": "/bank/task/checkpoints/step_00001000",
        "manifest_bytes": 300,
        "adapter_bytes": 2_654_208,
    }
    adapter = {"kind": TASK_EXPERT_ADAPTER_KIND, "tasks": [record]}
    evidence = {
        "schema_version": TASK_EXPERT_EPISODE_SCHEMA,
        **record,
        "init_state_id": 4,
    }
    assert validate_task_expert_episode(
        adapter,
        evidence,
        suite="libero_goal",
        task_id=2,
        init_state_id=4,
    )
    assert not validate_task_expert_episode(
        adapter,
        evidence,
        suite="libero_goal",
        task_id=2,
        init_state_id=5,
    )


def test_task_expert_runtime_uses_the_narrow_task_expert_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = {
        "config": {"path": str(tmp_path / "task-experts.json")},
        "bank_root": str(tmp_path / "bank"),
        "step": 1000,
        "tasks": [],
    }
    loaded: list[Path] = []
    monkeypatch.setattr(
        "ember.expert_manifold.evaluation.inspect_task_expert_bank",
        lambda **_kwargs: pytest.fail("worker repeated the launcher bank inspection"),
    )
    monkeypatch.setattr(
        "ember.expert_manifold.evaluation.load_task_expert_config",
        lambda path: loaded.append(path) or {"authorities": {}},
    )
    monkeypatch.setattr(
        "ember.expert_manifold.evaluation.authority_path",
        lambda _config, _name: tmp_path / "lora.json",
    )
    lora = object()
    monkeypatch.setattr(
        "ember.expert_manifold.evaluation.load_pi05_lora_contract",
        lambda _path: lora,
    )
    monkeypatch.setattr(
        "ember.expert_manifold.evaluation.inject_task_lora",
        lambda _policy, observed_lora: observed_lora is lora,
    )
    monkeypatch.setattr(
        "ember.expert_manifold.evaluation.task_lora_state_dict",
        lambda _policy: {},
    )
    policy = torch.nn.Linear(1, 1)
    adapter = FrozenTaskExpertAdapter(
        policy=policy,
        source={},
        evaluation_adapter=observed,
        task_keys=(),
        device=torch.device("cpu"),
        require_formal=False,
    )
    assert loaded == [Path(observed["config"]["path"])]
    assert adapter.lora is lora


def test_complete_hashless_task_expert_bank_is_inspectable(tmp_path: Path) -> None:
    config = load_task_expert_config(CONFIG)
    manifest = read_json(
        REPO_ROOT / config["authorities"]["target_data_manifest"]["path"]
    )
    rows = sorted(
        (row for row in manifest["tasks"] if row["split_role"] == "train"),
        key=lambda row: int(row["global_task_id"]),
    )
    lora = load_pi05_lora_contract(
        REPO_ROOT / config["authorities"]["lora_contract"]["path"]
    )
    source = {
        "source_run": str(tmp_path / "source"),
        "checkpoint": str(tmp_path / "source/checkpoints/step_00001000"),
        "model_path": str(tmp_path / "source/checkpoints/step_00001000/policy"),
    }
    assignments = tuple(tuple(range(worker, 24, 6)) for worker in range(6))
    for worker, ordinals in enumerate(assignments):
        worker_dir = tmp_path / "bank" / f"worker_{worker}"
        task_rows = []
        summary_rows = []
        for ordinal in ordinals:
            row = rows[ordinal]
            global_task_id = int(row["global_task_id"])
            task_rows.append(
                {
                    "ordinal": ordinal,
                    "global_task_id": global_task_id,
                    "suite": row["suite"],
                    "task_id": int(row["task_id"]),
                    "split_role": "train",
                    "language": row["language"],
                }
            )
            summary_rows.append(
                {
                    "task_ordinal": ordinal,
                    "global_task_id": global_task_id,
                    "completed_steps": 1000,
                }
            )
            checkpoint = (
                worker_dir
                / f"task_{ordinal:02d}_global_{global_task_id:02d}"
                / "checkpoints/step_00001000"
            )
            checkpoint.mkdir(parents=True)
            (checkpoint / "adapter.safetensors").write_bytes(b"a")
            (checkpoint / "trainer.pt").write_bytes(b"t")
            write_json_atomic(
                checkpoint / "manifest.json",
                {
                    "schema_version": "ember_pi05_task_expert_checkpoint_v1",
                    "step": 1000,
                    "task_ordinal": ordinal,
                    "global_task_id": global_task_id,
                    "state_tensor_count": lora.state_tensor_count,
                    "state_parameter_count": lora.parameter_count,
                    "files": {"adapter.safetensors": 1, "trainer.pt": 1},
                    "content_hash_policy": "disabled_by_owner",
                },
            )
        write_json_atomic(
            worker_dir / "run_contract.json",
            {
                "schema_version": "ember_pi05_task_expert_worker_launch_v1",
                "mode": "formal",
                "git": {"commit": "training-commit"},
                "config": {
                    "path": str(Path("/frozen/formal-worktree/configs") / CONFIG.name),
                    "schema": config["schema_version"],
                },
                "source": {
                    "run": source["source_run"],
                    "checkpoint": source["checkpoint"],
                    "model_path": source["model_path"],
                },
                "tasks": task_rows,
                "runtime": {
                    "per_task_batch_size": 16,
                    "task_parameter_sharing": "none",
                    "host": "gpu01",
                    "cuda_visible_device": str(worker),
                    "device_name": "NVIDIA A40",
                },
                "content_hash_policy": "disabled_by_owner",
            },
        )
        write_json_atomic(
            worker_dir / "worker_summary.json",
            {
                "schema_version": "ember_pi05_task_expert_worker_summary_v1",
                "tasks": summary_rows,
                "completed_task_count": 4,
                "selected_stop_step": 1000,
            },
        )
    observed = inspect_task_expert_bank(
        config_path=CONFIG,
        bank_root=tmp_path / "bank",
        step=1000,
        source=source,
        task_keys=tuple((str(row["suite"]), int(row["task_id"])) for row in rows),
        evaluation_role="development_train",
        require_formal=True,
    )
    assert observed["kind"] == TASK_EXPERT_ADAPTER_KIND
    assert observed["training_commit"] == "training-commit"
    assert len(observed["tasks"]) == 24
    assert observed["information_wall"]["validation_actions_read"] == 0

    projected_rows = []
    for row in observed["tasks"]:
        path = tmp_path / "projection" / f"task_{int(row['ordinal']):02d}.safetensors"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"projected")
        projected_rows.append(
            {
                "suite": row["suite"],
                "task_id": int(row["task_id"]),
                "ordinal": int(row["ordinal"]),
                "global_task_id": int(row["global_task_id"]),
                "expert_checkpoint": row["checkpoint"],
                "projected_adapter": str(path),
                "projected_adapter_bytes": path.stat().st_size,
            }
        )
    projection_manifest = tmp_path / "projection" / "projection_manifest.json"
    write_json_atomic(
        projection_manifest,
        {
            "schema_version": "ember_writer_fixed_head_reachability_oracle_v1",
            "repository": {"commit": "oracle", "dirty_paths": []},
            "writer_checkpoint": "/writer/checkpoints/macro_00000025",
            "optimization": {"factor_heads_frozen": True},
            "information_wall": {
                "role": "development_train_oracle_only",
                "deployment_carrier": False,
            },
            "tasks": projected_rows,
        },
    )
    projected = inspect_projected_task_expert_bank(observed, projection_manifest)
    assert projected["schema_version"] == PROJECTED_TASK_EXPERT_ADAPTER_SCHEMA
    assert projected["projection"]["deployment_carrier"] is False
    assert all("projected_adapter" in row for row in projected["tasks"])
