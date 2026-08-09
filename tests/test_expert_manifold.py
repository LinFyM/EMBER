from argparse import Namespace
import copy
import math
from pathlib import Path

import pytest
import torch

from ember.expert_manifold.contract import (
    ExpertTask,
    ExpertManifoldError,
    load_expert_manifold_config,
    parse_resume_task,
    parse_task_indices,
    resolve_runtime,
    worker_stage_resume_step,
)
from ember.writer.data import WriterTaskAuthority
from ember.expert_manifold.expert_training import _scheduler
from ember.expert_manifold.evaluation import (
    FrozenTaskExpertAdapter,
    TASK_EXPERT_ADAPTER_KIND,
    TASK_EXPERT_EPISODE_SCHEMA,
    inspect_task_expert_bank,
    validate_task_expert_episode,
)
from ember.pi05_lora import load_pi05_lora_contract
from ember.pi05_source_checkpoint import (
    read_json,
    write_json_atomic,
)
from ember.expert_manifold.sampler import TaskLocalEpochSampler
from ember.expert_manifold.model import (
    TopologicalLoRAChunkLayout,
    phase_centered_causal_memory,
)
from ember.expert_manifold.video_features import (
    FrozenPi05VideoInnovationEncoder,
    phase_resample,
)
from ember.expert_manifold.feature_cache import _feature_contract, _feature_runtime
from ember.expert_manifold.video_schedule import (
    condition_demo_index,
    reference_demo_index,
)
from ember.expert_manifold.inference import (
    EXPERT_MANIFOLD_ADAPTER_SCHEMA,
    EXPERT_MANIFOLD_EPISODE_SCHEMA,
    EXPERT_MANIFOLD_WRITER_KIND,
    expected_expert_manifold_episode_evidence,
    validate_expert_manifold_episode_evidence,
)
from ember.eval_adapters import expected_writer_episode
from ember.pi05_assets import Pi05EvaluationError
from ember.lora import expected_lora_state_shapes, identity_lora_state


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/pi05_video_expert_manifold_v1.json"


def test_video_expert_manifold_config_keeps_video_as_dynamic_value() -> None:
    config = load_expert_manifold_config(CONFIG)
    assert config["method"]["language_only_lora_path"] is False
    assert config["video_features"]["shots"] == 1
    assert config["topological_writer"]["chunk_count"] == 168
    assert config["topological_writer"]["valid_values"] == 1_287_168
    assert config["topological_writer"]["video_value_path"] == (
        "phase_centered_projected_video_sqrt_normalized_causal_prefix_integral_only"
    )
    assert config["video_features"]["feature_width"] == 3072
    assert config["video_features"]["expert_hidden_width"] == 1024
    assert config["information_wall"]["validation_actions_read"] == 0
    assert config["information_wall"]["writer_video_split_roles"] == [
        "train",
        "validation",
        "test",
    ]
    assert config["task_experts"]["profile_defaults"]["scheduler_total_steps"] == 2000


def test_profile_runtime_supports_fresh_then_exact_resume_boundary() -> None:
    config = load_expert_manifold_config(CONFIG)
    fresh = Namespace(mode="profile", batch_size=None, stop_after_step=1, resume=None)
    resumed = Namespace(
        mode="profile", batch_size=None, stop_after_step=3, resume=Path("x")
    )
    assert resolve_runtime(fresh, config) == (3, 16, (1, 3), 1)
    assert resolve_runtime(resumed, config) == (3, 16, (1, 3), 3)


def test_formal_experts_use_the_sealed_live_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_expert_manifold_config(CONFIG)
    monkeypatch.setattr(
        "ember.expert_manifold.contract.git_state",
        lambda _root: {
            "dirty_paths": [],
            "commit": "sealed",
            "upstream_commit": "sealed",
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
        lambda **_kwargs: observed,
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
    config = load_expert_manifold_config(CONFIG)
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


def _synthetic_lora_states():
    contract = load_pi05_lora_contract(REPO_ROOT / "configs/pi05_lora_v1.json")
    template = {}
    target = {}
    for ordinal, (name, shape) in enumerate(
        expected_lora_state_shapes(contract).items()
    ):
        base = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
        base = base.mul(1e-7 * (ordinal + 1))
        if name.endswith(".lora_B.default.weight"):
            base.zero_()
        template[name] = base
        target[name] = base + torch.full_like(base, 0.001 * (ordinal + 1))
    return contract, template, target


def test_topological_lora_layout_round_trips_full_rank16_state() -> None:
    contract, template, target = _synthetic_lora_states()
    layout = TopologicalLoRAChunkLayout(contract, chunk_width=512)
    assert layout.chunk_count == 168
    assert layout.valid_values == 1_287_168
    assert layout.padded_values == 1_376_256
    values = layout.tokenize(target, template)
    recovered = layout.detokenize(values, template)
    assert all(torch.equal(recovered[name], value) for name, value in target.items())


def test_identity_lora_state_matches_template_a_zero_b_contract() -> None:
    contract, template, _ = _synthetic_lora_states()
    identity = identity_lora_state(contract)
    assert set(identity) == set(template)
    assert all(
        torch.count_nonzero(value) == 0
        for name, value in identity.items()
        if name.endswith(".lora_B.default.weight")
    )
    assert all(
        torch.count_nonzero(value) > 0
        for name, value in identity.items()
        if name.endswith(".lora_A.default.weight")
    )


def test_causal_memory_forces_order_binding_without_static_value() -> None:
    constant = torch.randn(2, 1, 8).expand(2, 4, 8).clone()
    assert torch.count_nonzero(phase_centered_causal_memory(constant)) == 0
    ordered = torch.randn(2, 4, 8)
    forward = phase_centered_causal_memory(ordered)
    reverse = phase_centered_causal_memory(ordered.flip(1))
    assert not torch.allclose(forward, reverse)
    assert not torch.allclose(forward.mean(dim=1), reverse.mean(dim=1))


def test_phase_resample_preserves_video_endpoints_and_zero() -> None:
    value = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    aligned = phase_resample(value, 5)
    assert aligned.shape == (5, 2)
    assert torch.equal(aligned[0], value[0])
    assert torch.equal(aligned[-1], value[-1])
    assert torch.count_nonzero(phase_resample(torch.zeros_like(value), 5)) == 0


def test_video_encoder_retains_task_span_and_action_expert_widths() -> None:
    encoder = FrozenPi05VideoInnovationEncoder(
        image_width=2048,
        expert_width=1024,
        feature_width=3072,
        phase_slots=16,
        max_frames_per_encoder_call=4,
        action_horizon=50,
        padded_action_dim=32,
        initialization_seed=7,
    )
    assert encoder.image_width == 2048
    assert encoder.expert_width == 1024
    assert encoder.feature_width == 3072


def test_one_shot_video_schedule_covers_fifty_states_without_replacement() -> None:
    values = tuple(
        reference_demo_index(
            7,
            "libero_goal",
            3,
            state,
            demo_count=50,
            sampling_mode="without_replacement",
        )
        for state in range(50)
    )
    assert len(set(values)) == 50
    assert (
        condition_demo_index(
            7,
            "libero_goal",
            3,
            0,
            condition="same_task_other",
            demo_count=50,
            sampling_mode="without_replacement",
        )
        == (values[0] + 17) % 50
    )
    assert (
        condition_demo_index(
            7,
            "libero_goal",
            3,
            0,
            condition="no_video",
            demo_count=50,
            sampling_mode="without_replacement",
        )
        == values[0]
    )


def test_expert_manifold_episode_evidence_keeps_one_video_dynamic() -> None:
    adapter = {
        "schema_version": EXPERT_MANIFOLD_ADAPTER_SCHEMA,
        "kind": EXPERT_MANIFOLD_WRITER_KIND,
        "arm": "expert_manifold_v6_tangent_tube_correct",
        "video_condition": "correct",
        "writer_asset": {
            "reference": "v6-prior:historical-macro400",
            "kind": "historical_v6_macro400_load_only",
            "method_macro": 0,
            "writer_parameter_count": 10_775_296,
            "generated_lora_tensor_count": 76,
        },
        "lora_contract": {"reference": "lora:rank16"},
        "video_schedule": {
            "seed": 7,
            "demo_count": 50,
            "sampling_mode": "without_replacement",
        },
        "task_video_mapping": [
            {
                "suite": "libero_goal",
                "task_id": 3,
                "language_global_task_id": 13,
                "video_suite": "libero_goal",
                "video_task_id": 3,
                "video_global_task_id": 13,
                "video_split_role": "validation",
            }
        ],
        "task_video_mapping_reference": "mapping",
        "pairing_reference": "paired",
    }
    evidence = expected_expert_manifold_episode_evidence(
        adapter,
        suite="libero_goal",
        task_id=3,
        init_state_id=4,
        lora_reference="generated",
    )
    assert (
        expected_writer_episode(
            adapter,
            suite="libero_goal",
            task_id=3,
            init_state_id=4,
            lora_reference="generated",
            evidence_schema=EXPERT_MANIFOLD_EPISODE_SCHEMA,
        )
        == evidence
    )
    with pytest.raises(Pi05EvaluationError, match="evidence schema changed"):
        expected_writer_episode(
            adapter,
            suite="libero_goal",
            task_id=3,
            init_state_id=4,
            lora_reference="generated",
            evidence_schema="wrong_schema",
        )
    assert len(evidence["teacher_demo_indices"]) == 1
    assert evidence["language_global_task_id"] == 13
    assert evidence["video_global_task_id"] == 13
    assert validate_expert_manifold_episode_evidence(
        adapter,
        {**evidence, "writer_generation_seconds": 0.5},
        suite="libero_goal",
        task_id=3,
        init_state_id=4,
    )
    no_video_adapter = {**adapter, "video_condition": "no_video"}
    no_video = expected_expert_manifold_episode_evidence(
        no_video_adapter,
        suite="libero_goal",
        task_id=3,
        init_state_id=4,
        lora_reference="identity",
    )
    assert (
        no_video["teacher_demo_indices"] == no_video["teacher_reference_demo_indices"]
    )
    assert no_video["teacher_video_frames_used"] is False


def test_video_feature_profile_seal_keeps_formal_input_semantics() -> None:
    config = load_expert_manifold_config(CONFIG)
    assert _feature_runtime(config, "profile") == (4, 4, (0, 1, 2, 3))
    formal = config["video_features"]["formal_run"]
    assert formal["status"] == "sealed"
    assert formal["demo_count"] == 50
    evidence = formal["profile_evidence"]
    assert evidence["device"] == "NVIDIA A40"
    assert evidence["demo_count"] == 4
    assert evidence["feature_shape"] == [4, 16, 3072]
    assert evidence["peak_reserved_bytes"] < 46_068 * 1024**2
    assert all(
        evidence[name] == 0
        for name in (
            "teacher_action_reads",
            "teacher_state_reads",
            "reward_reads",
            "terminal_reads",
            "oom_count",
            "nonfinite_count",
        )
    )


def test_video_feature_contract_ignores_writer_only_sealing() -> None:
    config = load_expert_manifold_config(CONFIG)
    changed = copy.deepcopy(config)
    changed["meta_training"]["formal_run"]["selected_expert_step"] = 500
    changed["meta_training"]["formal_run"]["status"] = "sealed"
    assert _feature_contract(changed) == _feature_contract(config)

    changed["video_features"]["phase_slots"] += 1
    assert _feature_contract(changed) != _feature_contract(config)
