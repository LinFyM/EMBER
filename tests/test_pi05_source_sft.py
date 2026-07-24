from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from ember.lora import (
    LoRATarget,
    canonical_contract_sha256,
    expected_lora_state_shapes,
    inject_task_lora,
    task_lora_state_dict,
)
from ember.pi05_lora import Pi05LoRAContract, load_pi05_lora_contract
from ember.pi05_source_checkpoint import (
    DistributedContext,
    canonical_hash,
    sha256_file,
    write_json_atomic,
)
from ember.source_sft.checkpoint import (
    SOURCE_SFT_CHECKPOINT_SCHEMA,
    load_source_sft_checkpoint,
    save_source_sft_checkpoint,
    validate_source_sft_checkpoint_files,
)
from ember.source_sft.contract import (
    SOURCE_SFT_LAUNCH_SCHEMA,
    Pi05SourceSFTError,
    _contract_stop_step,
    _target_tasks,
    load_source_sft_config,
    reconcile_resume_contract,
    resolve_runtime as resolve_source_sft_runtime,
)
from ember.source_sft.inference import inspect_source_sft_evaluation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pi05_source_sft_development_v1.json"
FINAL_CONFIG = ROOT / "configs/pi05_source_sft_final_v1.json"
RANK128_CONFIG = ROOT / "configs/pi05_source_sft_rank128_capacity_v1.json"


def test_development_config_selects_only_sealed_train_actions() -> None:
    config = load_source_sft_config(CONFIG)
    tasks = _target_tasks(config, Path("/target-data"), "development")
    assert config["sealed_stage"] == "development"
    assert len(tasks) == 24
    assert {task.split_role for task in tasks} == {"train"}
    assert sorted(task.global_task_id for task in tasks) == [
        0,
        2,
        4,
        5,
        7,
        9,
        12,
        14,
        15,
        16,
        18,
        19,
        20,
        21,
        22,
        25,
        28,
        29,
        34,
        35,
        36,
        37,
        38,
        39,
    ]
    suite_counts = {
        suite: sum(task.suite == suite for task in tasks)
        for suite in {task.suite for task in tasks}
    }
    assert sorted(suite_counts.values()) == [6, 6, 6, 6]
    assert config["information_wall"]["test_actions_read"] == 0
    assert config["information_wall"]["test_video_values_read"] == 0


def test_development_formal_budget_is_independent_ceiling_search() -> None:
    config = load_source_sft_config(CONFIG)
    formal = config["stages"]["development"]["formal_run"]
    assert formal["status"] == "sealed"
    assert formal["expected_world_size"] == 8
    assert formal["total_steps"] == 800
    assert formal["per_rank_batch_size"] == 64
    assert formal["checkpoint_steps"] == [100, 200, 400, 600, 800]
    assert config["optimization"]["scheduler"]["warmup_steps"] == 100
    assert config["optimization"]["scheduler"]["decay_steps"] == 800
    assert "not matched to AS-Writer" in formal["selection_rule"]
    assert formal["prior_matched_scale_result"]["optimizer_steps"] == 63


def test_rank128_capacity_config_keeps_data_wall_and_changes_only_capacity() -> None:
    baseline = load_source_sft_config(CONFIG)
    capacity = load_source_sft_config(RANK128_CONFIG)
    rank128 = load_pi05_lora_contract(
        ROOT / capacity["authorities"]["lora_contract"]["path"]
    )
    assert capacity["sealed_stage"] == "development"
    assert capacity["information_wall"] == baseline["information_wall"]
    assert capacity["data"] == baseline["data"]
    assert capacity["optimization"] == baseline["optimization"]
    assert rank128.rank == rank128.alpha == 128
    assert rank128.parameter_count == 10_297_344
    assert (
        capacity["stages"]["development"]["formal_run"]["status"]
        == "sealed"
    )
    assert (
        capacity["stages"]["development"]["formal_run"]["selected_stop_step"]
        == 800
    )
    assert capacity["stages"]["development"]["formal_run"]["stage_stop_steps"] == [
        800,
        1100,
        1400,
        1700,
        2000,
        2300,
        2400,
    ]
    assert capacity["profile_evidence"]["rank"] == 128
    assert capacity["profile_defaults"]["expected_world_size"] == 4
    assert capacity["profile_defaults"]["per_rank_batch_size"] == 128
    assert sha256_file(RANK128_CONFIG) == (
        ROOT / "configs/pi05_source_sft_rank128_capacity_v1.sha256"
    ).read_text(encoding="utf-8").split()[0]


def test_source_sft_declared_stage_stop_can_extend_without_schedule_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_source_sft_config(RANK128_CONFIG)
    config["stages"]["development"]["formal_run"]["status"] = "sealed"
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=4,
        device=torch.device("cpu"),
        numa_node=0,
        cpu_affinity=(0,),
    )
    monkeypatch.setattr(
        "ember.source_sft.contract.git_state",
        lambda _root: {"dirty_paths": [], "commit": "sealed", "origin_main": "sealed"},
    )
    args = SimpleNamespace(
        stage="development",
        mode="formal",
        total_steps=None,
        batch_size=None,
        checkpoint_steps=None,
        stop_after_step=1100,
        resume=Path("/tmp/step_00000800"),
        skip_data_sha=False,
    )
    assert resolve_source_sft_runtime(args, config, context) == (
        2400,
        128,
        tuple(range(100, 2401, 100)),
    )
    assert args.stop_after_step == 1100
    args.stop_after_step = 500
    with pytest.raises(Pi05SourceSFTError, match="sealed profile"):
        resolve_source_sft_runtime(args, config, context)


def test_formal_stage_extension_does_not_change_source_sft_contract_stop() -> None:
    config = load_source_sft_config(RANK128_CONFIG)
    args = SimpleNamespace(
        stage="development", mode="formal", stop_after_step=1100
    )
    assert _contract_stop_step(args, config, 2400) == 800
    args.mode = "profile"
    assert _contract_stop_step(args, config, 2400) == 1100


def test_source_sft_code_compatible_resume_allows_only_commit_change(
    tmp_path: Path,
) -> None:
    existing = {
        "schema_version": "contract",
        "git": {"branch": "main", "commit": "old"},
        "runtime": {"selected_stop_step": 400, "total_steps": 800},
    }
    write_json_atomic(tmp_path / "run_contract.json", existing)
    args = SimpleNamespace(
        output_dir=tmp_path,
        resume=tmp_path / "checkpoints/step_00000400",
        allow_contract_compatible_code_resume=True,
    )
    candidate = {**existing, "git": {"branch": "main", "commit": "new"}}
    assert reconcile_resume_contract(args, candidate) == existing

    changed = {
        **candidate,
        "runtime": {"selected_stop_step": 400, "total_steps": 1000},
    }
    with pytest.raises(Pi05SourceSFTError, match="scientific contract"):
        reconcile_resume_contract(args, changed)

    args.allow_contract_compatible_code_resume = False
    with pytest.raises(Pi05SourceSFTError, match="launch contract changed"):
        reconcile_resume_contract(args, candidate)


def test_development_config_cannot_open_final_stage() -> None:
    config = load_source_sft_config(CONFIG)
    context = SimpleNamespace(world_size=8)
    args = SimpleNamespace(
        stage="final",
        mode="profile",
        total_steps=1,
        batch_size=1,
        checkpoint_steps="1",
        stop_after_step=1,
        resume=None,
        skip_data_sha=False,
    )
    from ember.source_sft.contract import resolve_runtime

    with pytest.raises(Pi05SourceSFTError, match="own immutable sealed config"):
        resolve_runtime(args, config, context)


def test_final_config_selects_32_source_actions_and_frozen_step_budget() -> None:
    config = load_source_sft_config(FINAL_CONFIG)
    tasks = _target_tasks(config, Path("/target-data"), "final")
    formal = config["stages"]["final"]["formal_run"]
    assert config["sealed_stage"] == "final"
    assert len(tasks) == 32
    assert {task.split_role for task in tasks} == {"train", "validation"}
    assert not any(task.split_role == "test" for task in tasks)
    assert formal["status"] == "sealed"
    assert formal["total_steps"] == 800
    assert formal["selected_stop_step"] == 400
    assert formal["per_rank_batch_size"] == 64
    assert formal["checkpoint_steps"] == [200, 400, 600, 800]
    assert formal["development_selection"]["selected_optimizer_step"] == 400
    assert sha256_file(FINAL_CONFIG) == (
        ROOT / "configs/pi05_source_sft_final_v1.sha256"
    ).read_text(encoding="utf-8").split()[0]


def test_final_formal_runtime_keeps_development_scheduler_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ember.source_sft import contract

    config = load_source_sft_config(FINAL_CONFIG)
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=8,
        device=torch.device("cpu"),
        numa_node=0,
        cpu_affinity=(0,),
    )
    monkeypatch.setattr(
        contract,
        "git_state",
        lambda _: {"dirty_paths": [], "commit": "pushed", "origin_main": "pushed"},
    )
    args = SimpleNamespace(
        stage="final",
        mode="formal",
        total_steps=None,
        batch_size=None,
        checkpoint_steps=None,
        stop_after_step=None,
        resume=None,
        skip_data_sha=False,
    )
    assert contract.resolve_runtime(args, config, context) == (
        800,
        64,
        (200, 400, 600, 800),
    )
    assert args.stop_after_step == 400


class _TinyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(3, 4, bias=False)


class _Sampler:
    per_rank_batch_size = 2
    seed = 17
    episodes_per_task = 2

    @staticmethod
    def coverage_for_steps(start: int, stop: int) -> dict[int, tuple[int, ...]]:
        assert (start, stop) == (0, 1)
        return {0: (0, 1)}

    @staticmethod
    def consumed_identity_summary(start: int, stop: int) -> dict:
        assert (start, stop) == (0, 1)
        return {
            "start_step": 0,
            "stop_step": 1,
            "global_examples": 2,
            "unique_query_rows": 2,
            "min_examples_per_task": 2,
            "max_examples_per_task": 2,
            "identity_sha256": "1" * 64,
        }


def _tiny_lora() -> Pi05LoRAContract:
    sealed = load_pi05_lora_contract(ROOT / "configs/pi05_lora_v1.json")
    return Pi05LoRAContract(
        targets=(LoRATarget("proj", 3, 4),),
        rank=2,
        alpha=2,
        dropout=0.0,
        identity_seed=7,
        foundation_repository=sealed.foundation_repository,
        foundation_revision=sealed.foundation_revision,
        foundation_weights_sha256=sealed.foundation_weights_sha256,
        foundation_config_sha256=sealed.foundation_config_sha256,
        source_base_config_sha256=sealed.source_base_config_sha256,
        recipe_sha256=sealed.recipe_sha256,
    )


def test_source_sft_checkpoint_roundtrip_and_tamper_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ember.source_sft.checkpoint as checkpoint_module

    monkeypatch.setattr(checkpoint_module, "capture_rng", lambda context: {"cursor": 9})
    monkeypatch.setattr(checkpoint_module, "restore_rng", lambda state, context: None)
    context = DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )
    policy = inject_task_lora(_TinyPolicy(), _tiny_lora())
    optimizer = torch.optim.AdamW(task_lora_state_dict(policy).values(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    contract = {
        "schema_version": SOURCE_SFT_LAUNCH_SCHEMA,
        "stage": "development",
        "runtime": {"total_steps": 1, "dataloader_generator_seed_base": 23},
    }
    expected = {
        name: value.detach().clone()
        for name, value in task_lora_state_dict(policy).items()
    }
    checkpoint = save_source_sft_checkpoint(
        output_dir=tmp_path,
        step=1,
        context=context,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=_Sampler(),  # type: ignore[arg-type]
        contract=contract,
        mode="formal",
        metrics_rows=1,
    )
    manifest = validate_source_sft_checkpoint_files(
        checkpoint,
        world_size=1,
        contract_sha256=canonical_hash(contract),
    )
    assert manifest["schema_version"] == SOURCE_SFT_CHECKPOINT_SCHEMA
    with torch.no_grad():
        for value in task_lora_state_dict(policy).values():
            value.add_(1)
    step, rng, rows = load_source_sft_checkpoint(
        checkpoint=checkpoint,
        context=context,
        policy=policy,
        lora_contract=_tiny_lora(),
        optimizer=optimizer,
        scheduler=scheduler,
        per_rank_batch_size=2,
        sampler_seed=17,
        dataloader_generator_seed=23,
        contract_sha256=canonical_hash(contract),
    )
    assert (step, rng, rows) == (1, {"cursor": 9}, 1)
    for name, value in task_lora_state_dict(policy).items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    (checkpoint / "trainer_state.pt").write_bytes(b"tampered")
    with pytest.raises(Pi05SourceSFTError, match="file changed"):
        validate_source_sft_checkpoint_files(
            checkpoint,
            world_size=1,
            contract_sha256=canonical_hash(contract),
        )


def _source() -> dict:
    return {
        "source_run_contract_sha256": "1" * 64,
        "checkpoint_manifest_sha256": "2" * 64,
        "optimizer_step": 30_000,
        "source_run_summary_sha256": "3" * 64,
        "source_training_commit": "4" * 40,
        "source_base_config_sha256": "5" * 64,
        "source_authority_hashes": {"normalization": "6" * 64},
        "model_files": [
            {
                "path": "policy/model.safetensors",
                "bytes": 1,
                "sha256": "7" * 64,
            }
        ],
        "model_path": "/source/policy",
    }


def _static_adapter_fixture(
    tmp_path: Path,
    *,
    config_path: Path = CONFIG,
    mode: str = "profile",
    world_size: int = 8,
    step: int = 4,
) -> tuple[Path, dict]:
    config = load_source_sft_config(config_path)
    lora = load_pi05_lora_contract(ROOT / config["authorities"]["lora_contract"]["path"])
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / f"step_{step:08d}"
    checkpoint.mkdir(parents=True)
    state = {
        name: torch.zeros(shape, dtype=torch.float32)
        for name, shape in expected_lora_state_shapes(lora).items()
    }
    save_file(state, str(checkpoint / "lora.safetensors"))
    (checkpoint / "trainer_state.pt").write_bytes(b"trainer")
    for rank in range(world_size):
        (checkpoint / f"rank_{rank:02d}_state.pt").write_bytes(f"rank-{rank}".encode())
    source = _source()
    training = {
        "schema_version": SOURCE_SFT_LAUNCH_SCHEMA,
        "mode": mode,
        "stage": "development",
        "config_sha256": sha256_file(config_path),
        "authorities": config["authorities"],
        "source": source,
        "information_wall": config["information_wall"],
        "stage_contract": config["stages"]["development"],
        "trainable": {
            "object": "one_shared_multitask_pi05_lora_only",
            "per_task_adapters": 0,
            "lora_contract_sha256": canonical_contract_sha256(lora),
        },
        "runtime": {"world_size": world_size, "checkpoint_steps": [step]},
    }
    write_json_atomic(run / "run_contract.json", training)
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(checkpoint.iterdir())
    }
    manifest = {
        "schema_version": SOURCE_SFT_CHECKPOINT_SCHEMA,
        "contract_sha256": canonical_hash(training),
        "stage": "development",
        "consumed": {"next_step": step},
        "files": files,
    }
    manifest["canonical_payload_sha256"] = canonical_hash(manifest)
    write_json_atomic(checkpoint / "checkpoint_manifest.json", manifest)
    return checkpoint, source


def test_formal_development_validation_accepts_published_checkpoint_before_summary(
    tmp_path: Path,
) -> None:
    checkpoint, source = _static_adapter_fixture(
        tmp_path,
        config_path=RANK128_CONFIG,
        mode="formal",
        world_size=4,
        step=100,
    )
    validation_keys = (
        ("libero_spatial", 1),
        ("libero_spatial", 3),
        ("libero_object", 1),
        ("libero_object", 3),
        ("libero_goal", 3),
        ("libero_goal", 6),
        ("libero_10", 1),
        ("libero_10", 2),
    )
    adapter = inspect_source_sft_evaluation(
        config_path=RANK128_CONFIG,
        checkpoint=checkpoint,
        source=source,
        task_keys=validation_keys,
        evaluation_role="validation",
        require_formal=True,
    )
    assert adapter["checkpoint"]["step"] == 100
    assert adapter["training_run"]["run_summary_sha256"] is None
    assert (
        adapter["training_run"]["completion_evidence"]
        == "published_checkpoint_before_run_completion"
    )

    seen_keys = (
        ("libero_spatial", 0),
        ("libero_spatial", 2),
        ("libero_object", 5),
        ("libero_object", 2),
        ("libero_goal", 1),
        ("libero_goal", 8),
        ("libero_10", 9),
        ("libero_10", 7),
    )
    with pytest.raises(Pi05SourceSFTError, match="completed Source-SFT run summary"):
        inspect_source_sft_evaluation(
            config_path=RANK128_CONFIG,
            checkpoint=checkpoint,
            source=source,
            task_keys=seen_keys,
            evaluation_role="seen_panel",
            require_formal=True,
        )


def test_static_source_sft_adapter_is_shared_and_role_gated(tmp_path: Path) -> None:
    checkpoint, source = _static_adapter_fixture(tmp_path)
    validation_keys = (
        ("libero_spatial", 1),
        ("libero_spatial", 3),
        ("libero_object", 1),
        ("libero_object", 3),
        ("libero_goal", 3),
        ("libero_goal", 6),
        ("libero_10", 1),
        ("libero_10", 2),
    )
    adapter = inspect_source_sft_evaluation(
        config_path=CONFIG,
        checkpoint=checkpoint,
        source=source,
        task_keys=validation_keys,
        evaluation_role="validation",
        require_formal=False,
    )
    assert adapter["kind"] == "shared_source_sft_lora"
    assert adapter["execution_backend"] == "materialized_once_per_worker_batched_replan"
    assert adapter["shared_adapter_count"] == 1
    assert adapter["per_task_adapter_count"] == 0
    assert adapter["teacher_video_reads"] == adapter["test_action_reads"] == 0

    exploratory_source = {**source, "source_run_summary_sha256": None}
    exploratory = inspect_source_sft_evaluation(
        config_path=CONFIG,
        checkpoint=checkpoint,
        source=exploratory_source,
        task_keys=validation_keys,
        evaluation_role="validation",
        require_formal=False,
    )
    assert exploratory["checkpoint"] == adapter["checkpoint"]

    seen_keys = (
        ("libero_spatial", 0),
        ("libero_spatial", 2),
        ("libero_object", 5),
        ("libero_object", 2),
        ("libero_goal", 1),
        ("libero_goal", 8),
        ("libero_10", 9),
        ("libero_10", 7),
    )
    seen = inspect_source_sft_evaluation(
        config_path=CONFIG,
        checkpoint=checkpoint,
        source=source,
        task_keys=seen_keys,
        evaluation_role="seen_panel",
        require_formal=False,
    )
    assert seen["evaluation_role"] == "seen_panel"

    with pytest.raises(Pi05SourceSFTError, match="cannot be evaluated"):
        inspect_source_sft_evaluation(
            config_path=CONFIG,
            checkpoint=checkpoint,
            source=source,
            task_keys=validation_keys,
            evaluation_role="test",
            require_formal=False,
        )
    changed = {**source, "optimizer_step": 29_999}
    with pytest.raises(Pi05SourceSFTError, match="source linkage"):
        inspect_source_sft_evaluation(
            config_path=CONFIG,
            checkpoint=checkpoint,
            source=changed,
            task_keys=validation_keys,
            evaluation_role="validation",
            require_formal=False,
        )
