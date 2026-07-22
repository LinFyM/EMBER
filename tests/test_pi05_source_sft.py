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
    _target_tasks,
    load_source_sft_config,
)
from ember.source_sft.inference import inspect_source_sft_evaluation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pi05_source_sft_development_v1.json"


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


def test_development_formal_budget_matches_selected_as_writer_query_scale() -> None:
    config = load_source_sft_config(CONFIG)
    formal = config["stages"]["development"]["formal_run"]
    budget = formal["matched_budget"]
    assert formal["status"] == "sealed"
    assert formal["expected_world_size"] == 8
    assert formal["total_steps"] == 63
    assert formal["per_rank_batch_size"] == 64
    assert formal["checkpoint_steps"] == [63]
    assert budget["as_writer_global_action_queries"] == 32_000
    assert budget["source_sft_global_action_queries"] == 32_256
    assert (
        formal["total_steps"]
        * formal["expected_world_size"]
        * formal["per_rank_batch_size"]
        == budget["source_sft_global_action_queries"]
    )


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


def _static_adapter_fixture(tmp_path: Path) -> tuple[Path, dict]:
    config = load_source_sft_config(CONFIG)
    lora = load_pi05_lora_contract(ROOT / config["authorities"]["lora_contract"]["path"])
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "step_00000004"
    checkpoint.mkdir(parents=True)
    state = {
        name: torch.zeros(shape, dtype=torch.float32)
        for name, shape in expected_lora_state_shapes(lora).items()
    }
    save_file(state, str(checkpoint / "lora.safetensors"))
    (checkpoint / "trainer_state.pt").write_bytes(b"trainer")
    for rank in range(8):
        (checkpoint / f"rank_{rank:02d}_state.pt").write_bytes(f"rank-{rank}".encode())
    source = _source()
    training = {
        "schema_version": SOURCE_SFT_LAUNCH_SCHEMA,
        "mode": "profile",
        "stage": "development",
        "config_sha256": sha256_file(CONFIG),
        "authorities": config["authorities"],
        "source": source,
        "information_wall": config["information_wall"],
        "stage_contract": config["stages"]["development"],
        "trainable": {
            "object": "one_shared_multitask_pi05_lora_only",
            "per_task_adapters": 0,
            "lora_contract_sha256": canonical_contract_sha256(lora),
        },
        "runtime": {"world_size": 8, "checkpoint_steps": [4]},
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
        "consumed": {"next_step": 4},
        "files": files,
    }
    manifest["canonical_payload_sha256"] = canonical_hash(manifest)
    write_json_atomic(checkpoint / "checkpoint_manifest.json", manifest)
    return checkpoint, source


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
