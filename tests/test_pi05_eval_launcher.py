from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import ember.pi05_eval.preparation as preparation_module
import ember.pi05_eval.reward_credit_gate as reward_gate_module
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval import launcher as runtime_launcher


ROOT = Path(__file__).resolve().parents[1]


def _launcher_module():
    path = ROOT / "scripts/evaluate_pi05.py"
    spec = importlib.util.spec_from_file_location("ember_evaluate_pi05", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_lock_precedes_queue_or_worker_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _launcher_module()
    lock_path = tmp_path / ".launcher.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reached_locked_body = False

        def unexpected(*args, **kwargs):
            nonlocal reached_locked_body
            reached_locked_body = True

        monkeypatch.setattr(module, "_start_workers_locked", unexpected)
        with pytest.raises(
            Pi05EvaluationError, match="another PI05 evaluator launcher"
        ):
            module.start_workers(tmp_path, resume=True)
        assert not reached_locked_body


def test_launcher_uses_the_contract_replica_profiles() -> None:
    module = _launcher_module()
    parser = argparse.ArgumentParser()
    module._add_prepare_arguments(parser)
    action = next(
        value for value in parser._actions if value.dest == "replicas_per_gpu"
    )
    assert tuple(action.choices) == module.RUNTIME_REPLICA_PROFILES
    assert 6 in action.choices


def test_no_video_control_is_scoped_to_expert_manifold() -> None:
    module = _launcher_module()
    parser = argparse.ArgumentParser()
    module._add_prepare_arguments(parser)
    choices = {action.dest: action.choices for action in parser._actions}
    assert "no_video" in choices["expert_manifold_video_condition"]


def test_gpu_preflight_queries_only_explicit_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setenv("EMBER_STORAGE_ROOT", str(tmp_path))

    def run(command, **kwargs):
        del kwargs
        calls.append(list(command))
        if command[0] == "df":
            return SimpleNamespace(
                stdout="size used avail pcent target\n1000 100 900 10% /data\n"
            )
        if "--query-gpu" in " ".join(command):
            return SimpleNamespace(
                stdout=(
                    "4, GPU-four, NVIDIA A40, 0, 46068, 0, 30, 550\n"
                    "7, GPU-seven, NVIDIA A40, 0, 46068, 0, 30, 550\n"
                )
            )
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(runtime_launcher.subprocess, "run", run)
    observed = runtime_launcher.gpu_preflight((4, 7))
    gpu_calls = [call for call in calls if call[0] == "nvidia-smi"]
    assert observed["storage_root"] == str(tmp_path.resolve())
    assert observed["storage_accounting"] == (
        "filesystem_capacity_only_no_recursive_personal_scan"
    )
    assert observed["physical_gpu_ids"] == [4, 7]
    assert observed["device_names"] == ["NVIDIA A40", "NVIDIA A40"]
    assert len(gpu_calls) == 2
    assert all(call[1:3] == ["-i", "4,7"] for call in gpu_calls)
    assert not any(call[0] == "du" for call in calls)
    assert any(
        call[0] == "df" and call[-1] == str(tmp_path.resolve()) for call in calls
    )


def test_storage_root_requires_explicit_host_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EMBER_STORAGE_ROOT", raising=False)
    with pytest.raises(Pi05EvaluationError, match="EMBER_STORAGE_ROOT must be set"):
        runtime_launcher._storage_root()


def test_gpu_preflight_rejects_duplicate_or_negative_gpu_indices() -> None:
    with pytest.raises(Pi05EvaluationError, match="selection is invalid"):
        runtime_launcher.gpu_preflight((0, 0))
    with pytest.raises(Pi05EvaluationError, match="selection is invalid"):
        runtime_launcher.gpu_preflight((-1,))


def test_writer_generation_batch_size_accepts_measured_positive_values() -> None:
    module = _launcher_module()
    assert module._positive_int("100") == 100
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        module._positive_int("0")


def test_resume_validation_preserves_dynamic_k_evaluation_cardinality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _launcher_module()
    normalization = tmp_path / "normalization.json"
    normalization.write_text("{}", encoding="utf-8")
    model = {
        "source_run": str(tmp_path / "source"),
        "checkpoint": str(tmp_path / "checkpoint"),
    }
    tokenizer = {"path": str(tmp_path / "tokenizer.model")}
    adapter = {
        "kind": module.DYNAMIC_K_WRITER_KIND,
        "config": {"path": str(tmp_path / "writer.json")},
        "writer_asset": {"checkpoint": str(tmp_path / "writer-checkpoint")},
        "video_data": {"root": str(tmp_path / "videos")},
        "video_condition": "correct",
        "video_schedule": {"seed": 7, "sampling_mode": "without_replacement"},
        "information_wall": {"evaluation_k": 4},
    }
    contract = {
        "authorities": {"config_path": str(tmp_path / "evaluation.json")},
        "git": {"commit": "sealed-commit"},
        "mode": "formal",
        "role": "validation",
        "model": model,
        "tokenizer": tokenizer,
        "normalization": {
            "path": str(normalization),
            "bytes": normalization.stat().st_size,
        },
        "adapter": adapter,
        "tasks": [{"suite": "libero_spatial", "task_id": 1}],
    }
    observed_k: list[int] = []

    monkeypatch.setattr(module, "load_evaluation_authorities", lambda *args: object())
    monkeypatch.setattr(
        module,
        "git_state",
        lambda path: {"commit": "sealed-commit", "dirty_paths": []},
    )
    monkeypatch.setattr(module, "inspect_source_checkpoint", lambda *args, **kwargs: model)
    monkeypatch.setattr(module, "inspect_tokenizer", lambda *args, **kwargs: tokenizer)

    def inspect_dynamic_k(**kwargs):
        observed_k.append(kwargs["evaluation_k"])
        return adapter

    monkeypatch.setattr(module, "_inspect_dynamic_k_writer_adapter", inspect_dynamic_k)
    module._validate_resume_inputs(contract)
    assert observed_k == [4]


def _registered_reward_credit_args(
    tmp_path: Path, *, macro: int, condition: str = "correct"
) -> tuple[argparse.Namespace, Path]:
    module = _launcher_module()
    config = tmp_path / "configs/reward-credit.json"
    config.parent.mkdir()
    registered_relative = f"registered-macro{macro}-{condition}"
    control_roots = {
        name: f"registered-macro{macro}-{name}"
        for name in (
            "same_task_other",
            "cross_suite_wrong",
            "shuffled",
            "reversed",
            "no_video",
        )
    }
    decision = {}
    for candidate_macro in (1, 2):
        candidate_controls = {
            name: f"registered-macro{candidate_macro}-{name}" for name in control_roots
        }
        decision[f"macro{candidate_macro}_registered_root"] = (
            f"registered-macro{candidate_macro}-correct"
        )
        decision[f"macro{candidate_macro}_control_registered_roots"] = (
            candidate_controls
        )
    config.write_text(
        json.dumps(
            {
                "schema_version": module.V6_PRIOR_CONFIG_SCHEMA,
                "initialization": {"checkpoint": "historical"},
                "formal_run": {
                    "registered_output_root": "formal",
                    "decision_evaluation": decision,
                    "decision_gates": {
                        "first_full_six_arm_correct_min": 144,
                        "goal_full_six_arm_correct_min": 151,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    training_root = tmp_path / "formal"
    checkpoint = training_root / f"checkpoints/macro_{macro:08d}"
    checkpoint.mkdir(parents=True)
    registered = tmp_path / registered_relative
    commit = "a" * 40
    run_decision = {
        key: (
            str(tmp_path / value)
            if isinstance(value, str)
            else {name: str(tmp_path / path) for name, path in value.items()}
        )
        for key, value in decision.items()
    }
    (training_root / "run_contract.json").write_text(
        json.dumps(
            {
                "schema_version": module.V6_PRIOR_RUN_SCHEMA,
                "mode": "formal",
                "git": {"commit": commit},
                "config": {"schema": module.V6_PRIOR_CONFIG_SCHEMA},
                "decision_evaluation": run_decision,
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": module.V6_PRIOR_CHECKPOINT_SCHEMA,
                "next_macro": macro,
                "metrics_rows": macro,
                "checkpoint_contract": {
                    "run_schema": module.V6_PRIOR_RUN_SCHEMA,
                    "mode": "formal",
                    "git_commit": commit,
                    "config": {"schema": module.V6_PRIOR_CONFIG_SCHEMA},
                },
            }
        ),
        encoding="utf-8",
    )
    return (
        argparse.Namespace(
            mode="formal",
            role="validation",
            state_count=50,
            expert_manifold_video_condition=condition,
            expert_manifold_video_sampling="without_replacement",
            expert_manifold_config=config,
            expert_manifold_checkpoint=checkpoint,
        ),
        registered,
    )


def _registered_evaluation_contract(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    run = json.loads(
        (args.expert_manifold_checkpoint.parent.parent / "run_contract.json").read_text(
            encoding="utf-8"
        )
    )
    commit = run["git"]["commit"]
    condition = args.expert_manifold_video_condition
    return {
        "mode": "formal",
        "role": "validation",
        "output_dir": str(output_dir.resolve()),
        "arm": f"expert_manifold_v6_condition_residual_{condition}",
        "git": {
            "branch": "",
            "commit": commit,
            "upstream": None,
            "authority_ref": "origin/codex/bci-continuation",
            "authority_contains_commit": True,
            "dirty_paths": [],
        },
        "tasks": [
            {
                "suite": f"suite-{index // 2}",
                "task_id": index,
                "split_role": "validation",
                "init_state_ids": tuple(range(50)),
            }
            for index in range(8)
        ],
        "adapter": {
            "schema_version": (
                "ember_pi05_v6_condition_program_residual_eval_adapter_v8"
            ),
            "config": {"schema": "ember_pi05_v6_reward_credit_program_cotangent_v1"},
            "writer_asset": {
                "checkpoint": str(args.expert_manifold_checkpoint.resolve())
            },
            "video_condition": condition,
            "video_schedule": {"sampling_mode": "without_replacement"},
        },
    }


@pytest.mark.parametrize("macro", (1, 2))
def test_reward_credit_evaluator_requires_its_training_registered_root(
    tmp_path: Path, macro: int
) -> None:
    module = _launcher_module()
    args, registered = _registered_reward_credit_args(tmp_path, macro=macro)
    contract = _registered_evaluation_contract(args, registered)
    module._validate_registered_reward_credit_output(
        args, registered.resolve(), contract
    )
    wrong = tmp_path / f"unregistered-macro{macro}"
    with pytest.raises(Pi05EvaluationError, match="pre-registered root"):
        module._validate_registered_reward_credit_output(
            args,
            wrong.resolve(),
            {**contract, "output_dir": str(wrong.resolve())},
        )
    assert not wrong.exists()


@pytest.mark.parametrize(
    "condition",
    ("same_task_other", "cross_suite_wrong", "shuffled", "reversed", "no_video"),
)
def test_reward_credit_controls_require_their_canonical_registered_root(
    tmp_path: Path, condition: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _launcher_module()
    args, registered = _registered_reward_credit_args(
        tmp_path, macro=1, condition=condition
    )
    monkeypatch.setattr(
        reward_gate_module,
        "load_reward_credit_control_trigger_evidence",
        lambda **_kwargs: {"macro": 1, "correct": 144},
    )
    module._validate_registered_reward_credit_output(
        args,
        registered.resolve(),
        _registered_evaluation_contract(args, registered),
    )


def test_reward_credit_gate_rejects_tampered_training_or_run_root(
    tmp_path: Path,
) -> None:
    module = _launcher_module()
    args, registered = _registered_reward_credit_args(tmp_path, macro=1)
    run_path = args.expert_manifold_checkpoint.parent.parent / "run_contract.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["decision_evaluation"]["macro1_registered_root"] = str(tmp_path / "tampered")
    run_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(Pi05EvaluationError, match="pre-registered root"):
        module._validate_registered_reward_credit_output(
            args,
            (tmp_path / "tampered").resolve(),
            _registered_evaluation_contract(args, tmp_path / "tampered"),
        )
    run["decision_evaluation"]["macro1_registered_root"] = str(registered)
    run_path.write_text(json.dumps(run), encoding="utf-8")
    copied = tmp_path / "copied/checkpoints/macro_00000001"
    copied.mkdir(parents=True)
    (copied.parent.parent / "run_contract.json").write_text(
        json.dumps(run), encoding="utf-8"
    )
    (copied / "manifest.json").write_bytes(
        (args.expert_manifold_checkpoint / "manifest.json").read_bytes()
    )
    args.expert_manifold_checkpoint = copied
    with pytest.raises(Pi05EvaluationError, match="pre-registered root"):
        module._validate_registered_reward_credit_output(
            args,
            registered.resolve(),
            _registered_evaluation_contract(args, registered),
        )


def test_reward_credit_gate_rejects_renamed_active_checkpoint(
    tmp_path: Path,
) -> None:
    module = _launcher_module()
    args, registered = _registered_reward_credit_args(tmp_path, macro=1)
    renamed = args.expert_manifold_checkpoint.parent / "renamed-reward-cycle"
    args.expert_manifold_checkpoint.rename(renamed)
    args.expert_manifold_checkpoint = renamed
    with pytest.raises(Pi05EvaluationError, match="pre-registered root"):
        module._validate_registered_reward_credit_output(
            args,
            registered.resolve(),
            _registered_evaluation_contract(args, registered),
        )


def test_reward_credit_registered_root_gate_is_scoped_to_formal_reward_cycles(
    tmp_path: Path,
) -> None:
    module = _launcher_module()
    args, registered = _registered_reward_credit_args(tmp_path, macro=1)
    contract = _registered_evaluation_contract(args, registered)
    args.mode = "smoke"
    module._validate_registered_reward_credit_output(
        args, registered.resolve(), contract
    )
    args.mode = "formal"
    nonreward_config = tmp_path / "configs/nonreward.json"
    nonreward_config.write_text(
        json.dumps({"schema_version": "historical_writer_v1"}), encoding="utf-8"
    )
    args.expert_manifold_config = nonreward_config
    module._validate_registered_reward_credit_output(
        args, registered.resolve(), contract
    )
    args.expert_manifold_config = tmp_path / "configs/reward-credit.json"
    args.expert_manifold_video_condition = "correct"
    args.expert_manifold_checkpoint = tmp_path / "historical"
    with pytest.raises(Pi05EvaluationError, match="registration is incomplete"):
        module._validate_registered_reward_credit_output(
            args, registered.resolve(), contract
        )


def test_formal_reward_credit_rejects_parser_registered_non_six_arm_condition(
    tmp_path: Path,
) -> None:
    module = _launcher_module()
    args, registered = _registered_reward_credit_args(
        tmp_path, macro=1, condition="shuffled_keep_first"
    )
    assert "shuffled_keep_first" in module.VIDEO_CONDITIONS
    with pytest.raises(Pi05EvaluationError, match="registered six-arm condition"):
        module._validate_registered_reward_credit_output(
            args,
            registered.resolve(),
            _registered_evaluation_contract(args, registered),
        )


@pytest.mark.parametrize(
    "mutation",
    ("role", "sampling", "commit", "tasks"),
)
def test_reward_credit_registration_binds_the_full_formal_invocation(
    tmp_path: Path,
    mutation: str,
) -> None:
    module = _launcher_module()
    args, registered = _registered_reward_credit_args(tmp_path, macro=1)
    contract = _registered_evaluation_contract(args, registered)
    if mutation == "role":
        contract["role"] = "test"
    elif mutation == "sampling":
        contract["adapter"]["video_schedule"]["sampling_mode"] = "with_replacement"
    elif mutation == "commit":
        contract["git"]["commit"] = "b" * 40
    else:
        contract["tasks"] = contract["tasks"][:-1]
    with pytest.raises(Pi05EvaluationError, match="pre-registered root"):
        module._validate_registered_reward_credit_output(
            args, registered.resolve(), contract
        )


def test_prepare_failure_never_publishes_the_canonical_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _launcher_module()
    output = tmp_path / "canonical"
    args = argparse.Namespace(output_dir=output, config=tmp_path / "config.json")
    monkeypatch.setattr(
        preparation_module, "adapter_requests", lambda _args: (None, False)
    )

    def reject(*_args, **_kwargs):
        raise Pi05EvaluationError("preflight rejected")

    monkeypatch.setattr(preparation_module, "load_evaluation_authorities", reject)
    with pytest.raises(Pi05EvaluationError, match="preflight rejected"):
        module.prepare_run(args)
    assert not output.exists()
    assert not tuple(tmp_path.glob(".canonical.prepare-*"))

    output.mkdir()
    with pytest.raises(Pi05EvaluationError, match="already exists"):
        module.prepare_run(args)
    assert output.is_dir()


def test_prepare_adapter_failure_releases_its_output_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _launcher_module()
    output = tmp_path / "canonical"
    args = argparse.Namespace(output_dir=output)

    def reject(_args: argparse.Namespace) -> tuple[None, bool]:
        raise Pi05EvaluationError("adapter rejected")

    monkeypatch.setattr(preparation_module, "adapter_requests", reject)
    with pytest.raises(Pi05EvaluationError, match="adapter rejected"):
        module.prepare_run(args)
    assert not output.exists()
    assert not (tmp_path / ".canonical.prepare.lock").exists()
    assert not tuple(tmp_path.glob(".canonical.prepare-*"))


def test_prepare_publishes_one_complete_staged_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _launcher_module()
    output = tmp_path / "canonical"
    args = argparse.Namespace(output_dir=output)
    monkeypatch.setattr(
        preparation_module, "adapter_requests", lambda _args: (None, False)
    )

    def payload(_args, *, staging: Path, **_kwargs):
        config = staging / "libero_config/config.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("assets: /sealed\n", encoding="utf-8")
        contract = {"contract_reference": "contract-a"}
        summary = {"event": "prepared", "output_dir": str(output.resolve())}
        return contract, (object(),), summary

    def initialize(path: Path, *_args, **_kwargs) -> None:
        path.write_bytes(b"queue")

    monkeypatch.setattr(preparation_module, "_prepared_payload", payload)
    monkeypatch.setattr(preparation_module, "initialize_queue", initialize)
    summary = module.prepare_run(args)
    assert summary["event"] == "prepared"
    assert json.loads((output / "run_contract.json").read_text(encoding="utf-8")) == {
        "contract_reference": "contract-a"
    }
    assert (output / "queue.sqlite3").read_bytes() == b"queue"
    assert (output / "libero_config/config.yaml").is_file()
    assert Path(os.environ["LIBERO_CONFIG_PATH"]) == output / "libero_config"
    assert not tuple(tmp_path.glob(".canonical.prepare-*"))
    assert not (tmp_path / ".canonical.prepare.lock").exists()


def test_staging_publish_failure_preserves_a_competing_final_root(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".canonical.prepare-owned"
    output = tmp_path / "canonical"
    staging.mkdir()
    (staging / "owned").write_text("owned\n", encoding="utf-8")
    output.mkdir()
    (output / "competing").write_text("competing\n", encoding="utf-8")
    lock = preparation_module._claim_prepare_lock(output)
    with pytest.raises(Pi05EvaluationError, match="already exists"):
        preparation_module._publish_staging(staging, output, lock=lock)
    assert (output / "competing").read_text(encoding="utf-8") == "competing\n"
    assert (staging / "owned").read_text(encoding="utf-8") == "owned\n"


def test_staging_publish_never_replaces_a_competing_empty_root(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".canonical.prepare-owned"
    output = tmp_path / "canonical"
    staging.mkdir()
    (staging / "owned").write_text("owned\n", encoding="utf-8")
    output.mkdir()
    lock = preparation_module._claim_prepare_lock(output)
    with pytest.raises(Pi05EvaluationError, match="already exists"):
        preparation_module._publish_staging(staging, output, lock=lock)
    assert output.is_dir()
    assert not tuple(output.iterdir())
    assert (staging / "owned").read_text(encoding="utf-8") == "owned\n"


def test_prepare_lock_is_exclusive_across_preparers(tmp_path: Path) -> None:
    output = tmp_path / "canonical"
    lock = preparation_module._claim_prepare_lock(output)
    with pytest.raises(Pi05EvaluationError, match="owns the output lock"):
        preparation_module._claim_prepare_lock(output)
    assert lock == tmp_path / ".canonical.prepare.lock"


def test_partial_launch_cleanup_stops_only_owned_processes() -> None:
    module = _launcher_module()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        module._terminate_owned_workers({"0-r0": process})
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_expert_manifold_prepare_arguments_are_all_or_none() -> None:
    module = _launcher_module()
    empty = argparse.Namespace(
        source_sft_config=None,
        source_sft_checkpoint=None,
        task_expert_config=None,
        task_expert_bank_root=None,
        task_expert_step=None,
        expert_manifold_config=None,
        expert_manifold_checkpoint=None,
        expert_manifold_video_data_root=None,
        expert_manifold_video_condition=None,
    )
    assert module._adapter_requests(empty) == (None, False)
    partial = argparse.Namespace(**vars(empty))
    partial.expert_manifold_config = Path("config.json")
    partial.expert_manifold_video_condition = "correct"
    with pytest.raises(Pi05EvaluationError, match="requires all declared assets"):
        module._adapter_requests(partial)


def test_source_sft_arguments_are_all_or_none_and_mutually_exclusive() -> None:
    module = _launcher_module()
    empty = argparse.Namespace(
        source_sft_config=None,
        source_sft_checkpoint=None,
        task_expert_config=None,
        task_expert_bank_root=None,
        task_expert_step=None,
        expert_manifold_config=None,
        expert_manifold_checkpoint=None,
        expert_manifold_video_data_root=None,
        expert_manifold_video_condition=None,
    )
    assert module._adapter_requests(empty) == (None, False)
    partial = argparse.Namespace(**vars(empty))
    partial.source_sft_config = Path("source_sft.json")
    with pytest.raises(Pi05EvaluationError, match="requires all declared assets"):
        module._adapter_requests(partial)
    both = argparse.Namespace(
        source_sft_config=Path("source_sft.json"),
        source_sft_checkpoint=Path("source-sft-step"),
        task_expert_config=None,
        task_expert_bank_root=None,
        task_expert_step=None,
        expert_manifold_config=Path("expert-manifold.json"),
        expert_manifold_checkpoint=Path("writer-checkpoint"),
        expert_manifold_video_data_root=Path("videos"),
        expert_manifold_video_condition="correct",
    )
    with pytest.raises(Pi05EvaluationError, match="mutually exclusive"):
        module._adapter_requests(both)

    expert = argparse.Namespace(**vars(empty))
    expert.task_expert_config = Path("expert.json")
    expert.task_expert_bank_root = Path("bank")
    expert.task_expert_step = 1000
    assert module._adapter_requests(expert) == ("task_expert", False)
    expert.source_sft_config = Path("source_sft.json")
    expert.source_sft_checkpoint = Path("source-sft-step")
    with pytest.raises(Pi05EvaluationError, match="mutually exclusive"):
        module._adapter_requests(expert)

    manifold = argparse.Namespace(**vars(empty))
    manifold.expert_manifold_config = Path("expert-manifold.json")
    manifold.expert_manifold_checkpoint = Path("writer-checkpoint")
    manifold.expert_manifold_video_data_root = Path("videos")
    manifold.expert_manifold_video_condition = "correct"
    assert module._adapter_requests(manifold) == ("expert_manifold_writer", False)


def test_retired_expert_manifold_deployment_assets_fail_closed() -> None:
    module = _launcher_module()
    args = argparse.Namespace(
        source_sft_config=None,
        source_sft_checkpoint=None,
        task_expert_config=None,
        task_expert_bank_root=None,
        task_expert_step=None,
        expert_manifold_config=Path("expert-manifold.json"),
        expert_manifold_checkpoint=Path("writer-checkpoint"),
        expert_manifold_video_data_root=Path("videos"),
        expert_manifold_video_condition="correct",
        expert_manifold_expert_bank_root=Path("experts"),
        expert_manifold_feature_cache_root=None,
    )
    with pytest.raises(Pi05EvaluationError, match="assets are retired"):
        module._adapter_requests(args)


def test_completed_queue_without_launcher_evidence_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _launcher_module()
    contract = {"contract_reference": "contract-a"}
    monkeypatch.setattr(module, "_active_worker_pids", lambda output_dir: [])
    monkeypatch.setattr(module, "load_run_contract", lambda path: contract)
    monkeypatch.setattr(module, "_validate_resume_inputs", lambda value: None)
    monkeypatch.setattr(module, "_shards_from_contract", lambda value: (object(),))
    monkeypatch.setattr(module, "initialize_queue", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "queue_summary",
        lambda path: {"status_counts": {"complete": 1}},
    )
    with pytest.raises(Pi05EvaluationError, match="without exact launcher"):
        module._recover_locked_queue(tmp_path, resume=True)
