from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

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


def test_gpu_preflight_rejects_more_than_the_owner_six_gpu_limit() -> None:
    with pytest.raises(Pi05EvaluationError, match="six-GPU limit"):
        runtime_launcher.gpu_preflight(tuple(range(7)))


def test_writer_generation_batch_size_accepts_measured_positive_values() -> None:
    module = _launcher_module()
    assert module._positive_int("100") == 100
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        module._positive_int("0")


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
    config.write_text(
        json.dumps(
            {
                "schema_version": module.V6_PRIOR_CONFIG_SCHEMA,
                "initialization": {"checkpoint": "historical"},
                "formal_run": {
                    "registered_output_root": "formal",
                    "decision_evaluation": {
                        f"macro{macro}_registered_root": (
                            f"registered-macro{macro}-correct"
                        ),
                        f"macro{macro}_control_registered_roots": control_roots,
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
    (training_root / "run_contract.json").write_text(
        json.dumps(
            {
                "schema_version": module.V6_PRIOR_RUN_SCHEMA,
                "mode": "formal",
                "git": {"commit": commit},
                "config": {"schema": module.V6_PRIOR_CONFIG_SCHEMA},
                "decision_evaluation": {
                    f"macro{macro}_registered_root": str(
                        tmp_path / f"registered-macro{macro}-correct"
                    ),
                    f"macro{macro}_control_registered_roots": {
                        name: str(tmp_path / path)
                        for name, path in control_roots.items()
                    },
                },
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
            expert_manifold_video_condition=condition,
            expert_manifold_config=config,
            expert_manifold_checkpoint=checkpoint,
        ),
        registered,
    )


@pytest.mark.parametrize("macro", (1, 2))
def test_reward_credit_evaluator_requires_its_training_registered_root(
    tmp_path: Path, macro: int
) -> None:
    module = _launcher_module()
    args, registered = _registered_reward_credit_args(tmp_path, macro=macro)
    module._validate_registered_reward_credit_output(args, registered.resolve())
    wrong = tmp_path / f"unregistered-macro{macro}"
    with pytest.raises(Pi05EvaluationError, match="pre-registered root"):
        module._validate_registered_reward_credit_output(args, wrong.resolve())
    assert not wrong.exists()


@pytest.mark.parametrize(
    "condition",
    ("same_task_other", "cross_suite_wrong", "shuffled", "reversed", "no_video"),
)
def test_reward_credit_controls_require_their_canonical_registered_root(
    tmp_path: Path, condition: str
) -> None:
    module = _launcher_module()
    args, registered = _registered_reward_credit_args(
        tmp_path, macro=1, condition=condition
    )
    module._validate_registered_reward_credit_output(args, registered.resolve())


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
            args, (tmp_path / "tampered").resolve()
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
        module._validate_registered_reward_credit_output(args, registered.resolve())


def test_reward_credit_gate_rejects_renamed_active_checkpoint(
    tmp_path: Path,
) -> None:
    module = _launcher_module()
    args, registered = _registered_reward_credit_args(tmp_path, macro=1)
    renamed = args.expert_manifold_checkpoint.parent / "renamed-reward-cycle"
    args.expert_manifold_checkpoint.rename(renamed)
    args.expert_manifold_checkpoint = renamed
    with pytest.raises(Pi05EvaluationError, match="pre-registered root"):
        module._validate_registered_reward_credit_output(args, registered.resolve())


def test_reward_credit_registered_root_gate_is_scoped_to_formal_reward_cycles(
    tmp_path: Path,
) -> None:
    module = _launcher_module()
    args, registered = _registered_reward_credit_args(tmp_path, macro=1)
    args.mode = "smoke"
    module._validate_registered_reward_credit_output(args, registered.resolve())
    args.mode = "formal"
    args.expert_manifold_video_condition = "unsupported"
    module._validate_registered_reward_credit_output(args, registered.resolve())
    args.expert_manifold_video_condition = "correct"
    args.expert_manifold_checkpoint = tmp_path / "historical"
    module._validate_registered_reward_credit_output(args, registered.resolve())


def test_writer_profile_requires_the_canonical_batch_floor() -> None:
    module = _launcher_module()
    assert module._profile_batch_sizes("8,16,32") == (8, 16, 32)
    with pytest.raises(Pi05EvaluationError, match="batch sizes are invalid"):
        module._profile_batch_sizes("1,8,16")
    with pytest.raises(Pi05EvaluationError, match="batch sizes are invalid"):
        module._profile_batch_sizes("8,16,24")


def test_writer_profile_launch_isolated_to_one_physical_gpu(tmp_path: Path) -> None:
    module = _launcher_module()
    command, environment = module._profile_worker_launch(
        output_dir=tmp_path,
        contract={
            "parallel": {
                "replicas_per_gpu": 1,
                "omp_threads_per_worker": {"1": 12},
            }
        },
        physical_gpu=5,
        batch_sizes=(8, 16, 32),
        warmup_runs=1,
        measured_runs=2,
    )
    assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert environment["CUDA_VISIBLE_DEVICES"] == "5"
    assert environment["OMP_NUM_THREADS"] == "12"
    assert command[2:7] == [
        "profile-writer-worker",
        "--output-dir",
        str(tmp_path.resolve()),
        "--worker-id",
        "5-r0",
    ]
    assert command[-6:] == [
        "--profile-batch-sizes",
        "8,16,32",
        "--profile-warmup-runs",
        "1",
        "--profile-measured-runs",
        "2",
    ]


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
