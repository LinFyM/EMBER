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
import ember.pi05_eval.recovery as recovery_module
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


def test_video_control_choices_cover_the_canonical_writer_backend() -> None:
    module = _launcher_module()
    parser = argparse.ArgumentParser()
    module._add_prepare_arguments(parser)
    choices = {action.dest: action.choices for action in parser._actions}
    assert set(choices["dynamic_k_writer_video_condition"]) == {
        "correct",
        "cross_suite_wrong",
        "endpoints_middle_shuffled",
        "final_frame_only",
        "first_final",
        "first_frame_only",
        "monotone_sparse",
        "no_video",
        "reversed",
        "same_task_other",
        "shuffled",
        "shuffled_keep_first",
    }
    assert {"language_only", "video_only", "wrong_task"}.issubset(
        choices["functional_writer_video_condition"]
    )


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
    assert runtime_launcher.evaluator_gpus_are_eligible(observed)
    assert len(gpu_calls) == 2
    assert all(call[1:3] == ["-i", "4,7"] for call in gpu_calls)
    assert not any(call[0] == "du" for call in calls)
    assert any(
        call[0] == "df" and call[-1] == str(tmp_path.resolve()) for call in calls
    )


@pytest.mark.parametrize(
    ("memory_used_mib", "utilization_percent", "eligible"),
    ((349, 0, True), (4_939, 0, True), (8_193, 0, False), (349, 11, False)),
)
def test_evaluator_gpu_admission_allows_only_safe_low_load_coscheduling(
    memory_used_mib: int,
    utilization_percent: int,
    eligible: bool,
) -> None:
    preflight = {
        "physical_gpu_ids": [1],
        "gpu_telemetry": [
            {
                "physical_gpu": 1,
                "uuid": "GPU-one",
                "memory_used_mib": memory_used_mib,
                "memory_total_mib": 46_068,
                "utilization_percent": utilization_percent,
            }
        ],
        "compute_applications": ["GPU-one, 123, python, 262, owner=other"],
    }
    assert runtime_launcher.evaluator_gpus_are_eligible(preflight) is eligible


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


def test_writer_profile_accepts_ordered_positive_candidate_batches() -> None:
    module = _launcher_module()
    assert module._profile_batch_sizes("8,16,32") == (8, 16, 32)
    assert module._profile_batch_sizes("1,2,4") == (1, 2, 4)
    with pytest.raises(Pi05EvaluationError, match="batch sizes are invalid"):
        module._profile_batch_sizes("8,8,32")
    with pytest.raises(Pi05EvaluationError, match="batch sizes are invalid"):
        module._profile_batch_sizes("0,2,4")


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

    monkeypatch.setattr(
        recovery_module, "load_evaluation_authorities", lambda *args: object()
    )
    monkeypatch.setattr(
        recovery_module,
        "git_state",
        lambda path: {"commit": "sealed-commit", "dirty_paths": []},
    )
    monkeypatch.setattr(
        recovery_module, "inspect_source_checkpoint", lambda *args, **kwargs: model
    )
    monkeypatch.setattr(
        recovery_module, "inspect_tokenizer", lambda *args, **kwargs: tokenizer
    )

    def inspect_dynamic_k(**kwargs):
        observed_k.append(kwargs["evaluation_k"])
        return adapter

    monkeypatch.setattr(
        recovery_module, "inspect_dynamic_k_writer_adapter", inspect_dynamic_k
    )
    module._validate_resume_inputs(contract)
    assert observed_k == [4]


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


def _empty_adapter_args() -> argparse.Namespace:
    return argparse.Namespace(
        source_sft_config=None,
        source_sft_checkpoint=None,
        task_expert_config=None,
        task_expert_bank_root=None,
        task_expert_step=None,
        dynamic_k_writer_config=None,
        dynamic_k_writer_checkpoint=None,
        dynamic_k_writer_video_data_root=None,
        dynamic_k_writer_video_condition=None,
        functional_writer_config=None,
        functional_writer_checkpoint=None,
        functional_writer_video_data_root=None,
        functional_writer_video_condition=None,
    )


def test_dynamic_k_writer_prepare_arguments_are_all_or_none() -> None:
    module = _launcher_module()
    empty = _empty_adapter_args()
    assert module._adapter_requests(empty) == (None, False)
    partial = argparse.Namespace(**vars(empty))
    partial.dynamic_k_writer_config = Path("config.json")
    partial.dynamic_k_writer_video_condition = "correct"
    with pytest.raises(Pi05EvaluationError, match="requires all declared assets"):
        module._adapter_requests(partial)

    complete = argparse.Namespace(**vars(empty))
    complete.dynamic_k_writer_config = Path("config.json")
    complete.dynamic_k_writer_checkpoint = Path("writer-checkpoint")
    complete.dynamic_k_writer_video_data_root = Path("videos")
    complete.dynamic_k_writer_video_condition = "correct"
    assert module._adapter_requests(complete) == (
        "layer_matched_memory_program_compiler_writer",
        False,
    )

    functional = argparse.Namespace(**vars(empty))
    functional.functional_writer_config = Path("config.json")
    functional.functional_writer_checkpoint = Path("writer-checkpoint")
    functional.functional_writer_video_data_root = Path("videos")
    functional.functional_writer_video_condition = "correct"
    assert module._adapter_requests(functional) == (
        "fixed_functional_code_writer",
        False,
    )


def test_source_task_expert_and_writer_adapters_are_mutually_exclusive() -> None:
    module = _launcher_module()
    empty = _empty_adapter_args()
    partial = argparse.Namespace(**vars(empty))
    partial.source_sft_config = Path("source_sft.json")
    with pytest.raises(Pi05EvaluationError, match="requires all declared assets"):
        module._adapter_requests(partial)

    expert = argparse.Namespace(**vars(empty))
    expert.task_expert_config = Path("expert.json")
    expert.task_expert_bank_root = Path("bank")
    expert.task_expert_step = 1000
    assert module._adapter_requests(expert) == ("task_expert", False)
    expert.source_sft_config = Path("source_sft.json")
    expert.source_sft_checkpoint = Path("source-sft-step")
    with pytest.raises(Pi05EvaluationError, match="mutually exclusive"):
        module._adapter_requests(expert)


def test_prepare_binds_projection_manifest_to_task_expert_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projection = tmp_path / "projection.json"
    projection.write_text("{}\n", encoding="utf-8")
    args = argparse.Namespace(
        source_sft_config=tmp_path / "source.json",
        source_sft_checkpoint=tmp_path / "source",
        task_expert_config=tmp_path / "expert.json",
        task_expert_bank_root=tmp_path / "bank",
        task_expert_step=2000,
        task_expert_projection_manifest=projection,
        role="development_train",
        mode="screen",
    )
    observed: list[dict[str, object]] = []

    def inspect_source(**kwargs):
        observed.append(kwargs)
        return {"source": True}

    def inspect_expert(**kwargs):
        observed.append(kwargs)
        return {"expert": True}

    monkeypatch.setattr(preparation_module, "inspect_source_sft_adapter", inspect_source)
    monkeypatch.setattr(preparation_module, "inspect_task_expert_adapter", inspect_expert)
    preparation_module._inspect_adapter(
        args,
        writer_kind=None,
        source_sft_requested=True,
        authorities=object(),
        model={},
        tasks=(),
    )
    assert "projection_manifest" not in observed[-1]
    preparation_module._inspect_adapter(
        args,
        writer_kind="task_expert",
        source_sft_requested=False,
        authorities=object(),
        model={},
        tasks=(),
    )
    assert observed[-1]["projection_manifest"] == projection.resolve()


def test_resume_rebinds_projected_task_expert_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _launcher_module()
    projection = tmp_path / "projection.json"
    projection.write_text("{}\n", encoding="utf-8")
    normalization = tmp_path / "normalization.json"
    normalization.write_text("{}\n", encoding="utf-8")
    adapter = {
        "kind": "task_local_expert_bank",
        "config": {"path": str(tmp_path / "expert.json")},
        "bank_root": str(tmp_path / "bank"),
        "step": 2000,
        "projection": {"manifest_path": str(projection)},
    }
    model = {
        "model": True,
        "source_run": str(tmp_path / "source"),
        "checkpoint": str(tmp_path / "source/checkpoint"),
    }
    tokenizer = {"tokenizer": True, "path": str(tmp_path / "tokenizer.model")}
    contract = {
        "authorities": {"config_path": str(tmp_path / "eval.json")},
        "git": {"commit": "commit"},
        "mode": "screen",
        "role": "development_train",
        "role_authority": None,
        "model": model,
        "tokenizer": tokenizer,
        "normalization": {
            "path": str(normalization),
            "bytes": normalization.stat().st_size,
        },
        "tasks": [{"suite": "libero_spatial", "task_id": 0}],
        "adapter": adapter,
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        recovery_module,
        "load_evaluation_authorities",
        lambda *args: SimpleNamespace(),
    )
    monkeypatch.setattr(
        recovery_module,
        "git_state",
        lambda root: {"commit": "commit", "dirty_paths": []},
    )
    monkeypatch.setattr(
        recovery_module, "inspect_source_checkpoint", lambda *a, **k: model
    )
    monkeypatch.setattr(
        recovery_module, "inspect_tokenizer", lambda *a, **k: tokenizer
    )

    def inspect_expert(**kwargs):
        captured.update(kwargs)
        return adapter

    monkeypatch.setattr(recovery_module, "inspect_task_expert_adapter", inspect_expert)
    module._validate_resume_inputs(contract)
    assert captured["projection_manifest"] == projection


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
