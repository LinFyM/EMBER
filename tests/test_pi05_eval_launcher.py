from __future__ import annotations

import fcntl
import importlib.util
import argparse
from pathlib import Path
import subprocess
import sys

import pytest

from ember.pi05_assets import Pi05EvaluationError


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
        with pytest.raises(Pi05EvaluationError, match="another PI05 evaluator launcher"):
            module.start_workers(tmp_path, resume=True)
        assert not reached_locked_body


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


def test_writer_prepare_arguments_are_all_or_none() -> None:
    module = _launcher_module()
    empty = argparse.Namespace(
        as_writer_config=None,
        as_writer_checkpoint=None,
        writer_feature_cache=None,
        writer_video_condition=None,
    )
    assert module._writer_requested(empty) is False
    partial = argparse.Namespace(
        as_writer_config=Path("config.json"),
        as_writer_checkpoint=None,
        writer_feature_cache=None,
        writer_video_condition="correct",
    )
    with pytest.raises(Pi05EvaluationError, match="requires all declared assets"):
        module._writer_requested(partial)

    rl_empty = argparse.Namespace(
        rl_writer_config=None,
        rl_writer_checkpoint=None,
        rl_writer_feature_cache=None,
        rl_writer_video_condition=None,
    )
    assert module._rl_writer_requested(rl_empty) is False
    rl_partial = argparse.Namespace(**vars(rl_empty))
    rl_partial.rl_writer_config = Path("rl.json")
    with pytest.raises(Pi05EvaluationError, match="requires all declared assets"):
        module._rl_writer_requested(rl_partial)


def test_source_sft_arguments_are_all_or_none_and_mutually_exclusive() -> None:
    module = _launcher_module()
    empty = argparse.Namespace(
        as_writer_config=None,
        as_writer_checkpoint=None,
        writer_feature_cache=None,
        writer_video_condition=None,
        rl_writer_config=None,
        rl_writer_checkpoint=None,
        rl_writer_feature_cache=None,
        rl_writer_video_condition=None,
        source_sft_config=None,
        source_sft_checkpoint=None,
    )
    assert module._adapter_requests(empty) == (None, False)
    partial = argparse.Namespace(**vars(empty))
    partial.source_sft_config = Path("source_sft.json")
    with pytest.raises(Pi05EvaluationError, match="requires all declared assets"):
        module._adapter_requests(partial)
    both = argparse.Namespace(
        as_writer_config=Path("as.json"),
        as_writer_checkpoint=Path("as-step"),
        writer_feature_cache=Path("cache"),
        writer_video_condition="correct",
        rl_writer_config=None,
        rl_writer_checkpoint=None,
        rl_writer_feature_cache=None,
        rl_writer_video_condition=None,
        source_sft_config=Path("source_sft.json"),
        source_sft_checkpoint=Path("source-sft-step"),
    )
    with pytest.raises(Pi05EvaluationError, match="mutually exclusive"):
        module._adapter_requests(both)

    as_and_rl = argparse.Namespace(**vars(empty))
    as_and_rl.as_writer_config = Path("as.json")
    as_and_rl.as_writer_checkpoint = Path("as-step")
    as_and_rl.writer_feature_cache = Path("cache")
    as_and_rl.writer_video_condition = "correct"
    as_and_rl.rl_writer_config = Path("rl.json")
    as_and_rl.rl_writer_checkpoint = Path("rl-update")
    as_and_rl.rl_writer_feature_cache = Path("cache")
    as_and_rl.rl_writer_video_condition = "correct"
    with pytest.raises(Pi05EvaluationError, match="mutually exclusive"):
        module._adapter_requests(as_and_rl)


def test_completed_queue_without_launcher_evidence_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _launcher_module()
    contract = {"contract_sha256": "a" * 64}
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
