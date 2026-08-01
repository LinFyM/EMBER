from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.pi05_source_checkpoint import read_json
from ember.writer.model import WriterModelError
from ember.writer.ucp_analysis_run import record_local_failure, seal_local_rows


def _load_analysis_script() -> object:
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_as_writer_ucp.py"
    spec = importlib.util.spec_from_file_location("test_ucp_analysis_run_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analysis_reference_failure_keeps_rank_task_and_reference_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_analysis_script()
    monkeypatch.setattr(
        script,
        "probe_reference",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("sentinel failure")),
    )
    context = SimpleNamespace(rank=2, world_size=4, device=torch.device("cpu"))
    task = {
        "suite": "libero_goal", "task_id": 3, "global_task_id": 23,
    }

    with pytest.raises(WriterModelError) as caught:
        script._analyze_local_tasks(
            args=SimpleNamespace(references_per_task=50),
            context=context,
            tasks=(task, task, task),
            adapters={},
            store=SimpleNamespace(),
            task_authorities=(SimpleNamespace(task_id=23),),
            policy=SimpleNamespace(),
            writer=SimpleNamespace(),
            identity={},
            lora=SimpleNamespace(),
            tokenizer=SimpleNamespace(),
            processor=SimpleNamespace(),
        )

    message = str(caught.value)
    assert "rank=2" in message
    assert "suite=libero_goal" in message
    assert "global_task_id=23" in message
    assert "reference_ordinal=0" in message
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_seal_rows_writes_before_waiting_on_gloo_control_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import ember.writer.ucp_analysis_run as analysis_run

    events = []
    control_group = object()

    def barrier(*, group: object) -> None:
        events.append(("barrier", group))
        assert (tmp_path / "rows_rank_01.json").is_file()

    monkeypatch.setattr(analysis_run.dist, "barrier", barrier)
    seal_local_rows(
        tmp_path,
        SimpleNamespace(rank=1, world_size=4),
        ({"finite": 1.0},),
        control_group,
    )

    assert events == [("barrier", control_group)]
    assert read_json(tmp_path / "rows_rank_01.json") == {
        "rank": 1, "rows": [{"finite": 1.0}],
    }


def test_rank_local_failure_is_persisted_without_collective(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import ember.writer.ucp_analysis_run as analysis_run

    def unexpected_collective(*_args: object, **_kwargs: object) -> None:
        pytest.fail("rank-local failure entered a collective")

    monkeypatch.setattr(analysis_run.dist, "barrier", unexpected_collective)
    monkeypatch.setattr(
        analysis_run.dist, "broadcast_object_list", unexpected_collective,
    )
    try:
        raise RuntimeError("reference sentinel")
    except RuntimeError as error:
        record_local_failure(tmp_path, 3, error)

    failure = read_json(tmp_path / "failure_rank_03.json")
    assert failure["rank"] == 3
    assert "reference sentinel" in failure["error"]
    assert "RuntimeError: reference sentinel" in failure["traceback"]
