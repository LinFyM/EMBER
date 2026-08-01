from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lerobot.utils.constants import ACTION

import ember.writer.endpoint_runtime as endpoint_runtime
from ember.pi05_source_checkpoint import canonical_hash, write_json_atomic
from ember.writer.endpoint_validation import (
    ENDPOINT_NOISE_SCHEMA,
    INFERENCE_TIMES,
    METRICS,
    SEALED_PANEL_PAYLOAD_SHA256,
    EndpointCandidate,
    _candidate_task_map,
    _validate_correct400_pairing,
    endpoint_metric_rows,
    endpoint_noise,
    endpoint_noise_seed,
    endpoint_schedule,
    exact_endpoint_actions,
    parse_endpoint_candidate_specs,
)
from ember.writer.endpoint_runtime import (
    _broadcast_rank_zero_validation,
    _predict_and_teacher_bridge,
    _task_authorities,
    _teacher_bridge_grid_losses,
    _validate_device_scope,
    _validate_endpoint_output_rows,
)
from ember.writer.model import WriterModelError
from ember.writer.validation import finalize_args


def _row(**overrides: int) -> dict[str, int]:
    result = {
        "global_task_id": 1,
        "video_group": 2,
        "query_ordinal": 3,
        "action_demo_index": 4,
        "action_frame_index": 5,
    }
    result.update(overrides)
    return result


def _candidate(candidate_id: str, family: str, cursor: int) -> EndpointCandidate:
    pairing = {"panel": "paired"}
    return EndpointCandidate(
        family=family,
        candidate_id=candidate_id,
        checkpoint_cursor=cursor,
        correct400=0,
        task_breadth=0,
        correct400_per_task=(),
        outcome_pairing_payload=pairing,
        outcome_pairing_sha256=canonical_hash(pairing),
        evaluation_root=Path("/tmp") / candidate_id,
        run_contract_file_sha256="1" * 64,
        run_contract_sha256="2" * 64,
        results_file_sha256="3" * 64,
        cache_manifest_file_sha256="4" * 64,
        entries={},
    )


def _panel_row(ordinal: int, video_group: int) -> dict[str, int]:
    return {
        "ordinal": ordinal,
        "global_task_id": 1,
        "video_group": video_group,
        "teacher_demo_index": video_group,
        "query_ordinal": 0,
        "action_demo_index": video_group + 10,
        "action_frame_index": 3,
        "dataset_row_index": ordinal,
        "policy_noise_seed": 7,
    }


def _output_row(
    candidate: EndpointCandidate,
    panel_row: dict[str, int],
) -> dict[str, object]:
    return {
        **panel_row,
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "checkpoint_cursor": candidate.checkpoint_cursor,
        "endpoint_noise_seed": endpoint_noise_seed(
            SEALED_PANEL_PAYLOAD_SHA256, panel_row
        ),
        "suite": "libero_spatial",
        "suite_task_id": 1,
        "group_wall_seconds": 0.1,
        "rank": 0,
        "metrics": {
            name: {
                "mse": 1.0,
                "per_action_dimension_mse": [1.0] * 7,
            }
            for name in METRICS
        },
    }


def test_endpoint_schedule_and_noise_are_exact_row_addressed_cpu_float32() -> None:
    times, dt = endpoint_schedule()
    assert times == INFERENCE_TIMES
    assert times == pytest.approx((1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1))
    assert dt == -0.1
    with pytest.raises(WriterModelError, match="exactly ten"):
        endpoint_schedule(9)

    rows = (_row(), _row(query_ordinal=6))
    left, seeds = endpoint_noise(SEALED_PANEL_PAYLOAD_SHA256, rows)
    right, right_seeds = endpoint_noise(SEALED_PANEL_PAYLOAD_SHA256, rows)
    assert left.device.type == "cpu"
    assert left.dtype == torch.float32
    assert left.shape == (2, 50, 32)
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert seeds == right_seeds
    assert seeds[0] != seeds[1]
    assert seeds[0] == endpoint_noise_seed(SEALED_PANEL_PAYLOAD_SHA256, rows[0])
    assert ENDPOINT_NOISE_SCHEMA in json.dumps(
        [ENDPOINT_NOISE_SCHEMA, SEALED_PANEL_PAYLOAD_SHA256]
    )


def test_endpoint_metrics_separate_padding_and_execution_horizons() -> None:
    predicted = torch.zeros((2, 50, 7))
    teacher = torch.zeros_like(predicted)
    teacher[0, :, :] = torch.arange(1, 51, dtype=torch.float32)[:, None]
    teacher[1, 5:, :] = 10.0
    padding = torch.ones((2, 50), dtype=torch.bool)
    padding[0, :3] = False
    padding[1, :8] = False
    grid = torch.full_like(predicted, 2.0)

    rows = endpoint_metric_rows(predicted, teacher, padding, grid)
    assert rows[0]["valid_action_steps"] == 3
    assert rows[0]["executed5_valid_steps"] == 3
    assert rows[1]["executed5_valid_steps"] == 5
    assert rows[1]["prefix10_valid_steps"] == 8
    first = rows[0]["metrics"]
    assert first[METRICS[0]]["mse"] == pytest.approx((1.0 + 4.0 + 9.0) / 3)
    assert first[METRICS[2]]["mse"] == pytest.approx((1.0 + 4.0 + 9.0) / 3)
    assert first[METRICS[1]]["mse"] == pytest.approx(
        sum(value * value for value in range(1, 51)) / 50
    )
    assert first[METRICS[4]]["mse"] == 2.0
    second = rows[1]["metrics"]
    assert second[METRICS[0]]["mse"] == 0.0
    assert second[METRICS[2]]["mse"] == pytest.approx(300.0 / 8.0)
    assert second[METRICS[1]]["mse"] == 90.0
    assert len(first[METRICS[0]]["per_action_dimension_mse"]) == 7


def test_endpoint_metrics_fail_closed_on_empty_valid_prefix() -> None:
    value = torch.zeros((1, 50, 7))
    with pytest.raises(WriterModelError, match="without valid"):
        endpoint_metric_rows(
            value,
            value,
            torch.ones((1, 50), dtype=torch.bool),
            value,
        )


def test_teacher_bridge_control_uses_the_exact_ten_point_grid() -> None:
    class Model:
        def __init__(self) -> None:
            self.times: list[float] = []

        def forward(self, _images, _image_masks, _tokens, _masks, actions, _noise, times):
            self.times.append(float(times[0]))
            return torch.ones_like(actions) * float(times[0])

    class Policy:
        def __init__(self) -> None:
            self.model = Model()

        def _preprocess_images(self, _batch):
            return [torch.zeros(1)], [torch.ones(1, dtype=torch.bool)]

        def prepare_action(self, batch):
            result = torch.zeros((batch["action"].shape[0], 50, 32))
            result[:, :, :7] = batch["action"]
            return result

    policy = Policy()
    batch = {
        "observation.language.tokens": torch.zeros((2, 4), dtype=torch.long),
        "observation.language.attention_mask": torch.ones((2, 4), dtype=torch.bool),
    }
    value = _teacher_bridge_grid_losses(
        policy,
        batch,
        torch.zeros((2, 50, 7)),
        torch.zeros((2, 50, 32)),
    )
    assert policy.model.times == pytest.approx(INFERENCE_TIMES)
    assert value.shape == (2, 50, 7)
    assert float(value[0, 0, 0]) == pytest.approx(sum(INFERENCE_TIMES) / 10)

    policy.model.forward = lambda *_args: torch.full(
        (2, 50, 32), float("inf")
    )
    with pytest.raises(WriterModelError, match="flow prediction"):
        _teacher_bridge_grid_losses(
            policy,
            batch,
            torch.zeros((2, 50, 7)),
            torch.zeros((2, 50, 32)),
        )


def test_exact_endpoint_wrapper_forces_ten_steps_and_no_grad() -> None:
    class Policy:
        def predict_action_chunk(self, _batch, *, noise, num_steps):
            assert not torch.is_grad_enabled()
            assert num_steps == 10
            assert noise.shape == (2, 50, 32)
            return torch.zeros((2, 50, 7), requires_grad=False)

    value = exact_endpoint_actions(
        Policy(),
        {},
        torch.zeros((2, 50, 32), dtype=torch.float32),
    )
    assert value.shape == (2, 50, 7)
    assert not value.requires_grad


def test_recursive_sampler_and_teacher_bridge_keep_distinct_precision_and_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autocast_active = False
    observations: list[tuple[str, bool]] = []

    @contextmanager
    def fake_autocast(**kwargs):
        nonlocal autocast_active
        assert kwargs["enabled"] is True
        assert not autocast_active
        autocast_active = True
        try:
            yield
        finally:
            autocast_active = False

    class Policy:
        def predict_action_chunk(self, batch, *, noise, num_steps):
            observations.append(("sampler", autocast_active))
            assert ACTION not in batch
            assert not torch.is_grad_enabled()
            assert num_steps == 10
            return torch.zeros((noise.shape[0], 50, 7))

    def fake_bridge(_policy, batch, teacher, _noise):
        observations.append(("teacher_bridge", autocast_active))
        assert ACTION not in batch
        assert not torch.is_grad_enabled()
        return torch.zeros_like(teacher)

    monkeypatch.setattr(endpoint_runtime.torch, "autocast", fake_autocast)
    monkeypatch.setattr(
        endpoint_runtime, "_teacher_bridge_grid_losses", fake_bridge
    )
    teacher = torch.zeros((2, 50, 7))
    predicted, grid = _predict_and_teacher_bridge(
        Policy(),
        {ACTION: teacher, "observation": torch.zeros(2, 1)},
        teacher,
        torch.zeros((2, 50, 32)),
        torch.device("cuda"),
    )
    assert observations == [("sampler", False), ("teacher_bridge", True)]
    assert predicted.shape == grid.shape == teacher.shape


def test_endpoint_finite_contract_rejects_noise_prediction_teacher_and_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch,
        "randn",
        lambda *_args, **_kwargs: torch.full((50, 32), float("nan")),
    )
    with pytest.raises(WriterModelError, match="noise is non-finite"):
        endpoint_noise(SEALED_PANEL_PAYLOAD_SHA256, (_row(),))

    class Policy:
        def predict_action_chunk(self, _batch, *, noise, num_steps):
            assert num_steps == 10
            return torch.full((noise.shape[0], 50, 7), float("inf"))

    with pytest.raises(WriterModelError, match="predicted action chunk"):
        exact_endpoint_actions(
            Policy(), {}, torch.zeros((1, 50, 32), dtype=torch.float32)
        )

    value = torch.zeros((1, 50, 7))
    padding = torch.zeros((1, 50), dtype=torch.bool)
    for position, label in ((0, "predicted"), (1, "teacher"), (2, "grid")):
        tensors = [value.clone(), value.clone(), value.clone()]
        tensors[position][0, 0, 0] = float("nan")
        with pytest.raises(WriterModelError, match=label):
            endpoint_metric_rows(
                tensors[0], tensors[1], padding, tensors[2]
            )


def test_endpoint_output_panel_is_exact_complete_and_profile_truncated() -> None:
    candidates = (
        _candidate("v52_step00000001", "v52", 1),
        _candidate("v6_step00000002", "v6", 2),
    )
    panel_rows = [_panel_row(0, 0), _panel_row(1, 1)]
    manifest = {"rows": panel_rows}
    rows = [_output_row(candidate, panel_rows[0]) for candidate in candidates]
    assert _validate_endpoint_output_rows(rows, candidates, manifest, 1) == 2

    with pytest.raises(WriterModelError, match="duplicated"):
        _validate_endpoint_output_rows(
            [rows[0], dict(rows[0])], candidates, manifest, 1
        )
    wrong_cursor = dict(rows[1])
    wrong_cursor["checkpoint_cursor"] = 99
    with pytest.raises(WriterModelError, match="candidate identity"):
        _validate_endpoint_output_rows(
            [rows[0], wrong_cursor], candidates, manifest, 1
        )
    nonfinite = dict(rows[1])
    nonfinite["metrics"] = {
        **rows[1]["metrics"],
        METRICS[0]: {
            "mse": float("nan"),
            "per_action_dimension_mse": [1.0] * 7,
        },
    }
    with pytest.raises(WriterModelError, match="non-finite"):
        _validate_endpoint_output_rows(
            [rows[0], nonfinite], candidates, manifest, 1
        )


def test_single_rank_profile_skips_distributed_broadcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        endpoint_runtime.dist,
        "broadcast_object_list",
        lambda *_args, **_kwargs: pytest.fail("single rank called broadcast"),
    )
    _broadcast_rank_zero_validation(
        [{"validated": True}],
        SimpleNamespace(world_size=1, is_main=True),
    )


def test_nonmain_hdf5_authority_uses_size_without_rehashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_ids = (1, 3, 11, 13, 23, 26, 31, 32)
    data_root = tmp_path / "data"
    data_root.mkdir()
    rows = []
    for task_id in task_ids:
        path = data_root / f"task_{task_id}.hdf5"
        path.write_bytes(bytes([task_id]) * 3)
        rows.append(
            {
                "global_task_id": task_id,
                "language": f"task {task_id}",
                "hdf5": {
                    "relative_path": path.name,
                    "bytes": 3,
                    "sha256": "a" * 64,
                },
            }
        )
    write_json_atomic(
        tmp_path / "target.json",
        {
            "summary": {"roles": {"validation": list(task_ids)}},
            "tasks": rows,
        },
    )
    panel = {
        "authorities": {
            "target_data_manifest": {"path": "target.json"}
        }
    }
    calls = 0

    def fake_sha256(_path):
        nonlocal calls
        calls += 1
        return "a" * 64

    monkeypatch.setattr(endpoint_runtime, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(endpoint_runtime, "sha256_file", fake_sha256)
    tasks = _task_authorities(
        panel, data_root, verify_sha256=False
    )
    assert len(tasks) == 8
    assert calls == 0
    _task_authorities(panel, data_root, verify_sha256=True)
    assert calls == 8


def test_correct400_pairing_rejects_policy_or_video_schedule_drift() -> None:
    contract = {
        "rng": {"inference_seed": 7},
        "policy": {"num_inference_steps": 10, "replan_steps": 5},
        "tasks": [
            {
                "suite": "libero_spatial",
                "task_id": 1,
                "init_state_ids": list(range(50)),
            }
        ],
        "adapter": {
            "video_schedule": {
                "sampling_mode": "without_replacement",
                "seed": 7,
            }
        },
    }
    reference, digest = _validate_correct400_pairing(contract, None)
    assert digest == canonical_hash(reference)
    changed = json.loads(json.dumps(contract))
    changed["adapter"]["video_schedule"]["seed"] = 8
    with pytest.raises(WriterModelError, match="paired correct400"):
        _validate_correct400_pairing(changed, reference)


def test_candidate_specs_and_task_video_pairing_fail_closed(tmp_path: Path) -> None:
    parsed = parse_endpoint_candidate_specs(
        [f"v52_new={tmp_path}", f"v6_fast={tmp_path}::{tmp_path / 'cache.json'}"]
    )
    assert parsed[0] == ("v52_new", tmp_path.resolve(), None)
    assert parsed[1][2] == (tmp_path / "cache.json").resolve()
    with pytest.raises(WriterModelError, match="unsafe"):
        parse_endpoint_candidate_specs([f"V5.2={tmp_path}"])

    mapping = []
    for global_task_id in (1, 3, 11, 13, 23, 26, 31, 32):
        suite = ("libero_spatial", "libero_object", "libero_goal", "libero_10")[
            global_task_id // 10
        ]
        task_id = global_task_id % 10
        mapping.append(
            {
                "suite": suite,
                "task_id": task_id,
                "language_global_task_id": global_task_id,
                "language_split_role": "validation",
                "video_suite": suite,
                "video_task_id": task_id,
                "video_global_task_id": global_task_id,
                "video_split_role": "validation",
            }
        )
    contract = {
        "adapter": {"task_video_mapping": mapping},
        "tasks": [
            {"suite": row["suite"], "task_id": row["task_id"]}
            for row in mapping
        ],
    }
    observed = _candidate_task_map(
        contract,
        [row["language_global_task_id"] for row in mapping],
    )
    assert set(observed.values()) == {
        row["language_global_task_id"] for row in mapping
    }
    contract["adapter"]["task_video_mapping"][0]["video_task_id"] += 1
    with pytest.raises(WriterModelError, match="mapping changed"):
        _candidate_task_map(contract, [row["language_global_task_id"] for row in mapping])


def test_validation_cli_keeps_functional_and_endpoint_inputs_disjoint(
    tmp_path: Path,
) -> None:
    common = {
        "panel_config": tmp_path / "panel.json",
        "source_run": tmp_path / "source",
        "source_checkpoint": tmp_path / "source-checkpoint",
        "tokenizer_path": tmp_path / "tokenizer",
        "data_root": tmp_path / "data",
        "output_dir": tmp_path / "output",
        "mode": "profile",
        "max_groups_per_task": 1,
    }
    endpoint = finalize_args(
        SimpleNamespace(
            **common,
            diagnostic="endpoint10",
            endpoint_candidates=[f"v52_new={tmp_path / 'evaluation'}"],
            training_run=None,
            checkpoints=None,
        )
    )
    assert endpoint.training_run is None
    assert endpoint.endpoint_candidates
    with pytest.raises(WriterModelError, match="no Writer checkpoint"):
        finalize_args(
            SimpleNamespace(
                **common,
                diagnostic="endpoint10",
                endpoint_candidates=[f"v52_new={tmp_path / 'evaluation'}"],
                training_run=tmp_path / "training",
                checkpoints=None,
            )
        )
    with pytest.raises(WriterModelError, match="training run"):
        finalize_args(
            SimpleNamespace(
                **common,
                diagnostic="functional_loss",
                endpoint_candidates=[],
                training_run=None,
                checkpoints=None,
            )
        )


def test_endpoint_device_scope_never_accepts_physical_gpus_zero_to_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(world_size=4)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    assert _validate_device_scope(SimpleNamespace(mode="formal")) == (
        4,
        5,
        6,
        7,
    )
    assert _validate_device_scope(
        SimpleNamespace(mode="formal"), context
    ) == (4, 5, 6, 7)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    with pytest.raises(WriterModelError, match="escaped"):
        _validate_device_scope(SimpleNamespace(mode="formal"), context)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    assert _validate_device_scope(
        SimpleNamespace(mode="profile"), SimpleNamespace(world_size=1)
    ) == (4,)
