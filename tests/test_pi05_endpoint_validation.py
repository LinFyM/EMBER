from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from ember.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    LoRATarget,
    SmolVLALoRAContract,
    canonical_contract_sha256,
    lora_state_sha256,
)
from ember.pi05_source_checkpoint import canonical_hash, sha256_file, write_json_atomic
from ember.writer.endpoint_validation import (
    ENDPOINT_NOISE_SCHEMA,
    INFERENCE_TIMES,
    METRICS,
    PORTABLE_CACHE_SCHEMA,
    SEALED_PANEL_PAYLOAD_SHA256,
    EndpointLoRAEntry,
    _candidate_task_map,
    _portable_cache_entries,
    _verify_lora_entry,
    _validated_payload,
    endpoint_metric_rows,
    endpoint_noise,
    endpoint_noise_seed,
    endpoint_schedule,
    exact_endpoint_actions,
    parse_endpoint_candidate_specs,
)
from ember.writer.endpoint_runtime import (
    _teacher_bridge_grid_losses,
    _validate_device_scope,
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

def test_payload_and_portable_cache_information_wall_fail_closed(tmp_path: Path) -> None:
    payload = {"schema_version": "example_v1", "value": 3}
    payload["canonical_payload_sha256"] = canonical_hash(payload)
    path = tmp_path / "payload.json"
    write_json_atomic(path, payload)
    assert _validated_payload(path, "example_v1")["value"] == 3
    payload["value"] = 4
    write_json_atomic(path, payload)
    with pytest.raises(WriterModelError, match="artifact changed"):
        _validated_payload(path, "example_v1")

    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    write_json_atomic(evaluation / "run_contract.json", {})
    write_json_atomic(evaluation / "results.json", {})
    lora = SmolVLALoRAContract(
        targets=(LoRATarget("proj", 2, 2),), rank=1, alpha=1, dropout=0.0, identity_seed=7
    )
    contract = {
        "contract_sha256": "1" * 64,
        "adapter": {
            "training_run": {"git_commit": "4" * 40},
            "checkpoint": {
                "cursor": 10,
                "manifest_file_sha256": "2" * 64,
                "writer_state_sha256": "3" * 64,
            }
        },
    }
    portable = {
        "schema_version": PORTABLE_CACHE_SCHEMA,
        "candidate": {
            "evaluation_root": str(evaluation),
            "run_contract_file_sha256": sha256_file(evaluation / "run_contract.json"),
            "run_contract_sha256": contract["contract_sha256"],
            "results_file_sha256": sha256_file(evaluation / "results.json"),
            "adapter_sha256": canonical_hash(contract["adapter"]),
            "writer_constructor_git_commit": "4" * 40,
        },
        "panel_manifest_payload_sha256": SEALED_PANEL_PAYLOAD_SHA256,
        "lora_contract_sha256": canonical_contract_sha256(lora),
        "information_wall": {
            "validation_action_values_read_during_generation": 0,
            "test_action_reads": 0,
            "test_video_value_reads": 0,
        },
        "entries": [
            {
                "global_task_id": task_id,
                "teacher_demo_index": demo,
                "lora_file": {
                    "path": f"loras/{task_id}_{demo}.safetensors",
                    "bytes": 10,
                    "sha256": str(task_id) * 64,
                },
                "lora_state_sha256": str(demo) * 64,
                "generation_evidence": {
                    "language_global_task_id": task_id,
                    "video_global_task_id": task_id,
                    "teacher_demo_index": demo,
                    "condition": "correct",
                    "writer_checkpoint_cursor": 10,
                    "writer_checkpoint_manifest_sha256": "2" * 64,
                    "writer_state_sha256": "3" * 64,
                    "lora_sha256": str(demo) * 64,
                },
            }
            for task_id, demo in ((1, 4), (3, 5))
        ],
    }
    portable["canonical_payload_sha256"] = canonical_hash(portable)
    portable_path = tmp_path / "portable.json"
    write_json_atomic(portable_path, portable)
    entries, _manifest_sha = _portable_cache_entries(
        portable_path,
        evaluation,
        contract,
        {(1, 4), (3, 5)},
        lora,
    )
    assert set(entries) == {(1, 4), (3, 5)}

    portable.pop("canonical_payload_sha256")
    portable["information_wall"][
        "validation_action_values_read_during_generation"
    ] = 1
    portable["canonical_payload_sha256"] = canonical_hash(portable)
    write_json_atomic(portable_path, portable)
    with pytest.raises(WriterModelError, match="authority changed"):
        _portable_cache_entries(
            portable_path,
            evaluation,
            contract,
            {(1, 4), (3, 5)},
            lora,
        )


def test_public_lora_tensor_hashes_and_state_hash_fail_closed(tmp_path: Path) -> None:
    lora = SmolVLALoRAContract(
        targets=(LoRATarget("proj", 2, 2),),
        rank=1,
        alpha=1,
        dropout=0.0,
        identity_seed=7,
    )
    state = {
        "proj" + LORA_A_SUFFIX: torch.tensor([[1.0, 2.0]]),
        "proj" + LORA_B_SUFFIX: torch.tensor([[3.0], [4.0]]),
    }
    path = tmp_path / "lora.safetensors"
    save_file(state, str(path))
    entry = EndpointLoRAEntry(
        path=path,
        bytes=path.stat().st_size,
        file_sha256=sha256_file(path),
        state_sha256=lora_state_sha256(state),
    )
    loaded = _verify_lora_entry(entry, lora, torch.device("cpu"))
    assert lora_state_sha256(loaded) == entry.state_sha256
    with pytest.raises(WriterModelError, match="state changed"):
        _verify_lora_entry(
            EndpointLoRAEntry(
                path=entry.path,
                bytes=entry.bytes,
                file_sha256=entry.file_sha256,
                state_sha256="0" * 64,
            ),
            lora,
            torch.device("cpu"),
        )
