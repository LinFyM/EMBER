from __future__ import annotations

from dataclasses import asdict
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.inference import (
    EXPERT_MANIFOLD_ADAPTER_SCHEMA,
    EXPERT_MANIFOLD_WRITER_KIND,
    expected_expert_manifold_episode_evidence,
)
import ember.expert_manifold.live_adapter as live_adapter_module
from ember.expert_manifold.live_adapter import (
    FrozenExpertManifoldTaskAdapter,
    _build_v6_writer,
    _ordered_video_tensors,
)
from ember.expert_manifold.v6_prior_checkpoint import PROGRAM_MEMORY_KEY
from ember.expert_manifold.video_schedule import (
    SAME_TASK_OTHER_OFFSET,
    reference_demo_index,
    shuffled_frame_permutation,
    task_video_mapping,
    video_schedule_contract,
    video_selection_seed,
)
from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_contract import RUN_CONTRACT_SCHEMA, policy_noise_seed
from ember.pi05_eval_queue import EvaluationShard
from ember.pi05_eval_results import _per_task_rows
from ember.pi05_evaluation import SHARD_RESULT_SCHEMA, validate_shard_result
from ember.writer.data import RawTeacherVideo


def _rows() -> list[dict]:
    return [
        {
            "suite": "libero_spatial",
            "task_id": 0,
            "split_role": "train",
            "language": "task zero",
            "init_state_id": state_id,
            "env_seed": 7,
            "policy_seed_root": 7,
            "policy_noise_seeds": [
                policy_noise_seed(7, "libero_spatial", 0, state_id, 0)
            ],
            "success": state_id == 0,
            "steps": 1,
            "wall_seconds": 0.1,
            "finished_at": 0.1,
        }
        for state_id in (0, 1)
    ]


def _writer_adapter(condition: str = "correct") -> dict:
    keys = tuple(
        (suite, 0)
        for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    )
    roles = {key: "train" for key in keys}
    schedule, pairing = video_schedule_contract(
        seed=7,
        demo_count=50,
        sampling_mode="without_replacement",
    )
    return {
        "schema_version": EXPERT_MANIFOLD_ADAPTER_SCHEMA,
        "kind": EXPERT_MANIFOLD_WRITER_KIND,
        "config": {"schema": "ember_pi05_v6_reward_credit_program_cotangent_v1"},
        "arm": f"expert_manifold_v6_condition_residual_{condition}",
        "video_condition": condition,
        "writer_asset": {
            "reference": "test:v6-prior:historical-macro400",
            "kind": "historical_v6_macro400_load_only",
            "method_macro": 0,
            "writer_parameter_count": 10_775_296,
            "program_residual_value_count": 20_971_520,
            "generated_lora_tensor_count": 76,
        },
        "lora_contract": {"reference": "rank16:76tensors"},
        "video_schedule": schedule,
        "task_video_mapping_reference": "next_suite_v1",
        "task_video_mapping": list(task_video_mapping(keys, roles, condition)),
        "pairing_reference": pairing,
    }


def _complete_writer():
    path = Path(__file__).with_name("test_writer_model.py")
    spec = importlib.util.spec_from_file_location("live_adapter_writer_helper", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._model()[0]


def test_no_video_prepare_uses_identity_without_writer_or_video_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FrozenExpertManifoldTaskAdapter.__new__(FrozenExpertManifoldTaskAdapter)
    adapter.evaluation_adapter = {"video_condition": "no_video"}
    adapter.identity_state = {
        "layer.lora_A.default.weight": torch.ones(2, 3),
        "layer.lora_B.default.weight": torch.zeros(4, 2),
    }
    adapter.device = torch.device("cpu")
    adapter.policy = object()
    adapter.lora_contract = object()
    adapter._last_generation_batch_profile = ()

    monkeypatch.setattr(
        adapter,
        "_episode_input",
        lambda **_identity: ({"schema_version": "test"}, None, "task"),
    )
    monkeypatch.setattr(
        live_adapter_module,
        "copy_task_lora_state_",
        lambda *_args, **_kwargs: None,
    )
    adapter.writer = SimpleNamespace(
        __call__=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Writer must not run")
        )
    )
    adapter.store = SimpleNamespace(
        load=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("video store must not run")
        )
    )
    adapter.tokenizer = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("tokenizer must not run")
    )

    prepared = adapter.prepare_episodes(
        ({"suite": "libero_spatial", "task_id": 0, "init_state_id": 0},)
    )

    assert len(prepared) == 1
    assert all(
        torch.equal(prepared[0].state[name], value)
        for name, value in adapter.identity_state.items()
    )


def test_no_video_episode_input_skips_the_real_video_store() -> None:
    adapter = FrozenExpertManifoldTaskAdapter.__new__(FrozenExpertManifoldTaskAdapter)
    adapter.evaluation_adapter = _writer_adapter("no_video")
    mapping = adapter.evaluation_adapter["task_video_mapping"][0]
    adapter.language_by_id = {int(mapping["language_global_task_id"]): "task zero"}
    adapter.store = SimpleNamespace(
        load=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("no-video _episode_input must not read the video store")
        )
    )

    row, video, language = adapter._episode_input(
        suite="libero_spatial", task_id=0, init_state_id=0
    )

    assert row["condition"] == "no_video"
    assert video is None
    assert language == "task zero"


@pytest.mark.parametrize("residual_value", (0.25, float("nan")))
def test_live_adapter_strict_loads_frozen_base_then_only_finite_residual_memory(
    monkeypatch: pytest.MonkeyPatch,
    residual_value: float,
) -> None:
    writer = _complete_writer()
    warm_state = {
        name: value.detach().cpu().clone()
        for name, value in writer.state_dict().items()
    }
    template = {
        name: value.detach().cpu().clone()
        for name, value in writer.template_state().items()
    }
    residual = torch.full((2, 320, 256), residual_value, dtype=torch.float32)
    writer.load_state_dict(
        {name: torch.zeros_like(value) for name, value in warm_state.items()},
        strict=True,
    )

    events: list[str] = []

    def load_warm_start(target, _path):
        events.append("historical_warm_start")
        target.load_state_dict(warm_state, strict=True)

    def load_trained(_path: str, *, device: str):
        assert device == "cpu"
        assert events == ["historical_warm_start"]
        events.append("residual_checkpoint")
        return {PROGRAM_MEMORY_KEY: residual}

    monkeypatch.setattr(
        "ember.expert_manifold.live_adapter.CompleteLoRAWriter",
        lambda *_args, **_kwargs: writer,
    )
    monkeypatch.setattr(
        "ember.expert_manifold.live_adapter.build_lora_tensor_specs",
        lambda _template: (),
    )
    monkeypatch.setattr(
        "ember.expert_manifold.live_adapter.load_v6_prior_warm_start_",
        load_warm_start,
    )
    monkeypatch.setattr(
        "ember.expert_manifold.live_adapter.load_file",
        load_trained,
    )
    bridge = SimpleNamespace(
        paligemma=SimpleNamespace(
            model=SimpleNamespace(language_model=object()),
        ),
        gemma_expert=SimpleNamespace(model=object()),
    )
    policy = SimpleNamespace(
        model=SimpleNamespace(paligemma_with_expert=bridge),
    )
    arguments = {
        "config": {
            "writer": {},
            "initialization": {"checkpoint": "/synthetic/historical-v6"},
            "condition_feature": {"feature_width": 2, "projection_seed": 17},
        },
        "observed": {
            "writer_asset": {
                "kind": "v6_condition_program_residual_checkpoint",
                "residual_state": {
                    "path": "/synthetic/trained/program_memory.safetensors",
                    "shape": [2, 320, 256],
                },
            },
        },
        "policy": policy,
        "template": template,
        "device": torch.device("cpu"),
    }
    if not np.isfinite(residual_value):
        with pytest.raises(ExpertManifoldError, match="evaluation state changed"):
            _build_v6_writer(**arguments)
        return
    result = _build_v6_writer(**arguments)

    assert events == ["historical_warm_start", "residual_checkpoint"]
    assert all(
        torch.equal(result.base_writer.state_dict()[name], value)
        for name, value in warm_state.items()
    )
    assert torch.equal(result.program_memory.value, residual)
    assert "projection" not in result.state_dict()
    assert all(not parameter.requires_grad for parameter in result.parameters())


def test_per_task_rows_summarizes_one_shot_teacher_videos() -> None:
    rows = _rows()
    for row, demo in zip(rows, (0, 1), strict=True):
        row["writer"] = {
            "condition": "correct",
            "teacher_demo_indices": [demo],
            "writer_generation_seconds": 0.25,
        }
    tasks = {
        ("libero_spatial", 0): {
            "split_role": "train",
            "language": "task zero",
        }
    }

    writer = _per_task_rows(rows, tasks)[0]["writer"]

    assert writer["videos_per_condition"] == 1
    assert writer["unique_teacher_videos"] == 2
    assert writer["teacher_demo_counts"] == {"0": 1, "1": 1}
    assert writer["unique_teacher_video_sets"] == 2
    assert writer["teacher_demo_set_counts"] == {"0": 1, "1": 1}
    assert writer["generation_wall_seconds"] == 0.5


def test_writer_video_schedule_and_wrong_map_are_order_independent() -> None:
    request = (7, "libero_spatial", 6, 0)
    assert video_selection_seed(
        *request, sampling_mode="without_replacement"
    ) == video_selection_seed(*request, sampling_mode="without_replacement")
    assert (
        0
        <= reference_demo_index(
            *request,
            demo_count=50,
            sampling_mode="without_replacement",
        )
        < 50
    )
    keys = (
        ("libero_spatial", 1),
        ("libero_spatial", 3),
        ("libero_object", 1),
        ("libero_object", 3),
        ("libero_goal", 3),
        ("libero_goal", 6),
        ("libero_10", 1),
        ("libero_10", 2),
    )
    roles = {key: "validation" for key in keys}
    forward = task_video_mapping(keys, roles, "cross_suite_wrong")
    reverse = task_video_mapping(tuple(reversed(keys)), roles, "cross_suite_wrong")
    shuffled = task_video_mapping(keys, roles, "shuffled")
    shuffled_keep_first = task_video_mapping(keys, roles, "shuffled_keep_first")
    reversed_video = task_video_mapping(keys, roles, "reversed")
    same_task_other = task_video_mapping(keys, roles, "same_task_other")
    assert forward == reverse
    assert shuffled == shuffled_keep_first == reversed_video == same_task_other
    assert all(row["suite"] == row["video_suite"] for row in shuffled)
    assert len({row["video_global_task_id"] for row in forward}) == len(keys)
    assert all(row["suite"] != row["video_suite"] for row in forward)
    assert all(row["language_split_role"] == row["video_split_role"] for row in forward)
    by_key = {(row["suite"], row["task_id"]): row for row in forward}
    assert (
        by_key[("libero_spatial", 1)]["video_suite"],
        by_key[("libero_spatial", 1)]["video_task_id"],
    ) == ("libero_object", 1)
    assert (
        by_key[("libero_goal", 6)]["video_suite"],
        by_key[("libero_goal", 6)]["video_task_id"],
    ) == ("libero_10", 2)


def test_shuffled_keep_first_changes_only_the_anchor_position() -> None:
    shuffled = shuffled_frame_permutation(20, 7, keep_first=False)
    keep_first = shuffled_frame_permutation(20, 7, keep_first=True)
    assert keep_first[0].item() == 0
    assert keep_first[1:].tolist() == [
        index for index in shuffled.tolist() if index != 0
    ]
    assert sorted(keep_first.tolist()) == list(range(20))


def test_temporal_controls_reorder_frames_but_keep_display_positions() -> None:
    frames = np.arange(4 * 3 * 2 * 2, dtype=np.uint8).reshape(4, 3, 2, 2)
    indices = np.asarray([0, 5, 10, 15], dtype=np.int64)
    video = RawTeacherVideo(
        frames=frames.copy(),
        frame_indices=indices.copy(),
        raw_frame_count=16,
    )
    correct_frames, correct_indices = _ordered_video_tensors(
        video,
        condition="correct",
        order_seed=7,
        device=torch.device("cpu"),
    )
    reversed_frames, reversed_indices = _ordered_video_tensors(
        video,
        condition="reversed",
        order_seed=7,
        device=torch.device("cpu"),
    )
    shuffled_frames, shuffled_indices = _ordered_video_tensors(
        video,
        condition="shuffled",
        order_seed=7,
        device=torch.device("cpu"),
    )
    permutation = shuffled_frame_permutation(4, 7, keep_first=False)

    assert torch.equal(correct_frames, torch.from_numpy(frames))
    assert torch.equal(reversed_frames, torch.from_numpy(frames).flip(0))
    assert torch.equal(
        shuffled_frames,
        torch.from_numpy(frames).index_select(0, permutation),
    )
    assert torch.equal(correct_indices, torch.from_numpy(indices))
    assert torch.equal(reversed_indices, correct_indices)
    assert torch.equal(shuffled_indices, correct_indices)
    assert np.array_equal(video.frames, frames)
    assert np.array_equal(video.frame_indices, indices)


def test_same_task_other_changes_only_the_teacher_demo() -> None:
    correct = expected_expert_manifold_episode_evidence(
        _writer_adapter("correct"),
        suite="libero_spatial",
        task_id=0,
        init_state_id=4,
        lora_reference="correct",
    )
    other = expected_expert_manifold_episode_evidence(
        _writer_adapter("same_task_other"),
        suite="libero_spatial",
        task_id=0,
        init_state_id=4,
        lora_reference="other",
    )
    assert other["teacher_demo_indices"] == [
        (value + SAME_TASK_OTHER_OFFSET) % 50
        for value in correct["teacher_demo_indices"]
    ]
    assert other["teacher_reference_demo_indices"] == correct["teacher_demo_indices"]
    assert other["video_global_task_id"] == correct["video_global_task_id"]
    assert other["teacher_video_order_seeds"] == correct["teacher_video_order_seeds"]


def test_writer_row_contract_recomputes_video_schedule_and_mapping(
    tmp_path: Path,
) -> None:
    contract = {
        "schema_version": RUN_CONTRACT_SCHEMA,
        "mode": "smoke",
        "arm": "expert_manifold_v6_condition_residual_correct",
        "role": "test",
        "output_dir": str(tmp_path),
        "content_hash_policy": "disabled_by_owner",
        "model": {"optimizer_step": 1},
        "normalization": {"bytes": 1},
        "tokenizer": {"bytes": 1},
        "rng": {"inference_seed": 7},
        "policy": {"replan_steps": 5},
        "tasks": [
            {
                "suite": "libero_spatial",
                "task_id": 0,
                "split_role": "train",
                "language": "task zero",
                "init_state_ids": [0, 1],
            }
        ],
        "adapter": _writer_adapter(),
        "contract_reference": f"{RUN_CONTRACT_SCHEMA}:test-contract",
    }
    shard = EvaluationShard(
        job_id="job",
        ordinal=0,
        suite="libero_spatial",
        task_id=0,
        horizon=220,
        init_state_ids=(0, 1),
        estimated_cost=440,
    )
    payload = {
        "schema_version": SHARD_RESULT_SCHEMA,
        "contract_reference": contract["contract_reference"],
        "job_id": shard.job_id,
        "shard": asdict(shard),
        "producer": {"worker_id": "0-r0", "claim_token": "a" * 32, "attempt": 1},
        "started_unix": 10.0,
        "finished_unix": 12.0,
        "rows": _rows(),
    }
    for row in payload["rows"]:
        row["writer"] = {
            **expected_expert_manifold_episode_evidence(
                contract["adapter"],
                suite=row["suite"],
                task_id=row["task_id"],
                init_state_id=row["init_state_id"],
                lora_reference=f"row:{row['init_state_id']}",
            ),
            "writer_generation_seconds": 0.25,
        }
    assert len(validate_shard_result(payload, contract=contract, shard=shard)) == 2
    payload["rows"][0]["writer"]["teacher_demo_indices"][0] += 1
    with pytest.raises(Pi05EvaluationError, match="row contract changed"):
        validate_shard_result(payload, contract=contract, shard=shard)
