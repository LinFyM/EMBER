from __future__ import annotations

from itertools import count
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ember.writer.evaluation_runtime as cache_runtime_module
import ember.writer.generation_profile as runtime_module
import ember.writer.live_adapter as live_adapter_module
from ember.writer.data import RawTeacherVideo
from ember.writer.evaluation_cache import (
    WriterCacheGenerationBatch,
    WriterCacheRequest,
)
from ember.writer.evaluation_runtime import run_writer_generation_phase
from ember.writer.generation_profile import profile_writer_generation
from ember.writer.live_adapter import FrozenDynamicKTaskAdapter


def test_profile_keeps_stable_candidates_when_larger_batch_ooms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    requests = tuple(
        SimpleNamespace(
            suite="libero_spatial",
            task_id=index,
            init_state_id=0,
            ordinal=index,
            entry_id=f"entry-{index}",
        )
        for index in range(32)
    )

    class Adapter:
        released = False

        def generation_request_profiles(self, identities):
            return tuple(
                {**identity, "sampled_frames": 64 - index, "raw_frames": 320}
                for index, identity in enumerate(identities)
            )

        def prepare_episodes(self, identities):
            if len(identities) == 32:
                raise torch.cuda.OutOfMemoryError("expected candidate OOM")
            self.profile = tuple(
                {**identity, "sampled_frames": 64 - int(identity["task_id"])}
                for identity in identities
            )
            return tuple(
                SimpleNamespace(state={"tensor": torch.ones(1)}) for _ in identities
            )

        def last_generation_batch_profile(self):
            return self.profile

        def release_generation_assets(self):
            self.released = True

    adapter = Adapter()
    contract = {
        "mode": "smoke",
        "contract_reference": "test-profile",
        "adapter": {"video_condition": "correct"},
        "parallel": {
            "physical_gpu_count": 1,
            "replicas_per_gpu": 1,
            "writer_generators_per_gpu": 1,
            "writer_generation_batch_size": 32,
        },
        "git": {"commit": "a" * 40, "dirty_paths": []},
    }
    runtime = SimpleNamespace(
        contract=contract,
        task_adapter=adapter,
        output_dir=tmp_path,
        gpu_uuid="GPU-test",
        gpu_index=0,
    )
    ticks = count()
    monkeypatch.setattr(
        runtime_module, "writer_cache_requests", lambda _contract: requests
    )
    monkeypatch.setattr(
        runtime_module,
        "stage_writer_lora_states_to_cpu",
        lambda states: tuple({} for _ in states),
    )
    monkeypatch.setattr(
        runtime_module,
        "git_state_is_clean_pushed_or_frozen_authority",
        lambda _git: True,
    )
    monkeypatch.setattr(runtime_module.time, "monotonic", lambda: float(next(ticks)))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "NVIDIA A40")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(total_memory=48 * 1024**3),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 8 * 1024**3)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 9 * 1024**3)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 8 * 1024**3)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 9 * 1024**3)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(runtime_module, "write_json_atomic", lambda *_args: None)

    result = profile_writer_generation(
        runtime,
        batch_sizes=(8, 16, 32),
        warmup_runs=1,
        measured_runs=2,
        preflight={"compute_applications": [], "device_names": ["NVIDIA A40"]},
    )

    assert result["selected_writer_model_batch_size"] == 16
    assert result["oom_count"] == 1
    assert result["writer_generation_measurements"][-1]["stable"] is False
    assert result["writer_generation_measurements"][-1]["oom_count"] == 1
    assert adapter.released is True


@pytest.mark.parametrize(
    ("complete_ordinals", "expected_generated", "expected_redundant"),
    (((), 8, 0), ((1, 6), 6, 2)),
)
def test_canonical_generation_keeps_full_batch_on_partial_resume(
    monkeypatch: pytest.MonkeyPatch,
    complete_ordinals: tuple[int, ...],
    expected_generated: int,
    expected_redundant: int,
) -> None:
    requests = tuple(
        WriterCacheRequest("libero_spatial", 1, ordinal, ordinal)
        for ordinal in range(8)
    )
    assigned = (
        WriterCacheGenerationBatch(
            ordinal=7,
            requests=requests,
            canonical_global=True,
        ),
    )
    prepared_identities: list[tuple[dict, ...]] = []
    writes: list[tuple[int, dict]] = []

    class Adapter:
        lora_contract = object()

        def prepare_episodes(self, identities):
            prepared_identities.append(tuple(identities))
            self.profile = tuple(
                {**identity, "raw_frames": 320, "sampled_frames": 64}
                for identity in identities
            )
            return tuple(
                SimpleNamespace(
                    state={"tensor": torch.tensor(float(index))},
                    evidence={"index": index},
                )
                for index, _identity in enumerate(identities)
            )

        def last_generation_batch_profile(self):
            return self.profile

    runtime = SimpleNamespace(
        contract={
            "parallel": {
                "writer_generators_per_gpu": 1,
                "writer_generation_batch_size": 8,
            }
        },
        gpu_slot=0,
        replica=0,
        worker_id="0-r0",
        task_adapter=Adapter(),
    )
    monkeypatch.setattr(
        cache_runtime_module,
        "assigned_writer_cache_batches",
        lambda *_args, **_kwargs: assigned,
    )
    monkeypatch.setattr(
        cache_runtime_module,
        "writer_cache_entry_is_complete",
        lambda _contract, request: request.ordinal in complete_ordinals,
    )
    monkeypatch.setattr(
        cache_runtime_module,
        "stage_writer_lora_states_to_cpu",
        lambda states: tuple(dict(state) for state in states),
    )
    monkeypatch.setattr(
        cache_runtime_module,
        "write_writer_cache_entry",
        lambda _contract, request, **kwargs: writes.append(
            (request.ordinal, dict(kwargs["generation"]))
        ),
    )
    monkeypatch.setattr(
        cache_runtime_module,
        "_finish_generation_handoff",
        lambda _runtime, *, invocation_id, generation, append_event: dict(generation),
    )
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)

    result = run_writer_generation_phase(
        runtime,
        invocation_id="a" * 32,
        append_event=lambda *_args: None,
    )

    assert len(prepared_identities) == 1
    assert tuple(row["init_state_id"] for row in prepared_identities[0]) == tuple(
        range(8)
    )
    assert [ordinal for ordinal, _generation in writes] == [
        ordinal for ordinal in range(8) if ordinal not in complete_ordinals
    ]
    assert all(
        generation["batch_ordinal"] == 7
        and generation["position_in_batch"] == ordinal
        and generation["batch_size"] == 8
        and len(generation["batch_entry_ids"]) == 8
        for ordinal, generation in writes
    )
    assert result["assigned_entries"] == 8
    assert result["generated_entries"] == expected_generated
    assert result["reused_entries"] == len(complete_ordinals)
    assert result["generated_batches"] == 1
    assert result["redundant_writer_forwards"] == expected_redundant
    assert result["batches"][0]["batch_ordinal"] == 7
    assert result["batches"][0]["batch_size"] == 8
    assert len(result["batches"][0]["entry_ids"]) == 8


def test_dynamic_k_control_reorders_frames_but_preserves_time_indices() -> None:
    frames = np.arange(4 * 3 * 2 * 2, dtype=np.uint8).reshape(4, 3, 2, 2)
    indices = np.asarray((0, 5, 10, 15), dtype=np.int64)
    video = RawTeacherVideo(
        frames=frames.copy(),
        frame_indices=indices.copy(),
        raw_frame_count=16,
    )
    reversed_frames, reversed_indices = live_adapter_module._ordered_video_tensors(
        video,
        condition="reversed",
        order_seed=7,
        device=torch.device("cpu"),
    )
    shuffled_frames, shuffled_indices = live_adapter_module._ordered_video_tensors(
        video,
        condition="shuffled",
        order_seed=7,
        device=torch.device("cpu"),
    )
    assert torch.equal(reversed_frames, torch.from_numpy(frames).flip(0))
    assert torch.equal(reversed_indices, torch.from_numpy(indices))
    assert not torch.equal(shuffled_frames, torch.from_numpy(frames))
    assert torch.equal(shuffled_indices, torch.from_numpy(indices))


def test_dynamic_k_no_video_returns_identity_without_writer_forward() -> None:
    adapter = FrozenDynamicKTaskAdapter.__new__(FrozenDynamicKTaskAdapter)
    adapter.evaluation_k = 4
    adapter.evaluation_adapter = {"video_condition": "no_video"}
    adapter._physical_lora_is_identity = True
    adapter.device = torch.device("cpu")
    adapter.policy = object()
    adapter.identity_state = {"generated": torch.ones(1)}
    adapter.lora_contract = object()
    adapter._episode_input = lambda **_identity: (
        {"lora_reference": "identity"},
        (),
        "task",
        (),
    )
    adapter.writer = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("no-video must not run the Writer")
    )
    prepared = adapter.prepare_episodes(
        ({"suite": "libero_spatial", "task_id": 1, "init_state_id": 0},)
    )
    assert len(prepared) == 1
    assert torch.equal(prepared[0].state["generated"], torch.ones(1))
    assert adapter.last_generation_batch_profile()[0]["sampled_frames"] == 0
