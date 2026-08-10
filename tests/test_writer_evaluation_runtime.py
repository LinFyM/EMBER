from __future__ import annotations

from itertools import count
from types import SimpleNamespace

import pytest
import torch

import ember.writer.generation_profile as runtime_module
from ember.writer.generation_profile import profile_writer_generation
from ember.writer.rank_reserved_vertical import (
    _effective_delta_moments,
    _moment_summary,
    _rank_reserved_cached_base_state,
    _rank_reserved_qv_only_state,
    _rank_reserved_zero_slots,
)


def _qv_states() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(41)
    before: dict[str, torch.Tensor] = {}
    after: dict[str, torch.Tensor] = {}
    for layer in range(18):
        for module in ("q", "v"):
            prefix = f"model.layers.{layer}.self_attn.{module}_proj"
            a_name = prefix + ".lora_A.default.weight"
            b_name = prefix + ".lora_B.default.weight"
            before_a = torch.randn(4, 5, generator=generator)
            before_b = torch.randn(6, 4, generator=generator)
            after_a = before_a + torch.randn(4, 5, generator=generator) * 0.03
            after_b = before_b + torch.randn(6, 4, generator=generator) * 0.02
            before[a_name], before[b_name] = before_a, before_b
            after[a_name], after[b_name] = after_a, after_b
    return before, after


def test_effective_delta_moments_match_explicit_ba_without_base_subtraction() -> None:
    old_before, old_after = _qv_states()
    new_before, new_after = _qv_states()
    for value in new_after.values():
        value.add_(0.01)

    observed = _effective_delta_moments(
        old_before,
        old_after,
        new_before,
        new_after,
    )
    old_delta = []
    new_delta = []
    for a_name in sorted(
        name for name in old_before if name.endswith(".lora_A.default.weight")
    ):
        b_name = a_name.replace(".lora_A.default.weight", ".lora_B.default.weight")
        old_delta.append(
            old_after[b_name] @ old_after[a_name]
            - old_before[b_name] @ old_before[a_name]
        )
        new_delta.append(
            new_after[b_name] @ new_after[a_name]
            - new_before[b_name] @ new_before[a_name]
        )
    old_flat = torch.cat([value.flatten() for value in old_delta])
    new_flat = torch.cat([value.flatten() for value in new_delta])

    assert observed[:3] == pytest.approx(
        (
            float(torch.dot(old_flat, new_flat)),
            float(old_flat.square().sum()),
            float(new_flat.square().sum()),
        ),
        rel=2e-5,
        abs=2e-5,
    )
    assert observed[3:] == (36, 36, 36)
    summary = _moment_summary((observed,))
    assert summary["new_nonzero_targets"] == 36
    assert summary["new_delta_l2_rms_across_panel"] > 0


def test_rank14_macro0_zero_slots_check_both_native_factors() -> None:
    before, _ = _qv_states()
    state = {
        name: (
            torch.cat((value, torch.zeros(2, value.shape[1])), dim=0)
            if name.endswith(".lora_A.default.weight")
            else torch.cat((value, torch.zeros(value.shape[0], 2)), dim=1)
        )
        for name, value in before.items()
    }

    observed = _rank_reserved_zero_slots((state,))
    assert observed["exact_zero"] is True
    state[next(iter(state))][-1, 0] = 1
    assert _rank_reserved_zero_slots((state,))["exact_zero"] is False


def test_qv_only_state_uses_reward_attention_and_base_action_tensors() -> None:
    base, reward = _qv_states()
    for prefix in ("action", "action2"):
        base[f"{prefix}.lora_A.default.weight"] = torch.zeros(4, 5)
        base[f"{prefix}.lora_B.default.weight"] = torch.zeros(6, 4)
        reward[f"{prefix}.lora_A.default.weight"] = torch.ones(4, 5)
        reward[f"{prefix}.lora_B.default.weight"] = torch.ones(6, 4)

    observed = _rank_reserved_qv_only_state(base, reward)

    assert (
        observed["action.lora_A.default.weight"] is base["action.lora_A.default.weight"]
    )
    qv_name = "model.layers.0.self_attn.q_proj.lora_A.default.weight"
    assert observed[qv_name] is reward[qv_name]


def test_cached_base_zeroes_only_reward_qv_residual_slots() -> None:
    action_base, cached_reward = _qv_states()
    for prefix in ("action", "action2"):
        action_base[f"{prefix}.lora_A.default.weight"] = torch.zeros(4, 5)
        action_base[f"{prefix}.lora_B.default.weight"] = torch.zeros(6, 4)
        cached_reward[f"{prefix}.lora_A.default.weight"] = torch.ones(4, 5)
        cached_reward[f"{prefix}.lora_B.default.weight"] = torch.ones(6, 4)

    observed = _rank_reserved_cached_base_state(action_base, cached_reward)

    a_name = "model.layers.0.self_attn.q_proj.lora_A.default.weight"
    b_name = "model.layers.0.self_attn.q_proj.lora_B.default.weight"
    assert torch.equal(observed[a_name][:-2], cached_reward[a_name][:-2])
    assert torch.count_nonzero(observed[a_name][-2:]) == 0
    assert torch.equal(observed[b_name][:, :-2], cached_reward[b_name][:, :-2])
    assert torch.count_nonzero(observed[b_name][:, -2:]) == 0
    assert (
        observed["action.lora_A.default.weight"]
        is action_base["action.lora_A.default.weight"]
    )


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
