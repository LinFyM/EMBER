from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from ember.writer.data import WriterTaskAuthority
from ember.writer.internal_analysis import (
    ATTENTION_PARITY_TOLERANCE,
    _attention_parity,
    _parity,
    _policy_attention_backends,
    _preserve_policy_attention_backends,
    fixed_policy_query,
)
from ember.writer.internal_metrics import CONDITIONS, attention_summary, routing_centered_energy, validate_finite_tree
from ember.writer.internal_results import lpt_assignment, summarize_rows, validate_rank_payloads
from ember.writer.model import WriterModelError


def _row(task: int, reference: int) -> dict:
    return {
        "global_task_id": task,
        "reference_ordinal": reference,
        "conditions": [{"condition": value} for value in CONDITIONS],
        "information_wall": {
            "teacher_action_values_read": 0,
            "teacher_state_values_sent_to_writer": 0,
            "teacher_reward_or_terminal_values_read": 0,
            "policy_query_observation_state_sent_to_writer": 0,
        },
        "metric": {"relative_l2": float(task + reference)},
    }


def test_lpt_and_four_rank_cartesian_sealing_are_exact() -> None:
    assignment = lpt_assignment({task: (task + 1) * 10 for task in range(8)})
    assert set(assignment) == {0, 1, 2, 3}
    assert all(len(value) == 2 for value in assignment.values())
    assert {value for values in assignment.values() for value in values} == set(range(8))
    payloads = [
        {
            "rank": rank,
            "assigned_task_ids": tasks,
            "rows": [_row(task, reference) for task in tasks for reference in range(3)],
        }
        for rank, tasks in assignment.items()
    ]
    rows = validate_rank_payloads(payloads, 3)
    assert len(rows) == 24
    summary = summarize_rows(rows)
    assert summary["tasks"] == 8
    assert summary["global_numeric"]["metric"]["relative_l2"]["count"] == 24
    payloads[0]["rows"][0]["conditions"][0]["condition"] = "reversed"
    with pytest.raises(WriterModelError, match="pairing"):
        validate_rank_payloads(payloads, 3)


def test_attention_summary_supports_batched_aed_programs() -> None:
    weights = torch.full((2, 4, 6, 15), 1.0 / 15)
    intervals = torch.ones(2, 3, dtype=torch.bool)
    semantics = torch.ones(2, 5, dtype=torch.bool)
    summary = attention_summary(weights, intervals, semantics)
    assert summary["normalized_entropy_mean"] == pytest.approx(1.0)
    assert summary["action_mass"] == pytest.approx(0.2)
    assert summary["effect_mass"] == pytest.approx(0.4)
    assert summary["change_mass"] == pytest.approx(0.4)


def test_program_reader_routing_energy_separates_target_and_rank() -> None:
    uniform = torch.full((1, 2, 6, 5), 0.2)
    baseline = routing_centered_energy(uniform, target_count=3, rank=2)
    assert baseline["target_centered_energy"] == pytest.approx(0)
    assert baseline["rank_centered_energy"] == pytest.approx(0)
    changed = uniform.clone().reshape(1, 2, 3, 2, 5)
    changed[:, :, 0, :, 0] += 0.05
    changed[:, :, 0, :, 1] -= 0.05
    target_only = routing_centered_energy(changed.reshape(1, 2, 6, 5), 3, 2)
    assert target_only["target_centered_energy"] > 0
    assert target_only["rank_centered_energy"] == pytest.approx(0)


class _Processor:
    def _tokenize_prompts(self, states: torch.Tensor, languages: list[str]):
        assert states.shape == (1, 8)
        assert languages == ["move the bowl"]
        return torch.tensor([[1, 2, 3]]), torch.ones(1, 3, dtype=torch.bool)


def test_fixed_policy_query_reads_observations_only(tmp_path: Path) -> None:
    path = tmp_path / "task.hdf5"
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", data=np.zeros((1, 4, 5, 3), dtype=np.uint8))
        obs.create_dataset("eye_in_hand_rgb", data=np.ones((1, 4, 5, 3), dtype=np.uint8))
        obs.create_dataset("ee_states", data=np.zeros((1, 6), dtype=np.float32))
        obs.create_dataset("gripper_states", data=np.zeros((1, 2), dtype=np.float32))
        demo.create_dataset("actions", data=np.full((1, 7), np.nan, dtype=np.float32))
    prepared, identity = fixed_policy_query(
        WriterTaskAuthority(4, "move the bowl", path, path.stat().st_size),
        _Processor(),
        torch.device("cpu"),
    )
    assert identity["observation_only"] is True
    assert identity["actions_dataset_opened"] is False
    assert set(prepared) == {
        "observation.images.base_0_rgb",
        "observation.images.left_wrist_0_rgb",
        "observation.language.tokens",
        "observation.language.attention_mask",
    }
    validate_finite_tree(identity)


def test_result_finite_gate_rejects_nested_nonfinite() -> None:
    with pytest.raises(WriterModelError, match="non-finite"):
        validate_finite_tree({"a": [{"b": float("nan")}]})


def test_bf16_sdpa_parity_tolerance_is_narrowly_scoped() -> None:
    reference = torch.ones(1024)
    within_bf16_roundoff = reference.clone()
    within_bf16_roundoff[::2] += ATTENTION_PARITY_TOLERANCE
    with pytest.raises(WriterModelError, match="parity failed"):
        _parity("strict", reference, within_bf16_roundoff)
    accepted = _attention_parity("attention", reference, within_bf16_roundoff)
    assert accepted["relative_l2"] < ATTENTION_PARITY_TOLERANCE
    assert accepted["tolerance"] == ATTENTION_PARITY_TOLERANCE

    outside_bf16_roundoff = reference.clone()
    outside_bf16_roundoff[::2] += 2 * ATTENTION_PARITY_TOLERANCE
    with pytest.raises(WriterModelError, match="BF16 SDPA parity failed"):
        _attention_parity("attention", reference, outside_bf16_roundoff)


def test_policy_attention_backend_mutation_is_scoped_to_action_probe() -> None:
    language_config = SimpleNamespace(_attn_implementation="sdpa")
    expert_config = SimpleNamespace(_attn_implementation="sdpa")
    policy = SimpleNamespace(
        model=SimpleNamespace(
            paligemma_with_expert=SimpleNamespace(
                paligemma=SimpleNamespace(
                    model=SimpleNamespace(
                        language_model=SimpleNamespace(config=language_config)
                    )
                ),
                gemma_expert=SimpleNamespace(
                    model=SimpleNamespace(config=expert_config)
                ),
            )
        )
    )
    assert _policy_attention_backends(policy) == {
        "language": "sdpa",
        "expert": "sdpa",
    }
    with _preserve_policy_attention_backends(policy) as before:
        assert before == {"language": "sdpa", "expert": "sdpa"}
        language_config._attn_implementation = "eager"
        expert_config._attn_implementation = "eager"
        assert _policy_attention_backends(policy) == {
            "language": "eager",
            "expert": "eager",
        }
    assert _policy_attention_backends(policy) == {
        "language": "sdpa",
        "expert": "sdpa",
    }
