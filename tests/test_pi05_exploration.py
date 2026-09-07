from __future__ import annotations

import copy
import json
from collections import deque
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval.exploration import (
    DIAGNOSTIC_STATES, add_exploration_noise, build_exploration_contract,
    episode_exploration_fields, exploration_metadata, exploration_noise_seed,
    validate_exploration_comparison, validate_exploration_contract,
)
from ember.pi05_eval.preparation import _explicit_diagnostic_states
from ember.pi05_eval_contract import RUN_CONTRACT_SCHEMA, load_run_contract, policy_noise_seed
from ember.pi05_eval_queue import EvaluationShard
from ember.pi05_eval_results import AGGREGATE_SCHEMA, paired_success_comparison
from ember.pi05_evaluation import SHARD_RESULT_SCHEMA, _plan_action_chunks, rollout_shard, validate_shard_result
from ember.writer.rl_math import exploration_covariance
from scripts.compare_pi05_results import compare


def _contract(enabled=False):
    task = {"suite": "libero_spatial", "task_id": 0, "split_role": "train", "language": "move bowl",
            "horizon": 220, "init_state_ids": list(DIAGNOSTIC_STATES)}
    value = {"role": "development_train", "mode": "screen", "tasks": [task],
        "contract_reference": "test", "model": {"checkpoint": "/source/checkpoint"},
        "environment": {"dummy_action": [0.] * 6 + [-1.], "dummy_settling_steps": 10},
        "policy": {"replan_steps": 5, "action_dim": 7, "chunk_size": 50, "num_inference_steps": 10},
        "rng": {"inference_seed": 7}, "adapter": None}
    value["diagnostic_exploration"] = build_exploration_contract(value, enabled=enabled)
    return value


def _slots(states=(32, 33)):
    return [{"init_state_id": state, "replan_index": 2} for state in states]


def test_correlated_noise_matches_canonical_covariance_and_only_changes_execution_block():
    contract = _contract(True)
    chunks = torch.full((2, 50, 32), 2.)
    before = torch.random.get_rng_state()
    slots = _slots()
    actual = add_exploration_noise(chunks, slots, task=contract["tasks"][0], contract=contract)
    sigma = exploration_covariance()
    assert sigma[0, 0] == pytest.approx(0.05 ** 2)
    assert sigma[0, 7] == pytest.approx(0.8 * 0.05 ** 2)
    assert sigma[6, 6] == pytest.approx(0.10 ** 2)
    assert sigma[0, 1] == 0
    standard = torch.stack([torch.randn(35, generator=torch.Generator().manual_seed(slot["exploration_noise_seeds"][0]))
                            for slot in slots])
    expected = 2 + (standard @ torch.linalg.cholesky(sigma).T).reshape(2, 5, 7)
    torch.testing.assert_close(actual[:, :5, :7], expected)
    assert torch.equal(actual[:, 5:], chunks[:, 5:]) and torch.equal(actual[:, :, 7:], chunks[:, :, 7:])
    assert torch.all(chunks == 2) and torch.all(actual[:, :5, :7] > 1)  # No clipping or gripper replacement.
    assert torch.equal(torch.random.get_rng_state(), before)
    reverse = add_exploration_noise(chunks.flip(0), _slots((33, 32)), task=contract["tasks"][0], contract=contract)
    torch.testing.assert_close(reverse.flip(0), actual)


def test_j0_is_exact_passthrough_and_records_the_same_independent_seed_schedule():
    chunks = torch.randn(2, 50, 7)
    zero, noisy = _slots(), _slots()
    j0, sigma = _contract(), _contract(True)
    assert add_exploration_noise(chunks, zero, task=j0["tasks"][0], contract=j0) is chunks
    add_exploration_noise(chunks, noisy, task=sigma["tasks"][0], contract=sigma)
    for left, right in zip(zero, noisy, strict=True):
        assert left["exploration_noise_seeds"] == right["exploration_noise_seeds"]
        assert left["exploration_noise_seeds"][0] != policy_noise_seed(7, "libero_spatial", 0, left["init_state_id"], 2)
    assert add_exploration_noise(chunks, zero, task=j0["tasks"][0], contract={}) is chunks


def _observation():
    return {"agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((4, 4, 3), dtype=np.uint8),
        "robot0_eef_pos": [0., 0., 0.], "robot0_eef_quat": [0., 0., 0., 1.],
        "robot0_gripper_qpos": [0., 0.]}


def _preprocess(value):
    return {key: item.unsqueeze(0) for key, item in value.items() if isinstance(item, torch.Tensor)}


class _Policy:
    config = SimpleNamespace(chunk_size=50, max_action_dim=32)

    def __init__(self):
        self.noise = []

    def reset(self):
        pass

    def predict_action_chunk(self, batch, *, noise, num_steps):
        assert num_steps == 10
        self.noise.append(noise.clone())
        return torch.full((noise.shape[0], 50, 7), 2.)


def test_planner_adds_noise_before_canonical_postprocess_and_preserves_flow_noise():
    planned, policies = [], []
    for enabled in (False, True):
        contract, policy = _contract(enabled), _Policy()
        slots = [slot | {"obs": _observation(), "action_plan": deque(), "policy_noise_seeds": []}
                 for slot in _slots()]
        _plan_action_chunks(slots, task=contract["tasks"][0], contract=contract, policy=policy,
            preprocess=_preprocess, postprocess=lambda chunks: 3 * chunks + 1,
            task_adapter=None, root_seed=7, replan_steps=5)
        planned.append(slots)
        policies.append(policy)
    torch.testing.assert_close(policies[0].noise[0], policies[1].noise[0], rtol=0, atol=0)
    expected_chunks = add_exploration_noise(torch.full((2, 50, 7), 2.), _slots(),
        task=_contract(True)["tasks"][0], contract=_contract(True))
    for index in range(2):
        assert planned[0][index]["policy_noise_seeds"] == planned[1][index]["policy_noise_seeds"]
        np.testing.assert_array_equal(np.asarray(planned[0][index]["action_plan"]), np.full((5, 7), 7.))
        np.testing.assert_allclose(np.asarray(planned[1][index]["action_plan"]),
                                   (3 * expected_chunks[index, :5] + 1).numpy())


class _Env:
    def __init__(self):
        self.actions = []

    def seed(self, seed):
        assert seed == 7

    def reset(self):
        return _observation()

    def set_init_state(self, state):
        assert state == 32
        return _observation()

    def step(self, action):
        self.actions.append(action.copy())
        return _observation(), 0., len(self.actions) == 12, {}


def test_early_success_retains_full_decision_noise_evidence_and_strict_row_validation():
    contract, env = _contract(True), _Env()
    task = contract["tasks"][0]
    rows = rollout_shard(envs=[env], init_states=tuple(range(50)), task=task, state_ids=(32,),
        contract=contract, policy=_Policy(), preprocess=_preprocess, postprocess=lambda chunks: chunks)
    assert len(env.actions) == 12 and rows[0]["steps"] == 2
    assert all(np.array_equal(action, contract["environment"]["dummy_action"]) for action in env.actions[:10])
    assert len(rows[0]["diagnostic_exploration"]["noise_seeds"]) == 1
    assert rows[0]["diagnostic_exploration"]["covariance"]["shape"] == [35, 35]
    shard = EvaluationShard("job", 0, "libero_spatial", 0, 220, (32,), 220)
    payload = {"schema_version": SHARD_RESULT_SCHEMA, "contract_reference": "test", "job_id": "job",
        "shard": asdict(shard), "rows": rows, "producer": {"worker_id": "0-r0", "claim_token": "a" * 32, "attempt": 1},
        "started_unix": 1., "finished_unix": 2.}
    assert validate_shard_result(payload, contract=contract, shard=shard) == rows
    rows[0]["diagnostic_exploration"]["noise_seeds"][0] += 1
    with pytest.raises(Pi05EvaluationError, match="row contract changed"):
        validate_shard_result(payload, contract=contract, shard=shard)


@pytest.mark.parametrize("change", ["validation", "test", "formal", "states", "rho", "seed", "flow", "action", "occupancy"])
def test_exploration_scope_and_sigma_cannot_drift(change):
    contract = _contract(True)
    if change in ("validation", "test"):
        contract["role"] = change
    elif change == "formal":
        contract["mode"] = change
    elif change == "states":
        contract["tasks"][0]["init_state_ids"] = list(range(5))
    elif change == "rho":
        contract["diagnostic_exploration"]["covariance"]["temporal_rho"] = 0.7
    elif change == "seed":
        contract["diagnostic_exploration"]["seed_root"] += 1
    elif change == "flow":
        contract["policy"]["num_inference_steps"] = 9
    elif change == "action":
        contract["policy"]["action_dim"] = 8
    else:
        contract["diagnostic_occupancy_capture"] = {"path": "/capture"}
    with pytest.raises(Pi05EvaluationError, match="exploration scope"):
        validate_exploration_contract(contract)


def test_flag_requires_explicit_diagnostic_states_and_normal_validation_remains_j0():
    with pytest.raises(Pi05EvaluationError, match="explicit development-train"):
        _explicit_diagnostic_states(SimpleNamespace(exploration_sigma=True, init_state_ids=None))
    contract = _contract()
    contract["role"] = "validation"
    contract["tasks"][0]["init_state_ids"] = list(range(50))
    assert build_exploration_contract(contract, enabled=False) is None
    with pytest.raises(Pi05EvaluationError, match="development_train screen"):
        build_exploration_contract(contract, enabled=True)


def test_sigma_cli_and_reloaded_contract_preserve_the_explicit_scope(tmp_path):
    import argparse
    from scripts.evaluate_pi05 import _add_prepare_arguments

    parser = argparse.ArgumentParser()
    _add_prepare_arguments(parser)
    args = parser.parse_args(["--source-run", "/source", "--checkpoint", "/checkpoint",
        "--tokenizer-path", "/tokenizer", "--output-dir", "/output", "--role", "development_train",
        "--mode", "screen", "--replicas-per-gpu", "1", "--state-count", "5",
        "--init-state-ids", "32,33,34,35,36", "--exploration-sigma"])
    assert args.exploration_sigma and _explicit_diagnostic_states(args) == DIAGNOSTIC_STATES
    path = tmp_path / "run_contract.json"
    contract = _contract(True) | {"schema_version": RUN_CONTRACT_SCHEMA,
        "contract_reference": f"{RUN_CONTRACT_SCHEMA}:fixture", "content_hash_policy": "disabled_by_owner",
        "output_dir": str(tmp_path)}
    path.write_text(json.dumps(contract))
    assert load_run_contract(path)["diagnostic_exploration"]["condition"] == "J_Sigma"
    path.write_text(json.dumps(contract | {"role": "validation"}))
    with pytest.raises(Pi05EvaluationError, match="exploration scope"):
        load_run_contract(path)


def _panels():
    result = []
    for enabled in (False, True):
        contract = _contract(enabled)
        rows = []
        for state in DIAGNOSTIC_STATES:
            replans = 2 if not enabled else 1
            row = {"suite": "libero_spatial", "task_id": 0, "init_state_id": state, "split_role": "train",
                "language": "move bowl", "env_seed": 7, "policy_seed_root": 7, "success": enabled,
                "policy_noise_seeds": [policy_noise_seed(7, "libero_spatial", 0, state, index) for index in range(replans)],
                "horizon_writer_lora": {"writer_checkpoint": {"path": "/writer/checkpoint"}, "K": 1,
                    "video_ordinal": 0, "selection_seed": 20260907, "selection_mode": "fixed_per_task",
                    "teacher_demo_indices": [46], "paired_correct_demos": [46], "paired_other_demos": [47]}}
            seeds = [exploration_noise_seed(contract["diagnostic_exploration"], suite="libero_spatial", task_id=0,
                     state_id=state, replan=index) for index in range(replans)]
            row.update(episode_exploration_fields(contract, {"exploration_noise_seeds": seeds}))
            rows.append(row)
        result.append({"rows": rows})
    return result


def test_only_explicit_comparison_accepts_declared_j0_sigma_pair():
    panels = _panels()
    with pytest.raises(Pi05EvaluationError, match="explicit exploration-pair"):
        paired_success_comparison(*panels)
    assert paired_success_comparison(*panels, allow_exploration_pair=True)["gained"] == 5
    with pytest.raises(Pi05EvaluationError, match="same checkpoint"):
        paired_success_comparison(panels[0], panels[0], allow_exploration_pair=True)


def test_legacy_source_rows_remain_comparable_to_declared_j0():
    current = _panels()[0]
    historical = copy.deepcopy(current)
    for row in historical["rows"]:
        row.pop("diagnostic_exploration")
        row.pop("horizon_writer_lora")
    assert paired_success_comparison(historical, current)["churn_count"] == 0


@pytest.mark.parametrize("change", ["teacher", "checkpoint", "seed", "flow", "rho"])
def test_explicit_comparison_does_not_relax_other_pairing(change):
    panels = _panels()
    row = panels[1]["rows"][0]
    if change == "teacher":
        row["horizon_writer_lora"]["teacher_demo_indices"] = [49]
    elif change == "checkpoint":
        row["horizon_writer_lora"]["writer_checkpoint"]["path"] = "/different"
    elif change == "seed":
        row["diagnostic_exploration"]["noise_seeds"][0] += 1
    elif change == "flow":
        row["policy_noise_seeds"][0] += 1
    else:
        row["diagnostic_exploration"]["covariance"]["temporal_rho"] = 0.7
    with pytest.raises(Pi05EvaluationError):
        paired_success_comparison(*panels, allow_exploration_pair=True)


def test_comparison_pipeline_requires_same_adapter_and_explicit_flag(tmp_path):
    roots = [tmp_path / "j0", tmp_path / "sigma"]
    normalizer = tmp_path / "normalizer.json"
    normalizer.write_text(json.dumps({"source_only": True}))
    for root, panel, enabled in zip(roots, _panels(), (False, True), strict=True):
        root.mkdir()
        contract = _contract(enabled) | {"normalization": {"path": str(normalizer)}, "contract_reference": str(root)}
        (root / "run_contract.json").write_text(json.dumps(contract))
        (root / "results.json").write_text(json.dumps(panel | {
            "schema_version": AGGREGATE_SCHEMA, "contract_reference": str(root), "arm": "correct"}))
        (root / "launcher_completion.json").write_text(json.dumps({
            "contract_reference": str(root), "return_codes": {"0-r0": 0}}))
    with pytest.raises(Pi05EvaluationError, match="explicit exploration-pair"):
        compare(*roots)
    observed = compare(*roots, allow_exploration_pair=True)
    assert observed["comparison_kind"] == "paired_training_J0_J_Sigma" and observed["comparison"]["gained"] == 5
    left, right = _contract(), _contract(True)
    right["adapter"] = {"checkpoint": "different"}
    with pytest.raises(Pi05EvaluationError, match="one adapter"):
        validate_exploration_comparison(left, right, allow_exploration_pair=True)
