from __future__ import annotations

import copy
import json
from collections import deque
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.stats import norm

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval import return_credit as capture
from ember.writer.return_credit import loo_advantages, score_cotangent
from ember.writer.score_conditioning import correlated_sign_score
from ember.pi05_eval.exploration import (
    DIAGNOSTIC_STATES, add_exploration_noise, alignment_metadata, build_exploration_contract,
    episode_exploration_fields, exploration_covariance, exploration_metadata, exploration_noise_seed,
    validate_exploration_comparison, validate_exploration_contract,
)
from ember.pi05_eval.trajectory_capture import initialize_capture
from ember.pi05_eval.preparation import _explicit_diagnostic_states
from ember.pi05_eval_contract import RUN_CONTRACT_SCHEMA, load_run_contract, policy_noise_seed
from ember.pi05_eval_queue import EvaluationShard
from ember.pi05_eval_results import AGGREGATE_SCHEMA, paired_success_comparison
from ember.pi05_evaluation import SHARD_RESULT_SCHEMA, _plan_action_chunks, rollout_shard, validate_shard_result
from scripts.compare_pi05_results import compare
from scripts.return_credit_analysis import _paired_native_noise
from scripts.return_score_update import SPEC_PATH as SCORE_UPDATE_SPEC, bank_keys, episode_keys
from scripts.return_score_update_analysis import _bootstrap as score_update_bootstrap, _compare as score_update_compare
from ember.pi05_source_checkpoint import read_json
from scripts.return_objective_alignment import (SPEC_PATH as ALIGNMENT_SPEC, bank_keys as alignment_banks,
    episode_keys as alignment_episodes, same_common_seed_prefix)


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


def _alignment_contract(cell="P_JS"):
    spec = read_json(ALIGNMENT_SPEC)
    output = spec["resources"]["study_root"] + f"/evaluation/{cell}/task_002_teacher_34/state_00"
    task = {"suite": "libero_spatial", "task_id": 2, "split_role": "train", "language": "move bowl",
            "horizon": 220}
    return {"role": "development_train", "mode": "formal", "adapter": None,
        "rng": {"inference_seed": 7},
        "policy": {"replan_steps": 5, "action_dim": 7, "chunk_size": 50, "num_inference_steps": 10},
        "diagnostic_exploration": alignment_metadata(enabled=cell.endswith("JS")),
        "diagnostic_occupancy_capture": {"mode": "compact", "trajectory_root": output + "/trajectories",
            "passive_trace": {"trace_root": output + "/continuous_traces"}},
        "diagnostic_stage_predicates": {"full_conditions_only": False},
        "frozen_objective_alignment": {"spec_path": str(ALIGNMENT_SPEC), "cell": cell,
            "model": cell.split("_")[0], "global_task": 2, "teacher_demo": 34,
            "init_state_id": 0, "replica": 4, "output": output}}, task


def test_objective_alignment_registered_scope_and_replica4_action_injection():
    spec = read_json(ALIGNMENT_SPEC)
    assert len(alignment_banks(spec)) == 16
    assert len(alignment_episodes(spec)) == 128
    for cell in ("P_J0", "P_JS", "RB_J0", "RB_JS"):
        contract, task = _alignment_contract(cell)
        validate_exploration_contract(contract, task=task, state_ids=(0,))
    j0, task = _alignment_contract("P_J0")
    js, _ = _alignment_contract("P_JS")
    chunks = torch.full((1, 50, 7), 2.)
    before = torch.random.get_rng_state()
    zero, noisy = [{"init_state_id": 0, "replan_index": 3}], [{"init_state_id": 0, "replan_index": 3}]
    initialize_capture(zero[0], "compact")
    initialize_capture(noisy[0], "compact")
    assert add_exploration_noise(chunks, zero, task=task, contract=j0) is chunks
    actual = add_exploration_noise(chunks, noisy, task=task, contract=js)
    assert zero[0]["exploration_noise_seeds"] == noisy[0]["exploration_noise_seeds"]
    from ember.pi05_eval.return_credit import exploration_seed
    seed = exploration_seed(2, 0, 4, 3)
    assert noisy[0]["exploration_noise_seeds"] == [seed]
    eta = torch.randn(35, generator=torch.Generator(device="cpu").manual_seed(seed)) @ torch.linalg.cholesky(exploration_covariance()).T
    torch.testing.assert_close(actual[0, :5], 2 + eta.reshape(5, 7))
    torch.testing.assert_close(actual[0, 5:], chunks[0, 5:])
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("change", ["teacher", "state", "replica", "model", "output", "flow", "capture", "task"])
def test_objective_alignment_rejects_scope_drift_before_rollout(change):
    contract, task = _alignment_contract()
    scope = contract["frozen_objective_alignment"]
    if change == "teacher":
        scope["teacher_demo"] = 46
    elif change == "state":
        scope["init_state_id"] = 32
    elif change == "replica":
        scope["replica"] = 0
    elif change == "model":
        scope["model"] = "RB"
    elif change == "output":
        scope["output"] += "_extra"
    elif change == "flow":
        contract["policy"]["num_inference_steps"] = 9
    elif change == "capture":
        contract["diagnostic_occupancy_capture"]["passive_trace"] = None
    else:
        task["task_id"] = 5
    with pytest.raises(Pi05EvaluationError, match="objective-alignment"):
        validate_exploration_contract(contract, task=task, state_ids=(0,))


def test_objective_alignment_planner_copies_one_pre_noise_mean_without_extra_forward():
    contract, task = _alignment_contract()
    policy = _Policy()
    slot = {"init_state_id": 0, "replan_index": 0, "obs": _observation(),
            "action_plan": deque(), "policy_noise_seeds": [], "steps": 0}
    initialize_capture(slot, "compact")
    _plan_action_chunks([slot], task=task, contract=contract, policy=policy,
        preprocess=_preprocess, postprocess=lambda chunks: 3 * chunks + 1,
        task_adapter=None, root_seed=7, replan_steps=5)
    assert len(policy.noise) == 1
    assert len(slot["pre_exploration_normalized_means"]) == 1
    torch.testing.assert_close(slot["pre_exploration_normalized_means"][0], torch.full((5, 7), 2.))
    assert len(slot["replay_action_chunks"]) == 1
    expected_slot = {"init_state_id": 0, "replan_index": 0}
    initialize_capture(expected_slot, "compact")
    expected = add_exploration_noise(torch.full((1, 50, 7), 2.),
        [expected_slot], task=task, contract=contract)
    torch.testing.assert_close(slot["replay_action_chunks"][0], expected)


def test_objective_alignment_pairs_only_replans_both_episodes_actually_reached():
    assert same_common_seed_prefix([11, 12], [11, 12, 13])
    assert same_common_seed_prefix([11, 12, 13], [11, 12])
    assert not same_common_seed_prefix([11, 12], [11, 99, 13])
    assert not same_common_seed_prefix([], [11])


def test_score_update_registered_96_scope_and_joint_teacher_state_bootstrap():
    spec = read_json(SCORE_UPDATE_SPEC)
    banks, episodes = bank_keys(spec), episode_keys(spec)
    assert len(banks) == len(set(banks)) == 48
    assert len(episodes) == len(set(episodes)) == 96
    assert len([key for key in episodes if key[1] in (2, 12, 22, 34)
                and key[2:] == (46, 32)]) == 12
    success = {}
    for task in spec["evaluation"]["task_ids"]:
        for state in (32, 33):
            for teacher in (46, 47):
                for arm, value in {"P": 0, "RAW": teacher == 46, "RB": 1}.items():
                    success[(task, state, teacher), arm] = int(value)
    bootstrap = score_update_bootstrap(spec, success)["comparisons"]["RB-RAW"]
    assert bootstrap["46"]["ci95"] == [0, 0]
    assert bootstrap["47"]["ci95"] == [1, 1]
    assert bootstrap["equal_two_teachers"]["ci95"] == [.5, .5]
    comparison = score_update_compare(success, spec["evaluation"]["task_ids"],
                                      (46, 47), "RB", "RAW")
    assert len(comparison["retained"]) == len(comparison["gained"]) == 16
    assert comparison["lost"] == []


def test_return_score_conditioning_independent_limit_and_temporal_correlation():
    threshold = -1. + 2. / (2. + 1e-6)
    signs = np.array([1., -1., 1., -1., 1.])
    x = np.array([.2, -.4, .8, 1.2, -.1])
    means = threshold + signs * .1 * x
    probability, score, _, _ = correlated_sign_score(means, signs, rho=0.)
    np.testing.assert_allclose(probability, np.prod(norm.cdf(x)), rtol=2e-4)
    np.testing.assert_allclose(score, signs/.1 * norm.pdf(x)/norm.cdf(x), rtol=2e-4)
    same_signs = np.ones(5)
    near_threshold = np.full(5, threshold + .02)
    correlated, revised, _, _ = correlated_sign_score(near_threshold, same_signs)
    independent, original, _, _ = correlated_sign_score(near_threshold, same_signs, rho=0.)
    assert correlated > independent
    assert not np.allclose(revised, original, atol=.01)
    assert np.isfinite(revised).all()


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


def test_loo_and_full_sigma_score_have_registered_sign_and_weight():
    advantages = loo_advantages([1, 0, 0, 0])
    torch.testing.assert_close(advantages, torch.tensor([1., -1/3, -1/3, -1/3]))
    old = torch.zeros(35)
    latent = torch.arange(35, dtype=torch.float32) / 1000
    precision = torch.eye(35)
    expected = latent * (1.5 * 20 / 4 / 128)
    torch.testing.assert_close(score_cotangent(latent, old, 1.5, Q=20, M=4,
                                               precision=precision), expected)
    with pytest.raises(ValueError):
        score_cotangent(latent, old, 1, Q=0, M=0)


def test_stateless_replica_noise_and_reward_independent_reservoir(tmp_path, monkeypatch):
    spec = capture.authority()
    spec["resources"]["study_root"] = str(tmp_path)
    monkeypatch.setattr(capture, "authority", lambda: spec)
    output = tmp_path / "collection" / "groups" / "task_002_state_00"
    task = {"suite": "libero_spatial", "task_id": 2}
    contract = {"return_credit_collection": {
        "global_task": 2, "init_state_id": 0, "replica": 0,
        "teacher_demo": 34, "output": str(output)},
        "rng": {"inference_seed": 7},
        "policy": {"num_inference_steps": 10, "replan_steps": 5}}
    slot = {"init_state_id": 0, "steps": 0, "replan_index": 0,
            "policy_noise_seeds": [], "obs": None}
    raw = {"observation.state": torch.zeros(8), "task": "pick"}
    processed = {"tokens": torch.zeros(1, 3)}
    for replan in range(6):
        slot["replan_index"], slot["steps"] = replan, replan * 5
        result = capture.explore_and_retain(torch.zeros(1, 50, 7), [slot],
            raw_inputs=[raw], processed=[processed], noise=torch.zeros(1, 50, 32),
            task=task, contract=contract)
        assert torch.isfinite(result).all()
        slot["policy_noise_seeds"].append(
            capture.policy_noise_seed(7, "libero_spatial", 2, 0, replan))
    assert slot["return_credit_reservoir"]["seen"] == 6
    assert len(slot["return_credit_reservoir"]["items"]) == 4
    assert capture.exploration_seed(2, 0, 0, 0) != capture.exploration_seed(2, 0, 1, 0)
    slot["steps"] = 27
    saved = capture.save_decisions(contract, task, slot)
    payload = torch.load(saved["path"], weights_only=True)
    assert (payload["Q"], payload["M"]) == (6, 4)
    assert all(len(row["executed_mask"]) == 5 for row in payload["decisions"])


def test_return_credit_scope_rejects_unregistered_task_state_and_teacher(tmp_path, monkeypatch):
    spec = capture.authority()
    spec["resources"]["study_root"] = str(tmp_path)
    monkeypatch.setattr(capture, "authority", lambda: spec)
    task = {"suite": "libero_spatial", "task_id": 2}
    scope = {"global_task": 2, "init_state_id": 0, "replica": 0,
             "teacher_demo": 34,
             "output": str(tmp_path / "collection/groups/task_002_state_00")}
    contract = {"return_credit_collection": scope, "rng": {"inference_seed": 7},
                "policy": {"num_inference_steps": 10, "replan_steps": 5}}
    assert capture.validate_collection(contract, task) == scope
    for field, invalid in (("global_task", 14), ("init_state_id", 32),
                           ("teacher_demo", 46), ("replica", 4)):
        changed = copy.deepcopy(contract)
        changed["return_credit_collection"][field] = invalid
        with pytest.raises(Pi05EvaluationError):
            capture.validate_collection(changed, task)


def test_return_credit_common_noise_allows_different_terminal_replan_counts():
    seeds = [policy_noise_seed(7, "libero_spatial", 2, 0, replan)
             for replan in range(3)]
    rows = [{"policy_noise_seeds": seeds[:2]},
            {"policy_noise_seeds": seeds}]
    _paired_native_noise(rows, 2, 0)
    rows[1]["policy_noise_seeds"][1] += 1
    with pytest.raises(ValueError, match="stateless"):
        _paired_native_noise(rows, 2, 0)
