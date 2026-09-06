from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ember.ecp.contracts import ACTION_HORIZON, TargetFamily, TargetOwner
from ember.ecp.policy_response_writer import (
    FrozenPolicyResponseVideo,
    PolicyResponseEvidence,
    CompletePolicyResponseWriter,
)
from ember.ecp.policy_response_writer.materialization import (
    VALIDATION_EVALUATION_SCHEMA,
    _validation_deployment_tasks,
    load_writer_evaluation_config,
)
from ember.ecp.policy_response_writer.shared import _optimizer, functional_objective
from ember.ecp.policy_response_writer.shared_execution import (
    assignment_makespan,
    cost_balanced_task_assignment,
)
from ember.ecp.policy_response_writer.shared_schedule import (
    counted_task_group,
    task_group_counts,
    task_occurrence_schedule,
    training_video_demos,
)
from ember.ecp.policy_response_writer.training import load_policy_response_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def _owners() -> tuple[TargetOwner, ...]:
    return (
        TargetOwner(0, "q", TargetFamily.Q, 0, 6, 16),
        TargetOwner(1, "v", TargetFamily.V, 1, 7, 8),
        TargetOwner(2, "action_in", TargetFamily.ACTION_IN, None, 2, 8),
        TargetOwner(3, "action_out", TargetFamily.ACTION_OUT, None, 5, 3),
    )


def _video(seed: int, *, frames: int = 6) -> FrozenPolicyResponseVideo:
    generator = torch.Generator().manual_seed(seed)
    return FrozenPolicyResponseVideo(
        patch_states=torch.randn(frames, 5, 10, generator=generator),
        language_states=torch.randn(frames, 3, 10, generator=generator),
        language_mask=torch.ones(frames, 3, dtype=torch.bool),
        layer_states=torch.randn(
            frames, 2, 19, ACTION_HORIZON, 12, generator=generator
        ),
        flow_velocity=torch.randn(
            frames, 2, ACTION_HORIZON, 32, generator=generator
        ),
        suffix_noise=torch.randn(2, ACTION_HORIZON, 32, generator=generator),
        frame_positions=torch.linspace(0.0, 1.0, frames),
    )


def _model() -> CompletePolicyResponseWriter:
    return CompletePolicyResponseWriter(
        _owners(), prefix_width=10, expert_width=12, rank=16,
        width=16, heads=4, blocks=2, process_tokens=3,
    )


def _group_gradient(model: torch.nn.Module, prefix: str) -> float:
    rows = [
        parameter.grad.detach().float().square().sum()
        for name, parameter in model.named_parameters()
        if name.startswith(prefix) and parameter.grad is not None
    ]
    return float(torch.stack(rows).sum().sqrt()) if rows else 0.0


def test_complete_factors_start_at_identity_and_functional_training_opens_the_graph() -> None:
    torch.manual_seed(7)
    model = _model()
    video = _video(7, frames=3)
    output = model((video,)).factors
    assert tuple(value.shape for value in output.a) == ((16, 6), (16, 7), (16, 2), (16, 5))
    assert tuple(value.shape for value in output.b) == ((16, 16), (16, 8), (16, 8), (16, 3))
    assert all(torch.count_nonzero(value) == 0 for value in output.b)
    assert all(torch.count_nonzero(value) > 0 for value in output.a)
    targets = tuple(torch.randn(owner.out_features, owner.in_features) for owner in _owners())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    for iteration in range(2):
        optimizer.zero_grad(set_to_none=True)
        factors = model((video,)).factors
        loss = sum((b.T @ a - target).square().mean()
                   for a, b, target in zip(factors.a, factors.b, targets, strict=True))
        loss.backward()
        assert _group_gradient(model, "factor_writer.factor_heads") > 0
        if iteration == 1:
            for prefix in ("evidence.response", "evidence.prefix", "factor_writer.blocks"):
                assert _group_gradient(model, prefix) > 0
        optimizer.step()
    updated = model((video,)).factors
    assert all(torch.count_nonzero(value) > 0 for value in updated.b)


def test_full_horizon_remains_explicit_before_the_learned_read() -> None:
    model = _model()
    video = _video(11, frames=3)
    response = model.evidence.response(video)
    assert response.shape == (video.frame_count, 78 * ACTION_HORIZON, 16)
    changed = video.layer_states.clone()
    changed[:, :, :, 17] += 3.0
    mutated = model.evidence.response(replace(video, layer_states=changed))
    mask = torch.arange(response.shape[1]) % ACTION_HORIZON == 17
    assert not torch.equal(response[:, mask], mutated[:, mask])
    torch.testing.assert_close(response[:, ~mask], mutated[:, ~mask])
    with pytest.raises(ValueError, match="full policy-response"):
        model((video,), representation="coarse")


def test_learned_video_set_read_is_permutation_invariant() -> None:
    model = _model().eval()
    # Open the heads so this assertion cannot pass merely because fresh B is zero.
    for head in model.factor_writer.factor_heads.values():
        torch.nn.init.normal_(head.a_head[-1].weight, std=0.03)
        torch.nn.init.normal_(head.b_head[-1].weight, std=0.03)
    left, right = _video(29, frames=3), _video(31, frames=4)
    with torch.no_grad():
        original = model((left, right)).factors
        permuted = model((right, left)).factors
    for expected, observed in zip((*original.a, *original.b), (*permuted.a, *permuted.b), strict=True):
        torch.testing.assert_close(observed, expected, rtol=2e-5, atol=2e-6)


def test_full_response_cache_roundtrip_has_no_unused_target_banks(tmp_path: Path) -> None:
    from ember.ecp.policy_response_writer.shared_video_cache import SharedPolicyResponseVideoCache
    video = _video(41, frames=3)
    cache = SharedPolicyResponseVideoCache(tmp_path / "video_cache", authority={"run": "test"})
    calls = []
    def build():
        calls.append(1)
        return video, {"sampled_frames": video.frame_count}
    first = cache.get_or_build(task=1, demo=2, builder=build)
    second = cache.get_or_build(task=1, demo=2, builder=build)
    assert calls == [1] and not first.hit and second.hit
    assert not hasattr(second.video, "native_inputs")
    torch.testing.assert_close(second.video.layer_states, video.layer_states)
    assert second.video.tensor_bytes == video.tensor_bytes


def test_shared_optimizer_owns_the_whole_writer_in_one_group() -> None:
    writer = _model()
    policy = torch.nn.Linear(2, 2).requires_grad_(False)
    stage0 = torch.nn.Linear(2, 2).requires_grad_(False)
    runtime = SimpleNamespace(
        writer=writer,
        policy=policy,
        stage0=stage0,
        config={
            "optimization": {
                "shared": {
                    "training_stage": "joint_functional_positive_only",
                    "learning_rate": 1e-4,
                    "decay_learning_rate": 1e-6,
                    "betas": [0.9, 0.95],
                    "weight_decay": 0.01,
                    "warmup_updates": 2,
                    "effective_updates": 4,
                }
            }
        },
    )
    parameters, optimizer, _ = _optimizer(runtime)
    assert {id(value) for value in parameters} == {
        id(value) for value in writer.parameters()
    }
    assert len(optimizer.param_groups) == 1


def test_positive_functional_objective_has_no_negative_video_term() -> None:
    row = functional_objective(
        generated_loss=1.2,
        normalizer=2.0,
        task_weight=0.25,
    )
    assert row["gradient_mass"] == pytest.approx(0.125)
    assert set(row) == {
        "functional_normalized",
        "gradient_mass",
    }


def test_task_batch_size_and_role_ratio_are_experiment_settings() -> None:
    meta = tuple(range(10))
    target = tuple(range(20, 26))
    assert task_group_counts(
        {
            "global_tasks_per_update": 5,
            "tasks_per_update_by_role": {"meta": 4, "target": 1},
        },
        meta=meta,
        target=target,
    ) == (4, 1)
    group = counted_task_group((meta, target), (4, 1), 0, seed=17)
    assert len(group) == 5
    assert len(set(group).intersection(meta)) == 4
    assert len(set(group).intersection(target)) == 1


def test_task_local_occurrence_drives_k_without_global_step_aliasing() -> None:
    groups = ((1, 2), (3, 4), (1, 3), (2, 4), (1, 4))
    occurrences = task_occurrence_schedule(groups)
    assert [row.get(1) for row in occurrences if 1 in row] == [0, 1, 2]
    selected = [
        training_video_demos(
            (0, 1, 2, 3),
            task_occurrence=row[1],
            task=1,
            cardinalities=(1, 2, 4),
            seed=19,
        )
        for row in occurrences
        if 1 in row
    ]
    assert {len(value) for value in selected} == {1, 2, 4}


def test_dynamic_cost_assignment_reduces_tail_without_changing_tasks() -> None:
    costs = {0: 19, 1: 17, 2: 13, 3: 11, 4: 7, 5: 5}
    eligibility = {task: (0, 1, 2) for task in costs}
    assignment = cost_balanced_task_assignment(
        tuple(costs), costs, eligibility, world_size=3
    )
    assert {task for row in assignment for task in row} == set(costs)
    assert assignment_makespan(assignment, costs) <= 25


def test_complete_config_is_canonical_and_residual_config_is_rejected() -> None:
    current = load_policy_response_config(REPO_ROOT / "configs/pi05_ecp_prw_complete_shared4_v1.json")
    assert current["model"]["complete_rank"] == 16
    assert current["model"]["carrier_installed"] is False
    assert "residual_rank" not in current["model"]
    with pytest.raises(ValueError, match="invalid Policy-Response Writer config"):
        load_policy_response_config(REPO_ROOT / "configs/pi05_ecp_prw_meta73_equal_exposure_v1.json")
    evaluation = load_writer_evaluation_config(
        REPO_ROOT / "configs/pi05_ecp_prw_complete_shared4_held_video_eval_v1.json"
    )
    assert evaluation["training_config"].endswith("prw_complete_shared4_v1.json")


def test_validation_materialization_config_keeps_test_and_gradients_closed(tmp_path: Path) -> None:
    import copy
    import json
    from ember.pi05_eval_contract import SUITE_ORDER
    from ember.static_task_lora import validation_task_keys

    config = json.loads((REPO_ROOT / "configs/pi05_ecp_prw_complete_shared4_held_video_eval_v1.json").read_text())
    config.update(schema_version=VALIDATION_EVALUATION_SCHEMA,
                  status="active_correct_only_validation_materialization",
                  evaluation_role="validation", task_subset=None, require_training_completion=True,
                  target_global_ids=[SUITE_ORDER.index(suite) * 10 + task
                                     for suite, task in validation_task_keys()])
    config["condition"].update(selection="predeclared_fixed_validation8_correct_video",
                               checkpoint_selection_use=True,
                               video_demos_by_global_task={str(task): [0] for task in config["target_global_ids"]})
    config["information_wall"].update(validation_or_test_use=True, test_use=False)
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(config))
    assert load_writer_evaluation_config(path)["evaluation_role"] == "validation"
    for group, key, value in (
        (None, "evaluation_role", "test"),
        (None, "target_global_ids", [2, 20, 38]),
        (None, "require_training_completion", False),
        ("condition", "gradient_use", True),
        ("condition", "outcome_dependence", True),
        ("information_wall", "test_use", True),
        ("information_wall", "held_action_or_reward_reads", 1),
    ):
        changed = copy.deepcopy(config)
        (changed if group is None else changed[group])[key] = value
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="unsupported Policy-Response Writer"):
            load_writer_evaluation_config(path)


def test_validation_deployment_uses_video_only_metadata_and_cannot_enter_training(tmp_path: Path) -> None:
    import json
    import h5py
    import numpy as np
    from ember.ecp.policy_response_writer.training import prepare_runtime, _functional_runtime_inputs
    from ember.pi05_eval_contract import SUITE_ORDER
    from ember.static_task_lora import validation_task_keys
    from ember.writer.data import RawTeacherVideoStore

    video = tmp_path / "video.h5"
    with h5py.File(video, "w") as handle:
        handle.create_dataset("data/demo_0/obs/agentview_rgb", data=np.zeros((2, 8, 8, 3), dtype=np.uint8))
    records = [{"global_task_id": SUITE_ORDER.index(suite) * 10 + task,
                "suite": suite, "task_id": task, "split_role": "validation",
                "language": f"exact {suite} {task}", "task_name": "fixture",
                "problem_folder": suite, "bddl": {"filename": "fixture.bddl"},
                "hdf5": {"relative_path": video.name, "bytes": video.stat().st_size},
                "demonstrations": {"episode_lengths": [2] * 50}}
               for suite, task in validation_task_keys()]
    (tmp_path / "target.json").write_text(json.dumps({"tasks": records}))
    (tmp_path / "base.json").write_text(json.dumps({"authorities": {"target_manifest": "target.json"}}))
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps({"authorities": {"base_g3_config": "base.json"}}))
    args = SimpleNamespace(config=config_path, asset_root=tmp_path, data_root=tmp_path, phase="shared")
    evaluation = {"evaluation_role": "validation", "target_global_ids": [r["global_task_id"] for r in records]}
    tasks = _validation_deployment_tasks(args, evaluation)
    assert len(tasks) == 8 and all(task.role == "target_validation" for task in tasks)
    store = RawTeacherVideoStore(tuple(task.writer_authority() for task in tasks), frame_stride=5)
    try:
        assert store.load(tasks[0].authority_id, 0).raw_frame_count == 2
    finally:
        store.close()
    assert _functional_runtime_inputs(authorities=(), source_config={}, base={}, args=None,
                                      context=None, enabled=False) == (None, None)
    with pytest.raises(ValueError, match="cannot enter Writer training"):
        prepare_runtime(args, None, deployment_global_ids=tuple(evaluation["target_global_ids"]),
                        deployment_tasks=tasks)
    with pytest.raises(ValueError, match="fixed validation8"):
        _validation_deployment_tasks(args, {**evaluation, "target_global_ids": evaluation["target_global_ids"][:-1]})
