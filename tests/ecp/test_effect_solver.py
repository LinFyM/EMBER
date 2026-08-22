import torch

from ember.ecp.effect_solver import (
    ExactPolicyEffectTargets,
    PolicyEffectResponse,
    evaluate_policy_effect_state,
    relative_effective_update_distance,
    solve_policy_effects,
)
from ember.lora import SmolVLALoRAContract, LoRATarget
from ember.expert_manifold.projection import inspect_projected_task_expert_bank
from ember.pi05_source_checkpoint import write_json_atomic


def _tiny_contract():
    target = LoRATarget("tiny", in_features=2, out_features=2)
    contract = SmolVLALoRAContract(
        targets=(target,), rank=1, alpha=1, dropout=0.0, identity_seed=1
    )
    state = {
        "tiny.lora_A.default.weight": torch.tensor([[0.5, 0.5]]),
        "tiny.lora_B.default.weight": torch.tensor([[0.2], [0.2]]),
    }
    return contract, state


def test_relative_effective_distance_is_gauge_invariant() -> None:
    contract, state = _tiny_contract()
    rescaled = {
        "tiny.lora_A.default.weight": 4.0 * state["tiny.lora_A.default.weight"],
        "tiny.lora_B.default.weight": 0.25 * state["tiny.lora_B.default.weight"],
    }
    assert torch.allclose(
        relative_effective_update_distance(rescaled, state, contract),
        torch.zeros(()),
        atol=1e-6,
    )


def test_fixed_solver_reduces_exact_policy_effect_error() -> None:
    contract, initial = _tiny_contract()
    events = 2
    owner = torch.zeros(events, 1, 1, 1)
    flow = torch.zeros(events, 1, 1, 1)
    action = torch.zeros(events, 1, 1, 1)
    targets = ExactPolicyEffectTargets(
        source_owner=owner,
        source_flow=flow,
        source_action=action,
        shared_owner=owner,
        shared_flow=flow,
        shared_action=action,
        mean_owner=torch.full_like(owner, 0.4),
        variance_owner=owner,
        mean_flow=torch.full_like(flow, 0.4),
        variance_flow=flow,
        mean_action=torch.full_like(action, 0.4),
        variance_action=action,
        presence=torch.ones(events),
    )

    def response(state, _event):
        value = (
            state["tiny.lora_B.default.weight"]
            @ state["tiny.lora_A.default.weight"]
        ).mean()
        value = value.reshape(1, 1, 1, 1)
        return PolicyEffectResponse(owner=value, flow=value, action=value)

    candidate, history = solve_policy_effects(
        initial_state=initial,
        targets=targets,
        contract=contract,
        response=response,
        steps=5,
        step_rms=0.01,
        step_decay_power=0.5,
        owner_weight=1.0,
        flow_weight=1.0,
        action_weight=1.0,
        shared_barrier_weight=0.0,
        trust_region=10.0,
        trust_weight=0.0,
    )
    final = evaluate_policy_effect_state(
        state=candidate,
        targets=targets,
        response=response,
        owner_weight=1.0,
        flow_weight=1.0,
        action_weight=1.0,
        shared_barrier_weight=0.0,
    )
    assert final["effect"] < history[0].effect
    assert all(
        right.effect < left.effect for left, right in zip(history, history[1:])
    )


def test_pecs_projection_binds_the_five_task_oracle(tmp_path) -> None:
    config = tmp_path / "config.json"
    base_manifest = tmp_path / "base.json"
    config.write_text("{}")
    base_manifest.write_text("{}")
    base_tasks = []
    projected_tasks = []
    for ordinal in range(5):
        checkpoint = tmp_path / f"checkpoint_{ordinal}"
        adapter = tmp_path / f"adapter_{ordinal}.safetensors"
        adapter.write_bytes(b"adapter")
        base_tasks.append(
            {
                "suite": "libero_train",
                "task_id": ordinal,
                "ordinal": ordinal,
                "global_task_id": ordinal,
                "checkpoint": str(checkpoint),
            }
        )
        projected_tasks.append(
            {
                "suite": "libero_train",
                "task_id": ordinal,
                "ordinal": ordinal,
                "global_task_id": ordinal,
                "expert_checkpoint": str(checkpoint),
                "projected_adapter": str(adapter),
                "projected_adapter_bytes": adapter.stat().st_size,
            }
        )
    manifest = tmp_path / "projection.json"
    write_json_atomic(
        manifest,
        {
            "schema_version": "ember_ecp_policy_effect_solver_oracle_projection_v1",
            "projection_kind": "ecp_policy_effect_solver_exact_oracle",
            "repository": {"dirty_paths": []},
            "effect_oracle_config": {
                "path": str(config),
                "bytes": config.stat().st_size,
            },
            "base_projection_manifest": {
                "path": str(base_manifest),
                "bytes": base_manifest.stat().st_size,
            },
            "optimization": {
                "fit_profile_task_count": 1,
                "held_task_count": 5,
                "held_shared_gradient_steps": 0,
                "solver_algorithm_frozen": True,
                "solver_steps": 12,
                "per_task_early_stop": False,
                "task_local_persistent_optimizer": False,
                "single_complete_lora": True,
                "final_lora_averaging": False,
                "rank": 16,
                "second_adapter_deployed": False,
                "parameterization": (
                    "one complete rank16 LoRA solved from exact policy effects"
                ),
            },
            "information_wall": {
                "role": "development_train_leave_task_out_oracle_only",
                "deployment_carrier": False,
                "exact_privileged_effects": True,
                "teacher_action_forward_reads": 0,
                "second_adapter_deployed": False,
            },
            "tasks": projected_tasks,
        },
    )
    observed = inspect_projected_task_expert_bank(
        {
            "tasks": base_tasks,
            "information_wall": {"evaluation_role": "development_train"},
        },
        manifest,
    )
    assert observed["schema_version"] == (
        "ember_pi05_ecp_policy_effect_solver_oracle_eval_adapter_v1"
    )
    assert observed["arm"] == "ecp_pecs_exact_effect_oracle"
