import json

import torch
from safetensors.torch import save_file

from ember.ecp.contracts import TargetFamily, TargetOwner
from ember.ecp.realizer_code import (
    fit_weighted_owner_pca,
    held_global_ids,
    task_equal_member_weights,
    transform_owner_particles,
)
from ember.ecp.realizer_evidence import (
    balanced_member_shards,
    effect_member_tensors,
    load_effect_member,
    save_effect_member,
)
from ember.ecp.realizer_model import (
    FixedEffectRealizer,
    fixed_effect_realizer_loss,
)
from ember.ecp.realizer_materialization import merge_carrier_residual
from ember.ecp.realizer_training_data import load_held_effect_code_batch
from ember.lora import (
    LORA_A_SUFFIX,
    LORA_B_SUFFIX,
    LoRATarget,
    SmolVLALoRAContract,
)


def _contract() -> SmolVLALoRAContract:
    return SmolVLALoRAContract(
        targets=(LoRATarget("tiny", in_features=3, out_features=2),),
        rank=5,
        alpha=5,
        dropout=0.0,
        identity_seed=1,
    )


def test_balanced_member_shards_use_trajectory_cost() -> None:
    rows = [
        {"index": index, "trajectories": [{}] * count}
        for index, count in enumerate((2, 2, 2, 1, 1, 1))
    ]
    shards = balanced_member_shards(rows, 3)
    assert sorted(index for shard in shards for index in shard) == list(range(6))
    loads = [sum(len(rows[index]["trajectories"]) for index in shard) for shard in shards]
    assert loads == [3, 3, 3]


def test_effect_member_tensors_keep_particles_and_rank4_targets(tmp_path) -> None:
    contract = _contract()
    residual = {
        "tiny.lora_A.default.weight": torch.ones(4, 3),
        "tiny.lora_B.default.weight": torch.ones(2, 4),
    }
    tensors = effect_member_tensors(
        owner_delta=torch.ones(4, 8, 38, 4, 128),
        residual=residual,
        trajectory_count=2,
        contract=contract,
    )
    assert tensors["owner_delta"].shape == (4, 8, 38, 4, 128)
    assert tensors["particle_trajectory_ids"].tolist() == [0, 0, 1, 1]
    assert tensors["particle_probe_signs"].tolist() == [1, -1, 1, -1]
    assert tensors["target_00_a"].shape == (4, 3)
    assert tensors["target_00_b"].shape == (2, 4)
    path = tmp_path / "member.safetensors"
    save_effect_member(
        path=path,
        owner_delta=torch.ones(4, 8, 38, 4, 128),
        residual=residual,
        trajectory_count=2,
        contract=contract,
    )
    loaded, _, trajectory_ids, probe_signs = load_effect_member(
        path, contract=contract
    )
    assert loaded.shape == (4, 8, 38, 4, 128)
    assert trajectory_ids.tolist() == [0, 0, 1, 1]
    assert probe_signs.tolist() == [1, -1, 1, -1]


def test_fold_and_task_weights_do_not_count_extra_members_as_tasks() -> None:
    config = {"target_train_global_ids_ordered": list(range(24))}
    assert held_global_ids(config, 0) == (0, 5, 10, 15, 20)
    rows = [
        {"asset_key": "a"},
        {"asset_key": "a"},
        {"asset_key": "b"},
    ]
    weights = task_equal_member_weights(rows, (0, 1, 2))
    assert torch.allclose(weights, torch.tensor([0.25, 0.25, 0.5]).double())


def test_weighted_owner_pca_uses_a_frozen_owner_local_basis() -> None:
    values = torch.tensor(
        [
            [[-2.0, 0.0], [0.0, -1.0]],
            [[-1.0, 0.0], [0.0, -2.0]],
            [[1.0, 0.0], [0.0, 2.0]],
            [[2.0, 0.0], [0.0, 1.0]],
        ]
    )
    mean, components, scales, explained = fit_weighted_owner_pca(
        values, torch.ones(4), width=1
    )
    assert mean.shape == (2, 2)
    assert components.shape == (2, 1, 2)
    assert scales.shape == (2, 1)
    assert torch.all(explained > 0.7)
    assert torch.allclose(components[..., 1].abs(), torch.tensor([[0.0], [1.0]]))


def test_effect_coordinate_transform_preserves_particle_event_and_owner_axes() -> None:
    code = transform_owner_particles(
        torch.ones(2, 8, 38, 4, 128),
        mean=torch.zeros(38, 512),
        components=torch.eye(512)[:128].expand(38, -1, -1),
        scales=torch.ones(38, 128),
    )
    assert code.shape == (2, 8, 38, 128)


def test_held_code_loader_never_opens_held_target_residuals(tmp_path) -> None:
    held_ids = (0, 9, 18, 25, 36)
    rows = [
        {
            "fold_role": "fit",
            "global_task_id": index,
            "member": "fit",
            "effect_code_path": str(tmp_path / "unused"),
        }
        for index in range(108)
    ]
    for global_id in held_ids:
        rows.append(
            {
                "fold_role": "held_transform_only",
                "global_task_id": global_id,
                "member": "earliest",
                "effect_code_path": str(tmp_path / "unused"),
            }
        )
        code_path = tmp_path / f"code_{global_id}.safetensors"
        save_file(
            {"effect_code": torch.ones(2, 8, 38, 128, dtype=torch.bfloat16)},
            str(code_path),
        )
        rows.append(
            {
                "fold_role": "held_transform_only",
                "global_task_id": global_id,
                "member": "latest",
                "effect_code_path": str(code_path),
                "tensor_path": str(tmp_path / "must_not_be_opened"),
                "reliability": 0.5,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "ember_ecp_fixed_effect_code_authority_v1",
                "status": "complete_fit_only_effect_code_coordinate",
                "held_tasks": 5,
                "held_global_task_ids": list(held_ids),
                "members": rows,
            }
        )
    )
    batch = load_held_effect_code_batch(
        manifest_path=manifest, device=torch.device("cpu")
    )
    assert batch.code.shape == (5, 4, 8, 38, 128)
    assert batch.particle_mask.sum().item() == 10
    assert tuple(int(row["global_task_id"]) for row in batch.rows) == held_ids


def test_merge_carrier_residual_replaces_only_mobile_rank4() -> None:
    contract = SmolVLALoRAContract(
        targets=(LoRATarget("tiny", in_features=3, out_features=2),),
        rank=16,
        alpha=16,
        dropout=0.0,
        identity_seed=1,
    )
    a_name = "tiny" + LORA_A_SUFFIX
    b_name = "tiny" + LORA_B_SUFFIX
    carrier = {
        a_name: torch.arange(48).reshape(16, 3).float(),
        b_name: torch.cat((torch.ones(2, 12), torch.zeros(2, 4)), dim=1),
    }
    residual_a = torch.full((4, 3), 7.0)
    residual_b = torch.full((2, 4), 9.0)
    merged = merge_carrier_residual(
        carrier=carrier,
        residual=((residual_a, residual_b),),
        contract=contract,
    )
    assert torch.equal(merged[a_name][:12], carrier[a_name][:12])
    assert torch.equal(merged[b_name][:, :12], carrier[b_name][:, :12])
    assert torch.equal(merged[a_name][12:], residual_a)
    assert torch.equal(merged[b_name][:, 12:], residual_b)


def test_fixed_effect_realizer_preserves_owner_outputs_and_has_finite_loss() -> None:
    targets = tuple(
        LoRATarget(f"tiny_{index}", in_features=2, out_features=3)
        for index in range(38)
    )
    contract = SmolVLALoRAContract(
        targets=targets,
        rank=4,
        alpha=4,
        dropout=0.0,
        identity_seed=1,
    )
    owners = tuple(
        TargetOwner(
            index=index,
            target_name=target.name,
            family=TargetFamily.Q,
            layer=index % 18,
            in_features=target.in_features,
            out_features=target.out_features,
        )
        for index, target in enumerate(targets)
    )
    model = FixedEffectRealizer(
        contract=contract,
        owners=owners,
        a_scales=torch.ones(38),
        b_scales=torch.ones(38),
        bottleneck=2,
    )
    code = torch.randn(2, 2, 8, 38, 128)
    mask = torch.ones(2, 2, dtype=torch.bool)
    prediction = model(code, mask, torch.ones(2))
    target = tuple((torch.randn_like(a), torch.randn_like(b)) for a, b in prediction)
    null = model(torch.zeros_like(code[:1]), mask[:1], torch.zeros(1))
    loss = fixed_effect_realizer_loss(
        prediction=prediction,
        target=target,
        null_prediction=null,
        a_scales=torch.ones(38),
        b_scales=torch.ones(38),
        null_weight=0.05,
    )
    assert len(prediction) == 38
    assert prediction[0][0].shape == (2, 4, 2)
    assert prediction[0][1].shape == (2, 3, 4)
    assert torch.isfinite(loss.total)
    loss.total.backward()
