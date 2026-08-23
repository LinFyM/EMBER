import torch

from ember.ecp.realizer_evidence import (
    balanced_member_shards,
    effect_member_tensors,
)
from ember.lora import LoRATarget, SmolVLALoRAContract


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


def test_effect_member_tensors_keep_particles_and_rank4_targets() -> None:
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
