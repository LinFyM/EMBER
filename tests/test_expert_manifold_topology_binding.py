from pathlib import Path

import torch

from ember.expert_manifold.contract import load_expert_manifold_config
from ember.expert_manifold.model import VideoConditionedTopologicalWriter
from ember.lora import identity_lora_state
from ember.pi05_lora import load_pi05_lora_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/pi05_video_expert_manifold_v1.json"


def _writer() -> VideoConditionedTopologicalWriter:
    contract = load_pi05_lora_contract(REPO_ROOT / "configs/pi05_lora_v1.json")
    return VideoConditionedTopologicalWriter(
        contract=contract,
        template_state=identity_lora_state(contract),
        phase_slots=4,
        feature_width=8,
        memory_width=16,
        attention_heads=4,
        axial_blocks=1,
        chunk_width=512,
    )


def test_address_binding_config_records_fresh_execution_evidence() -> None:
    config = load_expert_manifold_config(CONFIG)
    assert config["topological_writer"]["topology_address_binding"] == (
        "normalized_dynamic_times_normalized_chunk_plus_rank_address"
    )
    formal = config["meta_training"]["formal_run"]
    assert formal["status"] == "sealed"
    assert formal["profile_evidence"]["topology_address_binding"] == (
        config["topological_writer"]["topology_address_binding"]
    )
    assert formal["online_smoke_evidence"]["topology_address_binding"] == (
        config["topological_writer"]["topology_address_binding"]
    )


def test_dynamic_values_are_bound_to_chunk_and_rank_addresses() -> None:
    writer = _writer()
    dynamic = torch.randn(
        2, writer.layout.chunk_count, writer.layout.rank, writer.memory_width
    )
    dynamic = dynamic.mean(dim=(1, 2), keepdim=True).expand_as(dynamic).clone()
    address = writer.chunk_queries[:, None, :] + writer.rank_queries[None, :, :]
    bound = writer.bind_topology_address(dynamic, address)

    def centered_energy(value: torch.Tensor, dim: int) -> torch.Tensor:
        centered = value - value.mean(dim=dim, keepdim=True)
        return centered.square().sum() / value.square().sum()

    assert float(centered_energy(bound, 1).detach()) > 0.1
    assert float(centered_energy(bound, 2).detach()) > 0.1
    assert torch.count_nonzero(
        writer.bind_topology_address(torch.zeros_like(dynamic), address)
    ) == 0


def test_address_norm_receives_gradient_after_zero_output_head_opens() -> None:
    torch.manual_seed(20260808)
    writer = _writer()
    video = torch.randn(1, 4, 8)
    target = torch.randn_like(writer.forward_values(video))
    optimizer = torch.optim.SGD(writer.parameters(), lr=1e-2)

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        predicted, log_scale = writer.forward_values_with_scale(video)
        loss = (predicted - target).square().mean() + log_scale.square().mean()
        loss.backward()
        optimizer.step()

    assert bool(torch.count_nonzero(writer.address_norm.weight.grad))
