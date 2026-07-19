from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from ember.writer.core import (
    CompleteLoRAWriter,
    WriterColdStartError,
    build_lora_tensor_specs,
    physical_lora_delta_l2,
    load_writer_contract,
    load_writer_checkpoint,
    save_writer_checkpoint,
)
from ember.writer.data import (
    WriterQueryDataset,
    WriterSpecAuthority,
    WriterTaskBatchSampler,
    read_action_hidden_spec_frames,
)
from ember.writer.train import prepare_writer_images, repository_root


ROOT = Path(__file__).resolve().parents[1]


def test_writer_entrypoint_resolves_repository_root() -> None:
    assert repository_root() == ROOT


def test_writer_image_preparation_is_owned_by_policy_not_inner_flow_model() -> None:
    class Owner:
        def prepare_images(self, batch):
            return [batch["observation.images.camera1"] + 1], [torch.ones(1)]

    images = torch.zeros(1, 3, 4, 4)
    torch.testing.assert_close(prepare_writer_images(Owner(), images), images + 1)


def _tiny_lora_state() -> dict[str, torch.Tensor]:
    return {
        "base.block.q_proj.lora_A.default.weight": torch.randn(2, 3),
        "base.block.q_proj.lora_B.default.weight": torch.zeros(4, 2),
        "base.block.v_proj.lora_A.default.weight": torch.randn(2, 3),
        "base.block.v_proj.lora_B.default.weight": torch.zeros(5, 2),
    }


def _tiny_writer(state: dict[str, torch.Tensor] | None = None) -> CompleteLoRAWriter:
    state = state or _tiny_lora_state()
    return CompleteLoRAWriter(
        build_lora_tensor_specs(state),
        template_state=state,
        feature_dim=7,
        hidden_dim=12,
        module_embedding_dim=4,
        factor_embedding_dim=3,
        rank_embedding_dim=3,
        decoder_hidden_dim=10,
    )


def test_active_writer_contract_is_hash_bound_and_has_no_removed_mechanism() -> None:
    spec = load_writer_contract(
        ROOT / "configs/writer_cold_start.toml",
        phase0_path=ROOT / "configs/phase0.toml",
        split_path=ROOT / "configs/libero90_split_reseal.json",
        gate_zero_path=ROOT / "configs/gate_zero_oracle_pilot.toml",
        mature_lora_path=ROOT / "configs/gate_zero_mature_lora_positive_control.toml",
    )
    assert spec["train"]["world_size"] == 8
    assert spec["train"]["global_batch_size"] == 2048
    assert spec["train"]["per_rank_micro_batch_size"] == 256
    assert spec["validation"]["task_ids"] == [11, 21, 51, 70, 86]
    assert spec["writer"]["bank_geometry_or_shared_subspace"] is False
    assert spec["authority"]["test_held_numeric_access"] is False


def test_writer_physical_norm_recovery_contract_is_loadable_and_bounded() -> None:
    spec = load_writer_contract(
        ROOT / "configs/writer_cold_start_physical_norm_recovery.toml",
        phase0_path=ROOT / "configs/phase0.toml",
        split_path=ROOT / "configs/libero90_split_reseal.json",
        gate_zero_path=ROOT / "configs/gate_zero_oracle_pilot.toml",
        mature_lora_path=ROOT / "configs/gate_zero_mature_lora_positive_control.toml",
    )
    assert spec["train"]["physical_delta_l2_soft_cap"] == 2.0
    assert spec["train"]["physical_delta_excess_coefficient"] == 0.01
    assert spec["recovery"]["maximum_mechanism_variants"] == 1


def test_writer_emits_every_factor_and_starts_at_physical_zero() -> None:
    state = _tiny_lora_state()
    writer = _tiny_writer(state)
    generated = writer(torch.zeros(1, 7))
    assert set(generated) == set(state)
    assert {key: value.shape for key, value in generated.items()} == {
        key: (1, *value.shape) for key, value in state.items()
    }
    for key, value in generated.items():
        if ".lora_A." in key:
            torch.testing.assert_close(value[0], state[key])
        else:
            assert torch.count_nonzero(value) == 0
    for prefix in ("base.block.q_proj", "base.block.v_proj"):
        a = generated[f"{prefix}.lora_A.default.weight"][0]
        b = generated[f"{prefix}.lora_B.default.weight"][0]
        assert torch.count_nonzero(b @ a) == 0


def test_functional_loss_reaches_writer_without_training_base() -> None:
    writer = _tiny_writer()
    base_weight = torch.nn.Parameter(torch.randn(4, 3), requires_grad=False)
    generated = writer(torch.randn(1, 7))
    a = generated["base.block.q_proj.lora_A.default.weight"][0]
    b = generated["base.block.q_proj.lora_B.default.weight"][0]
    x = torch.randn(5, 3)
    target = torch.randn(5, 4)
    loss = torch.nn.functional.mse_loss(x @ (base_weight + 0.5 * (b @ a)).T, target)
    loss.backward()
    assert base_weight.grad is None
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in writer.parameters()
    )


def test_physical_lora_delta_norm_matches_materialized_update_and_has_gradient() -> None:
    a = torch.randn(2, 3, requires_grad=True)
    b = torch.randn(4, 2, requires_grad=True)
    state = {
        "base.block.q_proj.lora_A.default.weight": a,
        "base.block.q_proj.lora_B.default.weight": b,
    }
    norm = physical_lora_delta_l2(state, alpha=8, rank=2)
    expected = torch.linalg.vector_norm(4 * (b @ a))
    torch.testing.assert_close(norm, expected)
    norm.backward()
    assert a.grad is not None and torch.count_nonzero(a.grad)
    assert b.grad is not None and torch.count_nonzero(b.grad)


def _write_hdf5(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for demo_index in range(3):
            demo = data.create_group(f"demo_{demo_index}")
            demo.attrs["num_samples"] = 5
            obs = demo.create_group("obs")
            frames = np.arange(5 * 4 * 4 * 3, dtype=np.uint8).reshape(5, 4, 4, 3)
            obs.create_dataset("agentview_rgb", data=frames + demo_index)
            obs.create_dataset("eye_in_hand_rgb", data=frames)
            obs.create_dataset("ee_states", data=np.zeros((5, 6), dtype=np.float64))
            obs.create_dataset("gripper_states", data=np.zeros((5, 2), dtype=np.float64))
            demo.create_dataset("actions", data=np.zeros((5, 7), dtype=np.float64))


def test_action_hidden_reader_never_requires_action_or_proprio(tmp_path: Path) -> None:
    path = tmp_path / "spec.hdf5"
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        demo = data.create_group("demo_0")
        demo.attrs["num_samples"] = 3
        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", data=np.zeros((3, 4, 4, 3), dtype=np.uint8))
    authority = WriterSpecAuthority(3, "task", path, path.stat().st_size, None)
    frames = read_action_hidden_spec_frames(authority, [0], ["first", "middle", "last"])
    assert frames.shape == (3, 3, 4, 4)


def test_rank_task_sampler_is_resume_exact_and_single_task_per_batch(tmp_path: Path) -> None:
    path = tmp_path / "query.hdf5"
    _write_hdf5(path)
    authority = WriterSpecAuthority(3, "task", path, path.stat().st_size, None)
    dataset = WriterQueryDataset([authority], demo_indices=[1, 2], action_chunk_size=4)
    first = WriterTaskBatchSampler(
        dataset,
        task_ids=[3],
        per_rank_batch_size=4,
        start_step=0,
        stop_step=4,
        rank=0,
        world_size=1,
        seed=17,
    )
    resumed = WriterTaskBatchSampler(
        dataset,
        task_ids=[3],
        per_rank_batch_size=4,
        start_step=2,
        stop_step=4,
        rank=0,
        world_size=1,
        seed=17,
    )
    batches = list(first)
    assert batches[2:] == list(resumed)
    assert all(len(batch) == len(set(batch)) == 4 for batch in batches)
    assert all({dataset.frame_index[index][0] for index in batch} == {3} for batch in batches)
    dataset.close()


def test_atomic_writer_checkpoint_restores_model_optimizer_scheduler(tmp_path: Path) -> None:
    writer = _tiny_writer()
    optimizer = torch.optim.AdamW(writer.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 - 0.1 * step)
    loss = sum(value.square().mean() for value in writer(torch.randn(1, 7)).values())
    loss.backward()
    optimizer.step()
    scheduler.step()
    checkpoint = tmp_path / "checkpoints" / "000001"
    save_writer_checkpoint(
        checkpoint,
        step=1,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        rank_rng_states=None,
        metadata={"world_size": 1, "sampler": {"completed_step": 1}, "authority": "test"},
    )
    expected = {key: value.detach().clone() for key, value in writer.state_dict().items()}
    restored = _tiny_writer()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lambda step: 1.0 - 0.1 * step
    )
    step, chain = load_writer_checkpoint(
        checkpoint,
        writer=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        rank=0,
        world_size=1,
    )
    assert step == 1
    assert chain == ""
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    assert restored_optimizer.state_dict()["state"]
    assert restored_scheduler.last_epoch == scheduler.last_epoch
    assert json.loads((checkpoint / "scaler.json").read_text())["enabled"] is False


def test_lora_specs_reject_incomplete_pairs() -> None:
    state = _tiny_lora_state()
    del state["base.block.v_proj.lora_B.default.weight"]
    with pytest.raises(WriterColdStartError, match="pair"):
        build_lora_tensor_specs(state)
