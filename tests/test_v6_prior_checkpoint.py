from __future__ import annotations

import copy
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from ember.expert_manifold.contract import ExpertManifoldError
from ember.expert_manifold.v6_prior import (
    configure_v6_prior_trainability,
    v6_prior_trainable_parameters,
)
from ember.expert_manifold.v6_prior_checkpoint import (
    V6_PRIOR_CHECKPOINT_SCHEMA,
    V6_PRIOR_RNG_SCHEMA,
    V6_PRIOR_TRAINER_SCHEMA,
    compare_v6_prior_checkpoints,
    inspect_v6_prior_checkpoint,
    load_v6_prior_checkpoint,
    save_v6_prior_checkpoint,
)
from ember.expert_manifold.v6_prior_contract import (
    _checkpoint_comparison_evidence_matches,
)
from ember.pi05_source_checkpoint import (
    DistributedContext,
    read_json,
    write_json_atomic,
)


def _writer():
    path = Path(__file__).with_name("test_writer_model.py")
    spec = importlib.util.spec_from_file_location("v6_writer_test_helper", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._model()[0]


def _optimizer(writer):
    optimizer = torch.optim.AdamW(
        v6_prior_trainable_parameters(writer),
        lr=3e-5,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 1.0 if step >= 0 else 0.0,
    )
    return optimizer, scheduler


def _synthetic_formal_checkpoint(
    root: Path,
    *,
    writer_state: dict[str, torch.Tensor],
    trainer: dict,
    trainable_names: list[str],
    macro: int = 3,
) -> Path:
    checkpoint = root / "checkpoints" / f"macro_{macro:08d}"
    checkpoint.mkdir(parents=True)
    save_file(
        {
            name: value.detach().cpu().contiguous()
            for name, value in writer_state.items()
        },
        str(checkpoint / "writer.safetensors"),
    )
    torch.save(trainer, checkpoint / "trainer.pt")
    for rank in range(6):
        generator = torch.Generator(device="cpu").manual_seed(1000 + rank)
        torch.save(
            {
                "schema_version": V6_PRIOR_RNG_SCHEMA,
                "rank": rank,
                "world_size": 6,
                "python": random.Random(2000 + rank).getstate(),
                "numpy": np.random.RandomState(3000 + rank).get_state(),
                "torch_cpu": generator.get_state(),
                "torch_cuda": generator.get_state().clone(),
            },
            checkpoint / f"rng_rank_{rank:03d}.pt",
        )
    cursor = {
        "next_macro": macro,
        "task_visits_per_task": macro,
        "sampler_seed": 11,
        "teacher_video_seed": 13,
        "counterfactual_seed": 17,
        "counterfactual_phase": macro % 3,
        "videos_per_task_visit": 1,
        "action_queries_per_task": 20,
    }
    contract = {
        "run_schema": "ember_pi05_v6_prior_writer_launch_v1",
        "mode": "profile",
        "git_commit": "7778985",
        "config": {
            "path": "configs/pi05_v6_prior_policy_effective_writer_v1.json",
            "schema": "ember_pi05_v6_prior_policy_effective_writer_v1",
            "bytes": 1,
        },
        "source": {"model_path": "/synthetic/source"},
        "initialization": {
            "mode": "historical_v6_macro400_load_only",
            "checkpoint": "/synthetic/writer.safetensors",
            "writer_state_tensor_count": 600,
            "writer_state_value_count": sum(
                value.numel() for value in writer_state.values()
            ),
            "optimizer": "fresh",
            "scheduler": "fresh",
            "rng": "fresh_seed",
        },
        "expert_bank_root": "/synthetic/experts",
        "expert_step": 2000,
        "objective": {"positive_functional_weight": 1.0},
        "ownership": {
            "frozen_parameter_count": 7_060_992,
            "trainable_parameter_count": 3_714_304,
            "frozen_tensor_count": 482,
            "trainable_tensor_count": 41,
            "trainable_tensor_names": trainable_names,
            "source_policy_trainable_parameter_count": 0,
        },
        "world_size": 6,
    }
    names = [
        "writer.safetensors",
        "trainer.pt",
        *(f"rng_rank_{rank:03d}.pt" for rank in range(6)),
    ]
    write_json_atomic(
        checkpoint / "manifest.json",
        {
            "schema_version": V6_PRIOR_CHECKPOINT_SCHEMA,
            "next_macro": macro,
            "metrics_rows": macro,
            "world_size": 6,
            "cursor_contract": cursor,
            "checkpoint_contract": contract,
            "files": {name: (checkpoint / name).stat().st_size for name in names},
            "content_hash_policy": "disabled_by_owner",
        },
    )
    return checkpoint


def _synthetic_checkpoint_pair(tmp_path: Path) -> tuple[Path, Path]:
    writer = _writer()
    configure_v6_prior_trainability(writer)
    optimizer, scheduler = _optimizer(writer)
    for _ in range(3):
        for parameter in v6_prior_trainable_parameters(writer):
            parameter.grad = torch.full_like(parameter, 0.01)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    state = {
        name: value.detach().cpu().clone()
        for name, value in writer.state_dict().items()
    }
    trainable_names = [
        name for name, parameter in writer.named_parameters() if parameter.requires_grad
    ]
    trainer = {
        "schema_version": V6_PRIOR_TRAINER_SCHEMA,
        "next_macro": 3,
        "metrics_rows": 3,
        "world_size": 6,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "amp_scaler": {"enabled": False, "state": {}},
    }
    left = _synthetic_formal_checkpoint(
        tmp_path / "left",
        writer_state=state,
        trainer=trainer,
        trainable_names=trainable_names,
    )
    right_state = {name: value.clone() for name, value in state.items()}
    right_state[trainable_names[0]].view(-1)[0].add_(1e-7)
    right_trainer = copy.deepcopy(trainer)
    first_id = right_trainer["optimizer"]["param_groups"][0]["params"][0]
    right_trainer["optimizer"]["state"][first_id]["exp_avg"].view(-1)[0].add_(1e-8)
    right = _synthetic_formal_checkpoint(
        tmp_path / "right",
        writer_state=right_state,
        trainer=right_trainer,
        trainable_names=trainable_names,
    )
    return left, right


def _refresh_declared_size(checkpoint: Path, name: str) -> None:
    manifest = read_json(checkpoint / "manifest.json")
    manifest["files"][name] = (checkpoint / name).stat().st_size
    write_json_atomic(checkpoint / "manifest.json", manifest)


def test_v6_prior_checkpoint_restores_writer_optimizer_scheduler_and_cursor(
    tmp_path: Path,
) -> None:
    writer = _writer()
    configure_v6_prior_trainability(writer)
    optimizer, scheduler = _optimizer(writer)
    for parameter in v6_prior_trainable_parameters(writer):
        parameter.grad = torch.full_like(parameter, 0.01)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    expected = {
        name: value.detach().clone() for name, value in writer.state_dict().items()
    }
    cursor = {
        "next_macro": 1,
        "sampler_seed": 11,
        "teacher_video_seed": 13,
        "counterfactual_seed": 17,
        "task_visits_per_task": 1,
    }
    contract = {
        "schema": "v1",
        "warm_start": "historical_v6_macro400_load_only",
        "frozen_blocks": 4,
        "trainable_blocks": 2,
    }
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    checkpoint = save_v6_prior_checkpoint(
        output_dir=tmp_path,
        macro=1,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        context=context,
        metrics_rows=1,
        cursor_contract=cursor,
        checkpoint_contract=contract,
    )
    for parameter in writer.parameters():
        parameter.data.zero_()
    loaded_macro, rows = load_v6_prior_checkpoint(
        checkpoint=checkpoint,
        writer=writer,
        optimizer=optimizer,
        scheduler=scheduler,
        context=context,
        expected_cursor_contract=cursor,
        expected_checkpoint_contract=contract,
    )
    assert (loaded_macro, rows) == (1, 1)
    assert all(
        torch.equal(expected[name], value)
        for name, value in writer.state_dict().items()
    )
    assert scheduler.last_epoch == 1

    changed = {**cursor, "teacher_video_seed": 19}
    with pytest.raises(ExpertManifoldError, match="manifest changed"):
        load_v6_prior_checkpoint(
            checkpoint=checkpoint,
            writer=writer,
            optimizer=optimizer,
            scheduler=scheduler,
            context=context,
            expected_cursor_contract=changed,
            expected_checkpoint_contract=contract,
        )


def test_v6_prior_checkpoint_inspection_and_tolerant_semantic_comparison(
    tmp_path: Path,
) -> None:
    left, right = _synthetic_checkpoint_pair(tmp_path)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state()

    inspected = inspect_v6_prior_checkpoint(left)
    compared = compare_v6_prior_checkpoints(left, right)

    assert inspected["next_macro"] == 3
    assert inspected["world_size"] == 6
    assert inspected["writer"]["state_tensor_count"] == 600
    assert inspected["writer"]["trainable_tensor_count"] == 41
    assert inspected["writer"]["frozen_state_tensor_count"] == 483
    assert inspected["writer"]["frozen_parameter_tensor_count"] == 482
    assert inspected["writer"]["template_tensor_count"] == 76
    assert inspected["rng"]["rank_count"] == 6
    assert compared["cursor"]["semantic_equal"] is True
    assert compared["rng"]["semantic_equal"] is True
    assert compared["writer"]["frozen_exact"] is True
    assert compared["writer"]["state_tensor_count"] == 600
    assert compared["writer"]["tensor_count"] == 41
    assert 0.0 < compared["writer"]["max_abs"] <= 7.5e-6
    assert compared["trainer"]["optimizer"]["max_abs"] == pytest.approx(1e-8, abs=1e-10)
    assert _checkpoint_comparison_evidence_matches(
        {
            "macro": 3,
            "cursor_semantic_equal": compared["cursor"]["semantic_equal"],
            "checkpoint_contract_semantic_equal": compared["checkpoint_contract"][
                "semantic_equal"
            ],
            "rng_rank_count": compared["rng"]["rank_count"],
            "rng_semantic_equal": compared["rng"]["semantic_equal"],
            "scheduler_semantic_equal": compared["trainer"]["scheduler_semantic_equal"],
            "amp_semantic_equal": compared["trainer"]["amp_semantic_equal"],
            "optimizer": compared["trainer"]["optimizer"],
            "writer": compared["writer"],
        }
    )
    json.dumps(compared)
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_before[0]
    assert np.array_equal(numpy_after[1], numpy_before[1])
    assert numpy_after[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_v6_prior_checkpoint_inspection_and_comparison_fail_closed(
    tmp_path: Path,
) -> None:
    left, right = _synthetic_checkpoint_pair(tmp_path)

    manifest = read_json(left / "manifest.json")
    original_phase = manifest["cursor_contract"]["counterfactual_phase"]
    manifest["cursor_contract"]["counterfactual_phase"] = 2
    write_json_atomic(left / "manifest.json", manifest)
    with pytest.raises(ExpertManifoldError, match="cursor contract"):
        inspect_v6_prior_checkpoint(left)
    manifest["cursor_contract"]["counterfactual_phase"] = original_phase
    write_json_atomic(left / "manifest.json", manifest)

    trainer_path = left / "trainer.pt"
    trainer_bytes = trainer_path.stat().st_size
    with trainer_path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ExpertManifoldError, match="declared file trainer.pt"):
        inspect_v6_prior_checkpoint(left)
    with trainer_path.open("r+b") as handle:
        handle.truncate(trainer_bytes)

    rng_path = right / "rng_rank_004.pt"
    rng = torch.load(rng_path, map_location="cpu", weights_only=False)
    rng["torch_cuda"][0] ^= 1
    torch.save(rng, rng_path)
    _refresh_declared_size(right, "rng_rank_004.pt")
    with pytest.raises(ExpertManifoldError, match="rank 4 RNG"):
        compare_v6_prior_checkpoints(left, right)

    left_rng = torch.load(
        left / "rng_rank_004.pt", map_location="cpu", weights_only=False
    )
    torch.save(left_rng, rng_path)
    _refresh_declared_size(right, "rng_rank_004.pt")
    trainer_path = right / "trainer.pt"
    trainer = torch.load(trainer_path, map_location="cpu", weights_only=False)
    for values in trainer["optimizer"]["state"].values():
        values["exp_avg_sq"].zero_()
    torch.save(trainer, trainer_path)
    _refresh_declared_size(right, "trainer.pt")
    with pytest.raises(
        ExpertManifoldError, match="compared optimizer exp_avg_sq tolerance"
    ):
        compare_v6_prior_checkpoints(left, right)

    left_trainer = torch.load(
        left / "trainer.pt", map_location="cpu", weights_only=False
    )
    torch.save(left_trainer, trainer_path)
    _refresh_declared_size(right, "trainer.pt")
    right_state = load_file(str(right / "writer.safetensors"), device="cpu")
    frozen_name = next(name for name in right_state if name.startswith("procedure."))
    right_state[frozen_name].view(-1)[0].add_(0.01)
    save_file(
        {name: value.contiguous() for name, value in right_state.items()},
        str(right / "writer.safetensors"),
    )
    _refresh_declared_size(right, "writer.safetensors")
    with pytest.raises(ExpertManifoldError, match="frozen Writer tensor"):
        compare_v6_prior_checkpoints(left, right)

    left_state = load_file(str(left / "writer.safetensors"), device="cpu")
    trainable_name = next(name for name in left_state if name.startswith("compiler."))
    left_state[trainable_name].view(-1)[0] = torch.nan
    save_file(
        {name: value.contiguous() for name, value in left_state.items()},
        str(left / "writer.safetensors"),
    )
    _refresh_declared_size(left, "writer.safetensors")
    with pytest.raises(ExpertManifoldError, match="Writer tensor"):
        inspect_v6_prior_checkpoint(left)
