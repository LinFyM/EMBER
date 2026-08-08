from argparse import Namespace
from pathlib import Path

import pytest
import torch

from ember.expert_manifold.contract import (
    ExpertManifoldError,
    load_expert_manifold_config,
)
from ember.expert_manifold.writer_training import (
    _contract,
    _mean_writer_gradients,
    _validate_collective_environment,
)
from ember.pi05_source_checkpoint import DistributedContext


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pi05_video_expert_manifold_v1.json"


def test_writer_contract_seals_explicit_flat_gradient_reduction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "cache_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "ember.expert_manifold.writer_training.git_state",
        lambda _: {"branch": "branch", "commit": "commit"},
    )
    monkeypatch.setattr(
        "ember.expert_manifold.writer_training.visible_physical_cuda_index",
        lambda _: 4,
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA A40")
    contract = _contract(
        args=Namespace(mode="profile", config=CONFIG, feature_cache_root=tmp_path),
        config=load_expert_manifold_config(CONFIG),
        context=DistributedContext(0, 0, 1, torch.device("cuda", 0), 1, (48, 49)),
        source={},
        expert={},
        cache={
            "schema_version": "cache",
            "training_commit": "cache-commit",
            "task_count": 24,
            "demo_count": 50,
            "source": {},
        },
        scheduler_total=800,
        microbatch=1,
        checkpoints=(1, 3),
    )
    runtime = contract["runtime"]
    assert runtime["distributed_model_wrapper"] == "none"
    assert runtime["gradient_reduction"] == (
        "single_flat_parameter_ordered_allreduce_mean_after_local_task_mean"
    )


def test_writer_collective_environment_is_fixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = DistributedContext(0, 0, 6, torch.device("cpu"))
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
    monkeypatch.setenv("NCCL_ALGO", "Ring")
    monkeypatch.setenv("NCCL_PROTO", "Simple")
    _validate_collective_environment(context)
    monkeypatch.setenv("NCCL_ALGO", "Tree")
    with pytest.raises(ExpertManifoldError, match="collective environment"):
        _validate_collective_environment(context)


def test_writer_gradient_mean_uses_one_ordered_flat_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    second = torch.nn.Parameter(torch.tensor([0.0]))
    model = torch.nn.ParameterList((first, second))
    first.grad = torch.tensor([1.0, 2.0])
    second.grad = torch.tensor([3.0])

    def fake_all_reduce(value: torch.Tensor, *, op: object) -> None:
        assert op == torch.distributed.ReduceOp.SUM
        assert torch.equal(value, torch.tensor([1.0, 2.0, 3.0]))
        value.mul_(2).add_(torch.tensor([2.0, 4.0, 6.0]))

    monkeypatch.setattr(
        "ember.expert_manifold.writer_training.dist.all_reduce", fake_all_reduce
    )
    _mean_writer_gradients(
        model, DistributedContext(0, 0, 2, torch.device("cpu"))
    )
    assert torch.equal(first.grad, torch.tensor([2.0, 4.0]))
    assert torch.equal(second.grad, torch.tensor([6.0]))
