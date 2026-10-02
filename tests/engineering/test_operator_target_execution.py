"""Real distributed target replay preserves direct/indirect parameter credit."""
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX
from ember.operator_writer.conditional_read_write import ConditionalTarget
from ember.operator_writer.target_execution import (TargetShardClient, compile_targets,
                                                     serve_target_shards, target_partition)
from ember.writer.replay import sum_writer_gradients


class TinyWriter(nn.Module):
    def __init__(self, calibrated=False):
        super().__init__()
        self.names = tuple(f"target{i}" for i in range(4))
        self.factors = nn.ParameterList([nn.Parameter(torch.randn(shape) * .1)
                                        for _ in self.names for shape in ((128, 4), (3, 128))])
        self.conditional_targets = nn.ModuleList(ConditionalTarget(4, 3) for _ in self.names)
        self.gamma = nn.Linear(35, 35, bias=False) if calibrated else None
        if calibrated:
            for unit in self.conditional_targets:
                unit.ua, unit.ub = nn.Linear(35, 256, bias=False), nn.Linear(35, 256, bias=False)
        self.context_gain = nn.Parameter(torch.tensor(.7))
        with torch.no_grad():
            for unit in self.conditional_targets:
                unit.a_out.weight.normal_(std=.01)
                unit.b_out.weight.normal_(std=.01)

    def public_state(self):
        return dict(zip((name + suffix for name in self.names
                         for suffix in (LORA_A_SUFFIX, LORA_B_SUFFIX)), self.factors, strict=True))


def features(writer, owner):
    generator = torch.Generator().manual_seed(100 + owner)
    # Every target's public factors affect native X/H before the remote cut.
    native = sum((b @ a).square().mean().tanh()
                 for a, b in zip(writer.factors[::2], writer.factors[1::2], strict=True))
    h = torch.randn((3, 50, 1024), generator=generator) + .1 * native
    x = {name: torch.randn((3, 50, 4), generator=generator) + .1 * native for name in writer.names}
    c = h * writer.context_gain
    d = torch.cat((torch.zeros_like(h[:1]), h[1:] - h[:-1]))
    q = writer.gamma(torch.randn((2, 35), generator=generator)) if writer.gamma is not None else None
    return (x, h, c, d, q) if q is not None else (x, h, c, d)


def cotangents(state, owner):
    generator = torch.Generator().manual_seed(200 + owner)
    return tuple(torch.randn(value.shape, generator=generator) * .5 for value in state.values())


def distributed_worker(rank, rendezvous, result, calibrated):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    writer = TinyWriter(calibrated)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=3)
    try:
        if rank == 2:
            serve_target_shards(SimpleNamespace(writer=writer, device=torch.device("cpu")), owners=2, world=3)
        else:
            client = TargetShardClient(writer, owners=2, world=3)
            with torch.no_grad():
                state = client(writer, *features(writer, rank))
                direct = compile_targets(writer, range(4), *features(writer, rank))
                for name in state:
                    torch.testing.assert_close(state[name], direct[name], rtol=2e-5, atol=3e-6)
            state = client(writer, *features(writer, rank))
            # Dictionaries have local-first ordering; match the query's named cotangent.
            ordered = {name: state[name] for name in writer.public_state()}
            torch.autograd.backward(tuple(ordered.values()), cotangents(ordered, rank))
        parameters = tuple(writer.parameters())
        sum_writer_gradients(parameters, world_size=3, bucket_bytes=512 * 1024)
        if rank == 0:
            torch.manual_seed(17)
            reference = TinyWriter(calibrated)
            for owner in range(2):
                state = compile_targets(reference, range(4), *features(reference, owner))
                torch.autograd.backward(tuple(state.values()), cotangents(state, owner))
            for actual, expected in zip(parameters, reference.parameters(), strict=True):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=4e-5, atol=8e-6)
            assert writer.context_gain.grad.abs() > 0
            assert all(parameter.grad.norm() > 0 for parameter in writer.factors)
            Path(result).write_text("complete remote/local/native/public VJP and bucketed SUM passed\n")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("calibrated", (False, True))
def test_actual_distributed_composite_credit_and_bucketed_sum(tmp_path, calibrated):
    result = tmp_path / "result.txt"
    mp.spawn(distributed_worker, args=(str(tmp_path / "rendezvous"), str(result), calibrated), nprocs=3, join=True)
    assert result.read_text().startswith("complete remote/local/native/public VJP")


def test_topology_partition_covers_targets_without_idle_helpers():
    units = [SimpleNamespace(a_x=SimpleNamespace(in_features=1024 + i * 128),
                             b_out=SimpleNamespace(out_features=1024)) for i in range(38)]
    writer = SimpleNamespace(conditional_targets=units)
    for world in range(1, 7):
        owners = min(4, world)
        local, remote = target_partition(writer, owners=owners, world=world)
        groups = [local, *remote.values()]
        assert sorted(i for group in groups for i in group) == list(range(38))
        assert all(groups)
        assert set(remote) == set(range(owners, world))
