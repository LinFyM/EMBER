from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_distributed import (  # noqa: E402
    close_distributed_context,
    initialize_distributed_context,
    native_global_flow_inputs,
    topology_for_world_size,
    tensor_state_sha256,
    wrap_distributed_model,
)


class _NativeSampler(torch.nn.Module):
    def sample_noise(self, shape, device):
        return torch.normal(0.0, 1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, batch_size, device):
        beta = torch.distributions.Beta(1.5, 1.0).sample((batch_size,)).to(device)
        return beta * 0.999 + 0.001


class _FlowPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(()))
        self.model = _NativeSampler()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import tomllib

    spec = tomllib.loads(args.config.read_text(encoding="utf-8"))
    context = initialize_distributed_context(spec, backend="gloo")
    topology = topology_for_world_size(spec, context.world_size)
    try:
        torch.manual_seed(101)
        model = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.2, -0.1, 0.4], [-0.3, 0.5, 0.7]]))
        ddp = wrap_distributed_model(model, context)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.05)
        global_inputs = torch.arange(64 * 3, dtype=torch.float32).reshape(64, 3) / 100.0
        global_targets = torch.arange(64 * 2, dtype=torch.float32).reshape(64, 2) / 50.0
        local = topology.per_rank_micro_batch_size
        begin = context.rank * local
        end = begin + local
        loss = torch.nn.functional.mse_loss(ddp(global_inputs[begin:end]), global_targets[begin:end])
        loss.backward()
        optimizer.step()

        torch.manual_seed(303)
        flow_policy = _FlowPolicy()
        local_noise, local_time, flow_input_sha256 = native_global_flow_inputs(
            flow_policy,
            topology,
            context,
            action_shape=(3, 2),
            device=torch.device("cpu"),
        )
        gathered_noise = [torch.empty_like(local_noise) for _ in range(context.world_size)]
        gathered_time = [torch.empty_like(local_time) for _ in range(context.world_size)]
        torch.distributed.all_gather(gathered_noise, local_noise)
        torch.distributed.all_gather(gathered_time, local_time)
        if context.is_primary:
            torch.manual_seed(303)
            reference_policy = _FlowPolicy()
            reference_noise = reference_policy.model.sample_noise((64, 3, 2), torch.device("cpu"))
            reference_time = reference_policy.model.sample_time(64, torch.device("cpu"))
            reference_sha256 = tensor_state_sha256(
                {"noise": reference_noise, "time": reference_time}
            )
            output = {
                "world_size": context.world_size,
                "weight": model.weight.detach().tolist(),
                "flow_noise_exact": torch.equal(torch.cat(gathered_noise), reference_noise),
                "flow_time_exact": torch.equal(torch.cat(gathered_time), reference_time),
                "flow_input_sha256_exact": flow_input_sha256 == reference_sha256,
                "rank": int(os.environ["RANK"]),
            }
            args.output.write_text(json.dumps(output), encoding="utf-8")
    finally:
        close_distributed_context(context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
