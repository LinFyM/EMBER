from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ember.gate_zero_checkpoint import (  # noqa: E402
    load_source_base_training_state_without_rng,
    restore_source_base_checkpoint_rng,
    save_source_base_checkpoint,
    validate_source_base_checkpoint,
)
from ember.gate_zero_distributed import (  # noqa: E402
    broadcast_primary_error,
    close_distributed_context,
    gather_rank_objects,
    gather_rank_rng_states,
    initialize_distributed_context,
    topology_for_world_size,
    wrap_distributed_model,
)


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor([[0.2, -0.1, 0.4], [-0.3, 0.5, 0.7]])
            )
            self.linear.bias.copy_(torch.tensor([0.1, -0.2]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)

    def save_pretrained(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.json").write_text(
            json.dumps({"type": "tiny", "use_peft": False}), encoding="utf-8"
        )
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in self.state_dict().items()
            },
            directory / "model.safetensors",
        )


class _Processor:
    def __init__(self, name: str) -> None:
        self.name = name

    def save_pretrained(self, directory: Path) -> None:
        (directory / f"{self.name}.json").write_text("{}", encoding="utf-8")


def _metadata(world_size: int) -> dict:
    return {
        "checkpoint_role": "source_base_training_recovery",
        "topology": {
            "world_size": world_size,
            "micro_batch_size": 64 // world_size,
            "per_rank_micro_batch_size": 64 // world_size,
            "gradient_accumulation_steps": 1,
            "num_workers": 4 // world_size,
            "data_workers_per_rank": 4 // world_size,
            "global_effective_batch_size": 64,
            "total_num_workers": 4,
            "global_slot_algorithm": "absolute_optimizer_step_accumulation_rank_local_slot_v1",
            "flow_input_authority": "rank0_global_native_sample_then_contiguous_scatter_v1",
            "checkpoint_writer_rank": 0,
        },
        "authorities": {
            "base_revision": "c" * 40,
            "base_weight_sha256": "d" * 64,
            "normalization_sha256": "e" * 64,
            "gate_zero_contract_sha256": "f" * 64,
            "phase0_contract_sha256": "a" * 64,
            "canonical_manifest_sha256": "b" * 64,
            "topology_contract_sha256": "9" * 64,
        },
        "sampler": {
            "algorithm": "absolute_optimizer_step_accumulation_rank_local_slot_v1",
            "seed": 17,
            "next_optimizer_step": 1,
        },
    }


def _new_runtime(context):
    random.seed(700 + context.rank)
    np.random.seed(800 + context.rank)
    torch.manual_seed(900 + context.rank)
    policy = _Policy()
    wrapped = wrap_distributed_model(policy, context)
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))
    return policy, wrapped, optimizer, scheduler


def _step(wrapped, optimizer, scheduler, context, topology):
    inputs = torch.arange(64 * 3, dtype=torch.float32).reshape(64, 3) / 100.0
    targets = torch.arange(64 * 2, dtype=torch.float32).reshape(64, 2) / 50.0
    local = topology.per_rank_micro_batch_size
    begin = context.rank * local
    end = begin + local
    stochastic_targets = targets[begin:end] + torch.rand(local, 2) * 0.01
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(wrapped(inputs[begin:end]), stochastic_targets).backward()
    optimizer.step()
    scheduler.step()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import tomllib

    spec = tomllib.loads(args.config.read_text(encoding="utf-8"))
    context = initialize_distributed_context(spec, backend="gloo")
    topology = topology_for_world_size(spec, context.world_size)
    try:
        reference_policy, reference_wrapped, reference_optimizer, reference_scheduler = _new_runtime(context)
        _step(reference_wrapped, reference_optimizer, reference_scheduler, context, topology)
        _step(reference_wrapped, reference_optimizer, reference_scheduler, context, topology)
        reference_state = {
            name: value.detach().clone() for name, value in reference_policy.state_dict().items()
        }
        reference_lr = reference_optimizer.param_groups[0]["lr"]
        reference_next_rng = torch.rand(4)
        del reference_wrapped, reference_policy, reference_optimizer, reference_scheduler
        gc.collect()

        interrupted_policy, interrupted_wrapped, interrupted_optimizer, interrupted_scheduler = _new_runtime(context)
        _step(interrupted_wrapped, interrupted_optimizer, interrupted_scheduler, context, topology)
        rank_states = gather_rank_rng_states(context)
        primary_error = None
        if context.is_primary:
            try:
                save_source_base_checkpoint(
                    args.checkpoint,
                    step=1,
                    policy=interrupted_policy,
                    optimizer=interrupted_optimizer,
                    scheduler=interrupted_scheduler,
                    preprocessor=_Processor("policy_preprocessor"),
                    postprocessor=_Processor("policy_postprocessor"),
                    metadata=_metadata(context.world_size),
                    rank_rng_states=rank_states,
                )
            except BaseException as error:
                primary_error = error
        broadcast_primary_error(context, primary_error)
        del interrupted_wrapped, interrupted_policy, interrupted_optimizer, interrupted_scheduler
        gc.collect()

        random.seed(1)
        np.random.seed(2)
        torch.manual_seed(3)
        resumed_policy = _Policy()
        resumed_optimizer = torch.optim.AdamW(resumed_policy.parameters(), lr=1e-3)
        resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(
            resumed_optimizer, lambda step: 1.0 / (step + 1)
        )
        _, resumed_optimizer, resumed_scheduler = load_source_base_training_state_without_rng(
            args.checkpoint,
            policy=resumed_policy,
            optimizer=resumed_optimizer,
            scheduler=resumed_scheduler,
            expected=_metadata(context.world_size),
        )
        resumed_wrapped = wrap_distributed_model(resumed_policy, context)
        restore_source_base_checkpoint_rng(
            args.checkpoint, rank=context.rank, world_size=context.world_size
        )
        _step(resumed_wrapped, resumed_optimizer, resumed_scheduler, context, topology)
        local_result = {
            "model_exact": all(
                torch.equal(resumed_policy.state_dict()[name], value)
                for name, value in reference_state.items()
            ),
            "lr_exact": resumed_optimizer.param_groups[0]["lr"] == reference_lr,
            "rng_exact": torch.equal(torch.rand(4), reference_next_rng),
        }
        gathered = gather_rank_objects(local_result, context)
        if context.is_primary:
            manifest = validate_source_base_checkpoint(args.checkpoint)
            args.output.write_text(
                json.dumps(
                    {
                        "world_size": context.world_size,
                        "all_model_exact": all(item["model_exact"] for item in gathered),
                        "all_lr_exact": all(item["lr_exact"] for item in gathered),
                        "all_rng_exact": all(item["rng_exact"] for item in gathered),
                        "checkpoint_schema_version": manifest["schema_version"],
                        "checkpoint_world_size": manifest["topology"]["world_size"],
                        "distributed_rng_files": manifest["distributed_rng"]["files"],
                    }
                ),
                encoding="utf-8",
            )
    finally:
        close_distributed_context(context)
        if context.is_primary and args.checkpoint.parent.exists():
            shutil.rmtree(args.checkpoint.parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
