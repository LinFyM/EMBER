"""Non-NCCL readiness for outcome-skewed rank-local credit work."""

from __future__ import annotations

import hashlib
import os
import time
from typing import TYPE_CHECKING

import torch

from ember.pi05_source_checkpoint import write_json_atomic
from ember.reward.protocol import RewardProtocolError

if TYPE_CHECKING:
    from ember.rl_writer.runtime import RLWriterRuntime


def _credit_rendezvous_session() -> str:
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "")
    if not run_id:
        raise RewardProtocolError(
            "multi-rank credit rendezvous requires TORCHELASTIC_RUN_ID"
        )
    restart = os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
    return hashlib.sha256(f"{run_id}\0{restart}".encode("utf-8")).hexdigest()[:20]


def rank_local_credit_ready(
    runtime: RLWriterRuntime, *, cycle: int, epoch: int
) -> None:
    if runtime.context.world_size <= 1:
        return
    if runtime.context.device.type == "cuda":
        torch.cuda.synchronize(runtime.context.device)
    session = _credit_rendezvous_session()
    ready = (
        runtime.args.output_dir
        / ".rank-local-credit-ready"
        / session
        / f"cycle-{cycle:08d}-epoch-{epoch:04d}"
    )
    marker = ready / f"rank-{runtime.context.rank:02d}.json"
    write_json_atomic(
        marker,
        {
            "cycle": cycle,
            "epoch": epoch,
            "rank": runtime.context.rank,
            "session": session,
            "world_size": runtime.context.world_size,
        },
    )
    expected = tuple(
        ready / f"rank-{rank:02d}.json" for rank in range(runtime.context.world_size)
    )
    deadline = time.monotonic() + 30 * 60
    while True:
        missing = [path.name for path in expected if not path.is_file()]
        if not missing:
            return
        if time.monotonic() >= deadline:
            raise RewardProtocolError(
                "rank-local credit rendezvous timed out waiting for "
                + ",".join(missing)
            )
        time.sleep(0.05)
