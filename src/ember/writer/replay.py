"""Global task-weighted Writer gradient reduction."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.distributed as dist


def sum_writer_gradients(
    parameters: Sequence[torch.nn.Parameter], *, world_size: int,
) -> None:
    """SUM already globally weighted task gradients, including idle ranks.

    A task's weight is determined before device assignment. No extra world-size
    division is applied. All ranks must pass parameters in the same order.
    The caller checks the resulting global norm for finiteness after every rank
    has completed the collectives; local early raises can deadlock peer ranks.
    """

    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if world_size > 1:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
