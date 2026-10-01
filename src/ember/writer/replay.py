"""Global task-weighted Writer gradient reduction."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.distributed as dist


def sum_writer_gradients(
    parameters: Sequence[torch.nn.Parameter], *, world_size: int,
    bucket_bytes: int | None = None,
) -> None:
    """SUM already globally weighted task gradients, including idle ranks.

    A task's weight is determined before device assignment. No extra world-size
    division is applied. All ranks must pass parameters in the same order.
    The caller checks the resulting global norm for finiteness after every rank
    has completed the collectives; local early raises can deadlock peer ranks.
    """

    def reduce_bucket(bucket):
        flat = torch.cat([p.grad.reshape(-1) for p in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for parameter in bucket:
            parameter.grad.copy_(flat[offset:offset + parameter.numel()].view_as(parameter))
            offset += parameter.numel()

    bucket, size = [], 0
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if world_size > 1:
            if bucket_bytes is None:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            else:
                if bucket and (size + parameter.grad.nbytes > bucket_bytes
                               or parameter.grad.dtype != bucket[0].grad.dtype):
                    reduce_bucket(bucket)
                    bucket, size = [], 0
                bucket.append(parameter)
                size += parameter.grad.nbytes
    if bucket:
        reduce_bucket(bucket)
