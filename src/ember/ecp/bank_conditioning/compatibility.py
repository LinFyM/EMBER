"""Primary bounded Program-query/native-key scalar compatibility."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from ember.ecp.bank_conditioning.operator import BankConditioningError


class NormalizedBilinearCompatibility(torch.nn.Module):
    """Content-compute one bounded score with no residual bypass.

    Query and key projections receive the same primary gradient path.  The
    bounded temperature avoids recreating the previous tiny nonlinear
    residual beside a dominant dot-product route.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = int(width)
        if self.width <= 0:
            raise BankConditioningError("bilinear compatibility width is invalid")
        self.query_projection = torch.nn.Linear(self.width, self.width, bias=False)
        self.key_projection = torch.nn.Linear(self.width, self.width, bias=False)
        self.logit_scale = torch.nn.Parameter(torch.zeros(()))

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Score `[rank,event,branch,width]` against event-indexed candidates."""

        if (
            query.ndim != 4
            or key.ndim < 3
            or query.shape[1] != key.shape[0]
            or query.shape[-1] != self.width
            or key.shape[-1] != self.width
        ):
            raise BankConditioningError("bilinear compatibility axes changed")
        events, ranks, branches = key.shape[0], query.shape[0], query.shape[2]
        candidate_shape = key.shape[1:-1]
        query_by_event = query.permute(1, 0, 2, 3)
        query_shape = (
            events,
            ranks,
            branches,
            *((1,) * len(candidate_shape)),
            self.width,
        )
        key_shape = (events, 1, 1, *candidate_shape, self.width)
        query_view = functional.normalize(
            self.query_projection(query_by_event), dim=-1
        ).reshape(query_shape)
        key_view = functional.normalize(
            self.key_projection(key), dim=-1
        ).reshape(key_shape)
        # exp(tanh(. ) * log(4)) gives a smooth, positive [1/4, 4] scale.
        scale = torch.exp(
            torch.tanh(self.logit_scale) * self.logit_scale.new_tensor(4.0).log()
        )
        score = torch.tanh(scale * (query_view * key_view).sum(-1))
        order = (1, 0, 2, *range(3, score.ndim))
        return score.permute(order)
