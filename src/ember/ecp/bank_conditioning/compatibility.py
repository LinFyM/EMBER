"""Projected Program-query/native-key scalar compatibility."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from ember.ecp.bank_conditioning.operator import BankConditioningError


class ProjectedBilinearCompatibility(torch.nn.Module):
    """Expose the shared Q/K chart used by current-bank functional polar."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = int(width)
        if self.width <= 0:
            raise BankConditioningError("bilinear compatibility width is invalid")
        self.query_projection = torch.nn.Linear(self.width, self.width, bias=False)
        self.key_projection = torch.nn.Linear(self.width, self.width, bias=False)

    def project_query(self, query: torch.Tensor) -> torch.Tensor:
        if query.shape[-1] != self.width:
            raise BankConditioningError("bilinear query width changed")
        return self.query_projection(query)

    def project_key(self, key: torch.Tensor) -> torch.Tensor:
        if key.shape[-1] != self.width:
            raise BankConditioningError("bilinear key width changed")
        return functional.normalize(self.key_projection(key), dim=-1)

    def score_projected(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        """Score post-Wq polar coefficients against unit post-Wk keys."""
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
        query_view = query_by_event.reshape(query_shape)
        key_view = key.reshape(key_shape)
        score = torch.tanh((query_view * key_view).sum(-1))
        order = (1, 0, 2, *range(3, score.ndim))
        return score.permute(order)

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return self.score_projected(
            self.project_query(query), self.project_key(key)
        )
