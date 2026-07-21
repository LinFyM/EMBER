"""Official OpenPI tokenization and source-only normalization for pi0.5."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class Pi05LiberoProcessor:
    """Exact OpenPI pi0.5 prompt format with source-only quantile normalization."""

    def __init__(
        self,
        stats: dict[str, Any],
        tokenizer_path: Path,
        max_length: int,
        device: str,
    ) -> None:
        import sentencepiece
        import torch

        self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))
        self._max_length = max_length
        self._device = device
        self._state_q01 = torch.tensor(stats["observation.state"]["q01"], device=device)
        self._state_q99 = torch.tensor(stats["observation.state"]["q99"], device=device)
        self._action_q01 = torch.tensor(stats["action"]["q01"], device=device)
        self._action_q99 = torch.tensor(stats["action"]["q99"], device=device)

    @staticmethod
    def _quantile_transform(value: Any, q01: Any, q99: Any, inverse: bool) -> Any:
        import torch

        denominator = q99 - q01
        denominator = torch.where(
            denominator == 0,
            torch.tensor(1e-8, device=value.device, dtype=value.dtype),
            denominator,
        )
        if inverse:
            return (value + 1.0) * denominator / 2.0 + q01
        return 2.0 * (value - q01) / denominator - 1.0

    def __call__(self, value: dict[str, Any]) -> dict[str, Any]:
        import torch
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )

        state = value["observation.state"].to(self._device)
        normalized = self._quantile_transform(
            state, self._state_q01, self._state_q99, inverse=False
        )
        discretized = np.digitize(
            normalized.detach().cpu().numpy(),
            bins=np.linspace(-1, 1, 256 + 1)[:-1],
        ) - 1
        cleaned = value["task"].strip().replace("_", " ").replace("\n", " ")
        state_text = " ".join(map(str, discretized))
        prompt = f"Task: {cleaned}, State: {state_text};\nAction: "
        tokens = self._tokenizer.encode(prompt, add_bos=True)
        tokens = tokens[: self._max_length]
        mask = [True] * len(tokens)
        padding = self._max_length - len(tokens)
        tokens.extend([0] * padding)
        mask.extend([False] * padding)
        result = {
            key: tensor.unsqueeze(0).to(self._device)
            for key, tensor in value.items()
            if isinstance(tensor, torch.Tensor) and key.startswith("observation.images.")
        }
        result[OBS_LANGUAGE_TOKENS] = torch.tensor(
            tokens, dtype=torch.long, device=self._device
        ).unsqueeze(0)
        result[OBS_LANGUAGE_ATTENTION_MASK] = torch.tensor(
            mask, dtype=torch.bool, device=self._device
        ).unsqueeze(0)
        return result

    def unnormalize_action(self, action: Any) -> Any:
        return self._quantile_transform(
            action, self._action_q01, self._action_q99, inverse=True
        )
