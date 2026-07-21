"""Official OpenPI tokenization and source-only normalization for pi0.5."""

from __future__ import annotations

from collections.abc import Sequence
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

    def _tokenize_prompts(self, states: Any, tasks: Sequence[str]) -> tuple[Any, Any]:
        """Tokenize a batch with the exact OpenPI pi0.5 state prompt."""

        import torch

        if states.ndim != 2 or states.shape[0] != len(tasks):
            raise ValueError("pi0.5 prompt states and task strings must form one batch")
        normalized = self._quantile_transform(
            states, self._state_q01, self._state_q99, inverse=False
        )
        discretized = np.digitize(
            normalized.detach().cpu().numpy(),
            bins=np.linspace(-1, 1, 256 + 1)[:-1],
        ) - 1
        prompts = []
        for task, state in zip(tasks, discretized, strict=True):
            cleaned = str(task).strip().replace("_", " ").replace("\n", " ")
            prompts.append(
                f"Task: {cleaned}, State: {' '.join(map(str, state))};\nAction: "
            )
        encoded = self._tokenizer.encode(prompts, add_bos=True)
        tokens = torch.zeros(
            (len(prompts), self._max_length), dtype=torch.long, device=self._device
        )
        masks = torch.zeros_like(tokens, dtype=torch.bool)
        for row, values in enumerate(encoded):
            values = values[: self._max_length]
            length = len(values)
            if length:
                tokens[row, :length] = torch.as_tensor(
                    values, dtype=torch.long, device=self._device
                )
                masks[row, :length] = True
        return tokens, masks

    @staticmethod
    def _quantile_transform(value: Any, q01: Any, q99: Any, inverse: bool) -> Any:
        denominator = q99 - q01 + 1e-6
        if inverse:
            return (value + 1.0) * denominator / 2.0 + q01
        return 2.0 * (value - q01) / denominator - 1.0

    def __call__(self, value: dict[str, Any]) -> dict[str, Any]:
        import torch
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )

        state = value["observation.state"].to(self._device).unsqueeze(0)
        tokens, masks = self._tokenize_prompts(state, [value["task"]])
        result = {
            key: tensor.unsqueeze(0).to(self._device)
            for key, tensor in value.items()
            if isinstance(tensor, torch.Tensor) and key.startswith("observation.images.")
        }
        result[OBS_LANGUAGE_TOKENS] = tokens
        result[OBS_LANGUAGE_ATTENTION_MASK] = masks
        return result

    def training_batch(self, value: dict[str, Any]) -> dict[str, Any]:
        """Map sealed LIBERO HDF5 rows to the canonical LeRobot PI05 forward contract."""

        import torch
        from lerobot.utils.constants import (
            ACTION,
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )

        states = value["observation.state"].to(
            self._device, dtype=torch.float32, non_blocking=True
        )
        tasks = value["task"]
        if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
            raise ValueError("training batch must contain one task string per row")
        tokens, masks = self._tokenize_prompts(states, tasks)

        def image(key: str) -> Any:
            tensor = value[key].to(self._device, non_blocking=True)
            if tensor.dtype == torch.uint8:
                tensor = tensor.to(torch.float32).div_(255.0)
            elif tensor.dtype != torch.float32:
                tensor = tensor.to(torch.float32)
            if tensor.ndim != 4 or tensor.shape[1] != 3:
                raise ValueError(f"invalid PI05 training image batch: {key}")
            return tensor

        base = image("observation.images.camera1")
        wrist = image("observation.images.camera2")
        actions = value[ACTION].to(
            self._device, dtype=torch.float32, non_blocking=True
        )
        actions = self._quantile_transform(
            actions, self._action_q01, self._action_q99, inverse=False
        )
        return {
            "observation.images.base_0_rgb": base,
            "observation.images.left_wrist_0_rgb": wrist,
            "observation.images.right_wrist_0_rgb": torch.zeros_like(base),
            OBS_LANGUAGE_TOKENS: tokens,
            OBS_LANGUAGE_ATTENTION_MASK: masks,
            ACTION: actions,
        }

    def unnormalize_action(self, action: Any) -> Any:
        return self._quantile_transform(
            action, self._action_q01, self._action_q99, inverse=True
        )
