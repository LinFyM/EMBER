"""Official OpenPI tokenization and source-only normalization for pi0.5."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert one LIBERO end-effector quaternion to the OpenPI axis angle."""

    value = np.asarray(quat, dtype=np.float32).copy()
    if value.shape != (4,):
        raise ValueError("LIBERO end-effector quaternion must have shape (4,)")
    value[3] = np.clip(value[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - float(value[3] * value[3])))
    if denominator < 1e-10:
        return np.zeros(3, dtype=np.float32)
    return value[:3] * (2.0 * np.arccos(value[3]) / denominator)


def libero_policy_input(obs: Mapping[str, Any], language: str) -> dict[str, Any]:
    """Build the one-observation PI05 input shared by evaluation and reward RL."""

    import torch

    def image(value: Any) -> torch.Tensor:
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("LIBERO RGB observation must have shape (H,W,3)")
        rotated = np.ascontiguousarray(array[::-1, ::-1])
        return torch.from_numpy(rotated).permute(2, 0, 1).float().div_(255.0)

    cleaned = str(language).strip()
    if not cleaned:
        raise ValueError("LIBERO policy language must be non-empty")
    state = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    ).astype(np.float32)
    if state.shape != (8,):
        raise ValueError("LIBERO PI05 policy state must have exactly eight values")
    # Missing right wrist stays absent so PI05 creates a false image mask.
    return {
        "observation.images.base_0_rgb": image(obs["agentview_image"]),
        "observation.images.left_wrist_0_rgb": image(
            obs["robot0_eye_in_hand_image"]
        ),
        "observation.state": torch.from_numpy(state),
        "task": cleaned,
    }


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
        # OpenPI pads LIBERO's missing right wrist with image_mask=False. Omitting
        # the key is the pinned LeRobot PI05 preprocessor's exact equivalent.
        return {
            "observation.images.base_0_rgb": base,
            "observation.images.left_wrist_0_rgb": wrist,
            OBS_LANGUAGE_TOKENS: tokens,
            OBS_LANGUAGE_ATTENTION_MASK: masks,
            ACTION: actions,
        }

    def unnormalize_action(self, action: Any) -> Any:
        return self._quantile_transform(
            action, self._action_q01, self._action_q99, inverse=True
        )


class Pi05PureLanguageTokenizer:
    """Tokenize only task language for Writer features, with no policy state prompt."""

    def __init__(self, tokenizer_path: Path, max_length: int, device: str) -> None:
        import sentencepiece

        if max_length <= 0:
            raise ValueError("pure-language tokenizer max length must be positive")
        self._tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=str(tokenizer_path)
        )
        self._max_length = max_length
        self._device = device

    @staticmethod
    def format_prompt(task: str) -> str:
        cleaned = str(task).strip().replace("_", " ").replace("\n", " ")
        if not cleaned:
            raise ValueError("Writer task language must be non-empty")
        return f"Task: {cleaned}\n"

    def __call__(self, tasks: Sequence[str]) -> tuple[Any, Any]:
        import torch

        if not tasks or isinstance(tasks, (str, bytes)):
            raise ValueError("pure-language tokenizer requires a task sequence")
        prompts = [self.format_prompt(task) for task in tasks]
        encoded = self._tokenizer.encode(prompts, add_bos=True)
        lengths = [len(values) for values in encoded]
        if any(length > self._max_length for length in lengths):
            raise ValueError(
                "Writer task language exceeds the sealed tokenizer length: "
                f"{lengths} > {self._max_length}"
            )
        tokens = torch.zeros(
            (len(prompts), self._max_length),
            dtype=torch.long,
            device=self._device,
        )
        masks = torch.zeros_like(tokens, dtype=torch.bool)
        for row, values in enumerate(encoded):
            if values:
                tokens[row, : len(values)] = torch.as_tensor(
                    values, dtype=torch.long, device=self._device
                )
                masks[row, : len(values)] = True
        if not bool(masks.any(dim=1).all()):
            raise ValueError("pure-language tokenization produced an empty task")
        return tokens, masks
