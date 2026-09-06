"""Native Action Expert boundary capture, without a learned observer graph.

The capture contains the input boundary followed by all 18 post-layer
boundaries. Callers choose the required boundaries and whether to retain grads.
"""

from __future__ import annotations

from contextlib import AbstractContextManager

import torch


class ActionLayerStateCapture(AbstractContextManager["ActionLayerStateCapture"]):
    def __init__(self, expert_model: torch.nn.Module, *, detach: bool) -> None:
        self.expert_model = expert_model
        self.detach = detach
        self.values: list[torch.Tensor | None] = [None] * (len(expert_model.layers) + 1)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "ActionLayerStateCapture":
        modules = [layer.input_layernorm for layer in self.expert_model.layers]
        modules.append(self.expert_model.norm)
        for index, module in enumerate(modules):

            def hook(
                _module: torch.nn.Module,
                inputs: tuple[torch.Tensor, ...],
                *,
                selected: int = index,
            ) -> None:
                value = inputs[0]
                self.values[selected] = value.detach() if self.detach else value

            self.handles.append(module.register_forward_pre_hook(hook))
        return self

    def stacked(self) -> torch.Tensor:
        if any(value is None for value in self.values):
            raise RuntimeError("PI0.5 Action Expert layer capture is incomplete")
        return torch.stack([value for value in self.values if value is not None], dim=1)

    def __exit__(self, *args: object) -> None:
        for handle in self.handles:
            handle.remove()
