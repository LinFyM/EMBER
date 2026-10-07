"""Scoped intervention at the real PiGemma action block-output boundary."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from torch import Tensor, nn

from ember.operator_writer.native import _TEACHER_ATTENTION


KeyValues = tuple[tuple[Tensor, Tensor], ...]
_ACTIVE_EXECUTION: ContextVar[ExecutionScope | None] = ContextVar(
    "ember_native_video_control_execution", default=None)


@dataclass
class ExecutionScope:
    """One condition's cached projections and observable native Reader calls."""

    readers: Sequence[nn.Module]
    key_values: KeyValues
    calls: list[int] = field(default_factory=lambda: [0] * 18)

    def hook(self, layer: int):
        def intervene(_module, _arguments, output):
            if _ACTIVE_EXECUTION.get() is not self or _TEACHER_ATTENTION.get():
                return output
            if not isinstance(output, Tensor):
                raise ValueError("PiGemma action block output is no longer a Tensor")
            self.calls[layer] += 1
            return self.readers[layer](output, self.key_values[layer])
        return intervene


@contextmanager
def native_execution_scope(policy: nn.Module, readers: Sequence[nn.Module],
                           key_values: KeyValues) -> Iterator[ExecutionScope]:
    """Hook the existing suffix-only denoise consumer without copying its loop.

    NativeFlowPrediction and official sampling both call PI05.denoise_step,
    whose bridge ``[None, suffix]`` branch invokes all PiGemmaDecoderLayer
    modules. Each hook runs after the block's second gated residual, before
    the next layer; native final AdaRMS and action_out remain untouched.

    The teacher's ``[prefix, suffix]`` branch instead calls the installed
    compute_layer_complete function, which bypasses these module boundaries.
    Its teacher context also explicitly suppresses Readers. No layer placement
    is approximated with an MLP or projection hook.

    Lifetime is deliberately explicit: keep the scope through query backward.
    Whole-call checkpoint closures must reenter scope and reinstall functional
    beta on every replay. The current consumer disables native inner replay.
    """
    blocks = policy.model.paligemma_with_expert.gemma_expert.model.layers
    if len(blocks) != 18 or len(readers) != 18 or len(key_values) != 18:
        raise ValueError("execution must cover all 18 real native action blocks")
    if _ACTIVE_EXECUTION.get() is not None:
        raise ValueError("native memory execution scopes must not overlap")
    scope = ExecutionScope(readers, key_values)
    token = _ACTIVE_EXECUTION.set(scope)
    handles = []
    try:
        for layer, block in enumerate(blocks):
            handles.append(block.register_forward_hook(scope.hook(layer)))
        yield scope
    finally:
        for handle in handles:
            handle.remove()
        _ACTIVE_EXECUTION.reset(token)
