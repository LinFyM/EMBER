"""Bounded MT300 continuation and native video-memory control diagnostic."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ember.lora import validate_lora_state
from ember.pi05_lora import Pi05LoRAContract, pi05_target_names
from ember.writer.model import DirectLoRAParameters

from .hooks import ExecutionScope, KeyValues, native_execution_scope


def rms(value: Tensor) -> Tensor:
    """Fixed, non-affine RMS with FP32 statistics and the registered epsilon."""
    value = value.float()
    return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-6)


class NativeReader(nn.Module):
    """One block's independent state-dependent, eight-head memory read."""

    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(1024, 256, bias=False, dtype=torch.float32, device="cpu")
        self.k = nn.Linear(256, 256, bias=False, dtype=torch.float32, device="cpu")
        self.v = nn.Linear(256, 256, bias=False, dtype=torch.float32, device="cpu")
        self.o = nn.Linear(256, 1024, bias=False, dtype=torch.float32, device="cpu")
        nn.init.zeros_(self.o.weight)

    @staticmethod
    def _heads(value: Tensor) -> Tensor:
        return value.reshape(value.shape[0], value.shape[1], 8, 32).transpose(1, 2)

    def prepare_memory(self, memory: Tensor) -> tuple[Tensor, Tensor]:
        normalized = rms(memory)
        return self._heads(self.k(normalized)), self._heads(self.v(normalized))

    def forward(self, hidden: Tensor, key_values: tuple[Tensor, Tensor]) -> Tensor:
        if hidden.ndim != 3 or hidden.shape[1:] != (50, 1024):
            raise ValueError("native Reader needs the complete post-block action hidden")
        key, value = key_values
        if (key.ndim != 4 or key.shape != value.shape or key.shape[0:2] != (1, 8)
                or key.shape[-1] != 32 or key.device != hidden.device):
            raise ValueError("native Reader memory must retain one full condition")
        query = self._heads(self.q(rms(hidden)))
        attended = F.scaled_dot_product_attention(
            query, key.expand(len(hidden), -1, -1, -1), value.expand(len(hidden), -1, -1, -1),
            dropout_p=0.0, is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(len(hidden), 50, 256)
        # Preserve the native hidden's numerical policy and the next block's dtype.
        return hidden + self.o(attended).to(hidden.dtype)


class NativeVideoControl(nn.Module):
    """Own one complete FP32 beta, and only V's fresh video encoder/Readers.

    The physical source policy remains owned by the existing native runtime.
    ``public_state`` is substituted using torch.func by the actual consumer.
    """

    def __init__(self, contract: Pi05LoRAContract, template: Mapping[str, Tensor],
                 arm: str = "V") -> None:
        super().__init__()
        self.names = tuple(target.name for target in contract.targets)
        if arm not in {"M", "V"} or contract.rank != 128 or self.names != pi05_target_names():
            raise ValueError("native control requires M/V and the complete MT rank128 topology")
        validate_lora_state(template, contract)
        self.arm = arm
        self.common = DirectLoRAParameters({name: value.float() for name, value in template.items()})
        self.readers = nn.ModuleList()
        self.encoder_layers = nn.ModuleList()
        self.video_in: nn.Linear | None = None
        self.memory_norm: nn.LayerNorm | None = None
        if arm == "V":
            # CPU initialization neither consumes nor resets the policy's CUDA RNG.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(7)
                self.video_in = nn.Linear(1024, 256, dtype=torch.float32, device="cpu")
                self.encoder_layers.extend(nn.TransformerEncoderLayer(
                    d_model=256, nhead=8, dim_feedforward=1024, dropout=0.0,
                    activation="gelu", batch_first=True, norm_first=True,
                    dtype=torch.float32, device="cpu",
                ) for _ in range(2))
                self.memory_norm = nn.LayerNorm(256, dtype=torch.float32, device="cpu")
                self.readers.extend(NativeReader() for _ in range(18))
            self.register_buffer("probe", torch.randn(
                50, 32, generator=torch.Generator(device="cpu").manual_seed(1729), device="cpu"))
            self.register_buffer("inv_frequency", 10000. ** (
                -torch.arange(0, 128, 2, dtype=torch.float32, device="cpu") / 128), persistent=False)

    def public_state(self) -> dict[str, Tensor]:
        return self.common()

    def _position_code(self, positions: Tensor) -> Tensor:
        angles = positions.float()[..., None] * self.inv_frequency.float()
        return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)

    def encode(self, h: Tensor, indices: Tensor) -> Tensor:
        """Retain every real time/horizon slot; full bidirectional memory attention."""
        if self.arm != "V" or h.ndim != 3 or h.shape[1:] != (50, 1024) or len(h) < 2:
            raise ValueError("video encoder needs V and the complete native T x 50 x 1024 grid")
        indices = torch.as_tensor(indices, device=h.device)
        if (indices.shape != (len(h),) or indices.dtype == torch.bool
                or indices.is_floating_point() or indices.is_complex()
                or bool((indices < 0).any()) or bool((indices[1:] <= indices[:-1]).any())):
            raise ValueError("video encoder needs increasing real integer frame indices")
        time = self._position_code(indices.float() / 5)[:, None].expand(-1, 50, -1)
        horizon = self._position_code(torch.arange(50, device=h.device))[None].expand(len(h), -1, -1)
        memory = (self.video_in(rms(h)) + torch.cat((time, horizon), dim=-1)).flatten(0, 1)[None]
        for layer in self.encoder_layers:
            memory = layer(memory)
        return self.memory_norm(memory)

    def prepare_memory(self, memory: Tensor) -> KeyValues:
        """Cache K/V without detaching: training keeps all K/V -> C credit.

        Inference may call this once under no_grad and reuse the returned values
        across replans of the same fixed condition and Reader parameter version.
        """
        if (self.arm != "V" or memory.ndim != 3 or memory.shape[0] != 1
                or memory.shape[-1] != 256 or memory.shape[1] < 100 or memory.shape[1] % 50):
            raise ValueError("Reader preparation needs V's complete single-condition memory")
        return tuple(reader.prepare_memory(memory) for reader in self.readers)

    @contextmanager
    def execution_scope(self, policy: nn.Module, memory: Tensor | None,
                        *, key_values: KeyValues | None = None) -> Iterator[ExecutionScope | None]:
        """Apply Readers only to execution; keep this open through backward.

        An outer activation checkpoint must reenter this scope and torch.func
        substitution inside its complete call closure. A restored physical beta
        must never be used by a checkpoint's query recomputation.
        """
        if self.arm == "M":
            if memory is not None or key_values is not None:
                raise ValueError("M has no teaching input")
            yield None
            return
        if memory is None:
            raise ValueError("V execution requires its fixed teaching memory")
        prepared = self.prepare_memory(memory) if key_values is None else key_values
        with native_execution_scope(policy, self.readers, prepared) as scope:
            yield scope
