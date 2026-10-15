"""One anchored parameter encoder/decoder; no policy, task lookup or Reader."""
from __future__ import annotations

import math

import torch
from torch import nn

from ember.lora import LORA_A_SUFFIX, LORA_B_SUFFIX, validate_lora_state
from ember.proposal_writer.model import FactorEncoder, FactorLayout


class ControlSpace(nn.Module):
    """Fresh E/D modules; callers seed Torch before construction.

    identity is the canonical source A/nonzero, B/zero complete rank8 state.
    scales supplies one fixed positive RMS unit for each of its 76 tensors.
    Only this module owns the layout, so its buffers occur once in checkpoints.
    """
    def __init__(self, lora, identity, scales):
        super().__init__()
        if lora.rank != 8 or len(lora.targets) != 38:
            raise ValueError('control space requires the complete 38-target rank8 layout')
        validate_lora_state(identity, lora)
        if set(scales) != set(identity) or any(not math.isfinite(float(s)) or float(s) <= 0 for s in scales.values()):
            raise ValueError('control-space units must cover all 76 factors and be finite positive scalars')
        if any(bool(value.count_nonzero()) for name, value in identity.items() if name.endswith(LORA_B_SUFFIX)):
            raise ValueError('source identity requires zero B factors')
        if any(not bool(value.count_nonzero()) for name, value in identity.items() if name.endswith(LORA_A_SUFFIX)):
            raise ValueError('source identity requires nonzero A factors')
        self.layout = FactorLayout(lora, {k: v.detach() for k, v in identity.items()}, scales)
        self.encoder = FactorEncoder(self.layout, layers=2)
        self.to_code = nn.Linear(256, 64)
        self.decoder = FactorEncoder(self.layout, layers=6, endpoint=True)
        self.to_memory = nn.Linear(64, 256)

    def _normalized(self, state):
        packed = self.layout.pack(state).to(self.layout.center)
        return (packed - self.layout.center) / self.layout.scale

    def _encode(self, blocks):
        return self.to_code(self.encoder(blocks).mean(0))

    def _decode(self, code):
        if code.shape != (64,):
            raise ValueError('one complete control code must have shape (64,)')
        code = code.to(self.to_memory.weight)
        memory = self.to_memory(code).unsqueeze(0)
        return self.decoder(torch.zeros_like(self.layout.center), memory=memory)

    def _factors(self, blocks, anchor):
        packed = self.layout.center.float() + self.layout.scale.float() * (blocks.float() - anchor.float())
        return self.layout.unpack(packed)

    def encode(self, state):
        """E(r)=e(r)-e(0), retaining both current parameter graphs."""
        blocks = self._normalized(state)
        return self._encode(blocks) - self._encode(torch.zeros_like(blocks))

    def decode(self, code):
        """D(z)=identity+S*(h(z)-h(0)); all 76 output factors are FP32."""
        return self._factors(self._decode(code), self._decode(torch.zeros_like(code)))

    def forward(self, state):
        return self.forward_many([state])[0]

    def forward_many(self, states):
        """Share live anchors only within this call; backpropagate outputs jointly."""
        if not states:
            return []
        zero = torch.zeros_like(self.layout.center)
        encoder_anchor = self._encode(zero)
        decoder_anchor = self._decode(zero.new_zeros(64))
        return [self._factors(self._decode(self._encode(self._normalized(state)) - encoder_anchor),
                              decoder_anchor) for state in states]
