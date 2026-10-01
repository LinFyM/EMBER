"""Historical exploration RNG identity used to validate archived paired rows."""
from __future__ import annotations
from typing import Sequence

def _seed(parts: Sequence[int]) -> int:
    return int(np.random.SeedSequence(parts).generate_state(1, dtype=np.uint64)[0]) & ((1 << 63) - 1)

def exploration_seed(task: int, state: int, replica: int, replan: int) -> int:
    return _seed((20260925, task, state, replica, replan, 0x524C))
