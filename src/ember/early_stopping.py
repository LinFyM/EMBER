"""Decide from complete, regularly spaced paired validation panels only."""
from typing import Mapping, Sequence


def _slope(values):
    middle = (len(values) - 1) / 2
    return sum((i - middle) * value for i, value in enumerate(values)) / sum(
        (i - middle) ** 2 for i in range(len(values)))


def validation_decision(history: Sequence[Mapping], *, interval: int) -> dict:
    """Return a transparent stopping decision; never modify the training state."""
    if not history or interval <= 0:
        raise ValueError("completed validation history and a positive interval are required")
    for index, row in enumerate(history, 1):
        if (row.get("step") != index * interval or row.get("episodes") != 400
                or row.get("complete") is not True or type(row.get("successes")) is not int
                or not 0 <= row["successes"] <= 400):
            raise ValueError("early stopping requires every registered complete paired400 node")
    scores = [row["successes"] for row in history]
    best = max(scores)
    best_index = scores.index(best)
    reason = None
    if len(scores) >= 4:
        previous_best, recent = max(scores[:-3]), scores[-3:]
        if all(value < previous_best for value in recent) and previous_best - sum(recent) / 3 >= 8 and _slope(recent) <= 0:
            reason = "sustained_decline"
    if reason is None and len(scores) >= 6:
        previous_best, recent = max(scores[:-5]), scores[-5:]
        if max(recent) <= previous_best and max(recent) - min(recent) <= 8 and _slope(recent) <= 0:
            reason = "plateau"
    return {"stop": reason is not None, "reason": reason, "best_successes": best,
            "best_step": history[best_index]["step"],
            "tied_best_steps": [row["step"] for row in history if row["successes"] == best],
            "completed_nodes": len(history), "last_step": history[-1]["step"]}
