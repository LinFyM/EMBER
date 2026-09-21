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


def phase_validation_decision(
    history: Sequence[Mapping], *, parent_step: int = 1800, interval: int = 100
) -> dict:
    """Decide only from complete low-LR phase panels on their global-step axis."""
    if not history or parent_step < 0 or interval <= 0:
        raise ValueError("completed phase history and valid global-step cadence are required")
    for index, row in enumerate(history, 1):
        expected = parent_step + index * interval
        if (row.get("step") != expected or row.get("phase_step") != index * interval
                or row.get("episodes") != 400 or row.get("complete") is not True
                or type(row.get("successes")) is not int or not 0 <= row["successes"] <= 400):
            raise ValueError("phase stopping requires every registered complete paired400 node")
    scores = [row["successes"] for row in history]
    reason = None
    if len(scores) >= 4:
        previous_best, recent = max(scores[:-3]), scores[-3:]
        if (all(value < previous_best for value in recent)
                and previous_best - sum(recent) / 3 >= 8 and _slope(recent) <= 0):
            reason = "sustained_decline"
    if reason is None and len(scores) >= 6:
        previous_best, recent = max(scores[:-5]), scores[-5:]
        if max(recent) <= previous_best and max(recent) - min(recent) <= 8 and _slope(recent) <= 0:
            reason = "plateau"
    if reason is None and len(scores) >= 7:
        strict_best_index = 0
        best = scores[0]
        for index, score in enumerate(scores[1:], 1):
            if score > best:
                best, strict_best_index = score, index
        nodes_since_refresh = len(scores) - strict_best_index - 1
        if nodes_since_refresh >= 6 and _slope(scores[-3:]) <= 0:
            reason = "no_progress_review"
    best = max(scores)
    best_index = scores.index(best)
    return {
        "stop": reason is not None,
        "reason": reason,
        "status": "无进展，停止复核" if reason == "no_progress_review" else reason,
        "best_successes": best,
        "best_step": history[best_index]["step"],
        "best_phase_step": history[best_index]["phase_step"],
        "tied_best_steps": [row["step"] for row in history if row["successes"] == best],
        "completed_nodes": len(history),
        "last_step": history[-1]["step"],
        "last_phase_step": history[-1]["phase_step"],
    }
