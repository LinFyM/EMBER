"""Fixed video relations scored against complete, actual own observation traces.

Only predicted teacher geometry enters this module. The caller binds once at the
canonical initial state and supplies that same mapping and complete parent prefix
to every branch, including all real pre/post and terminal observations. No teacher
labels, environment registry, policy, or action-fitting consumer are read here.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


_GROUPS = (
    "relative_position", "relative_rotation", "entity_height", "joint_q",
    "hand_q", "hand_gravity",
)


def _array(value: Any, name: str, shape: tuple[int | None, ...]) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != len(shape) or any(
        size is not None and actual != size for actual, size in zip(result.shape, shape)
    ):
        raise ValueError(f"{name} has shape {result.shape}, expected {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _initial(value: Any, name: str, shape: tuple[int | None, ...]) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value)
    if value.ndim == len(shape) + 1 and value.shape[0]:
        value = value[0]
    return _array(value, name, shape)


def _smooth_l1(value: np.ndarray) -> np.ndarray:
    absolute = np.abs(value)
    return np.where(absolute < 1, 0.5 * value**2, absolute - 0.5)


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = left @ right.T
    denominator = np.linalg.norm(left, axis=-1)[:, None] * np.linalg.norm(right, axis=-1)
    return np.clip(
        np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0),
        -1, 1,
    )


def _joint_types(value: Any, count: int) -> np.ndarray:
    types = _array(value, "own joint_type", (count,))
    if not np.isin(types, (0, 1, 2)).all():
        raise ValueError("own joint_type must be 0=none, 1=slide, or 2=hinge")
    return types.astype(np.int64)


def _success(value: Any) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError("success must be the actual own-environment boolean")
    return bool(value)


def no_video_score(success: bool) -> dict[str, Any]:
    """The matched no-video reference uses only its actual terminal success."""
    success = _success(success)
    return {"status": "no_video", "success": success, "S_V": 0.0, "R": 2.0 * success}


def _edges(count: int, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right = np.triu_indices(count, k=1)
    return (
        np.concatenate((np.zeros(count, dtype=np.int64), left + 1)),
        np.concatenate((np.arange(1, count + 1), right + 1)),
        np.concatenate((weights, weights[left] * weights[right])),
    )


def _edge_relations(
    hand_p: np.ndarray, hand_R: np.ndarray, entity_p: np.ndarray, entity_R: np.ndarray,
    left: np.ndarray, right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.concatenate((hand_p[:, None], entity_p), axis=1)
    rotations = np.concatenate((hand_R[:, None], entity_R), axis=1)
    # World-relative displacement; Q_ab = R_a^T R_b, for hand->entity and i<j.
    displacement = positions[:, right] - positions[:, left]
    relative_rotation = np.swapaxes(rotations[:, left], -1, -2) @ rotations[:, right]
    return displacement, relative_rotation


def _dtw(cost: np.ndarray) -> tuple[float, dict[str, Any]]:
    n, m = cost.shape
    distance = np.full((n + 1, m + 1), np.inf)
    distance[0, 0] = 0.0
    predecessor = np.empty((n, m), dtype=np.uint8)
    for i in range(n):
        for j in range(m):
            neighbors = (distance[i, j], distance[i, j + 1], distance[i + 1, j])
            step = int(np.argmin(neighbors))  # diagonal, own-only, teacher-only
            distance[i + 1, j + 1] = cost[i, j] + neighbors[step]
            predecessor[i, j] = step
    i, j = n - 1, m - 1
    path = []
    while True:
        path.append([i, j])
        if i == 0 and j == 0:
            break
        step = int(predecessor[i, j])
        i -= step != 2
        j -= step != 1
    path.reverse()
    total = float(distance[n, m])
    normalizer = max(n, m)
    return total / normalizer, {
        "cumulative_cost": total, "normalizer": normalizer,
        "own_observations": n, "teacher_frames": m,
        "path_length": len(path), "path": path,
    }


class VideoTarget:
    """One complete, ordered Phi prediction and its fixed full-video weights.

    ``bind`` returns a JSON-compatible mapping. ``score`` never rematches: the
    caller must reuse the initial mapping and include the full branch prefix.
    An unmatched condition has ``S_V=R=None`` and must deliver the MT fallback.
    """

    def __init__(self, pred: Mapping[str, Any]):
        first = _array(pred["hand_p"], "teacher hand_p", (None, 3))
        self.frames = len(first)
        if not self.frames:
            raise ValueError("teacher video must contain actual frames")
        shapes = {
            "hand_p": (self.frames, 3), "hand_R": (self.frames, 3, 3),
            "hand_q": (self.frames, 2), "entity_p": (self.frames, 32, 3),
            "entity_R": (self.frames, 32, 3, 3), "semantic": (self.frames, 32, 64),
            "presence": (self.frames, 32), "relevance": (self.frames, 32),
            "joint_type": (self.frames, 32, 3), "joint_q": (self.frames, 32, 2),
        }
        self._pred = {key: _array(pred[key], f"teacher {key}", shape).copy()
                      for key, shape in shapes.items()}
        for key in ("presence", "relevance"):
            if ((self._pred[key] < 0) | (self._pred[key] > 1)).any():
                raise ValueError(f"teacher {key} must be probabilities, not logits")
        self._weights = (self._pred["presence"] * self._pred["relevance"]).mean(axis=0)
        self._types = self._pred["joint_type"].argmax(axis=-1)

    def bind(self, own_first: Mapping[str, Any]) -> dict[str, Any]:
        """Match once using initial semantic cosine and position SmoothL1/.25m."""
        position = _initial(own_first["entity_p"], "own initial entity_p", (None, 3))
        count = len(position)
        semantic = _initial(own_first["semantic"], "own initial semantic", (count, 64))
        types = _joint_types(own_first["joint_type"], count)
        presence = _initial(own_first.get("presence", np.ones(count)),
                            "own initial presence", (count,))
        if ((presence < 0) | (presence > 1)).any():
            raise ValueError("own initial presence must be boolean or probability")
        slots = np.flatnonzero(self._pred["presence"][0] >= 0.5)
        entities = np.flatnonzero(presence >= 0.5)
        rows = columns = np.empty(0, dtype=np.int64)
        if len(slots) and len(entities):
            cost = 1 - _cosine(self._pred["semantic"][0, slots], semantic[entities])
            delta = (self._pred["entity_p"][0, slots, None] - position[entities]) / 0.25
            cost += _smooth_l1(delta).mean(axis=-1)
            rows, columns = linear_sum_assignment(cost)
        matched = slots[rows]
        return {
            "schema": "ember_video_entity_mapping_v1",
            "status": "ok" if len(matched) else "no_match",
            "teacher_slots": matched.tolist(), "own_entities": entities[columns].tolist(),
            "weights": self._weights[matched].tolist(), "own_entity_count": count,
            "own_joint_type": types.tolist(),
            "unmatched_teacher_slots": np.setdiff1d(slots, matched).tolist(),
            "excluded_teacher_slots": np.flatnonzero(self._pred["presence"][0] < 0.5).tolist(),
        }

    def _trace(self, trace: Mapping[str, Any], count: int) -> dict[str, np.ndarray]:
        hand = _array(trace["hand_p"], "own hand_p", (None, 3))
        length = len(hand)
        if not length:
            raise ValueError("own trace must contain every actual observation")
        shapes = {
            "hand_p": (length, 3), "hand_R": (length, 3, 3), "hand_q": (length, 2),
            "entity_p": (length, count, 3), "entity_R": (length, count, 3, 3),
            "joint_q": (length, count, 2),
        }
        return {key: _array(trace[key], f"own {key}", shape) for key, shape in shapes.items()}

    def _joint_distances(
        self, own_q: np.ndarray, slots: np.ndarray, own_types: np.ndarray, weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        types = self._types[:, slots]
        joint_weights = weights[None] * ((types > 0) & (types == own_types[None]))
        denominator = joint_weights.sum(axis=-1)
        delta = (own_q[None] - self._pred["joint_q"][:, slots]) / np.array((0.25, np.pi))
        error = np.take_along_axis(_smooth_l1(delta), np.maximum(types - 1, 0)[..., None], -1)[..., 0]
        return np.divide((error * joint_weights).sum(axis=-1), denominator,
                         out=np.zeros(self.frames), where=denominator > 0), denominator > 0

    def _distance_grid(
        self, own: dict[str, np.ndarray], slots: np.ndarray, entities: np.ndarray,
        own_types: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights = self._weights[slots]
        left, right, edge_weights = _edges(len(slots), weights)
        video_delta, video_Q = _edge_relations(
            self._pred["hand_p"], self._pred["hand_R"],
            self._pred["entity_p"][:, slots], self._pred["entity_R"][:, slots], left, right,
        )
        own_delta, own_Q = _edge_relations(
            own["hand_p"], own["hand_R"], own["entity_p"][:, entities],
            own["entity_R"][:, entities], left, right,
        )
        edge_sum, entity_sum = edge_weights.sum(), weights.sum()
        length = len(own["hand_p"])
        groups = np.zeros((length, self.frames, len(_GROUPS)))
        valid = np.ones((self.frames, len(_GROUPS)), dtype=bool)
        valid[:, :2] = edge_sum > 0
        valid[:, 2] = entity_sum > 0
        for i in range(length):
            if edge_sum > 0:
                error = _smooth_l1((own_delta[i] - video_delta) / 0.25).mean(axis=-1)
                groups[i, :, 0] = (error * edge_weights).sum(axis=-1) / edge_sum
                error = 0.25 * ((own_Q[i] - video_Q)**2).sum(axis=(-1, -2))
                groups[i, :, 1] = (error * edge_weights).sum(axis=-1) / edge_sum
            if entity_sum > 0:
                error = _smooth_l1((own["entity_p"][i, entities, 2]
                                    - self._pred["entity_p"][:, slots, 2]) / 0.25)
                groups[i, :, 2] = (error * weights).sum(axis=-1) / entity_sum
            groups[i, :, 3], valid[:, 3] = self._joint_distances(
                own["joint_q"][i, entities], slots, own_types[entities], weights,
            )
            groups[i, :, 4] = _smooth_l1((own["hand_q"][i] - self._pred["hand_q"]) / 0.04).mean(-1)
            dot = self._pred["hand_R"][:, 2] @ own["hand_R"][i, 2]
            groups[i, :, 5] = (1 - np.clip(dot, -1, 1)) / 2
        cost = (groups * valid).sum(axis=-1) / valid.sum(axis=-1)
        return cost, groups, valid

    def score(self, trace: Mapping[str, Any], mapping: Mapping[str, Any]) -> dict[str, Any]:
        """Return R=2*success+exp(-DTW/max(n,m)), with fixed mapping diagnostics."""
        success = _success(trace["success"])
        slots = np.asarray(mapping["teacher_slots"], dtype=np.int64)
        entities = np.asarray(mapping["own_entities"], dtype=np.int64)
        unmatched = list(mapping["unmatched_teacher_slots"])
        if not len(slots):
            return {"status": "no_match", "success": success, "S_V": None, "R": None,
                    "d": None, "dtw": None, "unmatched_teacher_slots": unmatched,
                    "fallback": "MT", "matched_entities": 0}
        count = int(mapping["own_entity_count"])
        if (slots.ndim != 1 or entities.shape != slots.shape or len(set(slots)) != len(slots)
                or len(set(entities)) != len(entities) or ((slots < 0) | (slots >= 32)).any()
                or ((entities < 0) | (entities >= count)).any()):
            raise ValueError("mapping must contain unique, in-range teacher and own indices")
        types = _joint_types(trace["joint_type"], count)
        if types.tolist() != list(mapping["own_joint_type"]):
            raise ValueError("own physical registry changed after initial mapping")
        own = self._trace(trace, count)
        cost, groups, valid = self._distance_grid(own, slots, entities, types)
        distance, diagnostic = _dtw(cost)
        path = np.asarray(diagnostic["path"], dtype=np.int64)
        path_values, path_valid = groups[path[:, 0], path[:, 1]], valid[path[:, 1]]
        diagnostic["group_path_mean"] = {
            name: float(path_values[path_valid[:, group], group].mean())
            if path_valid[:, group].any() else None for group, name in enumerate(_GROUPS)
        }
        video_score = float(np.exp(-distance))
        return {"status": "ok", "success": success, "S_V": video_score,
                "R": 2.0 * success + video_score, "d": distance, "dtw": diagnostic,
                "matched_entities": len(slots), "unmatched_teacher_slots": unmatched}
