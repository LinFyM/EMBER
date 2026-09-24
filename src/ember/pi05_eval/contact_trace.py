"""Control-step MuJoCo robot/object contact samples for the sealed channel study."""

from __future__ import annotations

import math
from typing import Any

from ember.pi05_assets import Pi05EvaluationError


def _descends(model: Any, body_id: int, root_id: int) -> bool:
    while body_id:
        if body_id == root_id:
            return True
        body_id = int(model.body_parentid[body_id])
    return root_id == 0


def audit_contact_geometries(env: Any, object_names: tuple[str, ...]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    owner = env.env
    model = owner.sim.model
    robot_root = str(owner.robots[0].robot_model.root_body)
    robot_root_id = int(model.body_name2id(robot_root))
    object_roots = {name: int(owner.obj_body_id[name]) for name in object_names}
    indexed: dict[int, dict[str, Any]] = {}
    counts = {name: 0 for name in object_names}
    robot_count = 0
    for geom_id in range(int(model.ngeom)):
        body_id = int(model.geom_bodyid[geom_id])
        role = "robot" if _descends(model, body_id, robot_root_id) else None
        object_name = None
        if role is None:
            object_name = next((name for name, root in object_roots.items()
                                if _descends(model, body_id, root)), None)
            role = "object" if object_name is not None else None
        if role is None:
            continue
        indexed[geom_id] = {
            "geom_id": geom_id,
            "geom_name": str(model.geom_id2name(geom_id) or f"unnamed_geom_{geom_id}"),
            "body_id": body_id,
            "body_name": str(model.body_id2name(body_id) or f"unnamed_body_{body_id}"),
            "role": role,
            "object_name": object_name,
        }
        if role == "robot":
            robot_count += 1
        else:
            counts[object_name] += 1
    if robot_count == 0 or any(count == 0 for count in counts.values()):
        raise Pi05EvaluationError("approach-channel robot/object geom-body identity audit failed")
    return indexed, {
        "robot_root_body": robot_root, "robot_root_body_id": robot_root_id,
        "robot_geom_count": robot_count,
        "object_root_body_ids": object_roots,
        "object_geom_counts": counts,
        "sampling": "after settling and each executed control step",
        "limitation": "a control-step sample cannot exclude contacts between sampled physics substeps",
    }


def sample_robot_object_contacts(env: Any, indexed: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    data = env.env.sim.data
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for contact in data.contact[:int(data.ncon)]:
        left, right = indexed.get(int(contact.geom1)), indexed.get(int(contact.geom2))
        if left is None or right is None or {left["role"], right["role"]} != {"robot", "object"}:
            continue
        robot, obj = (left, right) if left["role"] == "robot" else (right, left)
        distance = float(contact.dist)
        if not math.isfinite(distance):
            raise Pi05EvaluationError("nonfinite robot/object contact distance")
        key = (robot["geom_id"], obj["geom_id"])
        if key not in pairs:
            pairs[key] = {
                "robot_geom_id": robot["geom_id"], "robot_geom_name": robot["geom_name"],
                "robot_body_id": robot["body_id"], "robot_body_name": robot["body_name"],
                "object_name": obj["object_name"],
                "object_geom_id": obj["geom_id"], "object_geom_name": obj["geom_name"],
                "object_body_id": obj["body_id"], "object_body_name": obj["body_name"],
                "minimum_distance_m": distance, "point_count": 1,
            }
        else:
            pairs[key]["minimum_distance_m"] = min(pairs[key]["minimum_distance_m"], distance)
            pairs[key]["point_count"] += 1
    return [pairs[key] for key in sorted(pairs)]
