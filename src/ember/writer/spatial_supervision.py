"""Training-only physical targets and KL on the Writer's actual visual reads."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


SCHEMA = "visible_object_attention_labels_v1"


def validate_spatial_config(config):
    expected = {"schema": SCHEMA, "label_root": "runs/analysis/native_correction_writer_20260913/spatial_labels",
                "weight": .1, "native_patches": 512, "prior_patches": 576,
                "objective": "mean_head_query_actual_attention_object_and_motion_kl"}
    if config != expected:
        raise ValueError("registered spatial supervision contract changed")


class SpatialLabelStore:
    """IDs select training targets in the data layer; targets never enter Writer."""

    def __init__(self, asset_root: Path, config, task_ids, demos):
        validate_spatial_config(config)
        self.root = asset_root / config["label_root"]
        registration = json.loads((self.root / "registration.json").read_text())
        completion = json.loads((self.root / "completion.json").read_text())
        if (registration["schema"] != SCHEMA or set(registration["tasks"]) != set(task_ids)
                or registration["demos"] != list(demos) or completion["status"] != "complete"
                or completion["episodes"] != len(task_ids) * len(demos)):
            raise ValueError("incomplete or incompatible training-only spatial labels")
        self.tasks, self.demos = set(task_ids), set(demos)

    def load(self, task, demos, frame_indices):
        if task not in self.tasks or not set(demos) <= self.demos or len(demos) != len(frame_indices):
            raise ValueError("spatial targets cross their training authority")
        labels = []
        for demo, indices in zip(demos, frame_indices, strict=True):
            with np.load(self.root / f"task{task}_demo{demo}.npz", allow_pickle=False) as stored:
                if not np.array_equal(stored["frame_indices"], indices.detach().cpu().numpy()):
                    raise ValueError("physical labels and actual teacher RGB frame positions differ")
                native, prior = (torch.from_numpy(stored[key].copy()).to(indices.device)
                                 for key in ("native_object", "prior_motion"))
            if (native.shape != (len(indices), 512) or prior.shape != (len(indices), 576)
                    or not torch.isfinite(native).all() or not torch.isfinite(prior).all()
                    or (native < 0).any() or (prior < 0).any()):
                raise ValueError("invalid spatial target layout or values")
            labels.append((native, prior))
        return tuple(labels)


def _spatial_kl(log_attention, mass, *, motion):
    if log_attention.shape != mass.shape or not torch.isfinite(log_attention).all():
        raise ValueError("actual attention and physical target patch layouts differ")
    row_mass = mass.sum(-1)
    valid = row_mass > 0
    if not valid.any():
        return log_attention.sum() * 0
    target = mass[valid] / row_mass[valid, None]
    entropy_term = torch.where(target > 0, target * target.clamp_min(torch.finfo(target.dtype).tiny).log(), 0.)
    divergence = (entropy_term - target * log_attention[valid]).sum(-1)
    weights = row_mass[valid] if motion else torch.ones_like(row_mass[valid])
    return (divergence * (weights / weights.sum())).sum()


def spatial_objective(distributions, labels):
    if not distributions or len(distributions) != len(labels):
        raise ValueError("one physical label pair is required per video")
    native, prior = [], []
    for (native_log, prior_log), (native_mass, prior_mass) in zip(distributions, labels, strict=True):
        native.append(_spatial_kl(native_log, native_mass, motion=False))
        prior.append(_spatial_kl(prior_log, prior_mass, motion=True))
    object_loss, motion_loss = torch.stack(native).mean(), torch.stack(prior).mean()
    return (object_loss + motion_loss) / 2, {
        "spatial_object_kl": float(object_loss.detach()), "spatial_motion_kl": float(motion_loss.detach()),
    }
