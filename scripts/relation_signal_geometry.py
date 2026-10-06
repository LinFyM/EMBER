"""Temporary CPU-only physical readback; no forward, GT injection, or update."""
from __future__ import annotations

import inspect
from collections.abc import Mapping

import numpy as np
import torch


MATCHER_SOURCE = (
    "/data1/user/ymdai/ember_runs/relation_grounded_writer_20261006/"
    "frozen_PEFTfix/src/ember/relation_writer/relation_loss.py"
)


def _stats(value):
    raw = np.asarray(value, dtype=np.float64).ravel()
    x = raw[np.isfinite(raw)]
    result = {"n": int(raw.size), "finite": int(x.size),
              "nonfinite": int(raw.size - x.size), "exact_zero": int((x == 0).sum())}
    for name in ("mean", "median", "min", "max", "p95", "rms", "sum"):
        result[name] = None
    if x.size:
        result.update(mean=float(x.mean()), median=float(np.median(x)), min=float(x.min()),
                      max=float(x.max()), p95=float(np.quantile(x, .95)),
                      rms=float(np.sqrt(np.mean(x * x))), sum=float(x.sum()))
    return result


def _numpy(tensor):
    return tensor.detach().to(dtype=torch.float64).numpy()


def _angle(a, b):
    """SO(3) geodesic via trace/skew atan2; supplied matrices are not projected."""
    relative = np.swapaxes(a, -1, -2) @ b
    skew = np.stack((relative[..., 2, 1] - relative[..., 1, 2],
                     relative[..., 0, 2] - relative[..., 2, 0],
                     relative[..., 1, 0] - relative[..., 0, 1]), -1) * .5
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1) * .5
    return np.arctan2(np.linalg.norm(skew, axis=-1), cosine)


def _masked(value, mask):
    return np.where(mask, value, np.nan).astype(np.float32)


def _motion_report(truth, predicted, mask, gap):
    paired = mask[1:] & mask[:-1]
    gt_delta, pred_delta = np.diff(truth, axis=0), np.diff(predicted, axis=0)
    vector = truth.ndim == 2
    gt_amp = np.linalg.norm(gt_delta, axis=-1) if vector else np.abs(gt_delta)
    pred_amp = np.linalg.norm(pred_delta, axis=-1) if vector else np.abs(pred_delta)
    error = np.linalg.norm(pred_delta - gt_delta, axis=-1) if vector else np.abs(pred_delta - gt_delta)
    static = paired & (gt_amp == 0)
    report = {"adjacent_sample_pairs": int(paired.size), "valid_pairs": int(paired.sum()),
              "no_label_pairs": int((~paired).sum()), "gt_amplitude": _stats(gt_amp[paired]),
              "pred_amplitude": _stats(pred_amp[paired]), "delta_error": _stats(error[paired]),
              "gt_per_raw_frame": _stats((gt_amp / gap)[paired]),
              "pred_per_raw_frame": _stats((pred_amp / gap)[paired]),
              "gt_exact_zero_pairs": int(static.sum()),
              "pred_on_gt_exact_zero": _stats(pred_amp[static]),
              "gt_nonzero_pairs": int((paired & (gt_amp != 0)).sum())}
    for prefix, amp in (("gt", gt_amp), ("pred", pred_amp)):
        report[prefix + "_total_path"] = (float(amp[paired].sum())
                                         if paired.any() and np.isfinite(amp[paired]).all() else None)
    denominator = report["gt_total_path"]
    report["path_ratio_pred_over_gt"] = (report["pred_total_path"] / denominator
                                        if denominator is not None and denominator > 0
                                        and report["pred_total_path"] is not None else None)
    return report, paired, gt_amp, pred_amp, error


def physical_report(pred, gt, registry_description, frame_indices, *, return_arrays=False):
    """One existing video, CPU torch mappings; optionally return NPZ-ready arrays.

    registry_description may be Registry.description() or the old label JSON
    containing registry and provenance. Caller retains arrays with
    np.savez_compressed; this function does not write files or alter its inputs.
    """
    from ember.relation_writer.relation_loss import clip_matching
    from ember.relation_writer.representation import check_physical

    matcher_source = inspect.getfile(inspect.unwrap(clip_matching))
    if matcher_source != MATCHER_SOURCE:
        raise ValueError("physical readback requires the original frozen_PEFTfix clip_matching")
    for source in (pred, gt):
        if not isinstance(source, Mapping) or any(
                not isinstance(v, torch.Tensor) or v.device.type != "cpu" for v in source.values()):
            raise ValueError("physical_report accepts CPU torch tensors only")
    shape = check_physical(pred)
    if len(shape) != 1 or check_physical(gt) != shape:
        raise ValueError("one paired whole video is required")
    n = shape[0]
    if isinstance(frame_indices, torch.Tensor) and frame_indices.device.type != "cpu":
        raise ValueError("frame_indices must be on CPU")
    frames = (frame_indices.detach().numpy() if isinstance(frame_indices, torch.Tensor)
              else np.asarray(frame_indices))
    if frames.shape != (n,) or not np.issubdtype(frames.dtype, np.integer) or np.any(np.diff(frames) <= 0):
        raise ValueError("raw frame_indices must be strictly increasing integers")
    if "frame_indices" in gt and not np.array_equal(gt["frame_indices"].numpy(), frames):
        raise ValueError("frame_indices differ from original labels")
    valid = gt["valid"].detach().numpy().astype(bool)
    if valid.shape != (n,) or not valid.any():
        raise ValueError("at least one valid original physical frame is required")
    pi, gi = clip_matching(pred, gt)
    pi, gi = pi.numpy(), gi.numpy()
    registry = registry_description.get("registry", registry_description)
    descriptions = {int(row["slot"]): dict(row) for row in registry["entities"]}
    descriptions[0] = {"slot": 0, "body": "hand", "semantic": "hand", "origin": "hand",
                       "p_source": "obs/ee_pos (grip site)", "R_source": "obs/ee_ori (eef body)"}
    if any(int(slot) not in descriptions for slot in gi):
        raise ValueError("matched GT slot lacks its original registry description")
    assignment = np.full(pred["presence"].shape[1], -1, np.int64)
    assignment[pi] = gi
    fields = {name: _numpy(pred[name])[:, pi] for name in ("p", "R", "semantic", "joint_q", "joint_type")}
    truth = {name: _numpy(gt[name])[:, gi] for name in fields}
    presence_gt = _numpy(gt["presence"])
    masks = {name: gt[name + "_mask"].numpy().astype(bool)[:, gi]
             & valid[:, None] & (presence_gt[:, gi] > .5)
             for name in ("p", "R", "semantic", "joint_type")}
    q_mask = gt["joint_q_mask"].numpy().astype(bool)[:, gi] & valid[:, None, None]
    q_mask &= presence_gt[:, gi, None] > .5
    gap = np.diff(frames).astype(np.float64)
    p_error_vector = fields["p"] - truth["p"]
    p_error = np.linalg.norm(p_error_vector, axis=-1)
    r_error = _angle(truth["R"], fields["R"])
    r_fro = np.linalg.norm((fields["R"] - truth["R"]).reshape(n, len(gi), 9), axis=-1)
    semantic_pred = fields["semantic"] / np.maximum(np.linalg.norm(fields["semantic"], axis=-1, keepdims=True), 1e-6)
    semantic_gt = truth["semantic"] / np.maximum(np.linalg.norm(truth["semantic"], axis=-1, keepdims=True), 1e-6)
    semantic_cos = np.clip((semantic_pred * semantic_gt).sum(-1), -1, 1)
    probability = _numpy(pred["presence"])
    relative_p_error = np.full_like(p_error, np.nan)
    arrays = {"frame_indices": frames.astype(np.int64), "valid": valid, "pred_slots": pi,
              "gt_slots": gi, "assignment": assignment, "raw_frame_gaps": gap.astype(np.int64),
              "p_pred_m": fields["p"].astype(np.float32), "p_gt_m": truth["p"].astype(np.float32),
              "R_pred": fields["R"].astype(np.float32), "R_gt": truth["R"].astype(np.float32),
              "p_mask": masks["p"], "R_mask": masks["R"], "semantic_mask": masks["semantic"],
              "p_error_m": _masked(p_error, masks["p"]),
              "p_error_vector_m": _masked(p_error_vector, masks["p"][..., None]),
              "R_error_rad": _masked(r_error, masks["R"]), "R_error_fro": _masked(r_fro, masks["R"]),
              "semantic_cos": _masked(semantic_cos, masks["semantic"]),
              "joint_q_pred": fields["joint_q"].astype(np.float32),
              "joint_q_gt": truth["joint_q"].astype(np.float32), "joint_q_mask": q_mask,
              "joint_q_signed_error": _masked(fields["joint_q"] - truth["joint_q"], q_mask),
              "joint_type_pred_probability": fields["joint_type"].astype(np.float32),
              "joint_type_gt": truth["joint_type"].astype(np.float32), "joint_type_mask": masks["joint_type"]}
    report = {"schema": "ember_relation_signal_physical_readback_v1", "status": "complete",
              "statistics_dtype": "float64 (NPZ floats float32); model tensors unchanged",
              "pred_input_dtypes": {k: str(v.dtype) for k, v in pred.items()},
              "matcher_source": matcher_source, "matching": "one whole-video Hungarian; hand fixed 0",
              "source": {k: v for k, v in registry_description.items() if k not in ("registry", "frames")},
              "registry": registry, "assignment_pred_to_gt": assignment.tolist(),
              "frames": {"rgb": n, "valid_geometry": int(valid.sum()), "no_geometry": int((~valid).sum()),
                         "raw_first": int(frames[0]), "raw_last": int(frames[-1])},
              "units": {"p": "m", "R_angle": "rad and degree", "R_fro": "dimensionless",
                        "joint_q_channels": ["slide m", "hinge rad"], "hand_q": "raw two finger qpos, m",
                        "rate": "per raw frame, not per second"},
              "rotation_formula": "atan2(norm(vee(Ra.T@Rb-(Ra.T@Rb).T))/2,(trace(Ra.T@Rb)-1)/2); no projection",
              "boundaries": ["GT matching/masks are statistics only; no GT enters model conditions",
                             "body positions are MuJoCo body origins; moving links retain original registry meaning",
                             "static means exact zero labelled adjacent movement; no grasp/phase threshold",
                             "motion only uses adjacent sampled frames with both labels; no gap bridging"],
              "entities": []}
    for j, (pred_slot, gt_slot) in enumerate(zip(pi, gi, strict=True)):
        expected = valid & (presence_gt[:, gt_slot] > .5)
        row = {"pred_slot": int(pred_slot), "gt_slot": int(gt_slot), **descriptions[int(gt_slot)],
               "eligible_frames": int(expected.sum()), "field_counts": {},
               "p_absolute_error_m": _stats(p_error[masks["p"][:, j], j]),
               "R_error_rad": _stats(r_error[masks["R"][:, j], j]),
               "R_error_degree": _stats(np.degrees(r_error[masks["R"][:, j], j])),
               "R_error_fro": _stats(r_fro[masks["R"][:, j], j]),
               "semantic_cos": _stats(semantic_cos[masks["semantic"][:, j], j])}
        for name, mask in masks.items():
            row["field_counts"][name] = {"valid": int(mask[:, j].sum()),
                                         "missing_on_present_valid": int((expected & ~mask[:, j]).sum())}
        labelled = np.flatnonzero(masks["p"][:, j])
        first = int(labelled[0]) if labelled.size else None
        row["relative_p_first_raw_frame"] = int(frames[first]) if first is not None else None
        row["relative_first_gt_amplitude_m"] = _stats([])
        row["relative_first_pred_amplitude_m"] = _stats([])
        if first is not None:
            pred_displacement = fields["p"][:, j] - fields["p"][first, j]
            gt_displacement = truth["p"][:, j] - truth["p"][first, j]
            relative_p_error[:, j] = np.linalg.norm(pred_displacement - gt_displacement, axis=-1)
            row["relative_first_gt_amplitude_m"] = _stats(np.linalg.norm(gt_displacement[labelled], axis=-1))
            row["relative_first_pred_amplitude_m"] = _stats(np.linalg.norm(pred_displacement[labelled], axis=-1))
        row["relative_first_p_error_m"] = _stats(relative_p_error[masks["p"][:, j], j])
        motion, pair, gt_amp, pred_amp, error = _motion_report(truth["p"][:, j], fields["p"][:, j], masks["p"][:, j], gap)
        row["p_motion_m"] = motion
        arrays.setdefault("p_step_mask", np.zeros((n - 1, len(gi)), bool))[:, j] = pair
        for name, value in (("p_step_gt_m", gt_amp), ("p_step_pred_m", pred_amp), ("p_step_error_m", error)):
            arrays.setdefault(name, np.full((n - 1, len(gi)), np.nan, np.float32))[:, j] = _masked(value, pair)
        r_pair = masks["R"][1:, j] & masks["R"][:-1, j]
        arrays.setdefault("R_step_mask", np.zeros((n - 1, len(gi)), bool))[:, j] = r_pair
        gt_r = _angle(truth["R"][:-1, j], truth["R"][1:, j])
        pred_r = _angle(fields["R"][:-1, j], fields["R"][1:, j])
        gt_rf = np.linalg.norm(np.diff(truth["R"][:, j], axis=0).reshape(n - 1, 9), axis=-1)
        pred_rf = np.linalg.norm(np.diff(fields["R"][:, j], axis=0).reshape(n - 1, 9), axis=-1)
        row["R_motion"] = {"valid_pairs": int(r_pair.sum()), "no_label_pairs": int((~r_pair).sum()),
                           "gt_rad": _stats(gt_r[r_pair]), "pred_rad": _stats(pred_r[r_pair]),
                           "gt_rad_per_raw_frame": _stats((gt_r / gap)[r_pair]),
                           "pred_rad_per_raw_frame": _stats((pred_r / gap)[r_pair]),
                           "gt_fro": _stats(gt_rf[r_pair]), "pred_fro": _stats(pred_rf[r_pair]),
                           "gt_exact_zero_fro_pairs": int((r_pair & (gt_rf == 0)).sum()),
                           "pred_rad_on_gt_exact_zero_fro": _stats(pred_r[r_pair & (gt_rf == 0)])}
        for name, value in (("R_step_gt_rad", gt_r), ("R_step_pred_rad", pred_r),
                            ("R_step_gt_fro", gt_rf), ("R_step_pred_fro", pred_rf)):
            arrays.setdefault(name, np.full((n - 1, len(gi)), np.nan, np.float32))[:, j] = _masked(value, r_pair)
        type_mask = masks["joint_type"][:, j]
        types = truth["joint_type"][:, j].argmax(-1)
        row["joint_type"] = {"missing_on_present_valid": int((expected & ~type_mask).sum()),
                             "none_slide_hinge_counts": [int((type_mask & (types == k)).sum()) for k in range(3)],
                             "pred_none_slide_hinge_probability": [_stats(fields["joint_type"][type_mask, j, k]) for k in range(3)]}
        row["joint_q"] = []
        for channel, unit in enumerate(("m", "rad")):
            mask = q_mask[:, j, channel]
            typed = type_mask & (types == channel + 1)
            difference = fields["joint_q"][:, j, channel] - truth["joint_q"][:, j, channel]
            motion, _, _, _, _ = _motion_report(truth["joint_q"][:, j, channel], fields["joint_q"][:, j, channel], mask, gap)
            row["joint_q"].append({"channel": channel, "type": ("slide", "hinge")[channel], "unit": unit,
                                   "valid_q_frames": int(mask.sum()), "expected_by_labelled_type": int(typed.sum()),
                                   "missing_q_on_labelled_type": int((typed & ~mask).sum()),
                                   "q_without_labelled_type": int((mask & ~type_mask).sum()),
                                   "signed_error": _stats(difference[mask]), "absolute_error": _stats(np.abs(difference[mask])),
                                   "gt_q": _stats(truth["joint_q"][mask, j, channel]),
                                   "pred_q": _stats(fields["joint_q"][mask, j, channel]), "motion": motion})
        row["presence"] = {"actual_target": _stats(presence_gt[valid, gt_slot]),
                           "pred_probability": _stats(probability[valid, pred_slot]),
                           "probability_error": _stats(probability[valid, pred_slot] - presence_gt[valid, gt_slot])}
        report["entities"].append(row)
    arrays["p_relative_first_error_m"] = _masked(relative_p_error, masks["p"])
    hand_mask = gt["hand_q_mask"].numpy().astype(bool) & valid[:, None] & (presence_gt[:, 0, None] > .5)
    hand_pred, hand_gt = _numpy(pred["hand_q"]), _numpy(gt["hand_q"])
    report["hand_raw_qpos"] = []
    for finger in range(2):
        mask = hand_mask[:, finger]
        difference = hand_pred[:, finger] - hand_gt[:, finger]
        report["hand_raw_qpos"].append({"finger": finger + 1, "unit": "m", "valid_frames": int(mask.sum()),
                                       "missing_on_present_valid": int((valid & (presence_gt[:, 0] > .5) & ~mask).sum()),
                                       "gt_qpos": _stats(hand_gt[mask, finger]), "pred_qpos": _stats(hand_pred[mask, finger]),
                                       "signed_error": _stats(difference[mask]), "absolute_error": _stats(np.abs(difference[mask]))})
    arrays.update(hand_q_pred_m=hand_pred.astype(np.float32), hand_q_gt_m=hand_gt.astype(np.float32),
                  hand_q_mask=hand_mask, hand_q_signed_error_m=_masked(hand_pred - hand_gt, hand_mask))
    target = np.zeros_like(probability)
    target[:, pi] = presence_gt[:, gi]
    arrays.update(presence_pred_probability=probability.astype(np.float32), presence_actual_target=target.astype(np.float32))
    report["presence"] = {}
    clipped = np.clip(probability[valid], 1e-6, 1 - 1e-6)
    labels = target[valid]
    bce = -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
    balanced = [bce[mask].mean() for mask in (labels > .5, labels <= .5) if mask.any()]
    balanced_bce = float(np.mean(balanced))
    report["presence"]["original_balanced_bce"] = balanced_bce if np.isfinite(balanced_bce) else None
    report["presence"]["original_balanced_bce_nonfinite"] = not bool(np.isfinite(balanced_bce))
    report["presence"]["total_pred_mass_per_valid_frame"] = _stats(probability[valid].sum(-1))
    partitions = {"hand": np.array([0]), "matched_real_entities": pi[gi != 0],
                  "empty_anonymous_slots": np.flatnonzero(assignment < 0)}
    for name, slots in partitions.items():
        values, labels = probability[valid][:, slots], target[valid][:, slots]
        safe = np.clip(values, 1e-6, 1 - 1e-6)
        bce = -(labels * np.log(safe) + (1 - labels) * np.log1p(-safe))
        report["presence"][name] = {"pred_slots": slots.tolist(), "slot_count": int(slots.size),
                                   "valid_frames": int(valid.sum()), "target_positive_cells": int((labels > .5).sum()),
                                   "probability": _stats(values), "total_mass_per_frame": _stats(values.sum(-1)),
                                   "target_mass_per_frame": _stats(labels.sum(-1)),
                                   "brier": _stats((values - labels) ** 2), "bce": _stats(bce),
                                   "positive_bce": _stats(bce[labels > .5]), "negative_bce": _stats(bce[labels <= .5])}
    return (report, arrays) if return_arrays else report
