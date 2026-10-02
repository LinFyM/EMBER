"""Last native attention's fixed-query response to the next real prefix."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

TASK = "native_prefix_change_value_20261002"
ROOT = Path("/data1/user/ymdai/ember_runs") / TASK
MODE = "native_prefix_change"
SEEN_MODE = MODE + "_seen"
MODES = (MODE, SEEN_MODE)
SPEC_NAME = "native_prefix_change_spec.json"
JOINT = {"loss_variant": "full", "full_loss_coefficient": 1.0,
         "public_loss_coefficient": 0.0, "public_credit": "none_no_public_objective",
         "query_reuse": "112_cross_episode_query_action_tau_noise", "internal_mode": MODE,
         "initialization": "fresh_original_T_then_independent_seed7_zero_E"}
FEATURE = {"block": 17, "native_reads": 1, "query": "actual_rotated_current_suffix",
           "replacement": "next_real_whole_prefix_KV_only",
           "suffix": "current_native_KV", "attention": "native_RoPE_mask_GQA_scaling_dropout0",
           "output": "frozen_action_o_proj_with_bias_before_residual_MLP",
           "difference": "token_RMS_eps1e-6_u_cf_minus_u", "shape": ["N-1", 50, 1024],
           "E": [1024, 256], "E_bias": False, "E_initialization": "zero_after_all_original_T_modules",
           "added_trainable_parameters": 9961472, "prefix_credit": "frozen",
           "suffix_credit": "live_Q_K_V_to_public_beta", "chunk_boundary_transitions": True,
           "last_real_frame": True, "horizon_average": False}


def expected_spec(base: dict) -> dict:
    return {**base, "task": TASK, "design": "docs/designs/native_prefix_change_value_design.md",
            "run_root": str(ROOT), "joint": JOINT,
            "operator": {**base["operator"], "native_prefix_change": FEATURE},
            "execution": {**base["execution"], "modes": [MODE]},
            "evaluation": {**base["evaluation"], "bank_macros": [270, 450],
                           "seen_macro": 450, "only_selected_checkpoint": None},
            "budget": {"new_gpu_hours_hard": 14, "peak_new_gib": 48,
                       "expected_wall_hours": [3, 5], "report_gpu_hours": 12}}


def rms(value):
    value = value.float()
    return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)


class NativeAttentionCapture:
    """Observe actual rotated Q/K/V and actual expert projection, without a forward."""
    def __init__(self, bridge):
        self.attention = bridge.paligemma.model.language_model.layers[17].self_attn
        self.projection = bridge.gemma_expert.model.layers[17].self_attn.o_proj
        self.fields = {}

    def attention_call(self, module, query, key, value, attention_mask, scaling, **_):
        if module is not self.attention:
            return
        if self.fields or query.shape[1:] != (8, key.shape[2], 256) or key.shape[1] != 1:
            raise ValueError("last native attention capture repeated or changed geometry")
        # Prefix never consumes action-side beta. Current suffix remains differentiable.
        self.fields = {"q": query[:, :, -50:], "kp": key[:, :, :-50].detach(),
                       "vp": value[:, :, :-50].detach(), "ks": key[:, :, -50:],
                       "vs": value[:, :, -50:], "mask": attention_mask[:1, :, -50:]}
        if scaling != self.attention.scaling:
            raise ValueError("native attention scaling changed")

    def projection_call(self, _module, _arguments, result):
        if "u" in self.fields or not self.fields or result.shape[1:] != (50, 1024):
            raise ValueError("actual action attention projection capture incomplete")
        self.fields["u"] = result

    def outputs(self):
        if set(self.fields) != {"q", "kp", "vp", "ks", "vs", "mask", "u"}:
            raise ValueError("last native attention fields missing")
        return tuple(self.fields[key] for key in ("q", "kp", "vp", "ks", "vs", "u", "mask"))


def response_difference(chunks, attention, projection, *, frame_chunk):
    """Concatenate ordered fields before pairing, including every chunk boundary."""
    q, kp, vp, ks, vs, u = (torch.cat([chunk[index] for chunk in chunks], 0)
                             for index in range(6))
    mask = chunks[0][6]
    if len(q) < 2 or projection.weight.requires_grad:
        raise ValueError("prefix response needs ordered real frames and frozen projection")
    changes = []
    for start in range(0, len(q) - 1, frame_chunk):
        stop = min(start + frame_chunk, len(q) - 1)
        key = torch.cat((kp[start + 1:stop + 1], ks[start:stop]), 2)
        value = torch.cat((vp[start + 1:stop + 1], vs[start:stop]), 2)
        heads = F.scaled_dot_product_attention(
            q[start:stop], key.repeat_interleave(attention.num_key_value_groups, 1),
            value.repeat_interleave(attention.num_key_value_groups, 1), attn_mask=mask,
            dropout_p=0.0, is_causal=False, scale=attention.scaling)
        merged = heads.transpose(1, 2).contiguous().flatten(2).to(projection.weight.dtype)
        u_cf = projection(merged)
        changes.append(rms(u_cf) - rms(u[start:stop]))
    return torch.cat(changes, 0)


def passive_condition(condition, mode):
    """Eight preregistered seen mappings; never infer fixed demo ordinals."""
    from . import scope
    from ember.writer.materialization import planned_episodes
    if mode != SEEN_MODE or condition["global_task_id"] not in (0, 12, 20, 32):
        return False
    episodes = planned_episodes(scope.selection(), condition["global_task_id"])
    return any(condition["condition_id"] == row["condition_id"]
               for row in episodes if row["init_state_id"] in (32, 33))
