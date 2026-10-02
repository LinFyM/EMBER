"""Control content and its contract inside the canonical conditional Writer.

Gamma reads only bare-source H/mu. Its ordered 35-vector gates both existing
Value latents; it never becomes an execution adapter or a LoRA parameter label.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

MODE = "control_calibrated_read_write"
SEEN_MODE = MODE + "_seen"
TASK = "control_calibrated_read_write_20261003"
ROOT = Path("/data1/user/ymdai/ember_runs") / TASK
SPEC_NAME = "control_calibrated_read_write_spec.json"
MODES = (MODE, SEEN_MODE)
CONTRACT = {
    "bare_native": "frozen_Source1000_no_LoRA_statefree_RGB_exact_language_probe1729_tau1",
    "feature_cache": "H0_mu0_indices_only_no_action_state_reward",
    "gamma": "fresh_seed7_shared_QKV1024_256_Wo256_four_heads_RMS_eps1e-6_W1_768_1024_GELU_W2_1024_7",
    "gamma_output_init": "W2_and_bias_zero",
    "control": "mu0_departure_plus_Gamma_H0_departure_H0_arrival_first5_ordered35",
    "value_gates": "per_target_1_plus_Ua_q_and_1_plus_Ub_q_35_256_nobias_zero",
    "short_gap": "keep_original_c_d_write_gate_identity_no_label",
    "auxiliary": "legal_offset1_5x7_interval_mean_condition_mean_task1_over4_once",
    "auxiliary_credit": "Gamma_only", "full_FM_credit": "all_original_Writer_Gamma_U_and_public_native_beta",
    "coefficient": 1.0,
}


def expected_spec(parent):
    joint = {**parent["joint"], "internal_mode": MODE,
             "initialization": "complete_original_fresh_modules_then_independent_seed7_Gamma_U_zero"}
    return {**parent, "task": TASK, "design": "docs/designs/control_calibrated_read_write_design.md",
            "run_root": str(ROOT), "joint": joint,
            "operator": {**parent["operator"], "additional_loss": True, "control_calibration": CONTRACT},
            "execution": {**parent["execution"], "modes": [MODE],
                          "checkpoints": [0, 90, 180, 270, 360, 450],
                          "profile_updates_max": 3},
            "budget": {"new_gpu_hours_hard": 32, "peak_new_gib": 80,
                       "expected_wall_hours": [6, 10]}}


def bank_path(mode, checkpoint):
    if (checkpoint is None or checkpoint.name != "macro_00000450"
            or not checkpoint.resolve().is_relative_to(ROOT / MODE / "train/attempts")):
        raise ValueError("calibrated Writer readers require the unique owned complete450 endpoint")
    return ROOT / mode / "banks/450/manifest.json"


class ControlCalibration(nn.Module):
    """Same mathematical P residual function, with fresh parameters."""
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(1024, 256, bias=False)
        self.k = nn.Linear(1024, 256, bias=False)
        self.v = nn.Linear(1024, 256, bias=False)
        self.o = nn.Linear(256, 256, bias=False)
        self.w1 = nn.Linear(768, 1024)
        self.w2 = nn.Linear(1024, 7)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    @staticmethod
    def rms(x):
        x = x.float()
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

    @staticmethod
    def attention(q, k, v):
        def heads(x):
            return x.reshape(-1, 50, 4, 64).transpose(1, 2)
        out = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), dropout_p=0.)
        return out.transpose(1, 2).reshape(q.shape)

    def forward(self, departure, arrival):
        x, z = self.rms(departure), self.rms(arrival)
        q = self.q(x)
        c0 = self.o(self.attention(q, self.k(x), self.v(x)))
        c1 = self.o(self.attention(q, self.k(z), self.v(z)))
        return self.w2(F.gelu(self.w1(torch.cat((q, c0, c1), -1))))

    def controls(self, bare, frame_indices):
        h, mu = bare["H0"], bare["mu0"]
        if (h.shape != (len(frame_indices), 50, 1024) or mu.shape != (len(h), 5, 7)
                or h.requires_grad or mu.requires_grad
                or not torch.equal(bare["frame_indices"], frame_indices)):
            raise ValueError("control features must be aligned frozen bare-source H0/mu0")
        valid = frame_indices[1:] - frame_indices[:-1] == 5
        q = mu[:-1].float() + self(h[:-1], h[1:])[:, :5].float()
        # Zeroing this control leaves identity gates for true short final gaps.
        return q * valid[:, None, None], valid


def append_modules(writer):
    """Finish all old initialization before the independent new CPU RNG scope."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(7)
        writer.gamma = ControlCalibration()
        for unit in writer.conditional_targets:
            unit.ua = nn.Linear(35, 256, bias=False)
            unit.ub = nn.Linear(35, 256, bias=False)
            nn.init.zeros_(unit.ua.weight)
            nn.init.zeros_(unit.ub.weight)


def auxiliary_loss(q, valid, target, *, condition_weight=.25):
    if (target.shape != (int(valid.sum()), 5, 7) or target.requires_grad
            or not valid.any()):
        raise ValueError("auxiliary labels must contain only complete gap5 offset1 intervals")
    return (q[valid].float() - target.float()).square().mean() * condition_weight


def training_labels(data, runtime, event, indices):
    """Only the authorized train-side loss may open teacher actions."""
    import h5py
    import numpy as np
    if data.role != "train" or data.queries is None:
        raise ValueError("control labels are inaccessible to deployment/held consumers")
    task, demo = event["task"], event["teacher_demo"]
    starts = indices[:-1][indices[1:] - indices[:-1] == 5].cpu().tolist()
    with h5py.File(data.tasks[task].authority.path, "r") as f:
        actions = f[f"data/demo_{demo}/actions"]
        if not starts or any(p + 6 > len(actions) for p in starts):
            raise ValueError("complete control label must retain offset1 and all five actions")
        raw = np.stack([np.asarray(actions[p + 1:p + 6], dtype=np.float32) for p in starts])
    target = runtime.processor.normalize_action(torch.from_numpy(raw)).detach().to(runtime.device)
    path = Path(data.spec["run_root"]) / "control_labels" / f"task{task:03d}_demo{demo:02d}.pt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"target": target.cpu(), "departure_frames": torch.tensor(starts),
                    "action_indices": torch.tensor(starts)[:, None] + torch.arange(1, 6),
                    "mask": torch.ones(len(starts), 5, dtype=torch.bool),
                    "normalization": data.spec["source"]["normalization"],
                    "labels_only": True, "deployment_feature_cache": False}, path)
    return target
