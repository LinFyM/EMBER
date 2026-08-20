from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from ember.functional_adaptation.fingerprint_codes import (
    FINGERPRINT_CODE_SCHEMA,
    load_functional_fingerprint_code_targets,
    uniformly_spaced_task_ids,
    whiten_functional_fingerprints,
)


def test_uniform_anchor_selection_uses_only_stable_task_order() -> None:
    assert uniformly_spaced_task_ids(tuple(range(56)), 8) == (
        0,
        8,
        16,
        24,
        31,
        39,
        47,
        55,
    )


def test_train_only_pca_whitens_train_and_transforms_held() -> None:
    generator = torch.Generator().manual_seed(17)
    train = torch.randn(56, 80, generator=generator)
    held = torch.randn(15, 80, generator=generator) + 0.2

    result = whiten_functional_fingerprints(train, held, code_width=32)

    assert result.train_codes.shape == (56, 32)
    assert result.held_codes.shape == (15, 32)
    assert torch.allclose(result.train_codes.mean(dim=0), torch.zeros(32), atol=1e-5)
    covariance = result.train_codes.T @ result.train_codes / 55
    assert torch.allclose(covariance, torch.eye(32), atol=2e-5)
    expected_held = (
        (held - result.mean) @ result.components.T / result.scales
    )
    assert torch.allclose(result.held_codes, expected_held)
    assert 0.0 < result.explained_variance_fraction <= 1.0


def test_code_authority_preserves_the_fixed_task_mapping(tmp_path: Path) -> None:
    codes = tmp_path / "fingerprint_codes.safetensors"
    save_file(
        {"train_codes": torch.randn(3, 2), "held_codes": torch.randn(2, 2)},
        str(codes),
    )
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "schema_version": FINGERPRINT_CODE_SCHEMA,
                "formal_authority": True,
                "fit_surface": "meta_train_only_pca_whitening",
                "repository": {"dirty_paths": []},
                "train_global_task_ids": [1, 2, 3],
                "held_global_task_ids": [4, 5],
                "files": {"fingerprint_codes.safetensors": codes.stat().st_size},
            }
        ),
        encoding="utf-8",
    )

    result = load_functional_fingerprint_code_targets(
        tmp_path,
        expected_train_task_ids=(1, 2, 3),
        expected_held_task_ids=(4, 5),
        code_width=2,
        device="cpu",
    )

    assert result.train_task_ids == (1, 2, 3)
    assert result.held_task_ids == (4, 5)


def test_code_authority_accepts_explicit_role_disjoint_surface(
    tmp_path: Path,
) -> None:
    codes = tmp_path / "fingerprint_codes.safetensors"
    save_file(
        {"train_codes": torch.randn(3, 2), "held_codes": torch.randn(2, 2)},
        str(codes),
    )
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "schema_version": FINGERPRINT_CODE_SCHEMA,
                "formal_authority": True,
                "fit_surface": "train24_fit_only_pca_whitening",
                "repository": {"dirty_paths": []},
                "train_global_task_ids": [1, 2, 3],
                "held_global_task_ids": [4, 5],
                "files": {"fingerprint_codes.safetensors": codes.stat().st_size},
            }
        ),
        encoding="utf-8",
    )

    result = load_functional_fingerprint_code_targets(
        tmp_path,
        expected_train_task_ids=(1, 2, 3),
        expected_held_task_ids=(4, 5),
        code_width=2,
        device="cpu",
        expected_fit_surface="train24_fit_only_pca_whitening",
    )

    assert result.train_codes.shape == (3, 2)
    assert result.held_codes.shape == (2, 2)
