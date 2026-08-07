"""Immutable PI05 model, normalization, and tokenizer setup for rollout workers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_processing import Pi05LiberoProcessor


def load_policy(
    model_path: Path,
    stats: Mapping[str, Any],
    tokenizer_path: Path,
    policy_contract: Mapping[str, Any],
) -> tuple[Any, Pi05LiberoProcessor, Any]:
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.utils.constants import ACTION, OBS_STATE

    config = PreTrainedConfig.from_pretrained(model_path)
    if not isinstance(config, PI05Config):
        raise Pi05EvaluationError("evaluation checkpoint did not resolve to PI05Config")
    config.device = "cuda:0"
    config.dtype = str(policy_contract["precision"])
    config.chunk_size = int(policy_contract["chunk_size"])
    config.n_action_steps = int(policy_contract["n_action_steps"])
    config.num_inference_steps = int(policy_contract["num_inference_steps"])
    config.input_features[OBS_STATE] = PolicyFeature(
        type=FeatureType.STATE, shape=(int(policy_contract["state_dim"]),)
    )
    config.output_features[ACTION] = PolicyFeature(
        type=FeatureType.ACTION, shape=(int(policy_contract["action_dim"]),)
    )
    policy = PI05Policy.from_pretrained(
        model_path,
        config=config,
        local_files_only=True,
        strict=True,
    ).to("cuda:0").eval()
    if hasattr(policy.model, "gradient_checkpointing_disable"):
        policy.model.gradient_checkpointing_disable()
    processor = Pi05LiberoProcessor(
        stats,
        tokenizer_path,
        config.tokenizer_max_length,
        "cuda:0",
    )
    return policy, processor, processor.unnormalize_action


def validate_worker_assets(
    contract: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], Path]:
    normalization_path = Path(contract["normalization"]["path"])
    if (
        not normalization_path.is_file()
        or normalization_path.stat().st_size
        != int(contract["normalization"]["bytes"])
    ):
        raise Pi05EvaluationError("source-only normalization changed after queue creation")
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    model_path = Path(contract["model"]["model_path"])
    frozen_policy_subdir = contract["model"].get("frozen_policy_subdir")
    if frozen_policy_subdir != model_path.name:
        raise Pi05EvaluationError("frozen source-policy subdirectory changed")
    for record in contract["model"]["model_files"]:
        relative = Path(record["path"]).relative_to(frozen_policy_subdir)
        path = model_path / relative
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
        ):
            raise Pi05EvaluationError(
                f"PI05 model file changed after queue creation: {path}"
            )
    tokenizer_path = Path(contract["tokenizer"]["path"])
    if (
        not tokenizer_path.is_file()
        or tokenizer_path.stat().st_size != int(contract["tokenizer"]["bytes"])
    ):
        raise Pi05EvaluationError("OpenPI tokenizer changed after queue creation")
    return model_path, normalization, tokenizer_path
