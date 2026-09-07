"""Fixed normalized action exploration for the paired training diagnostic."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping, Sequence

from ember.pi05_assets import Pi05EvaluationError


EXPLORATION_SCHEMA = "ember_horizon_train_exploration_v1"
EXPLORATION_SUBSTREAM = 0x5349474D41
DIAGNOSTIC_STATES = tuple(range(32, 37))


def exploration_metadata(*, enabled: bool, flow_seed: int) -> dict[str, Any]:
    return {
        "schema_version": EXPLORATION_SCHEMA,
        "condition": "J_Sigma" if enabled else "J0", "enabled": enabled,
        "action_space": "normalized_chunk_before_canonical_source_postprocess",
        "executed_steps": 5, "action_dim": 7, "flatten_order": "time_major",
        "covariance": {"shape": [35, 35], "formula": "C_rho_kron_diag_std_squared",
                       "temporal_rho": 0.8, "action_std": [0.05] * 6 + [0.10]},
        "sampling": "standard_normal35_times_cholesky_transpose",
        "seed_root": int(flow_seed) ^ EXPLORATION_SUBSTREAM,
        "seed_schedule": "policy_noise_seed(seed_root,suite,task_id,init_state_id,replan_index)",
        "generator": "independent_cpu_torch_generator_per_decision",
        "flow_noise_preserved": True, "additional_clipping": False,
        "gripper_override": False, "training_gradient_use": False,
        "checkpoint_qualification_use": False,
    }


def _diagnostic_scope(contract: Mapping[str, Any]) -> bool:
    return (contract.get("role") == "development_train" and contract.get("mode") == "screen"
            and bool(contract.get("tasks")) and all(
                task.get("split_role") == "train" and tuple(task.get("init_state_ids", ())) == DIAGNOSTIC_STATES
                for task in contract["tasks"])
            and contract.get("diagnostic_occupancy_capture") is None
            and contract.get("diagnostic_stage_predicates") is None)


def build_exploration_contract(contract: Mapping[str, Any], *, enabled: bool) -> dict[str, Any] | None:
    if not _diagnostic_scope(contract):
        if enabled:
            raise Pi05EvaluationError("exploration Sigma requires development_train screen states32..36")
        return None
    return exploration_metadata(enabled=enabled, flow_seed=int(contract["rng"]["inference_seed"]))


def validate_exploration_contract(contract: Mapping[str, Any]) -> None:
    metadata = contract.get("diagnostic_exploration")
    if metadata is None:
        return
    policy = contract["policy"]
    if (not isinstance(metadata, Mapping) or type(metadata.get("enabled")) is not bool
            or not _diagnostic_scope(contract)
            or any(policy.get(key) != value for key, value in {
                "replan_steps": 5, "action_dim": 7, "chunk_size": 50, "num_inference_steps": 10}.items())
            or metadata != exploration_metadata(enabled=metadata["enabled"],
                                                flow_seed=int(contract["rng"]["inference_seed"]))):
        raise Pi05EvaluationError("training exploration scope, Sigma, or independent RNG contract changed")


def exploration_noise_seed(metadata, *, suite: str, task_id: int, state_id: int, replan: int) -> int:
    from ember.pi05_eval_contract import policy_noise_seed

    return policy_noise_seed(int(metadata["seed_root"]), suite, task_id, state_id, replan)


@lru_cache(maxsize=1)
def _cpu_cholesky():
    import torch
    from ember.writer.rl_math import exploration_covariance

    return torch.linalg.cholesky(exploration_covariance(device="cpu", dtype=torch.float32))


def add_exploration_noise(chunks, slots: Sequence[dict[str, Any]], *, task, contract):
    """Perturb only the normalized 5x7 execution block; J0 is an exact passthrough."""
    import torch

    metadata = contract.get("diagnostic_exploration")
    if metadata is None:
        return chunks
    seeds = [exploration_noise_seed(metadata, suite=str(task["suite"]), task_id=int(task["task_id"]),
             state_id=int(slot["init_state_id"]), replan=int(slot["replan_index"])) for slot in slots]
    for slot, seed in zip(slots, seeds, strict=True):
        slot.setdefault("exploration_noise_seeds", []).append(seed)
    if not metadata["enabled"]:
        return chunks
    if chunks.ndim != 3 or chunks.shape[0] != len(slots) or chunks.shape[1] < 5 or chunks.shape[2] < 7:
        raise Pi05EvaluationError("exploration requires a batched normalized action chunk containing 5x7 values")
    standard = torch.stack([torch.randn(35, dtype=torch.float32, device="cpu",
                                       generator=torch.Generator(device="cpu").manual_seed(seed))
                            for seed in seeds])
    noise = (standard @ _cpu_cholesky().T).reshape(len(slots), 5, 7)
    result = chunks.clone()
    result[:, :5, :7] += noise.to(device=chunks.device, dtype=chunks.dtype)
    return result


def episode_exploration_fields(contract: Mapping[str, Any], slot: Mapping[str, Any]) -> dict[str, Any]:
    metadata = contract.get("diagnostic_exploration")
    return {} if metadata is None else {"diagnostic_exploration": {
        **metadata, "noise_seeds": list(slot.get("exploration_noise_seeds", ()))}}


def validate_episode_exploration(contract, row, *, replans: int) -> bool:
    metadata = contract.get("diagnostic_exploration")
    if metadata is None:
        return row.get("diagnostic_exploration") is None
    seeds = [exploration_noise_seed(metadata, suite=str(row["suite"]), task_id=int(row["task_id"]),
             state_id=int(row["init_state_id"]), replan=index) for index in range(replans)]
    return row.get("diagnostic_exploration") == {**metadata, "noise_seeds": seeds}


def _same_sigma_contract(left, right) -> bool:
    ignored = {"condition", "enabled"}
    return {key: value for key, value in left.items() if key not in ignored} == {
        key: value for key, value in right.items() if key not in ignored}


def validate_exploration_comparison(left, right, *, allow_exploration_pair: bool) -> None:
    """Only the explicit diagnostic comparison may vary J0 versus J_Sigma."""
    for contract in (left, right):
        validate_exploration_contract(contract)
    a, b = left.get("diagnostic_exploration"), right.get("diagnostic_exploration")
    if allow_exploration_pair:
        if (a is None or b is None or {a["condition"], b["condition"]} != {"J0", "J_Sigma"}
                or not _same_sigma_contract(a, b) or left.get("adapter") != right.get("adapter")
                or left["tasks"] != right["tasks"]):
            raise Pi05EvaluationError("exploration pairing requires declared J0/J_Sigma with one adapter and exact task states")
    elif a != b and (a is not None and a["enabled"] or b is not None and b["enabled"]
                     or a is not None and b is not None):
        raise Pi05EvaluationError("exploration condition changed; use the explicit exploration-pair comparison")


def validate_exploration_row_pair(left, right, *, allow_exploration_pair: bool) -> None:
    metadata = []
    for row in (left, right):
        evidence = row.get("diagnostic_exploration")
        if evidence is None:
            metadata.append(None)
            continue
        if not isinstance(evidence, Mapping) or type(evidence.get("enabled")) is not bool:
            raise Pi05EvaluationError("paired exploration evidence is invalid")
        expected = exploration_metadata(enabled=evidence["enabled"], flow_seed=int(row["policy_seed_root"]))
        if (row["split_role"] != "train" or row["init_state_id"] not in DIAGNOSTIC_STATES
                or not validate_episode_exploration({"diagnostic_exploration": expected}, row,
                                                   replans=len(row["policy_noise_seeds"]))):
            raise Pi05EvaluationError("paired exploration Sigma or seed trace changed")
        metadata.append(expected)
    a, b = metadata
    if allow_exploration_pair:
        fields = ("horizon_writer_lora", "policy_adapter_sha256", "static_task_lora", "task_expert")
        if (a is None or b is None or {a["condition"], b["condition"]} != {"J0", "J_Sigma"}
                or not _same_sigma_contract(a, b) or any(left.get(key) != right.get(key) for key in fields)):
            raise Pi05EvaluationError("exploration row pair must preserve the same checkpoint and teacher condition")
    elif a != b and (a is not None and a["enabled"] or b is not None and b["enabled"]
                     or a is not None and b is not None):
        raise Pi05EvaluationError("exploration condition changed; use the explicit exploration-pair comparison")
