import copy
import json
import subprocess

import pytest

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_results import paired_success_comparison
from scripts import compare_pi05_results


def panels():
    common = {"suite": "libero_goal", "task_id": 0, "language": "open drawer",
              "env_seed": 7, "policy_seed_root": 7, "split_role": "train", "policy_noise_seeds": [8, 9, 10]}
    before = {"rows": [{**common, "init_state_id": i, "success": i in (0, 1)} for i in range(3)]}
    after = copy.deepcopy(before)
    after["rows"][0]["success"] = False
    after["rows"][2]["success"] = True
    after["rows"][2]["policy_noise_seeds"] = [8, 9]  # Earlier success may shorten rollout.
    return before, after


def test_success_exchange_is_distinct_from_equal_total_scores():
    result = paired_success_comparison(*panels())
    assert (result["retained"], result["gained"], result["lost"]) == (1, 1, 1)
    assert result["churn_count"] == 2
    assert result["success_set_jaccard"] == pytest.approx(1 / 3)
    assert result["candidate_breadth"] == 1


@pytest.mark.parametrize("change", ["rng", "rows", "language"])
def test_unpaired_evidence_is_rejected(change):
    before, after = panels()
    if change == "rng":
        after["rows"][0]["policy_noise_seeds"][0] = 10
    elif change == "rows":
        after["rows"].pop()
    else:
        after["rows"][0]["language"] = "close drawer"
    with pytest.raises(Pi05EvaluationError):
        paired_success_comparison(before, after)


def test_removed_worktree_normalizer_is_read_from_recorded_commit(tmp_path, monkeypatch):
    relative = "configs/pi05_source_corpus_v1/source_normalization.json"
    repo = tmp_path / "repo"
    path = repo / relative
    path.parent.mkdir(parents=True)
    payload = {"mean": [1.0], "std": [2.0]}
    path.write_text(json.dumps(payload))
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo).decode().strip()
    git("init", "-q")
    git("add", relative)
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "Record normalizer")
    commit = git("rev-parse", "HEAD")
    contract = {"git": {"commit": commit, "dirty_paths": []},
                "normalization": {"path": str(tmp_path / "removed-worktree" / relative), "bytes": path.stat().st_size}}
    # The current checkout is not the historical evidence, even if its config
    # has the same filename. Recover the original version without a worktree.
    path.write_text(json.dumps({"mean": [9.0], "std": [2.0]}))
    monkeypatch.setattr(compare_pi05_results, "__file__", str(repo / "scripts/compare_pi05_results.py"))
    restored, provenance = compare_pi05_results.normalization_evidence(contract)
    assert restored == payload
    assert provenance["git_blob"] == f"{commit}:{relative}"
    contract["normalization"]["bytes"] += 1
    with pytest.raises(ValueError, match="size differs"):
        compare_pi05_results.normalization_evidence(contract)
    contract["git"]["dirty_paths"] = [relative]
    with pytest.raises(ValueError, match="clean Git provenance"):
        compare_pi05_results.normalization_evidence(contract)
