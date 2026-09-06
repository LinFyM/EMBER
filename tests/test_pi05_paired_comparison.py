import copy

import pytest

from ember.pi05_assets import Pi05EvaluationError
from ember.pi05_eval_results import paired_success_comparison


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
