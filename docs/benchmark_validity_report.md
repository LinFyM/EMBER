# EMBER Gate -1 Benchmark and Specification Validity Report

Status: **specification-only split permanently resealed; Gate -1 remains in progress**
Evidence cutoff: 2026-07-18

## Technical summary

The reproducible SmolVLA/LIBERO path and pinned LIBERO-90 specification
authority are mechanically valid. A two-task overlap-trained pilot also shows a
large descriptive dependence on the policy-visible prompt. Those results are
substrate evidence, not a LIBERO-90 Gate -1 pass.

The original predeclared 60/15/15 split failed its stronger source-atom
coverage contract before any LIBERO-90 policy training or evaluation result was
read. That failure is preserved below and in Git commit `5897406`. Under the
owner-authorized one-time recovery, EMBER used only the pinned English task
specifications and scene identities to build a strict role table for all 90
tasks and deterministically search a replacement split. BDDL goals, actions,
proprioception, rewards, terminal flags, normalization values, and policy
outcomes were not inputs to the redesign.

The replacement is now permanently sealed in `configs/phase0.toml`; the full
factor table, rejected prior split, search contract, diagnostics, and active
task IDs are in `configs/libero90_split_reseal.json` with SHA256
`9f5bc62e15e2cb07887e97bc98630a3f527ac6b5e253f41c203cf37459568428`.
The reseal repairs a benchmark-design defect. It does not itself pass Gate -1,
demonstrate policy competence, or authorize Writer center training.

## Permanent specification-only reseal

### Authority and role definitions

- Task ordering and text come from pinned task-map Git blob
  `08144b4dd01d91fb0ca40e2c1d93ccaa85025fbc`; the canonical ordered
  `task_index/scene/language` surface has SHA256
  `9ec40758b7b5c2a6c3c0aacb5e41c2a0bd30a21e702e9b0f1187c1adeeb8ea39`.
- The parser schema is `libero90_role_factors_v1`. It normalizes explicit
  operations, moved grammatical patients, target receptacles, target
  relations, source selectors, target selectors, actuated fixtures, and
  actuated subregions. A scene distractor never counts as role exposure.
- `put` and `place` normalize to `place`. Explicit language order determines
  step order. `pick up … and place …` and `stack … and place them …` retain two
  ordered steps. Full ordered compositions are reported separately from
  primitive role atoms.
- All 90 instructions parse exactly once. Unknown or ambiguous templates fail
  closed; tests reparse every checked-in factor record exactly.

### Frozen search contract and result

The search algorithm is
`sha256_multistart_greedy_plus_steepest_swap_v1`, seed `20260718`, with 16,384
deterministic candidates. The hard requirement is that every primitive role
used by validation or held tasks occurs task-relevantly in at least two source
tasks. Subject to that constraint, the fixed lexicographic objective maximizes
unseen full compositions, same-scene semantic hard negatives, and same-scene
source controls, then balances scenes and operation count, then preserves old
assignments where possible. Validation/held partitioning has its own frozen
scene, difficulty, role-balance, and old-role-retention priorities.

| Reseal diagnostic | Result |
| --- | ---: |
| Source / validation / held tasks | 60 / 15 / 15 |
| Evaluation primitive roles | 41 |
| Minimum source-task occurrences for every evaluation role | 2 |
| Coverage violations | 0 |
| Evaluation tasks with source-unseen full composition | 30 / 30 |
| Evaluation tasks with a different same-scene source task | 30 / 30 |
| Evaluation tasks with a role-sharing same-scene hard negative | 28 / 30 |
| Two-operation tasks in source / validation / held | 29 / 7 / 8 |
| Prior evaluation tasks retained anywhere in evaluation | 8 / 30 |
| Prior exact validation/held assignments retained | 3 / 30 |

The active IDs are:

- Source: `3, 4, 5, 6, 9, 10, 12, 15, 16, 17, 19, 20, 23, 24, 25, 26,
  27, 30, 31, 32, 33, 34, 35, 37, 38, 39, 41, 42, 43, 44, 46, 47, 49, 50,
  52, 53, 54, 57, 58, 62, 63, 64, 67, 68, 69, 71, 72, 73, 74, 75, 77, 78,
  79, 80, 81, 82, 83, 84, 87, 89`.
- Validation: `2, 8, 11, 13, 21, 22, 28, 40, 51, 59, 60, 65, 70, 76, 86`.
- Held: `0, 1, 7, 14, 18, 29, 36, 45, 48, 55, 56, 61, 66, 85, 88`.

No threshold changed: Gate -1 retains 20 percentage points for full versus
no-spec, 20 points for correct versus swapped, and 0.80 for counterfactual
correct-switch fraction. Split selection is never reopened after policy
results.

## Rejected prior split and preserved failure evidence

The following language-only checks rejected the old split. `Source tasks`
counts task-relevant grammatical roles, not fixture or distractor presence.

| Atom and role | Exact language evidence | Source tasks | Validation tasks | Held tasks | Result |
| --- | --- | ---: | ---: | ---: | --- |
| verb `stack` | instruction begins `stack` | none | 16, 63 | 17, 64 | fail: 0 source |
| moved object `tomato_sauce` | `tomato sauce` | none | 49, 59 | 54 | fail: 0 source |
| target relation `under` | ` under ` | none | 42, 89 | none | fail: 0 source |
| target relation `front_of` | `to the front of` | none | none | 34 | fail: 0 source |
| target receptacle `wine_rack` | `wine rack` | none | none | 27 | fail: 0 source |
| moved object `moka_pot` | `moka pot` | 38 | 19 | none | fail: 1 source |
| moved object `wine_bottle` | `wine bottle` | 26 | none | 27 | fail: 1 source |
| moved object `white_bowl` | `white bowl` | 43 | 36 | 37 | fail: 1 source |

The complete strict parser reproduces these failures and finds additional
selector-level constraints; the original table remains a high-confidence
minimal failure packet rather than being rewritten after recovery.

## Evidence status by Gate -1 component

| Component | Status | Evidence and authorized interpretation |
| --- | --- | --- |
| Official checkpoint mechanics | complete | All ten `libero_spatial` task/BDDL/init-state/camera/controller paths and videos execute; 9/10 one-episode successes are mechanics-only. |
| Evaluation identity | diagnosed | Reset rendering has sparse one-level RGB nondeterminism; model actions are batch-shape sensitive. The fixed-batch statistical/functional contract is frozen, and cross-batch results are not pooled. |
| Prior canonical LIBERO-90 authority | complete but tied to rejected split | The 90-file audit validated data integrity and leakage boundaries, but its source-only normalization belongs to the rejected split and must not be reused under the reseal. |
| LIBERO-90 role-factor table and split design | recovery mechanics pass | All 90 specifications parse, the permanent seal has zero role-coverage violations, and its deterministic regeneration/hash checks pass. This is not policy evidence. |
| Fresh canonical manifest under resealed split | pass with documented upstream notes | Clean commit `23f3301` re-audited all 90 files, recomputed 183,555 source-only rows, kept all 30 validation/held tasks metadata-only, and recorded zero evaluation numeric access with valid checksums. |
| Prompt-path specification pilot | scale candidate only | On overlap-trained `libero_spatial` tasks 0/1, correct is 6/8 and 6/8; no-spec is 0/8 and 0/8; scene-only is 2/8 and 0/8; swapped is 0/8 and 0/8. This is not a LIBERO-90 result. |
| Same-init paired executable goals | not measured | Prompt swap kept the environment goal fixed; it cannot establish correct counterfactual goal switching. |
| Video content and temporal necessity | not measured | Wrong/same-scene/shuffled/reversed/first/last/scene-only video controls remain pending. |

## Method, robustness, and leakage boundary

The task-map blob is parsed as a Python literal after verifying its Git-blob
identity. The reseal program has no dataset, BDDL, simulator, model, reward, or
normalization input. It serializes `numeric_or_privileged_fields_read: []` and
only `task_index`, `scene`, and `language` as authority fields. Contract tests
pin the record hash, parser schema, source-occurrence threshold, algorithm,
seed, candidate count, prior split commit, active split, and unchanged Gate -1
thresholds. The record regenerates byte-for-byte after the active contract is
installed because the rejected split is preserved explicitly as `prior_split`.

The split audit uses task as the independent unit. Primitive exposure is a
necessary source-supervision condition, not proof that a learned policy has
mastered the primitive. Full-composition novelty is exact over the sealed
ordered role records; it is not inferred from policy behavior.

## Limitations and next evidence

- The split and data-access mechanics are valid under the declared role grammar
  and fresh canonical manifest. This still does not establish that a policy has
  learned the primitives or uses the specification causally.
- Language-derived roles cannot prove policy competence or causal use of task
  information. Same-init executable-goal controls and video content/temporal
  controls remain required.
- The overlap specification pilot has two tasks and an overlap-trained policy;
  it cannot estimate LIBERO-90 validity, Writer utility, or video utility.
- No LIBERO-90 policy success, oracle, Writer, or held reward result was read in
  selecting or validating the resealed task IDs.

Next, continue the predeclared same-init specification and language/video
causal controls under the sealed split and fixed-batch evaluator contract. Gate
0 and Gate 1 remain downstream; Writer center training remains unauthorized.
