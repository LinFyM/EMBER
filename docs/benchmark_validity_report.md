# EMBER Gate -1 Benchmark and Specification Validity Report

Status: **split resealed and source paired-goal mechanics valid; Gate -1 remains in progress**
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

A subsequent source-only mechanics probe establishes that a feasible same-state
paired-goal surface can be executed with the pinned native evaluator. It keeps
policy causal behavior separate: no result yet shows that a learned policy
switches correctly when only the task specification changes.

## Source same-state executable-goal mechanics

The overlap-trained `libero_spatial` suite is unsuitable for this particular
control because all ten tasks share one native success predicate. EMBER
therefore selected the smallest predeclared legal alternative after the reseal:
source tasks 3 and 4 share `KITCHEN_SCENE10`, an identical 77-dimensional
MuJoCo state layout, and the same drawer target, but task 3 targets the back
butter (`butter_2`) while task 4 targets the front butter (`butter_1`). No
validation or held semantics or numeric values are used.

The probe freezes the task/data/BDDL hashes, first eight task-3 init states,
first eight source demonstrations per task, final-recorded-state selector,
exact flattened-state identity, native `check_success`, and the unchanged 0.80
counterfactual threshold. Results are:

| Mechanics diagnostic | Result |
| --- | ---: |
| Native model-layout hashes equal | yes |
| Shared init states exact after injection | 8 / 8 |
| Shared init states neutral under both goals | 8 / 8 |
| Task-3 terminal states pass task 3 only | 8 / 8 |
| Task-4 terminal states pass task 4 only | 8 / 8 |
| Minimum bidirectional specificity fraction | 1.00 |

This passes only the paired-goal evaluator mechanics prerequisite. Demonstration
terminal states certify that the two native predicates are executable and
specific on exactly the same physical state; they do not show policy behavior
under an instruction switch. The canonical artifact and local HTML report are
`$EMBER_OUTPUT_ROOT/gate_minus1/specification/source_same_init_goal_20260718T050511Z`
and its `source_same_init_goal_latest` link. They contain hashes and boolean
matrices but no raw states, actions, model XML, or private paths.

## Same-observation language-to-action path

The prior overlap prompt pilot used matched seeds and init-state indices but
reset each arm independently. Because Gate -1 identity work found sparse
one-level renderer variation across resets, a cached-observation follow-up was
required before attributing the full prompt effect solely to language.

For each overlap task, the follow-up resets one async batch of eight once, then
uses the exact cached two-camera/state observation, fixed batch shape, and fixed
policy RNG for correct, no-spec, scene-only, swapped, and a correct repeat. It
retains the first ten actions of each one-forward SmolVLA action chunk. The
diagnostic threshold is a per-episode maximum absolute plan delta of 0.01,
which is over four times the previously observed 0.002254 batch-shape artifact.

Correct-repeat plans have zero delta for all 16 samples. Every correct-versus-
swapped, correct-versus-no-spec, and correct-versus-scene-only sample exceeds
0.01, including the first action. Overall maximum deltas are 0.452167, 0.342599,
and 0.318189 respectively. This establishes a same-observation language-to-
action causal path on the overlap-trained checkpoint and rules out repeated
rendering as the prompt-effect explanation. Because the overlap environment
goal remains fixed, it still does not establish correct counterfactual goal
switching or a LIBERO-90 Gate -1 pass.

The checksummed result and compact local report are
`$EMBER_OUTPUT_ROOT/gate_minus1/specification/language_action_20260718T052249Z`
and `language_action_latest/index.html`. The existing overlap gallery remains
the video authority; this action-only diagnostic does not duplicate media.

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
| Same-observation language-to-action path | diagnostic pass | Exact correct repeats and 16/16 substantive action-plan contrasts remove repeated-reset rendering as the prompt-effect explanation. Goal correctness is not measured. |
| Same-init paired executable goals | source evaluator mechanics pass; policy behavior pending | Tasks 3/4 have exact shared-state compatibility and 16/16 native-goal specificity. This does not show a policy follows the switched spec. |
| Video content and temporal necessity | protocol frozen; canonical result pending | Source tasks 3/4, disjoint support/query demos, RGB-only frozen encoder/readout, wrong/same-scene/shuffled/reversed/first/last/static/drop-last controls, thresholds, and seeds are sealed before outcomes. This cannot itself pass Gate -1. |

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
  and fresh canonical manifest. This still does not establish that a LIBERO-90
  source policy has learned the primitives.
- Language-derived roles cannot prove policy competence or causal use of task
  information. The overlap policy now has a same-observation language-action
  path, but correct paired-goal behavior and video content/temporal controls
  remain required even though the native paired-goal surface is valid.
- The overlap specification pilot has two tasks and an overlap-trained policy;
  it cannot estimate LIBERO-90 validity, Writer utility, or video utility.
- No LIBERO-90 policy success, oracle, Writer, or held reward result was read in
  selecting or validating the resealed task IDs.

Next, execute the frozen action-hidden video content/temporal controls, then obtain legal
source-task competence and evaluate correct behavior on the paired native-goal
surface without changing its state or evaluator. Gate 0 and Gate 1 remain
downstream; Writer center training remains unauthorized.
