# EMBER Gate -1 Benchmark and Specification Validity Report

Status: **recovery decision required; Gate -1 is not passed**  
Evidence cutoff: 2026-07-17

## Technical summary

The reproducible SmolVLA/LIBERO path and canonical LIBERO-90 data authority are
mechanically valid, and a two-task overlap-trained pilot shows a large
descriptive dependence on the policy-visible prompt. Those results are useful
substrate evidence, not a LIBERO-90 Gate -1 pass.

The current predeclared LIBERO-90 60/15/15 split fails its stronger validity
contract before any LIBERO-90 policy result is read. A conservative,
language-only role audit finds validation or held tasks whose task-relevant
atoms have fewer than two source-task occurrences. Several have zero: `stack`,
moving `tomato_sauce`, target relation `under`, target relation `front_of`, and
target receptacle `wine_rack`. Moving `moka_pot`, `wine_bottle`, and
`white_bowl` each has only one source task.

This is a benchmark/specification failure, not a Writer, representation, or
optimization failure. Writer center training remains unauthorized. The next
scientific decision is whether to reseal the split once using task
specifications only, before any LIBERO-90 policy training/evaluation, or retain
the split and explicitly abandon the source-covered compositional-held-out
claim.

## The current split violates source-atom coverage

The table reports task IDs at the task grain. `Source count` is the number of
source task specifications in which the atom has the indicated task-relevant
role; distractor presence in a scene or BDDL object list does not count as
executable exposure. The predeclared requirement is at least two source tasks
for every validation/held atom.

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

An exact lookup table is used instead of a chart because the decision depends
on categorical constraint violations and task identities, not a continuous
trend. These eight checks are a high-confidence lower bound, not a claim that a
complete role parser would find only eight failures.

## Evidence status by Gate -1 component

| Component | Status | Evidence and authorized interpretation |
| --- | --- | --- |
| Official checkpoint mechanics | complete | All ten `libero_spatial` task/BDDL/init-state/camera/controller paths and videos execute; 9/10 one-episode successes are mechanics-only. |
| Evaluation identity | diagnosed | Reset rendering has sparse one-level RGB nondeterminism; model actions are batch-shape sensitive. The fixed-batch statistical/functional contract is frozen, and cross-batch results are not pooled. |
| Canonical LIBERO-90 authority | pass with documented notes | 90 tasks, 4,500 demos, 669,043 frames, exact 66,658,085,995 bytes, source-only normalization, checksums, and leakage checks pass. Legacy producer-path and wording notes remain visible. |
| Prompt-path specification pilot | scale candidate only | On overlap-trained `libero_spatial` tasks 0/1, correct is 6/8 and 6/8; no-spec is 0/8 and 0/8; scene-only is 2/8 and 0/8; swapped is 0/8 and 0/8. This is not a LIBERO-90 result. |
| LIBERO-90 split factor coverage | fail | The table above disproves the at-least-two source-atom condition for the current split. |
| Same-init paired executable goals | not measured | Prompt swap kept the environment goal fixed; it cannot establish correct counterfactual goal switching. |
| Video content and temporal necessity | not measured | Wrong/same-scene/shuffled/reversed/first/last/scene-only video controls remain pending after split repair. |

## Scope, authority, and definitions

- The canonical task grain is one LIBERO-90 task ID. The fixed split is 60
  source, 15 validation, and 15 reporting-only held tasks in
  `configs/phase0.toml`.
- The factor audit uses only the English language specification and scene
  identity that the scientific contract already permits as held-task input. It
  does not parse held BDDL goals or read held actions, proprioception, rewards,
  terminal flags, filenames as model inputs, or normalization contributions.
- An **atom** here is a role-specific primitive verb, moved object,
  receptacle, or spatial relation. An ordered multi-step composition is reported
  separately and is not required to have appeared as a source composition.
- Source exposure means that the atom is task-relevant in a source instruction.
  Merely appearing as a distractor object or fixture is not executable bridge
  supervision.
- Gate decisions use task as the independent unit. Episode-level counts in the
  overlap pilot are descriptive and cannot supply a LIBERO-90 task-level
  confidence interval.

## Method and robustness checks

The data authority comes from the immutable dataset revision
`f13aa24a3da8c43c7225569f28c562979fa0e35a`. The recovery audit at commit
`d6cdac7` validates every LFS hash, HDF5 schema, task/BDDL/init-state binding,
camera/controller field, and source-only normalization record. Its artifact is
`$EMBER_OUTPUT_ROOT/phase0/libero90_manifest/latest`.

The eight factor checks above are deterministic exact-prefix or exact-substring
queries over the canonical `language` field. Exact task lists are included so a
future complete parser can reproduce or refute each finding. The listed zero-
and one-source violations survive reasonable normalization such as treating
`right moka pot` as `moka_pot`; none can reach the required source count of two
without changing task roles, counting distractors, adding source tasks, or
changing the split.

The prompt-path result is separately checksummed under
`$EMBER_OUTPUT_ROOT/gate_minus1/specification/latest`. It uses an official
overlap-trained checkpoint and therefore measures only whether the evaluator's
policy-visible prompt path can affect behavior under one fixed batch contract.

## Limitations and uncertainty

- This is not yet a complete 90-task verb/object/receptacle/relation/order
  annotation. The eight violations are sufficient to reject the current split
  criterion, but a resealed split still needs a complete deterministic table and
  tests for every instruction template.
- Language-only atom extraction does not prove that a source policy has learned
  the primitive. It is a necessary coverage check, not a sufficient competence
  check.
- The overlap specification pilot has two tasks and an overlap-trained policy.
  It cannot estimate LIBERO-90 benchmark validity or Writer/video utility.
- Same-init executable goal switching and video temporal/content controls remain
  unmeasured. Gate -1 remains incomplete even after a split is resealed.
- No LIBERO-90 policy success, oracle, Writer, or held reward result was read in
  selecting this diagnosis.

## Recommended next step

**Recommended: authorize one specification-only split redesign now.** Build a
complete language-derived factor table, require every validation/held primitive
atom to occur in at least two source tasks, preserve same-scene hard negatives
where possible, then freeze task IDs, factor rules, thresholds, and seeds before
any LIBERO-90 base/oracle/evaluation result. Record both the rejected split and
the resealing rule; do not optimize the new split against policy outcomes.

If the split must remain unchanged, narrow the paper contract to a mixed
novel-atom/novel-composition benchmark. That removes the clean
source-covered-atoms premise and requires reconsidering whether Gate 0/Writer
training can identify the claimed transfer mechanism.

Disallowed recoveries are counting distractor presence as executable exposure,
parsing held privileged labels, redefining atoms after seeing policy results,
or lowering the at-least-two threshold solely to retain the current split.

## Further questions requiring owner decision

1. May the current 60/15/15 split be replaced once, using only task
   specifications and before any LIBERO-90 policy result, then permanently
   resealed?
2. If not, should the project explicitly narrow its first-paper claim to permit
   genuinely unseen primitive atoms, accepting that this is a different and
   less identifiable transfer question?

