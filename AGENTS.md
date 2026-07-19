# EMBER Repository Instructions

## Purpose

EMBER is a standalone embodied-learning research project. Treat every proposed
mechanism and novelty statement as a hypothesis to test, not an established
result. The repository is now in the execution phase: turn the research design
into reproducible evidence without weakening the held-out contract.

## Required reading order

Before changing code, data, environments, or experiment state, read:

1. `README.md`
2. `task_plan.md`
3. `findings.md`
4. `progress.md`
5. `docs/origin_and_general_thesis.md`
6. `docs/concept.md`
7. `docs/prior_work_memllm_lessons.md`
8. `docs/novelty_and_landscape.md`
9. `docs/decisions_and_open_questions.md`
10. `docs/expert_plan.md`
11. `docs/execution_brief.md`
12. `docs/benchmark_validity_report.md`

`docs/expert_plan.md` is preserved expert advice. Where its eight-GPU resource
assumption or stage details conflict with the active execution contract,
`docs/execution_brief.md` and this file take precedence.

The owner corrected the research authority on 2026-07-18 after reviewing the
original design conversation. Any current Goal text or historical document that
makes a canonical bank, shared task-update subspace, soft geometry, residual
escape path, or a mandatory Gate 1 a prerequisite is superseded. Those
mechanisms were assistant/expert additions, are outside the current EMBER
project and long-term Goal, and must not be implemented, trained, benchmarked,
scheduled, or anticipated by reserved code paths. `docs/expert_plan.md` remains
unchanged only as historical provenance.

`configs/phase0.toml` is already hash-bound into completed mechanics/base
evidence. Its legacy future-stage `task_geometry` string is preserved only to
avoid rewriting provenance and is non-authoritative/non-executable. Do not use
it to schedule work, and do not retroactively mutate sealed evidence merely to
remove the string.

## Active objective

Advance EMBER from a research proposal through the complete corrected research
program. Begin with a faithful SmolVLA plus LIBERO development path and
establish:

1. benchmark and task-specification validity;
2. a useful task-specific update oracle;
3. direct Writer acquisition of complete task-specific LoRA tensors at the
   predeclared target matrices, with immediate zero-interaction utility;
4. an independent source-only Writer-only RL phase that freezes the base,
   treats generated LoRA as functional output rather than a separately updated
   variable, and updates Writer parameters from rollout reward;
5. matched-budget ordinary task-local LoRA RL from the Writer initialization,
   with the Writer and base frozen and only task-local LoRA updated in place;
6. source-only reward/delayed outer learning that updates the Writer through
   task-local LoRA adaptation while the shared base remains frozen; and
7. shared-frozen held-task evaluation; and
8. OpenVLA-OFT scale confirmation after the lower-cost mechanism survives.

Phase 0, exact resume, throughput work, Gate -1, Gate 0, one training run, or
authorization of a later stage cannot by itself complete the long-term Goal.

Do not implement the complete system merely because it appears in the expert
plan. Each expensive component needs evidence from its scientific predecessor.

## Hard resource constraint

- At most **four NVIDIA A100 80GB GPUs** may be allocated to EMBER at any time,
  across all concurrent EMBER processes and agents.
- Default to one GPU for smoke tests and one or two GPUs for pilots. Use three or
  four only when measured memory or throughput justifies it.
- After a minimal correctness smoke establishes the real memory baseline, tune
  useful batch size, environment/task parallelism, gradient accumulation, and
  caching so that each allocated A100 normally uses about 70GB and retains about
  10GB average headroom against OOM. This is a steady-state efficiency target,
  not permission to allocate dummy tensors or sacrifice correctness and
  reproducibility merely to fill memory.
- Do not copy eight-GPU launch commands from `docs/expert_plan.md`. Recompute
  batch size, gradient accumulation, parallelism, GPU-hours, and wall-clock for
  the four-GPU ceiling.
- Before a substantial download or run, check available storage and estimate
  data, cache, checkpoint, and log growth.

## Scientific invariants

- Writer-visible held-task input may contain language and action-hidden video,
  but never held actions, proprioceptive trajectories, rewards, terminal flags,
  task IDs, filenames, or hidden normalization statistics.
- Shared base, Writer, encoders, and every other shared state are frozen during
  held-task evaluation. Only predeclared task-local state may adapt from held
  rewards.
- Keep source, validation, and reporting-only held surfaces separate. Do not
  tune architecture, thresholds, seeds, or checkpoints after reading held
  results.
- Immediate functional utility is required before claiming bootstrapping.
- LoRA is the only adaptation mechanism in the current project. Do not add a
  bottleneck adapter, IA3, prefix tuning, shared base adapter, shared LoRA, or
  another parameter-efficient state alongside it.
- The Writer must emit the complete declared task-specific LoRA, not coefficients
  in a shared task-update bank or subspace.
- The only structural search space is the common predeclared LoRA contract:
  selected target layers plus rank/parameter count. Ordinary task-local RL
  updates those same LoRA parameters in place from the Writer initialization.
  The Writer emits no second object that constrains RL.
- Primary Writer supervision is independent source-query action,
  flow-matching, behavioral, or return loss differentiated through the
  functional adapter. Oracle-update or physical-delta imitation is auxiliary
  only when predeclared; raw LoRA-factor MSE is never the primary objective.
- Parameter distance and raw LoRA-factor MSE are diagnostic or auxiliary, not
  the primary scientific objective.
- Gate 0 must include a bounded LoRA capacity audit on source/validation:
  zero/base versus independently trained task-local LoRA is the matched test;
  action-expert partial/full updates or full fine-tuning may appear only as
  explicitly non-matched capability upper bounds. If only those upper bounds
  work, classify the fixed LoRA target/rank contract as too narrow before
  attributing failure to the Writer.
- Gate 0 is a useful-update Gate, not an SFT-only Gate. A bounded source-only,
  matched-control task-local LoRA RL recovery may establish that the declared
  LoRA space contains a useful behavioral update. Such a result is not
  supervised zero-interaction utility and is not Writer evidence.
- The last-two q/v rank-8 pilot and its rank-16 support recovery test only the
  frozen 12-demo, 750-step custom acquisition recipe. Their outcomes cannot by
  themselves establish a final LoRA, Gate 0, Writer, or EMBER negative. A
  support/query partition is an independence rule, not a permanently small
  support or optimization budget.
- Before sealing the Writer contract or recording a final LoRA-capacity
  negative, run one primary-source-anchored mature task-local LoRA positive
  control. The bounded primary recipe uses only source-task action labels,
  keeps an independent source query and fresh closed-loop surface, covers the
  SmolVLA v0.6.0 default-like action support (all action-expert q/v plus
  state/action/time projections), and imports only architecture-compatible
  rank/capacity, duration, initialization, scheduling, and augmentation
  principles from empirically successful OpenVLA/OpenVLA-OFT LoRA recipes.
  SmolVLA's roughly 50-episode, batch-64, 20k-step recipe trains the action
  expert/projections and is not LoRA evidence; the LeRobot PEFT defaults are an
  implementation/API anchor, not published behavioral proof.
- The mature primary is predeclared as 40 source support demonstrations, six
  independent source query demonstrations, 20k steps, the 37 default-like
  targets, rank 32/alpha 16/dropout 0, Gaussian exact-physical-zero LoRA,
  SmolVLA AdamW warmup/cosine scheduling, compatible 90--100% random-resized
  crops, fixed-final-step selection, and fresh init-state 40--47 closed-loop
  evaluation. Source-side Gate 0 may use actions from episode roles 0--39;
  those actions remain forbidden as Writer-visible inputs. Held and validation
  numeric access remain zero. At most one predeclared all-action-expert-linear
  rank-32 compatibility recovery is allowed after a mechanically valid primary
  failure; no grid search or threshold reduction is allowed.
- Permanently seal the empirically successful LoRA support/rank/scale rather
  than shrinking back toward the smallest adapter merely to simplify Writer
  generation. If the bounded positive controls show that no declared LoRA
  space has behavioral capacity, preserve the failure packet and escalate the
  scientific decision; do not blame Writer acquisition before it is tested.
- After that seal, Writer, zero-init ordinary LoRA RL, average/retrieval,
  language-only HyPoGen/DISC-style direct generators, and every other matched
  arm must use exactly the same LoRA support and trainable-parameter budget. If
  broad support is hard to generate, use layer/module-aware, chunked or
  type-specific generation rather than silently shrinking targets; this is an
  output-architecture remedy, not permission to add a bank, subspace, mask, or
  later-RL constraint.
- Keep EMBER independent from MemLLM code and Wiki/QA mechanisms. Reuse lessons,
  not implementation dependencies.

## Gate recovery protocol

A failed or ambiguous gate is a diagnosis trigger, not permission to declare
failure casually. Before stopping:

1. verify mechanics, data authority, split integrity, seeds, metrics, and the
   exact failed claim;
2. classify the failure as benchmark/specification, information sufficiency,
   representation, Writer acquisition, optimization/RL, implementation, or
   resource/throughput;
3. record the evidence in `findings.md` and the attempted remedy in
   `progress.md`;
4. explore bounded remedies on source and validation surfaces, such as a
   predeclared target-layer/rank expansion, better functional adapter
   parameterization, better temporal video representation, a stable RL
   estimator, or a smaller faithful model;
5. rerun the gate with matched controls and fresh evidence.

Recovery must not violate the core thesis or evaluation contract. Do not expose
held labels, update shared parameters on held tasks, silently change the task
distribution after seeing held results, remove strong baselines, or lower a
threshold solely to turn a failure into a pass. If bounded remedies are
exhausted, preserve the negative result and narrow the claim explicitly.

## Current implementation decisions

- Use SmolVLA as the development and mechanism-validation policy. Scale a
  surviving result to OpenVLA-OFT as confirmation rather than starting with the
  7B path.
- Treat language/video-to-complete-LoRA as a task-conditioned hypernetwork or
  amortized LoRA initializer, not a universal meta-optimizer, learned update
  direction, or learned optimizer.
- Make neutral-prompt parameter compilation a co-primary mechanism test; also
  report the practical setting where the online policy receives the task
  instruction.
- Include a strong language-only HyPoGen/DISC-style parameter-generator
  baseline, not only task-ID, retrieval, or direct-conditioning controls.
- After Gate 0, train a direct Writer that emits all LoRA factors for the
  declared target matrices. Differentiate independent source-query functional
  loss through adapter application into the Writer.
- After positive zero-interaction Writer utility, compare ordinary task-local
  LoRA RL from the Writer initialization against matched standard-LoRA and
  direct-generator baselines.
- The core RL causal arms are: zero-LoRA initialization plus ordinary RL;
  cold-start Writer LoRA initialization plus identical RL; and
  reward-outer-trained Writer LoRA initialization plus identical RL. Hold LoRA
  targets/rank/count, RL algorithm and hyperparameters, seeds, reward, and
  interaction budget fixed. Report zero-step utility, learning-curve AUC,
  time-to-threshold, matched-budget final success, and final-minus-initial gain.
  A claim about the learning process beyond a better starting point also needs
  a matched-initial-performance or equivalent control.
- During direct Writer cold-start and source reward/meta-RL, the shared base is
  frozen. Inner adaptation updates task-local LoRA; the source-only outer
  objective updates Writer parameters, differentiating through the inner loop
  or using a predeclared estimator. No shared base adapter or shared policy
  parameter is trained in the default Writer stage.
- Canonical banks, shared update subspaces, soft geometry, and residual escape
  are out of scope. Historically, the bank supplied a shared basis/span,
  geometry scaled or preconditioned its coordinates, and residual escape could
  leave that span; together they imposed a second, narrower Writer-conditioned
  RL search space. Their removal means current RL has no Writer-predicted bank,
  basis, mask, metric, radius, or learning-rate object. A future separate
  research program would need a newly demonstrated bottleneck and matched
  evidence before considering any shared structure; a bank is not presumed and
  this repository reserves no path for it. Updating shared base weights or a
  shared LoRA during source outer RL is likewise a separate matched ablation,
  not the default or a completion requirement.

## Experiment and systems practice

- Reproduce an official known checkpoint or recipe before interpreting a custom
  source-only result.
- Pin repository commits, dataset revisions, file hashes, environment versions,
  task manifests, normalization authority, seeds, and exact commands.
- Treat exact-resume, RNG digests, and telemetry binding as infrastructure, not
  scientific Gates. Operational sufficiency is a loadable checkpoint with
  correct model/optimizer/scheduler/global-step/data-cursor state and a short
  non-crashing resume whose loss or functional behavior stays within a
  predeclared tolerance. For non-scientific anomalies, default to one
  reproduction, one narrow repair, and one verification; then move on unless
  the issue can change recoverability, sampled data, closed-loop success, a
  Gate decision, matched fairness, or the held boundary. Do not add identity
  surfaces merely to make engineering evidence cosmetically exact.
- Run multi-hour scientific fits as predeclared exact-resume segments at
  meaningful candidate boundaries. The first blind segment should normally be
  at most about 30 minutes; inspect only legal source mechanics/query trends,
  continue the same trajectory when its frozen rule passes, and reserve final
  closed-loop surfaces until the candidate is frozen. Staging changes execution
  cadence, not model capacity, data authority, or maximum scientific budget.
- Optimize the full training and inference path for useful throughput. Reuse
  canonical model loads, manifests, decoded or preprocessed observations,
  cached features, and other scientifically equivalent intermediates whenever
  safe; avoid duplicate downloads, repeated preprocessing, idle simulator/GPU
  pipelines, and redundant computation, while recording cache provenance and
  invalidation rules.
- Use durable logs for long runs. Store temporary output under `.codex/tmp/` or
  ignored output directories; never commit datasets, checkpoints, credentials,
  private infrastructure, or raw copyrighted papers.
- Make simulator and policy runs reviewable after execution with compact video
  galleries or equivalent visual artifacts, but keep that surface bounded.
  Preserve metrics, manifests, failure packets, the current `latest` gallery,
  and explicitly designated evidence; after artifact-heavy phases, inventory
  accumulated media and remove only verified-regenerable, unpinned older videos
  when they become numerous or large. Record every such prune with paths,
  hashes, reason, and bytes released, and never delete ambiguous scientific
  evidence automatically.
- Keep one canonical implementation path. Put unselected alternatives in
  `docs/decisions_and_open_questions.md` until evidence selects them.
- Use isolated branches or worktrees for overlapping write-capable tasks. Never
  let multiple writers edit one worktree concurrently.
- Update `task_plan.md`, `findings.md`, and `progress.md` after meaningful state
  changes so a resumed agent can recover without conversation history.

## Current-goal completion evidence

The current long-term execution goal is complete only when the repository
contains:

- a reproducible four-GPU-compatible environment and data manifest;
- a benchmark/specification validity report;
- a closed-loop useful-update oracle report;
- recorded recovery attempts for any failed or ambiguous gate; and
- a direct full-LoRA Writer report with independent-query zero-interaction
  utility and all required modality/negative/direct-generator baselines;
- a matched-budget ordinary task-local LoRA RL report;
- a source-only reward/delayed outer-learning report in which task-local LoRA
  is the model-side adaptive state, Writer receives the outer update, and the
  shared base stays frozen;
- a shared-frozen held-task report under the permanently sealed contract; and
- a predeclared-seed, confidence-interval, causal-control, and reproducible
  OpenVLA-OFT scale confirmation of the surviving mechanism.

Positive completion additionally requires held-frozen evidence that Writer
initialization beats zero/base, average, retrieval, and a capacity-matched
language-only HyPoGen/DISC-style direct LoRA generator; matched A/B/C ordinary
RL evidence for zero-step utility and adaptation efficiency/final behavior; and
a cold-start versus source-reward-outer-trained Writer comparison showing that
delayed reward improves the generated initialization. If bounded recovery
instead falsifies a core hypothesis or requires changing the paper claim,
preserve the negative evidence and escalate the decision rather than marking
the Goal complete.
