# EMBER Active Execution Brief

## 1. Authority and objective

This file converts the independent expert plan into the active execution
contract. `docs/expert_plan.md` remains the full expert record. This brief
overrides its eight-GPU resource assumption and resolves several implementation
gaps identified before experiments begin.

The active objective is to determine whether a task specification containing
language and action-hidden video can be amortized into:

1. a parameter update with positive zero-target-interaction utility; and
2. a soft task-conditioned local geometry that improves matched-budget
   reward-driven adaptation.

The first claim is intentionally limited to held-out task compositions under
one robot embodiment and one simulator dynamics family.

## 2. Active terminology and contribution boundary

The static Writer is an **amortized task-conditioned parameter-update
generator**, implemented as a task-conditioned hypernetwork and adapter
initializer. It is not called a general meta-optimizer. A later feedback-aware
Writer may approach a learned update rule only if it repeatedly reads adaptation
history or gradient/reward summaries and produces further updates.

The broad information-to-parameter thesis is scientifically meaningful but not
itself novel. The candidate contribution is the complete controlled contract:

```text
language + action-hidden task video
    -> immediately useful task-local adapter center
    -> soft task-conditioned parameter geometry with residual escape
    -> matched-budget task-local RL
    -> source-only shared Writer refinement
    -> shared-frozen held-task evaluation
```

Language-only policy generation, action-free video learning, LoRA bases, and
demonstration-plus-reward adaptation all have close prior work. Each component
must earn its place through matched controls.

## 3. Model and benchmark path

### Development path

- Policy: immutable revision of `lerobot/smolvla_base`.
- Simulator: a pinned LIBERO revision with an audited LIBERO-90 task map, BDDL
  files, init states, cameras, controller, and dataset manifest.
- First targets: a small predeclared set of action-expert matrices enumerated
  from the actual checkpoint, not guessed from documentation.
- Purpose: environment reproduction, benchmark/specification validity, useful
  update oracle, canonical representation, Writer acquisition, and geometry
  mechanics under a one-to-four-GPU budget.

### Scale confirmation

- Policy: `openvla/openvla-7b` with a pinned OpenVLA-OFT revision and a
  source-only fine-tuning/data-statistics pipeline.
- Initial targets: the two 4096-to-4096 residual-block linear layers in the OFT
  L1 action head.
- Start only after the mechanism passes the lower-cost gates and a four-GPU
  memory/throughput pilot establishes a feasible recipe.

SmolVLA is not a disposable toy: it is the primary scientific development
surface. OpenVLA-OFT provides scale confirmation if justified by evidence.

## 4. Corrected information-flow experiment

The task specification and online policy prompt are two distinct causal paths.
Evaluate both:

1. **Parameter-compilation setting:** the Writer sees language/video, then the
   online policy receives a fixed neutral prompt. This is a co-primary mechanism
   test of whether task information entered the generated parameters.
2. **Practical setting:** both Writer and online policy receive the task
   instruction. Compare against capacity- and inference-matched direct
   conditioning.

Required controls include language-only, video-only, combined, wrong same-scene
video, shuffled/reversed video, first/last frame, task-ID/scene-only, nearest
adapter, average adapter, direct conditioning, and a language-only
HyPoGen/DISC-style parameter generator.

The first paper uses same-embodiment successful robot videos with actions hidden
from the Writer. This tests information-to-supervision conversion; it does not
establish that the videos were cheaper to collect or that human-video transfer
works.

## 5. Corrected geometry training path

The expert pseudocode emitted a geometry `p_T` during supervised Writer
bootstrap but did not give it a gradient before the geometry evaluation stage.
The active path is:

1. train and validate the Writer center independently;
2. freeze the center checkpoint for the first geometry comparison;
3. on source tasks, unroll a short inner optimizer only in the low-dimensional
   bank-coordinate state using cached policy features and support action loss;
4. optimize predicted positive scales/gates on independent source query action
   loss, KL, and non-harm objectives;
5. compare the learned diagonal geometry with unit and fixed-global geometry;
6. only after this offline mechanism works, refine center/geometry through
   source-only reward outer learning and evaluate task-local RL curves.

Because the differentiable loop is confined to roughly 24--64 coordinates, it
does not require second-order differentiation through the full VLA or simulator.
If offline geometry does not transfer to reward adaptation, test one bounded
source-reward estimator redesign before removing the geometry claim.

## 6. Staged authorization

### Phase 0: reproducible substrate

Produce an environment lock, exact revisions, dataset/file hashes, task-factor
manifest, normalization authority, known-checkpoint smoke test, measured GPU
memory, simulator throughput, and storage estimate. No scientific conclusion is
allowed before this passes.

### Gate -1: benchmark and specification validity

Construct same-scene hard negatives and, where feasible, paired goals under the
same initial state. Measure language/spec necessity, scene-only shortcuts,
spec-swap behavior, and video temporal/content necessity. Treat the proposed
60/15/15 split as unvalidated until its task-factor coverage is generated and
reviewed.

### Gate 0: useful-update oracle

On a small predeclared source-task subset first, show that a task-specific
low-rank update at the intended target matrices improves independent query loss
and closed-loop success without excessive policy drift. Scale to the complete
source set only after the pilot is positive and stable.

### Gate 1: canonical representation

Extract physical updates rather than matching arbitrary LoRA factors. Test
whether a canonical bank preserves the oracle's functional gain, with dimension
and rank selected on source/validation only. Report average-direction and
task-specificity controls.

### Stage 2 and later

Writer center training is authorized only after Gates -1, 0, and 1 provide a
learnable and identifiable target. Ordinary task-local RL follows positive
zero-step utility. Predicted geometry follows a demonstrated center-to-RL
benefit. Source-reward outer refinement and an optional tiny shared base adapter
come last. Shared modules remain frozen on held tasks.

## 7. Four-GPU execution rules

- Never allocate more than four A100s across concurrent EMBER jobs.
- Start smoke tests on one GPU and pilots on one or two.
- Use gradient accumulation, frozen-feature caching, task parallelism, and
  staggered jobs before increasing GPU count.
- Every multi-GPU command records expected peak memory, process count, GPU IDs,
  wall-clock, stop condition, and rollback/cleanup procedure.
- Re-estimate the expert plan's GPU-hours and wall-clock from measured pilots;
  its eight-GPU numbers are not an active schedule.

## 8. Gate recovery without scientific drift

When a gate fails or is ambiguous, do not immediately abandon the project and
do not hide the result. Produce a failure packet containing:

- the exact failed metric and confidence interval;
- mechanics and data-authority checks;
- qualitative rollouts or action diagnostics;
- classification of the likely failure layer;
- a ranked list of bounded remedies;
- the next cheapest discriminating experiment.

Allowed remedies include correcting implementation defects, improving task
counterfactuals, changing the predeclared target layer/rank once, increasing bank
dimension or operator rank once, canonicalization/whitening changes, a stronger
temporal video encoder, stable exploration or RL estimators, and falling back to
a smaller faithful policy.

Disallowed remedies include held-label access, shared held-task updates,
post-hoc task/split replacement after reading held results, deleting strong
baselines, unreported budget increases, or lowering pass criteria solely to
manufacture success. A genuine negative result is recorded only after the
bounded recovery path is exhausted.

## 9. Immediate work package

The next agent should not start a full training run. It should:

1. inventory the host software/GPU/storage state without recording private
   infrastructure in public files;
2. create a reproducible environment and pin upstream revisions;
3. reproduce one official SmolVLA/LIBERO inference or evaluation path;
4. inventory and hash the exact LIBERO-90 demonstrations, BDDL, and init states;
5. generate the task-factor table and audit the proposed source/validation/held
   split before reading held outcomes;
6. implement benchmark/specification probes and the smallest closed-loop
   useful-update oracle pilot;
7. update `task_plan.md`, `findings.md`, and `progress.md` with evidence and the
   next gate decision.
