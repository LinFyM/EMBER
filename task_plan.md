# EMBER Execution Plan

## Goal

Advance EMBER from a research design to a reproducible, four-GPU-compatible
experimental program. Establish benchmark validity, useful-update existence,
and canonical representation before authorizing Writer and RL complexity, then
progress through the complete design only when predecessor evidence supports it.

## Definition of success for the current goal

- Reproducible environment, upstream revisions, dataset/task manifests, and
  measured systems envelope under at most four A100 80GB GPUs.
- Benchmark/specification validity evidence with causal hard-negative controls.
- A useful-update oracle that improves independent closed-loop behavior.
- A canonical update representation that preserves task-specific oracle utility.
- Documented diagnosis and bounded recovery for every failed or ambiguous gate.
- An evidence-based decision to proceed to Writer center training, redesign the
  target, or narrow the claim.

## Boundaries

- No real robot, human-video transfer, cross-embodiment claim, or arbitrary web
  video in the first study.
- No held actions or shared held-task updates.
- No more than four concurrent A100s for all EMBER work.
- No full-system implementation before its scientific predecessors pass.
- No datasets, checkpoints, secrets, or private infrastructure in Git.
- After correctness baselines, tune useful GPU work toward roughly 70GB per
  allocated A100 with about 10GB average OOM headroom; do not count dummy
  allocations as utilization.
- Retain compact task videos and a `latest` gallery for review, while pruning
  only verified-regenerable, unpinned old media when it accumulates.

## Phases

| Phase | Status | Required evidence |
| --- | --- | --- |
| Phase 0. Reproducible substrate | in progress | environment lock, revisions, hashes, known-path smoke test, VRAM/throughput/storage measurements |
| Gate -1. Benchmark/spec validity | in progress under fixed statistical contract | task-factor audit, counterfactual/spec-swap/no-language/video controls |
| Gate 0. Useful-update oracle | in progress; source-only pilot frozen before outcomes | independent query and closed-loop gain with drift/non-harm diagnostics |
| Gate 1. Canonical representation | pending | functional preservation, conditioning, task-specificity, dimension/rank decision |
| Stage 2. Writer center | pending | zero-interaction utility over retrieval, average, direct-conditioning, and DISC/HyPoGen-style baselines |
| Stage 3. Center plus ordinary local RL | pending | matched-budget adaptation AUC/steps-to-threshold benefit |
| Stage 4. Predicted soft geometry | pending | trained task metric beats unit/global metrics without final-performance ceiling |
| Stage 5. Source-reward Writer outer loop | pending | source-trained reward update improves locked validation without zero-step collapse |
| Stage 6. Optional shared base adapter | pending | incremental benefit without coordinate drift or interference |
| Stage 7. Frozen held evaluation | pending | sealed primary comparison, full controls, resource and failure report |

## Evidence policy

- Use source data for fitting, validation for decisions, and held tasks once for
  final reporting after the method and thresholds are frozen.
- Treat task as the primary independent statistical unit.
- Record exact commands, revisions, seeds, budgets, artifacts, and failed
  attempts. Summarize durable evidence in `findings.md`; record current state and
  handoff in `progress.md`.
- On this host, long-run telemetry wrappers must preserve the scientific
  command return code explicitly and finalize samplers/artifacts in a normal
  post-command path with `INT`/`TERM` handlers. Do not combine `set -e` with a
  function-based `EXIT` trap: Bash 5.2.21 can turn successful canonical work
  into a false outer failure during function-context unwinding.
- A gate failure starts the recovery protocol in `AGENTS.md`. Do not silently
  change the scientific contract to create a pass.

## Current next actions

1. [x] Bootstrap a Python 3.12 environment from the locked project definition;
   verify package consistency and the four-GPU/storage contracts.
2. [x] Pin immutable LeRobot, LIBERO runtime/official, SmolVLA, SmolVLM,
   LIBERO-90, and simulator-asset revisions and hashes.
3. [x] Reproduce one official one-episode LIBERO inference/evaluation path from
   local verified snapshots before any custom policy changes.
4. [x] Calibrate useful single-GPU vector-environment concurrency and batching.
   Matched batch-eight sync/async evaluation selected asynchronous environments;
   measured batches 32, 96, and 112 established 112 as the resource-rich
   throughput rung (68,080 MiB peak) and 96 as the conservative rung.
5. [x] Exercise all ten `libero_spatial` task IDs with the validated offline
   entrypoint and retained resource/video artifacts. Treat this as a mechanics
   sweep, not policy-quality evidence. All ten paths and videos completed; the
   single task-5 failure is retained without tuning.
6. [ ] Build the data/task/BDDL/init-state/controller/normalization/split manifest
   and implement Gate -1 probes, including batch-invariant seed/init-state,
   reset-observation, initial-action, and short-rollout identity checks. The
   identity checks are implemented and have isolated renderer and model-batch
   effects. The leakage-safe manifest builder, source-only normalization, hash
   audit, and local filterable HTML report are implemented and tested; the
   factor summary distinguishes task-name scene coverage from held semantic
   dimensions that are deliberately not evaluated under the access policy. The
   pinned 66.66GB LIBERO-90 download is complete at exactly 90 files and
   66,658,085,995 bytes; the canonical 90-file audit remains active, followed by
   the remaining specification probes and the finer predeclared
   verb/object/receptacle/relation/order coverage audit. The first full audit
   stopped on a legacy producer-path basename check; canonical HDF5/task-map
   authority remained exact, so the bounded repair records that legacy field as
   provenance while preserving canonical mismatch as fatal. The fresh recovery
   completed all 90 tasks from the repaired clean commit with checksums and
   source-only normalization verified. A language-only role audit then found
   decisive zero/one-source atom violations in the original split before any
   LIBERO-90 policy outcome. The authorized one-time recovery is complete: all
   90 instructions parse under a fail-closed role grammar, the old split remains
   frozen as provenance, and a deterministic seed-20260718 search permanently
   sealed a replacement with 41 evaluation roles all covered by at least two
   source tasks, 30/30 source-unseen full compositions, 30/30 same-scene source
   controls, and 28/30 role-sharing same-scene hard negatives. A fresh canonical
   audit from clean commit `23f3301` now verifies the resealed IDs, 183,555
   source-only normalization rows, 30 metadata-only validation/held tasks, zero
   evaluation numeric access, full checksums, and a filterable role-aware HTML
   view. A source-only same-state native-BDDL probe now establishes an exact
   paired executable-goal surface on tasks 3/4: the MuJoCo state layout is
   identical, 8/8 shared initial states are neutral under both goals, and 16/16
   source demonstration terminal states pass only their originating goal. This
   is evaluator mechanics, not policy behavior. A subsequent cached-observation
   probe isolates a stable language-to-action path on the overlap checkpoint.
   What remains is correct paired-goal behavior with legal source competence
   plus action-hidden video content/temporal controls.
7. [x] Run the bounded evaluation-identity recovery on the same official-overlap
   surface: first compare the frozen policy under one exactly repeated reset
   observation across the predeclared batch ladder, then record matched actual
   reset actions and five-step trajectories. Preserve the strict mechanics
   failure and stop before changing evaluator semantics or observation
   acceptance tolerances. The batch ladder stopped at batch 2 and the mechanism
   probe localized the difference to batch-shape-dependent model forward
   numerics, not input preprocessing or flow-matching noise.
8. [x] Select and freeze one evidence-backed evaluation recovery contract before
   further Gate runs: fixed-batch statistical/functional reproducibility,
   per-sample policy wrapping, or a deterministic-render/precision fork. Run
   matched controls only after this decision; do not reinterpret the strict
   identity failures as passes. The selected contract uses the unchanged async
   evaluator, fixes one measured-safe batch/mode across every arm in a
   comparison, and judges task-level functional estimates with uncertainty;
   batch-1 exactness remains a small audit rather than the primary Gate.
9. [x] Design and permanently predeclare the smallest useful-update oracle
   pilot with explicit pass, diagnosis, and recovery criteria. The canonical
   contract uses the all-60-source base-fit surface at demos 8--27, source tasks
   3/4 for independent rank-8 support at 28--39, fixed-noise selection only at
   40--45, and a report-access freeze before 46--49 or official init 16--23.
   It discloses the prior RGB-only access to 40--47, keeps validation/held
   numeric access at zero, and cannot authorize Writer from a two-task pilot.
10. [x] Run the predeclared official-overlap specification pilot before scaling
    Gate -1: `libero_spatial` tasks 0/1, one fixed async batch of eight per arm,
    correct/no-spec/scene-only/swapped prompts, paired seeds/init states, one
    video per arm, and no Gate decision from this overlap-trained surface. The
    scientific command completed all 64 episodes: correct was 6/8 on each task,
    no-spec was 0/8 on each, scene-only was 2/8 then 0/8, and swapped was 0/8 on
    each. This authorizes only a prompt-path/specification scale candidate on
    the overlap-trained surface. The outer long-run state remains a retained
    wrapper-failure packet because its one-off `EXIT` cleanup returned 1 after
    the canonical command had exited 0 and written checksummed artifacts.
11. [x] Establish a native paired-goal mechanics surface without held access.
    Resealed source tasks 3/4 share one exact MuJoCo layout but have different
    butter-object BDDL goals. Eight shared init states and eight terminal states
    per direction were cross-evaluated with the unmodified pinned native
    evaluator; exact state identity and 16/16 bidirectional specificity pass.
    The 0.80 threshold is descriptive mechanics only and cannot pass Gate -1.
12. [x] Run the smallest same-observation language-to-action causal-path probe.
    On the fixed overlap batch-8 surface, each task is reset exactly once and
    the cached observation/RNG feeds correct/no-spec/scene-only/swapped plus a
    correct repeat. Repeat plans are exactly stable; all 16 samples in every
    comparison exceed the predeclared 0.01 plan-delta scale. This excludes
    repeated-reset rendering as the prompt-effect explanation but still does
    not establish correct paired-goal switching.
13. [ ] Complete the predeclared action-hidden video content and temporal
    controls, then obtain legal source-task policy competence before interpreting the
    paired native-goal surface behaviorally. Keep this Gate -1 dependency
    explicit when designing the Gate 0 oracle pilot. The source-only video
    protocol is now frozen before outcomes: same-scene tasks 3/4, support demos
    0--23, query demos 24--47, reserved demos 48--49, 16 third-person RGB
    frames, one neutral frozen SmolVLM2 encoder, one fixed dual-ridge readout,
    and ordered/reversed/shuffled/first/last/static-median/drop-last-20%
    controls. The first canonical execution is mechanically valid but fails the
    frozen evidence thresholds: ordered balanced accuracy is 0.625 and temporal
    controls do not degrade. Preserve this failure packet and run one bounded
    representation recovery using the same RGB cache, task/demo split,
    readout, thresholds, and model weights: replace only the collapsed final
    causal-context token with fixed framewise visual-connector temporal moments.
    That recovery is now complete: ordered and wrong-video specificity each
    improve to 0.7917, and the temporal-order/static-control criteria pass, but
    only 15/24 paired queries are jointly correct and the two 0.80 content
    thresholds remain unmet. Preserve both readers and stop reader selection;
    the remaining item is legal source competence and closed-loop behavior.
14. [x] Close Gate 0 model mechanics before multi-step training. The first
    adapter invocation failed before forward/backward because preprocessing
    correctly removed provenance that fixed-noise keying still needed; capture
    row keys before preprocessing and retain that implementation packet. The
    recovered probe then exposed that PEFT 0.19.1 `orthogonal` initialization is
    not the contract's stated functional-zero update. Preserve that diagnostic,
    amend only the initializer to nonzero-`A`/zero-`B`, require exact zero
    physical delta and exact base-loss identity, and rerun from a clean commit.
    Recovery from `2d6d3d3` passes: all four initial deltas and the loss delta
    are exactly zero, one step creates finite nonzero updates, and saved deltas
    reload bit-exactly. The independent shared-base one-step path also passes
    with finite loss, gradient, and optimizer update.
15. [x] After exact functional-zero mechanics pass, calibrate source-base
    microbatch candidates `[8,16,32,64]` in one model-loading process for three
    technical steps each. The first run remains valid throughput/memory
    diagnostics (microbatch 64 reached 92.19 samples/s with 61,712 MiB free),
    but it cannot select a launch shape because candidates reused successively
    updated model/optimizer state and accumulation-dependent draws. Run the one
    predeclared matched recovery from identical trainable state, fresh AdamW,
    and identical effective-batch/noise authorities. All three row-key digests
    match across all four candidates; microbatch 64 remains fastest at 92.17
    samples/s with 61,750 MiB free. This freezes only the batch shape, keeps
    effective batch 64, and does not authorize the formal fit before resume
    identity passes.
16. [x] Build and mechanically verify the single canonical source-base trainer,
    including scheduler, Trackio, atomic checkpoints, optimizer/scheduler/RNG
    and sampler-step resume, rotating two recovery checkpoints, one retained
    scientific checkpoint, and a hash-bound runtime/evaluator manifest. Only
    after item 15 is resealed and resume identity passes may the 10,000-step
    all-60-source base fit launch. The canonical trainer/checkpoint path and CPU
    regression suite are implemented. The clean single-GPU real-SmolVLA probe
    now matches full model, optimizer, scheduler, RNG, next raw batch, and next
    row keys exactly across uninterrupted and checkpoint/resume branches; the
    validated 1.32GB transient checkpoint is cleaned.
17. [ ] Launch the now-authorized 10,000-step all-60-source base fit from the
    frozen batch-64 contract, retain two rotating recovery checkpoints plus the
    final scientific candidate, then evaluate only the predeclared source tasks
    3/4 competence surface before any task-local oracle interpretation. The
    single-GPU reference is currently running from clean commit `8ff06f2` and
    must not be interrupted or rebound to new code; it remains the recovery and
    efficiency reference, not evidence that one GPU is the long-term default.
18. [ ] After item 17 commits its final hash, run the outcome-free fixed
    1/2/4-GPU topology probe sequentially on the same available device set.
    Preserve global effective batch 64, optimizer steps, sample/flow authority,
    total loader workers, and scheduler. Freeze the fastest safe topology only
    if it passes the predeclared speedup, efficiency, headroom, global-slot, and
    same-topology resume rules; otherwise use the four-GPU allowance for
    independent arm/task/seed work rather than leaving useful devices idle.
19. [ ] Run the already frozen source-only tasks-3/4 competence comparison on
    the final 10k checkpoint, preferably four-way arm parallel after live GPU
    preflight. A correct-arm minimum pass authorizes only task-local oracle
    fitting; a failure triggers the one bounded identical-contract extension to
    20k. Neither branch passes Gate -1, Gate 0, or authorizes Writer.
