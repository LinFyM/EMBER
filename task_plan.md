# EMBER Execution Plan

## Goal

Advance EMBER from a research design to a reproducible, four-GPU-compatible
experimental program. Establish benchmark validity and useful-update existence,
then train a direct Writer to emit the complete task-specific LoRA allowed by a
common predeclared target-layer/rank contract. Demonstrate immediate utility,
ordinary task-local LoRA RL, source-only Writer reward/meta learning, a
shared-frozen held evaluation, and OpenVLA-OFT scale confirmation after the
mechanism survives.

The 2026-07-18 owner correction supersedes the original Goal text wherever it
made canonical representation or geometry mandatory. Canonical banks, shared
update subspaces, task-conditioned geometry, and residual escape are outside
the current project and long-term Goal, not pending milestones.

## Definition of success for the current goal

- Reproducible environment, upstream revisions, dataset/task manifests, and
  measured systems envelope under at most four A100 80GB GPUs.
- Benchmark/specification validity evidence with causal hard-negative controls.
- A useful-update oracle that improves independent closed-loop behavior, plus a
  bounded LoRA capacity audit against non-matched partial/full-update upper
  bounds so an underspecified LoRA contract is not misdiagnosed as Writer
  failure.
- Documented diagnosis and bounded recovery for every failed or ambiguous gate.
- A direct Writer whose complete generated LoRA improves independent
  zero-interaction behavior over all required matched baselines.
- Matched-budget A/B/C ordinary task-local LoRA RL from zero-LoRA, cold-start
  Writer, and source-reward-outer-trained Writer initializations, with J0, AUC,
  time-to-threshold, J_K, and J_K-J0 reported under identical budgets.
- Source-only reward/meta learning that improves Writer initializations while
  the shared base remains frozen and task-local LoRA is the model-side state.
- A sealed held-task evaluation with base, Writer, and all shared state frozen.
- A reproducible OpenVLA-OFT scale confirmation of the surviving mechanism.

No preparatory milestone completes this long-term Goal. In particular, code or
environment completion, exact resume, throughput calibration, Gate -1, Gate 0,
one successful run, or authorization to start Writer is insufficient. Positive
completion requires the held-frozen Writer and A/B/C evidence above, causal
language/video controls with predeclared seeds and confidence intervals, the
cold-start versus reward-outer Writer comparison, and the scale confirmation.

## Boundaries

- No real robot, human-video transfer, cross-embodiment claim, or arbitrary web
  video in the first study.
- No held actions or shared held-task updates.
- No more than four concurrent A100s for all EMBER work.
- LoRA is the only adaptation mechanism in the current project; do not add a
  bottleneck adapter, IA3, prefix tuning, shared base adapter, shared LoRA, or a
  parallel parameter-efficient path.
- No full-system implementation before its scientific predecessors pass.
- No canonical bank, shared task-update subspace, Writer-predicted basis/mask/
  metric/radius/learning-rate object, soft geometry, or residual escape in the
  current project. Do not reserve implementation paths for them.
- The only common structural search space is the predeclared LoRA target layers
  and rank. Writer initializes all task-local LoRA parameters in that space;
  ordinary RL updates the same parameters without a second constraint object.
- The frozen last-two action-expert q/v rank-8 contract is only the Gate 0
  pilot. Before Writer acquisition, run one bounded source/validation-only
  support audit across the pilot set, all action-expert q/v, and near-official
  SmolVLA PEFT support; then permanently reseal exact targets, rank, alpha,
  dropout, and parameter count. All Writer/RL/direct-generator arms share that
  final contract exactly.
- The shared base stays frozen throughout direct Writer training and the default
  source reward/meta-RL stage. Shared-base/shared-LoRA updates are future
  separate matched ablations, not this plan or its completion criteria.
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
| Stage 1. Direct full-LoRA Writer | pending Gate 0 | independent-query zero-interaction utility over retrieval, average, direct-conditioning, standard-LoRA, and capacity-matched DISC/HyPoGen-style baselines |
| Stage 2. Ordinary task-local LoRA RL | pending Writer utility | matched-budget adaptation AUC/steps-to-threshold from Writer versus standard initialization; no predicted search constraint |
| Stage 3. Source-only Writer reward/meta learning | pending ordinary RL | outer reward improves future Writer initializations while shared base remains frozen |
| Stage 4. Frozen held evaluation | pending complete freeze | sealed primary comparison, full controls, resources, uncertainty, and failure report |
| Stage 5. OpenVLA-OFT confirmation | pending mechanism survival | same mechanism survives a re-pinned four-GPU-compatible scale test |

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
- Infrastructure work is timeboxed to one reproduction, one narrow repair, and
  one verification unless it can change recoverability, sampled data,
  closed-loop success, a Gate decision, matched fairness, or held isolation.
  Ambient RNG digests, telemetry bytes, and bitwise identity are residual
  diagnostics once checkpoint load/state/cursor and short-resume functional
  behavior are operationally sufficient.

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
13. [x] Complete the predeclared action-hidden video content and temporal
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
    legal source competence and closed-loop behavior are now established by the
    frozen tasks-3/4 run: correct prompts achieve 8/8 and 5/8 while both
    same-scene swapped prompts achieve 0/8. This closes the prerequisite but
    does not erase the retained 0.7917 video-content threshold miss or pass
    Gate -1.
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
17. [x] Launch the now-authorized 10,000-step all-60-source base fit from the
    frozen batch-64 contract, retain two rotating recovery checkpoints plus the
    final scientific candidate, then evaluate only the predeclared source tasks
    3/4 competence surface before any task-local oracle interpretation. The
    single-GPU reference completed from clean commit `8ff06f2` with rc 0,
    checksummed step-10000 final state, validated recovery steps 8000/9000,
    Trackio completion, and released resources. It remains the recovery and
    efficiency reference, not evidence that one GPU is the long-term default.
    The frozen source competence surface now passes, completing this item.
18. [x] Close infrastructure selection with at most one necessary short
    four-GPU throughput/stability window, then stop topology benchmarking. The
    clean world-size-4 recovery at commit `9a8a8f5` already returned rc 0 and
    matched model, optimizer, scheduler, global step, data, and all recorded
    resume surfaces under contract SHA256
    `04bf00a4326f62119b32ca22ef9836980d5743e61eb2f1366e85ae4feae25e9d`.
    This is operationally sufficient; no new RNG/identity surface may block
    science. If the one short window has reasonable throughput and headroom,
    use world-size 4 for fitting where beneficial; otherwise fill up to four
    GPUs with independent arm/task/seed jobs through canonical entries. Do not
    rerun a full 1/2/4 scaling curve merely to beautify engineering evidence.
    The single window reached about 294 global samples/s and safe headroom, but
    its step-31 continuous-versus-resumed model hash was not bitwise exact.
    Because the checkpoint validated, loaded, and ran the resumed step, stop
    exact-resume work and use four independent arm/task/seed jobs for long
    science rather than selecting world-size-4 DDP for resumable training.
19. [x] Run the already frozen source-only tasks-3/4 competence comparison on
    the final 10k checkpoint, preferably four-way arm parallel after live GPU
    preflight. A correct-arm minimum pass authorizes only task-local oracle
    fitting; a failure triggers the one bounded identical-contract extension to
    20k. The clean four-GPU arm-parallel result passes without recovery: task 3
    correct 8/8, task 4 correct 5/8, both swapped 0/8, all mechanics valid.
    This authorizes task-local oracle fitting only; it does not pass Gate -1,
    Gate 0, or authorize Writer.
20. [ ] Complete the Gate 0 task-local rank-8 LoRA oracle and LoRA capacity
    audit on the frozen source support/query/locked-report split. Require
    independent query selection, locked closed-loop gain, non-harm/drift
    diagnostics, confidence intervals, and matched zero/base versus task-local
    LoRA controls. Record action-expert partial/full update or full fine-tuning
    only as non-matched capability upper bounds. If only an upper bound works,
    trigger target/rank recovery rather than blaming Writer. The four frozen
    fit/select jobs are complete and checksummed. Task 3 LoRA selected exact
    step 0 because its first improving query candidate exceeded the drift cap;
    task 4 selected step 250 with only a 1.12% fixed-query reduction. Both
    partial-update jobs selected step 0 because trained support improvement
    reversed on independent query data. Keep this item open until the immutable
    selection grant, matched locked offline/closed-loop report, failure
    classification, and one bounded recovery decision are complete. The final
    report now completes rc 0 with all mechanics valid but fails the frozen
    Gate: median success gain 12.5pp, one of two positive tasks, and median
    locked-flow reduction 0.41%. Task 4 improves 1/8 to 3/8 while task 3 remains
    5/8 to 5/8 at its selected zero update; both partial selected states equal
    base. Gate 0 and Writer stay unauthorized. Use the single bounded
    acquisition/target-support audit in item 21 as recovery, then rerun matched
    query and fresh recovery-init closed-loop controls without changing these
    thresholds.
21. [ ] Before freezing the Writer architecture, complete one bounded,
    predeclared source/validation-only target-support audit with held numeric
    access at zero. Compare at least current last-two q/v rank 8, all
    action-expert q/v, and support close to the SmolVLA v0.6.0 default; adjust
    rank only if needed, choose the smallest robust closed-loop oracle support,
    and permanently reseal exact target names/rank/alpha/dropout/count. The
    current Gate 0 pilot is not automatically the winner. Then freeze the
    direct Writer meta-episode and architecture contract. Writer sees language,
    action-hidden video, or both and emits every LoRA factor for that resealed
    space. Primary supervision is
    independent source-query action/flow/behavioral loss through functional
    adapter application; factor MSE is prohibited as primary and oracle-delta
    imitation is auxiliary only if predeclared. Use layer/module-aware,
    chunked/Perceiver-style or type-specific generation if wide support needs
    it; never shrink support merely because a flat generator is inconvenient.
    This item is now the active Gate 0 recovery: predeclare acquisition
    stabilization (lower learning-rate/denser early query candidates or an
    equally bounded remedy) together with the three required support scopes;
    choose only on legal source/query evidence, then use the already reserved
    recovery init states for the closed-loop check. Do not reuse locked report
    demos for selection. The pre-outcome authority is now frozen in
    `configs/gate_zero_target_support_audit.toml` and its concise decision record
    in `docs/gate_zero_target_support_audit.md`: rank-8 supports contain
    40,320/322,560/371,328 parameters, use one lower-LR/dense-early-candidate
    remedy, screen on init states 24--31, and confirm only after support freeze
    on 32--39. Exactly one conditional rank-16 escalation is allowed only when
    no rank-8 support passes; no other layer/rank/optimizer search is scheduled.
    All six rank-8 fits now complete with checksum-valid, drift-safe nonzero
    selections. Median fixed-query reductions rank official-default-like 4.79%,
    all-expert q/v 4.67%, and last-two q/v 2.36%. Immutable screening grant
    SHA256 `fd8e28a7f0b828e14ff7cfb794a047409b6e8e96562646b38aef232b65332992`
    freezes all six states and authorizes only the matched init-24--31
    closed-loop screen; locked report, rank 16, final support, Gate 0, and
    Writer remain unauthorized until that result is classified. The first
    four-rank attempt is retained as an implementation failure: the reused
    locked-report helper performed only one warm-up reset, so all 64 episodes
    actually used init states 16--23 and every arm correctly failed mechanics.
    Its outcomes are quarantined from selection. A single narrow repair now
    derives the exact deterministic warm-up reset count from the frozen target
    batch (two warm-ups before the 24--31 rollout), with regression coverage
    for both the original 16--23 report and recovery surface. Run the untouched
    grant once in a fresh output directory; do not reinterpret or overwrite the
    failed attempt. Recovery run
    `gate0_support_rank8_screen_recovery1_20260718_151530` is now complete rc 0
    with all eight arms mechanically valid on exact init states 24--31. No
    rank-8 support passes: last-two has task gains -12.5/0pp, all-expert q/v
    -25/+12.5pp, and official-default-like -12.5/+12.5pp; their median query
    reductions remain 2.36%/4.67%/4.79% versus 20%. The frozen decision
    authorizes exactly one `official_default` rank-16 fit/screen and nothing
    else. That post-outcome authority is now frozen in
    `configs/gate_zero_target_support_rank16.toml`: exact same 37 targets,
    support/query rows, optimizer, LR, sampler and candidate steps; only
    rank/alpha/count become 16/16/742,656. Screening uses untouched init states
    32--39 and confirmation, only after selection, uses 40--47. A rank-16
    screening failure authorizes no rank/support search under that immutable
    small-recipe contract. Gate 0, final target sealing, confirmation, and
    Writer remain false. Both rank-16 fits and the fresh init-32--39 screen are
    now complete: task 3 and task 4 each improve from frozen-base 2/8 to own-LoRA
    3/8, while fixed-query reductions are 5.76% and 3.27%. The median 12.5pp
    gain and 4.52% query reduction fail the unchanged 15pp/20% criteria. This is
    a mechanically valid negative for the 12-demo, 750-step acquisition recipe,
    not for LoRA capacity or Gate 0.

    The 2026-07-18 owner decision supersedes the old final-negative stop clause
    because no known-positive SmolVLA task-local LoRA behavioral recipe had been
    reproduced. The active bounded recovery is now
    `configs/gate_zero_mature_lora_positive_control.toml` (SHA256
    `882db40dca9ced15cf2b567f9fa57bf2c36c66e64654eef55c067d6485b4b259`):
    tasks 3/4, support demos 0--39 drawn from the three legal source roles,
    independent query demos
    40--45, 20k steps, effective batch 64, the 37 SmolVLA default-like targets,
    rank 32/alpha 16/dropout 0, Gaussian exact-physical-zero initialization,
    SmolVLA AdamW warmup/cosine scheduling, compatible 90--100% random-resized
    crops, fixed step-20k selection, and fresh source init states 40--47. Source
    actions are legal for this oracle but remain hidden from Writer inputs;
    validation/held numeric access is zero. A mechanics-valid primary failure
    permits at most one predeclared all-action-expert-linear rank-32 compatibility
    recovery with unchanged thresholds. Success seals this empirically proven
    full LoRA contract for Writer and all matched arms; failure remains a bounded
    LoRA-capacity diagnosis, not an automatic EMBER negative. Keep items 20 and
    21 open until that closed-loop evidence exists.

    Execute the unchanged mature trajectory through the exact-resume ladder
    1k -> 2k -> 5k -> 10k -> 20k rather than a blind 20k run. The frozen
    operational authority is `configs/gate_zero_mature_lora_stage_ladder.toml`
    (SHA256 `0db007a1e9403902b99e5b6f106f7556d087fe41c188742f94547c986bf6a9eb`).
    Stage 1k may continue to 2k only if both task query reductions are
    nonnegative and their median is at least 2%; later absolute/slope criteria
    are in that file. Every failure stops for bounded mechanics/data/
    augmentation/optimizer diagnosis. Staging uses query only and never consumes
    final init states 40--47. The launch race was stopped by SIGINT after about
    10 volatile steps; both outputs retain validated atomic step-0 candidate,
    optimizer, scheduler, RNG, and state hashes, so resume those outputs rather
    than restarting.
    Stage 1k is now complete rc 0 in 13:03 per task. Task 3/4 independent-query
    reductions are 7.96%/5.75% (median 6.86%), both positive, with drift
    0.01676/0.01883. All candidate/recovery/telemetry hashes validate and no
    final rollout surface was opened. The frozen continuation rule passes;
    exact-resume the same states to step 2k next.
    Stage 2k is now complete rc 0 in 12:59/12:44. Task 3/4 query reductions
    from the unchanged step-0 anchors are 8.04%/4.50% (median 6.27%); both
    remain positive. The median is 0.59 percentage points below stage 1k,
    within the frozen at-most-1pp regression allowance, although task 4 alone
    regressed by 1.25pp and remains a diagnostic signal. Candidate, recovery,
    state and telemetry hashes validate; GPU 4/5 returned to 0MiB and no final
    rollout surface was opened. The predeclared 2k-to-5k rule therefore passes:
    exact-resume these same states to step 5k, then require median reduction at
    least 10%, every task at least 2%, and at most 1pp median regression before
    authorizing 10k.
    Stage 5k completed mechanically rc 0 in 37:32/37:45, but the frozen
    continuation rule failed and this trajectory is stopped. Task 3/4 query
    reductions are 0.31%/-2.96% (median -1.32%), a 7.59pp median regression
    from 2k; drift rose to 0.04563/0.04393 while mean support loss continued to
    fall. Query/anchor identities stayed exact, all candidate/recovery hashes
    validate, and validation/held/final rollout access remained zero. Classify
    this as a mechanics-valid optimization/generalization overrun, not a LoRA,
    Gate-0, or EMBER negative; never resume this primary trajectory to 10k or
    20k. Activate only the already predeclared conditional compatibility
    recovery: rank-32 LoRA over every action-expert q/k/v/o and gate/up/down
    linear plus the five SmolVLA state/action/time projections (117 exact
    targets, 7,027,200 parameters), with all data, optimizer, augmentation,
    isolation, staged thresholds and final Gate thresholds otherwise unchanged.
    This is the sole remaining target-support variant, not the sealed Writer
    support, and it must first pass a one-step target/count/memory smoke and the
    1k query boundary before any continuation or final rollout.
    The sealed fit contract is
    `configs/gate_zero_mature_lora_all_linear_recovery.toml` (SHA256
    `82f5203ed86a25dac386bde68cb8a76efaba03c0f230fe2bd0249bb8d64fe15c`)
    and its result-blind ladder is
    `configs/gate_zero_mature_lora_all_linear_stage_ladder.toml` (SHA256
    `f3b66cff59135f52e81ab9ef387230381662fad6797e5c63557f791bd015739f`).
    Its one permitted live smoke passed: 117 resolved targets and 7,027,200
    parameters, exact-zero physical initialization, finite one-step loss and
    gradient, and 17.93/18.56GiB peak allocation/reservation. The first-row
    digest exactly matches the primary trajectory and GPU 4 returned to 0MiB.
    The two task-1k staged fits are therefore mechanically authorized.
    Both recovery fits reached the atomic step-1k boundary with rc 0 in
    13:35/13:21. Task 3/4 independent-query reductions are 8.381%/1.276%
    (median 4.829%), so both are nonnegative and the median exceeds the frozen
    2% threshold. Task 4's weak positive response is retained as a diagnostic
    signal without changing the aggregate rule. Candidate, trainable-state,
    recovery and telemetry hashes validate; query/anchor identities and sample
    counts exactly match the primary recipe. Peak device memory is
    19,143/19,379MiB and memory-active mean utilization is 89.39%/90.69%; both
    devices were released. No final rollout, validation, or held surface was
    opened. Exact-resume these same states from step 1k to step 2k next, then
    apply only the frozen all-linear 2k-to-5k rule.
22. [ ] Train and evaluate direct Writer zero-interaction utility on source and
    validation surfaces against language-only, video-only, combined,
    wrong/shuffled/reversed/first/last/scene/task-ID controls, average/retrieval,
    direct conditioning, ordinary task-specific LoRA, and capacity-matched
    DISC/HyPoGen-style generation. Freeze all choices before held outcomes.
23. [ ] Run the fixed causal arms: A) zero-LoRA initialization plus ordinary
    RL; B) cold-start Writer LoRA initialization plus identical RL; and C)
    reward-outer-trained Writer LoRA initialization plus identical RL. Update
    the same LoRA parameters in place under identical target layers/rank/count,
    algorithm, hyperparameters, seeds, reward, and interaction budget. Include
    average/retrieval/language-only direct-generator baselines; report J0, AUC,
    time-to-threshold, J_K, J_K-J0, uncertainty, resources, and failures. Add a
    matched-initial-performance or equivalent control before claiming a better
    learning process rather than only a better starting point. Writer emits no
    RL constraint object.
24. [ ] Run source-only reward/delayed outer learning with the shared base
    frozen. Inner adaptation updates task-local LoRA; the outer objective updates
    Writer parameters through a predeclared differentiable path or estimator.
    Shared base/shared LoRA training is not part of the mainline.
25. [ ] Permanently freeze base, Writer, encoders, all shared state, target/rank,
    optimizer, budgets, thresholds, and baselines before held evaluation. Only
    predeclared task-local LoRA may adapt from held reward. Require Writer-start
    to beat zero/base, average, retrieval, and the capacity-matched language-only
    direct LoRA generator with predeclared seeds, confidence intervals, causal
    controls, isolation audit, and reproducible rerun.
26. [ ] After the mechanism survives, re-pin and execute the four-GPU-compatible
    OpenVLA-OFT scale confirmation without changing the causal or held contract.
