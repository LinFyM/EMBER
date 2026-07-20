# EMBER Execution Plan

## 2026-07-20 task-22 result and systems handoff closure

- Task 22 (`close the bottom drawer of the cabinet`) now has a complete matched
  h50 comparison on init states 32--47 and policy RNG seeds 2026072085/86:
  foundation base `0/32`, frozen full-video Writer `4/32`, and task-22
  action-supervised direct LoRA `12/32`. The canonical base/direct packet is
  `/data/ymdai/ember_outputs/foundation_source_screen/task22_base_direct_20260720T105838Z`;
  all checksums and 64 episode identities validate and test/held access is false.
- The canonical evaluator now keeps one async LIBERO worker pool alive across
  matched policy seeds, explicitly rewinds and audits the physical init-state
  cursor, renders only the retained first-worker video, and binds policy plus
  simulator children to the GPU-local NUMA node. Matched mechanics-only
  benchmarks reduced two-seed wall time from `113.59s` to `106.28s` and
  one-video wall time from `66.54s` to `64.32s`; all four benchmark cells passed
  reset/mechanics checks and performance outcomes were intentionally omitted.
- The Writer launcher now defaults to the active foundation/full-video source
  config; validation requires an explicit config instead of silently selecting
  the retired three-frame path. An eight-rank one-step smoke completed rc 0 in
  29.64s, reached 460.87 global samples/s and 66,860.9MiB peak/card, retained
  roughly 14GiB headroom, kept the base frozen, and produced finite Writer
  gradients. No more GPU work is scheduled during handoff cleanup.

## 2026-07-20 task-22 base/direct launch contract

- From clean pushed revision `8ad5be0`, compare the immutable foundation
  SmolVLA base with a task-22 action-supervised direct LoRA. The direct fit uses
  validation episode 8--39 actions, the mature 37-target rank-32/alpha-16/
  dropout-0 space, batch 128, 300 AdamW steps and the already frozen schedule.
  This is an action-supervised task-local upper bound; it does not train or
  evaluate EMBER again.
- Evaluate both arms at h50 on exactly the same init states 32--47 and policy
  RNG seeds 2026072085/86 used by the frozen-Writer probe: 32 episodes/arm and
  64 rows total. GPUs 0--3 run four independent ranks: rank 0 fits direct LoRA,
  ranks 1--2 concurrently collect the two base seed shards, then ranks 0/3
  collect the two direct seed shards. No DDP or sample-budget change is made.
- Canonical output is
  `/data/ymdai/ember_outputs/foundation_source_screen/task22_base_direct_20260720T105838Z`.
  The temporary runner SHA256 is `2c645049...6d63` and copies its exact source
  into the output packet before work. Estimated wall time is under 12 minutes,
  retained growth under 1 GiB, personal usage is 310 GiB under the 500 GiB cap,
  and any mechanics failure stops without reinterpreting partial outcomes.

## 2026-07-20 additional validation probe result

- The ten-minute frozen-Writer probe on eight previously untested validation
  tasks completed in 281 seconds: `4/256` overall, all from task 22 (`4/32`),
  with seven tasks at zero. Preserve this as sparse-transfer evidence; it does
  not justify another evaluation of the unchanged Writer or weaken the current
  diagnosis that validation generalization is the next mechanism problem.

## 2026-07-20 foundation source/validation localization result

- The requested comparison is complete. On all 16 selected source tasks,
  base/Writer/direct scored `0/512`, `55/512`, and `51/512`. On five
  cross-category validation tasks, the same arms scored `0/160`, `1/160`, and
  `18/160`; only direct LoRA was allowed validation teacher actions. Both
  packets use h50, 32 paired rollouts/task/arm, unique episode identities,
  retained videos, and no test/held access.
- This freezes the present diagnosis: the step-300 full-video Writer has clear
  source utility but negligible validation transfer. Do not spend another
  evaluation cycle on the same checkpoints; the next Writer change must answer
  the observed generalization failure while preserving the full-video,
  complete-LoRA and information-isolation contract.

## 2026-07-20 foundation validation three-arm launch

- Evaluate validation tasks 11/21/51/70/86, chosen before outcomes to span
  open-drawer, toggle-then-place, basket pick/place, spatial relation, and
  shelf placement. The source-trained step-300 Writer is frozen and consumes
  only language plus action-hidden episode 8--39 videos. A separate direct
  LoRA per validation task may use episode 8--39 actions and is explicitly an
  action-supervised upper bound, not information-matched to Writer.
- Use the same 37-target rank-32/alpha-16/dropout-0 LoRA space and foundation
  base, h50, 32 paired rollouts/task/arm, five direct-fit ranks plus three
  concurrent base/Writer evaluation ranks, and no test/held access. Retained
  storage is estimated below 3 GiB; exact task-local fit checkpoints and
  atomic evaluation shards permit unchanged resume.
- Canonical command is `scripts/run_writer_cold_start.sh --mode=validate
  --config=/data/ymdai/projects/EMBER/configs/writer_foundation_validation_comparison.toml
  --output-dir=/data/ymdai/ember_outputs/foundation_source_screen/validation_three_arm_20260720T102000Z
  --writer-checkpoint=/data/ymdai/ember_outputs/foundation_source_screen/writer_fullvideo_source16_20260720T080457Z/checkpoints/000300`.

## 2026-07-20 foundation-source three-arm evaluation launch

- Canonical workspace is this checkout on branch
  `phase0/reproducible-substrate`; launch only from the committed and pushed
  revision containing `configs/writer_foundation_source_comparison.toml`.
- Compare the immutable SmolVLA foundation base, step-300 full-video Writer,
  and sixteen immutable step-300 action-supervised direct LoRAs on all 16
  selected source tasks. Use official SmolVLA execution horizon 50, 32 paired
  rollouts/task/arm (16 init states by two policy RNG seeds), 1,536 episodes
  total, binary simulator success plus paired intervals, and no
  validation/test/held access.
- Run eight independent evaluation ranks on physical GPUs 0--7. Each rank sees
  one CUDA device and owns sixteen asynchronous simulator environments; 48
  task-arm shards are balanced six per rank. This changes scheduling and
  denominator only, not any frozen model, LoRA capacity, task, arm, evaluator,
  or metric. Estimated wall time is 20--35 minutes and additional retained
  storage is below 1 GiB.
- Canonical command is `scripts/run_writer_cold_start.sh --mode=validate
  --config=/data/ymdai/projects/EMBER/configs/writer_foundation_source_comparison.toml
  --output-dir=/data/ymdai/ember_outputs/foundation_source_screen/source_three_arm_eval_v2_20260720T094500Z
  --writer-checkpoint=/data/ymdai/ember_outputs/foundation_source_screen/writer_fullvideo_source16_20260720T080457Z/checkpoints/000300`.
  Complete shards are atomic and may be resumed unchanged with `--resume`;
  incompatible or failed roots remain provenance and are never mixed into the
  published result.

## Goal

Advance EMBER from a research design to a reproducible, eight-GPU-ceiling-compatible
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
  measured systems envelope under at most eight A100 80GB GPUs.
- Benchmark/specification validity evidence with causal hard-negative controls.
- A useful-update oracle that improves independent closed-loop behavior, plus a
  bounded LoRA capacity audit against non-matched partial/full-update upper
  bounds so an underspecified LoRA contract is not misdiagnosed as Writer
  failure.
- Documented diagnosis and bounded recovery for every failed or ambiguous gate.
- A direct Writer whose complete generated LoRA clearly improves independent
  zero-interaction behavior over frozen base across multiple task categories;
  its gap to the same-space action-supervised direct-LoRA upper bound guides
  optimization but is not an automatic rejection rule.
- A separate source-only Writer-RL result after supervised cold start: the
  shared base stays frozen, the generated LoRA is used functionally but is not
  optimized as an independent inner variable, and rollout reward updates only
  Writer parameters. This tests whether reward improves the LoRA that the
  Writer emits immediately.
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
- No more than eight concurrent A100s for all EMBER work. Prefer independent
  task/arm/evaluation-shard/training-seed parallelism through the existing
  canonical entrypoint when a single four-rank job does not use all devices.
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
- The last-two q/v rank-8 setting was only a Gate-0 pilot. The bounded support
  audit is complete and the active Writer/RL/direct-generator contract is
  permanently resealed to the mature 37-target, rank-32, alpha-16, dropout-0,
  1,485,312-parameter LoRA space for every compared arm.
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
| Gate -1. Benchmark/spec validity | passed with recorded residuals; no longer a Writer blocker | immutable 19/24 ordered and wrong-video, 15/24 paired, original 0.80 threshold and drop-last sensitivity preserved |
| Gate 0. Useful-update oracle | passed for development with limited coverage | task3/task4 SFT-LoRA improves over base; retain the near-similar-task and n=32/arm limitation |
| Stage 1. Direct full-LoRA Writer | in progress | independent-query zero-interaction utility across several validation categories, with frozen-base and matched direct-LoRA comparisons |
| Stage 2. Ordinary task-local LoRA RL | pending Writer utility | matched-budget adaptation AUC/steps-to-threshold from Writer versus standard initialization; no predicted search constraint |
| Stage 3. Source-only Writer reward/meta learning | pending ordinary RL | outer reward improves future Writer initializations while shared base remains frozen |
| Stage 4. Frozen held evaluation | pending complete freeze | sealed primary comparison, full controls, resources, uncertainty, and failure report |
| Stage 5. OpenVLA-OFT confirmation | pending mechanism survival | same mechanism survives a re-pinned eight-GPU-ceiling-compatible scale test |

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
- The completed Gate-0 collection treated every n<32 run as mechanics-only and
  did not inspect its performance. That historical rule prevented small-sample
  checkpoint or seed selection; it is not a universal denominator for later
  EMBER stages. Each new validation contract chooses enough independent
  rollouts from expected effect and observed variance, retains episode rows and
  uncertainty, and covers multiple categories and policy RNG seeds.
- Both Gate-0 training seeds are frozen at step 32 and may not receive more
  interaction. The formal paired development evaluation in
  `configs/gate_zero_formal_development_evaluation.toml` is complete with four
  policy RNG seeds, 32 rollouts/task/arm, h16 primary, h50 robustness, fixed
  initialization arms evaluated once, and no partial performance report.
- Use horizon 16 for RL training and the primary Gate comparison; report
  horizon 50 separately as canonical/deployment robustness. Call the existing
  mean-before-ratio implementation the custom chunk-level flow-loss PPO pilot.
  A valid negative cannot represent ordinary LoRA RL until the bounded faithful
  per-flow-sample/group-size-one FPO++ core is also tested.
- Infrastructure work is timeboxed to one reproduction, one narrow repair, and
  one verification unless it can change recoverability, sampled data,
  closed-loop success, a Gate decision, matched fairness, or held isolation.
  Ambient RNG digests, telemetry bytes, and bitwise identity are residual
  diagnostics once checkpoint load/state/cursor and short-resume functional
  behavior are operationally sufficient.

## Current next actions

### 2026-07-19 formal Gate 0 decision boundary

- [x] Collect the first admissible Gate 0 performance packet from the two
  frozen step-32 training seeds. Seed `2026071830` completed 512 rows on GPUs
  4--7; after the owner raised the project ceiling to eight GPUs, seed
  `2026072030` ran concurrently on GPUs 0--3 and completed 256 rows. The
  aggregate contains 768 unique source-development rows, n=32 in every declared
  task/arm/training-seed/horizon cell, four policy RNG seeds, paired episode
  identity, and h16/h50 results. Fixed base/SFT rows have `training_seed=null`.
- [x] Verify the packet before interpretation: both run rc values are zero,
  checksums and JSON pass, 24 videos decode, no grain-key duplicates or init
  hash inconsistencies exist, and an independent in-memory recomputation
  exactly matches the saved result.
- [x] Owner interpretation supersedes the former strict-CI decision: Gate 0 is
  passed for rapid development from the positive SFT-LoRA point gains, while
  retaining the two-near-similar-task and n=32/arm limitation. Gate -1 remains
  passed with residuals. Do not rewrite either immutable result packet.
- [x] Run the first Writer cold start through the frozen
  `configs/writer_cold_start.toml` contract. The eight-GPU step-1000 segment
  completed in 48.2 minutes, then 1,920 paired validation rows covered five
  categories, 64 rollouts/task/arm and horizons 16/50. At h16, base/Writer/
  direct-LoRA scored `33/320`, `21/320`, and `150/320`; at h50 they scored
  `18/320`, `21/320`, and `140/320`. The Writer therefore failed the current
  cold-start criterion, while the matched LoRA space and direct acquisition
  remain clearly capable.
- [x] Run the single predeclared physical-update recovery in
  `configs/writer_cold_start_physical_norm_recovery.toml`. Offline independent
  query diagnostics found that the original Writer reduced loss on all five
  validation tasks but emitted physical updates of norm about `30--39`, versus
  `1.41--2.00` for successful direct LoRAs. A post-hoc `0.05` factor scale
  restored direct-like norms but erased nearly all query gain, so it is not a
  rollout candidate. Retrain once from the prescribed initialization with a
  soft `2.0` physical-norm cap and `0.01` excess penalty. The eight-GPU segment
  completed rc 0 in 48.8 minutes and kept updates near the cap. On independent
  query demos, step 250 improved all five categories by 7.5%--14.4% (10.35%
  unweighted mean) with norms 1.96--2.33, so it is the frozen rollout candidate.
- [x] Evaluate only the frozen step-250 recovered Writer on the same five
  categories and 64 paired rollouts/task at h16/h50. Reuse hash-bound base and
  direct-LoRA episode rows from the completed packet; do not repeat their
  rollouts or fits. The 1,920-row packet completed rc 0. At h16 base/Writer/
  direct scored `33/320`, `31/320`, `150/320`; at h50 they scored `18/320`,
  `21/320`, `140/320`. Norm control preserved offline transfer but did not
  produce cross-category closed-loop utility, so Writer-only RL remains blocked.
- [x] Fit matched direct-LoRA teachers on the frozen 15-task/five-category
  source subset. All 15 fixed-step fits completed in two GPU-parallel waves;
  their state hashes are bound by `teacher_bundle.json`, and physical update
  norms span `0.703--0.955` (median `0.797`). No validation/test/held numeric
  surface was used.
- [x] Complete the single frozen source-teacher auxiliary Writer recovery in
  `configs/writer_cold_start_source_teacher_auxiliary_recovery.toml`. Independent
  source-query functional loss remains primary; coefficient `0.1` applies only
  to gauge-invariant relative physical-Delta-W error, raw factor MSE is absent,
  and a `1.25` soft norm cap is a runaway safeguard. Keep the full 37-target
  rank-32 space and leave later task-local RL unconstrained.
  The eight-GPU step-1000 segment completed rc 0 in 48.7 minutes with four
  validated exact-resume checkpoints. A five-category validation-query rule
  frozen before result access selected step 500. Its complete 64-rollout/task
  closed-loop packet did not establish utility: at h16 base/Writer/direct were
  `33/320`, `32/320`, and `150/320`; at h50 they were `18/320`, `22/320`, and
  `140/320`. Writer success remained confined to the spatial-relation task.
  A no-rollout source-teacher reconstruction diagnostic, frozen before its
  outcome, found mean normalized physical-update errors of `0.741`, `0.555`,
  `0.490`, and `0.466` at steps 250/500/750/1000. The best value remains above
  the predeclared `0.25` underfit boundary, while step-1000 validation-query
  improvement remains positive on all five categories and averages `7.12%`.
  The exact same eight-rank trajectory then resumed once from step 1000 to
  2000, rc 0 in 48.9 minutes. Source-teacher error improved only to `0.420`,
  still above the frozen `0.25` boundary; all five validation query tasks stayed
  better than base but mean improvement was flat at `7.04%`. Step 2000 is not
  rollout-eligible and step scaling stops. Run one fresh-objective recovery in
  `configs/writer_cold_start_source_teacher_weight_recovery.toml`: change only
  the gauge-invariant teacher auxiliary coefficient from `0.1` to `0.3`, keep
  functional query loss primary and every model/data/LoRA/norm-cap field fixed,
  and evaluate offline before authorizing any rollout.
  That fresh eight-GPU step-1000 segment completed rc 0 in 49.2 minutes. The
  best source-teacher error was still `0.418`, so no checkpoint qualified for
  validation closed-loop; all five validation-query tasks remained better than
  base with `6.93%` mean reduction. Per the owner's localization request,
  freeze step 1000 only as a source-diagnostic checkpoint and compare frozen
  base, Writer, and immutable direct teacher on source tasks 6/19/46/34/73,
  64 paired rollouts/task/arm at h16/h50. This result may diagnose acquisition
  versus generalization but cannot retroactively make the checkpoint eligible.
  The complete source packet has 1,920 rows and passes all hashes. At primary
  h16, base/Writer/direct teacher score `141/320`, `127/320`, and `137/320`;
  at h50 they score `104/320`, `101/320`, and `111/320`. Writer therefore also
  lacks aggregate source-task utility, while these source teachers themselves
  are not a reliable closed-loop upper bound. The failure is not merely
  validation generalization. Before another Writer recovery, establish a
  behaviorally useful direct source-LoRA target on multiple categories using
  the existing mature LoRA path; do not start Writer-only RL from this
  checkpoint.
- [x] Correct the eight-rank Writer evaluator's device topology. The previous
  job put each policy rank on its intended CUDA device but left every
  MuJoCo/EGL worker on physical GPU0. Bind both PyTorch and
  `MUJOCO_EGL_DEVICE_ID` from `LOCAL_RANK` before project imports. A real
  eight-rank EGL smoke now shows exactly one matching `C+G` process on each
  GPU; use this path for subsequent validation rather than accepting GPU0
  renderer contention.
- [ ] Screen the ten remaining immutable source teachers before retraining any
  target or Writer. Contract
  `configs/writer_cold_start_source_teacher_coverage_screen.toml` binds tasks
  23/35/20/38/52/54/37/69/87/89, three frozen arms, disjoint init/policy RNG
  identities, h16 only, 64 paired rollouts/task/arm, 1,920 total rows, and the
  corrected eight-rank CUDA/EGL topology. Read no partial performance; use the
  complete result only to decide whether the existing teacher bundle already
  contains useful targets across categories or requires a data/fit recovery.

1. [x] Bootstrap a Python 3.12 environment from the locked project definition;
   verify package consistency and the active GPU/storage contracts.
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
    Stage 2k completed rc 0 in 13:04/13:08, but the recovery trajectory now
    stops permanently. Task 3/4 query reductions reverse to -3.149%/-12.093%
    (median -7.621%), a 12.450pp median regression from 1k. This fails all three
    frozen 2k-to-5k conditions: median at least 5%, every task nonnegative, and
    median regression at most 1pp. Candidate/recovery hashes and query identity
    validate, support loss continues downward, and drift rises to
    0.05816/0.06011, so the bounded classification is mechanics-valid
    optimization/generalization overrun rather than implementation failure.
    Never resume this recovery to 5k/10k/20k and do not consume its formal
    closed-loop surface. No further target/rank variant is authorized. Preserve
    both mature failures, then execute only the already predeclared non-matched
    task-local action-expert capacity upper bound needed to distinguish LoRA
    space from acquisition/data/optimization failure; surface the Gate-recovery
    decision afterward without changing thresholds or blaming Writer.
    That final diagnostic is now predeclared in
    `configs/gate_zero_mature_action_expert_upper_bound.toml` (SHA256
    `8fd7f3a5fac0bbfef6fb7281e48b7ef9df7e5b95a74e9446d1e4c8e8ed72327d`)
    with staged ladder SHA256
    `69640a07e97915e9ac51ac31153d13f4df4e3154845afdb2a136def230f4bc98`.
    It reuses the exact 40 support demos, independent query, effective batch,
    sampler/noise, augmentation and optimizer schedule, but trains the existing
    99,880,992 action-expert/projection parameters instead of LoRA. It is
    explicitly non-matched and cannot pass Gate 0, seal Writer targets, or
    authorize Writer. Run one exact-identity/count/finite-step smoke, then only
    tasks 3/4 to the 1k query boundary; continue under the same frozen ladder
    and never consume the formal closed-loop surface during staging.
    The sole smoke passes on one A100: 99,880,992 trainable parameters in 155
    tensors, first-row digest identical to both mature LoRA recipes, finite
    loss/gradient, 62.3 samples/s, and 17.86/18.93GiB peak allocation/reservation.
    It saves no candidate and opens no rollout/validation/held surface; GPU 4
    returns to 0MiB. The two task-1k staged fits are mechanically authorized.
    Both task-1k fits complete rc 0 in 13:19/13:06 but fail the first frozen
    continuation rule and stop permanently. Task 3/4 query reductions are
    -5.221%/-19.788% (median -12.504%), with drift 0.06313/0.06903, while
    support loss falls substantially. Candidate/recovery/telemetry hashes and
    query identity validate; GPU 4/5 return to 0MiB and the 2.0GiB output is
    retained. This mechanics-valid result shows that the current mature
    task-local acquisition/data/optimization recipe fails to generalize even in
    the non-matched action-expert space; it does not isolate LoRA capacity and
    cannot be promoted to a LoRA, Gate 0, Writer, or EMBER negative. The frozen
    contract now requires a Gate-recovery decision before any new recipe. Do
    not resume to 2k or consume closed loop. Present the owner with the bounded
    choice between one result-blind lower-LR/dense-early acquisition recovery
    (upper bound first, then matched LoRA only if the upper bound becomes
    positive) and recording the current task-local supervised acquisition
    surface as insufficient and revising the Gate 0 evidence plan; no silent
    third target/rank or hyperparameter grid is allowed.
    A post-hoc, source-query-only physical-update scale probe now resolves that
    decision without new training or closed-loop access. Its scale-0 and
    scale-1 endpoints exactly reproduce both saved candidates; at scale 0.25,
    task 3/4 query reductions become +2.705%/+1.311% (median +2.008%), while
    scale 0.5 already makes task 4 negative. This is diagnostic evidence that
    the learned direction contains utility but the 1k update magnitude
    overruns; interpolation is explicitly not equivalent to lower-LR training.
    Freeze one final optimization-only recovery in
    `gate_zero_mature_action_expert_lr_recovery.toml` (SHA256
    `30e2c575...a3c7c`): same data, sampler/noise, augmentation, AdamW, seed and
    99,880,992 parameters, but peak/decay LR scaled by 0.25 to
    2.5e-5/6.25e-7. Run the same trajectory at 250/500/750/1000 boundaries
    under ladder SHA256 `6947a9a8...ebc2c`; stop at the first failed rule. A
    step-1000 pass authorizes only the identical schedule on LoRA, not Gate 0,
    Writer, action-expert rollout, another LR trial, or a target/rank search.
    The exact-resume action-expert ladder has now completed through step 1000
    and stops there. Task 3/4 query reductions at steps 250, 500, 750, and 1000
    remain positive; step 1000 is +7.883%/+3.710% (median +5.797%), above the
    frozen 2%/nonnegative criterion, although it regresses from the step-750
    median of +6.974%. All stage commands return 0, final candidate/recovery/
    telemetry hashes validate, GPU 4/5 are released, and no formal rollout,
    validation, or held surface was opened. This proves only that the bounded
    acquisition schedule can improve independent query loss in the non-matched
    action-expert space and authorizes no action-expert 2k continuation.
    The matched 37-target rank-32 LoRA recovery is now permanently predeclared
    in `gate_zero_mature_lora_lr_recovery.toml` (SHA256
    `693cd61457ec5ec0aafb1c72837899c58f44f2c90e812cb14d115385393fafca`)
    with ladder SHA256
    `436bae48c5c9f754346b18bd424378d70eee5900eb0add96f6a4ea99104817d3`.
    It changes only the matched LoRA state class relative to the successful
    action-expert schedule and uses the same 250/500/750/1000 continuation
    rules. Run tasks 3/4 first to step 250 on two GPUs, exact-resume only after
    each frozen rule passes, and do not consume closed loop during staging.
    Even a step-1000 pass authorizes only a separately predeclared headroom-safe
    source rollout contract: source task 3 is already 8/8 under the base and
    cannot satisfy a positive-gain-on-both-tasks rule.
    The matched LoRA step-250 segment now completes rc 0 on both tasks in
    3:54/3:53. Query reductions are +0.892%/+1.038% (median +0.965%), with
    drift 0.00167/0.000925; both per-task and median nonnegative conditions
    pass. Candidate/recovery hashes validate, peak memory is
    18,485/18,545MiB, GPU 4/5 are released, and the output is only 58MiB.
    Exact-resume these same states to step 500; do not start a new trajectory
    or access closed loop.
    Step 500 also completes rc 0 by exact resume in 3:45/3:41. Task 3/4 query
    reductions rise to +3.388%/+2.611% (median +2.999%) with drift
    0.00572/0.00495, passing the frozen 0.5%-median/nonnegative rule. Artifacts
    validate and GPU 4/5 again release; exact-resume to step 750 next.
    Step 750 completes rc 0 in 3:43/3:44 and improves further to
    +4.925%/+3.728% (median +4.326%) with drift 0.00768/0.00684. This passes
    the frozen 1%-median/nonnegative rule; exact-resume the same states to the
    final authorized source-query boundary at step 1000.
    Step 1000 completes rc 0 in 3:46/3:45 and passes the frozen offline-success
    criterion: task 3/4 query reductions are +5.798%/+4.236% (median +5.017%),
    both positive, with drift 0.01056/0.00858. Candidate/recovery/telemetry
    hashes validate, GPU 4/5 release, and the full output is 92MiB. Stop this
    ladder at 1000; it authorizes only a separately frozen, headroom-safe
    source closed-loop contract. Do not resume to 2k or authorize Gate 0/Writer.
    The owner approved the source-only ceiling-aware Proposal A on 2026-07-19
    Asia/Singapore before any new LoRA closed-loop outcome. The active
    `gate_zero_mature_lora_headroom_screen.toml` SHA256 is
    `1f92f80ddcc63be7c6a3ef3da1fe63f9870df27a0537ec02c3429bab71440a52`;
    pending proposal SHA256 `8c7ae12b...075d5c` and commit `108ce65` remain
    provenance.
    It binds both immutable step-1000 candidate/recovery/telemetry packets and
    uses paired fresh source init states 40--47 with seeds 5800--5807. Task 3
    is a maintenance control requiring nonnegative paired net wins; task 4 is
    the improvement control and must expose at least two frozen-base failures
    and gain at least two paired net wins; aggregate net wins must be at least
    two. Both query reductions must remain at least 2% and both drift proxies
    at most 0.02. If task 4 exposes fewer than two failures, classify headroom
    as absent and predeclare a source-task extension rather than lowering the
    rule. The completed two-GPU run
    `gate0_mature_lora_headroom_owner_a_20260719_025534` returned rc 0 and
    exposed real headroom on both tasks: frozen base was 3/8 on task 3 and
    3/8 on task 4. Step-1000 LoRA was 2/8 and 4/8, respectively, so paired net
    wins are -1/+1 and aggregate zero. Query and drift safeguards pass, but the
    maintenance, improvement, and aggregate behavioral checks fail. Gate 0,
    target sealing, and Writer authorization remain false. Result SHA256 is
    `84116faa...c98f`; all result, gallery, video, freeze, and telemetry hashes
    validate, wall time was 171.95s, and GPU 4/5 released.
    A later Option-B instruction to replace task 3 raced with this run and was
    withdrawn once the owner learned that task 3's fresh base is 3/8 rather
    than the old competence slice's 8/8. No replacement selection rollout was
    launched. Its four-file WIP is recoverable only as local stash
    `201d097e...1476` and is not an active repository path. Keep tasks 3/4;
    do not change task, threshold, or seed to evade the failed A result.
    The active cheapest discriminating recovery is the pre-outcome candidate-
    step diagnostic in
    `gate_zero_mature_lora_candidate_step_diagnostic.toml` (SHA256
    `4445664f...6a46`): reuse A's base and
    step-1000 arms, evaluate only existing exact step-500/750 LoRA states on
    the same source development slice, and test update-duration/magnitude
    overrun. Both tasks must improve, median gain must retain the original
    15pp rule, and every query reduction/drift must retain 2%/0.02 safeguards;
    select maximum aggregate net wins with an earliest-step tie-break. This
    diagnostic cannot pass Gate 0. Only a passing candidate may receive a
    separately hash-bound fresh-seed matched recovery Gate; no default 2k
    continuation is authorized.
    The canonical diagnostic
    `step500_750_20260719T035642Z` completed main rc 0 on clean `19e5ea2`.
    Both earlier states reproduce the same source-development behavior:
    task 3 is 2/8 versus the frozen 3/8 base and task 4 is 3/8 versus the
    frozen 3/8 base. Thus each step has zero positive tasks, -6.25pp median
    gain, and aggregate paired net -1. The pre-existing query and drift
    safeguards pass, but the behavioral rules fail; result SHA256 is
    `aae6e19f...b11cf`. No checkpoint is selected, no fresh recovery Gate is
    granted, and Gate 0/Writer remain false. This bounded evidence does not
    support step-1000 update magnitude or duration as the sufficient failure
    explanation, so stop the supervised trajectory and proceed to the matched
    ordinary-RL recovery below.
    If this same-task candidate diagnostic still lacks credible closed-loop
    positive evidence, do not reinterpret Gate 0 as an SFT-only Gate. Before
    any new outcome, freeze one small-budget source-only ordinary-RL recovery
    on the unchanged task-3/task-4, LoRA, evaluator, init-state, seed, and
    success contract. Its four matched arms are frozen base; supervised LoRA;
    zero-init LoRA plus ordinary task-local RL; and supervised-LoRA-init plus
    the identical ordinary task-local RL. The two RL arms must share estimator,
    optimizer, reward, interaction budget, compute accounting, and all LoRA
    structure; only initialization differs. Start with a 10--30 minute early
    check and keep each resumable segment within one to two hours. If only RL
    arms improve, conclude only that the LoRA space contains a useful RL
    oracle; support for helpful initialization requires supervised-init plus RL
    to beat matched zero-init plus RL. No Writer participates in this Gate-0
    recovery, and no task/seed/threshold change, validation/held/locked access,
    or arbitrary 2k/5k continuation is authorized.
    The original pre-outcome recovery was sealed in
    `configs/gate_zero_task_local_rl_recovery.toml` at SHA256
    `75ceeec398f472d53fb1c7b88b4dd135469b0f841bbf8ac3dfc0ac4b13cd5c68`.
    It uses episodic Monte-Carlo AWR-style reward-weighted flow regression,
    because SmolVLA exposes a per-sample flow loss but no trustworthy exact
    action likelihood: binary source success, an in-batch mean baseline,
    temperature 0.5, deterministic Gaussian raw-action exploration, eight
    optimizer updates per eight-episode round, and fresh identical AdamW state
    for both initializations. Four independent ranks run task 3/4 crossed with
    zero/supervised initialization; there is no cross-rank gradient reduction,
    critic, Writer, shared update, or demonstration action in the RL optimizer.
    Atomic nodes are 16 and at most 32 episodes per task/initialization. The
    source development slice can only stop/continue/select a candidate for a
    separately hash-bound fresh Gate; it can never pass Gate 0 itself. The
    launcher retains Trackio, one video per arm/node, bounded galleries,
    telemetry, and about 10GiB device headroom.
    Its first four-card collection failed closed before any optimizer update or
    development rollout. A repeated task-4/zero-init training slice proved the
    reset-after IDs were exactly 8--15, while successful sub-environments were
    legitimately auto-reset before the end-of-rollout ID read. It also localized
    1195/1208 clipped scalars to the binary gripper dimension versus only 13 in
    all six continuous dimensions. The active mechanical recovery retains
    Gaussian std 0.05 on dimensions 0--5, preserves the policy gripper command
    on dimension 6, audits identity at reset, and leaves every scientific field
    and threshold unchanged. The amended config SHA256 is
    `e138b7d649c192d4618a8e5b9c0f8fe29b60c95a5117815313f271f405d4d406`;
    it binds the predecessor, failure packet, and localization packet before
    rerun.
    Recovery2 passed those collection guards but failed before its first
    optimizer update because the replay action is 7D before preprocessing while
    SmolVLA pads the model action tensor to 32D; deterministic flow noise had
    incorrectly retained the preprocessor input width. This is a single tensor-
    shape implementation error, not an RL result. The final bounded repair
    derives and validates noise shape `[50,32]` from the processed action tensor;
    all scientific fields remain unchanged. Active contract SHA256 is
    `504d20bc371078b5ffeabaad84eb1e041423c5167cd7331b91e047a3324f673d`,
    binding both prior contracts and failure packets. Run one final canonical
    verification; any further mechanical failure stops this implementation
    attempt for diagnosis rather than triggering another blind rerun.
    That final verification failed at its explicit shape guard and exposed the
    remaining layer boundary: preprocessing correctly preserves replay actions
    as `[64,50,7]`; `SmolVLAPolicy.prepare_action` pads to the frozen
    `max_action_dim=32` inside `model.forward`. The repaired code now validates
    the 7D input and derives noise width 32 from the pinned model config. Active
    contract SHA256 is
    `b08a85b8de1bf04c788d217cfab8d34bb984d0f70ab8795e8c0aaf0f19820a37`.
    Before any environment rerun, require a real-model synthetic-batch
    forward/backward with noise `[64,50,32]`, no optimizer step, and no
    environment/validation/held access. Do not relaunch the canonical rollout
    unless this exact integration check passes.
    That integration check passed from clean commit `cd95342` in 29.06 seconds:
    input `[64,50,7]`, noise `[64,50,32]`, per-sample loss `[64]`, finite loss
    and gradient norm, rc 0, and no optimizer/environment/validation/held
    access. Result SHA256 is `64e522b8...a689c`. This reopens only the already
    frozen 16-episode source-development verification; it is not Gate-0 or
    behavioral evidence.
    The canonical stage-16 recovery then completed rc 0 from clean commit
    `a581eea` in 4:38.96. All checksums and mechanics passed, all four arms made
    16 finite optimizer updates, and reward varied, but each arm reproduced its
    own initial development success count: task 3 supervised 2/8 -> 2/8 and
    zero 3/8 -> 3/8; task 4 supervised 4/8 -> 4/8 and zero 3/8 -> 3/8. Thus
    paired net wins are zero for all arms, the supervised-init advantage is
    -1/+1 across tasks, and `task_local_rl_early_check_not_supported` stops the
    trajectory at 16 without stage 32 or a fresh Gate. Result SHA256 is
    `aab151ea...b57`. Treat this as bounded negative evidence about this small-
    budget AWR-style estimator, not a final negative about LoRA or EMBER.
    Only the step-16 model/recovery state is retained; step 8 has round metrics
    but no loadable weights. Before any further source interaction, use the
    retained step-16 state plus round-8/round-16 reward, loss, gradient, query,
    and drift evidence to diagnose whether the update is merely too weak or
    directionally unhelpful. Freeze any later optimizer/estimator recovery
    before outcomes and do not reopen stage 32 by default.
    The cheapest directional discriminator is now frozen before outcomes at
    config SHA256
    `d322339eb417536a8b96b124b3c8d6324c4b25b95e89f4a3cffb5d6cadce200c`.
    It replaces only the ineffective AWR optimizer objective with one bounded
    per-sample conditional-flow-loss-ratio check using signed within-batch
    binary-reward advantages: successful trajectories are pulled toward the
    policy while failed trajectories are pushed away. It is anchored to the
    FPO++ paper and official code commit
    `b80112be1e8362263c4cd176e7aef21a275ff1c6`, but explicitly omits the
    critic, GAE, many-flow-sample estimator, entropy term, and long training,
    so it is not a full FPO++ reproduction or performance claim. Tasks 3/4,
    four initial/paired arms, 37-target rank-32 LoRA, initialization states,
    source interaction/development slices, seeds, exploration, reward,
    optimizer hyperparameters, 16-episode budget, query/drift guards, and Gate
    thresholds remain unchanged. There is no stage-32 fallback. Focused/full
    tests and the required no-environment real-model forward/backward have
    passed: 64 unique source rows, exact repeated old/current loss, finite
    gradient norm 0.0803, no optimizer step, no simulator, and result SHA256
    `b1e75b43...b4687`. This permits exactly one four-GPU source-only 16-episode
    run. That run completed mechanically but did not select a candidate:
    supervised-init stayed 2/8 and 4/8 on task 3/4; zero-init changed 3/8 to
    2/8 and stayed 3/8. The status is
    `task_local_rl_early_check_not_supported`, result SHA256 is
    `73d681ca...8703`, and stage 32/fresh Gate remain closed. Do not increase
    signed update steps blindly: its physical LoRA displacement was already
    about 10% of the supervised physical-update norm, while AWR produced about
    17%, yet neither changed behavior positively. The cheapest next
    discriminator is the already mandated non-matched lower-LR action-expert
    capacity upper bound on the exact same source-development slice. Its
    result-blind contract is frozen at SHA256
    `e313e437fe57f20d2cd390fbede0c89432bb89f1d40dd7d37bcf8156e1af9f3a`:
    exact 99,880,992-parameter state hashes, tasks 3/4, init states 40--47,
    seeds 5800--5807, warmup 5792, policy seed 2026071836, paired 3/8 base
    vectors, and the original two-positive-task/15pp-median rule. Run exactly
    16 new source episodes on two GPUs with bounded videos/gallery/Trackio.
    If it converts its
    positive independent-query evidence into behavior while LoRA does not,
    classify the LoRA contract/acquisition as too narrow; if it also fails,
    classify query-to-closed-loop conversion or credit assignment as the
    primary bottleneck before considering a materially fuller flow-policy RL
    method. This non-matched diagnostic cannot pass Gate 0, seal LoRA targets,
    or authorize Writer; neither branch weakens the full EMBER objective.
    No validation, held, locked-report, or step-2k access is authorized. The
    original SHA256 `ba3ee431...f132f`
    reached no episode because its last-warm-up seed was not stride-adjacent to
    the unchanged report seed 5800; retain that failure packet and use the
    corrected last warm-up seed 5792 without changing any scientific surface.
    The corrected run completed main rc 0 in 1:37.99 from clean commit
    `7e5f905`; all checksums, mechanics, two videos/gallery, Trackio, and
    telemetry validate. Task 3 exactly matches its paired base vector at 3/8;
    task 4 exchanges one success for one failure and also remains 3/8. Both
    paired net wins and the median gain are zero, so status is
    `nonmatched_action_expert_capacity_behavioral_signal_absent`, result SHA256
    is `9a91fbb8...170ad`, and Gate 0/Writer remain false. The wider
    query-positive partial-update state therefore does not support “LoRA
    parameter count alone” as the current bottleneck. Stop the target/rank/
    supervised-step route and diagnose the action/flow-query-to-closed-loop
    and temporal-credit mismatch on these same source tasks. Any further RL
    mechanism must be frozen before outcomes, preserve the four matched Gate-0
    arms and fast resumable cadence, and cannot be called mature FPO++ unless
    its critic/temporal-credit estimator and other claimed essentials are
    actually implemented. No validation, held, locked-report, arbitrary
    step-2k continuation, or fresh Gate is authorized by this diagnostic.
    Before any further source interaction, run the no-environment query/action
    alignment audit frozen at config SHA256
    `a85de2e89ae0e5477e931cf887b79b6b756aa0c090bf0903353c7bf475262c3d`.
    On source query demos 40--45 only, compare frozen base, step-1000 supervised
    rank-32 LoRA, and step-1000 partial action-expert states using the same 48
    fixed anchors and inference noise. Report normalized generated-action-chunk
    MSE by row, episode, action dimension, and four contiguous time partitions,
    alongside the immutable flow-query reductions. It opens zero new simulator
    episodes and cannot alter a threshold or authorize Gate 0/Writer. If flow
    improves but generated-action error does not, repair the selection/
    acquisition surrogate before more RL; if generated actions improve but
    closed loop does not, predeclare one task-local RL recovery with genuine
    temporal credit; if signs are mixed, treat aggregate averaging as hiding
    behavior-critical heterogeneity. Retire the one-time audit controller and
    launcher after its immutable packet and the selected next recovery contract
    are frozen; retain the general fixed-anchor action-error metric for direct
    Writer query supervision/diagnostics.
    The audit completed main rc 0 from clean `9568921` in 43.77 seconds with
    result SHA256 `95f8adfc...c1c6`. Supervised-LoRA flow-query reductions are
    +5.798%/+4.236% on tasks 3/4, but generated-action MSE changes are
    -3.114%/-2.479% (negative means worse). The non-matched action-expert flow
    reductions are +7.883%/+3.710%, while action MSE changes are
    -0.886%/-0.304%. Thus all four query-positive updates worsen the actual
    fixed-noise generated-action metric; Gate 0/Writer remain false. Because
    both flow and action metrics currently use one deterministic random draw,
    the final no-rollout discriminator before changing acquisition is a small
    predeclared multi-inference-noise replication on the same 48 anchors and
    immutable states. If the sign is stable, test a no-update differentiable
    full-sampler action-loss mechanics smoke, then train only a staged
    action-aligned acquisition recovery. If signs are unstable, treat sampling
    variance/multi-sample estimation as the repair target. Do not reopen SFT
    steps, targets, ranks, simulator outcomes, or RL budget meanwhile.
    The replication is frozen before outcomes at config SHA256
    `d436e17f2a5b91b8cdf22806e3967fc1f0f170590ba8a96692c610c7ef42212f`.
    It reuses the same controller and adds exactly inference-noise seeds
    `[2026071835, 2026071935, 2026072035, 2026072135]`; no data, state, metric,
    or environment surface changes. Confirm robust mismatch only when all four
    candidate-task mean action-MSE reductions are negative and each is negative
    on at least three of four draws; otherwise classify inference sampling
    variance. Neither branch changes Gate authority.
    The four-draw run completed rc 0 from clean `ccb2934` in 52.31 seconds,
    result SHA256 `c1fc3ab4...ae4b`. Supervised LoRA action MSE worsens on 4/4
    draws for both tasks and by 1.901%/3.062% in the mean, establishing a robust
    LoRA flow-to-generated-action acquisition mismatch. The non-matched action
    expert instead improves on 3/4 draws and by 1.680%/1.428% in the mean, yet
    its closed-loop net gain is already zero. Therefore two layers coexist:
    repair LoRA acquisition toward generated actions, while retaining temporal
    closed-loop credit as a separate necessary test. Next run only a no-update,
    one-GPU mechanics smoke that differentiates through the pinned full 10-step
    sampler into the same 37-target rank-32 LoRA and records finite gradient,
    exact source authority, wall time, and peak memory. If it passes below the
    10GiB-headroom limit, freeze a short resumable action-aligned acquisition
    ladder; if it fails, use a multi-sample flow estimator rather than a hidden
    sampler approximation. No new closed-loop or RL outcome precedes that
    mechanics decision.
    The smoke is frozen at config SHA256
    `2dab3cd4399cd93daa26725b3c7ea50d07e555ee70f027ac53c622ac3bc10f25`
    and exact temporary source SHA256 `1e5a8542...c5468`. It uses only
    `task3/demo0/frame0` and `frame1`, batch 2, noise seed 2026072235, zero
    optimizer steps, and zero environment episodes. The source is copied into
    the immutable output packet and then removed from workspace; no parallel
    retained trainer path is created.
    The scientific command passed on clean `fe2d270`: full 10-step action shape
    `[2,50,7]`, finite loss 0.21559, all 74 LoRA gradient tensors present with
    norm 0.52946, identical trainable-state digests before/after backward, and
    only 2,796MiB peak reserved. Result SHA256 is `b4e6fcef...4fe4`; checksums
    and copied source/config pass. The outer ad-hoc telemetry wrapper alone
    reported rc 1 after the timed command printed exit 0 and checksum cleanup
    completed. Retain that immutable wrapper failure packet, do not rerun GPU
    work, and use the canonical fit launcher for training. The smoke authorizes
    predeclaration of a short resumable action-aligned LoRA ladder, not Gate 0,
    Writer, or closed-loop access by itself.
    The result-blind recovery is now frozen in
    `configs/gate_zero_action_aligned_lora_acquisition.toml` at SHA256
    `3d5b54be47c20bf29e356395f43ad2c9d43834b90eded994e68b141be0902246`.
    It keeps tasks 3/4, support/query authority, all 37 targets, rank 32/alpha
    16/dropout 0, effective batch 64, augmentation, evaluator, and four fixed
    action-inference noise seeds. The bounded acquisition repair changes the
    objective to differentiable 10-step generated-action MSE and compresses
    only the warmup/decay horizon to the fixed 200-step ladder while preserving
    the prior 2.5e-5/6.25e-7 peak/decay LR magnitudes; flow-query loss is
    diagnostic. The same
    canonical oracle fitter owns atomic recovery and candidates at steps
    `0/1/5/10/25/50/100/200`, with peak LR 2.5e-5, five-step warmup, and a
    200-step cosine maximum. Step 1 is the batch-64 memory/recovery boundary;
    do not continue scientifically until both tasks reach the same boundary.
    Both task action-MSE reductions must be nonnegative at step 5 to open step
    10 and at least 1% at step 10 to open step 25. Only a candidate with at
    least 2% mean generated-action-MSE reduction on each task and drift at most
    0.02 may open one unchanged source-development closed-loop check; it cannot
    itself pass Gate 0 or authorize Writer. Each blind segment remains below
    30 minutes initially and two hours thereafter. The one-time query/action
    audit controller, launcher, configs, and experiment-only tests are retired
    from the active tree now that immutable packets and this canonical recovery
    are frozen; Git history retains them, while the reusable action-error metric
    remains in the canonical evaluator.
    The exact-resume ladder stopped at its frozen step-10 rule. Step 1 completed
    rc 0 for task 3/4 with 64/64 unique rows, 50,697/50,657MiB peak device
    memory, and valid atomic candidates/recovery. Step 5 improved mean four-noise
    query action MSE by 0.210%/0.219%, passing only the non-regression rule.
    Step 10 improved every fixed-noise draw and the means by 0.881%/0.928%, with
    drift `1.19e-4`/`4.38e-5`; flow-query changes were +0.065%/-0.033% and remain
    diagnostic. Candidate SHA256 values are `c9b0d940...e5571` and
    `292781f4...acbb`; recovery manifest SHA256 values are
    `6fdf67de...b4a49` and `4412cba7...f265c`. Both tasks miss the predeclared
    1% step-10-to-25 continuation floor, so stop without rounding, threshold
    change, step 25, or closed-loop access. This is bounded negative evidence
    for the fixed 10-step action-aligned acquisition ladder, not a final LoRA or
    EMBER negative. Gate 0 and Writer remain false. The next source-only recovery
    must address temporal credit in ordinary task-local LoRA RL under matched
    zero/supervised initialization; do not add blind supervised steps.
    The result-blind temporal-credit recovery is now frozen at config SHA256
    `0cfd1c74ced6b5cdc0e792d1af48555df6f2346527377cdc753ba46fc35955d2`.
    It keeps tasks 3/4, both immutable initializations, the exact 37-target
    rank-32/alpha-16/dropout-0 LoRA, exploration, evaluator, source identities,
    four-arm comparisons, query/drift safeguards, and original two-task
    positive-improvement rule. It changes only the failed RL estimator: each
    arm receives a task-local 512/256 critic over detached frozen SmolVLA
    visual features plus state/progress, action-chunk GAE (`0.99/0.95`), eight
    matched conditional-flow samples, chunk-level PPO clipping, and separate
    fresh actor/critic AdamW state. This is an FPO++-anchored source mechanism
    probe, not a reproduction or performance claim. Stage 8 is the first
    atomic actor+critic checkpoint and development decision; a passing
    behavioral candidate freezes immediately, otherwise only finite,
    nondegenerate temporal-credit mechanics with safe drift may exact-resume
    the same trajectories to stage 16. Failure at 16 stops; episodes 24 and
    later are outside this recovery. No Writer or shared parameter is updated.
    The required real-model mechanics smoke is complete on clean `8237bed`:
    64/64 unique legal source rows produced `[64,1953]` frozen two-camera
    critic features and `[64,8]` matched flow losses, actor/critic gradients
    were finite and nonzero, trainable LoRA state was unchanged, optimizer and
    environment counts stayed zero, and peak reserved memory was 5,268MiB.
    The first fail-closed attempt exposed that the policy's declared trailing
    empty-camera slot had been included in the critic vector; the narrow fix
    now excludes only declared all-false empty slots and rejects any such slot
    that becomes observation-bearing. This authorizes the frozen stage-8
    source run only; it is not Gate-0 evidence.
    Stage 8 and its exact-resume stage 16 are now complete on clean `86b3e40`.
    Stage 8 was mechanically healthy but behaviorally negative, so the frozen
    rule continued once. Stage 16 terminated with status
    `task_local_rl_early_check_not_supported`: zero-init RL changed task3/4
    from `3/8,3/8` to `2/8,3/8`; supervised-init RL changed its own SFT starts
    from `2/8,4/8` to `2/8,4/8`. No task has a positive paired gain. Result
    SHA256 is `e1345634...a14c`; all packet checksums and eight retained videos
    validate. Do not continue this trajectory to episode 24 or reinterpret the
    task4 one-success supervised-vs-zero difference as a useful-update pass.
    The failure is not caused by a zero physical update: final zero-init LoRA
    operator norms are `0.0701/0.0743`, and the RL increment moves the SFT
    operator by `11.3%/11.7%` in task3/4. Before another RL outcome, freeze a
    new, explicitly separate compatibility recovery only if it corrects a
    primary-source-supported mechanism gap rather than merely extending this
    failed budget. The first such bounded candidate is critic-only warmup before
    actor updates with source-result-blind interaction nodes; keep task3/4,
    LoRA, initializations, evaluator, Gate rule, and held isolation unchanged.
    That compatibility recovery is now outcome-free frozen in
    `configs/gate_zero_task_local_rl_critic_warmup.toml` at SHA256
    `51fc9a009d0fa93476ba47a22d86e95a5d89f32182057843c3129e4147725a8a`.
    It starts a new trajectory from the immutable zero/SFT initializations,
    trains only the task-local critic in round 0, requires exact LoRA identity
    at episode 8, uses `gamma=lambda=0.99`, and removes the added Gaussian
    action perturbation in favor of SmolVLA's native stochastic flow sampling.
    Healthy stage 8 exact-resumes to 16; healthy stage 16 exact-resumes once to
    24; episode 24 opens 32 only if at least one initialization has positive
    aggregate paired gain and neither task is below -1. Episode 32 is terminal
    and 40+ is forbidden. A passing node freezes immediately. Run one real-
    model, no-environment critic-warmup smoke before the first source rollout;
    it must show zero actor optimizer steps, exact trainable-state identity,
    finite positive critic gradients, and at least 10GiB device headroom.
    This is the current cheapest source-only Gate-0 recovery, not a Writer
    experiment and not permission to reinterpret the completed negative run.
    The real-model smoke passed on clean `faf1564` in 28.95 seconds: 64/64
    unique source rows, exact actor/LoRA identity, zero actor optimizer state
    and steps, 40 finite critic updates (minimum gradient norm 0.835), and only
    4,128MiB peak reserved. Result SHA256 is `91db6430...3c07`; all three packet
    checksums pass and GPU4 released. This authorizes only the already frozen
    episode-8 stage on GPUs 4--7; the stage must start fresh and stop at its
    predeclared decision.
    The staged recovery completed rc 0 at episodes 8, 16, and 24 on clean
    `2d103d6`; episode 24 is terminal with result SHA256
    `98688726...b1a8`. Zero-init ends with task3/4 paired gains `[0,0]` and
    supervised-init with `[0,-1]`; mechanics, critic/GAE/PPO gradients, drift,
    atomic recovery, checksums, and source/held boundaries remain valid. The
    frozen trend rule therefore stops without episode 32. Gate 0 and Writer
    remain false. Before considering more interaction or another optimizer,
    run the outcome-free support-replay discriminator frozen in
    `configs/gate_zero_task_local_rl_support_replay.toml` at SHA256
    `f539b7376dd1e265076941d7b45022934802f2931bdb54b866b9b97e1a533909`.
    It loads the four immutable episode-24 states and, with zero optimizer
    steps, replays exactly round-0 init states 8--15/seeds 6200--6207 using the
    canonical collector. Positive paired support replay with negative fixed
    development behavior diagnoses generalization/coverage; no positive arm
    diagnoses reward-credit/optimizer acquisition. It consumes 32 source
    episodes, changes no Gate threshold, and cannot authorize Writer.
    The diagnostic has now completed rc 0 on clean `0804f21`. All four replay
    arms are non-positive on the exact round-0 source support slice: supervised
    task3/task4 paired net wins are `[0,-1]`, and zero-init task3/task4 are
    `[-1,-1]`. Status is `support_replay_no_improvement`; result SHA256 is
    `7e92b745...414e`. The four actor states and optimizer states remain exact,
    optimizer steps are zero, all packet checksums pass, and validation/held
    access remains zero. This rejects the coverage-only explanation: the
    current 24-episode reward-credit/optimizer acquisition does not improve
    even its seen support slice. Do not open episode 32 or blindly add scale.
    A read-only action-authority audit also rules out an action-coordinate
    mismatch: LIBERO's env postprocessor is empty, the pinned SmolVLA action
    pre/post statistics are bit-identical, unnormalize-to-normalize round-trip
    error is at most `7.2e-7`, replay consumes the postprocessed environment
    actions, and the model honors the action padding mask. The next recovery
    must therefore change a primary-source-supported credit/acquisition
    mechanism under a new result-before-outcome contract, while keeping tasks
    3/4, all four arms, the canonical reporting evaluator, LoRA support, and
    source/held boundaries fixed.
    The next result-blind recovery is now frozen in
    `configs/gate_zero_task_local_rl_horizon_credit.toml` at SHA256
    `491d0315...e05b`. The official FPO++ manipulation recipe at commit
    `b80112be...f1c6` uses `n_action_steps=16`; EMBER previously executed 50
    actions per inference, leaving only eight reward-credit transitions in a
    400-step episode. The new path changes training collection/credit resolution
    only: execute 16 actions, retain the native 50-slot SmolVLA model output,
    mark only the 16 executed actions valid for flow loss, and collect 25
    ordered transitions per episode. The canonical 50-step development/fresh
    evaluator, four arms, task3/task4, LoRA, optimizer, seeds, thresholds, and
    held boundary stay unchanged. Stage 8 remains critic-only; one exact resume
    to stage 16 is the only actor check. Stage 16 either selects a passing
    checkpoint or terminates; episode 24+ is forbidden. Before any source
    outcome, require one real-model zero-environment smoke proving 200 replay
    rows, ordered 16/50 masking, actor identity, finite critic updates, bounded
    microbatches, and at least 10GiB device headroom.
    The clean-commit smoke passed in 31.42 seconds on one A100: 200/200 replay
    row identities, exact 16-step scoped execution with restoration to 50,
    16-by-8 finite real-model flow losses, 130 finite critic updates, exact
    actor and empty actor-optimizer identity, and only 4,004MiB peak reserved.
    Result SHA256 is `29528c5f...844a`; all result/source/config/telemetry
    checksums pass and GPU4 released. Two preceding no-outcome mechanics
    failures are preserved: telemetry precreated the supposedly new directory,
    then repeated support-loader provenance keys violated the production replay
    uniqueness check. Each received one narrow smoke-only fix; neither changed
    trainer/config/Gate semantics or produced flow/critic/rollout outcomes.
    This authorizes only the frozen four-GPU stage 8 on a new output root.
    Stage 8 then completed rc 0 on clean `ac9cf2f` in 3m34s. All four arms
    collected 200 replay rows at execution horizon 16, made 90 finite critic
    updates, kept actor state exact with zero actor optimizer updates, and
    passed temporal/mechanical safeguards. As required for critic-only warmup,
    both initialization families have task3/task4 paired gains `[0,0]`; status
    is `horizon_credit_warmup_complete_continue_to_16`. Stage-result SHA256 is
    `a3b93ebf...1b0d`. Fifteen JSON files, four atomic recoveries, four
    candidates, four videos, and telemetry validate; peak memory is 19,266MiB
    and all GPUs release. This authorizes exactly one same-output exact resume
    to terminal stage 16, with no configuration or checkpoint selection.
    Terminal stage 16 then exact-resumed from the same output and completed rc 0
    in 7m58s. Mechanics, temporal credit, critic/actor gradients, and all
    artifact checks remained healthy, but the fixed development paired gains
    were `[-2,+1]` for zero-init and `[0,-1]` for supervised-init. The result
    therefore stopped with `task_local_rl_early_check_not_supported`, no
    selected checkpoint, Gate 0/Writer/validation/held all false, and SHA256
    `771eb3b9...f496`. Stage 24 remains forbidden. Before any further outcome,
    freeze one zero-update replay of the exact round-1 horizon-16 training slice
    from the immutable step-16 actors. This diagnostic may distinguish
    support acquisition from development generalization, but cannot pass Gate
    0, select a checkpoint, alter tasks/seeds/thresholds, or authorize Writer.
    That zero-update diagnostic is now frozen before outcome in
    `configs/gate_zero_task_local_rl_horizon_support_replay.toml`, SHA256
    `7e676c52...a9cb`. It binds the terminal result/recovery hashes, round-1
    init states 16--23, seeds 6208--6215, policy RNG 2026071961, horizon 16,
    all four initial success vectors, four-rank topology, zero updates, and a
    32-episode maximum. No video or final Gate surface is consumed.
    The diagnostic completed rc 0 in 2m01s. Exact round-1 paired net wins were
    `[-1,+1]` for supervised-init and `[+1,0]` for zero-init, so status is
    `horizon_support_replay_improves_but_development_does_not`; result SHA256
    is `4a0c13a0...f7ef`. This is partial seen-support acquisition, not a Gate
    pass: fixed development remains negative and no initialization improves
    both tasks. The next bounded recovery may only test data coverage by adding
    fresh legal source training slices under the same trainer, four arms,
    LoRA/optimizer/evaluator/Gate, with result-blind staged stop rules. It must
    start a new immutable trajectory rather than append to the terminal
    stage-16 packet, and it cannot use validation/held/locked surfaces.
    The result-blind coverage contract is now frozen in
    `configs/gate_zero_task_local_rl_horizon_coverage.toml`, SHA256
    `72e4f13e...f241`. It starts a new trajectory at node 8, repeats the sealed
    critic-warmup/first-actor mechanics at nodes 8/16, and adds only source init
    states 24--31 at node 24 and 32--39 at conditional node 32. Node 24 must
    show positive aggregate paired gain in at least one initialization with no
    task below -1 to open 32; otherwise it stops. The original two-task/15pp
    candidate rule, development states 40--47, all four arms, LoRA, optimizer,
    evaluator, query/drift safeguards, and fresh-Gate boundary are unchanged.
    Fresh node 8 completed rc 0 in 3m36s with status
    `horizon_coverage_warmup_complete_continue_to_16`: every arm collected 200
    replay rows, made 90 finite critic updates, and preserved the complete
    actor plus empty actor optimizer exactly. Both initialization families have
    fixed development gains `[0,0]`. Stage-result SHA256 is
    `219eff17...89dd`; 15 JSON files, four candidates/recoveries, four videos,
    and telemetry validate. This opens only exact-resume node 16 on the same
    output, not Gate 0 or a checkpoint selection.
    Node 16 exact-resumed and completed rc 0 in 8m03s. It exactly reproduces
    the sealed horizon result: zero-init `[-2,+1]`, supervised-init `[0,-1]`,
    with healthy mechanics and no selected checkpoint. Stage-result SHA256 is
    `3d4bf5a1...5050`; all new candidate/recovery/video/telemetry evidence
    validates. This reproducibility boundary opens exactly node 24, the first
    new disjoint source-coverage slice; its frozen trend rule decides whether
    node 32 remains closed or opens.
    Node 24 then completed rc 0 at the same clean trajectory and atomically
    stopped. Supervised-init task3/task4 paired gains are `[0,0]`, zero-init
    gains are `[0,-1]`, and supervised-init advantages over zero-init are
    `[-1,+2]`. All four candidates/recoveries, JSON, video, gallery, telemetry,
    and hashes validate; the result SHA256 is `9b738193...0a94`. Under the
    repaired authority this n=8 surface is only a mechanism diagnostic and
    cannot pass/reject Gate 0, select a checkpoint, or open Writer. Do not run
    node 32 under the superseded small-sample rule.
    Before any new Gate outcome, freeze and use
    `configs/gate_zero_evidence_repair.toml` at SHA256
    `0196419d...aa4e`: preserve Gate -1 as passed with
    residuals; keep task3/task4 as development; require n>=32 paired evidence,
    multiple policy RNG and independent training seeds, disjoint source
    confirmation, horizon-16 primary evaluation with horizon-50 robustness,
    process diagnostics, and truthful independent-unit accounting. The old RL
    code is a custom chunk-level flow-loss PPO pilot, not FPO++; a bounded
    faithful per-flow-sample/group-size-one core is required before an ordinary
    task-local RL negative claim.
    The first result-blind confirmation prerequisite is frozen before outcomes
    in `configs/gate_zero_base_difficulty_audit.toml`, SHA256
    `ae73a4b0...6443`. It evaluates only the frozen source base on candidate
    tasks `[6,9,16,20,23,33,39,46,63]`, using 32 unique SHA-partitioned physical
    train states/task, four policy RNG batches, horizon 16, and 288 total source
    episodes. A deterministic base-only competence/headroom rule selects at
    most four distinct primitive signatures; no LoRA, validation, held, or
    locked outcome participates. Keep one video/task plus Trackio/gallery, and
    fail closed on checkpoint/state identity. This audit cannot pass Gate 0.
    Its first launch failed before reset/outcome because LeRobot's lazy async
    wrapper omitted `set_attr`; preserve the rc-1 packet. Materialize the owned
    Gymnasium vector env, set/read back the counters, and require one actual-env
    no-policy identity smoke before a fresh-root relaunch. No scientific
    contract field changes.
    Recovery1 completed rc 0 from clean `3cbb975` in 645.34s and selected
    source confirmation tasks `[6,16,33,39]` with distinct signatures
    `[open,stack,close,turn_off]`. Frozen-base successes across the nine
    candidates are `{6:4,9:27,16:17,20:32,23:0,33:13,39:24,46:4,63:0}`;
    eligible IDs are `[6,16,33,39,46]`, then the predeclared distance/task-ID
    ranking selects four. The canonical selection manifest SHA256 is
    `a4d57cf9...f670`; the result SHA256 is `240de313...dd61`. Gate 0 and Writer
    remain false. Next bind evaluation seeds, h16/h50, >=2 training seeds, and
    LoRA-matched arms to these disjoint confirmation tasks before any LoRA
    outcome.
22. [ ] Train and evaluate supervised direct-Writer cold-start
    zero-interaction utility. The rapid-development comparison is frozen base,
    the language+action-hidden-video Writer, and exact-capacity direct
    task-local LoRA on five distinct validation categories. Writer emits the
    complete declared LoRA; attach it to the frozen base and backpropagate
    independent source-query flow/behavioral loss into Writer. Add the full
    modality/negative/retrieval/direct-generator matrix only after the complete
    EMBER mechanism proves useful and the formal experiment is frozen.
23. [ ] After supervised cold start, run a separate source-only Writer-only RL
    stage. Freeze the shared base; use each generated LoRA for environment
    rollout, but do not optimize that LoRA in place or maintain it as a second
    independent variable. Rollout reward updates Writer parameters only. Report
    immediate generated-LoRA utility before/after this stage and its effect on
    later adaptation. This stage tests reward acquisition by the generator and
    is not ordinary task-local LoRA RL.
24. [ ] Run the fixed ordinary task-local LoRA RL causal arms: A) zero-LoRA
    initialization plus ordinary
    RL; B) cold-start Writer LoRA initialization plus identical RL; and C)
    reward-outer-trained Writer LoRA initialization plus identical RL. Update
    only the same task-local LoRA parameters in place while Writer and base are
    frozen, under identical target layers/rank/count,
    algorithm, hyperparameters, seeds, reward, and interaction budget. Include
    the final predeclared strong baselines; report J0, AUC,
    time-to-threshold, J_K, J_K-J0, uncertainty, resources, and failures. Add a
    matched-initial-performance or equivalent control before claiming a better
    learning process rather than only a better starting point. Writer emits no
    RL constraint object.
25. [ ] Run source-only reward/delayed meta-outer learning with the shared base
    frozen. Inner adaptation updates task-local LoRA; the outer objective updates
    Writer parameters through a predeclared differentiable path or estimator.
    This is distinct from the Writer-only RL stage because it explicitly asks
    whether reward through a task-local adaptation process improves future
    Writer initializations. Shared base/shared LoRA training is not part of the
    mainline.
26. [ ] Permanently freeze base, Writer, encoders, all shared state, target/rank,
    optimizer, budgets, thresholds, and baselines before held evaluation. Only
    predeclared task-local LoRA may adapt from held reward. Require Writer-start
    to beat zero/base and the final predeclared strong baselines with
    predeclared seeds, confidence intervals, causal
    controls, isolation audit, and reproducible rerun.
27. [ ] After the mechanism survives, re-pin and execute the
    eight-GPU-ceiling-compatible OpenVLA-OFT scale confirmation without changing
    the causal or held contract.

## Post-smoke matched Gate 0 contract (frozen before new LoRA outcomes)

- `configs/gate_zero_matched_evidence.toml` (SHA256 `625db578...e0038`) is the only active post-selection
  evidence contract. It hash-binds the repaired Gate-0 contract, permanent
  split, base-difficulty result, confirmation-selection manifest, and mature
  37-target LoRA support.
- Development stays on tasks 3/4. Independent confirmation is fixed to source
  tasks 6/16/33/39 and their deterministic 32/16/2 physical-state partitions.
  No task, partition, seed, checkpoint, or threshold may be selected from a
  task-specific LoRA outcome.
- Run required training replicates `2026071830` and `2026072030`. Use
  `2026072130` only if the two required replicates leave the predeclared matched
  effect ambiguous. Evaluation RNG never counts as training replication.
- Each trainable arm/task/seed needs at least 32 paired episodes and at least
  two policy RNG seeds. Evaluate frozen base once per task/horizon with
  `training_seed=null`; reuse it for paired contrasts without duplicating it
  into false independent seed observations. Retain paired rows and
  bootstrap/exact intervals.
- The primary RL/evaluation horizon is 16. Report horizon 50 only as frozen
  deployment robustness, never for checkpoint selection. Binary success stays
  primary; process diagnostics and drift to both base and each RL
  initialization explain failures without weakening success.
- The canonical trainer now has an explicit surrogate selector. Historical
  packets retain the custom chunk-mean pilot; new matched runs must use the
  faithful per-flow-sample/group-size-one FPO++ core. Use the same LoRA space,
  interaction budget, optimizer, estimator, and evaluation episodes for
  zero-init and supervised-init RL.
- Stage development before confirmation. A credible development candidate
  opens the already frozen confirmation; persistent flat/negative evidence
  with healthy mechanics gets one cheapest source-only diagnostic, not seed,
  node, task, or threshold shopping.
- Required-seed-1 node 16 completed rc 0 in 11m03s with 16 interactions per
  arm and a fully healthy faithful per-flow-sample update path. Its n=8 paired
  changes are supervised-init `[-1,-1]` and zero-init `[-2,+2]`; supervised
  query reductions remain +5.70%/+3.90%. These are explicitly ambiguous smoke
  signals, so the next action is one exact-resume segment to node 24 from the
  same four checkpoints. Do not open node 32, confirmation, Gate 0, or Writer
  until the node-24 review supports it.
- Seed-1 node 24 then met the frozen continuation trend with zero-init paired
  wins `[+1,+2]`; after one fail-closed nonterminal-publication repair, the same
  trajectory reached its hard node-32 maximum. Final n=8 changes are
  supervised-init `[0,+1]` and zero-init `[+1,+3]`, with no supervised-init
  advantage over zero-init. Treat this only as a promising ordinary-RL smoke.
- Launch required training seed `2026072030` next through the same trainer and
  staged nodes. The master seed must drive distinct critic/minibatch/flow and
  training-policy RNG streams while evaluation RNG remains matched. Do not
  expand seed-1 interaction; a scientific candidate still needs >=32 paired
  rollouts/task/arm, multiple evaluation RNG seeds, and both required training
  seeds before disjoint confirmation.
