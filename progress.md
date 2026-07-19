# EMBER Progress and Handoff

## Current state

- The first real matched faithful-FPO++ development segment is complete:
  `gate0_matched_dev_seed2026071830_ep16_20260719_134125`, clean `77e15c7`,
  main rc 0, 11m03s, GPUs 4--7, 16 source interactions per each of four arms.
  All per-flow-sample actor updates, temporal-credit/mechanics checks, atomic
  recovery hashes, JSON, telemetry, Trackio, and the four-video gallery
  validate; GPUs are released. The n=8 horizon-16 paired gains are supervised
  `[-1,-1]` and zero-init `[-2,+2]`, while supervised source-query reductions
  remain +5.70%/+3.90%. By the frozen smoke-only rule this is mixed/ambiguous,
  not a Gate decision. Exact-resume the same seed/arms once from 16 to 24;
  review before any node 32 or confirmation. Gate 0 and Writer remain false,
  with validation/held/locked numeric access still zero.

- After the candidate-step diagnostic failed, the next source-only Gate-0
  recovery was frozen before any new outcome. Config
  `configs/gate_zero_task_local_rl_recovery.toml` initially had SHA256
  `75ceeec398f472d53fb1c7b88b4dd135469b0f841bbf8ac3dfc0ac4b13cd5c68`.
  It keeps task 3/4 and the exact 37-target rank-32 LoRA, and compares frozen
  base, supervised LoRA, zero-init LoRA plus ordinary RL, and supervised-init
  LoRA plus identical ordinary RL. Four independent GPU ranks run an episodic
  AWR-style reward-weighted native flow-loss update with matched exploration,
  reward, optimizer, interaction, and compute; Writer/shared state are absent.
  The 16-episode development node is a 10--30 minute continuation decision and
  32 episodes is the hard maximum; only a later separately bound fresh Gate can
  pass Gate 0. Atomic checkpoints, Trackio, compact videos/gallery, telemetry,
  and exact-resume are part of the canonical launcher.
- First launch `gate0_task_local_rl_ep16_20260719_044843` failed closed after
  rank 2 collected its first eight source episodes, with zero optimizer updates,
  zero development rollouts, no checkpoint, and no Gate result; all GPUs were
  released. A bounded single-card repeat localized both guards. Reset-after IDs
  were exactly 8--15; final IDs changed only because successful vector-env
  members auto-reset. Proposed clipping was `[0,3,10,0,0,0,1195]` by dimension,
  making the binary gripper 98.9% of the false saturation signal. Active config
  SHA256 `e138b7d649c192d4618a8e5b9c0f8fe29b60c95a5117815313f271f405d4d406`
  now audits identity at reset, explores only continuous dimensions 0--5 with
  the unchanged std 0.05, and preserves the policy gripper command. Every
  scientific arm, seed, budget, optimizer, surface, and threshold is unchanged;
  the predecessor/failure/localization hashes are bound in the config before
  rerun. One temporary diagnostic attempt lacked a multiprocessing main guard,
  was terminated before a result, and released GPU 6; the guarded repeat rc 0.
- Canonical recovery2
  `gate0_task_local_rl_ep16_recovery2_20260719_050401` passed the repaired
  collection guards but failed before the first optimizer update or development
  rollout. The replay action tensor `[64,50,7]` is correctly padded by SmolVLA
  preprocessing to `[64,50,32]`; the deterministic flow-noise helper was
  mistakenly called with the original width 7. Ranks 1--3 recorded identical
  fail-closed tracebacks and all GPUs released. The final bounded implementation
  repair validates the processed `[64,50,32]` tensor and derives noise shape
  `[50,32]`; active config SHA256 is
  `504d20bc371078b5ffeabaad84eb1e041423c5167cd7331b91e047a3324f673d`.
  No scientific field changes. Exactly one final canonical verification is
  authorized; another mechanical failure stops for diagnosis instead of a
  further automatic rerun.
- Recovery3 `gate0_task_local_rl_ep16_recovery3_20260719_051103` stopped at the
  new guard, again before optimizer/development evidence, and released all
  GPUs. Source inspection localized the exact API boundary: preprocessing
  retains `[64,50,7]`, while `SmolVLAPolicy.prepare_action` pads internally to
  `max_action_dim=32`. The active code now audits 7D replay input and creates
  `[64,50,32]` noise from the pinned model config. Config SHA256 is
  `b08a85b8de1bf04c788d217cfab8d34bb984d0f70ab8795e8c0aaf0f19820a37`.
  The next action is not another environment rerun: first execute one single-GPU
  real-model synthetic forward/backward with no optimizer step and no simulator
  interaction. Only a pass reopens the canonical 16-episode verification.
- Real-model shape smoke `gate0_flow_shape_real_model_smoke_20260719_052352`
  passed rc 0 from clean commit `cd95342` in 29.06 seconds on one A100. It
  validated `[64,50,7]` replay input, `[64,50,32]` deterministic noise,
  per-sample loss `[64]`, and a finite backward pass, with no optimizer step or
  simulator/validation/held access. Result SHA256 is
  `64e522b8863527234e7633a1c8ea72482459b57d43d4b129e4a67c60668a689c`.
  GPU memory was released. The already frozen canonical stage-16 source-only
  verification is mechanically reopened; no Gate or Writer claim follows from
  this smoke.
- Canonical stage-16 recovery `gate0_task_local_rl_ep16_recovery4_20260719_052739`
  completed rc 0 from clean `a581eea` in 4:38.96 and released GPUs 4--7. All
  checksums, mechanics, atomic states, Trackio runs, gallery, and four videos
  are valid. Peak device memory was 22.1/19.0/18.8/18.8 GiB and retained output
  is 93 MiB. Each arm completed 16 finite optimizer steps with varied source
  reward, but none changed its development success count from its own start:
  task3 supervised 2/8 -> 2/8, zero 3/8 -> 3/8; task4 supervised 4/8 -> 4/8,
  zero 3/8 -> 3/8. Frozen decision
  `task_local_rl_early_check_not_supported` prohibits stage 32/fresh Gate;
  result SHA256 is `aab151ea...b57`. Gate 0 and Writer remain unauthorized.
- Only step 16 has a retained loadable model/optimizer/RNG state; step 8 has
  round metrics only. Next is a no-new-interaction source-only diagnosis using
  the step-16 state and the two rounds' reward/loss/gradient/query/drift evidence
  to separate weak or directionally unhelpful updates. Any later estimator or
  optimizer recovery must be sealed before outcomes; no automatic interaction
  extension is permitted.
- Architecture ownership is deliberately narrow: the
  `ember.gate_zero_task_local_rl` package owns only this bounded Gate-0 recovery
  contract, reward-weighted replay mechanics, and orchestration. It reuses the
  existing canonical LoRA session, evaluator, simulator rollout, atomic
  checkpoint, gallery, and launcher conventions rather than creating another
  trainer. There was no earlier ordinary task-local RL implementation to
  retire; the failed supervised candidate diagnostic remains immutable
  evidence, not an alternate executable RL path. The package is removed from
  the active plan after this bounded recovery is answered and its reusable
  invariant tests/checkpoint primitives have been folded into the direct
  Writer/ordinary-RL owner, rather than being extended into a generic RL
  framework.

- General thesis, embodied concept, novelty landscape, decisions, and prior
  Writer-program lessons are documented.
- An independent expert completed `docs/expert_plan.md` with a conditional-go
  recommendation and a full staged design; it is now historical provenance.
- The owner rechecked the original design on 2026-07-18. The active contract is
  direct language/action-hidden-video to complete task-specific LoRA, immediate
  utility, an independent source-only Writer-only RL stage, ordinary task-local
  LoRA RL, source-only Writer reward/meta learning,
  shared-frozen held evaluation, and conditional OpenVLA-OFT confirmation.
  Canonical bank/shared subspace/soft geometry/residual escape was a later
  assistant/expert addition and is outside the current project/Goal. The
  corrected authority is recorded in `AGENTS.md` and
  `docs/execution_brief.md`.
- The active compute ceiling is four A100 80GB GPUs.
- A clean execution clone and user-home workspace entry have been verified on
  the target GPU host. Work is isolated on
  `phase0/reproducible-substrate`.
- `configs/phase0.toml`, `pyproject.toml`, and `uv.lock` now pin the scientific
  contract and the Python/simulator environment. Contract and runtime-repair
  tests pass in the locked environment.
- The environment bootstrap is reproducible without system privileges. It
  includes content-guarded repairs for the malformed BDDL metadata payload and
  robosuite's shared `/tmp` log collision; `uv pip check` and locked-environment
  checks report no drift afterward.
- Immutable model, dataset, asset, task-map, and official semantic-tree hashes
  are recorded. The pinned SmolVLM constructor snapshot has been downloaded and
  its 2,029,990,624-byte weight file matches the declared SHA256.
- The official offline SmolVLA/LIBERO mechanics path now completes on
  `libero_spatial` task 0 at seed 1000. The one-episode run succeeded, generated
  a valid video/gallery, used 2,029 MiB peak GPU memory, and released its sole
  GPU afterward. This overlap-trained checkpoint remains mechanics-only; no
  EMBER training or scientific Gate evaluation has started.
- The initial attempt's invalid `env.seed` field was diagnosed at the config
  layer and removed; global `seed=1000` is the validated entrypoint. The failed
  and recovered local long-run records are both retained.
- `scripts/run_phase0_eval.sh` is the canonical single-process offline
  evaluation entrypoint. It validates GPU/task/batch arguments, refuses rollout
  batches that LeRobot would partly discard, rechecks pinned runtime assets, and
  generates a safe post-run HTML/video gallery with an atomic `latest` link.
- A synchronous batch-8 calibration on task 0 completed with 7/8 successes in
  119.131 evaluation seconds. Peak GPU memory was 6,003 MiB and active-window
  average GPU utilization was only 5.60%; the 280-step failed seed exposes a
  simulator/straggler bottleneck rather than a memory ceiling.
- The matched asynchronous batch-8 run preserved all eight outcomes and reduced
  evaluation time to 63.403 seconds (1.879x faster). Subsequent asynchronous
  batches 32, 96, and 112 reached 0.2667, 0.3850, and 0.3958 episodes/s. Batch
  112 peaked at 68,080 MiB with 13,076 MiB free and is the measured
  resource-rich rung; batch 96 is the conservative shared-host rung.
- Calibration also exposed a Gate -1 reproducibility ambiguity: fixed
  seed/init-state prefixes were not outcome-identical across all batch sizes.
  The first 32 outcomes differed twice between batches 32 and 96, and the first
  96 differed ten times between batches 96 and 112. These runs remain
  mechanics/throughput evidence only; a reset-observation/initial-action/short-
  rollout probe is now required before scientific metric comparison.
- The official ten-task `libero_spatial` mechanics sweep completed from clean
  commit `342b256` in one sequential process. Every task/BDDL/init-state/camera/
  controller path and all ten videos completed; 9/10 single episodes succeeded,
  while task 5 ran 280 steps and failed. The result is mechanics-only evidence
  from an overlap-trained checkpoint, so no task-5 tuning or policy-quality
  claim is authorized. Evaluation time was 97.362 seconds, wall time was 116.77
  seconds, and peak GPU memory was 2,029 MiB.
- Direct TorchCodec import still fails for lack of compatible shared FFmpeg
  libraries, so it is not the selected backend. The contract and direct project
  dependency now pin PyAV 15.1.0; generated-video round-trip, timestamped
  LeRobot selection, and live artifact decode/throughput checks pass.
- `configs/gate_minus1_identity.toml` now predeclares the first overlap-only
  evaluation-identity diagnostic: exact reset/fixed-action comparisons across
  repeated sync/async batches 1 and 2, a `1e-6` initial-action tolerance, a
  batch ladder capped at the measured-safe 112, five-step policy trajectories,
  and stop-before-policy rules for mechanical mismatches.
- The frozen mechanics layer completed from clean commit `07baaca` and stopped
  as predeclared before policy load. Seed/init-state/BDDL/controller/state and
  outcome identity held, but sparse camera-render differences prevented strict
  bitwise observation identity: all deltas were pixel-only, at most 1/255, and
  at most 28/388,800 values in any leaf. Same-mode repeats also exhibited the
  variation, localizing it to renderer-level nondeterminism rather than
  sync/async or batch semantics alone. A checksummed failure packet records the
  unresolved action and short-trajectory effects; no tolerance was relaxed.
- `configs/gate_minus1_identity_recovery.toml` is the predeclared action-layer
  recovery spec. It is byte-for-byte equivalent in parsed experiment fields to
  the strict spec except for one bounded state transition: a mechanics failure
  may continue to policy diagnostics only when every mismatch domain is
  `pixels`; any state/outcome/other mismatch still stops. The strict mechanics
  status and original stop reason remain failed in the result.
- The recovery first repeats the same frozen reset observation across policy
  batches 1, 2, 8, 32, 96, and 112 with RNG seed 20260717 and action
  `atol=rtol=1e-6`; the ladder stops at its first outside-tolerance action. It
  then records reset digest, init-state identity, initial action values, and
  five-step policy trajectories for the same six matched environment
  conditions. These are localization diagnostics, not a tolerant Gate pass.
- The canonical recovery completed and stopped its identical-observation batch
  ladder at batch 2: all seven action dimensions changed beyond `1e-6`, with a
  maximum absolute delta of 0.002254. Across the seven matched actual-environment
  comparisons, maximum initial-action and five-step action deltas were 0.004378
  and 0.009512. Five-step reward/done records remained exact, but actions and
  subsequent robot state did not; Gate -1 remains failed/ambiguous.
- The bounded mechanism probe then proved that the raw/preprocessed first
  sample, the first `50 x 32` flow-matching noise sample, and a repeated
  same-shape batch-1 forward are bitwise identical. With explicit matched noise,
  changing only model batch shape from 1 to 2 still changed all seven action
  dimensions (maximum 0.004668). The unresolved choice is now the evaluation
  contract, not seed, init-state, preprocessing, or random-noise authority.
- The pinned LIBERO-90 download completed in the resumable zero-GPU long-run
  `libero90_canonical_download_20260717_163546`. It is restricted to the exact
  90 HDF5 files at revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`, uses
  up to eight network workers, returned main rc 0 at exactly 90 files and
  66,658,085,995 bytes, and left no incomplete residue.
- The first 8-worker canonical audit stopped after 12.01 seconds on task 14
  before artifact generation: a legacy producer `env_args` basename differed
  even though canonical HDF5,
  task-map, and official BDDL names matched. A full metadata scan found six
  finite legacy basename differences across four source, one validation, and
  one held metadata record; all 90 canonical bindings and HDF5 instructions
  matched.
- The bounded fix treats only the legacy producer basename as a provenance
  note, retains canonical HDF5 mismatch as fatal, and is covered in both
  directions. A real task-14 audit passes with the two expected legacy notes
  and source-only normalization access. All 11 manifest tests and all 54
  repository tests pass; the failed long-run state remains untouched and the
  recovery must use a fresh artifact directory from a clean commit.
- The fresh recovery completed from clean commit `d6cdac7` with main rc 0 and
  checksum follow-up rc 0. In 12.45 seconds, eight CPU auditors and zero GPUs
  validated 90 tasks, 4,500 demonstrations, 669,043 frames, and the exact
  66,658,085,995-byte input. Peak host RSS was 864,788 KiB. The final
  `pass_with_documented_notes` artifact is 1,307,643 bytes and its atomic local
  `latest` report is ready for later review.
- Source-only normalization contains 166,475 finite state/action rows from the
  declared 60 tasks and episodes 8–27. Validation/held are 30/30 metadata-only,
  held BDDL remains identity-hash-only, checksums pass, and serialized outputs
  contain no local path, host identity, producer model path, or raw data. The
  coarse validation scene/object/fixture/predicate presence sets are covered by
  source, but the finer relation/order and at-least-two atomic coverage proof is
  still open and Gate -1 is not passed.
- A high-confidence language-only role audit now proves the current 60/15/15
  split violates the at-least-two source-atom rule. Five atom roles have zero
  source tasks (`stack`, moved `tomato_sauce`, `under`, `front_of`, and
  `wine_rack`), while moved `moka_pot`, `wine_bottle`, and `white_bowl` have one
  source task each. No LIBERO-90 policy result or held executable label was used.
  `docs/benchmark_validity_report.md` records the evidence and recovery fork.
- `scripts/build_libero_manifest.sh` is now the sole manifest entrypoint. The
  implementation performs Hub LFS hash/byte validation, HDF5 schema and task
  mapping checks, BDDL/init-state authority capture, source-only episodes 8–27
  normalization, data-quality aggregation, checksums, and an atomic local HTML
  `latest` report. Eight focused manifest tests and all 41 repository tests
  pass; the wrapper dry-run, Python compilation, and shell syntax checks pass.
- A live source/validation/held probe passed on tasks 0, 3, and 1. Only source
  task 0 produced numeric normalization arrays; validation and held returned
  none. Held BDDL semantics are not parsed. The probe also surfaced task-map /
  parsed-BDDL wording notes on non-held tasks 14, 84, and 85; task mapping and
  splits remain unchanged.
- The coarse factor summary no longer represents unread held semantics as an
  empty uncovered set. Task-name scene coverage remains evaluated, while held
  object/fixture/goal-predicate coverage is explicitly `null` and
  `not_evaluated_due_to_access_policy`. A regression test fixes this leakage-safe
  report contract; the finer atom-multiplicity and relation/order split audit
  remains a distinct Gate -1 deliverable. All nine manifest tests and all 52
  repository tests pass.
- The next Gate -1 diagnostic is predeclared in
  `configs/gate_minus1_specification_pilot.toml`. It uses the overlap-trained
  official checkpoint only on `libero_spatial` tasks 0 and 1, with correct,
  empty, scene-only, and within-pair swapped prompts. Every arm is one async
  batch of eight with identical seeds, init-state indices, batch shape, mode,
  and policy RNG seed. The pilot stops if either correct arm has zero successes;
  it cannot authorize a Gate pass or any Writer/video/policy-quality claim.
- The specification harness changes only each environment's policy-visible
  `task_description` and proves the underlying task identity is unchanged. A
  transparent reset-audit proxy records the exact first upstream reset seed and
  init-state stride without replacing the upstream rollout. It writes paired
  bootstrap diagnostics, one video per arm, a local HTML gallery, and checksums.
  The first launch stopped before GPU allocation on an exact checkpoint-role
  string mismatch between the new TOML and `phase0.toml`; its failure packet is
  retained. The string is corrected without changing the experiment, a
  cross-contract regression test now covers it, and nine focused tests plus all
  50 repository tests pass. The recovered launch then exposed a second
  implementation-layer issue before any rollout: LeRobot's lazy async wrapper
  omits `set_attr` even though its underlying Gymnasium vector env implements
  it. That 6.9KB failure packet is retained.
- The prompt override now explicitly reaches the pinned lazy wrapper's
  underlying `AsyncVectorEnv.set_attr` after `_ensure`; a faithful lazy-wrapper
  regression test passes without changing upstream rollout semantics. The
  launch wrapper also hard-rejects any preexisting compute PID or at least
  1,000 MiB GPU allocation. It correctly rejected the active MemLLM PID on GPU
  0, and all 51 repository tests pass.
- The recovery2 scientific command then completed all 64 predeclared episodes
  on an empty GPU 4. Correct was 6/8 on both tasks; no-spec was 0/8 on both;
  scene-only was 2/8 then 0/8; swapped was 0/8 on both. Its 1,166,960-byte
  canonical artifact contains valid result/eval JSON, telemetry, a local HTML
  gallery, and eight checksum-verified videos. It is only an overlap-trained
  prompt-path/specification scale candidate, not a LIBERO-90 Gate -1 pass or
  evidence about Writer, video, policy quality, or same-init goal switching.
- The evaluator printed completion and GNU `time` recorded exit 0. A subsequent
  Bash 5.2.21 function-context error in the one-off outer telemetry `EXIT` trap
  made the long-run wrapper record rc=1. That state remains untouched as a
  wrapper failure packet. The canonical launcher was not the cause and is not
  forked or patched. Future long-run templates explicitly capture the main rc,
  perform normal-path telemetry cleanup with `INT`/`TERM` handlers, and return
  the captured rc; shell smokes preserve 0/7/124 for success/failure/timeout and
  leave no sampler process.
- Owner authorization resumed the same first-stage Goal and selected the
  recommended one-time specification-only split recovery. The original split,
  failure report, and commit provenance are preserved; no LIBERO-90 policy
  outcome or held privileged surface was read.
- New strict owners separate language-role parsing (`libero_task_factors.py`),
  pure deterministic search/audit (`libero_split.py`), and the sole pinned
  task-map reseal/verification CLI (`libero_split_reseal.py`). The checked-in
  90-task record is `configs/libero90_split_reseal.json`; the active contract
  pins its SHA256, specification hash, parser schema, coverage threshold,
  algorithm, seed, candidate count, old split commit, and active IDs.
- The deterministic full search completed in 20.39 seconds on one CPU core with
  18,240 KiB peak RSS and no GPU. Its result has zero role-coverage violations,
  minimum source exposure two across 41 evaluation roles, 30/30 unseen full
  compositions, 30/30 same-scene source controls, and 28/30 role-sharing
  same-scene hard negatives. A second regeneration from the modified active
  contract matched the 144KB record byte-for-byte at SHA256 `9f5bc62e...8428`.
- The canonical manifest now binds the sealed language factors by task
  index/scene/instruction before data audit and will expose role atoms/order plus
  split metrics in the dependency-free local HTML report. The previous
  normalization is explicitly tied to the rejected split; a fresh clean-commit
  manifest run is the next required canonical artifact.
- Milestone verification passes all 68 repository tests, full contract/record
  validation, Python compilation, shell syntax, rendered-report JavaScript
  syntax, diff whitespace, and secret/path scans. Cleanup removed the verified
  duplicate 144,387-byte temporary candidate plus five regenerable pytest-cache
  files (150,794 bytes total); all long-run and canonical evidence was retained.
- Commit `23f3301` was pushed, leaving a clean tree for the formal resealed
  manifest. Longrun `libero90_resealed_manifest_20260718_044358` completed with
  main rc 0 in 12.34 seconds: eight CPU workers averaged 806% CPU, peaked at
  840,572 KiB RSS, and used no GPU. The fresh 1,537,733-byte artifact is
  `reseal1_20260718T044331Z`, and `latest` now points to it atomically.
- All four artifact checksums and JSON parses pass. The manifest binds all 90
  factors/splits to the permanent seal and clean generation commit; source
  episodes 8–27 provide 183,555 finite normalization rows, validation/held are
  exactly 30 metadata-only tasks, evaluation numeric access is zero, and held
  BDDL remains identity-hash-only. Private-path/producer-path scanning is clean.
  The new HTML view exposes split metrics, operation order, and role atoms; its
  embedded JavaScript passes a fresh syntax check.

## Gate -1 source same-state goal mechanics milestone

- Commit `25b1276` adds the sole source same-state goal-mechanics owner, frozen
  config, thin CPU-only launcher, and six focused tests. The full repository
  suite passes 74 tests, contract validation, compilation, shell syntax, diff
  whitespace, and secret/path scans. The architecture guard has no hard or
  function-size signals; its only review flag is the 774-line source/test/
  launcher growth. The 534-line core has one current responsibility and reuses
  native LIBERO BDDL evaluation plus the existing atomic `latest` owner, so no
  parallel evaluator or compatibility path was introduced.
- The first longrun invocation (`..._050456`) expanded an empty output variable;
  the launcher rejected the non-absolute output with rc 2 before data or
  simulator access. The corrected invocation preserved the same scientific
  contract and completed as `gate_minus1_source_same_init_goal_20260718_050511`
  with main rc 0 from the clean pushed commit.
- Canonical task-3/task-4 mechanics are exact: shared model-layout hashes match;
  all eight shared initial states have zero cross-environment state delta and
  are unsuccessful under both native goals; all sixteen source terminal states
  are successful only under their originating goal. The minimum bidirectional
  fraction is 1.0 versus the frozen 0.80 threshold. The run used no GPU, took
  4.73 seconds, peaked at 2,748,868 KiB RSS, and retained a 21,482-byte
  checksummed JSON/HTML report with an atomic local review link.

## Gate -1 same-observation language-action milestone

- Commit `2038129` adds a frozen action-path config, one focused owner, a thin
  offline single-GPU launcher, and four pure contract/comparison tests. The
  repository passes 78 tests, contract validation, compilation, shell syntax,
  diff checks, and secret/path scans. Architecture review finds no hard,
  complexity, or function-size signal; the only flag is the 805-line current
  source/test/launcher surface. It reuses the existing fixed-batch policy
  forward and does not add another rollout evaluator or goal implementation.
- Longrun `gate_minus1_language_action_20260718_052249` completed with main rc 0
  from the clean pushed commit. For each overlap task, one batch-8 reset supplies
  exactly the same cached observation to all prompt arms and the correct repeat.
  Correct repeats are exact; correct versus swapped/no-spec/scene-only is
  substantive for 16/16 samples in every comparison, including the first
  action. The primary swapped maximum action delta is 0.452167.
- The run took 41.72 seconds, peaked at 3,980,224 KiB host RSS, and reports
  1,232 MiB Torch peak reserved memory on GPU 4. The 268,038-byte checksummed
  JSON/HTML artifact has an atomic `language_action_latest` report; the GPU is
  released. No video was duplicated because the frozen prior pilot supplies
  matched rollout videos.

## Gate -1 action-hidden video prelaunch contract

- Before reading any video-probe outcome, the checked-in contract freezes
  resealed source tasks 3/4, support demonstrations 0--23, disjoint query
  demonstrations 24--47, and unused rows 48--49. Each encoder input contains
  only 16 uniformly sampled `agentview_rgb` frames with the final recorded
  frame excluded and standardized 1 fps metadata. Task language, task ID,
  action, proprioception, reward, terminal flags, original trajectory length,
  filename, and normalization statistics are absent from the encoder input;
  source task labels are used only to fit and score the declared readout.
- One frozen pinned SmolVLM2-500M causal-context feature and a single balanced
  dual-ridge readout with fixed lambda 1.0 are primary. Only ordered support
  clips fit the readout. The same query clips then generate ordered, reversed,
  deterministic shuffled, repeated first-frame, repeated last-frame, repeated
  temporal-median, and independently resampled drop-last-20% controls. The
  wrong-video control swaps the two same-scene source labels. Thresholds and
  bootstrap seeds are fixed in
  `configs/gate_minus1_video_information_probe.toml`; the bidirectional query
  pairs are demo-index matched but are not claimed to share simulator state.
  This two-task diagnostic
  cannot pass Gate -1 or authorize Writer.
- A non-scientific duplicated-clip systems sweep selected batch 48 on one A100:
  device-side preprocessing plus inference measured 17.37 clips/s with 49,624
  MiB peak reserved memory. Batch 64 was slower at 14.89 clips/s despite 65,924
  MiB reserved; batch 32 reached 16.64 clips/s. The canonical path therefore
  uses the throughput optimum rather than dummy allocation and retains at least
  10 GiB measured total-device headroom.
- The video surface has one CLI and launcher. `video_probe_core.py` owns the
  fail-closed protocol/statistics, `video_probe_runtime.py` owns pinned
  authority/RGB extraction/frozen encoding/GPU telemetry, and
  `video_information_probe.py` owns orchestration, checksums, compact MP4/HTML,
  and the atomic latest link. The architecture guard has no hard violation;
  its review flag is the approximately 1.5k-line config/source/test/launcher
  addition. No prior video-probe path is superseded. If the video hypothesis is
  rejected, retire this entrypoint after its immutable failure packet and
  report are preserved; if it survives, these caches become the sole upstream
  video input owner for Writer acquisition.
- Before formal launch, all 84 repository tests pass along with Python
  compilation, shell syntax, launcher dry-run, diff whitespace, and the
  architecture review. Canonical output is still pending and no scientific
  metric has been read.

## Gate -1 action-hidden video failure packet and recovery

- Clean commit `b256227` produced canonical longrun
  `gate_minus1_video_information_20260718_060251` with main rc 0 and external
  artifact `video_information_20260718T060241Z`. All 21 listed files pass
  checksums; both NPZ caches load without pickle, all 384x960 features are
  finite, all fourteen 16-frame 128x128 MP4 controls decode, the latest link is
  exact, and GPU 4 was released. Validation and held numeric access remain zero.
- The predeclared result is negative/ambiguous rather than a mechanics failure.
  Ordered query balanced accuracy is 0.625 with a 95% source-stratified
  bootstrap interval of [0.50, 0.75], bidirectional query-pair correctness is
  9/24, and wrong-video specificity is 0.625. First-frame and static-median are
  0.50, but last-frame is 0.729, reversed is 0.729, shuffled is 0.708, and
  drop-last-20% is 0.542. Thus content, temporal order, endpoint independence,
  and the 0.80 primary threshold all remain unestablished; Gate -1 and Writer
  authorization stay false.
- The failure is classified primarily as representation/acquisition. The same
  fixed readout reaches only 0.8125 on ordered support and 0.625 on query; a
  no-hyperparameter nearest-centroid diagnostic is also 0.625. Frozen
  causal-context class centroids are almost collapsed (support cosine 0.999601,
  query cosine 0.999775). Exact same-batch repeats, source hashes, balanced
  partitions, finite caches, and video decoding rule out the mechanical layers;
  closed-form fitting rules out iterative optimization instability. Subtle
  same-color objects and robot occlusion remain a data/information alternative,
  so this result does not yet prove that RGB itself is insufficient.
- The lowest-cost discriminating recovery is frozen before rerun: reuse the
  exact checksummed RGB cache and all task/demo/control/threshold/readout
  choices, retain the same SmolVLM2 weights and matched batch 48, and replace
  only the final LM context token with per-frame visual-connector embeddings
  summarized by fixed mean/first/last/signed-delta/linear-time-slope moments.
  No language, task ID, privileged field, held data, threshold search, model
  fine-tuning, or alternative task pair is introduced. If ordered performance
  remains below 0.80 or temporal/static criteria still fail, retain that result
  and return to information/data diagnosis rather than changing the standard.
- The run used one A100 for 38.80 seconds, reached 100% sampled utilization,
  50,143 MiB total memory used (49,624 MiB Torch reserved), kept 31,777 MiB
  headroom, peaked at 3,857,596 KiB host RSS, and retained 58,908,493 bytes.
  The gallery is available at `video_information_latest/index.html`; the RGB
  and feature caches are intentionally retained for the bounded recovery.
- Recovery implementation is now sealed at config SHA256
  `bf19485beffbfa88657ea3acf793798277f0b3590f3d6c650ec3711d493e9d03`.
  It validates the prior result/cache/gallery hashes, reuses the exact 96-clip
  RGB cache, and symlinks the unchanged fourteen videos instead of duplicating
  media. A two-clip technical smoke produced finite 4800-dimensional features
  with exact repeat delta zero; it did not inspect a classification metric.
  The full suite passes 87 tests, compilation, shell syntax, recovery dry-run,
  and diff checks. Architecture review has no hard violation; its function-size
  flags cover the single recovery overlay, immutable cache transaction, and
  shared run orchestrator. This is one config-selected strategy under the same
  CLI, not a parallel probe. It retires with the base video probe after Gate -1
  evidence is preserved if the hypothesis is rejected.
- The first recovery launch is retained as wrapper/publication failure packet
  `gate_minus1_video_information_recovery1_20260718_062439`. Its model and
  scoring work completed, but the CLI resolved the existing `latest` symlink to
  its old target before the atomic publisher could replace it, so the outer
  command correctly returned rc 1. No scientific result was promoted from that
  run. Commit `eff4269` changes only that path handling from symlink-following
  resolution to a lexical absolute path and adds a red-then-green CLI
  regression test; shared atomic-link tests also pass.
- The clean repaired commit produced canonical longrun
  `gate_minus1_video_information_recovery1_20260718_062728` with main rc 0 and
  artifact `video_information_recovery1_20260718T062713Z`. All six retained
  artifact checksums pass; the 384x4800 float32 feature cache is finite and
  pickle-free, all fourteen reused videos match their prior hashes and decode,
  the relative media symlink and atomic `video_information_latest` link are
  exact, validation/held numeric access is zero, and GPU 4 is released.
- Recovery ordered accuracy is 0.7917 with bootstrap interval
  [0.6667, 0.8958], wrong-video specificity is 0.7917, and paired correctness is
  15/24. All remain below their fixed content thresholds. Ordered-minus-first,
  ordered-minus-static, and temporal-order gaps are 0.2708, 0.25, and 0.2292,
  so the temporal representation remedy works, while drop-last retention is
  only 0.8158. Gate -1 and Writer authorization remain false; no further
  post-outcome reader or pair selection will be performed.
- The successful run took 28.77 seconds internally (29.71 seconds under GNU
  time), processed 25.94 clips/s, reached 100% sampled utilization and 50,143
  MiB peak device usage with 31,777 MiB headroom, and retained only 4,564,411
  bytes by reusing the prior RGB cache and videos. Result SHA256 is
  `f80d2b2b4c8ac1bb1b9dea5755178d1648fca381de52917708ec8fcc440bbde1`.

## Current phase

Phase 0, reproducible substrate, is in progress. The immutable contract, first
official mechanics smoke, explicit PyAV decoder path, and useful single-GPU
concurrency envelope are established. The ten-task spatial mechanics sweep is
also complete. Gate -1 evaluation identity has completed its bounded mechanism
diagnosis. The selected recovery is fixed-contract statistical/functional
reproducibility: the unchanged async evaluator uses one predeclared measured-safe
batch/mode across every arm, task-level repetition and confidence intervals are
primary, and batch-1 exactness is a small audit. The official-overlap
specification pilot is complete and authorizes scaling only that diagnostic.
The previous canonical LIBERO-90 manifest is complete as data-integrity
evidence, and its role-aware audit falsified the original split. The authorized
one-time specification-only recovery is now permanently sealed with no policy
outcome input. The fresh canonical manifest and source-only normalization under
the new IDs now pass. The source same-state native-goal surface passes its
mechanics contract, and the fixed-batch overlap policy has an exact-repeat,
    same-observation language-to-action path. The action-hidden video probe and
    its only same-cache representation recovery are complete: temporal signal
    is established on one source pair, but content thresholds still fail.
    Gate -1 is therefore still in progress on correct paired-goal behavior with
    legal source competence. Writer training remains unauthorized.

## Implementation ownership review

- `src/ember/contracts.py` is the single owner of immutable scientific,
  resource, split, and artifact-surface invariants.
- `src/ember/runtime_env.py` owns only version- and content-guarded repairs to
  the pinned third-party Python installation. It does not own simulator or
  policy behavior.
- `src/ember/phase0_runtime.py` is the single owner of checked external-asset
  materialization: the offline SmolVLA view and LIBERO path/asset binding.
- `src/ember/eval_artifacts.py` owns only post-run video validation, gallery
  generation, media hashes, and the safe `latest` symlink. It never changes
  evaluation metrics or simulator behavior.
- `src/ember/identity_evidence.py` owns canonical tree hashing, numeric
  difference summaries, and validation of the bounded overlap-only probe spec.
  `src/ember/evaluation_identity.py` owns the staged mechanics/policy diagnostic
  and atomic result/runtime-error packets; the canonical stopped-Gate failure
  packet is derived from its immutable result. The shell script is its only
  offline single-GPU entrypoint. These files observe upstream semantics and do
  not replace or patch the evaluator.
- `scripts/bootstrap_env.sh` remains a thin environment entrypoint and
  `scripts/zig-cxx` is only the pinned user-space compiler adapter.
  `scripts/run_phase0_eval.sh` is the one thin evaluation entrypoint over the
  runtime and gallery owners. Tests mirror these ownership surfaces; the local
  launch scripts used during diagnosis are ignored evidence, not parallel
  retained implementations.
- `src/ember/libero_data.py` owns the only HDF5/LFS integrity and source-only
  normalization implementation. `src/ember/libero_manifest.py` owns task/BDDL/
  init-state authority and the single manifest orchestration path;
  `src/ember/libero_report.py` owns only the dependency-free read-only HTML
  projection. The shell wrapper is thin and no alternate downloader, converter,
  or manifest format was added.
- `src/ember/libero_task_factors.py` owns the strict fail-closed language-role
  grammar. `src/ember/libero_split.py` owns pure role-coverage audit and
  deterministic split search. `src/ember/libero_split_reseal.py` is the only
  pinned task-map record generator/verifier and does not read BDDL, dataset, or
  policy surfaces. `configs/libero90_split_reseal.json` is generated evidence,
  while `configs/phase0.toml` remains the active contract. These are distinct
  current responsibilities, not parallel split implementations.
- This manifest surface adds five active files and roughly 1.3k source/test/
  wrapper lines, so the architecture guard requires an explicit rationale. The
  size is driven by the current second-use boundary between leakage enforcement,
  task authority, artifact rendering, and synthetic HDF5 contract tests. Core
  retained modules are 337 and 504 lines. The 70-line audit function is one
  file-open/access-policy transaction with schema and environment work already
  delegated; the 90-line report function is a cohesive declarative HTML
  template, and the long test helper materializes the complete synthetic HDF5
  schema. There is no superseded path to retire.
  Retire this builder only if a pinned upstream exporter reproduces the same
  source/held access separation, immutable hashes, normalization provenance,
  and local checksummed report; otherwise it remains the canonical owner.
- `src/ember/specification_probe.py` owns only the prompt-path diagnostic and
  reuses the existing pinned policy/runtime builders plus LeRobot's unchanged
  evaluator and existing gallery owner. Its shell wrapper is the single launch
  entrypoint. The 823-line source/test/wrapper addition has no fallback or
  parallel evaluator; the module remains below the architecture review size
  boundary, and its retirement trigger is completion of Gate -1 or an upstream
  evaluator gaining an equivalently audited prompt-override hook.
- `src/ember/counterfactual_goal_probe.py` owns only source-task exact-state
  cross-evaluation under two unmodified native BDDL goals. Its frozen config and
  CPU-only shell launcher are the sole active path; it does not load a policy,
  invent a goal heuristic, or expose validation/held numeric data. The module
  is retired after Gate -1 once its hashes and result summary are preserved.
- `src/ember/language_action_probe.py` owns only one cached-reset observation
  and matched prompt-conditioned action plans. It imports the existing pinned
  policy builder/action postprocessor, links the prior competence artifact by
  hash, and never steps a rollout or changes goal semantics. Its config and
  launcher are the sole current path and retire with Gate -1 evidence.
- The action-hidden video diagnostic has three non-overlapping owners described
  in its prelaunch section: pure protocol/statistics, pinned data/model runtime,
  and one canonical artifact/CLI path. The split resolves the architecture
  guard's former >1,000-line single-file hard violation; no alternate encoder,
  reader, launcher, or legacy compatibility path remains active.
- The action-hidden video diagnostic has three non-overlapping owners described
  in its prelaunch section: pure protocol/statistics, pinned data/model runtime,
  and one canonical artifact/CLI path. The split resolves the architecture
  guard's former >1,000-line single-file hard violation; no alternate encoder,
  reader, launcher, or legacy compatibility path remains active.
- Retirement triggers are explicit: remove the BDDL and robosuite repairs after
  a pinned dependency upgrade proves the upstream wheels no longer contain the
  duplicate metadata/shared-log defects; remove the local SmolVLA/LIBERO
  bindings when pinned upstream APIs propagate constructor/tokenizer revisions
  and asset revisions end-to-end; retire the local gallery owner only after a
  selected durable experiment system preserves the same local-video hashes,
  safe path checks, and bounded retention contract. Git preserves old evidence
  rather than retaining superseded executable paths.
- The architecture guard reports no hard violations, large functions, dense
  directories, or parallel implementation families. Its sole review flag is
  the four-file surface: one canonical eval shell entrypoint, one reusable
  artifact owner, and their two focused test files.

## Gate 0 pilot frozen before policy outcomes

- Added `configs/gate_zero_oracle_pilot.toml` and its explanatory contract before
  any LIBERO-90 source policy training or outcome access. It binds the permanent
  split, canonical manifest, source-only normalization, base/model/data
  revisions, source task pair 3/4, episode partitions, thresholds, target
  matrices, seeds, optimizer budgets, matched rollout init states, failure
  classes, and single bounded recovery. The two-task result cannot authorize
  Writer training.
- The access amendment preserves the earlier video evidence without pretending
  it never happened: demos 40--47 had RGB-only exposure, demos 46--49 still
  have locked actions/rewards/policy outcomes, and only 48--49 are untouched in
  every field. The streaming surface factory derives tasks from the checksummed
  canonical manifest, exposes all 60 source tasks only for base-fit, restricts
  support/query to source tasks 3/4, and refuses report construction without a
  selected-adapter freeze record.
- Added a deterministic, resumable task→demo→frame sampler and a lazy per-worker
  HDF5 dataset. It constructs state from `ee_states + gripper_states`, applies
  exactly one H/W camera flip, emits 50-step action chunks with repeat-last
  padding plus an explicit mask, and never creates a duplicate converted video
  corpus. Synthetic tests cover partition denial, unknown demos, image/state
  transforms, padding, and O(1) step-resume identity.
- Pinned PEFT 0.19.1 in the project and Phase 0 environment lock. Static
  safetensors inspection verifies the four declared action-expert q/v matrices
  and exactly 40,320 rank-8 parameters. Fixed query noise/time is keyed by
  task/demo/frame rather than batch order, preventing the earlier batch-shape
  mechanics issue from changing the common random numbers.
- Source-only mechanics checks, still before any policy outcome, established
  that HDF5 demo init rows are not official pruned-init rows and that HDF5 RGB
  matches raw simulator orientation. A real source support batch then passed
  the exact SmolVLA tokenizer/normalizer interface as finite two-camera 128x128,
  8D state, 50x7 action, padding-mask, and 48-token tensors. Live preflight
  selected one free device for the short EGL probe, released it afterward, and
  did not interfere with unrelated jobs.
- Added local-only Trackio 0.30.1 logging under
  `$EMBER_OUTPUT_ROOT/trackio`; model mechanics runs can be reviewed with
  `trackio show --project EMBER_gate0`, without uploading runs or exposing host
  paths through a remote service.
- The first adapter model probe at clean commit `bb1a0bb` stopped before model
  forward/backward on missing post-preprocessor provenance. Commit `51b9405`
  moves fixed-noise row-key capture before preprocessing and adds the regression
  test. Its recovery completed with exact target/count resolution, finite
  gradient/update, checksums, and bit-exact adapter reload.
- That recovery also showed PEFT `orthogonal` was not functional-zero. The
  source contract is narrowly amended to the standard nonzero-`A`/zero-`B`
  no-op initializer, with exact physical-delta and fixed-loss assertions. This
  is a pre-training mechanics correction after one disclosed support batch, not
  a threshold, split, target, rank, optimizer, or outcome-selected change; a
  clean-commit recovery was required before batch calibration.
- Commit `2d6d3d3` and long-run
  `gate_zero_adapter_model_mechanics_exactzero_20260718_073720` close that
  recovery. The real four-target adapter starts with exact zero physical update
  and exact base-loss identity, takes a finite nonzero step, saves/reloads
  bit-exactly, passes every checksum, records Trackio metrics, and releases its
  sole GPU. No Gate or Writer decision is implied.
- The source-base one-step probe completed separately from clean commit
  `51b9405`: 99,880,992 trainable parameters, finite loss and gradient, 1,818
  MiB peak PyTorch reserved memory, 2,345 MiB peak sampled device memory, and
  rc 0 with GPU 4 released. It does not establish source competence.
- The batch-calibration path loads the all-source dataset, model, normalization,
  and Trackio process once; evaluates microbatches 8/16/32/64 at effective batch
  64 using four persistent HDF5 workers; includes data loading in timing; records
  no loss or policy outcome; and stops after the first OOM. One warmup plus two
  measured optimizer steps per candidate make worker startup non-comparative.
  The shared model/preprocessor/loss owner is reused from validated mechanics.
- Clean commit `394ef4a` completed the canonical calibration long-run
  `gate_zero_source_base_batch_calibration_20260718_075123`. All candidates
  passed headroom; measured throughput was 49.03, 76.25, 86.30, and 92.19
  samples/s for microbatch 8, 16, 32, and 64. The selected microbatch 64 uses
  accumulation 1, peaks at 19,441 MiB sampled device memory, and retains 61,712
  MiB free. The 71.44-second result and four-step Trackio series are checksummed,
  outcome-free, and the sole GPU was released.
- A post-run matching audit showed that this first run successively reused the
  updated model/optimizer and changed samples with accumulation partition. It
  is retained as resource telemetry, but its selection authority is now
  `superseded_pending_matched_recovery`; no base training is authorized from it.
  The recovery was frozen before any source training: each candidate restores
  the same trainable snapshot, resets global RNG, starts a new empty AdamW, and
  uses absolute effective-batch slots plus matching fixed-noise row keys. The
  result builder compares every per-step row digest and fails closed on drift.
- Clean commit `e3a653a` completed long-run
  `gate_zero_source_base_batch_calibration_matched_recovery_20260718_081737`
  with main rc 0. The checksummed result is
  `gate_zero/batch_calibration/source_base_matched_20260718T081723Z` at SHA256
  `849b0ad2dc3ff8d2eb4088e570a50d331e8c86da1524515a4b46945722b40ead`.
  All four candidates share all three row digests and fixed flow seed; the
  frozen winner is microbatch 64 / accumulation 1 at 92.17 samples/s, 19,403
  MiB sampled peak, and 61,750 MiB minimum free. The run recorded no outcome,
  updated the atomic `latest`, and released GPU 4.
- The same result measures mixed native AdamW state: 96,607,440 bf16 and
  3,273,552 fp32 parameter elements, with both moments following parameter
  dtype plus 155 fp32 step scalars. The contract precision label is amended
  before formal source training. Batch shape is now authorized; formal base fit
  remains false until the real stochastic checkpoint/resume identity probe.
- The retained implementation has one optimization path:
  `gate_zero_base_runtime.py` supplies the loader, optimizer, pinned upstream
  scheduler, component loader, and optimizer step to calibration and training;
  `gate_zero_checkpoint.py` owns LeRobot-format atomic full-state artifacts,
  sidecar/whole-tree hashes, explicit late RNG restore, and validated rotation;
  `gate_zero_base_train.py` is the sole CLI owner for both the exact resume probe
  and the later authorized fit. There is one shell entrypoint and no fallback
  trainer, alternate sampler, or evaluator.
- The trainer slice adds three active source/test/entrypoint files and about
  1.3k net source/test lines, so the architecture guard requires an explicit
  rationale. The 695-line orchestrator is above the review signal but below the
  escalation boundary: its two modes deliberately share component loading,
  stochastic optimizer steps, checkpoint metadata, state hashing, Trackio, and
  result publication, while serialization remains in the separate 334-line
  checkpoint owner. Splitting the two modes into runners would create the
  parallel paths this contract forbids. The guard reports review signals only,
  no hard violation. Retire the probe-only branch after Gate 0 evidence is
  frozen; retire the local checkpoint/runtime owners only if pinned LeRobot
  gains equivalent absolute-step sampling, atomic commit, hash validation, and
  late-RNG resume semantics. Git and result manifests preserve old evidence.
- The prelaunch regression surface now has 131 passing tests plus Python compile
  and shell dry-run checks. It covers scalar/mixed-dtype state hashing, model/
  optimizer/scheduler loading without RNG mutation, explicit late RNG restore,
  payload/manifest tamper rejection, overwrite refusal, atomic `last`, and
  validated two-checkpoint rotation. Formal training authorization remains
  false; no source-base multistep run has started.
- Clean commit `a4689b7` completed
  `gate_zero_source_base_resume_probe_20260718_084514` with main rc 0 in 182.32
  seconds. All seven recorded comparisons (completed step plus six exact state/
  data surfaces) pass; the two branch records are identical. The canonical
  result is `gate_zero/resume_probe/source_base_resume_20260718T084500Z` at
  SHA256 `bc7a17cd3ddb0b8c3f6daf5f529b0357ff65fa426c85d90d90b6592ecbe5d3ed`.
  Trackio contains both pass records and 156 system rows. Peak sampled device
  memory was 19,467 MiB, the validated 1.32GB checkpoint was removed, the final
  output is 48KB, and GPU 4 is released.
- The resume evidence changes only `formal_base_fit_authorized` to true. Gate
  -1, Gate 0, source competence, and Writer authorization remain open/false.
  The next operation is the canonical 10,000-step all-source fit; no task-local
  adapter or held surface may be accessed first.
- Before launching that fit, a dry-run audit found the trainer wrote Trackio on
  every step even though the frozen tracking contract says every 10 optimizer
  steps, while stdout had no incremental progress. The unique mechanical fix
  now logs/flushes step 1, each tenth step, and final/checkpoint boundaries to
  both Trackio and durable JSON lines. Sampling, optimization, checkpoints,
  thresholds, access surfaces, and the 10,000-step budget are unchanged.
- Since that logging-only patch changed a manifest-bound trainer hash, clean
  commit `d71c9ce` ran one exact revalidation as long-run
  `gate_zero_source_base_resume_probe_revalidation_20260718_085806`. Main rc is
  zero; all branch digests are unchanged and exact, Trackio has both pass rows
  plus 155 system rows, peak memory is again 19,467 MiB, the validated transient
  checkpoint is cleaned, and GPU 4 is released. The active checksummed result is
  `gate_zero/resume_probe/source_base_resume_revalidation_20260718T085753Z` at
  SHA256 `fab1eb111b5b2edf32d9103a51ff0e4ec6783ead1a1d2a09c372ca4a6e3ceab1`.
  No executable source is changed after this revalidation; the remaining edits
  only reseal evidence metadata and prose.

## Immediate handoff

1. Preserve the permanent reseal and fresh canonical manifest as the only active
   split/normalization path; do not reuse the rejected split's normalization.
2. Preserve both same-state native-goal mechanics and the same-observation
   language-action path, then run action-hidden video content/temporal controls.
   Do not use held results to choose thresholds, task IDs, or remedies.
3. Preserve the completed single-GPU 10,000-step reference. Finish the clean
   world-size-4 resume recovery and matched 1/2/4 topology selection, then run
   the frozen source competence arms. Only a competence pass permits the
   predeclared task-local Gate 0 oracle; direct Writer authorization remains
   false until locked Gate 0 evidence exists.

## Last verified handoff facts

- The repository now contains only lightweight environment/contract code and
  tests; external models, data, caches, and outputs remain outside Git.
- The permanent split record is `configs/libero90_split_reseal.json` at SHA256
  `9f5bc62e15e2cb07887e97bc98630a3f527ac6b5e253f41c203cf37459568428`.
  It is specification-only, reproduces the old failure, validates the new split,
  and does not constitute a Gate -1 or Writer authorization decision.
- The active canonical audit is longrun
  `.codex/longrun/libero90_resealed_manifest_20260718_044358` plus
  `$EMBER_OUTPUT_ROOT/phase0/libero90_manifest/reseal1_20260718T044331Z`.
  Its local review page is the corresponding `latest/index.html`; the rejected
  split's prior artifact is retained by its immutable directory, not deleted.
- `docs/expert_plan.md` uses an obsolete eight-GPU planning envelope; active
  execution must recalculate every launch for at most four GPUs.
- A gate failure requires diagnosis and bounded recovery, not immediate
  abandonment and not post-hoc weakening of held-out constraints.
- The canonical strict-identity evidence is
  `.codex/longrun/gate_minus1_identity_mechanics_20260717_160231` plus
  `$EMBER_OUTPUT_ROOT/gate_minus1/evaluation_identity_mechanics_20260717T160231Z`.
  It stopped before policy load, used one GPU, and retains result, telemetry,
  resource summary, failure packet, and checksums.
- The canonical policy and mechanism evidence is in the two corresponding
  long-run records ending `161316` and `162212`, with checksummed external
  artifacts ending `T161316Z` and `T162212Z`. The first proves action/trajectory
  batch sensitivity; the second localizes it to model-forward batch shape.
- Cleanup removed 15 verified-regenerable temporary files totaling 622,289
  bytes: the preliminary mechanics smoke, recovered mechanism-launch residue,
  copied local diagnostic source/cache, old smoke launch wrappers/contact sheet,
  and pytest cache. Canonical outputs and all long-run state/logs were retained.
- The latest completed visual review page is available locally at
  `$EMBER_OUTPUT_ROOT/gate_minus1/specification/latest/index.html`; the earlier
  Phase 0 gallery remains available under its own `latest` link. The canonical
  data/task audit is separately available at
  `$EMBER_OUTPUT_ROOT/phase0/libero90_manifest/latest/index.html`. Historical
  run directories are retained until verified-regenerable, unpinned media
  becomes large or numerous enough for a recorded cleanup.
- The source paired-goal mechanics record is
  `.codex/longrun/gate_minus1_source_same_init_goal_20260718_050511` plus
  `$EMBER_OUTPUT_ROOT/gate_minus1/specification/source_same_init_goal_20260718T050511Z`.
  The atomic `source_same_init_goal_latest/index.html` report is 21KB and needs
  no cleanup; it is mechanics-only and leaves the Gate/Writer decision false.
- The same-observation language-action record is
  `.codex/longrun/gate_minus1_language_action_20260718_052249` plus
  `$EMBER_OUTPUT_ROOT/gate_minus1/specification/language_action_20260718T052249Z`.
  Its atomic `language_action_latest/index.html` report is 268KB; correct repeat
  is exact and all three prompt contrasts are substantive for 16/16 samples,
  while correct paired-goal switching and the Gate/Writer decisions remain open.

## 2026-07-18 multi-GPU efficiency work

- The original 10,000-step source-base fit completed from clean commit
  `8ff06f2` under long-run ID
  `gate_zero_source_base_fit_10k_20260718_090528`, with main rc 0 and no restart
  or exposure to later implementation changes. The checksummed result reports
  10,000 steps and 7036.82 seconds; full-tree validation passes for retained
  schema-2 recovery steps 8000/9000 and final step 10000. The final role is
  `source_base_candidate_pending_competence`, result SHA256 is
  `0db5485707711657ecaad2806019c0d28d3a2ec9b94973a5c4aa7b327dc2a1b2`,
  and final manifest SHA256 is
  `ca0c83abd8d4b46cf59e8f0a01bd267f7f0e019d3e2bfea8c8baeb2e851d4d00`.
  Active telemetry averaged 96.31% GPU utilization, peaked at 19,311 MiB, and
  Trackio reached step 10000 before GPU release. No loss value is interpreted
  as policy competence.
- The pre-outcome 1/2/4-GPU amendment was developed in the isolated
  `codex/gate0-multigpu` branch, validated, pushed as `cc4ba36`, and integrated
  only after the final reference hashes froze, as implementation commit
  `39bfee9`. It adds one canonical
  torchrun/DDP trainer topology, deterministic global sharding, rank-0 flow
  generation/scatter, rank-aware atomic checkpoint/resume, fixed topology
  report generation, and source-arm parallel competence evaluation.
- Verification currently passes: Python compilation, shell syntax/dry-runs,
  diff whitespace checks, architecture guard with zero hard violations, and
  all 161 repository tests. Real CPU/gloo subprocess tests cover world sizes 2
  and 4 for gradient aggregation, global data/flow identity, checkpoint
  publication, post-publication rollback with prior-`last` restoration, and
  exact same-topology resume. The report refuses telemetry that differs from
  its run-finalized checksum. INT/TERM cleanup handlers stop telemetry samplers
  without reintroducing the prohibited Bash `EXIT` trap.
- Final topology selection remains pending. The first fixed probes and bounded
  recovery are recorded below; repeat world sizes 1, 2, and 4 under one amended
  contract before freezing the selection report/config. Then run the source-only
  competence arms and follow the already frozen pass/recovery decision; Gate -1,
  Gate 0, and Writer remain unauthorized.

## 2026-07-18 topology-probe recovery

- The first matched world-size 1/2 probes under contract `e334d4e8...` completed
  with rc 0, exact resume, checksum-bound telemetry, and 93.353/170.822 global
  samples/s. The world-size-4 run
  `gate_zero_topology_probe_w4_20260718_111048` completed its fixed 30 steps and
  valid schema-3 checkpoint but failed at the continuous-versus-resumed step-31
  model hash. No policy outcome was recorded and no selection was made.
- A four-GPU read-only checkpoint diagnostic then ran two independent fresh
  resumes. Model, optimizer, scheduler, per-rank RNG, next batch, row keys,
  flow inputs, and all 64 slots were exact, with zero differing tensors. This
  excludes checkpoint/data/RNG recovery and isolates the continuous-versus-new
  DDP reducer bucket lifecycle. The failure packet and telemetry are checksummed
  under `$EMBER_OUTPUT_ROOT/gate_zero/topology_probe/world4_20260718T110600Z`;
  the validated 1.32GB transient checkpoint was removed, leaving 96KB.
- Tests now require the topology contract and both 2/4-rank DDP paths to use a
  static graph. The single trainer binds this setting into schema-3 checkpoint
  topology. Before resealing any selection, run the live four-rank resume check
  and repeat all three matched throughput candidates under amended contract
  SHA256 `84c5bcf7...`; the prior throughput numbers are diagnostic only.

## 2026-07-18 direct-Writer scope correction

- Active contracts now remove mandatory Gate 1 and every planned
  canonical-bank/shared-update-subspace/soft-geometry/residual-escape milestone.
  The old route supplied a shared span, a coordinate preconditioner, and an
  escape from that span; it was a second Writer-conditioned RL search space and
  is not merely an optional later experiment.
- The current structural space is only the frozen LoRA target-layer/rank
  contract. Writer emits every task-specific LoRA factor in that space;
  ordinary task-local RL updates those same factors in place with no predicted
  bank, basis, mask, metric, radius, or learning-rate object.
- During direct Writer and default source reward/meta-RL, the shared base stays
  frozen. Inner learning updates task-local LoRA and the outer source objective
  updates Writer parameters. Shared-base/shared-LoRA source outer training is a
  future separate matched ablation, not the mainline or completion evidence.
- Gate -1 and Gate 0 remain unchanged. Gate 0 independently establishes useful
  task-local LoRA and upper-bound/baseline evidence; a positive locked result
  leads directly to full-LoRA Writer acquisition with independent source-query
  functional supervision. `docs/expert_plan.md` is deliberately unchanged as
  historical advice.
- The owner deleted the obsolete Goal. The Goal service now stores a new active,
  unbudgeted full-lifecycle EMBER objective spanning base/spec validity, Gate 0,
  direct Writer, A/B/C task-local LoRA RL, source reward/meta outer learning,
  frozen-held evaluation, and OpenVLA-OFT confirmation. It explicitly excludes
  Gate 1/bank/geometry and forbids completion at an infrastructure or early-Gate
  milestone.

## 2026-07-18 static world-size-4 RNG recovery

- Live long-run `gate_zero_world4_static_resume_20260718_112741` showed that
  DDP static graph fixed the prior continuous/resumed model mismatch. Model,
  optimizer, scheduler, next batch, and row keys were exact; only the aggregate
  per-rank RNG digest differed. Telemetry peaked at 12,162 MiB and retained
  68,993 MiB free, excluding OOM.
- Diagnostic long-run
  `gate_zero_world4_static_rng_diagnostic_20260718_113544` completed rc 0 and
  localized the sole difference to Python `random`; NumPy, CPU Torch, CUDA RNG,
  flow inputs, loss, gradient, all model tensors, and state/data surfaces were
  exact. The checkpoint exactly matched resumed pre-step RNG on every rank.
- A regression first failed because a fresh runtime seeded only before one-time
  setup. The unique fix reseeds fresh training RNG after model/DDP/loader/
  authority setup, while resume restores checkpoint RNG at that same boundary.
  The focused four-test slice is green. The 1.32GB diagnostic checkpoint was
  whole-tree validated and removed; a checksummed 96KB failure packet remains.
- No topology has been selected. After documentation/tests and a clean commit,
  run one live four-rank exact-resume check; only then repeat world sizes 1/2/4
  under the new contract SHA256
  `04bf00a4326f62119b32ca22ef9836980d5743e61eb2f1366e85ae4feae25e9d`
  and build the local topology report.
- Verification passes all 165 tests, Python compilation, all shell syntax, and
  diff whitespace checks. Three superseded temporary diagnostic scripts plus
  two bytecode files (33,379 bytes) were removed after their compact result and
  failure evidence froze; the small video-inspection images remain because they
  are distinct visual diagnosis artifacts rather than duplicate run output.

## 2026-07-18 bounded world-size-4 closeout and science-priority reset

- Clean long-run `gate_zero_world4_rng_boundary_resume_20260718_120328` from
  commit `9a8a8f5` completed with main rc 0 under topology contract SHA256
  `04bf00a4326f62119b32ca22ef9836980d5743e61eb2f1366e85ae4feae25e9d`.
  Its result SHA256 is
  `e996dd9f5290bbb8302c1a051d81783a3353ad433d15530742ae436f6ef60a12`.
  Completed step, model, optimizer, scheduler, aggregate RNG, next raw batch,
  and next row keys all match; more importantly, the schema-3 checkpoint loaded,
  restored world size 4, and reproduced the declared training/data state.
- The run used GPUs 4--7 only, peaked at 12,162 MiB per sampled device, retained
  at least 68,993 MiB free, completed in 93.19 seconds of measured probe time,
  released all four GPUs, and removed the validated 1,319,495,706-byte transient
  checkpoint. Checksummed result/telemetry and Trackio provenance remain in a
  100KB canonical output; no EMBER process remains.
- Owner stop rule closes exact-resume work here. Ambient RNG, bitwise, telemetry,
  and extra identity surfaces are no longer research blockers once checkpoint
  load/state/cursor and short-resume functional behavior are operationally
  sufficient. Non-scientific anomalies receive one reproduction, one narrow
  repair, and one verification unless they can affect recoverability, sampled
  data, closed-loop success, a Gate decision, matched fairness, or held
  isolation.
- The previously planned full matched 1/2/4 rerun is superseded. Run at most one
  necessary short world-size-4 throughput/stability window, then proceed to the
  frozen source tasks-3/4 competence arms, remaining Gate -1 evidence, and Gate
  0 LoRA oracle/capacity audit. Use four-way arm/task/seed parallelism if a
  single DDP job does not scale well; do not leave safe available GPUs idle.
- Active contracts now fix LoRA as the only adaptation mechanism, the Gate 0
  capacity audit, direct complete-LoRA Writer supervision, the A/B/C ordinary-RL
  causal arms and metrics, frozen-base source outer learning, frozen-held
  isolation, positive long-term completion evidence, and OpenVLA-OFT scale
  confirmation. Historical `docs/expert_plan.md` remains unchanged provenance.

## 2026-07-18 source competence unlocks Gate 0 oracle fitting

- The one permitted four-card short window ran from clean commit `e386925` and
  measured roughly 294 global samples/s with safe 68,787 MiB minimum free. Its
  step-30 checkpoint passed full-tree validation and loaded/reran step 31, but
  the continuous and restarted model hashes were not bitwise equal. The failed
  long-run state, log, and telemetry are retained; the validated 1.32GB
  checkpoint was cleaned. Per the owner timebox there is no rerun or new
  instrumentation. Four-card independent science parallelism is selected over
  long world-size-4 DDP.
- Formal source competence then completed in four-way arm parallel under
  long-run `gate_zero_source_competence_base10k_20260718_122116`, main rc 0.
  Task 3 correct/swapped was 8/8 versus 0/8; task 4 was 5/8 versus 0/8. All
  mechanics are valid, the frozen minimum of 2/8 per correct task passes, and no
  20k recovery is authorized or needed.
- The 836KB canonical artifact is
  `$EMBER_OUTPUT_ROOT/gate_zero/source_competence/base10k_20260718T122059Z`;
  `latest` resolves there. Result SHA256 is
  `c9697c4cf71d452c431424be4cd12fd6a869ac4fd58755d07152ee6928da83cc`.
  Checksums cover result, eval info, gallery, telemetry, and four videos. Trackio
  run `base10k_20260718T122059Z` is available in project `EMBER_gate0`; GPUs
  4--7 are released.
- Decision scope is narrow: task-local LoRA oracle fitting is now authorized.
  Gate -1, Gate 0, Writer, validation, and held outcomes remain unauthorized.
  Next implement the sole frozen Gate 0 oracle path and bounded LoRA capacity
  audit, then launch task 3/4 fits as independent GPU jobs with locked query and
  rollout reporting.

## 2026-07-18 Writer target-support authority clarified

- The owner has frozen the current last-two q/v rank-8 target set as a Gate 0
  pilot only, not the final Writer contract. No healthy Gate 0 run is changed or
  interrupted by this clarification.
- Active planning now requires one bounded, predeclared source/validation-only
  support audit before Writer target sealing. It must include the pilot set,
  all action-expert q/v, and near-default SmolVLA v0.6.0 PEFT support, may adjust
  rank only within the bounded audit, and then permanently reseals exact target
  names, rank, alpha, dropout, and parameter count with held access still zero.
- All Writer, zero-init ordinary RL, average/retrieval, language-only
  HyPoGen/DISC-style generation, and other matched arms will use that same final
  support. Broad-output difficulty will be addressed with structured
  layer/module-aware generation rather than silently shrinking capacity or
  reintroducing any bank/subspace/geometry path.
- The single Gate 0 fit/select/checkpoint implementation is now under TDD. Its
  contract and artifact tests pass 10/10; long-run launch remains pending full
  repository verification and a clean commit.
- Architecture ownership is explicit despite the necessarily broad scientific
  surface: `gate_zero_oracle_contract` owns sealed authority checks,
  `gate_zero_oracle_session` owns live model/data/optimizer lifetime,
  `gate_zero_oracle_metrics` owns fixed query/drift evaluation,
  `gate_zero_oracle_artifacts` owns atomic state publication, and
  `gate_zero_oracle_fit` is the sole CLI orchestration path. The launcher adds
  no second trainer. The older one-step `gate_zero_model_probe` remains only as
  already-published mechanics provenance until the first completed LoRA and
  partial fit plus locked report verify the replacement; at that trigger,
  retire its executable launcher/module while preserving durable invariants and
  Git evidence. No bank/geometry or hypothetical Writer code path is reserved.

## 2026-07-18 Gate 0 oracle fit closeout and locked-report preparation

- Four independent long-runs filled GPUs 4--7 without touching the unrelated
  GPU 0--3 jobs: `gate_zero_oracle_fit_lora_task3_20260718_131711`,
  `gate_zero_oracle_fit_lora_task4_20260718_132013`,
  `gate_zero_oracle_fit_partial_task3_20260718_132014`, and
  `gate_zero_oracle_fit_partial_task4_20260718_132014`. All completed main rc 0
  and released their GPUs. LoRA fit bodies took 1153.34/1151.11 seconds with
  about 3.70GB peak Torch allocation; partial fits took 1468.24/1478.24 seconds
  with about 18.27GB. The devices stayed compute-saturated during fitting;
  fixed effective batch 64 and independent-job parallelism preserved the
  scientific sample budget rather than inflating memory with dummy work.
- Query-only selections are task 3 LoRA step 0, task 4 LoRA step 250, and both
  partial upper bounds step 0. Task 3's step-250 LoRA query gain was 3.27% but
  violated drift 0.02; task 4's selected gain was 1.12% at drift 0.01995. Both
  partial fits sharply reduced support loss but worsened independent query loss
  by more than 54% at step 250. Gate 0, Writer, and final Writer target support
  remain false.
- Full checksum validation passes for all four outputs. Completed recovery
  directories and unselected bulky partial states were removed by the canonical
  fitter; selected states, compact candidate metrics, LoRA candidates,
  telemetry, and result provenance remain. The full oracle-fit tree is about
  397MB, so no further evidence cleanup is warranted now.
- The locked-report implementation was developed in isolated worktree/branch
  `codex/gate-zero-locked-report` so the running launchers stayed on commit
  `96cd0f9`. Its sole path first validates all four completed fit results and
  selected hashes, atomically publishes an immutable selection-freeze grant,
  then uses four fixed shards for base/own/swapped/non-matched-partial offline
  and closed-loop arms. It enforces matching report rows, sample counts,
  seeds/init states, base losses within functional tolerance, correct prompts,
  and the two-reset transition to official init states 16--23. Trackio and one
  video per arm provide bounded live/later visualization.
- Architecture ownership is intentionally narrow: `gate_zero_oracle_report`
  owns the irreversible grant and frozen decision arithmetic;
  `gate_zero_oracle_report_runtime` owns the only live report evaluator;
  `gate_zero_oracle_session` remains the shared variant-construction owner; and
  one shell launcher owns GPU preflight, telemetry, cleanup, and torchrun. This
  adds no alternate trainer or evaluator mode. The older one-step model probe
  keeps its previously recorded removal trigger after the first locked report;
  the new report path remains only while Gate 0 evidence is an active project
  requirement. The architecture guard reports review signals (large but below
  hard limits) and no hard violation; the cohesive exception is justified by
  the irreversible data gate plus simulator/report lifecycle.
- Synthetic grant/decision/shard/reset tests pass 5/5; the full repository suite
  passes 181/181 with Python compilation and shell syntax checks. The report
  surface is still unopened. After integrating this clean implementation,
  create the grant exactly once, run the fixed four-card locked report, and only
  then freeze the Gate 0 failure/recovery decision.
- The first report launch permanently created and validated the selection grant,
  then all four ranks failed before model/data loading because the new runtime
  hard-coded `checkpoint_manifest.json` instead of using the checkpoint
  owner's canonical manifest constant. Long-run
  `gate_zero_oracle_locked_report_20260718_135155` and its four compact failure
  packets/telemetry are retained; no rollout or scientific report metric was
  produced. One red regression, one narrow replacement with
  `CHECKPOINT_MANIFEST`, and one real read-only authority preflight now pass.
  Retry must reuse the immutable grant and write a fresh report directory; no
  second selection-freeze artifact is permitted.
- Recovery long-run
  `gate_zero_oracle_locked_report_recovery1_20260718_135618` completed all eight
  offline/closed-loop arms with mechanics valid and generated all eight videos,
  then failed only during rank-0 aggregation because the first implementation
  required cross-GPU/zero-PEFT base flow MSE to match at rtol `1e-7`. Observed
  harmless relative variation was about `2e-5`--`4e-5`, far below the frozen
  20% utility threshold; task 3's zero-LoRA own arm also matched base success
  exactly. Per the owner stop rule, a red/green regression now fixes functional
  matching at rtol `1e-4`, atol `1e-8`, while explicitly rejecting a `1e-3`
  discrepancy. The tolerance is emitted in the final decision. The failed
  aggregation packet, telemetry, and videos remain; run one fresh final report
  from the same immutable grant because per-episode arm arrays were not
  published before aggregation.

## 2026-07-18 final Gate 0 locked report

- Final long-run `gate_zero_oracle_locked_report_final_20260718_140322` reused
  the sole immutable grant, completed all eight arms and 64 episodes with main
  rc 0, and released GPUs 4--7. The result is
  `$EMBER_OUTPUT_ROOT/gate_zero/oracle_report/locked_final_20260718T140322Z`
  with result SHA256
  `b7fcfc6227ba7fd6fc2e9ad21b2e55978b54d668476c9c520e216536739e9d91`;
  `latest/index.html` exposes the bounded eight-video gallery and Trackio run
  `locked_final_20260718T140322Z` remains under project `EMBER_gate0`.
- All result/gallery/video/telemetry checksums pass. The four-card report body
  took 199.17 seconds. Sampled peak memory was 21,932 MiB on rank 0 and about
  4.3--4.6GiB on the other ranks; simulator-bound active utilization was lower
  than training, but all four cards concurrently executed disjoint fixed
  shards and no additional legal arm remained to fill memory. The complete
  report/failure/grant tree is only 3.3MB, so no evidence cleanup is warranted.
- Task 3 base/own/swapped/partial successes are 5/5/4/5 of eight; task 4 is
  1/3/1/1. Task 4 own LoRA yields +25pp and 0.82% locked-flow reduction, while
  task 3 own is the selected zero update. Aggregate median gain is 12.5pp,
  positive-task fraction 0.5, median locked-flow reduction 0.00410, and median
  selection drift 0.00998. Only the drift threshold passes; the own-minus-base
  bootstrap 95% CI is [0, 37.5]pp.
- The frozen decision is `gate_zero_pilot_failed` with failure class
  `task_local_lora_oracle_utility_not_established`. Gate 0, Writer, validation,
  held evaluation, and final Writer target sealing remain unauthorized. This is
  not a reason to stop the long-term Goal: fit/query evidence localizes the next
  bounded recovery to acquisition generalization plus target support, not
  benchmark mechanics, resources, or Writer acquisition.
- Active priority moves to one predeclared source/query-only audit combining a
  bounded early-optimization remedy with last-two q/v, all action-expert q/v,
  and near-official SmolVLA support. Rank changes only if that audit requires
  them. Selection cannot read the locked report again; the matched closed-loop
  recovery uses only the already reserved fresh recovery init states. Writer
  remains blocked until a useful independent LoRA oracle is established and
  the final support is permanently resealed.

## 2026-07-18 target-support audit predeclaration

- Added the pre-outcome authority
  `configs/gate_zero_target_support_audit.toml` and concise rationale
  `docs/gate_zero_target_support_audit.md`. It binds the failed locked result
  SHA256 `b7fcfc6227ba7fd6fc2e9ad21b2e55978b54d668476c9c520e216536739e9d91`,
  immutable grant, source competence, base checkpoint, split, and held-zero
  access. The active long-term Goal already covers this bounded LoRA capacity
  audit and remains unchanged, active, and unbudgeted.
- Extended the sole recoverable oracle fitter/launcher by configuration rather
  than adding a parallel trainer. Legacy `lora` and `partial_upper_bound`
  contracts still resolve identically; named support variants share the same
  model/data/optimizer/artifact owners. Path-safe variant IDs remain contract
  restricted.
- Unit contract tests pass for exact target nesting, frozen counts, prior-report
  hash failure, legacy/new single-path resolution, and launcher dispatch. The
  previous ten oracle execution tests, Python compilation, shell syntax, and
  diff checks pass. Read-only inspection of the frozen checkpoint confirms
  target/count tuples `(4, 40320)`, `(32, 322560)`, and `(37, 371328)`.
- Next: run one live zero-step target-resolution smoke, then launch the six
  750-step source fits in at most four concurrent one-GPU jobs with Trackio and
  recoverable state. No locked-report, validation, or held numeric result is
  available to fitting or selection.
- The one permitted live zero-step smoke completed rc 0 on physical GPU 4 and
  released it back to 0MiB. The three variants resolved exactly 4/32/37 target
  modules, 8/64/74 trainable LoRA tensors, and
  40,320/322,560/371,328 parameters; all physical deltas were exact zero.
  Peak Torch allocation was only 902MiB because no batch or outcome surface was
  loaded. No additional target-identity instrumentation is planned.
- Final verification passes 185/185 tests in 44.60 seconds, all Python
  compilation, all shell syntax checks, and diff whitespace checks. The
  architecture guard reports REVIEW with no hard violation: net active-source
  growth is 559 lines across seven changed and two new files. The new
  `gate_zero_support.contract` has one cohesive owner--the irreversible
  audit authority--and avoids mixing the superseded pilot contract with its
  bounded recovery. The fitter, session, artifacts, and launcher remain single
  shared paths. After the final LoRA support is sealed and its report exported,
  remove or freeze the audit dispatch in the next retirement review rather
  than extending it into a general experiment framework.

### Formal rank-8 support-fit launch contract

- Canonical implementation revision is clean commit
  `c8c0ad0e4c2ca7845e2ef90a96302fd5257abe8a` on
  `phase0/reproducible-substrate`; launch occurs from its clean documentation-only
  descendant, whose exact HEAD is captured by every long-run record. The six outputs live below
  `$EMBER_OUTPUT_ROOT/gate_zero/target_support_audit/fit/rank8_20260718T143438Z`;
  stable `latest_<variant>_task<id>` links live in the parent fit directory.
- Every job calls the sole `scripts/run_gate_zero_oracle_fit.sh` with
  `--config=$PWD/configs/gate_zero_target_support_audit.toml`, one declared
  support variant, one source task, one physical GPU, and a fresh output. The
  inputs are the frozen 10k source-base checkpoint, source demos 28--39,
  query-only demos 40--45, fixed noise/time, effective batch 64, LR `1e-4`, and
  750 optimizer steps. Locked demos, validation, held, rollout reward, and
  Writer surfaces remain unavailable.
- Wave 1 assigns last-two task 3/4 and all-expert task 3/4 independently to
  GPUs 4/5/6/7. Wave 2 assigns official-default task 3/4 to GPUs 4/5 only after
  wave 1 releases them. Expected peak is 20--22GiB and roughly 10--15 minutes
  per job; each wave stops at step 750 or on first failure. Four distinct jobs
  already saturate useful compute in wave 1. The two-job second wave does not
  duplicate work merely to occupy GPUs 6/7, and colocating compute-saturated
  jobs would reduce throughput despite consuming more memory.
- Current personal storage is 294GiB against the 500GiB cap; projected peak
  addition is below 4GiB. Each long-run records command/Git/environment/logs,
  the fitter publishes atomic candidate and recovery states, and `--resume`
  is legal only for the identical variant/task/config output. On success,
  recovery state is removed after selected-state validation; compact candidates,
  selected state, Trackio metrics, telemetry, and checksums remain. A failed or
  partial output is never overwritten and receives a fresh recovery decision.

### Target-support screening implementation ownership

- The bounded audit now has one cohesive `ember.gate_zero_support` package:
  `contract` owns the pre-outcome authority, `screen` owns six-fit validation,
  the irreversible screening grant, and frozen decision arithmetic, while
  `screen_runtime` owns only distributed model/rollout lifecycle. The shell
  launcher owns live GPU conflict checks, four-rank topology, telemetry, and
  cleanup. The existing oracle fitter remains the only trainer.
- The screening runtime imports the already validated task-authority and
  `_closed_loop_metrics` implementation from the canonical locked-report
  runtime, including prompt override, two-reset init-state identity, upstream
  evaluator, seed handling, and bounded video generation. It does not create a
  second evaluator or read HDF5/report demos. Six fit states must hash-validate
  before the one screening grant can authorize init states 24--31.
- Architecture guard status is REVIEW with no hard violation. Relative to the
  clean launch base, the isolated implementation adds five active files and
  about 1.7k source/test/shell lines. This growth is confined to the current
  irreversible Gate recovery, not a general experiment framework: grant/
  decision code is pure and tested, runtime is thin orchestration over reused
  mechanics, and the package removes two new modules from the crowded flat
  `ember` namespace. After confirmation permanently seals or rejects the final
  support, perform a retirement review before Writer work: preserve config,
  hashes, reports, and decision tests, but remove the screening launcher/runtime
  and fit-dispatch surface from active code once no canonical rerun depends on
  them. No bank/geometry or hypothetical architecture path is reserved.
- The isolated implementation passes 192/192 repository tests in 44.53
  seconds, Python compilation, launcher shell syntax, and diff checks. Synthetic
  tests cover six-state atomic grant publication, hash/missing-state failure,
  query ranking, four-rank arm partitioning, smallest-passing-support selection,
  bounded rank-16 authorization, state authority, and the single dry-run path.

### Rank-8 fit completion and closed-loop screening launch contract

- Wave-1 long-runs `gate0_support_last2_t3_20260718_143657`,
  `gate0_support_last2_t4_20260718_143657`,
  `gate0_support_allqv_t3_20260718_143657`, and
  `gate0_support_allqv_t4_20260718_143657`; wave-2 runs
  `gate0_support_official_t3_20260718_144758` and
  `gate0_support_official_t4_20260718_144758` all completed main rc 0 and
  released GPUs 4--7. Runtime was 507--625 seconds; last-two peak Torch
  allocation was 3.70GiB and broad supports 17.23--17.36GiB. The complete fit
  tree is only 53MB, so no evidence cleanup is warranted.
- Every fit checksum and selected-state authority validates. The one screening
  grant is
  `$EMBER_OUTPUT_ROOT/gate_zero/target_support_audit/screening_freeze/rank8_20260718T150007Z/screening_grant.json`,
  SHA256 `fd8e28a7f0b828e14ff7cfb794a047409b6e8e96562646b38aef232b65332992`.
  It freezes six selected state/result/manifest hashes, records the exact query
  ranking, authorizes only source rollout init states 24--31, and explicitly
  leaves report access, rank 16, Gate 0, final target support, Writer,
  validation, and held access false.
- The next canonical command is the sole
  `scripts/run_gate_zero_target_support_screen.sh` from clean screening code
  revision `1722b9d`, with `--gpus=4,5,6,7`, the rank-8 fit root above,
  `--reuse-screening-freeze`, and output
  `$EMBER_OUTPUT_ROOT/gate_zero/target_support_audit/screening/rank8_20260718T150007Z`.
  Four ranks evaluate eight disjoint arms/64 episodes: one frozen base and all
  three supports per task, exact init states 24--31 and seeds 5500--5507. No
  HDF5/query/report/validation/held data is opened. Expected peak is below
  22GiB per rank, wall time 3--5 minutes, and storage below 1GiB. Trackio
  project `EMBER_gate0`, one video per arm, telemetry, checksums, and a `latest`
  gallery provide live/later inspection. Stop on first mechanics/authority
  failure or after all 64 episodes; never overwrite the grant or partial output.

### Rank-8 screening implementation failure and bounded recovery

- `gate0_support_rank8_screen_20260718_150220` ran from clean commit `55a18a6`
  on GPUs 4--7 and released all four devices. It finished 64 episodes in 186
  seconds but exited 1 during decision publication because all eight arms were
  mechanically invalid. The failed output, eight diagnostic videos, telemetry,
  long-run state/log, and rank-0 failure packet are retained in place; no
  scientific result, gallery `latest`, support choice, or downstream grant was
  published.
- Read-only diagnosis found one cause at the canonical reset boundary. The
  imported locked-report evaluator always performed one warm-up plus the
  rollout reset, which deterministically reaches init states 16--23. The
  screening authority requires 24--31 and therefore needs two warm-ups plus the
  rollout reset. Logged success totals (task 3 base/all-qv/last-two/official
  4/3/2/4 of 8; task 4 4/3/4/4 of 8) are explicitly quarantined because they
  came from the wrong surface and cannot influence ranking or rank escalation.
- Following the infrastructure stop rule, one test-first narrow repair now
  computes the warm-up sequence from the already frozen target batch. Existing
  16--23 behavior and the new 24--31 three-event identity both pass focused
  tests; a missing warm-up is rejected. A policy-free live async LIBERO reset
  probe also observes exact transitions 0--7 -> 8--15 -> 16--23 -> 24--31 with
  the derived seed batches and leaves no simulator process behind. The next
  action is one fresh-output replay of the same immutable grant on the intended
  24--31 surface, not a rerun or reinterpretation of the invalid outcomes.

### Formal rank-8 screening recovery launch contract

- The repaired implementation is committed as `a4d9aac` on clean
  `phase0/reproducible-substrate`; the documentation-only launch freeze is its
  clean descendant. The command reuses immutable grant SHA256
  `fd8e28a7f0b828e14ff7cfb794a047409b6e8e96562646b38aef232b65332992`
  and the six checksum-valid rank-8 selected states, but writes only to fresh
  output
  `$EMBER_OUTPUT_ROOT/gate_zero/target_support_audit/screening/rank8_recovery1_20260718T151445Z`.
  The failed output and long-run state remain unchanged.
- Four independent ranks use physical GPUs 4/5/6/7 for eight arms and 64
  episodes on source init states 24--31, seeds 5500--5507. The support states,
  thresholds, query evidence, task IDs, prompts, evaluator, policy RNG, and
  scientific budget are unchanged; the sole repair is the extra deterministic
  warm-up reset required to reach the frozen surface. Expected peak is below
  22GiB per GPU, wall time 3--5 minutes, and additional storage below 1GiB.
  Trackio plus one video per arm and a checksummed gallery provide live/later
  inspection. Stop after 64 episodes or first mechanics/authority error; no
  resume or overwrite of either output is allowed.
- Immediate preflight: GPUs 4--7 are 0MiB and idle, while an unrelated MEMLLM
  job owns GPUs 0--3 and is untouched. Personal usage is 294GiB of the 500GiB
  cap; `/data` has 2.9TiB free. Git is clean, all upstream checksums pass, the
  reset-only live probe passes, and the repository suite passes 193/193 tests.

### Rank-8 screening recovery result

- Long-run `gate0_support_rank8_screen_recovery1_20260718_151530` completed from
  clean commit `6f3cf52` with main rc 0 in 187.79 seconds. All eight arms and 64
  episodes are mechanically valid on init states 24--31/seeds 5500--5507. All
  output, gallery, telemetry, and eight video checksums pass; `screening/latest`
  resolves to the 1.7MB canonical output, and GPUs 4--7 returned to 0MiB.
- Task-3 base/last-two/all-qv/official successes are 4/3/2/3 of 8; task-4 are
  3/3/4/4. Candidate median success gains are -6.25pp, -6.25pp, and 0pp;
  positive-task counts are 0/2, 1/2, and 1/2. Combined with frozen query
  reductions 2.36%, 4.67%, and 4.79%, no rank-8 candidate passes. All three
  satisfy only the drift check.
- `support_screening_result.json` SHA256 is
  `0df3acb8d3fd5f94507921298940281c7430eedc359869d3918a0f2c012c6efb`.
  Its immutable status is `rank8_support_screen_failed_rank16_authorized` and
  sole scope is `official_default_r8` -> rank 16. Confirmation, locked-report
  access, Gate 0, final target support, Writer, validation, and held remain
  unauthorized. Next, add one hash-bound rank-16 contract using the same
  fitter/query rows/optimizer schedule and fresh init states 32--39; no other
  search is scheduled.

### Conditional rank-16 contract and implementation ownership

- Added the hash-bound post-rank-8 contract
  `configs/gate_zero_target_support_rank16.toml`. It resolves one
  `official_default_r16` candidate by inheriting the exact 37 targets and all
  fit/query mechanics from the frozen rank-8 contract, changing only rank,
  alpha and declared count to 16/16/742,656. It predeclares init 32--39 with
  seeds 5600--5607 for screening and init 40--47 with seeds 5700--5707 for a
  conditional confirmation. Failure closes all further layer/rank search.
  Contract SHA256 is
  `c205b4ef1a49670b55af4eeb829555774a1ed7c786f4a1c5ec87c64aec4048ad`.
- The sole fitter, grant builder, screening evaluator, shell entrypoint, and
  artifact schema are reused through a bounded config/stage dispatch; no
  second trainer/evaluator or rank-specific launcher was added. Targeted tests
  cover config hashes/inheritance, exact target/count, canonical fitter
  dispatch, two-rank nonduplicated shards, two-state atomic grant validation,
  and the no-further-search failure decision. The full repository suite passes
  199/199 tests; Python compilation, both launcher syntax checks, and diff
  whitespace checks pass.
- Architecture guard is REVIEW with no hard violation. Net active-source/test
  growth is 572 lines and no source module or entrypoint was added; existing
  `screen.py` and `screen_runtime.py` now slightly exceed 600 lines because the
  current second use is expressed as configuration rather than duplicated
  orchestration. Ownership remains: contract resolves authority, screen owns
  pure grant/decision arithmetic, runtime owns model/rollout lifecycle. After
  rank-16 confirmation seals or rejects support and evidence is exported,
  perform the already scheduled retirement review and remove active audit
  dispatch/launcher surfaces that no canonical rerun needs before Writer work.
- The sole live model-level smoke on physical GPU 4 passed and released the
  device to 0MiB: PEFT resolved exactly 37 targets, exposed exactly 742,656
  trainable parameters, and `configure_oracle_variant` verified every physical
  LoRA delta is exact zero. Peak Torch allocation/reservation was 902/928MiB;
  no demonstrations, rollout outcomes, validation, or held surface was opened.

### Formal conditional rank-16 fit launch contract

- Launch revision is clean commit `e3b2efa` on
  `phase0/reproducible-substrate`. Two independent canonical fitter jobs write
  fresh outputs below
  `$EMBER_OUTPUT_ROOT/gate_zero/target_support_audit/fit/rank16_20260718T153934Z/official_default_r16_task{3,4}`
  and bind contract SHA256 `c205b4ef...4048ad`, rank-8 result SHA256
  `0df3acb8...12c6efb`, source-base manifest `ca0c83ab...51d4d00`, and source
  competence result `c9697c4c...83cc`.
- Each job uses the sole oracle fitter, one physical GPU (task 3 on GPU 4,
  task 4 on GPU 5), 37 exact targets, rank/alpha 16, dropout 0, AdamW `1e-4`,
  support demos 28--39, query demos 40--45, effective/micro batch 64, and the
  frozen candidates through step 750. They share no trainable state; expected
  peak is below 24GiB per job, wall time 10--15 minutes, combined output below
  2GiB, and stop is step 750 or first failure. GPUs 6--7 remain available
  because no additional legal fit exists and duplicating either task would
  change neither evidence nor throughput.
- Immediate preflight finds GPUs 4--7 empty while unrelated MEMLLM owns 0--3;
  personal storage is 294GiB with 2.9TiB filesystem free. Git is clean, all
  named authorities validate, exact-zero smoke passes, and 199/199 tests pass.
  Both jobs retain Trackio metrics, compact selected/candidate states,
  telemetry, atomic recovery state and checksums; successful completion removes
  only validated recovery state. No rollout, locked report, validation, held,
  Gate 0, final target, or Writer authority is available during fitting.

### Rank-16 small-recipe completion and owner-corrected recovery

- Fit long-runs `gate0_support_rank16_t3_20260718_154106` and
  `gate0_support_rank16_t4_20260718_154106` completed main rc 0; screen long-run
  `gate0_support_rank16_screen_20260718_155426` also completed main rc 0. All fit,
  selected-state, screen, gallery, telemetry, and eight video checksums validate,
  and all GPUs were released. The result is task 3/4 base 2/8 and own 3/8,
  median gain 12.5pp, median query reduction 4.517%, and status
  `rank16_support_screen_failed`.
- The owner froze the correct interpretation before any final LoRA/Gate 0
  negative: rank 8/rank 16 tested only 12 demos, 750 steps, and a custom
  acquisition schedule. The old rank-16 config/result remain immutable, but its
  clause forbidding a mature positive-control recovery is superseded. No result
  is rerun or overwritten, no threshold is lowered, and Gate 0/Writer remain
  false.

### Mature LoRA positive-control implementation and prelaunch state

- Added the strict, hash-bound
  `configs/gate_zero_mature_lora_positive_control.toml` and one narrow contract
  module. Its frozen prelaunch SHA256 is
  `882db40dca9ced15cf2b567f9fa57bf2c36c66e64654eef55c067d6485b4b259`.
  The existing `ember.gate_zero_oracle_fit` remains the only trainer;
  the existing candidate/recovery artifact path now optionally saves scheduler
  state, and the existing target-support screen handles the mature decision.
  No second trainer, evaluator, launcher, bank, geometry, or future architecture
  path was added.
- The primary recipe is tasks 3/4, source support demos 0--39, query demos
  40--45, 20k steps, effective/micro batch 64, 37 default-like targets, rank
  32/alpha 16/dropout 0, 1,485,312 trainable parameters, Gaussian exact-zero
  physical initialization, AdamW `1e-4`, 1k warmup/cosine decay, deterministic
  90--100% crops, fixed step-20k selection, and fresh init 40--47 screening.
  The source support union is explicitly checked against `writer_spec`,
  `source_base_fit`, and `oracle_support`; those actions are legal only for the
  source-side oracle and remain hidden from Writer input. Query, locked report,
  validation, and held isolation remain fail-closed.
- The primary may be followed by at most one all-action-expert-linear rank-32
  compatibility recovery after a mechanics-validated failure. Passing the
  unchanged query and closed-loop thresholds seals the same LoRA contract for
  Writer, zero-init ordinary task-local RL, average/retrieval, and direct
  generator baselines. Failure cannot be promoted directly to an EMBER negative.
- Fresh verification passes 206/206 repository tests in 44.91 seconds, Python
  compilation, every shell launcher's syntax check, and diff whitespace checks.
  Tests cover strict source-role/data authority, fixed-final selection,
  deterministic augmentation, scheduler-bound atomic resume, mature Gate 0/
  Writer sealing arithmetic, and reuse of the two canonical launchers.
- The single permitted live smoke passed on physical GPU 4: exactly 37 targets,
  1,485,312 trainable parameters, exact-zero physical delta, one finite
  augmented 64-sample optimizer/scheduler step, and 17.35/17.47GiB peak Torch
  allocation/reservation. It evaluated no query or rollout outcome and left the
  GPU at 0MiB with no workers. A CPU-only augmentation microbenchmark measured
  0.04--0.05s/batch, disproving the initial suspicion that the per-row crop was
  a material bottleneck; no unneeded rewrite was made. Existing 750-step broad
  LoRA timings support the predeclared approximately five-hour fit estimate.
  The subsequent launch cadence is superseded by the staged race recovery below.

### Staged mature-fit race recovery

- Initial long-runs `gate0_mature_r32_t3_20260718_162854` and
  `gate0_mature_r32_t4_20260718_162854` entered startup immediately before the
  owner changed the execution cadence. Both received orderly SIGINT at about
  50 seconds and ended rc 130; GPU 4/5 returned to 0MiB. The outputs and logs
  remain intact. Each has a hash-valid step-0 candidate plus atomic optimizer,
  scheduler, RNG and trainable-state recovery; approximately 10 later volatile
  steps are correctly discarded.
- Added one optional `--stop-after-step` argument to the existing fitter and
  launcher. It accepts only a future predeclared non-final candidate, validates
  the candidate and recovery manifests, returns rc 0 with a resumable stage
  summary, and leaves the output legal for the next `--resume`. No second
  trainer/checkpoint format was added. Stage telemetry hashes are written to the
  long-run log rather than into the resumable output.
- The stage ladder contract SHA256 is
  `0db007a1e9403902b99e5b6f106f7556d087fe41c188742f94547c986bf6a9eb`.
  The first exact-resume segment stops normally at step 1000 and is expected to
  take about 12--18 minutes from measured 77--89 samples/s, below the 30-minute
  cap. It evaluates only the independent source query/drift candidate; formal
  init states 40--47 remain untouched. Continue to 2k only if both task query
  reductions are nonnegative and the median is at least 2%.
- Stage-1k long-runs `gate0_mature_r32_t3_stage1k_20260718_163856` and
  `gate0_mature_r32_t4_stage1k_20260718_163856` completed main rc 0 in 13:03.
  Task 3/4 query reductions are 7.96%/5.75% (median 6.86%), so the frozen
  continuation rule passes. Candidate, trainable-state, recovery and telemetry
  hashes validate; peak device memory is 18,585/18,545MiB and effective-window
  mean utilization is about 89.9%. GPU 4/5 returned to 0MiB. Resume the same
  outputs from step 1000 to the next stop at step 2000; do not open final
  closed-loop init states.
- Stage-2k long-runs `gate0_mature_r32_t3_stage2k_20260718_165425` and
  `gate0_mature_r32_t4_stage2k_20260718_165425` completed main rc 0 in
  12:59/12:44. Task 3/4 query reductions are 8.036%/4.504% (median 6.270%).
  The median change from 1k is -0.587pp, within the frozen 1pp limit; both task
  reductions remain positive, so the 2k-to-5k continuation rule passes. Task
  4's -1.251pp individual change is recorded rather than hidden or used to
  rewrite the aggregate rule. Candidate-manifest SHA256 values are
  `4f8b3536e0d028afad9aaa646d585d81829aa3ebe067495b5355f8171f8004e6` and
  `adb91eae7afe134fbf8446003e3c981d5475633f89d2c9584bb4e18a12c4b31a`;
  recovery-manifest SHA256 values are
  `1317ea1f7f90a3cf04ea96897a13f6b46f2ca9b9bd73891d9f629c4c3aec5d6b` and
  `0f189357f809e8d602adaff78865859168671cec39e1f00f8449b1a0cd247791`;
  telemetry SHA256 values are
  `a05400a7552bd9b77a96567239fd215ffc608db73e8d1021f5776aa998a15967` and
  `8629a7388a45980778640075551fc3e515645977cd50f96db556b20488441c95`.
  GPU 4/5 returned to 0MiB. Resume the same outputs from step 2000 to step 5000;
  expected wall time is about 38--45 minutes, with no final closed-loop access.
- Stage-5k long-runs `gate0_mature_r32_t3_stage5k_20260718_171212` and
  `gate0_mature_r32_t4_stage5k_20260718_171211` completed main rc 0 in
  37:32/37:45 on clean commit `7ed1282`. Task 3/4 query reductions are
  0.315%/-2.956% (median -1.321%); the 7.591pp median regression from 2k
  violates the frozen continuation rule, so neither output may resume to 10k
  or 20k. Candidate-manifest SHA256 values are
  `3868717b3c31c88565df3d0006c6eadfb7609cc5d29a990fd975a39e1983ea8a` and
  `7bbc34d47d358ba76cc94bc06693e8cfc7ac8e6692535e9efe2acd0f7cd7642d`;
  recovery-manifest values are
  `70d09c578e36679c9a79fccb8098ec9ee8dbeaa94c9007c00a1c45a9d7a3cd16` and
  `b8caebcac4b817c92d31e785debdfcb58d0c43beeec727737987ae5200f1c32e`;
  telemetry values are
  `9822b4041712bbfa94db1b7090efb3f717467e23105bd3e254601219132649e0` and
  `1ca56a87dfb2afd239743a665c756617eb287f1345bad694d9aff59324b86c37`.
  Peak memory was 18,585/18,545MiB and active-window utilization averaged
  91.79%/91.04%; GPU 4/5 returned to 0MiB with no worker residue. Final
  init states 40--47, validation and held surfaces remain untouched.
- A one-GPU structure-only probe then loaded the same frozen source checkpoint,
  enumerated 112 action-expert linear modules and the five existing projection
  targets, computed 7,027,200 rank-32 parameters, and released the GPU. Use this
  exact 117-target set for the already-authorized
  `all_action_expert_linear_r32_same_recipe` recovery. Before any fit outcome,
  freeze its parent failure hashes and a matching staged ladder; run one finite
  step/count/memory smoke, then tasks 3/4 to the 1k boundary only. No additional
  rank/support search or final rollout is authorized at this point.
- Added the strict recovery contract and reused the single canonical oracle
  fitter/launcher. The fit contract SHA256 is
  `82f5203ed86a25dac386bde68cb8a76efaba03c0f230fe2bd0249bb8d64fe15c`;
  the result-blind ladder SHA256 is
  `f3b66cff59135f52e81ab9ef387230381662fad6797e5c63557f791bd015739f`.
  Loader checks bind the primary config/ladder and both task-5k candidate,
  recovery and telemetry hashes. The same support/query rows, optimizer,
  augmentation, seed/noise and rollout identities are retained; only the exact
  target set and resulting parameter count change.
- The single live recovery smoke passed on physical GPU 4: exact target/count,
  exact-zero physical initialization, finite 64-sample loss/gradient, matched
  primary step-1 row digest, 63.4 samples/s, and 17.93/18.56GiB peak Torch
  allocation/reservation. It opened no query candidate or rollout surface and
  left GPU 4 at 0MiB with no worker residue. The task-3/task-4 1k stage may now
  launch on two independent GPUs under the frozen ladder.
- Recovery stage-1k long-runs
  `gate0_mature_all_linear_t3_stage1k_20260718_181417` and
  `gate0_mature_all_linear_t4_stage1k_20260718_181417` completed main rc 0 in
  13:35/13:21. Task 3/4 query reductions are 8.381%/1.276% (median 4.829%),
  satisfying the result-blind 1k-to-2k rule of median at least 2% and every task
  nonnegative. Task 4's weak response is logged but does not change that rule.
  Candidate/recovery artifacts, state hashes, telemetry hashes, primary-matched
  query/anchor digests and sample counts all validate. Peak memory is
  19,143/19,379MiB and memory-active utilization averages 89.39%/90.69%; GPU
  4/5 returned to 0MiB. The next authorized action is exact-resume of these same
  outputs to step 2k. No final rollout, validation, held, Gate 0, final target,
  or Writer authority has been opened.
- Recovery stage-2k long-runs
  `gate0_mature_all_linear_t3_stage2k_20260718_183319` and
  `gate0_mature_all_linear_t4_stage2k_20260718_183319` completed main rc 0 in
  13:04/13:08. Task 3/4 query reductions reverse to -3.149%/-12.093% (median
  -7.621%), 12.450pp below the 1k median. This fails every frozen 2k-to-5k
  condition, so both recovery outputs are permanently stopped at 2k and no
  formal closed-loop surface is opened. Candidate, trainable-state, recovery,
  query/anchor and telemetry evidence validates; support loss falls while
  drift rises, classifying the result as an optimization/generalization overrun
  rather than an implementation failure. Telemetry SHA256 values are
  `eb0171439c881e445e6e0e74cd242d7e9fd08163647956490579d979cb7f0897`
  and `1c8acee4e1ff8737ed8f995cd09efaded43315ae18aac74a8ed4ad1d332871ac`;
  peak memory is 19,099/19,379MiB and both GPUs returned to 0MiB. No additional
  target/rank variant is legal. Next preserve this failure packet, then run only
  the predeclared non-matched task-local action-expert capacity upper bound and
  surface the bounded Gate-recovery decision.
- Added the one final mature capacity-diagnostic config and its result-blind
  exact-resume ladder before any new fit outcome. Config/ladders SHA256 values
  are `8fd7f3a5fac0bbfef6fb7281e48b7ef9df7e5b95a74e9446d1e4c8e8ed72327d`
  and `69640a07e97915e9ac51ac31153d13f4df4e3154845afdb2a136def230f4bc98`.
  They bind both all-linear 2k failure packets, retain source/validation/held
  isolation, and explicitly prohibit Gate 0, Writer, or target-seal authority
  from the non-matched arm. The same canonical fitter now accepts the existing
  99,880,992-parameter action-expert/projection update state with the mature
  data, optimizer, augmentation and staged budget; no parallel implementation
  was created. Focused tests pass 15/15, the full repository suite passes
  214/214, Python compilation/shell syntax/diff checks pass, and a real
  prerequisite validation resolves the frozen step-10k source checkpoint. The
  canonical launch dry-run also resolves the exact upper-bound config and 1k
  stop. Next run one live finite-step count/identity/resource smoke, then launch
  only the 1k stage on two GPUs if it passes.
- Architecture guard is REVIEW with no hard violation: net active source/test
  growth is 272 lines, no module or entrypoint was added, the existing mature
  contract owner remains below 600 lines, and the one already-over-600 support
  dispatcher grew by a single declarative name. The capacity config is a
  bounded third use of the same fitter rather than a parallel path. After its
  Gate-recovery decision and evidence export, review whether the config mode
  still needs active rerun support before starting Writer implementation.
- A launch-preflight review caught that the shared dataset loader originally
  recognized only the LoRA mature-stage name for the 0--39 support / 40--45
  query split. A red-then-green regression extends that same fail-closed mature
  stage set to the non-matched capacity diagnostic; no split, row, dataset, or
  loader implementation changed. The focused file and full 214-test suite pass
  after the repair.
- The sole mature action-expert upper-bound smoke then passed on GPU 4 from
  clean commit `d7f77ba`: exact 99,880,992-parameter/155-tensor identity,
  mature-control-matched first-batch digest, finite loss 0.42178 and gradient
  norm 1.65898, 62.3 samples/s, and 17.86/18.93GiB peak Torch allocation/
  reservation. It published no scientific candidate, consumed no rollout,
  validation, or held surface, and left GPU 4 at 0MiB with no worker residue.
  Tasks 3/4 may now launch only to the 1k staged boundary on two independent
  GPUs under the frozen capacity ladder.
- Stage-1k long-runs `gate0_mature_action_expert_t3_stage1k_20260718_190809`
  and `gate0_mature_action_expert_t4_stage1k_20260718_190809` complete main rc 0
  in 13:19/13:06. Independent-query reductions are -5.221%/-19.788% (median
  -12.504%), so both the per-task nonnegative condition and median >=2% rule
  fail; the outputs are stopped permanently at 1k. Candidate/recovery states,
  query/anchor identities and telemetry hashes validate. Support loss falls
  while query and drift worsen, classifying this as a mechanics-valid
  task-local acquisition/generalization failure. Telemetry SHA256 values are
  `e8b5d4df1d0d67f684d5ce773f5677d421edb3a399fae12d0c705c82886f7da2`
  and `d5907ccb9be8f398ce5edce3f940a45d98763790175703a2d28d168dff31a977`;
  peak memory is 19,455/20,195MiB, GPUs return to 0MiB, and the retained output
  is 2.0GiB. No 2k resume, rollout, validation, held, Gate 0, Writer, or target
  seal is authorized. The next action is an owner Gate-recovery decision on a
  single lower-LR/dense-early acquisition recovery versus revising the current
  supervised Gate 0 evidence plan; no further target/rank or grid search runs.
- Source-query update-scale diagnostics
  `gate0_action_expert_update_scale_t3_20260718_193359` and task 4 complete rc 0
  in 83/81 seconds on GPUs 4/5 and release both devices. Scale 0.25 converts the
  saved update to +2.705%/+1.311% query reduction (median +2.008%); scale 0.5
  already makes task 4 negative. Scale-0/1 endpoint identity and all artifact
  checksums pass; only 48KB is retained, including exact probe source SHA256
  `f03738cc...bb7`. Trackio group `gate0_update_scale_diagnostic` provides the
  live/later curve. No training or closed-loop/validation/held surface was used.
- Added one strict optimization-only recovery contract and ladder at SHA256
  `30e2c575...a3c7c` / `6947a9a8...ebc2c`. They reuse the canonical fitter and
  preserve data, sampler/noise, augmentation, optimizer, seed, parameter
  identity and 20k maximum budget, changing only peak/decay LR by 0.25. The
  staged authority is 250 -> 500 -> 750 -> 1000 with exact resume and early
  stopping. Focused contract tests pass 18/18; real prerequisite loading and
  canonical launcher dry-run pass. The next launch is task 3/4 to step 250 on
  two GPUs; no LR grid, action-expert rollout, or LoRA run is yet authorized.

### Action-expert lower-LR ladder completed; matched LoRA ladder frozen

- Long-runs `gate0_action_expert_lr_recovery_t3_stage250_20260718_195048`
  through `...stage1000_20260718_200640`, with the corresponding task-4 runs,
  all complete main rc 0. Query reductions at step 1000 are +7.883%/+3.710%
  (median +5.797%), passing the frozen source-query criterion; step 750 was the
  observed peak (+8.771%/+5.177%, median +6.974%). Stop the non-matched
  diagnostic at 1000 rather than burning to 2k. Final state/recovery hashes are
  bound in the next contract, both GPUs are 0MiB after exit, and closed-loop,
  validation, held, Gate 0, and Writer surfaces remain unopened.
- Added the dedicated strict config validator
  `mature_lora_lr_contract.py`; it only validates a new mode of the existing
  oracle fitter. Config/ladders SHA256 values are
  `693cd61457ec5ec0aafb1c72837899c58f44f2c90e812cb14d115385393fafca`
  and `436bae48c5c9f754346b18bd424378d70eee5900eb0add96f6a4ea99104817d3`.
  The contract binds all upstream hashes and both action-expert step-1000
  candidate/recovery/telemetry packets, keeps 37 targets/rank32/1,485,312
  parameters, and reuses exactly the validated lower-LR schedule.
- Focused tests pass 20/20 and the full repository suite passes 219/219 in
  44.38s. Real prerequisite validation resolves source checkpoint step 10k;
  the sole launcher dry-run resolves the exact config/variant and
  `--stop-after-step 250`. Next, commit/push this clean pre-outcome contract,
  perform live GPU/storage preflight, and launch task 3/4 to step 250 on two
  independent GPUs. Continue the same outputs only if the result-blind ladder
  passes. Do not open the formal closed-loop surface until a separate
  headroom-safe source-only contract is frozen.
- Step-250 long-runs `gate0_mature_lora_lr_recovery_t3_stage250_20260718_203014`
  and `...t4...203015` complete main rc 0 on clean commit `a392634` in
  3:54/3:53. Task 3/4 query reductions are +0.892%/+1.038% (median +0.965%),
  satisfying the frozen nonnegative continuation rule; drift is
  0.00167/0.000925. Candidate and recovery validators pass, current recovery
  pointers resolve to step 250, peak memory is 18,485/18,545MiB, active-window
  utilization is 89.44%/88.89%, GPU 4/5 return to 0MiB, and retained output is
  58MiB. Exact-resume the same output to step 500 next; no formal rollout,
  validation, held, Gate 0, or Writer authority was opened.
- Step-500 long-runs `gate0_mature_lora_lr_recovery_t3_stage500_20260718_203639`
  and its task-4 peer complete main rc 0 in 3:45/3:41 from the step-250 atomic
  states. Query reductions improve to +3.388%/+2.611% (median +2.999%), above
  the frozen 0.5%-median/nonnegative continuation rule; drift is
  0.00572/0.00495. Candidate/recovery validation passes, peak memory is
  18,585/18,545MiB, active-window utilization is 88.84%/90.14%, current
  recovery pointers resolve to step 500, retained output is 69MiB, and both
  GPUs return to 0MiB. Exact-resume to step 750 next; no rollout or non-source
  surface has been accessed.
- Step-750 long-runs `gate0_mature_lora_lr_recovery_t3_stage750_20260718_204212`
  and its task-4 peer complete main rc 0 in 3:43/3:44. Query reductions reach
  +4.925%/+3.728% (median +4.326%), passing the frozen 1%-median/nonnegative
  rule; drift is 0.00768/0.00684. Candidate/recovery validators pass, peak
  memory is 18,585/18,545MiB, active utilization is 89.74%/88.96%, current
  recovery pointers resolve to 750, and both GPUs are released. Exact-resume
  to the final authorized step-1000 query boundary next; no closed loop or
  validation/held access occurred.
- Step-1000 long-runs `gate0_mature_lora_lr_recovery_t3_stage1000_20260718_204726`
  and its task-4 peer complete main rc 0 in 3:46/3:45 on clean `57c44c3`.
  Task 3/4 query reductions are +5.798%/+4.236% (median +5.017%), passing the
  frozen 2%-median/nonnegative success rule; drift is 0.01056/0.00858.
  Candidate/recovery validation passes, peak memory is 18,585/18,545MiB,
  active utilization is 89.06%/89.07%, current recovery pointers resolve to
  1000, retained output is 92MiB, and GPU 4/5 return to 0MiB. The ladder is
  complete and must not resume to 2k. Next freeze a headroom-safe source-only
  closed-loop contract; Gate 0, final Writer support, and Writer remain false.

### Owner approved the headroom-safe matched-LoRA Gate before outcomes

- Active `configs/gate_zero_mature_lora_headroom_screen.toml` SHA256 is
  `1f92f80ddcc63be7c6a3ef3da1fe63f9870df27a0537ec02c3429bab71440a52`
  before any new LoRA closed-loop result. It binds both task step-1000
  candidate/state/recovery/telemetry hashes, fixes source init states 40--47,
  seeds 5800--5807, and predeclares the maintenance-task, improvement-task,
  aggregate, query and drift rules plus the distinct no-headroom recovery.
- The existing target-support grant/runtime/launcher remains the sole path.
  It now loads grant-bound staged candidates directly, while the same strict
  freeze validates upstream artifacts and source/validation/held authority.
  A red-then-green launcher regression removed the obsolete assumption that
  every valid fit must expose `selected/`; stage-specific candidate identity
  remains fail-closed in the freeze validator. Direct state-load smoke resolves
  74 tensors and exactly 1,485,312 parameters for each task with the frozen
  state hashes.
- Before the supervisory pause, focused screening tests passed 38/38 and the
  full repository suite passed 226/226 after the mechanical seed regression;
  Python compilation, shell syntax, shellcheck, diff checks and the canonical
  two-GPU dry-run passed. The pending-owner fail-closed guard had its focused
  file passing 8/8 plus Python compilation and diff checks.
  Architecture guard is REVIEW with no hard violation: the new contract owner
  is bounded and reuses the existing evaluator rather than adding a second
  rollout path. Gate 0 and Writer remain false until the owner-approved screen
  is complete and mechanically validated.
- The first launch `gate0_mature_lora_headroom_screen_20260718_211501` stopped
  rc 1 before any episode on the evaluator's warm-up-seed adjacency check.
  GPUs 4/5 released to 0MiB. Failure-packet SHA256 values are
  `c6fc083a...463c`/`7ba07798...91e`, with telemetry `d2b56c28...479`.
  A red-then-green regression fixes only the last warm-up batch start from
  5760 to 5792 so it ends immediately before unchanged report seed 5800. The
  old contract hash and clean commit `5681819` remain provenance; no rollout
  result, threshold, task/init identity, LoRA state, validation, held, or
  locked-report surface was consumed or changed.
- Supervisory review correctly identified that the ceiling-aware rule is a
  scientific Gate redesign, not part of the mechanical seed fix. Commit
  `aa16f20` had already been pushed before that instruction arrived, but no
  relaunch occurred. The owner then explicitly approved Proposal A on
  2026-07-19 Asia/Singapore and resumed the unchanged full EMBER objective.
  Active config changes only authorization state to
  `owner_decision_required=false` and `screening_rollout_authorized=true`,
  preserving pending SHA256 `8c7ae12b...075d5c` and commit `108ce65` as
  provenance. Focused screening tests pass 41/41; Python compilation, shell
  syntax, diff checks and the canonical two-GPU dry-run pass. A disposable
  real grant validates both staged candidates and its own checksum, while
  keeping Gate 0/Writer and validation/held access false. Next commit/push,
  then launch only the two-GPU short screen.
- Launch review caught that the prior runtime evaluated base and LoRA arms
  before classifying missing headroom. The owner-approved recovery forbids even
  reading a LoRA outcome in that case, so the same canonical runtime now uses
  one distributed base-only barrier: task-4 base failures >=2 opens the LoRA
  stage; otherwise both LoRA arms remain unopened and the published
  `headroom_absent` packet contains base arms only. Red/green tests cover both
  branches and reject any full-arm result whose base lacks headroom. No Gate
  number, task, seed, init state, evaluator, LoRA state, or data surface changed.
  Architecture guard is REVIEW with no hard violation; the existing runtime
  and contract owner remain the single canonical path.

### Proposal A completed, failed Gate 0, and entered same-task bounded recovery

- Long-run `gate0_mature_lora_headroom_owner_a_20260719_025534` completed
  main rc 0 on clean `dbcb729`. The base-first barrier measured task 3/4 at
  3/8 and 3/8, then authorized the matched step-1000 LoRA arms at 2/8 and 4/8.
  Paired net wins are -1/+1 and aggregate zero, so the frozen A Gate fails even
  though both query-reduction and drift safeguards pass. Gate 0, Writer, and
  final support sealing remain false; the ladder stays stopped at step 1000.
- Both output and freeze checksum manifests pass; result SHA256 is
  `84116faa...c98f`, four bounded videos/gallery are present, the latest link is
  canonical, output is 636KiB plus a 12KiB freeze, wall time is 171.95s, and
  GPU 4/5 returned to 0MiB. Unrelated MemLLM jobs on GPU 0--3 were untouched.
- An owner instruction to replace task 3 arrived after the run because of a
  remote-turn race, then was explicitly withdrawn once the fresh 3/8 base
  result was known. No Option-B selection experiment launched. Its uncommitted
  config/launcher/module/test are recoverable as local stash
  `201d097ec542b76014eb885dbd04ffb166221476`, but are absent from the active
  worktree and must not be committed or run.
- Added a narrow post-failure candidate-step diagnostic contract, runtime,
  launcher, and red-then-green tests. It keeps task 3/4, binds A and the exact
  step-500/750 candidate manifests/states, reuses the A source development
  slice without rerunning base/step1000, and evaluates only 32 new episodes on
  two GPUs. Its deterministic rule preserves the original 15pp median and two
  positive tasks plus per-task 2% query and 0.02 drift safeguards. The result
  can only authorize a hash-bound fresh-seed recovery Gate, never Gate 0 or
  Writer directly; step 2k remains forbidden. Contract SHA256 is
  `4445664f652744829e219fd483cde7820cfde1d588157d730e4322811e916a46`;
  focused tests pass 36/36, the full repository suite passes 235/235, real
  loading validates all four 74-tensor candidate states at 1,485,312 trainable
  parameters, and the canonical two-GPU dry-run passes.
- Before any candidate-step outcome, the owner separated two later reward
  stages. After supervised Writer cold start, a dedicated source-only
  Writer-only RL phase freezes the base, does not optimize generated task-local
  LoRA in place, and updates only Writer from rollout reward. Ordinary
  task-local LoRA RL remains a later, separate phase that freezes base/Writer
  and updates only the same LoRA from Writer or matched-zero initialization.
  A subsequent adaptation-aware reward/meta outer stage may differentiate
  through the inner LoRA learner, but must not be conflated with Writer-only RL.
- The owner also clarified before any new recovery outcome that Gate 0 is not
  SFT-only. The candidate-step diagnostic remains first and unchanged. If it
  does not show credible closed-loop utility, the next bounded recovery is a
  pre-outcome-frozen, same-task four-arm source comparison: base; supervised
  LoRA; zero-init LoRA plus ordinary task-local RL; and supervised-init LoRA
  plus identical RL. This phase has no Writer, matches LoRA/reward/estimator/
  optimizer/seed/init/interaction/compute, starts with a 10--30 minute early
  check, and limits resumable segments to one to two hours. No task/seed/
  threshold evasion, validation/held/locked access, or arbitrary long training
  is authorized.
- Candidate-step long-run
  `gate0_candidate_step_diagnostic_20260719T035642Z_20260719_035642` completed
  main rc 0 on clean `19e5ea2` in 2:57.68. Step 500 and 750 both give task-3
  2/8 and task-4 3/8 versus the reused frozen bases 3/8 and 3/8, so neither has
  a positive task, both have -6.25pp median gain and paired net -1, and the
  frozen decision selects no state. Query/drift safeguards pass, but no fresh
  recovery Gate, Gate 0, target seal, or Writer authorization is issued.
  Result SHA256 is `aae6e19f...b11cf`; result/eval/gallery/video/telemetry
  checksums and JSON parse checks pass, four videos and `index.html` are
  viewable locally and through Trackio run `step500_750_20260719T035642Z`, the
  retained packet is 600KiB, and GPU 4/5 released to 0MiB. The supervised
  trajectory remains stopped; next freeze the already authorized same-task
  four-arm matched ordinary task-local RL recovery before any new outcome.

## 2026-07-19 signed flow-ratio Gate-0 recovery prelaunch

- The prior AWR-style 16-episode four-arm run was validated and closed at
  result SHA256
  `aab151ea503dbada6eaf3a2242301562a47052e1399ec10986c2279425c13b57`:
  rc 0, complete checksums, finite updates, four videos/gallery/Trackio, but
  zero paired development change in all arms. Only the step-16 recovery state
  is loadable; step 8 retains metrics but no model checkpoint. Stage 32 and a
  fresh Gate remain closed.
- Before any further environment outcome, the single active trainer was
  changed from positive-only AWR weighting to a bounded signed conditional-
  flow-loss-ratio objective. The pre-outcome contract SHA256 is
  `d322339eb417536a8b96b124b3c8d6324c4b25b95e89f4a3cffb5d6cadce200c`;
  it binds the AWR result and official FPO-control commit
  `b80112be1e8362263c4cd176e7aef21a275ff1c6`, while declaring that this is not
  full FPO++. No second trainer or long-run path was added, and 32 episodes are
  removed from the active recovery rather than silently reopened.
- Red-then-green tests cover signed normalized advantages, flat-reward
  fail-closed behavior, success/failure gradient direction, frozen hashes, the
  single 16-episode node, and canonical launcher topology. Focused tests pass
  30/30 and the full non-DDP suite passes 247/247; Python compilation, shell
  syntax, dry-run authority paths, diff checks, and the architecture guard
  pass with REVIEW/no hard violation. A no-environment real-model smoke with no
  optimizer step is the final prelaunch requirement; no new source rollout,
  validation, held, locked-report, or Writer outcome has been consumed yet.
- The active plan also records the owner-mandated independent Writer-only RL
  stage after supervised Writer cold start. That stage updates Writer only and
  cannot be conflated with Gate-0/task-local LoRA RL, which updates LoRA only,
  or with later adaptation-aware reward/meta outer learning.
- Long-run `gate0_signed_flow_ratio_real_model_smoke_20260719_060242`
  completed rc 0 from clean `2a72bd4` in 30.07 seconds on GPU 6. The real model
  accepted `[64,50,7]` source actions with `[64,50,32]` flow noise, repeated
  old/current losses exactly, and produced finite nonzero gradient norm 0.0803
  without optimizer or environment interaction. Checksums pass; result SHA256
  is `b1e75b43...b4687`, peak reserved CUDA memory is 17,948MiB, and GPU 6 is
  released. The temporary source is durably copied into the mechanics packet;
  remove the workspace copy and proceed to the sole frozen four-card stage-16
  run after a fresh GPU/storage preflight.
- Long-run `gate0_signed_flow_ratio_ep16_20260719_060519` completed main rc 0
  from clean `30d9e22` in 4:52.99. All four arms reached loadable atomic step
  16, output checksums and recovery validators pass, mechanics/nonfinite guards
  pass, four videos plus `index.html` and Trackio are retained, output is
  93MiB, and GPUs 4--7 returned to 0MiB. Result SHA256 is
  `73d681ca...8703`.
- The frozen source-development result is negative: supervised task3/4 remain
  2/8 and 4/8; zero task3 falls 3/8 to 2/8 and zero task4 remains 3/8. Status is
  `task_local_rl_early_check_not_supported`; Gate 0, fresh Gate, Writer, stage
  32, validation, held, and locked-report access remain unauthorized.
- A no-GPU factor/physical-delta audit compared both retained RL estimators to
  their exact zero/supervised initialization states. Signed physical updates
  are 0.067--0.072 L2 versus AWR 0.096--0.120; supervised-relative moves are
  about 10% versus 17%. Since the larger AWR move also failed behaviorally,
  do not simply multiply optimizer steps. Next implement only the lowest-cost
  source diagnostic already required by the capacity audit: evaluate the
  immutable lower-LR action-expert step-1000 state on the same frozen task3/4
  development identities as a non-matched upper bound. Its contract and hashes
  must be frozen before any rollout, and it cannot pass Gate 0 or authorize
  Writer alone.
- Added the selected one-time action-expert capacity closed-loop controller,
  config, launcher, and red-then-green tests. It reuses the existing policy
  loader, variant configuration, state restoration and `_closed_loop_metrics`
  evaluator; it adds no policy or trainer implementation. Contract SHA256 is
  `e313e437fe57f20d2cd390fbede0c89432bb89f1d40dd7d37bcf8156e1af9f3a`.
- Focused tests pass 22/22 with active-contract checks, and the full non-DDP
  suite passes 253/253. Python compilation, shell syntax, two-GPU dry-run,
  diff checks, and real no-GPU authority loading pass. The real load validates
  both 155-tensor/99,880,992-parameter step-1000 states and exact query metrics.
  Architecture guard is REVIEW with no hard violation: this is the single
  selected capacity diagnostic and all rollout behavior delegates to the
  canonical evaluator. No capacity outcome exists yet. Next commit/push
  cleanly, then launch exactly two source arms if live GPU/storage preflight
  passes.
- Long-run `gate0_action_expert_capacity_closed_loop_20260719_062725`
  subsequently completed main rc 0 from clean `7e5f905` in 1:37.99. All
  result/eval/gallery/video/telemetry checksums pass; the two videos and local
  gallery are retained, Trackio run is `capacity_20260719T062725Z`, and GPUs
  4/5 released. Result SHA256 is `9a91fbb8...170ad`.
- The non-matched 99,880,992-parameter partial action-expert state gives task 3
  3/8 with the exact base vector and task 4 3/8 with one paired loss and one
  paired win. Both net gains and the median gain are zero; frozen status is
  `nonmatched_action_expert_capacity_behavioral_signal_absent`. Gate 0,
  Writer, fresh Gate, target sealing, validation, held, and locked-report
  access remain false.
- This completes the planned bounded target/rank/capacity discriminator. Do
  not add supervised steps, rank sweeps, or target expansions: even the much
  wider query-positive state failed to convert into stable closed-loop gain.
  The next scientific contract must instead isolate query/behavior alignment
  or use a pre-outcome-frozen task-local RL estimator with genuine temporal
  credit, keeping tasks 3/4 and the matched base/supervised/zero-init-RL/
  supervised-init-RL comparison. The already documented later stages remain
  separate: supervised Writer cold start, Writer-only RL updating Writer only,
  ordinary LoRA-only task-local RL, then adaptation-aware source meta-outer
  learning.
- Froze the next no-environment discriminator before outcomes at config SHA256
  `a85de2e89ae0e5477e931cf887b79b6b756aa0c090bf0903353c7bf475262c3d`.
  It reuses the canonical fixed-query evaluator and immutable candidate states
  to compare generated action-chunk MSE for base, supervised LoRA, and partial
  action expert on 48 source-query anchors per task, with row/episode/action-
  dimension/time-partition breakdowns and zero new simulator episodes.
- Red-then-green tests cover aggregation identity, shape failure, all three
  decision branches, frozen source-only authority, and the two-rank dry-run.
  Focused tests pass 15/15; shell syntax, Python compilation, dry-run, and diff
  checks pass. Architecture guard is REVIEW with no hard violation. The
  one-time 506-line controller owns only authority validation, two-task
  orchestration, and compact output; it adds no trainer/evaluator path and will
  be retired after its packet and the selected recovery contract are frozen.
  The 50x7 fixed-anchor action-error summary remains because it is a current
  second use for later direct-Writer query supervision/diagnostics.
- Long-run `gate0_query_action_alignment_20260719_065332` completed main rc 0
  from clean `9568921` in 43.77 seconds. Both task ranks finished, checksums
  pass, output is 56KiB, Trackio run is `alignment_20260719T065332Z`, and
  GPUs 4/5 returned to 0MiB. Result SHA256 is `95f8adfc...c1c6`.
- Status is `fixed_flow_query_surrogate_misaligned`: task-3/4 supervised-LoRA
  fixed-flow loss improves 5.798%/4.236% while generated-action MSE worsens
  3.114%/2.479%; action-expert loss improves 7.883%/3.710% while action MSE
  worsens 0.886%/0.304%. Gate 0, Writer, targets, validation, held, locked
  report, and new closed-loop access remain false.
- Next freeze one small multi-inference-noise replication on the same 48
  source-query anchors and immutable states. It must consume zero environment
  episodes and is the last variance check before either a differentiable full-
  sampler action-loss acquisition smoke (stable mismatch) or a multi-sample
  flow estimator repair (unstable signs). Do not add supervised steps, target/
  rank variants, or task-local RL budget before this discriminator.
- The four-draw replication contract is frozen at SHA256
  `d436e17f2a5b91b8cdf22806e3967fc1f0f170590ba8a96692c610c7ef42212f`.
  It reuses the existing two-task audit entrypoint, binds the single-noise
  result, and adds only seeds `[2026071835, 2026071935, 2026072035,
  2026072135]`. Robust mismatch requires worse mean action MSE for all four
  candidate-task pairs and at least three worsening draws per pair; otherwise
  status is sampling variance. No rollout, new model state, or threshold is
  opened. Focused tests pass 17/17; compile, shell syntax, dry-run, and diff
  checks pass. Architecture guard is REVIEW/no hard violation; the 626-line
  one-time controller remains under its documented retirement trigger.
- Long-run `gate0_query_action_alignment_robustness_20260719_070642` completed
  main rc 0 from clean `ccb2934` in 52.31 seconds. Checksums pass, output is
  144KiB, Trackio run is `robust_20260719T070642Z`, and GPU4/5 released. Result
  SHA256 is `c1fc3ab4...ae4b`; no environment episode was opened.
- Supervised-LoRA action MSE worsens on all four noise draws for each task and
  by 1.901%/3.062% in mean task-3/4 error. Partial action expert improves on
  three draws per task and by 1.680%/1.428% in the mean despite its earlier
  zero closed-loop gain. Gate 0/Writer remain false. This confirms a robust
  LoRA acquisition-surrogate failure plus a distinct teacher-forced-to-closed-
  loop/temporal-credit gap.
- Next freeze and run only a one-GPU, no-optimizer, no-environment mechanics
  smoke for differentiating normalized action-chunk MSE through the pinned
  full 10-step sampler into the same rank-32 LoRA. It must record finite loss/
  gradient, exact source rows, wall time, and peak memory with at least 10GiB
  free. Only a passing smoke may authorize a short resumable action-aligned
  acquisition ladder; no simulator/RL budget is opened yet.
- Froze that smoke at config SHA256
  `2dab3cd4399cd93daa26725b3c7ea50d07e555ee70f027ac53c622ac3bc10f25`
  and script SHA256 `1e5a8542...c5468`. Static compilation passes. It uses one
  GPU, task-3 demo-0 frames 0/1, batch 2, the full 10-step sampler, unchanged
  rank-32 LoRA, no optimizer step, no simulator, and a 10GiB-headroom memory
  stop. A pass authorizes only a separately predeclared short action-aligned
  acquisition ladder.
- Long-run `gate0_differentiable_action_loss_smoke_20260719_071538` produced a
  valid scientific packet from clean `fe2d270`: full 10-step `[2,50,7]`
  sampling, loss 0.21559, 74 finite LoRA gradients with norm 0.52946, unchanged
  parameters, 2,796MiB peak reserved, zero optimizer/environment steps, and
  result SHA256 `b4e6fcef...4fe4`. All output/source/config/telemetry checksums
  pass and GPU4 released. The exact temporary script was copied into the packet
  and removed from workspace.
- The outer ad-hoc telemetry command is retained as a wrapper failure packet:
  longrun rc 1 conflicts with the inner GNU-time exit 0 and completed checksum
  cleanup. One no-GPU shell-skeleton verification exits 0; per the engineering
  stop rule, do not investigate or rerun further because checkpoint/data/
  behavior/Gate/held integrity is unaffected. Use the canonical oracle-fit
  launcher for the newly authorized predeclared action-aligned ladder.
- Added the result-blind action-aligned acquisition contract at SHA256
  `3d5b54be47c20bf29e356395f43ad2c9d43834b90eded994e68b141be0902246`.
  It freezes the unchanged task3/task4 37-target rank-32 LoRA and batch-64
  support/query surface, a full-sampler generated-action-MSE objective, a
  5/200-step warmup/decay horizon with unchanged peak/decay LR magnitudes, four
  query inference-noise seeds, and candidates `0/1/5/10/25/50/100/200`.
  Step 1 is memory/recovery only; later continuation requires two-task action-
  MSE evidence under the frozen 0%/1% ladder, and closed-loop access requires
  at least 2% per-task action-MSE improvement plus drift no greater than 0.02.
- Extended the single canonical oracle fitter and evaluator rather than adding
  a trainer: objective dispatch, deterministic row/step/slot sampler noise,
  multi-noise action-query metrics, action-MSE selection, Trackio logging, and
  existing atomic resume are one path. Retired the one-time query/action audit
  controller, launcher, two configs, and experiment-only tests after their
  packets and next contract were frozen; Git history remains the rollback.
- Red/green focused tests pass 6/6; the full suite passes 263/263. Python
  compilation, shell dry-run, prerequisite/hash validation, and diff checks
  pass. Architecture guard is REVIEW with no hard violation and net active
  source shrinkage. No new GPU training, simulator, validation, held, locked-
  report, Gate, or Writer outcome was accessed while freezing this contract.
- Committed/pushed the clean pre-outcome implementation as `16b6e14`. Ran
  task3/task4 on GPU4/5 through separate durable exact-resume segments at steps
  1, 5, and 10; all six longruns completed rc 0 and released both GPUs. Output
  root is `gate_zero/action_aligned_lora_acquisition/fit/action_20260719T074144Z`.
  Step-1 telemetry peaks at 50,697/50,657MiB and the total output remains small;
  no additional GPU was used for duplicate work.
- Step-1 action-MSE reductions are +0.074%/+0.109%; step-5 reductions are
  +0.210%/+0.219%; step-10 reductions are +0.881%/+0.928%. Step-10 candidate
  SHA256 values are `c9b0d940...e5571`/`292781f4...acbb`, and current recovery
  manifest SHA256 values are `6fdf67de...b4a49`/`4412cba7...f265c`. Candidate,
  optimizer, scheduler, RNG, and telemetry hashes validate; GPUs 4/5 return to
  0MiB.
- Enforced the frozen stop: both tasks miss the 1% step-10-to-25 action-MSE
  floor, despite positive signs on all four inference-noise seeds. Did not run
  step 25, round up the metric, lower the threshold, or access source closed-
  loop/validation/held/locked surfaces. Gate 0/Writer remain false. Next freeze
  the smallest ordinary task-local LoRA RL recovery that adds genuine temporal
  credit and preserves the matched zero-init versus supervised-init comparison.

## 2026-07-19 temporal-credit Gate-0 recovery frozen before outcomes

- Rechecked the active unbudgeted full-EMBER Goal and clean `c370df1` base.
  The already documented owner contract remains active: Writer supervised cold
  start, a separate Writer-only RL stage that updates Writer only, later
  ordinary LoRA-only RL, and adaptation-aware source meta-outer learning are
  distinct experiments. Gate 0 is not SFT-only and currently contains no
  Writer or shared-state update.
- Replaced the canonical active task-local RL estimator in the existing
  `ember.gate_zero_task_local_rl` entrypoint; no second trainer or launcher was
  added. Historical AWR/signed-ratio configs and immutable result packets
  remain provenance, but their executable objective code and experiment-only
  tests were retired. Runtime/contract source shrank while temporal mechanics
  moved into one cohesive owner module.
- Froze `configs/gate_zero_task_local_rl_temporal_credit.toml` at SHA256
  `0cfd1c74ced6b5cdc0e792d1af48555df6f2346527377cdc753ba46fc35955d2`.
  It binds the negative AWR (`aab151ea...b57`), signed-ratio
  (`73d681ca...8703`), and stopped action-aligned step-10 manifests; task3/4,
  supervised step-1000 versus physical zero initialization, LoRA structure,
  source seeds/init states, evaluator, exploration, drift and Gate thresholds
  are unchanged.
- Added ordered action-chunk replay, task-local 512/256 critic, masked GAE,
  eight-sample matched flow ratios, PPO clipping, and separate actor/critic
  AdamW. Atomic recovery schema 3 now includes critic and critic optimizer;
  old schemas remain readable. Stage 8 must start fresh; stage 16 must resume
  the exact stage-8 actor+critic state. A passing stage-8 candidate stops;
  otherwise only healthy mechanics may continue once to 16.
- Red/green focused tests pass 15/15; the full suite passes 264/264. Python
  compilation, shell syntax, contract/hash dry-run, and diff checks pass. No
  simulator, validation, held, locked-report, Gate, or Writer outcome was
  consumed while freezing this contract. One real-model no-environment/no-step
  smoke remains before Git pre-outcome delivery and any source rollout.
- Committed and pushed the outcome-free temporal-credit contract and canonical
  implementation as `3590f70`. Long-run
  `gate0_temporal_credit_real_model_smoke_20260719_082844` then failed closed
  before gradients because the configured empty third camera was included in
  the otherwise two-camera critic vector. It used zero optimizer steps and
  zero environment episodes; GPUs released. The immutable run remains the
  mechanical failure packet.
- Added one narrow production regression: only declared trailing all-false
  empty-camera slots are removed, while an observation-bearing empty slot is
  rejected. Focused temporal tests pass 16/16; committed/pushed as `8237bed`.
  The single allowed retry,
  `gate0_temporal_credit_real_model_smoke_retry_20260719_083159`, completed rc 0
  in 35.70s. It verified 64/64 unique source rows, `[64,1953]` critic features,
  `[64,8]` matched flow losses, finite nonzero actor/critic gradients, unchanged
  LoRA state, zero optimizer/environment steps, and only 5,268MiB peak reserved
  memory. GPU4 returned to 0MiB. This authorizes only the already frozen stage-8
  source run and does not change Gate-0/Writer status.
- Launched the canonical four-arm stage 8 from clean `86b3e40` on GPUs 4--7.
  Long-run `gate0_temporal_credit_stage8_20260719_083749` completed rc 0 in
  4m46s. Fifteen JSON files and 20 candidate/recovery payloads validated; the
  stage packet SHA256 is `30629d47...f21f`. Its fixed decision was
  `task_local_rl_temporal_credit_continue_to_16`, so no configuration or task
  changed before exact resume.
- Long-run `gate0_temporal_credit_stage16_20260719_084442` exact-resumed the
  same four actor/critic/optimizer/RNG states and completed rc 0 in 5m01s.
  Terminal result SHA256 is `e13456343564880e6ef02d48119636774e0a06b783e6f4b0218692f104afa14c`;
  all `checksums.sha256` entries pass, all eight bounded videos decode, output
  is 169MiB, and GPUs 4--7 returned to 0MiB. The local gallery is
  `gate_zero/task_local_lora_rl_recovery/temporal_credit_ep8_20260719T083749Z/index.html`;
  Trackio project `EMBER_gate0` contains all eight stage runs.
- Recorded the terminal negative without changing the Gate: zero-init task3/4
  is `3→2,3→3`; supervised-init task3/4 is `2→2,4→4`; status is
  `task_local_rl_early_check_not_supported`, Gate 0 and Writer remain false,
  and validation/held numeric access remains zero. Stage 24 is forbidden for
  this trajectory.
- Completed one artifact-only physical-update audit. The RL state is not
  mechanically inert: zero-init operator norms reach 0.0701/0.0743 and the
  supervised-init operator moves 11.3%/11.7% from its SFT value. Official-code
  reconciliation identifies critic-only first-iteration warmup and far larger
  collection scale as the cheapest primary-source-supported gaps. Freeze any
  corrected bounded recovery as a new contract before new outcomes; do not
  append interaction to the failed trajectory.

## 2026-07-19 critic-warmup recovery frozen before outcomes

- Reconfirmed the active unbudgeted full-EMBER Goal. The fixed task3/task4
  Gate-0 surface and the completed temporal-credit negative remain authoritative;
  no replacement-task selection, validation, held, locked-report, Writer, or
  new closed-loop outcome was opened during this implementation.
- Added the result-blind compatibility contract
  `configs/gate_zero_task_local_rl_critic_warmup.toml`, SHA256
  `51fc9a009d0fa93476ba47a22d86e95a5d89f32182057843c3129e4147725a8a`.
  It binds the immutable episode-16 negative, starts a new trajectory, runs one
  critic-only round, sets GAE lambda to `0.99`, removes external Gaussian action
  noise, and caps exact-resume decisions at episodes `8/16/24/32` under the
  predeclared trend rule. It preserves the exact four matched Gate-0 arms and
  all source/held scientific boundaries.
- Kept one canonical trainer and launcher. The first round skips all actor
  forward/backward/optimizer work, verifies exact LoRA identity, and records
  actor/critic optimizer counts separately; later rounds reuse the same PPO
  actor path. Exact resume now requires the immediately preceding atomic node,
  and only contract-declared nonterminal statuses can continue. The retired
  external-noise processor was removed rather than retained as a second path.
- Focused recovery tests pass 16/16; the full suite passes 265/265. Python
  compilation, shell syntax, config/hash validation, canonical four-GPU
  launcher dry-run, and diff whitespace checks pass. The dry-run binds the
  new temporal predecessor packet and retains Trackio plus bounded video
  gallery output. Next, after clean Git delivery, run one GPU real-model
  no-environment/no-actor-optimizer smoke for critic update and exact actor identity;
  only that mechanics result may open the staged source rollout.
- The active research sequence remains explicit: supervised Writer cold start;
  independent Writer-only RL that updates Writer only; ordinary task-local LoRA
  RL that freezes Writer/base and updates LoRA only; then adaptation-aware
  source meta-outer learning. Gate 0 may establish a useful LoRA update through
  supervised training or matched ordinary RL, but must report which mechanism
  supplied the evidence and cannot convert RL utility into a supervised
  zero-interaction claim.
- Committed and pushed the outcome-free contract/implementation as `faf1564`.
  Long-run `gate0_critic_warmup_real_model_smoke_20260719_092504` then completed
  rc 0 on GPU4 in 28.95 seconds. It used 64/64 unique source rows and zero
  environment episodes; actor state remained exact, actor optimizer state and
  updates remained zero, while the critic made 40 finite updates with minimum
  gradient norm 0.835. Peak allocated/reserved memory was 2,689/4,128MiB.
  Result SHA256 is `91db6430...3c07`; result/source/config checksums pass, the
  packet is only 32KiB, and GPU4 returned to 0MiB. The first checksum command
  was invoked from the repository rather than the packet directory and could
  not resolve its relative paths; immediate verification from the correct
  directory passed all entries. This operator cwd residual did not alter the
  packet. The frozen four-GPU episode-8 source stage is now mechanically
  authorized; it remains a development decision, not Gate-0 evidence by itself.
- Ran the clean four-arm trajectory through its evidence-gated ladder. Stage 8
  long-run `gate0_critic_warmup_stage8_20260719_092947` completed rc 0 in 3m24s,
  reproduced all initial vectors exactly, recorded zero actor optimizer state,
  and returned the frozen continue-to-16 status. Stage 16
  `gate0_critic_warmup_stage16_20260719_093614` exact-resumed all four
  actor/critic/optimizer/RNG states, completed rc 0 in 5m02s, and produced
  zero/supervised paired gains `[-1,0]`/`[0,0]`; its fixed contract continued
  once to 24.
- Stage 24 `gate0_critic_warmup_stage24_20260719_094255` exact-resumed, completed
  rc 0 in 5m16s, and terminated with
  `task_local_rl_early_check_not_supported`. Final zero-init paired gains are
  `[0,0]`; supervised-init gains are `[0,-1]`. Result SHA256 is
  `98688726...b1a8`; Gate 0/Writer remain false and validation/held access
  remains zero. The trend gate correctly prevents stage 32.
- The terminal launcher checksum pass covers all candidates, latest schema-3
  recoveries, round/stage JSON, 12 retained stage videos, gallery/index,
  terminal result, and telemetry. All videos were separately decoded at their
  stages. Peak memory across the ladder is 19,274MiB on GPU4 and at most
  6,075MiB on GPUs5--7; total output is 193MiB. GPUs4--7 returned to 0MiB after
  every segment. The gallery remains bounded to the latest four videos and is
  available at the run's `index.html`; Trackio project `EMBER_gate0` contains
  the 12 staged arm runs.
- Completed one artifact-only physical-update audit: episode-8-to-24 LoRA
  operator increments are nonzero (`0.0586/0.0644` zero-init and
  `0.0732/0.0841` supervised-init for task3/4). Combined with finite gradients,
  safe KL/drift, and low but improving critic explained variance, this rules out
  a no-update implementation failure while preserving the behavioral negative.
- Froze the cheapest next source-only discriminator before its outcome in
  `configs/gate_zero_task_local_rl_support_replay.toml`, SHA256
  `f539b7376dd1e265076941d7b45022934802f2931bdb54b866b9b97e1a533909`.
  A four-rank one-time script (SHA256 `1a410a6a...37c4`) loads the immutable
  episode-24 states and reuses the canonical round-0 collector on exact init
  states 8--15/seeds 6200--6207 with zero updates. It consumes only 32 source
  episodes and classifies credit/optimizer versus coverage/generalization;
  it cannot change the failed Gate result or authorize Writer.

## 2026-07-19 support-replay diagnostic closed and next boundary narrowed

- Long-run `gate0_support_replay_diagnostic_20260719_095823` completed rc 0 in
  1m46s from clean `0804f21` on GPUs 4--7 and released all four devices. Its
  immutable output is the bounded 32-episode source-only packet
  `support_replay_20260719T095739Z`; all checksums pass.
- Status is `support_replay_no_improvement`. Supervised task3/task4 paired net
  wins on the exact round-0 support slice are `[0,-1]`; zero-init task3/task4
  are `[-1,-1]`. Actor/trainable and optimizer states are unchanged because
  this diagnostic performs zero updates. Result SHA256 is
  `7e92b745...414e`; Gate 0 and Writer remain false, and validation/held/locked
  numeric access remains zero.
- The result rejects the coverage-only branch: the existing reward update does
  not improve even the seen support slice. No episode-32 or blind scale
  extension is authorized. The same task3/task4, exact mature LoRA, fixed
  reporting evaluator, four matched arms, query/drift safeguards, and original
  success-gain rule remain active.
- Completed one read-only action-authority audit before another recovery. The
  rollout action is the postprocessed environment action; LIBERO adds no env
  action postprocessing; checkpoint action normalization statistics match
  bit-exactly in both directions; numerical round-trip error is at most
  `7.2e-7`; replay uses those actions; and SmolVLA honors the padding mask.
  Action-coordinate mismatch is therefore excluded without new GPU outcomes.
- The full research plan now explicitly keeps four different optimization
  questions separate: supervised Writer cold start; Writer-only RL with frozen
  base and reward updating Writer only; ordinary task-local LoRA RL with
  frozen Writer/base and LoRA-only updates; and adaptation-aware source
  meta-outer learning. Gate 0 is not SFT-only: the already executed four-arm
  recovery is base, supervised LoRA, zero-init LoRA plus ordinary RL, and
  supervised-init LoRA plus identical RL. Because none is yet positive, the
  next source-only recovery must be a new primary-source-supported
  credit/acquisition correction frozen before outcome, not a threshold, task,
  seed, or surface change.

## 2026-07-19 horizon-credit recovery frozen before outcomes

- Verified the primary FPO++ manipulation recipe at commit
  `b80112be1e8362263c4cd176e7aef21a275ff1c6`: its main benchmark uses a
  16-step execution horizon. This motivates one bounded compatibility
  discriminator after the support replay ruled out coverage-only failure. It
  does not authorize copying the official multi-million-step budget.
- Added the outcome-free contract
  `configs/gate_zero_task_local_rl_horizon_credit.toml`, SHA256
  `491d031565409962cfb96cea09f6ac73ae636a1fe87a14aeb441b18c2d15e05b`.
  The same trainer now scopes training rollout execution to 16 actions while
  preserving a 50-slot model action tensor and restoring the canonical 50-step
  evaluator immediately afterward. Replay contains 25 transitions per episode
  and 200 rows per arm-round; only the 16 executed actions are unmasked.
- Kept one canonical four-GPU launcher and made its allowed stage nodes come
  from the selected sealed contract. For this recovery only 8/16 are legal;
  stage 24 fails before launch. The prior critic-warmup contract still validates
  unchanged. Full-replay feature and old-loss capture is microbatched to cap
  memory without changing row order or the 16-sample actor minibatches.
- Focused recovery tests pass 21/21; the full repository unittest suite, Python
  compilation, shell syntax, predecessor-artifact validation, canonical dry
  run, forbidden-stage dry run, and whitespace checks are the pre-launch gates.
  After clean Git delivery, run exactly one real-model no-environment smoke on
  one free GPU. Only a passing smoke with actor identity, finite critic updates,
  exact 200-row/16-of-50 mechanics, and at least 10GiB headroom may open the
  four-GPU stage-8 source run.

## 2026-07-19 horizon-credit real-model smoke passed

- Preflight found clean `b5aaaea`, 306GiB personal usage against the 500GiB
  cap, 2.9TiB free on `/data`, and all eight A100s idle. The canonical smoke
  used only GPU4, zero environment episodes, and a new mechanics output root.
- The first long-run failed rc 1 in 1.47s before model load because the telemetry
  sampler precreated a directory that the smoke required to create atomically.
  The second failed rc 1 before flow/critic work because the support-loader
  surrogate legitimately repeated provenance keys across batches while
  production rollout keys are task/seed/anchor unique. Both packets and logs
  are retained. Narrow smoke-only fixes allowed a telemetry-only directory and
  suffixed deterministic smoke slots; no trainer or contract path changed.
- Long-run `gate0_horizon_credit_real_model_smoke_recovery2_20260719_104517`
  then completed rc 0 in 31.42s. It verified 200/200 replay identities,
  16-step scoped execution and 50-step restoration, 16-of-50 processed action
  masking, finite `[16,8]` real-model flow losses, 130 critic updates, exact
  actor identity, empty actor optimizer, and healthy temporal credit. Peak
  allocated/reserved memory was 2,858/4,004MiB; GPU4 released.
- Result SHA256 is `29528c5f...844a`; result/source/config/telemetry checksums
  all pass. This mechanically authorizes only the already frozen four-GPU
  stage 8 from a new output root. Stage 8 remains critic-only and cannot pass
  Gate 0 or authorize Writer; only its frozen decision may open exact-resume
  stage 16.

## 2026-07-19 horizon-credit stage 8 completed

- Long-run `gate0_horizon_credit_stage8_20260719_104802` completed rc 0 from
  clean `ac9cf2f` in 3m34s on GPUs 4--7. It wrote a new 99MiB output root and
  released every GPU. Trackio project `EMBER_gate0` contains the four arm runs;
  the local `index.html` gallery contains one bounded video per arm.
- Every arm collected 200 replay rows at execution horizon 16 and made 90
  critic updates. Mechanics and temporal credit are healthy, actor state is
  exact, actor optimizer updates are zero, saturation is zero, and fixed
  development paired gains are `[0,0]` for both initialization families.
  Frozen status is `horizon_credit_warmup_complete_continue_to_16`; Gate 0 and
  Writer remain false.
- Stage-result SHA256 is `a3b93ebf...1b0d`. Validated 15 JSON files, four
  latest recoveries, four candidate packets, four decodable videos, and the
  telemetry checksum. Peak memory is 19,266MiB on GPU4 and <=5,121MiB on the
  other ranks, retaining >60GiB headroom. The only next action is an exact
  same-output resume to terminal stage 16 after clean documentation delivery
  and fresh GPU preflight; no parameter, task, seed, threshold, or surface may
  change.

## 2026-07-19 horizon-credit stage 16 terminated negative

- Long-run `gate0_horizon_credit_stage16_20260719_105345` exact-resumed the
  clean `b50458d` stage-8 output on GPUs 4--7 and completed rc 0 in 7m58s. Peak
  memory was 19,278MiB on GPU4 and at most 6,439MiB on the other ranks; all
  devices released. The same output is 170MiB and the bounded Trackio/video
  evidence remains available.
- The fixed development gains are zero-init `[-2,+1]` and supervised-init
  `[0,-1]`. Status is `task_local_rl_early_check_not_supported`; no checkpoint
  was selected, and Gate 0, Writer, validation, held, and fresh Gate remain
  closed. Result SHA256 is `771eb3b9...f496`.
- Revalidated root checksums, every recovery/candidate/round JSON, all eight
  videos, gallery, and telemetry. Actor/critic update mechanics are healthy,
  so no stage-24 scale extension is permitted. The next source-only action is
  to freeze and run one zero-update replay of the exact round-1 training slice
  from the immutable step-16 actors; it is a diagnostic only and consumes no
  final closed-loop surface.
- Frozen that diagnostic before outcome in
  `configs/gate_zero_task_local_rl_horizon_support_replay.toml`, SHA256
  `7e676c52...a9cb`. Its script is hash-bound at `54e43db0...2599`; the
  no-GPU authority dry-run confirms the terminal result, four recovery
  manifests, initial vectors, 32-episode budget, and validation/held closure.
  Python compilation and whitespace checks pass. Commit and push this frozen
  boundary before allocating GPUs 4--7.

## 2026-07-19 horizon support replay completed

- Committed/pushed the outcome-free boundary as `bff88bd`. Live preflight kept
  the unrelated MemLLM processes on GPUs 0--3 untouched and selected idle GPUs
  4--7. Long-run `gate0_horizon_support_replay_20260719_111434` completed rc 0
  in 2m01s with 32 source episodes and exactly zero policy/critic/optimizer
  updates; all four GPUs returned to 0MiB.
- Output is the 140KiB packet
  `horizon_support_replay_20260719T111331Z`. All four checksums pass. Peak GPU
  usage was 16,679MiB on rank0 and 4,317MiB on ranks1--3, preserving more than
  64GiB headroom; no video or final Gate surface was consumed.
- Result SHA256 is `4a0c13a0...f7ef`, status
  `horizon_support_replay_improves_but_development_does_not`. Supervised-init
  task3/4 paired wins are `[-1,+1]`; zero-init wins are `[+1,0]`. This records
  partial seen-support acquisition but no two-task or development utility.
  Gate 0 and Writer remain false.
- Next freeze, test, and cleanly deliver one coverage-only recovery that starts
  a new trajectory and adds disjoint legal source training slices under the
  unchanged canonical trainer and Gate. It may run only staged evidence nodes
  and must stop before blind scale, fresh Gate, validation, held, or locked
  access.

## 2026-07-19 horizon coverage recovery frozen before outcomes

- Added `configs/gate_zero_task_local_rl_horizon_coverage.toml`, SHA256
  `72e4f13e...f241`. It binds both the terminal horizon result and the partial
  support-replay result, preserves all earlier failure provenance, and changes
  only the number of disjoint legal source training rounds from two to at most
  four in a new trajectory.
- Reused the single canonical trainer/launcher. The validator distinguishes the
  sealed two-node horizon probe from the new four-node coverage recovery;
  launcher preflight now hash-checks both new evidence packets before GPU use.
  No predecessor actor is selected or appended, and validation/held/locked
  access stays false.
- Red/green focused tests now pass 22/22. Python compilation, shell syntax,
  node-8 and resume-node-24 dry-runs, config/hash authority, and diff checks
  pass. The frozen nodes are 8/16/24/32; stage 24 must show the predeclared
  positive aggregate trend to open 32, while the original two-task candidate
  rule remains unchanged. Deliver clean Git before launching node 8.

## 2026-07-19 horizon coverage node 8 completed

- Committed/pushed the pre-outcome implementation as `4b8dde6`. Live
  preflight found all GPUs free; long-run
  `gate0_horizon_coverage_stage8_20260719_112918` used GPUs 4--7 and completed
  rc 0 in 3m36s. The 99MiB output is
  `horizon_coverage_ep8_20260719T112902Z`, with local gallery and Trackio
  records retained; all GPUs released.
- Each arm collected 200 replay rows, performed 90 critic optimizer updates,
  and performed zero actor optimizer updates with exact actor identity.
  Mechanics/temporal credit are healthy, saturation is zero, and fixed
  development gains are `[0,0]` for both initialization families. Frozen
  status is `horizon_coverage_warmup_complete_continue_to_16`; Gate 0 and
  Writer remain false.
- Stage SHA256 is `219eff17...89dd`. Validated 15 JSON files, four candidate
  packets, four atomic recovery packets, four fully decoded videos, and the
  telemetry checksum. Peak memory is 19,267MiB on GPU4 and <=5,122MiB on the
  other ranks. After clean documentation delivery and fresh preflight, exact
  resume only to node 16; do not alter config, checkpoint, task, or surface.

## 2026-07-19 horizon coverage node 16 reproduced the sealed trajectory

- Long-run `gate0_horizon_coverage_stage16_20260719_113650` exact-resumed the
  same output from clean `52e1efc` and completed rc 0 in 8m03s. GPU4/5/6/7
  peak memory was 19,278/6,417/6,439/6,439MiB; mean utilization was
  65.4/51.3/55.2/55.7%, and every GPU released.
- The stage exactly reproduces the predecessor outcome: zero-init task3/4
  paired gains `[-2,+1]`, supervised-init `[0,-1]`. Status is
  `horizon_coverage_recovery_continue_to_24`, but no checkpoint, Gate, Writer,
  validation, held, or fresh-Gate authority is selected.
- Stage-result SHA256 is `3d4bf5a1...5050`. Validated all four new candidate
  and atomic recovery packets, four fully decoded stage-16 videos, 24 JSON
  files, and telemetry. Output is 170MiB. After clean documentation delivery
  and live preflight, exact-resume once to node 24, the first new source
  coverage slice; its frozen trend rule controls any node-32 continuation.

## 2026-07-19 node 24 closed and evidence authority repaired

- Coverage node 24 exact-resumed the clean `9c9f239` trajectory and completed
  rc 0 in 8m14s. Supervised-init task3/task4 paired gains are `[0,0]`, zero-init
  gains are `[0,-1]`, and supervised-init advantages are `[-1,+2]`. Result
  SHA256 is `9b738193...0a94`; four candidates, four recovery packets, 34 JSON
  files, four videos, gallery, telemetry, and checksums validate. Output is
  196MiB and GPUs 4--7 released.
- This n=8 outcome is mechanism smoke only. It cannot pass/reject Gate 0,
  select a checkpoint, authorize Writer, or open node 32. No validation, held,
  locked-report, task, seed, or threshold authority was accessed or changed.
- Added the result-before-outcome contract
  `configs/gate_zero_evidence_repair.toml`, SHA256 `0196419d...aa4e`. Gate -1
  is now passed with its
  immutable 19/24 ordered/wrong-video, 15/24 paired, original 0.80 threshold,
  and drop-last residual explicitly retained; no more compute is authorized to
  polish it. Gate 0 requires n>=32 paired rollouts/task/arm, multiple policy RNG
  seeds, >=2 independent training seeds, paired intervals, disjoint
  source-only confirmation, horizon-16 primary evaluation plus horizon-50
  robustness, process metrics, dual-reference drift, and truthful interaction
  accounting.
- Added one cohesive fail-closed evidence/statistics validator and focused
  tests. The existing mean-before-ratio code is now explicitly a custom
  chunk-level flow-loss PPO pilot. A minimal faithful FPO++ loss core uses
  per-flow-sample/group-size-one ratios, modified Huber, old-loss/log-ratio
  clamps, and PPO clipping; it remains inside the existing temporal-credit
  owner rather than creating a second trainer. The next action is test and
  clean Git delivery of this frozen boundary, followed by a result-blind
  source difficulty/partition audit—not another n=8 or blind node continuation.

## 2026-07-19 source difficulty audit frozen before outcomes

- Added `configs/gate_zero_base_difficulty_audit.toml` at SHA256
  `ae73a4b0...6443`, plus one thin orchestration entry inside the existing
  `gate_zero_support` owner and a single canonical shell launcher. It reuses the
  upstream evaluator and does not add a trainer or LoRA path.
- The contract covers nine source-only factor-derived candidates, 32 unique
  physical train states/task, four policy RNG batches, horizon 16, four-GPU
  task parallelism, 288 total episodes, deterministic base-only headroom
  selection, one retained video/task, Trackio, gallery, telemetry, and atomic
  result/selection manifests. Validation/held/locked access and all LoRA
  outcomes remain forbidden.
- Focused tests pass for contract authority, world-size assignment, physical
  state setting, dry-run routing, deterministic selection, pairing statistics,
  and SHA checks. The dry run binds the exact 10k base checkpoint and competence
  packet; their live manifest/result hashes validate. Next: full repository
  verification, clean commit/push, then live GPU/storage preflight and one
  <=30-minute four-GPU audit. This audit cannot itself pass Gate 0.
- The first launch `gate0_source_base_difficulty_audit_20260719_124341` failed
  rc 1 in 20.22 seconds before reset/outcome because LeRobot's lazy async wrapper
  exposes no `set_attr`. Four failure packets and telemetry are retained; all
  GPUs returned to 0MiB. Applied one narrow wrapper-materialization repair and
  added a faithful lazy-wrapper regression. After clean delivery, run one
  actual-env no-policy state-assignment/reset smoke, then relaunch to a new
  output root only if the physical before/after IDs match exactly.

## 2026-07-19 source difficulty audit completed and selection frozen

- The actual lazy-vector no-policy smoke passed, then longrun
  `gate0_source_base_difficulty_audit_recovery1_20260719_124855` completed rc 0
  from clean `3cbb975` in 645.34s. All 288 horizon-16 source episodes, nine
  task packets, and deterministic state/seed identities are present; GPUs 4--7
  released.
- Base successes are `{6:4,9:27,16:17,20:32,23:0,33:13,39:24,46:4,63:0}`.
  The frozen base-only rule yields eligible `[6,16,33,39,46]` and selects
  confirmation tasks `[6,16,33,39]` (`open/stack/close/turn_off`). Result SHA256
  is `240de313...dd61`; selection manifest SHA256 is `a4d57cf9...f670`.
- Checksums/JSON/state uniqueness/policy-seed coverage/gallery/latest and nine
  video decodes pass. Output is 2.3MiB. Telemetry peaks are 20,021MiB on GPU4
  and 1,752MiB on each other GPU, with a final three-rank idle tail; record this
  as an efficiency residual and do not rerun valid science to polish it.
- Next freeze the matched development/confirmation evaluation contract around
  task3/task4 and selected tasks6/16/33/39: >=32 paired rollouts/task/arm,
  multiple policy RNG seeds, >=2 independent training seeds, horizon16 primary,
  horizon50 robustness, process diagnostics, and LoRA-identical arms. Gate 0
  and Writer remain unauthorized; validation/held/locked access remains zero.

## 2026-07-19 matched Gate 0 evidence contract frozen

- Added `configs/gate_zero_matched_evidence.toml` (SHA256
  `625db578...e0038`), binding the evidence-repair
  contract, permanent split, mature LoRA support, base-difficulty result, and
  selected confirmation manifest. Tasks 3/4 remain development; source tasks
  6/16/33/39 and their deterministic partitions are confirmation.
- Required trainable checkpoint seeds are `2026071830` and `2026072030`; the
  predeclared ambiguity-only replicate is `2026072130`. Core contract tests
  bind those seeds, the disjoint task sets, n>=32, horizon 16, and the exact
  LoRA support. Frozen base is recorded once rather than relabeled as multiple
  training replicates.
- Wired the already implemented modified-Huber per-flow-sample FPO++ loss into
  the existing temporal-credit trainer through one explicit surrogate
  selector. Historical configs receive the accurate
  `historical_chunk_mean_flow_ppo` label at load time; no second trainer or RL
  path was added.
- Updated the benchmark report and README to reflect the owner-resolved Gate -1
  status: passed with immutable 19/24 ordered/wrong-video, 15/24 paired,
  original 0.80 threshold, and drop-last residual. Gate 0 and Writer remain
  unauthorized.
- The existing staged launcher now consumes the sealed faithful contract
  without a second trainer. The first fresh segment reaches 16 interaction
  episodes/arm (critic warmup plus one actor round), executes all four matched
  development conditions at horizon 16, and remains n=8 early-check evidence
  only. Targeted tests and the canonical dry-run pass. Next: clean commit/push,
  live preflight, and immediate launch on GPUs 4--7; do not consume
  confirmation or h50 for candidate selection.
