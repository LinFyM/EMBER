# EMBER Progress and Handoff

## Current state

- General thesis, embodied concept, novelty landscape, decisions, and prior
  Writer-program lessons are documented.
- An independent expert completed `docs/expert_plan.md` with a conditional-go
  recommendation and a full staged design; it is now historical provenance.
- The owner rechecked the original design on 2026-07-18. The active contract is
  direct language/action-hidden-video to complete task-specific LoRA, immediate
  utility, ordinary task-local LoRA RL, source-only Writer reward/meta learning,
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
