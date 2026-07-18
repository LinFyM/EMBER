# EMBER Progress and Handoff

## Current state

- General thesis, embodied concept, novelty landscape, decisions, and prior
  Writer-program lessons are documented.
- An independent expert completed `docs/expert_plan.md` with a conditional-go
  recommendation and a full staged design.
- The expert plan was reviewed before implementation. The active corrections
  are recorded in `docs/execution_brief.md`.
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
same-observation language-to-action path. Gate -1 is still in progress on
correct paired-goal behavior with legal source competence and on video causal
probes. Writer training remains unauthorized.

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

## Immediate handoff

1. Preserve the permanent reseal and fresh canonical manifest as the only active
   split/normalization path; do not reuse the rejected split's normalization.
2. Preserve both same-state native-goal mechanics and the same-observation
   language-action path, then run action-hidden video content/temporal controls.
   Do not use held results to choose thresholds, task IDs, or remedies.
3. Predeclare the smallest source-task closed-loop useful-update oracle so its
   legal competence can later complete paired-goal behavior without weakening
   the Gate -1 or Gate 0 contracts.

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
