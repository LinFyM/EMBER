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
- The pinned LIBERO-90 download is active in the resumable zero-GPU long-run
  `libero90_canonical_download_20260717_163546`. It is restricted to the exact
  90 HDF5 files at revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`, uses
  up to eight network workers, retains partials for resume, and verifies the
  declared 66,658,085,995-byte surface before completion.
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
  Eight focused tests and all 49 repository tests pass; the run is not yet
  launched.

## Current phase

Phase 0, reproducible substrate, is in progress. The immutable contract, first
official mechanics smoke, explicit PyAV decoder path, and useful single-GPU
concurrency envelope are established. The ten-task spatial mechanics sweep is
also complete. Gate -1 evaluation identity has completed its bounded mechanism
diagnosis. The selected recovery is fixed-contract statistical/functional
reproducibility: the unchanged async evaluator uses one predeclared measured-safe
batch/mode across every arm, task-level repetition and confidence intervals are
primary, and batch-1 exactness is a small audit. The canonical LIBERO-90
manifest and remaining Gate -1 specification probes are now the active path.

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

1. Download and audit canonical LIBERO-90, generate the full task/BDDL/init
   state/controller/normalization/split manifest with the implemented canonical
   builder, inspect the quality report, then complete the remaining Gate -1
   probes.
2. Predeclare matched fixed-batch controls and uncertainty criteria on the
   official-overlap/source surface. Preserve both strict identity failures and
   do not use held results to choose thresholds or remedies.
3. Predeclare and run the smallest closed-loop useful-update oracle only after
   benchmark/specification probes are mechanically valid.

## Last verified handoff facts

- The repository now contains only lightweight environment/contract code and
  tests; external models, data, caches, and outputs remain outside Git.
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
  `$EMBER_OUTPUT_ROOT/phase0/latest/index.html`; historical run directories are
  retained until verified-regenerable, unpinned media becomes large or
  numerous enough for a recorded cleanup.
