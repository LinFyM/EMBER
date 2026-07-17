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

## Current phase

Phase 0, reproducible substrate, is in progress. The immutable contract, first
official mechanics smoke, explicit PyAV decoder path, and useful single-GPU
concurrency envelope are established. The ten-task spatial mechanics sweep is
also complete. Gate -1 evaluation identity is in bounded diagnostic recovery:
the strict mechanics layer has isolated sparse renderer variation, while the
frozen-policy action and short-trajectory layers remain open. That recovery and
the canonical LIBERO-90 manifest are the active critical path.

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

1. Continue the bounded Gate -1 identity recovery without reinterpreting the
   mechanics failure: compare the frozen policy on one exactly repeated reset
   observation across the predeclared batch ladder, then record matched actual
   initial actions and five-step trajectories across sync/async batches 1 and 2.
2. Stop for a recorded scientific decision before changing upstream evaluator
   semantics, accepting nonzero RGB identity tolerance, or choosing among
   deterministic-render workarounds.
3. Download and audit canonical LIBERO-90, generate the full task/BDDL/init
   state/controller/normalization/split manifest, then complete the remaining
   Gate -1 probes.
4. Predeclare and run the smallest closed-loop useful-update oracle only after
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
- The latest completed visual review page is available locally at
  `$EMBER_OUTPUT_ROOT/phase0/latest/index.html`; historical run directories are
  retained until verified-regenerable, unpinned media becomes large or
  numerous enough for a recorded cleanup.
