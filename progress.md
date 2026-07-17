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
- No EMBER training or scientific evaluation has started. No GPU has been
  allocated by the Phase 0 bootstrap work.

## Current phase

Phase 0, reproducible substrate, is in progress. The environment and immutable
revision contract are established; the official one-episode SmolVLA/LIBERO
mechanics reproduction and its resource measurements are the active critical
path.

## Implementation ownership review

- `src/ember/contracts.py` is the single owner of immutable scientific,
  resource, split, and artifact-surface invariants.
- `src/ember/runtime_env.py` owns only version- and content-guarded repairs to
  the pinned third-party Python installation. It does not own simulator or
  policy behavior.
- `src/ember/phase0_runtime.py` is the single owner of checked external-asset
  materialization: the offline SmolVLA view and LIBERO path/asset binding.
- `scripts/bootstrap_env.sh` remains a thin environment entrypoint and
  `scripts/zig-cxx` is only the pinned user-space compiler adapter. Tests mirror
  these three Python ownership surfaces; no legacy or parallel implementation
  path was added.
- Retirement triggers are explicit: remove the BDDL and robosuite repairs after
  a pinned dependency upgrade proves the upstream wheels no longer contain the
  duplicate metadata/shared-log defects; remove the local SmolVLA/LIBERO
  bindings when pinned upstream APIs propagate constructor/tokenizer revisions
  and asset revisions end-to-end. Git preserves the old evidence rather than
  retaining superseded executable paths.

## Immediate handoff

1. Finish and hash-check the pinned official LIBERO smoke checkpoint, SmolVLA
   base, and LIBERO assets. Use standard HTTP when Xet reconstruction fails.
2. Materialize an offline runtime view that maps both SmolVLA's VLM constructor
   and tokenizer to the pinned local SmolVLM snapshot.
3. Create an isolated `LIBERO_CONFIG_PATH` pointing to pinned BDDL, init-state,
   asset, and data roots; verify EGL on only the selected free GPU.
4. Run the official one-episode `libero_spatial` mechanics smoke on one GPU and
   record peak VRAM, wall time, simulator throughput, output size, and failure
   evidence.
5. Download and audit canonical LIBERO-90, generate the full manifest, then
   implement Gate -1 probes and the smallest closed-loop useful-update oracle.

## Last verified handoff facts

- The repository now contains only lightweight environment/contract code and
  tests; external models, data, caches, and outputs remain outside Git.
- `docs/expert_plan.md` uses an obsolete eight-GPU planning envelope; active
  execution must recalculate every launch for at most four GPUs.
- A gate failure requires diagnosis and bounded recovery, not immediate
  abandonment and not post-hoc weakening of held-out constraints.
