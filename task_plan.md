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
| Gate 0. Useful-update oracle | pending | independent query and closed-loop gain with drift/non-harm diagnostics |
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
   pinned 66.66GB LIBERO-90 download and canonical 90-file run remain active,
   followed by the remaining specification probes.
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
9. [ ] Design the smallest useful-update oracle pilot with explicit pass,
   diagnosis, and recovery criteria.
10. [ ] Run the predeclared official-overlap specification pilot before scaling
    Gate -1: `libero_spatial` tasks 0/1, one fixed async batch of eight per arm,
    correct/no-spec/scene-only/swapped prompts, paired seeds/init states, one
    video per arm, and no Gate decision from this overlap-trained surface.
