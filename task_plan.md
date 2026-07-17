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
| Gate -1. Benchmark/spec validity | pending | task-factor audit, counterfactual/spec-swap/no-language/video controls |
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
   reset-observation, initial-action, and short-rollout identity checks.
7. [ ] Design the smallest useful-update oracle pilot with explicit pass,
   diagnosis, and recovery criteria.
