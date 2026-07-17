# EMBER Progress and Handoff

## Current state

- General thesis, embodied concept, novelty landscape, decisions, and prior
  Writer-program lessons are documented.
- An independent expert completed `docs/expert_plan.md` with a conditional-go
  recommendation and a full staged design.
- The expert plan was reviewed before implementation. The active corrections
  are recorded in `docs/execution_brief.md`.
- The active compute ceiling is four A100 80GB GPUs.
- No EMBER implementation, environment lock, dataset manifest, checkpoint, or
  experimental result exists yet.

## Current phase

Phase 0, reproducible substrate, is in progress. The next executor should build
and verify the SmolVLA/LIBERO development path before making scientific claims or
starting expensive jobs.

## Immediate handoff

Read `AGENTS.md`, then follow the repository reading order. Start by recording:

- exact upstream commits and model/dataset revisions;
- package, CUDA, PyTorch, simulator, renderer, and driver compatibility;
- known-checkpoint inference/evaluation reproduction;
- task and data manifests with hashes and split authority;
- measured one-GPU VRAM, throughput, disk growth, and failure logs.

Then implement Gate -1 probes and design the smallest closed-loop useful-update
oracle pilot. Update this file after each material transition.

## Last verified handoff facts

- The repository is documentation-only apart from planning state.
- `docs/expert_plan.md` uses an obsolete eight-GPU planning envelope; active
  execution must recalculate every launch for at most four GPUs.
- A gate failure requires diagnosis and bounded recovery, not immediate
  abandonment and not post-hoc weakening of held-out constraints.
