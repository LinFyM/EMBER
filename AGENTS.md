# EMBER Repository Instructions

## Purpose

EMBER is a standalone embodied-learning research project. Treat every proposed
mechanism and novelty statement as a hypothesis to test, not an established
result. The repository is now in the execution phase: turn the research design
into reproducible evidence without weakening the held-out contract.

## Required reading order

Before changing code, data, environments, or experiment state, read:

1. `README.md`
2. `task_plan.md`
3. `findings.md`
4. `progress.md`
5. `docs/origin_and_general_thesis.md`
6. `docs/concept.md`
7. `docs/prior_work_memllm_lessons.md`
8. `docs/novelty_and_landscape.md`
9. `docs/decisions_and_open_questions.md`
10. `docs/expert_plan.md`
11. `docs/execution_brief.md`
12. `docs/benchmark_validity_report.md`

`docs/expert_plan.md` is preserved expert advice. Where its eight-GPU resource
assumption or stage details conflict with the active execution contract,
`docs/execution_brief.md` and this file take precedence.

## Active objective

Advance EMBER from a research proposal to a reproducible staged experiment.
Begin with a faithful SmolVLA plus LIBERO development path and establish:

1. benchmark and task-specification validity;
2. a useful task-specific update oracle;
3. a stable canonical update representation;
4. only then, amortized Writer acquisition and later RL components.

Do not implement the complete system merely because it appears in the expert
plan. Each expensive component needs evidence from its scientific predecessor.

## Hard resource constraint

- At most **four NVIDIA A100 80GB GPUs** may be allocated to EMBER at any time,
  across all concurrent EMBER processes and agents.
- Default to one GPU for smoke tests and one or two GPUs for pilots. Use three or
  four only when measured memory or throughput justifies it.
- After a minimal correctness smoke establishes the real memory baseline, tune
  useful batch size, environment/task parallelism, gradient accumulation, and
  caching so that each allocated A100 normally uses about 70GB and retains about
  10GB average headroom against OOM. This is a steady-state efficiency target,
  not permission to allocate dummy tensors or sacrifice correctness and
  reproducibility merely to fill memory.
- Do not copy eight-GPU launch commands from `docs/expert_plan.md`. Recompute
  batch size, gradient accumulation, parallelism, GPU-hours, and wall-clock for
  the four-GPU ceiling.
- Before a substantial download or run, check available storage and estimate
  data, cache, checkpoint, and log growth.

## Scientific invariants

- Writer-visible held-task input may contain language and action-hidden video,
  but never held actions, proprioceptive trajectories, rewards, terminal flags,
  task IDs, filenames, or hidden normalization statistics.
- Shared base, Writer, encoders, and bank are frozen during held-task
  evaluation. Only predeclared task-local state may adapt from held rewards.
- Keep source, validation, and reporting-only held surfaces separate. Do not
  tune architecture, thresholds, seeds, or checkpoints after reading held
  results.
- Immediate functional utility is required before claiming bootstrapping.
- A task-conditioned geometry must beat unit or fixed-global geometry under a
  matched center, optimizer, parameter count, interaction budget, and escape
  path.
- Parameter distance and raw LoRA-factor MSE are diagnostic or auxiliary, not
  the primary scientific objective.
- Keep EMBER independent from MemLLM code and Wiki/QA mechanisms. Reuse lessons,
  not implementation dependencies.

## Gate recovery protocol

A failed or ambiguous gate is a diagnosis trigger, not permission to declare
failure casually. Before stopping:

1. verify mechanics, data authority, split integrity, seeds, metrics, and the
   exact failed claim;
2. classify the failure as benchmark/specification, information sufficiency,
   representation, Writer acquisition, optimization/RL, implementation, or
   resource/throughput;
3. record the evidence in `findings.md` and the attempted remedy in
   `progress.md`;
4. explore bounded remedies on source and validation surfaces, such as a
   predeclared target-layer/rank expansion, bank dimension or canonicalization
   change, better temporal video representation, stable RL estimator, or a
   smaller faithful model;
5. rerun the gate with matched controls and fresh evidence.

Recovery must not violate the core thesis or evaluation contract. Do not expose
held labels, update shared parameters on held tasks, silently change the task
distribution after seeing held results, remove strong baselines, or lower a
threshold solely to turn a failure into a pass. If bounded remedies are
exhausted, preserve the negative result and narrow the claim explicitly.

## Current implementation decisions

- Use SmolVLA as the development and mechanism-validation policy. Scale a
  surviving result to OpenVLA-OFT as confirmation rather than starting with the
  7B path.
- Treat language/video-to-center as a task-conditioned hypernetwork or
  amortized adapter initializer, not a universal meta-optimizer.
- Make neutral-prompt parameter compilation a co-primary mechanism test; also
  report the practical setting where the online policy receives the task
  instruction.
- Include a strong language-only HyPoGen/DISC-style parameter-generator
  baseline, not only task-ID, retrieval, or direct-conditioning controls.
- Train the geometry before evaluating it: first use a differentiable
  low-dimensional source support/query inner loop over bank coordinates, then
  optionally refine the Writer with source-only reward outer learning.
- Prefer a center over a canonical bank plus soft task-conditioned geometry and
  a residual escape path. A hard inescapable subspace is not the default.

## Experiment and systems practice

- Reproduce an official known checkpoint or recipe before interpreting a custom
  source-only result.
- Pin repository commits, dataset revisions, file hashes, environment versions,
  task manifests, normalization authority, seeds, and exact commands.
- Optimize the full training and inference path for useful throughput. Reuse
  canonical model loads, manifests, decoded or preprocessed observations,
  cached features, and other scientifically equivalent intermediates whenever
  safe; avoid duplicate downloads, repeated preprocessing, idle simulator/GPU
  pipelines, and redundant computation, while recording cache provenance and
  invalidation rules.
- Use durable logs for long runs. Store temporary output under `.codex/tmp/` or
  ignored output directories; never commit datasets, checkpoints, credentials,
  private infrastructure, or raw copyrighted papers.
- Make simulator and policy runs reviewable after execution with compact video
  galleries or equivalent visual artifacts, but keep that surface bounded.
  Preserve metrics, manifests, failure packets, the current `latest` gallery,
  and explicitly designated evidence; after artifact-heavy phases, inventory
  accumulated media and remove only verified-regenerable, unpinned older videos
  when they become numerous or large. Record every such prune with paths,
  hashes, reason, and bytes released, and never delete ambiguous scientific
  evidence automatically.
- Keep one canonical implementation path. Put unselected alternatives in
  `docs/decisions_and_open_questions.md` until evidence selects them.
- Use isolated branches or worktrees for overlapping write-capable tasks. Never
  let multiple writers edit one worktree concurrently.
- Update `task_plan.md`, `findings.md`, and `progress.md` after meaningful state
  changes so a resumed agent can recover without conversation history.

## Current-goal completion evidence

The first execution goal is complete only when the repository contains:

- a reproducible four-GPU-compatible environment and data manifest;
- a benchmark/specification validity report;
- a closed-loop useful-update oracle report;
- a canonical representation report with functional preservation metrics;
- recorded recovery attempts for any failed or ambiguous gate; and
- an evidence-based decision on whether Writer center training is authorized.
