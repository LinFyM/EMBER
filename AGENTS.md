# EMBER Repository Instructions

## Purpose

This repository is a standalone research project. Treat all proposed mechanisms
and novelty statements as hypotheses to be criticized and tested, not as
established results.

## Required reading order

Before proposing code or experiments, read:

1. `README.md`
2. `docs/origin_and_general_thesis.md`
3. `docs/concept.md`
4. `docs/prior_work_memllm_lessons.md`
5. `docs/novelty_and_landscape.md`
6. `docs/decisions_and_open_questions.md`
7. `docs/expert_review_brief.md`

## Current task for a remote expert

Independently assess whether the general research thesis and its embodied
instantiation are reasonable, identify obvious flaws and improvements, select
models and data that fit the available maximum of eight A100 80GB GPUs, and
plan a staged path from the smallest decisive experiment to the complete EMBER
design. Produce an executable research plan at `docs/expert_plan.md` using the
requested structure in `docs/expert_review_brief.md`.

Do not begin a full implementation before the plan resolves the P0 decisions.
Small read-only inspections, feasibility calculations, and disposable probes
are allowed when they materially support the plan.

## Reasoning constraints

- Separate verified literature facts, reasoned inferences, and untested design
  assumptions.
- Prefer primary papers, official project pages, and official repositories.
- Do not claim novelty from any one ingredient such as hypernetworks, LoRA,
  video conditioning, policy subspaces, or reinforcement learning.
- Define exactly what distribution shift and held-out task mean in every
  proposed evaluation.
- Do not assume access to a real robot, unlimited demonstrations, or unlimited
  compute. The declared maximum is eight A100 80GB GPUs; state all other
  resource assumptions and provide a lower-cost path.
- Preserve the distinction between shared outer-loop parameters and task-local
  inner-loop adaptation state.
- Include negative-result criteria and matched-budget baselines.

## Repository boundaries

- Keep this project self-contained and independent of unrelated repositories.
- Do not add checkpoints, datasets, credentials, private infrastructure details,
  or raw copyrighted paper files.
- Use links and concise technical summaries for literature evidence.
- Preserve one canonical design path; alternatives belong in the decision
  document until selected.
