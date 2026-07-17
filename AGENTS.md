# EMBER Repository Instructions

## Purpose

This repository is a standalone research project. Treat all proposed mechanisms
and novelty statements as hypotheses to be criticized and tested, not as
established results.

## Required reading order

Before proposing code or experiments, read:

1. `README.md`
2. `docs/concept.md`
3. `docs/novelty_and_landscape.md`
4. `docs/decisions_and_open_questions.md`
5. `docs/expert_review_brief.md`

## Current task for a remote expert

Independently assess whether the research problem is real, whether the proposed
method has a defensible contribution, and what the smallest decisive experiment
should be. Produce an executable research plan at `docs/expert_plan.md` using
the requested structure in `docs/expert_review_brief.md`.

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
  compute. State resource assumptions explicitly and provide a lower-cost path.
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
