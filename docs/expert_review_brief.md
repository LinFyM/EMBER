# Remote Expert Review Brief

## Your role

Act as an independent embodied-AI and robot-learning research expert. You have no
access to the conversation that produced this repository; the checked-in files
are the complete handoff context.

The goal is not to endorse the proposal. Determine whether it is coherent,
meaningfully differentiated, falsifiable, and feasible, then choose a concrete
path forward.

## Requested deliverable

Create `docs/expert_plan.md`. It should make one primary recommendation rather
than presenting an unranked catalog of possibilities.

Use the following structure.

### 1. Executive judgment

- Is EMBER solving a real VLA/embodied-learning problem?
- What is the strongest defensible claim?
- What part is most likely already covered by prior work?
- Is the project worth a mechanism-level pilot? Answer yes, no, or only after a
  named prerequisite.

### 2. Corrections to the current formulation

- Identify invalid assumptions, hidden information leakage, ill-defined
  gradients, or mismatches between video information and robot control.
- State whether "bootstrapping" and "adaptation geometry" are technically
  justified.
- Specify whether the base and Writer can realistically be jointly optimized,
  and through which outer-loop estimator.

### 3. Exact first claim and task split

- Write a one-paragraph candidate paper claim.
- Define source tasks, held-out tasks, and the precise distribution shift.
- Explain why success would demonstrate meta-generalization rather than task
  recognition, retrieval, or ordinary fine-tuning.

### 4. Minimum falsification experiment

Select one primary environment/benchmark and one base policy. Specify:

- source and held-out tasks;
- how action-hidden task videos and language are produced;
- what the Writer sees;
- what supervision and reward are available only during training;
- which layers receive adapters;
- interaction budget, number of seeds, and success criteria;
- why this is the cheapest faithful test of the central hypothesis.

If a smaller non-VLA sandbox is necessary before the VLA experiment, describe it
as a short Gate 0 probe and explain exactly what it can and cannot validate.

### 5. Concrete model design

Provide a module-level architecture with approximate tensor shapes or parameter
counts where possible:

- language/video encoders and temporal aggregation;
- Writer architecture;
- initial adapter-center representation;
- shared basis bank, gates, scales, and residual;
- critic/value function;
- frozen versus trainable components.

Explicitly choose one of these as the first implementation:

1. direct LoRA only;
2. direct LoRA plus ordinary LoRA RL;
3. center plus predicted gates/scales over a shared basis bank;
4. a different design, with a clear reason.

### 6. Training algorithm

Provide pseudocode for:

- supervised Writer bootstrap;
- one meta-training episode;
- task-local RL adaptation;
- shared outer-loop update;
- held-out evaluation.

State whether gradients pass through the inner RL loop. If not, identify the
first-order, evolutionary, implicit, or black-box alternative.

### 7. Baselines, ablations, and metrics

Give a matched-budget comparison table. It must isolate:

- conditional adapter generation;
- the value of predicted geometry;
- the value of video beyond language/task identity;
- the value of meta-training beyond adapter retrieval;
- final performance versus adaptation speed and stability.

### 8. Data and compute plan

Estimate a low-cost pilot and a publication-scale experiment:

- GPU memory, GPU-hours, simulator steps, storage, and wall-clock time;
- demonstration counts and task counts;
- caching or frozen-feature strategies;
- the main scaling bottleneck.

Do not assume a real robot or unrestricted compute. State all assumptions.

### 9. Milestones and stop conditions

Propose three or four milestones, each with:

- artifact to produce;
- quantitative pass criterion;
- estimated effort;
- failure interpretation;
- whether to proceed, redesign, or stop.

The milestones should follow the go/no-go gates in
`decisions_and_open_questions.md`, not a calendar-first roadmap.

### 10. Literature gaps

- Add any closer work missing from `novelty_and_landscape.md`.
- For every proposed novelty statement, identify the strongest competing paper.
- Distinguish peer-reviewed evidence from very recent preprints.

## Constraints

- Do not merge this project concept with another research agenda.
- Do not propose building a foundation VLA from scratch.
- Do not start with arbitrary internet videos or cross-embodiment transfer.
- Do not use access to action labels in the Writer input when evaluating the
  action-free-video claim.
- Do not count updating shared parameters on held-out tasks as meta-learning
  generalization.
- Do not call a single LoRA update a subspace.
- Do not rely only on parameter distance; use behavioral constraints and
  behavior-level evaluation.
- Prefer a decisive small experiment over an ambitious full system whose
  failure would be uninterpretable.

## Desired review style

Be skeptical and concrete. Lead with decisions and evidence. When several
options are viable, recommend one and explain the decisive tradeoff. Flag any
point that requires information not present in the repository instead of
silently inventing it.
