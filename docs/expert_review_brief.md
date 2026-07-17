# Remote Expert Review Brief

## Your role

Act as an independent expert in embodied AI, VLA systems, meta-learning,
parameter-efficient adaptation, and reinforcement learning. You have no access
to the conversation that produced this repository; the checked-in files are the
complete handoff context.

The goal is not to endorse the proposal. Determine whether the **general
information-to-parameter meta-learning thesis** and the **EMBER embodied
instantiation** are reasonable, identify obvious flaws and improvements, choose
models and data that fit the available resources, and plan a staged path toward
the complete design.

Read `prior_work_memllm_lessons.md` as empirical and process background. Do not
copy its language-model memory architecture into EMBER.

## Known resource envelope

- Maximum compute: **8 x NVIDIA A100 80GB GPUs**.
- Real-robot access: not assumed.
- Storage and guaranteed wall-clock allocation: not yet specified; state your
  assumptions.
- The plan should include a low-cost path that does not occupy all eight GPUs
  for every exploratory run.

## Requested deliverable

Create `docs/expert_plan.md`. Make one primary recommendation rather than an
unranked catalog. The plan should begin with decisive falsification gates, but
its endpoint should cover the complete target design: multimodal Writer,
immediate update, adaptation geometry, task-local RL, reward-shaped outer
learning, and frozen held-task evaluation.

Use the following structure.

## 1. Executive judgment

- Is the original cross-distribution information-to-parameter thesis
  scientifically meaningful?
- Is EMBER solving a real VLA/embodied-learning problem?
- What are the strongest and weakest assumptions?
- What is the strongest defensible paper claim?
- Is the complete project worth pursuing? Answer yes, no, or only after a named
  prerequisite.

## 2. Correct abstraction and terminology

- Decide whether this should be called a meta-optimizer, amortized meta-learner,
  task-conditioned hypernetwork, learned update rule, or something else.
- Define the distributions being crossed: information modality, supervision,
  task, environment, or embodiment.
- State the assumptions under which an informative non-label input can identify
  a beneficial parameter update.
- Explain what one successful embodied experiment would and would not establish
  about the broader thesis.

## 3. Problems and improvements

Identify:

- invalid assumptions or missing variables;
- information leakage and shortcut risks;
- mismatches between video information and executable robot control;
- parameter non-identifiability or hypernetwork-output scaling problems;
- ill-defined gradients through the inner RL loop;
- collapse to task-ID lookup, generic policy bias, or adapter retrieval;
- safety/stability problems from restricting RL to a predicted geometry;
- changes that would make the concept simpler, more defensible, or more likely
  to work.

State whether "bootstrapping" and "adaptation geometry" are technically
justified. Explain whether the shared base and Writer can realistically be
jointly optimized, and with which estimator.

## 4. Model, data, and benchmark selection under 8 x A100 80GB

Recommend one primary stack and one lower-cost fallback. Compare only serious
candidates, such as an OpenVLA-OFT-class model, SmolVLA-class model, another
open VLA, or a smaller visuomotor policy used as a Gate 0 sandbox.

For the chosen stack, specify:

- model and checkpoint;
- parameter count, precision, and expected per-GPU memory;
- which modules are frozen;
- exact LoRA/adaptation targets;
- distributed strategy and required GPU count;
- simulator and benchmark;
- robot demonstration and language data;
- how action-hidden videos are produced;
- train/validation/held-out task split;
- licenses and practical data-access constraints;
- why this pairing tests EMBER rather than merely ordinary VLA fine-tuning.

Explicitly consider whether Meta-World, LIBERO, or a staged pair is preferable.
If a public robot dataset such as BridgeData, DROID, Open X-Embodiment, or a
benchmark-native dataset is proposed, explain exactly how it enters
meta-training and how task leakage is prevented.

## 5. Exact first claim and minimum falsification experiment

Write a one-paragraph candidate first-paper claim and define the precise
distribution shift.

Then select the cheapest faithful experiment. Specify:

- source and held-out tasks;
- Writer-visible language/video;
- supervision and rewards available only during training;
- useful-update oracle and representation gate;
- adapter location and output size;
- interaction budget and reset assumptions;
- number of seeds;
- quantitative pass/fail thresholds;
- why a pass is more than task recognition, retrieval, or ordinary fine-tuning.

If a non-VLA sandbox is needed, call it **Gate 0** and state exactly what it can
and cannot validate. Do not let a sandbox result substitute for the VLA claim.

## 6. Complete model design

Provide a module-level design with approximate tensor shapes and parameter
counts:

- language/video encoders and temporal aggregation;
- handling of missing modalities;
- Writer architecture;
- initial adapter-center representation;
- shared basis bank, gates, per-direction scales, and residual escape path;
- policy-KL or behavioral trust region;
- critic/value function;
- feedback-history representation if an iterative Writer is eventually needed;
- frozen, shared-trainable, and task-local parameters.

Explicitly choose the first implementation among:

1. direct LoRA only;
2. direct LoRA plus ordinary LoRA RL;
3. center plus predicted gates/scales over a shared basis bank;
4. a different design, with a clear reason.

Then state which later milestone adds the remaining components of the complete
design.

## 7. Training and evaluation algorithm

Provide pseudocode for:

- source-task useful-update oracle construction;
- supervised Writer bootstrap;
- one meta-training episode;
- task-local RL adaptation;
- shared outer-loop Writer/base update;
- held-out evaluation.

State whether gradients pass through inner RL. If not, choose and justify a
first-order, implicit, evolutionary, critic-based, or black-box alternative.

Clearly separate:

- Writer-visible input;
- training-only bridge supervision;
- task-local state;
- shared parameters;
- meta-train, selection, and reporting-only data.

## 8. Baselines, ablations, and metrics

Provide a matched-budget table that isolates:

- conditional adapter generation;
- the value of predicted geometry;
- video beyond language or task identity;
- meta-training beyond nearest-adapter retrieval;
- RL refinement beyond Writer zero-step gain;
- final performance versus adaptation speed and stability.

At minimum address:

- base policy;
- direct language/video conditioning;
- standard supervised or behavioral-cloning adaptation;
- random and nearest-task adapters;
- matched-parameter standard LoRA RL;
- Writer center followed by unconstrained LoRA RL;
- fixed global subspace;
- action-labeled one-demonstration oracle;
- language-only, video-only, language-plus-video, shuffled-video, and task-ID
  controls.

Metrics should include zero-step gain, success-versus-interactions AUC,
episodes to threshold, final success under a fixed budget, seed variance,
control harm, policy KL/action drift, wall-clock time, and peak memory.

## 9. Resource and systems plan

Estimate both a low-cost pilot and the full design under the eight-A100 ceiling:

- GPU count per phase and whether data/model parallelism is required;
- GPU-hours and wall-clock time;
- simulator environment steps and rollout throughput;
- demonstration counts, task counts, and seeds;
- video-feature caching and storage;
- checkpoint and optimizer-state storage;
- which phases can run concurrently and which must be serial because they read
  held surfaces;
- primary scaling and systems bottlenecks.

Separate one-time meta-training cost from per-task Writer inference and RL
adaptation cost.

## 10. Staged plan to complete the full design

Propose milestones from mechanism isolation to the complete EMBER system. Each
milestone must include:

- artifact to produce;
- components implemented;
- data and compute used;
- quantitative pass criterion;
- failure interpretation;
- whether to proceed, redesign, narrow the claim, or stop.

The plan should include, when scientifically authorized:

1. useful-update oracle and representation Gate 0;
2. direct Writer zero-step utility;
3. task-local RL from Writer initialization;
4. predicted adaptation geometry;
5. reward-trained shared Writer and optional base outer loop;
6. multimodal and held-task full evaluation;
7. only then, a justified human-video or cross-embodiment extension if needed.

Use scientific predecessor gates rather than a calendar-first roadmap, but give
rough effort and wall-clock estimates so the plan is actionable.

## 11. Literature and novelty gaps

- Add any closer work missing from `novelty_and_landscape.md`.
- For every proposed novelty statement, identify the strongest competing paper.
- Distinguish peer-reviewed work from recent preprints.
- Assess whether lessons from `prior_work_memllm_lessons.md` transfer correctly
  or whether the embodied setting changes the diagnosis.

## Constraints

- Keep EMBER independent from the prior project; use its lessons, not its code
  or Wiki/QA architecture.
- Do not propose building a foundation VLA from scratch.
- Do not start the main claim with arbitrary internet video or cross-embodiment
  transfer.
- Do not expose action labels to the Writer when evaluating the action-free
  input claim.
- Do not count updating shared parameters on held-out tasks as meta-learning
  generalization.
- Do not call a single LoRA update a subspace.
- Do not rely only on parameter distance or teacher-adapter MSE.
- Do not stop the plan at a toy pilot; use the pilot as a gate toward the full
  design.
- Do not hide one-time meta-training cost behind fast deployment claims.

## Desired review style

Be skeptical and concrete. Lead with decisions and evidence. When several
options are viable, recommend one and explain the decisive tradeoff. Flag any
missing information instead of silently inventing it.
