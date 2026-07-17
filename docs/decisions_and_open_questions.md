# Decisions, Open Questions, and Go/No-Go Gates

## Settled working decisions

These are current design constraints, not experimentally validated conclusions.

1. **General thesis first.** The underlying research goal is a meta-learned
   optimizer that translates informative, non-label inputs across task and
   supervision distributions into beneficial parameter updates. EMBER is its
   embodied test case, not the origin of the idea.
2. **Standalone project.** EMBER is developed and evaluated as an independent
   embodied-learning project. Prior-project lessons are evidence and process
   guidance, not a code or architecture dependency.
3. **Task specification is multimodal.** The Writer may consume language, video,
   or both. It must not depend on video being present in every episode.
4. **Immediate utility is required.** The generated update should directly
   improve relevant task behavior, even if the gain is small.
5. **Executable source-task supervision is allowed.** Early Writer training may
   use action-labeled robot trajectories, teacher policies, and reward signals.
   Human or action-free video alone is not assumed to identify robot actions.
6. **Reward shapes both local and shared learning, at different levels.** Inner
   RL initially updates task-local state. Across a meta-batch of source tasks,
   the outer objective may update the shared Writer and shared base policy.
7. **Direct update and geometry are complementary.** The preferred target is an
   initial adapter center plus a soft adaptation region, not a choice between a
   point update and an inescapable hard subspace.
8. **Held-out evaluation freezes shared modules.** Updating the shared Writer or
   base online is reserved for a separate continual-meta-learning study.
9. **The first study is same-embodiment and simulated.** Human internet video,
   cross-embodiment transfer, tactile feedback, and real-robot RL come only after
   the core mechanism survives falsification.
10. **Active compute ceiling.** EMBER may use at most four NVIDIA A100 80GB GPUs
    concurrently. Exploratory runs should normally use one or two; any three- or
    four-GPU run requires a measured memory or throughput reason.
11. **Development before scale confirmation.** Use SmolVLA plus LIBERO for the
    first faithful benchmark, oracle, representation, and Writer gates. Scale a
    surviving mechanism to OpenVLA-OFT rather than beginning with the 7B path.
12. **Gate failure triggers bounded recovery.** A failed gate must be diagnosed
    and explored on source/validation surfaces before stopping, without leaking
    held data, weakening matched baselines, or changing the core claim after
    seeing held results.

## Target complete design

The staged plan should ultimately implement and evaluate all of the following,
unless an earlier gate falsifies the corresponding hypothesis:

1. language-only, video-only, and combined task specifications;
2. supervised Writer bootstrapping on source tasks with executable labels or
   functional query supervision;
3. an immediately useful generated adapter center;
4. a task-conditioned soft adaptation geometry;
5. reward-driven task-local refinement;
6. an outer loop in which source-task outcomes improve the shared Writer and,
   if justified, the shared base policy;
7. held-out evaluation with shared modules frozen;
8. matched-budget baselines, modality controls, and complete interaction curves.

The minimum experiment is the first gate toward this end state, not the final
scope of the requested expert plan.

## P0 decisions: resolve before implementation

### P0.0 Scientific abstraction and terminology

**Question:** Is the general claim best formalized as a meta-optimizer,
amortized meta-learner, task-conditioned hypernetwork, learned update rule, or a
more restricted concept?

**Why blocking:** Calling the system a cross-distribution optimizer may imply
universality that one embodied experiment cannot establish. The expert must
state the necessary assumptions and the exact level of generalization tested.

**Current default:** Describe the Writer operationally as an amortized,
task-conditioned parameter-update generator; reserve the broader
cross-distribution meta-optimizer claim for a clearly scoped thesis statement.

### P0.1 Exact scientific claim and task distribution

**Question:** What is the smallest task distribution on which "new task" is
meaningful and train/test tasks share enough structure for a Writer to
meta-learn?

The plan must distinguish:

- unseen object/state variations of known skills;
- new compositions of known primitives;
- held-out task primitives;
- new embodiments.

**Why blocking:** Without an explicit distribution, the project cannot define
meta-learning, leakage, or generalization. A random episode split is not enough.

**Current default:** Target held-out task compositions or held-out tasks within
one embodiment. Treat new embodiment as out of scope.

### P0.2 Minimal environment and benchmark

**Question:** Should the first mechanism test use a fast multi-task simulator
such as Meta-World, a VLA-oriented benchmark such as LIBERO, or a staged pair?

The selection must provide:

- enough source tasks for meta-training;
- held-out tasks with a defensible structural relation to source tasks;
- deterministic access to actions, rewards, resets, and query rollouts;
- videos that can be shown to the Writer while hiding action labels;
- feasible online interaction throughput.

**Why blocking:** The benchmark determines whether a real VLA can be used, how
the Writer is supervised, and whether RL experiments are computationally
feasible.

**Current default:** Use action-hidden videos generated from robot trajectories
in LIBERO. Use SmolVLA as the faithful development policy; Meta-World is only an
optional unit-test environment for an estimator and cannot replace the VLA
claim.

### P0.3 Base policy and adaptation target

**Question:** Which pretrained VLA or visuomotor base is small enough to train
but realistic enough to support the intended claim? Which layers receive
adapters?

Candidates may include an OpenVLA-OFT-style base or a smaller VLA/visuomotor
policy. The plan must specify whether LoRA is placed in:

- only the action expert/head;
- cross-modal projection and action layers;
- selected attention/MLP blocks;
- or the full VLA stack.

**Why blocking:** Generating and adapting a full-VLA adapter may dominate memory
and compute, while updating only an action head may weaken the multimodal claim.

**Current default:** Start with predeclared matrices in the SmolVLA action
expert/head. If the mechanism survives, confirm it in the two 4096-to-4096
residual-block linear layers of an OpenVLA-OFT L1 action head. Expand targets
only through a recorded gate-recovery decision.

### P0.4 Writer supervision and meta-episode construction

**Question:** What exactly is one meta-training episode, and which observations
are visible to the Writer versus the policy and loss?

The plan must define:

- support specification: language, one or more videos, viewpoints, duration;
- hidden executable labels available only to training objectives;
- query trajectories or online rollouts;
- how source-task demonstrations are split to avoid video/query leakage;
- whether teacher adapters are generated and, if so, whether they are auxiliary
  targets or only baselines.

**Why blocking:** "Train the Writer on labeled data" is not yet a reproducible
learning problem.

**Current default:** The Writer sees language plus action-hidden robot video.
Functional action/query loss supplies the primary supervised signal; raw teacher
adapter MSE is auxiliary at most.

### P0.5 Inner-loop RL contract

**Question:** Which RL algorithm, reward, critic, reset mechanism, and interaction
budget make the geometry claim testable?

The plan must decide:

- on-policy versus off-policy adaptation;
- sparse outcome reward versus shaped/process reward;
- whether a critic is shared, task-conditioned, or relearned;
- which variables are task-local;
- the KL/trust-region constraint and residual escape schedule;
- whether meta-gradients pass through inner optimization or use a first-order or
  black-box outer estimator.

**Why blocking:** A poor RL implementation can make every Writer geometry look
bad, while excessive reward shaping can hide the claimed bootstrapping problem.

**Current default:** Use the simplest stable task-local algorithm compatible with
the selected action policy, report sparse-reward results separately, and avoid
updating shared policy parameters inside a single task.

## P1 decisions: resolve during the first prototype

### P1.1 Writer architecture

- Which encoders are frozen or trainable?
- Does video use pooled features, temporal tokens, object tracks, latent actions,
  or a learned resampler?
- How are missing modalities represented?
- Does the Writer output adapter tensors directly, or coefficients over a basis
  bank?

**Default:** Reuse frozen language/video features when possible and predict
coefficients/gates over a learned shared bank before generating full tensors.

### P1.2 Geometry representation

Compare in increasing complexity:

1. direct task-conditioned LoRA center;
2. center plus unconstrained standard LoRA RL;
3. center plus fixed global basis;
4. center plus Writer-predicted gates/scales over a global basis;
5. center plus fully generated task-specific directions;
6. soft geometry plus an annealed residual.

Do not implement level 5 before levels 1-4 establish value.

Before evaluating level 4, train its task-conditioned geometry. The active
default is a short differentiable source support/query inner loop in the
low-dimensional bank-coordinate space using cached policy features. This gives
the predicted preconditioner a direct meta-training signal before any
source-reward outer refinement.

### P1.3 Static versus feedback-aware Writer

A static Writer can learn from rewards across training tasks even if reward is
not an input. It cannot respond specifically to the current task's failed
attempts. Decide whether the first paper needs:

- one-shot `Writer(specification)` followed by coefficient-only RL; or
- iterative `Writer(specification, rollout history, adapter state)` updates.

**Default:** Start static. Add feedback-aware rewriting only after the basic
center and geometry are validated.

### P1.4 Baselines and ablations

At minimum include:

- base policy without adaptation;
- direct language/video conditioning without generated parameters;
- a capacity-matched language-only HyPoGen/DISC-style parameter generator;
- supervised fine-tuning or behavioral cloning with matched source supervision;
- random and nearest-task adapters;
- standard task-specific LoRA RL with matched trainable parameter count;
- Writer center followed by unconstrained LoRA RL;
- fixed/global policy subspace;
- action-labeled one-demonstration oracle;
- language-only, video-only, language-plus-video, and task-ID controls.

### P1.5 Metrics

Primary metrics should include:

- zero-step success or return;
- success-versus-interactions AUC;
- episodes and environment steps to a target success threshold;
- final success under a fixed budget;
- seed variance, failure rate, and catastrophic policy drift;
- policy KL or action-distribution change;
- gradient variance and critic error where relevant;
- wall-clock update time and peak memory.

### P1.6 Resource envelope

The active maximum is **4 x NVIDIA A100 80GB**. Recalculate all recipes from the
eight-GPU expert plan for this ceiling and estimate:

- GPU type/count and training hours for the Writer and VLA;
- simulator throughput and total environment steps;
- storage for demonstrations, video features, and checkpoints;
- expected number of tasks, trajectories, seeds, and ablations;
- a low-cost falsification path and a publication-scale path;
- whether each phase needs 1, 2, 3, or 4 GPUs and whether model/data parallelism
  is actually necessary;
- which encoders or features should be frozen/cached to reduce repeated cost.

Storage capacity, real-robot access, and guaranteed wall-clock allocation have
not been fixed, so those assumptions must remain explicit.

## P2 extensions: explicitly defer

- arbitrary third-person human tutorial videos;
- human-to-robot embodiment transfer;
- multiple robot embodiments;
- tactile and proprioceptive task specifications;
- real-robot online RL and autonomous resets;
- continual updates to the shared Writer/base during deployment;
- long-horizon planning and external memory systems.

## Go/no-go gates

Gate outcomes are evidence, not an instruction to give up after one run. Follow
the recovery protocol in `AGENTS.md`: verify mechanics, classify the failure,
try bounded source/validation remedies that preserve the scientific invariants,
and rerun with matched controls. Stop or narrow the claim only after those
remedies are exhausted or the underlying information/representation hypothesis
is genuinely falsified.

### Gate A: direct-update value

On held-out tasks, the generated center must outperform the base, random
adapters, and a nearest-task retrieval adapter before any new-task RL.

If Gate A fails, stop geometry work and diagnose task encoding, supervision,
and whether direct parameter generation is appropriate.

### Gate B: adaptation-geometry value

With the same interaction and trainable-parameter budget, predicted geometry
must outperform Writer initialization followed by ordinary LoRA RL on AUC,
stability, or time-to-threshold.

If Gate B fails, retain conditional adapter generation but remove the learned
geometry claim.

### Gate C: meta-generalization

Gates A and B must hold on a predeclared held-out task split with all shared
modules frozen.

If Gate C fails, do not frame the system as cross-task meta-learning.

### Gate D: multimodal necessity

Video must contribute beyond language or task-ID controls on tasks where motion
information is actually relevant.

If Gate D fails, narrow the project to task-conditioned parameter adaptation or
redesign the video representation.
