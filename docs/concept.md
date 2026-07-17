# EMBER Research Concept

This document specifies the embodied instance. The broader motivation and the
meaning of cross-distribution information-to-parameter transfer are defined in
[`origin_and_general_thesis.md`](origin_and_general_thesis.md).

## 1. Motivation

Action-labeled robot trajectories contain synchronized observations, robot
states, actions, and often success labels. They are substantially more expensive
to collect than language descriptions or action-free videos. Action-free data,
however, is not directly executable: it lacks calibrated robot actions, contact
forces, embodiment constraints, and reward information.

At the same time, sparse-reward RL may fail to improve a VLA whose initial
success probability is effectively zero. EMBER targets the gap between these two
facts. It asks whether a meta-trained model can transform a task specification
into a rough but executable parameter prior, after which interaction can correct
the remaining details.

This is a concrete test of the general Writer thesis: the input distribution
(language/video) and the useful supervision distribution (robot actions and
returns) differ, and the learned cross-task bridge is expressed as a parameter
update rather than direct label prediction.

The human-learning analogy is deliberately limited: watching an instructional
video may provide the rough structure of a new movement, but physical practice
and feedback are still needed to correct timing, force, and coordination.

## 2. Problem statement

Let each task be denoted by \(T\), drawn from a meta-training task distribution
\(p_{train}(T)\). A task has:

- a specification \(x_T = (l_T, v_T)\), where either language \(l_T\), video
  \(v_T\), or both may be present;
- an environment with observations, actions, transitions, and reward;
- source-task executable supervision during early training, such as robot
  trajectories or a teacher policy;
- query rollouts used to measure behavior after the Writer update and after
  task-local adaptation.

The shared base VLA policy is \(\pi_\theta\). The shared Writer is
\(H_\psi\). Task-local state is \(z_T\). A critic or value estimator is kept
separate from the policy adapter unless an experiment justifies coupling them.

At held-out evaluation, tasks are sampled from a separately defined
\(p_{test}(T)\). Shared parameters \(\theta\) and \(\psi\) are frozen; only
task-local state may adapt. Updating the global Writer or base policy on the
held-out task is a different continual-meta-learning problem.

## 3. Proposed parameterization

### 3.1 Writer output

The Writer consumes the task specification and optionally task-specific
feedback history:

\[
(\Delta\theta_T^0, \mathcal{G}_T)
= H_\psi(x_T, h_T).
\]

- \(\Delta\theta_T^0\) is an initial low-rank adapter center. Applying it must
  improve the task policy before any new-task RL.
- \(\mathcal{G}_T\) is a soft adaptation geometry: directions, gates, scales,
  or a local metric that prioritizes how task-local RL should update the policy.
- \(h_T\) may contain rollout summaries, rewards, or the current adapter. It is
  empty in the static one-shot Writer and becomes necessary only if the Writer
  must react differently to the current task's observed failures.

### 3.2 Recommended first geometry

The first tractable version should use a globally learned bank of low-rank
adapter directions \(\{B_i\}_{i=1}^m\). The Writer predicts an initial center,
task-specific gates \(g_{T,i}\), and per-direction scales. Task-local RL updates
coefficients \(\alpha_{T,i}\):

\[
\Delta\theta_T
= \Delta\theta_T^0
+ \sum_{i=1}^{m} g_{T,i}\alpha_{T,i}B_i
+ \rho_T.
\]

The residual \(\rho_T\) is initially small and behaviorally constrained by a
policy-KL or action-distribution trust region. Its constraint may be annealed so
that an incorrect Writer geometry does not permanently trap the policy.

This is an **affine soft adaptation region**, not a strict hard subspace. A
single LoRA update is only a point. Conversely, allowing both factors of an
ordinary LoRA adapter to change freely does not preserve a claimed task-specific
subspace.

Generating a complete, task-specific bank of LoRA bases from scratch is a later
variant. It should not be attempted before a shared basis bank with predicted
coefficients and gates demonstrates value.

## 4. Training contract

### 4.1 Phase A: supervised bootstrapping

Early training requires executable supervision on source tasks. Candidate
signals include paired robot demonstrations, teacher-policy actions, query-set
behavioral loss, or differentiable return estimates.

The preferred objective evaluates behavior after applying the generated update:

\[
\mathcal{L}_{boot}
= \mathcal{L}_{action/query}
  (\pi_{\theta + \Delta\theta_T^0}, D_T^{query}).
\]

Raw MSE to a teacher LoRA is not sufficient as the main objective because many
parameter updates can implement equivalent policies. Teacher adapters may be
used as auxiliary supervision, but the primary criterion should be behavioral
or return-based.

If the objective explicitly compares the Writer-on policy with the base policy,
the base reference must be frozen or stop-gradient. Otherwise the shared base
could degrade merely to make the Writer's relative gain appear larger.

### 4.2 Phase B: task-local RL and outer meta-optimization

For a sampled source task:

1. encode the language/video specification;
2. generate the initial adapter center and adaptation geometry;
3. measure zero-interaction query performance;
4. collect task rollouts;
5. update only task-local coefficients and the constrained residual for \(K\)
   inner-loop steps;
6. evaluate the adapted policy on fresh query rollouts;
7. update the shared Writer and, if selected, the shared base policy through an
   outer objective across a meta-batch of tasks.

A schematic objective is:

\[
\max_{\theta,\psi}\ \mathbb{E}_{T}
\left[
\lambda_0 J_T(\theta + \Delta\theta_T^0)
+ \lambda_K J_T(\theta + \Delta\theta_T^K)
- \beta\,\Omega_T
\right],
\]

where the zero-step term teaches immediate usefulness, the post-adaptation term
teaches improvability, and \(\Omega_T\) regularizes behavior change or geometry.

Jointly optimizing the Writer and base policy refers to this cross-task outer
loop. The initial inner loop should not update shared \(\theta\) or shared
\(\psi\) from a single task.

### 4.3 Phase C: held-out evaluation

For every held-out task:

1. freeze the shared base, Writer, basis bank, and critic training procedure;
2. generate one task-local center and geometry from the task specification;
3. measure zero-step performance;
4. adapt only task-local state using the predeclared interaction budget;
5. report the complete success-versus-interactions curve and final performance.

The evaluation must distinguish at least:

- new states, viewpoints, or object instances within a known task;
- new compositions of known skill primitives;
- genuinely new task primitives;
- new robot embodiments.

These are not interchangeable forms of "cross-distribution" generalization.

## 5. What bootstrapping means in EMBER

Bootstrapping is the deployment-time Writer step that moves the base VLA from
no useful behavior to minimum viable competence:

\[
J_T(\theta + \Delta\theta_T^0) > J_T(\theta).
\]

The supervised source-task phase teaches the Writer to do this; it is not itself
the deployment bootstrap. Subsequent reward-driven improvement is experience
refinement.

If the generated update does not directly improve held-out-task behavior and
only acts as a regularizer for RL, the term "bootstrapping" is too strong and
the project should be reframed as guidance or adaptation shaping.

## 6. Falsifiable hypotheses

- **H1: Immediate utility.** The Writer update improves held-out-task action or
  return metrics over the base policy, random adapters, and retrieved
  nearest-task adapters before interaction.
- **H2: Adaptation value.** Writer-predicted geometry improves interaction AUC,
  stability, or episodes-to-threshold over Writer initialization followed by
  unconstrained, matched-budget LoRA RL.
- **H3: Meta-generalization.** H1 and H2 persist when shared modules are frozen
  and tasks are held out according to a predeclared task split.
- **H4: Multimodal value.** Video provides improvement beyond language-only
  task identification, rather than merely leaking the task label.

Failure of H1 reduces the method to an RL guidance mechanism. Failure of H2
removes the need for task-conditioned geometry. Failure of H3 invalidates a
meta-generalization claim. Failure of H4 limits the motivation for video.

## 7. Non-goals for the first study

- Training a new foundation VLA from scratch.
- Solving arbitrary internet-video-to-robot transfer immediately.
- Claiming cross-embodiment generalization from a same-embodiment experiment.
- Updating all global parameters online on one held-out task.
- Treating parameter distance alone as evidence of behavioral safety.
- Demonstrating only final success while omitting zero-step performance and
  interaction curves.
