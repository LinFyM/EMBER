# EMBER Research Concept

This document specifies the active embodied instance. The broader motivation is
defined in [`origin_and_general_thesis.md`](origin_and_general_thesis.md). The
2026-07-18 owner correction supersedes historical proposals for a canonical
bank, shared update subspace, soft geometry, residual escape, or mandatory
canonical-representation Gate.

## 1. Motivation

Action-labeled robot trajectories contain synchronized observations, robot
states, actions, and often success labels. Language and action-hidden videos can
describe a task but do not directly provide calibrated robot actions, contact
dynamics, or reward. At the same time, sparse-reward RL may not improve a VLA
whose initial success probability is effectively zero.

EMBER asks whether source-task executable supervision can train a Writer to
transform language/action-hidden video into a rough but functional task-specific
LoRA. The Writer's deployment output should cross the zero-competence barrier;
ordinary interaction then refines that same LoRA. This is a concrete
information-to-parameter test, not a claim that action-hidden video alone
contains every robot-control detail.

## 2. Problem statement

For each task \(T\):

- \(x_T=(l_T,v_T)\) is legal language, action-hidden robot video, or both;
- \(D_T^{support}\) and \(D_T^{query}\) are independent executable source
  surfaces used only by training objectives;
- \(E_T\) supplies rollouts and reward; and
- \(\pi_\theta\) is a shared pretrained VLA base.

The shared Writer \(H_\psi\) emits task-local LoRA parameters \(a_T^0\):

\[
a_T^0 = H_\psi(x_T), \qquad
J_T(\pi_{\theta,a_T^0}) > J_T(\pi_{\theta,0}).
\]

The target matrices, rank, scaling, and parameter count are common and frozen
before outcomes. The values of all task-local LoRA factors are task-specific.
The shared base \(\theta\) remains frozen during direct Writer training, source
reward/meta learning, and held evaluation.

## 3. Exact adaptation parameterization

Let the predeclared target set be \(\mathcal{M}\). For every matrix
\(W_m\in\mathcal{M}\), rank-\(r\) LoRA applies

\[
W'_m = W_m + s_m B_{T,m}A_{T,m}.
\]

The Writer emits every \(A_{T,m}\) and \(B_{T,m}\) required by this contract.
"Complete task-specific LoRA" means complete within \(\mathcal{M}\), rank
\(r\), and the fixed parameter budget; it does not mean modifying every base
weight.

After zero-step evaluation, ordinary task-local RL updates the same
\(\{A_{T,m},B_{T,m}\}\) in place. The Writer emits no additional bank, basis,
mask, gate, metric, radius, learning rate, preconditioner, or residual object.

Historical assistant/expert planning proposed a canonical bank that supplied a
shared span, a task-conditioned geometry that scaled/preconditioned directions
inside it, and a residual escape that could leave it. That was a second,
narrower Writer-conditioned RL search space. It is outside the current project
and long-term Goal; no implementation path should be reserved for it.

## 4. Training contract

### 4.1 Gate 0: independent useful-update oracle

Before amortization, independently fit one task-local LoRA per source task at
the exact declared targets/rank. Use a support surface for optimization,
independent query data for selection, and a locked closed-loop surface for the
report. Record immediate gain, task specificity, drift/non-harm, confidence
intervals, and resource cost.

This oracle establishes that useful updates exist and provides an upper bound
and baseline. It does not show that Writer-visible information can predict them.

### 4.2 Direct Writer cold-start

For source tasks, the Writer sees only legal \(x_T\). Its generated LoRA is
applied functionally to the frozen base and evaluated on independent executable
query data:

\[
\mathcal{L}_{writer}
= \mathbb{E}_{T}\left[
  \mathcal{L}_{action/flow/behavior}
  (\pi_{\theta,H_\psi(x_T)},D_T^{query})
\right].
\]

Gradient through functional adapter application is the primary bridge signal.
Raw factor MSE is prohibited as the primary objective because LoRA gauge and
parameter non-identifiability need not align with behavior. Oracle physical
delta/update imitation may be a predeclared auxiliary, never a substitute for
independent functional utility.

Report language-only, video-only, language-plus-video, wrong-video,
same-scene, shuffled/reversed, first/last/scene-only, task-ID, average,
retrieval, direct-conditioning, standard task-specific LoRA, and
capacity-matched DISC/HyPoGen-style parameter-generator baselines. Neutral-prompt
parameter compilation and practical instruction prompting are co-primary
settings.

### 4.3 Ordinary task-local LoRA RL

Initialize \(a_T\leftarrow H_\psi(x_T)\), collect the predeclared reward
budget, and run an ordinary LoRA optimizer over \(a_T\) only. Compare the full
interaction curve against the same optimizer, parameter count, and budget from
standard and other declared initializations. No Writer-predicted object may
change the optimizer or constrain its search.

Primary evidence includes zero-step success/return, success-versus-interaction
AUC, steps/episodes to threshold, final performance, seed uncertainty,
catastrophic drift, wall time, and peak memory.

### 4.4 Source-only reward or delayed outer learning

For a source meta-batch:

1. generate each task's initial LoRA;
2. adapt only that task-local LoRA from source reward;
3. evaluate fresh source query return; and
4. update \(\psi\) so future zero-step initializations and subsequent
   adaptation improve.

The estimator may differentiate through the inner loop or use one predeclared
stable alternative. In the default mainline, no shared base weight, shared
adapter, or other shared policy state is trained. The model-side update is the
task-local LoRA; the shared learning update is the Writer.

Updating shared base weights/shared LoRA would be a separate justified matched
ablation, not the default and not required for completion.

### 4.5 Held-task evaluation

Before held outcomes, freeze the base, Writer, encoders, target/rank contract,
normalization authority, optimizer, budgets, thresholds, and all shared state.
For each held task:

1. generate a zero-step LoRA from legal language/action-hidden video;
2. measure immediate utility;
3. adapt only that task-local LoRA from the declared held reward budget; and
4. report the complete curve and matched controls.

Held actions, labels, proprioceptive trajectories, terminals, filenames, task
IDs, and hidden normalization information are never Writer inputs or tuning
surfaces.

## 5. What bootstrapping means

Bootstrapping is the deployment-time Writer step that moves the base VLA to
minimum viable competence before target-task interaction:

\[
J_T(\pi_{\theta,H_\psi(x_T)}) > J_T(\pi_{\theta,0}).
\]

Source supervision teaches this transformation; it is not itself the held-task
bootstrap. If the generated LoRA has no immediate functional utility and only
changes later optimization, the bootstrapping claim fails.

## 6. Falsifiable hypotheses

- **H1: Information validity.** Legal language/video contains task-relevant
  information beyond scene/task-ID shortcuts.
- **H2: Useful-update existence.** Independent task-local LoRA oracles improve
  locked source behavior at the declared targets/rank.
- **H3: Direct Writer utility.** Generated LoRA improves zero-interaction
  behavior over base, average, retrieval, direct-conditioning, standard-LoRA,
  and capacity-matched direct-generator controls.
- **H4: Local-RL value.** Writer initialization improves matched-budget
  ordinary LoRA RL AUC or time-to-threshold without hiding a final-performance
  loss.
- **H5: Source reward learning.** Source-only outer reward improves future
  Writer initializations while the shared base remains frozen.
- **H6: Frozen meta-generalization.** The declared gains persist on sealed held
  compositions with all shared state frozen.
- **H7: Multimodal value.** Action-hidden video contributes causally beyond
  language/task-ID controls.

Each failure localizes a different claim and triggers bounded recovery; none
authorizes held leakage, threshold weakening, or substitution of direct
conditioning for parameter-generation evidence.

## 7. Non-goals for the current project

- Canonical banks, shared task-update subspaces, predicted RL geometry,
  residual escape, or any second Writer-conditioned search object.
- Shared-base/shared-LoRA source outer training in the default mainline.
- A universal optimizer across arbitrary modalities or distributions.
- Human-video or cross-embodiment transfer, arbitrary web video, or real-robot
  RL in the first claim.
- Training a foundation VLA from scratch.
- Updating shared state on held tasks.
- Treating parameter distance, factor MSE, healthy gradients, or final success
  alone as proof of the mechanism.
